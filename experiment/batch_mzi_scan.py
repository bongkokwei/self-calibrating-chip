"""
batch_mzi_scan_refactored.py

Refactored voltage scan script to characterise V_2π and φ_init for individual MZIs.

This version uses the modular utilities from photonic_fir.utils:
- mzi_characterisation: Nonlinear fitting functions
- mzi_plotting: Plotting functions

Key improvements:
1. Proper tan² model fitting instead of linear approximation
2. Simultaneous estimation of V_2π and φ_init
3. Cleaner separation of concerns (measurement, fitting, plotting)
4. Plot shows PSR vs V² only (no phase plot in main figure)

Usage:
    # Configure parameters in main() and run
    python batch_mzi_scan_refactored.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict

from voltage_ctrl import VoltageController
from luna_ova import LunaOVA

from photonic_fir import (
    ExperimentConfig,
    config_from_dict,
    measure_spectrum,
    recover_impulse_response_from_df,
    detect_taps,
    tap_coeffs_to_power_splitting_ratios,
    power_splitting_ratio_to_mzi_phase,
    load_config,
    get_next_run_dir,
    setup_logging,
)

# Import new utilities
from photonic_fir.utils.mzi_characterisation import (
    fit_mzi_v2pi_and_phi_init,
    print_fit_results,
)

from photonic_fir.utils.mzi_plotting import (
    plot_mzi_characterisation,
)


@dataclass
class V2piScanConfig:
    """Configuration for V_2π voltage scan."""

    # Target MZI to characterise
    mzi_id: str = "4-5"

    # Voltage scan parameters
    v_min: float = 0.0
    v_max: float = 30.0
    n_points: int = 51

    # Timing
    settling_time_sec: float = 2.0

    # Output
    output_dir: str = "./v2pi_scan_results"
    save_raw_data: bool = True

    # MZI tree structure parameters
    min_stage: int = 1
    max_stage: int = 4

    def get_voltage_range(self) -> np.ndarray:
        """Generate voltage array from parameters."""
        voltage_squared = np.linspace(
            self.v_min**2,
            self.v_max**2,
            self.n_points,
        )
        return np.sqrt(voltage_squared)

    def get_all_mzi_ids(self) -> List[str]:
        """Generate all MZI IDs based on stage configuration."""
        mzi_ids = []
        for stage in range(self.min_stage, self.max_stage + 1):
            for pos in range(1, 2 ** (stage - 1) + 1):
                mzi_ids.append(f"{stage}-{pos}")
        return mzi_ids

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "V2piScanConfig":
        """Create from dictionary."""
        return cls(**config_dict)


def set_mzi_voltage(
    mzi_id: str,
    voltage: float,
    exp_config: ExperimentConfig,
    settling_time_sec: float = 2.0,
    v_max: float = 30.0,
) -> None:
    """
    Set voltage on a specified MZI.

    Parameters
    ----------
    mzi_id : str
        MZI identifier (e.g., "1-1", "2-1", "4-6")
    voltage : float
        Voltage to apply (V)
    exp_config : ExperimentConfig
        Experiment configuration object
    settling_time_sec : float
        Time to wait for thermal settling (seconds)
    v_max : float
        Maximum allowed voltage (V)
    """

    # Get MZI channel from mapping
    mzi_device_id = f"MZI_{mzi_id}"
    mzi_channel = exp_config.channel_mapping.get_channel(mzi_device_id)

    print(f"Setting MZI {mzi_id} (channel {mzi_channel}) to {voltage:.2f} V")

    # Apply voltage
    with VoltageController(
        com_port=exp_config.measurement.voltage_controller_port,
        baud_rate=exp_config.measurement.voltage_controller_baudrate,
        zero_on_exit=False,  # Don't zero when we're done
    ) as v_ctrl:
        v_ctrl.set_voltages([mzi_channel], [voltage], v_max=v_max)
        print(f"✓ Voltage applied")

        if settling_time_sec > 0:
            print(f"Waiting {settling_time_sec} s for thermal settling...")
            time.sleep(settling_time_sec)
            print(f"✓ Settled")


def perform_voltage_sweep(
    scan_config: V2piScanConfig,
    exp_config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.DataFrame]]:
    """
    Perform voltage sweep on a single MZI, measuring spectrum at each point.

    This is the low-level measurement function that sweeps through voltages,
    acquires spectra, and extracts power splitting ratios.

    Parameters
    ----------
    scan_config : V2piScanConfig
        Scan configuration (voltage range, MZI ID, etc.)
    exp_config : ExperimentConfig
        Experiment configuration (chip parameters, instruments, etc.)

    Returns
    -------
    voltages : np.ndarray
        Voltage values scanned
    power_splitting_ratios : np.ndarray
        Measured power splitting ratios (dB)
    mzi_phases : np.ndarray
        Calculated MZI phases (radians)
    dataframes : List[pd.DataFrame]
        List of measurement DataFrames for each voltage
    """

    # Create output directory
    output_path = Path(scan_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get voltage range
    voltage_range = scan_config.get_voltage_range()
    n_voltages = len(voltage_range)

    # Get MZI channel from chip configuration
    mzi_device_id = f"MZI_{scan_config.mzi_id}"
    mzi_channel = exp_config.channel_mapping.get_channel(mzi_device_id)

    # Build MZI tree structure for PSR calculations
    mzi_tree = exp_config.full_mzi_tree.tree

    # Storage for results
    power_splitting_ratios = np.zeros(n_voltages)
    mzi_phases = np.zeros(n_voltages)
    dataframes = []

    print(f"\n{'='*70}")
    print(f"V_2π Voltage Sweep for MZI {scan_config.mzi_id}")
    print(f"{'='*70}")
    print(f"MZI ID: {scan_config.mzi_id}")
    print(f"Channel: {mzi_channel}")
    print(f"Voltage range: {scan_config.v_min:.2f} - {scan_config.v_max:.2f} V")
    print(f"Number of points: {n_voltages}")
    print(f"Settling time: {scan_config.settling_time_sec} s")
    print(f"Output directory: {output_path}")
    print(f"{'='*70}\n")

    # Measure DUT length for delay calculations
    with LunaOVA(ip=exp_config.measurement.ova_address) as ova:
        ova.set_dut_length()

    # Scan through voltages
    for i, voltage in enumerate(voltage_range):
        print(f"[{i+1}/{n_voltages}] Voltage: {voltage:.3f} V")

        # Initialise voltage controller, exit after measurement to ensure heaters are zeroed
        with VoltageController(
            com_port=exp_config.measurement.voltage_controller_port,
            baud_rate=exp_config.measurement.voltage_controller_baudrate,
            zero_on_exit=True,
        ) as v_ctrl:
            # a. Set voltage
            init_mzi_channels = list(exp_config.calibration.initial_mzi_voltages.keys())

            init_psu_channels = [
                exp_config.channel_mapping.get_channel(f"MZI_{mzi_id}")
                for mzi_id in init_mzi_channels
            ]
            init_mzi_voltages = list(
                exp_config.calibration.initial_mzi_voltages.values()
            )
            v_ctrl.set_voltages(
                init_psu_channels + [mzi_channel],
                init_mzi_voltages + [voltage],
                v_max=scan_config.v_max,
            )

            # b. Wait for thermal settling
            time.sleep(scan_config.settling_time_sec)

            # c. Measure spectrum
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if scan_config.save_raw_data:
                file_name = (
                    f"v2pi_scan_mzi_{scan_config.mzi_id}_{voltage:.3f}v_{timestamp}"
                )
                folder_dir = str(output_path)
            else:
                file_name = None
                folder_dir = None

            df = measure_spectrum(
                center_wavelength_nm=exp_config.measurement.center_wavelength_nm,
                wavelength_span_nm=exp_config.measurement.wavelength_span_nm,
                num_averages=exp_config.measurement.num_averages,
                edfa_port=exp_config.measurement.edfa_port,
                edfa_baudrate=exp_config.measurement.edfa_baudrate,
                edfa_output_power_dbm=exp_config.measurement.edfa_output_power_dbm,
                ova_ip=exp_config.measurement.ova_address,
                folder_dir=folder_dir,
                file_name=file_name,
            )

            time.sleep(scan_config.settling_time_sec)
            print("Exit voltage controller context - heaters should be zeroed")

        dataframes.append(df)

        # d. Recover tap coefficients via Kramers-Kronig
        time_ps, h_time = recover_impulse_response_from_df(
            df=df,
            fsr_hz=exp_config.chip.fsr_hz,
            wavelength_col=exp_config.measurement.wavelength_col,
            freq_col=exp_config.measurement.frequency_col,
            insertion_loss_col=exp_config.measurement.insertion_loss_col,
        )

        # Detect taps
        tap_times, tap_coeffs = detect_taps(
            time_ps=time_ps,
            h_time=h_time,
            fsr_hz=exp_config.chip.fsr_hz,
            delay_step_s=exp_config.chip.delay_step_s,
            n_taps=exp_config.chip.n_taps,
            prominence_factor_db=exp_config.measurement.prominence_factor_db,
            height_threshold_db=exp_config.measurement.height_threshold_db,
        )

        # Get all power splitting ratios from tap coefficients
        psr_dict = tap_coeffs_to_power_splitting_ratios(tap_coeffs, mzi_tree)

        # Extract the specific MZI's PSR
        psr_db = psr_dict.get(scan_config.mzi_id, 0.0)
        power_splitting_ratios[i] = psr_db

        # Convert PSR to MZI phase
        mzi_phase_rad = power_splitting_ratio_to_mzi_phase(psr_db)
        mzi_phases[i] = mzi_phase_rad

        print(f"  PSR: {psr_db:+.3f} dB, MZI phase: {mzi_phase_rad:.4f} rad")
        print()

    print(f"✓ Scan complete - voltage controller channels zeroed\n")

    return voltage_range, power_splitting_ratios, mzi_phases, dataframes


def save_results(
    scan_config: V2piScanConfig,
    exp_config: ExperimentConfig,
    voltages: np.ndarray,
    psr: np.ndarray,
    phases: np.ndarray,
    v_2pi: float,
    phi_init: float,
    r_squared: float,
    fit_info: Dict,
):
    """Save scan results to YAML file."""

    results = {
        "scan_parameters": scan_config.to_dict(),
        "mzi_info": {
            "mzi_id": scan_config.mzi_id,
            "channel": exp_config.channel_mapping.get_channel(
                f"MZI_{scan_config.mzi_id}"
            ),
        },
        "fit_results": {
            "v_2pi_volts": float(v_2pi),
            "phi_init_rad": float(phi_init),
            "phi_init_deg": float(np.degrees(phi_init)),
            "p_2pi_watts": float(fit_info["p_2pi_watts"]),
            "p_2pi_mw": float(fit_info["p_2pi_watts"] * 1000),
            "r_squared": float(r_squared),
            "rmse_db": float(fit_info["rmse_db"]),
            "resistance_ohm": float(exp_config.chip.heater_resistance_ohm),
        },
        "data_ranges": {
            "voltage_range": {
                "min": float(np.min(voltages)),
                "max": float(np.max(voltages)),
                "n_points": int(len(voltages)),
            },
            "psr_range": {
                "min": float(np.min(psr)),
                "max": float(np.max(psr)),
            },
            "phase_range": {
                "min": float(np.min(phases)),
                "max": float(np.max(phases)),
            },
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_path = (
        Path(scan_config.output_dir) / f"scan_results_{scan_config.mzi_id}.yaml"
    )
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Results summary saved: {output_path}")


def characterise_mzi(
    mzi_id: str,
    base_output_dir: str,
    exp_config: ExperimentConfig,
    v_min: float = 0.0,
    v_max: float = 30.0,
    n_points: int = 51,
    settling_time: float = 2.0,
    save_raw_data: bool = True,
):
    """
    Full characterisation workflow for a single MZI's V_2π and φ_init.

    This is the high-level orchestration function that:
    1. Sets up the scan configuration
    2. Initialises reference MZIs (1-1, 2-1, 3-1, 4-1) to known states
    3. Performs the voltage sweep
    4. Fits V_2π and φ_init using nonlinear least squares
    5. Plots results (PSR vs V² and fit overlay)
    6. Saves summary

    Parameters
    ----------
    mzi_id : str
        MZI identifier (e.g., "4-6", "2-1")
    base_output_dir : str
        Base directory for outputs
    exp_config : ExperimentConfig
        Experiment configuration object (instruments, chip params, etc.)
    v_min : float
        Minimum voltage (V)
    v_max : float
        Maximum voltage (V)
    n_points : int
        Number of voltage points to measure
    settling_time : float
        Thermal settling time after voltage change (seconds)
    save_raw_data : bool
        Whether to save individual CSV files for each voltage point
    """

    # ============================================================
    # CREATE SCAN CONFIGURATION
    # ============================================================

    # Output - use subdirectory for each MZI
    output_dir = str(Path(base_output_dir) / f"mzi_{mzi_id}")

    # Create scan configuration
    scan_config = V2piScanConfig(
        mzi_id=mzi_id,
        v_min=v_min,
        v_max=v_max,
        n_points=n_points,
        settling_time_sec=settling_time,
        output_dir=output_dir,
        save_raw_data=save_raw_data,
        min_stage=1,
        max_stage=4,
    )

    print(f"Chip: {exp_config.chip.n_taps}-tap FIR")
    print(f"FSR: {exp_config.chip.fsr_hz/1e9:.1f} GHz")

    R = exp_config.chip.heater_resistance_ohm

    # ============================================================
    # PERFORM VOLTAGE SWEEP
    # ============================================================

    voltages, psrs, phases, dataframes = perform_voltage_sweep(
        scan_config=scan_config,
        exp_config=exp_config,
    )

    # ============================================================
    # FIT V_2π AND φ_init USING NONLINEAR LEAST SQUARES
    # ============================================================

    # Let the function auto-estimate initial guesses from the data
    v_2pi, phi_init, r_squared, fit_info = fit_mzi_v2pi_and_phi_init(
        voltages=voltages,
        psr_db=psrs,
        resistance_ohm=R,
    )

    # Check if fit was successful
    fit_successful = r_squared > 0.5 and fit_info["fitted_psr"] is not None

    if not fit_successful:
        print("\n⚠ Warning: Nonlinear fit did not converge well (R² < 0.5)")
        print("   Proceeding with data-only plot")

    # Print results
    print_fit_results(
        mzi_id=scan_config.mzi_id,
        v_2pi=v_2pi,
        phi_init=phi_init,
        r_squared=r_squared,
        fit_info=fit_info,
        resistance_ohm=R,
    )

    # ============================================================
    # PLOT RESULTS
    # ============================================================

    # Single plot: PSR vs V² with data and fitted curve
    # Only show fit if it was successful
    fig = plot_mzi_characterisation(
        voltages=voltages,
        psr_db=psrs,
        mzi_id=scan_config.mzi_id,
        v_2pi=v_2pi if fit_successful else None,
        phi_init=phi_init if fit_successful else None,
        fit_info=fit_info if fit_successful else None,
        resistance_ohm=R,
        output_dir=scan_config.output_dir,
    )

    # ============================================================
    # SAVE SUMMARY
    # ============================================================

    save_results(
        scan_config=scan_config,
        exp_config=exp_config,
        voltages=voltages,
        psr=psrs,
        phases=phases,
        v_2pi=v_2pi,
        phi_init=phi_init,
        r_squared=r_squared,
        fit_info=fit_info,
    )

    print(f"\n{'='*70}")
    print("MZI Characterisation Complete!")
    print(f"{'='*70}")
    print(f"MZI {scan_config.mzi_id}")
    print(f"V_2π:   {v_2pi:.3f} V")
    print(f"φ_init: {phi_init:.4f} rad ({np.degrees(phi_init):.2f}°)")
    print(f"P_2π:   {fit_info['p_2pi_watts']*1000:.2f} mW")
    print(f"R²:     {r_squared:.6f}")
    print(f"Results saved to: {scan_config.output_dir}")
    print(f"{'='*70}\n")


def main():
    """
    Main execution - batch scan all MZIs on the chip.
    """

    # ============================================================
    # CONFIGURATION - Edit these parameters for your batch scan
    # ============================================================

    # Experiment configuration file
    CONFIG_PATH = "measurements/experiment_config_shorter_range_reduce_num_avg.yaml"

    # Voltage scan parameters
    V_MIN = 0.0  # Minimum voltage (V)
    V_MAX = 20.0  # Maximum voltage (V)
    N_POINTS = 31  # Number of voltage points (31 = ~0.67V steps for 0-20V)

    # Timing
    SETTLING_TIME = 2.0  # Thermal settling time after voltage change (seconds)

    # Output
    SAVE_RAW_DATA = True  # Save individual CSV files for each voltage point

    # ============================================================
    # LOAD CONFIGURATION (once for all MZIs)
    # ============================================================

    exp_config = load_config(CONFIG_PATH)

    print(f"\n{'='*70}")
    print(f"CONFIGURATION LOADED")
    print(f"{'='*70}")
    print(f"Config file: {CONFIG_PATH}")
    print(f"Chip: {exp_config.chip.n_taps}-tap FIR filter")
    print(f"FSR: {exp_config.chip.fsr_hz/1e9:.3f} GHz")
    print(f"P_2π: {exp_config.chip.p2pi_watts*1000:.1f} mW")
    print(f"Heater resistance: {exp_config.chip.heater_resistance_ohm} Ω")
    print(f"Initial MZI voltages: {exp_config.calibration.initial_mzi_voltages}")
    print(f"{'='*70}\n")

    # ============================================================
    # DETERMINE MZIs TO SCAN
    # ============================================================

    # Get next available run directory
    base_output_dir = get_next_run_dir(
        base_dir="measurements",
        prefix="v2pi_batch_scan_results",
    )

    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=f"{base_output_dir}/batch_mzi_scan.log", level="INFO")

    # All MZIs on the chip
    all_mzi_ids = exp_config.chip.get_all_mzi_ids()

    # Exclude MZIs in the first position of each stage (plus reference MZI)
    # excluded_mzis = {"1-1", "2-1", "3-1", "4-1", "4-5", "4-6", "4-7", "4-8"}
    excluded_mzis = {}
    mzi_ids = [mzi_id for mzi_id in all_mzi_ids if mzi_id not in excluded_mzis]

    print(f"\n{'='*70}")
    print(f"BATCH V_2π CHARACTERISATION")
    print(f"{'='*70}")
    print(f"Number of MZIs to scan: {len(mzi_ids)}")
    print(f"MZI IDs: {mzi_ids}")
    print(f"Skipping: 1-1 (reference MZI)")
    print(f"Base output directory: {base_output_dir}")
    print(f"{'='*70}\n")

    # ============================================================
    # RUN CHARACTERISATION FOR EACH MZI
    # ============================================================

    # Run scan for each MZI
    for i, mzi_id in enumerate(mzi_ids):
        print(f"\n{'#'*70}")
        print(f"# CHARACTERISING MZI {i+1}/{len(mzi_ids)}: {mzi_id}")
        print(f"{'#'*70}\n")

        try:
            characterise_mzi(
                mzi_id=mzi_id,
                base_output_dir=base_output_dir,
                exp_config=exp_config,
                v_min=V_MIN,
                v_max=V_MAX,
                n_points=N_POINTS,
                settling_time=SETTLING_TIME,
                save_raw_data=SAVE_RAW_DATA,
            )
        except Exception as e:
            print(f"⚠ FAILED to characterise MZI {mzi_id}: {e}")
            print(f"Continuing with next MZI...\n")
            continue

        # Add delay between scans to allow thermal settling
        if i < len(mzi_ids) - 1:
            print("\nWaiting 5 seconds before next scan...\n")
            time.sleep(5)

    # ============================================================
    # ZERO ALL HEATERS AT END OF BATCH, redundant safety
    # ============================================================

    with VoltageController(
        com_port=exp_config.measurement.voltage_controller_port,
        baud_rate=exp_config.measurement.voltage_controller_baudrate,
        zero_on_exit=True,
    ) as v_ctrl:
        v_ctrl.set_voltages(
            channels=np.arange(1, 32 + 1),
            voltages=[0.0] * 32,
            v_max=30.0,
        )

    print(f"\n{'='*70}")
    print(f"BATCH SCAN COMPLETE!")
    print(f"{'='*70}")
    print(f"Characterised {len(mzi_ids)} MZIs")
    print(f"Results saved to: {base_output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
