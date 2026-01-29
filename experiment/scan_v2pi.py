"""
scan_v2pi.py

Voltage scan script to characterise P_2π (V_π) for individual MZIs.

Workflow:
1. Load experiment configuration
2. Scan voltage range for specified MZI
3. For each voltage setting:
   a. Set voltage on MZI
   b. Wait for thermal settling
   c. Measure optical spectrum
   d. Recover tap coefficients via Kramers-Kronig
   e. Calculate power splitting ratio for the target MZI
4. Plot results and estimate V_2π from linear fit

Usage:
    # Configure parameters in main() and run
    python scan_v2pi.py
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

    # Experiment config path
    experiment_config_path: str = "configs/v2pi_scan_config.yaml"

    # MZI tree structure parameters
    min_stage: int = 1
    max_stage: int = 4

    def get_voltage_range(self) -> np.ndarray:
        """Generate voltage array from parameters."""
        return np.linspace(self.v_min, self.v_max, self.n_points)

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


def scan_mzi_v2pi(
    scan_config: V2piScanConfig,
    exp_config: ExperimentConfig,
    mzi_ids: List[str] = None,  # ← New parameter
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.DataFrame]]:
    """
    Scan voltage on a single MZI to characterise P_2π.

    Parameters
    ----------
    scan_config : V2piScanConfig
        Scan configuration (voltage range, MZI ID, etc.)
    exp_config : ExperimentConfig
        Experiment configuration (chip parameters, instruments, etc.)
    mzi_ids : List[str], optional
        List of MZI IDs to include in tree structure.
        If None, uses exp_config.chip.get_mzi_ids() (signal MZIs only)

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
    if mzi_ids is None:
        mzi_ids = exp_config.chip.get_mzi_ids()  # Default: signal MZIs only

    mzi_tree = exp_config.signal_mzi_tree.tree

    # Storage for results
    power_splitting_ratios = np.zeros(n_voltages)
    mzi_phases = np.zeros(n_voltages)
    dataframes = []

    print(f"\n{'='*70}")
    print(f"V_2π Scan for MZI {scan_config.mzi_id}")
    print(f"{'='*70}")
    print(f"MZI ID: {scan_config.mzi_id}")
    print(f"Channel: {mzi_channel}")
    print(f"Voltage range: {scan_config.v_min:.2f} - {scan_config.v_max:.2f} V")
    print(f"Number of points: {n_voltages}")
    print(f"Settling time: {scan_config.settling_time_sec} s")
    print(f"Output directory: {output_path}")
    print(f"{'='*70}\n")

    # Get voltage controller parameters
    resistance = exp_config.chip.heater_resistance_ohm

    # Measure DUT length for delay calculations
    with LunaOVA(ip=exp_config.measurement.ova_address) as ova:
        ova.set_dut_length()

    # Initialise voltage controller
    with VoltageController(
        com_port=exp_config.measurement.voltage_controller_port,
        baud_rate=exp_config.measurement.voltage_controller_baudrate,
        zero_on_exit=True,
    ) as v_ctrl:

        print("✓ Voltage controller initialised\n")

        # Scan through voltages
        for i, voltage in enumerate(voltage_range):
            print(f"[{i+1}/{n_voltages}] Voltage: {voltage:.3f} V")

            # a. Set voltage
            v_ctrl.set_voltages([mzi_channel], [voltage], v_max=scan_config.v_max)

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

            # # e. Calculate power splitting ratio for target MZI
            # # Extract signal processing taps (indices 8-15 for 16-tap chip)
            # signal_tap_indices = exp_config.chip.signal_tap_indices
            # signal_taps = tap_coeffs[list(signal_tap_indices)] # Might be the cause of the issue

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


def plot_v2pi_scan(
    voltages: np.ndarray,
    power_splitting_ratios: np.ndarray,
    mzi_phases: np.ndarray,
    scan_config: V2piScanConfig,
    v_2pi_estimate: float = None,
):
    """
    Plot V_2π scan results.

    Parameters
    ----------
    voltages : np.ndarray
        Voltage values
    power_splitting_ratios : np.ndarray
        Power splitting ratios in dB
    mzi_phases : np.ndarray
        MZI phases in radians
    scan_config : V2piScanConfig
        Scan configuration
    v_2pi_estimate : float, optional
        Estimated V_2π to mark on plot
    """

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Power splitting ratio vs voltage
    ax1 = axes[0]
    ax1.plot(voltages, power_splitting_ratios, "bo-", linewidth=2, markersize=4)
    ax1.set_ylabel("Power Splitting Ratio (dB)", fontsize=12)
    ax1.set_title(
        f"V_2π Scan for MZI {scan_config.mzi_id}", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="50:50 split")
    ax1.legend()

    # Plot 2: MZI phase vs voltage
    ax2 = axes[1]
    ax2.plot(voltages, mzi_phases, "ro-", linewidth=2, markersize=4)
    ax2.set_xlabel("Voltage (V)", fontsize=12)
    ax2.set_ylabel("MZI Phase (rad)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Mark 0, π, and 2π phase levels
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3, label="0")
    ax2.axhline(y=np.pi, color="gray", linestyle="--", alpha=0.3, label="π")
    ax2.axhline(y=2 * np.pi, color="gray", linestyle="--", alpha=0.3, label="2π")

    # Mark V_2π estimate if provided
    if v_2pi_estimate is not None:
        ax2.axvline(
            x=v_2pi_estimate,
            color="green",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=f"V_2π ≈ {v_2pi_estimate:.2f} V",
        )

    ax2.legend()

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = (
        Path(scan_config.output_dir) / f"v2pi_scan_{scan_config.mzi_id}_{timestamp}.png"
    )
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"✓ Figure saved: {fig_path}")

    # plt.show()


def estimate_v2pi(voltages: np.ndarray, mzi_phases: np.ndarray) -> float:
    """
    Estimate V_2π from phase vs voltage data using linear fit.

    The MZI phase should follow φ = (π/V_π) × V for small phases,
    so we fit φ = slope × V and extract V_2π = 2π/slope.

    Parameters
    ----------
    voltages : np.ndarray
        Voltage values
    mzi_phases : np.ndarray
        MZI phases in radians

    Returns
    -------
    v_2pi : float
        Estimated V_2π in volts
    """

    # Linear fit: phase = slope * voltage + intercept
    coeffs = np.polyfit(voltages, mzi_phases, deg=1)
    slope = coeffs[0]  # rad/V
    intercept = coeffs[1]  # rad

    # V_2π from slope
    v_2pi = 2 * np.pi / slope

    # Calculate R² for goodness of fit
    phase_fit = slope * voltages + intercept
    ss_res = np.sum((mzi_phases - phase_fit) ** 2)
    ss_tot = np.sum((mzi_phases - np.mean(mzi_phases)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\n{'='*70}")
    print("V_2π Estimation (Linear Fit)")
    print(f"{'='*70}")
    print(f"Slope: {slope:.4f} rad/V")
    print(f"Intercept: {intercept:.4f} rad")
    print(f"R²: {r_squared:.4f}")
    print(f"Estimated V_2π: {v_2pi:.3f} V")

    # Corresponding power (P = V²/R)
    # Typical resistance ~600 Ω, but this depends on your chip
    print(f"\nNote: For power calculation, check your heater resistance")
    print(f"{'='*70}\n")

    return v_2pi


def save_results(
    scan_config: V2piScanConfig,
    exp_config: ExperimentConfig,
    voltages: np.ndarray,
    psr: np.ndarray,
    phases: np.ndarray,
    v_2pi: float,
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
        "results": {
            "v_2pi_estimate": float(v_2pi),
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


def run_v2pi_scan(mzi_id: str, base_output_dir: str):
    """
    Run V_2π scan for a single MZI.

    Parameters
    ----------
    mzi_id : str
        MZI identifier (e.g., "4-6", "2-1")
    """

    # ============================================================
    # CONFIGURATION - Edit these parameters for your scan
    # ============================================================

    # Voltage scan parameters
    V_MIN = 0.0  # Minimum voltage (V)
    V_MAX = 30.0  # Maximum voltage (V)
    N_POINTS = 51  # Number of voltage points (51 = ~0.6V steps for 0-30V)

    # Timing
    SETTLING_TIME = 2.0  # Thermal settling time after voltage change (seconds)

    # Output - use subdirectory for each MZI
    OUTPUT_DIR = str(Path(base_output_dir) / f"mzi_{mzi_id}")
    SAVE_RAW_DATA = True  # Save individual CSV files for each voltage point

    # Experiment configuration file
    CONFIG_PATH = "measurements/experiment_config_shorter_range.yaml"
    MIN_STAGE = 1
    MAX_STAGE = 4

    # ============================================================

    # Create scan configuration
    scan_config = V2piScanConfig(
        mzi_id=mzi_id,
        v_min=V_MIN,
        v_max=V_MAX,
        n_points=N_POINTS,
        settling_time_sec=SETTLING_TIME,
        output_dir=OUTPUT_DIR,
        save_raw_data=SAVE_RAW_DATA,
        experiment_config_path=CONFIG_PATH,
        min_stage=MIN_STAGE,
        max_stage=MAX_STAGE,
    )

    exp_config = load_config(CONFIG_PATH)

    # Get all MZI IDs from scan config
    all_mzi_ids = scan_config.get_all_mzi_ids()

    print(f"Chip: {exp_config.chip.n_taps}-tap FIR")
    print(f"FSR: {exp_config.chip.fsr_hz/1e9:.1f} GHz")
    print(f"Including ALL MZIs in tree structure ({len(all_mzi_ids)} total)\n")

    # Run voltage scan
    voltages, psrs, phases, dataframes = scan_mzi_v2pi(
        scan_config=scan_config,
        exp_config=exp_config,
        mzi_ids=all_mzi_ids,
    )

    # Estimate V_2π from linear fit
    v_2pi = estimate_v2pi(voltages, phases)

    # Plot results
    plot_v2pi_scan(
        voltages=voltages,
        power_splitting_ratios=psrs,
        mzi_phases=phases,
        scan_config=scan_config,
        v_2pi_estimate=None,
    )

    # Save summary
    save_results(
        scan_config=scan_config,
        exp_config=exp_config,
        voltages=voltages,
        psr=psrs,
        phases=phases,
        v_2pi=v_2pi,
    )

    print(f"\n{'='*70}")
    print("Scan Complete!")
    print(f"{'='*70}")
    print(f"MZI {scan_config.mzi_id}")
    print(f"Estimated V_2π: {v_2pi:.3f} V")
    print(f"Results saved to: {scan_config.output_dir}")
    print(f"{'='*70}\n")


def main():
    """
    Main execution - batch scan all MZIs.
    """
    # Get next available run directory
    base_output_dir = get_next_run_dir(
        base_dir="measurements",
        prefix="v2pi_batch_scan_results",
    )

    # All MZIs on the chip
    # Create a temporary config just to generate MZI IDs
    temp_config = V2piScanConfig(min_stage=1, max_stage=4)
    mzi_ids = temp_config.get_all_mzi_ids()

    print(f"\n{'='*70}")
    print(f"BATCH V_2π CHARACTERISATION")
    print(f"{'='*70}")
    print(f"Number of MZIs to scan: {len(mzi_ids)}")
    print(f"MZI IDs: {mzi_ids}")
    print(f"{'='*70}\n")

    # Run scan for each MZI
    for i, mzi_id in enumerate(mzi_ids):
        print(f"\n{'#'*70}")
        print(f"# SCANNING MZI {i+1}/{len(mzi_ids)}: {mzi_id}")
        print(f"{'#'*70}\n")

        run_v2pi_scan(mzi_id, base_output_dir)

        # Add delay between scans to allow thermal settling
        if i < len(mzi_ids) - 1:
            print("\nWaiting 5 seconds before next scan...\n")
            time.sleep(5)

    print(f"\n{'='*70}")
    print(f"BATCH SCAN COMPLETE!")
    print(f"{'='*70}")
    print(f"Scanned {len(mzi_ids)} MZIs")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
