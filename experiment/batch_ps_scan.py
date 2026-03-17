"""
batch_ps_scan.py

Batch voltage scan to characterise V_2π and φ_init for individual phase shifters (PSs).

Unlike MZIs, phase shifters have a LINEAR relationship between applied power and phase:

    φ(V) = 2π · V² / V_2π² + φ_init

This means:
  - No tan² nonlinearity to deal with
  - The observable is the tap PHASE (not PSR)
  - Fitting is a simple linear regression on unwrapped phase vs V²
  - φ_init is the y-intercept; V_2π is derived from the slope

Each phase shifter (PS) is associated with one tap in the signal processing core.
The tap phase is extracted from the complex tap coefficients after impulse response
recovery. Phase wrapping must be handled carefully before fitting.

Usage:
    # Configure parameters in main() and run
    python batch_ps_scan.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from scipy.stats import linregress

from voltage_ctrl import VoltageController
from luna_ova import LunaOVA

from photonic_fir import (
    ExperimentConfig,
    load_config,
    get_next_run_dir,
    setup_logging,
)

from photonic_fir.calibration import measure_and_detect_taps

# ==============================================================================
# CONFIGURATION DATACLASS
# ==============================================================================


@dataclass
class PSScanConfig:
    """Configuration for a single phase shifter V_2π voltage scan."""

    # Target phase shifter tap number (1-indexed, matches ps_channels mapping)
    tap_num: int = 9

    # Voltage scan parameters
    v_min: float = 0.0
    v_max: float = 20.0
    n_points: int = 31  # 31 points → ~0.67 V steps over 0-20 V

    # Timing
    settling_time_sec: float = 2.0

    # Output
    output_dir: str = "./ps_v2pi_scan_results"
    save_raw_data: bool = True

    def get_voltage_range(self) -> np.ndarray:
        """
        Generate voltage array with uniform V² spacing.

        Uniform spacing in V² ensures uniform power steps (P = V²/R),
        giving even phase increments (φ ∝ P for a PS).
        """
        v_squared = np.linspace(self.v_min**2, self.v_max**2, self.n_points)
        return np.sqrt(v_squared)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PSScanConfig":
        return cls(**d)


# ==============================================================================
# PHYSICS / FITTING
# ==============================================================================


def ps_phase_model(
    voltage_squared: np.ndarray, v_2pi_squared: float, phi_init: float
) -> np.ndarray:
    """
    Linear phase model for a thermo-optic phase shifter.

    The phase shift is proportional to applied optical power:

        φ(V²) = 2π · V² / V_2π² + φ_init

    Parameters
    ----------
    voltage_squared : np.ndarray
        Applied V² values (V²)
    v_2pi_squared : float
        V_2π² — the V² that produces a 2π phase shift (V²)
    phi_init : float
        Initial phase offset at V=0 (radians)

    Returns
    -------
    phase_rad : np.ndarray
        Predicted phase (radians), NOT wrapped
    """
    return 2 * np.pi * voltage_squared / v_2pi_squared + phi_init


def fit_ps_v2pi_and_phi_init(
    voltages: np.ndarray,
    phases_rad: np.ndarray,
    resistance_ohm: float,
) -> Tuple[float, float, float, Dict]:
    """
    Fit V_2π and φ_init for a phase shifter using linear regression.

    Because φ = (2π/V_2π²)·V² + φ_init, this is a simple linear regression
    of unwrapped phase against V². The slope gives V_2π directly.

    Parameters
    ----------
    voltages : np.ndarray
        Applied voltage values (V)
    phases_rad : np.ndarray
        Measured tap phases (radians) — will be unwrapped internally
    resistance_ohm : float
        Heater resistance (Ω), used to compute P_2π

    Returns
    -------
    v_2pi : float
        Voltage for 2π phase shift (V)
    phi_init : float
        Initial phase offset (radians), wrapped to [-π, π]
    r_squared : float
        Coefficient of determination (goodness of fit)
    fit_info : Dict
        slope, intercept, p_2pi_watts, residuals, rmse_rad, fitted_phases
    """
    voltage_squared = voltages**2

    # Unwrap phase to remove ±π jumps before fitting
    phases_unwrapped = np.unwrap(phases_rad)

    # Linear regression: φ = slope · V² + intercept
    result = linregress(voltage_squared, phases_unwrapped)
    slope = result.slope  # rad / V²
    intercept = result.intercept  # rad (= φ_init)
    r_squared = result.rvalue**2

    # Derive V_2π from slope: slope = 2π / V_2π²
    if slope > 1e-12:
        v_2pi = np.sqrt(2 * np.pi / slope)
    else:
        # Degenerate case — no measurable phase change
        v_2pi = voltages[-1]

    # Wrap φ_init back to [-π, π]
    phi_init = (intercept + np.pi) % (2 * np.pi) - np.pi

    # Fitted phases (unwrapped)
    fitted_phases_unwrapped = slope * voltage_squared + intercept

    # Residuals
    residuals = phases_unwrapped - fitted_phases_unwrapped
    rmse = np.sqrt(np.mean(residuals**2))

    p_2pi = v_2pi**2 / resistance_ohm

    fit_info = {
        "slope_rad_per_v2": slope,
        "intercept_rad": intercept,
        "p_2pi_watts": p_2pi,
        "residuals": residuals,
        "rmse_rad": rmse,
        "fitted_phases_unwrapped": fitted_phases_unwrapped,
        "phases_unwrapped": phases_unwrapped,
        "voltage_squared": voltage_squared,
    }

    return v_2pi, phi_init, r_squared, fit_info


def print_ps_fit_results(
    tap_num: int,
    v_2pi: float,
    phi_init: float,
    r_squared: float,
    fit_info: Dict,
    resistance_ohm: float,
) -> None:
    """Print formatted fitting results for a phase shifter."""
    p_2pi = fit_info["p_2pi_watts"]
    rmse = fit_info["rmse_rad"]

    print(f"\n{'='*60}")
    print(f"  Phase Shifter Tap {tap_num} — Fit Results")
    print(f"{'='*60}")
    print(f"  V_2π:      {v_2pi:.3f} V")
    print(f"  P_2π:      {p_2pi * 1000:.2f} mW")
    print(f"  φ_init:    {phi_init:.4f} rad  ({np.degrees(phi_init):.2f}°)")
    print(f"  Slope:     {fit_info['slope_rad_per_v2']:.6f} rad/V²")
    print(f"  R²:        {r_squared:.6f}")
    print(f"  RMSE:      {rmse:.4f} rad")
    print(f"  R (heater): {resistance_ohm:.0f} Ω")
    print(f"{'='*60}\n")


# ==============================================================================
# PLOTTING
# ==============================================================================


def plot_ps_characterisation(
    voltages: np.ndarray,
    phases_rad: np.ndarray,
    tap_num: int,
    v_2pi: Optional[float] = None,
    phi_init: Optional[float] = None,
    fit_info: Optional[Dict] = None,
    resistance_ohm: Optional[float] = None,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Plot phase shifter characterisation data with linear fit overlay.

    Shows measured tap phase vs V² (power axis) with fitted line.

    Parameters
    ----------
    voltages : np.ndarray
        Applied voltages (V)
    phases_rad : np.ndarray
        Measured tap phases (radians)
    tap_num : int
        Phase shifter tap number
    v_2pi : float, optional
        Fitted V_2π (V)
    phi_init : float, optional
        Fitted φ_init (radians)
    fit_info : dict, optional
        Additional fit information from fit_ps_v2pi_and_phi_init()
    resistance_ohm : float, optional
        Heater resistance (Ω)
    output_dir : str, optional
        Directory to save figure

    Returns
    -------
    fig : plt.Figure
    """
    voltage_squared = voltages**2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Phase Shifter Tap {tap_num} Characterisation", fontsize=14, fontweight="bold"
    )

    # --- Left panel: unwrapped phase vs V² ---
    ax = axes[0]

    phases_unwrapped = (
        fit_info["phases_unwrapped"] if fit_info is not None else np.unwrap(phases_rad)
    )

    ax.plot(
        voltage_squared,
        phases_unwrapped,
        "o",
        color="steelblue",
        markersize=5,
        label="Measured (unwrapped)",
        zorder=3,
    )

    if fit_info is not None and v_2pi is not None:
        v2_fine = np.linspace(0, voltage_squared[-1], 300)
        phase_fit = ps_phase_model(v2_fine, v_2pi**2, fit_info["intercept_rad"])
        ax.plot(
            v2_fine,
            phase_fit,
            "-",
            color="crimson",
            linewidth=1.8,
            label=f"Linear fit\nV_2π={v_2pi:.2f} V, φ_init={np.degrees(phi_init):.1f}°\nR²={fit_info.get('rmse_rad', 0):.4f} rad RMSE",
        )

    ax.set_xlabel("V² (V²)")
    ax.set_ylabel("Tap phase, unwrapped (rad)")
    ax.set_title("Phase vs V² (unwrapped)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    # --- Right panel: residuals ---
    ax2 = axes[1]
    if fit_info is not None and fit_info.get("residuals") is not None:
        ax2.plot(
            voltage_squared,
            np.degrees(fit_info["residuals"]),
            "s",
            color="darkorange",
            markersize=5,
        )
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        rmse_deg = np.degrees(fit_info["rmse_rad"])
        ax2.axhline(
            +rmse_deg,
            color="grey",
            linewidth=0.8,
            linestyle=":",
            label=f"±RMSE = {rmse_deg:.2f}°",
        )
        ax2.axhline(-rmse_deg, color="grey", linewidth=0.8, linestyle=":")
        ax2.set_xlabel("V² (V²)")
        ax2.set_ylabel("Residual (°)")
        ax2.set_title("Fit Residuals")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.35)
    else:
        # Fallback: raw (wrapped) phase
        ax2.plot(
            voltage_squared,
            np.degrees(phases_rad),
            "o",
            color="steelblue",
            markersize=5,
        )
        ax2.set_xlabel("V² (V²)")
        ax2.set_ylabel("Tap phase (°)")
        ax2.set_title("Measured Phase (wrapped)")
        ax2.grid(True, alpha=0.35)

    plt.tight_layout()

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = output_path / f"ps_scan_tap{tap_num}_{timestamp}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"✓ Plot saved: {fig_path}")

    return fig


# ==============================================================================
# MEASUREMENT
# ==============================================================================


def perform_ps_voltage_sweep(
    scan_config: PSScanConfig,
    exp_config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, List[pd.DataFrame]]:
    """
    Perform voltage sweep on a single phase shifter, measuring tap phase at each step.

    At each voltage:
      1. Set the target PS channel to the current voltage (all other PSs at 0 V)
      2. Wait for thermal settling
      3. Measure spectrum and recover impulse response → tap coefficients
      4. Extract the phase of the tap associated with this PS

    Parameters
    ----------
    scan_config : PSScanConfig
        Scan configuration
    exp_config : ExperimentConfig
        Experiment configuration

    Returns
    -------
    voltages : np.ndarray
        Voltage values scanned
    tap_phases : np.ndarray
        Measured tap phases (radians, wrapped)
    dataframes : List[pd.DataFrame]
        Raw measurement DataFrames for each voltage point
    """
    output_path = Path(scan_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    voltage_range = scan_config.get_voltage_range()
    n_voltages = len(voltage_range)

    # Resolve hardware channel for this PS
    ps_device_id = f"PS_{scan_config.tap_num}"
    ps_channel = exp_config.channel_mapping.get_channel(ps_device_id)

    # The tap index in the tap_coeffs array.
    # tap_coeffs are 0-indexed; tap_num is 1-indexed.
    tap_index = scan_config.tap_num - 1

    tap_phases = np.zeros(n_voltages)
    dataframes = []

    print(f"\n{'='*70}")
    print(f"V_2π Voltage Sweep — Phase Shifter Tap {scan_config.tap_num}")
    print(f"{'='*70}")
    print(f"PS channel:    {ps_channel}")
    print(f"Tap index:     {tap_index}  (0-indexed)")
    print(f"Voltage range: {scan_config.v_min:.2f} – {scan_config.v_max:.2f} V")
    print(f"N points:      {n_voltages}")
    print(f"Settling time: {scan_config.settling_time_sec} s")
    print(f"Output dir:    {output_path}")
    print(f"{'='*70}\n")

    # Set OVA DUT length once before sweep
    with LunaOVA(ip=exp_config.measurement.ova_address) as ova:
        ova.set_dut_length()

    for i, voltage in enumerate(voltage_range):
        print(
            f"  [{i+1:02d}/{n_voltages}] V = {voltage:.3f} V  (V² = {voltage**2:.2f} V²)"
        )

        with VoltageController(
            com_port=exp_config.measurement.voltage_controller_port,
            baud_rate=exp_config.measurement.voltage_controller_baudrate,
            zero_on_exit=True,  # Safety: zero all heaters on context exit
        ) as v_ctrl:

            init_mzi_channels = list(exp_config.calibration.initial_mzi_voltages.keys())

            init_psu_channels = [
                exp_config.channel_mapping.get_channel(f"MZI_{mzi_id}")
                for mzi_id in init_mzi_channels
            ]
            init_mzi_voltages = list(
                exp_config.calibration.initial_mzi_voltages.values()
            )

            # Apply voltage to this PS only; all others remain at 0 V (zeroed on entry)
            v_ctrl.set_voltages(
                channels=init_psu_channels + [ps_channel],
                voltages=init_mzi_voltages + [voltage],
                v_max=scan_config.v_max,
            )

            # Thermal settling
            time.sleep(scan_config.settling_time_sec)

            # Measure spectrum and detect taps
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if scan_config.save_raw_data:
                file_name = (
                    f"ps_scan_tap{scan_config.tap_num}_{voltage:.3f}v_{timestamp}"
                )
                folder_dir = str(output_path)
            else:
                file_name = None
                folder_dir = None

            df, tap_times, tap_coeffs, _, _ = measure_and_detect_taps(
                config=exp_config,
                file_name=file_name,
                folder_dir=folder_dir,
            )

            time.sleep(scan_config.settling_time_sec)

        dataframes.append(df)

        # Extract phase from the complex tap coefficient for this tap
        if tap_index < len(tap_coeffs):
            coeff = tap_coeffs[tap_index]
            phase = np.angle(coeff)  # wrapped to [-π, π]
        else:
            print(
                f"  ⚠ tap_index {tap_index} out of range (len={len(tap_coeffs)}), storing NaN"
            )
            phase = np.nan

        tap_phases[i] = phase
        print(f"       → tap phase = {np.degrees(phase):+.2f}°  ({phase:+.4f} rad)")

    print(f"\n✓ Sweep complete — PS heater zeroed\n")
    return voltage_range, tap_phases, dataframes


# ==============================================================================
# SAVE RESULTS
# ==============================================================================


def save_ps_results(
    scan_config: PSScanConfig,
    exp_config: ExperimentConfig,
    voltages: np.ndarray,
    phases_rad: np.ndarray,
    v_2pi: float,
    phi_init: float,
    r_squared: float,
    fit_info: Dict,
) -> None:
    """Save phase shifter scan results to YAML."""
    R = exp_config.chip.heater_resistance_ohm

    results = {
        "scan_parameters": scan_config.to_dict(),
        "ps_info": {
            "tap_num": scan_config.tap_num,
            "channel": exp_config.channel_mapping.get_channel(
                f"PS_{scan_config.tap_num}"
            ),
        },
        "fit_results": {
            "v_2pi_volts": float(v_2pi),
            "phi_init_rad": float(phi_init),
            "phi_init_deg": float(np.degrees(phi_init)),
            "p_2pi_watts": float(fit_info["p_2pi_watts"]),
            "p_2pi_mw": float(fit_info["p_2pi_watts"] * 1000),
            "slope_rad_per_v2": float(fit_info["slope_rad_per_v2"]),
            "r_squared": float(r_squared),
            "rmse_rad": float(fit_info["rmse_rad"]),
            "resistance_ohm": float(R),
        },
        "data_ranges": {
            "voltage_range": {
                "min": float(np.min(voltages)),
                "max": float(np.max(voltages)),
                "n_points": int(len(voltages)),
            },
            "phase_range_deg": {
                "min": float(np.degrees(np.nanmin(phases_rad))),
                "max": float(np.degrees(np.nanmax(phases_rad))),
            },
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_path = (
        Path(scan_config.output_dir) / f"scan_results_tap{scan_config.tap_num}.yaml"
    )
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Results saved: {output_path}")


# ==============================================================================
# HIGH-LEVEL CHARACTERISATION WORKFLOW
# ==============================================================================


def characterise_ps(
    tap_num: int,
    base_output_dir: str,
    exp_config: ExperimentConfig,
    v_min: float = 0.0,
    v_max: float = 20.0,
    n_points: int = 31,
    settling_time: float = 2.0,
    save_raw_data: bool = True,
) -> None:
    """
    Full characterisation workflow for a single phase shifter's V_2π and φ_init.

    Steps:
      1. Set up scan configuration
      2. Perform voltage sweep, measuring tap phase at each point
      3. Unwrap and linearly fit phase vs V²
      4. Plot and save results

    Parameters
    ----------
    tap_num : int
        Phase shifter tap number (1-indexed)
    base_output_dir : str
        Base output directory (a subdirectory per tap will be created)
    exp_config : ExperimentConfig
        Experiment configuration
    v_min, v_max : float
        Voltage sweep range (V)
    n_points : int
        Number of voltage points
    settling_time : float
        Thermal settling time per step (seconds)
    save_raw_data : bool
        Whether to save individual CSV files per voltage step
    """

    output_dir = str(Path(base_output_dir) / f"ps_tap{tap_num}")

    scan_config = PSScanConfig(
        tap_num=tap_num,
        v_min=v_min,
        v_max=v_max,
        n_points=n_points,
        settling_time_sec=settling_time,
        output_dir=output_dir,
        save_raw_data=save_raw_data,
    )

    R = exp_config.chip.heater_resistance_ohm
    print(f"Chip: {exp_config.chip.n_taps}-tap FIR, R = {R} Ω")

    # ---- MEASUREMENT ----
    voltages, phases_rad, dataframes = perform_ps_voltage_sweep(
        scan_config=scan_config,
        exp_config=exp_config,
    )

    # ---- FIT ----
    v_2pi, phi_init, r_squared, fit_info = fit_ps_v2pi_and_phi_init(
        voltages=voltages,
        phases_rad=phases_rad,
        resistance_ohm=R,
    )

    fit_successful = r_squared > 0.8 and not np.isnan(v_2pi)

    if not fit_successful:
        print(f"\n⚠ Warning: Linear fit quality poor (R² = {r_squared:.4f})")
        print("  Phase may be wrapping non-monotonically — check raw data")

    print_ps_fit_results(
        tap_num=tap_num,
        v_2pi=v_2pi,
        phi_init=phi_init,
        r_squared=r_squared,
        fit_info=fit_info,
        resistance_ohm=R,
    )

    # ---- PLOT ----
    plot_ps_characterisation(
        voltages=voltages,
        phases_rad=phases_rad,
        tap_num=tap_num,
        # v_2pi=v_2pi if fit_successful else None,
        # phi_init=phi_init if fit_successful else None,
        # fit_info=fit_info if fit_successful else None,
        v_2pi=None,
        phi_init=None,
        fit_info=None,
        resistance_ohm=R,
        output_dir=output_dir,
    )

    # ---- SAVE ----
    save_ps_results(
        scan_config=scan_config,
        exp_config=exp_config,
        voltages=voltages,
        phases_rad=phases_rad,
        v_2pi=v_2pi,
        phi_init=phi_init,
        r_squared=r_squared,
        fit_info=fit_info,
    )

    print(f"\n{'='*60}")
    print(f"  Phase Shifter Tap {tap_num} — Summary")
    print(f"{'='*60}")
    print(f"  V_2π:   {v_2pi:.3f} V")
    print(f"  P_2π:   {fit_info['p_2pi_watts']*1000:.2f} mW")
    print(f"  φ_init: {phi_init:.4f} rad  ({np.degrees(phi_init):.2f}°)")
    print(f"  R²:     {r_squared:.6f}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    """
    Batch characterise all signal-processing phase shifters on the chip.

    Phase shifters in the signal processing core are taps 9–16 (8 taps).
    Taps 1–8 are unused in the standard 16-tap configuration and are skipped.
    """

    # ============================================================
    # CONFIGURATION — edit these parameters
    # ============================================================

    CONFIG_PATH = "measurements/calibration_config.yaml"  # Reuse same config file

    # Voltage scan parameters
    V_MIN = 0.0  # V
    V_MAX = 20.0  # V  (PS heaters typically need less than MZIs)
    N_POINTS = 31  # 31 points → ~0.67 V steps (uniform in V²)

    # Timing
    SETTLING_TIME = 2.0  # seconds per voltage step

    # Output
    SAVE_RAW_DATA = True

    # ============================================================
    # LOAD CONFIGURATION
    # ============================================================

    exp_config = load_config(CONFIG_PATH)

    print(f"\n{'='*70}")
    print(f"CONFIGURATION LOADED")
    print(f"{'='*70}")
    print(f"Config file:       {CONFIG_PATH}")
    print(f"Chip:              {exp_config.chip.n_taps}-tap FIR filter")
    print(f"FSR:               {exp_config.chip.fsr_hz/1e9:.3f} GHz")
    print(f"P_2π (chip):       {exp_config.chip.p2pi_watts_ps*1000:.1f} mW")
    print(f"Heater resistance: {exp_config.chip.heater_resistance_ohm} Ω")
    print(f"{'='*70}\n")

    # ============================================================
    # DETERMINE PHASE SHIFTERS TO SCAN
    # ============================================================

    base_output_dir = get_next_run_dir(
        base_dir="measurements",
        prefix="v2pi_batch_ps_scan_results",
    )
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=f"{base_output_dir}/batch_ps_scan.log", level="INFO")

    # Signal processing core: taps 9–16 (1-indexed)
    # Taps 1–8 are unused in the standard 16-tap FIR layout
    all_ps_tap_nums = sorted(exp_config.channel_mapping.ps_channels.keys())
    ps_tap_nums = [t for t in all_ps_tap_nums if t >= 9]

    # ps_tap_nums = [9]  # TEMP: test with a single tap first

    print(f"\n{'='*70}")
    print(f"BATCH V_2π CHARACTERISATION — PHASE SHIFTERS")
    print(f"{'='*70}")
    print(f"Phase shifters to scan: {ps_tap_nums}")
    print(f"N taps: {len(ps_tap_nums)}")
    print(f"Base output directory: {base_output_dir}")
    print(f"{'='*70}\n")

    # ============================================================
    # RUN CHARACTERISATION FOR EACH PHASE SHIFTER
    # ============================================================

    for i, tap_num in enumerate(ps_tap_nums):
        print(f"\n{'#'*70}")
        print(f"# CHARACTERISING PHASE SHIFTER {i+1}/{len(ps_tap_nums)}: Tap {tap_num}")
        print(f"{'#'*70}\n")

        try:
            characterise_ps(
                tap_num=tap_num,
                base_output_dir=base_output_dir,
                exp_config=exp_config,
                v_min=V_MIN,
                v_max=V_MAX,
                n_points=N_POINTS,
                settling_time=SETTLING_TIME,
                save_raw_data=SAVE_RAW_DATA,
            )
        except Exception as e:
            print(f"⚠ FAILED to characterise PS tap {tap_num}: {e}")
            print("  Continuing with next tap...\n")
            continue

        if i < len(ps_tap_nums) - 1:
            print("Waiting 5 s before next scan...\n")
            time.sleep(5)

    # ============================================================
    # ZERO ALL HEATERS AT END OF BATCH (redundant safety)
    # ============================================================

    with VoltageController(
        com_port=exp_config.measurement.voltage_controller_port,
        baud_rate=exp_config.measurement.voltage_controller_baudrate,
        zero_on_exit=True,
    ) as v_ctrl:
        v_ctrl.set_voltages(
            channels=np.arange(1, 32 + 1).tolist(),
            voltages=[0.0] * 32,
            v_max=30.0,
        )

    print(f"\n{'='*70}")
    print(f"BATCH PS SCAN COMPLETE!")
    print(f"{'='*70}")
    print(f"Characterised {len(ps_tap_nums)} phase shifters")
    print(f"Results saved to: {base_output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
