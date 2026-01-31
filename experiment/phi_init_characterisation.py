"""
phi_init_characterisation.py

Two-step method for determining initial phase offsets (φ_init) of MZIs
BEFORE creating chip state. This is a pre-calibration characterisation step.

Key principle: We don't use ChipState during characterisation because we don't
yet know the accurate φ_init values that ChipState would need!
"""

import numpy as np
import time
from typing import Dict
from dataclasses import dataclass

from voltage_ctrl import VoltageController

# Import from your existing package
from photonic_fir.hardware.measurement import measure_spectrum
from photonic_fir.processing.tap_recovery import (
    recover_impulse_response_from_df,
    detect_taps,
)
from photonic_fir.core.power_splitting_ratio import (
    tap_coeffs_to_power_splitting_ratios,
)
from photonic_fir.core.data_structure import ExperimentConfig


@dataclass
class MZICharacterisation:
    """Results from MZI initial phase characterisation."""

    mzi_id: str
    phi_init_rad: float  # Initial phase offset
    psr_at_zero_db: float  # PSR when MZI at 0W
    psr_at_perturbation_db: float  # PSR when MZI at 0.05W
    delta_psr_db: float  # Change in PSR


def apply_raw_voltages(
    mzi_powers: Dict[str, float],
    config: ExperimentConfig,
    voltage_ctrl: VoltageController,
) -> None:
    """
    Apply raw voltages to MZI heaters without using chip_state.

    This is the low-level function used during characterisation when we
    don't yet have accurate φ_init values for chip_state.

    Args:
        mzi_powers: Dictionary mapping MZI IDs to powers in watts
        config: Experiment configuration (for channel mapping and resistance)
        voltage_ctrl: VoltageController instance
    """
    R = config.chip.heater_resistance_ohm

    print("  Applying voltages:")
    for mzi_id, power_watts in mzi_powers.items():
        voltage = np.sqrt(power_watts * R)
        channel = config.channel_mapping.get_channel(f"MZI_{mzi_id}")
        voltage_ctrl.set_voltages(channels=[channel], voltages=[voltage], v_max=30)
        print(f"    MZI {mzi_id} (ch {channel}): {voltage:.4f} V ({power_watts:.4f} W)")

    # Wait for thermal settling
    settling_time = config.chip.thermal_time_constant_s * 5  # 5 time constants
    print(f"  Waiting {settling_time:.3f}s for thermal settling...")
    time.sleep(settling_time)


def measure_and_extract_psrs(
    config: ExperimentConfig,
    mzi_tree: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Measure spectrum and extract PSRs for all MZIs.

    Args:
        config: Experiment configuration
        mzi_tree: MZI tree structure from config.signal_mzi_tree.tree

    Returns:
        Dictionary mapping MZI IDs to PSRs in dB
    """
    # 1. Measure spectrum
    df = measure_spectrum(
        center_wavelength_nm=config.measurement.center_wavelength_nm,
        wavelength_span_nm=config.measurement.wavelength_span_nm,
        num_averages=config.measurement.num_averages,
        edfa_port=config.measurement.edfa_port,
        edfa_baudrate=config.measurement.edfa_baudrate,
        edfa_output_power_dbm=config.measurement.edfa_output_power_dbm,
        ova_ip=config.measurement.ova_address,
        folder_dir=None,  # Don't save during characterisation
        file_name=None,
    )

    # 2. Recover impulse response
    time_ps, h_time = recover_impulse_response_from_df(
        df=df,
        fsr_hz=config.chip.fsr_hz,
        wavelength_col=config.measurement.wavelength_col,
        freq_col=config.measurement.frequency_col,
        insertion_loss_col=config.measurement.insertion_loss_col,
    )

    # 3. Detect taps
    tap_times, tap_coeffs = detect_taps(
        time_ps=time_ps,
        h_time=h_time,
        fsr_hz=config.chip.fsr_hz,
        delay_step_s=config.chip.delay_step_s,
        n_taps=config.chip.n_taps,
        prominence_factor_db=config.measurement.prominence_factor_db,
        min_distance_ps=config.measurement.min_distance_ps,
        height_threshold_db=config.measurement.height_threshold_db,
    )

    # 4. Extract signal processing taps
    signal_tap_coeffs = tap_coeffs[list(config.chip.signal_tap_indices)]

    # 5. Calculate PSRs
    psr_dict = tap_coeffs_to_power_splitting_ratios(signal_tap_coeffs, mzi_tree)

    return psr_dict


def characterise_mzi_phi_init(
    config: ExperimentConfig,
    perturbation_power_watts: float = 0.05,
) -> Dict[str, float]:
    """
    Determine initial phase offsets (φ_init) for all MZIs using two-step method.

    This runs BEFORE creating ChipState because we need φ_init values to
    create an accurate ChipState.

    The method:
    1. Baseline: All MZIs at 0W → measure PSRs
    2. Perturbations: Each MZI at 0.05W → measure its PSR
    3. From ΔPSR, determine branch and calculate φ_init

    Total measurements: 8 (1 baseline + 7 individual perturbations)

    Args:
        config: Experiment configuration
        perturbation_power_watts: Perturbation power to apply (default 0.05W)

    Returns:
        Dictionary mapping MZI IDs to φ_init in radians

    Example:
        >>> config = load_config("config.yaml")
        >>> phi_init = characterise_mzi_phi_init(config)
        >>> # Now create chip_state with these values
        >>> chip_state = ChipState.create_initial_state(
        ...     chip_params=config.chip,
        ...     p_fixed_watts=0.3,
        ...     mzi_phi_init=phi_init,  # Use measured values!
        ... )
    """
    mzi_tree = config.signal_mzi_tree.tree
    mzi_ids = list(mzi_tree.keys())
    p2pi_nominal_watts = config.chip.p2pi_watts

    print("\n" + "=" * 70)
    print("TWO-STEP φ_init CHARACTERISATION")
    print("=" * 70)
    print(f"MZIs to characterise: {mzi_ids}")
    print(f"Perturbation power: {perturbation_power_watts:.3f} W")
    print(f"P_2π: {p2pi_nominal_watts:.3f} W")

    # Calculate expected phase shift
    phi_perturbation = (perturbation_power_watts / p2pi_nominal_watts) * 2 * np.pi
    print(
        f"Expected phase shift: {phi_perturbation:.3f} rad "
        f"({np.degrees(phi_perturbation):.1f}°)"
    )

    # Create voltage controller
    voltage_ctrl = VoltageController(
        com_port=config.measurement.voltage_controller_port,
        baud_rate=config.measurement.voltage_controller_baudrate,
        zero_on_exit=True,
    )

    # -------------------------------------------------------------------------
    # STEP 1: Baseline measurement (all MZIs at 0W)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: Baseline measurement (all MZIs at 0W)")
    print("-" * 70)

    baseline_powers = {mzi_id: 0.0 for mzi_id in mzi_ids}
    apply_raw_voltages(baseline_powers, config, voltage_ctrl)

    print("Measuring spectrum and recovering tap coefficients...")
    psr_baseline = measure_and_extract_psrs(config, mzi_tree)

    print("\nBaseline PSRs:")
    for mzi_id in mzi_ids:
        psr = psr_baseline.get(mzi_id)
        if psr is not None:
            print(f"  {mzi_id}: {psr:7.2f} dB")
        else:
            print(f"  {mzi_id}: NOT FOUND")

    # -------------------------------------------------------------------------
    # STEP 2: Individual perturbations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: Individual perturbations (one MZI at a time)")
    print("-" * 70)

    results = {}

    for mzi_id in mzi_ids:
        print(f"\n--- Characterising MZI {mzi_id} ---")

        # Set only this MZI to perturbation power
        perturbed_powers = {mid: 0.0 for mid in mzi_ids}
        perturbed_powers[mzi_id] = perturbation_power_watts
        apply_raw_voltages(perturbed_powers, config, voltage_ctrl)

        # Measure PSRs
        psr_perturbed = measure_and_extract_psrs(config, mzi_tree)

        # Get PSRs for this MZI
        psr_0 = psr_baseline.get(mzi_id)
        psr_1 = psr_perturbed.get(mzi_id)

        if psr_0 is None or psr_1 is None:
            print(f"ERROR: Could not measure PSR for MZI {mzi_id}")
            continue

        delta_psr = psr_1 - psr_0

        print(f"  PSR at 0W:     {psr_0:7.2f} dB")
        print(f"  PSR at 0.05W:  {psr_1:7.2f} dB")
        print(f"  ΔPSR:          {delta_psr:+7.2f} dB")

        # -------------------------------------------------------------------------
        # Calculate φ_init from PSR and branch determination
        # -------------------------------------------------------------------------
        # PSR = 10·log₁₀[tan²(φ_total/2)] where φ_total = φ_MZI + φ_init

        # Convert PSR₀ from dB to linear
        tan_squared = 10 ** (psr_0 / 10)
        tan_value = np.sqrt(tan_squared)

        if delta_psr > 0:
            # PSR increased → positive slope branch
            # φ_init/2 is between 0 and π/2
            half_phi = np.arctan(tan_value)
            phi_init = 2 * half_phi
            print(f"  → ΔPSR > 0: positive slope branch")
        else:
            # PSR decreased → negative slope branch
            # φ_init/2 is between π/2 and π
            half_phi = np.pi - np.arctan(tan_value)
            phi_init = 2 * half_phi
            print(f"  → ΔPSR < 0: negative slope branch")

        # Wrap to [-π, π]
        phi_init = np.arctan2(np.sin(phi_init), np.cos(phi_init))

        print(f"  → φ_init = {phi_init:+7.3f} rad ({np.degrees(phi_init):+7.1f}°)")

        # Store results
        results[mzi_id] = MZICharacterisation(
            mzi_id=mzi_id,
            phi_init_rad=phi_init,
            psr_at_zero_db=psr_0,
            psr_at_perturbation_db=psr_1,
            delta_psr_db=delta_psr,
        )

    # Reset all MZIs to 0W
    baseline_powers = {mzi_id: 0.0 for mzi_id in mzi_ids}
    apply_raw_voltages(baseline_powers, config, voltage_ctrl)
    print("\nReset all MZIs to 0W")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CHARACTERISATION SUMMARY")
    print("=" * 70)

    phi_init_dict = {}
    for mzi_id, result in results.items():
        phi_init_dict[mzi_id] = result.phi_init_rad
        print(
            f"{mzi_id}: φ_init = {result.phi_init_rad:+7.3f} rad "
            f"({np.degrees(result.phi_init_rad):+7.1f}°), "
            f"ΔPSR = {result.delta_psr_db:+6.2f} dB"
        )

    print("\n" + "=" * 70)
    print("READY TO CREATE CHIP STATE")
    print("=" * 70)
    print("Use these φ_init values to create ChipState:")
    print(
        f"  chip_state = ChipState.create_initial_state(..., mzi_phi_init={phi_init_dict})"
    )

    return phi_init_dict


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Example showing the complete workflow:
    1. Characterise φ_init (without ChipState)
    2. Create ChipState with measured φ_init
    3. Run calibration
    """
    from photonic_fir import load_config
    from photonic_fir.core.data_structure import ChipState

    # Load configuration
    config = load_config(
        "measurements/experiment_config_shorter_range_reduce_num_avg.yaml"
    )

    print("=" * 70)
    print("COMPLETE CALIBRATION WORKFLOW")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: Characterise φ_init (no ChipState needed!)
    # -------------------------------------------------------------------------
    print("\n### STEP 1: Characterise φ_init ###")
    phi_init_dict = characterise_mzi_phi_init(
        config=config,
        perturbation_power_watts=0.05,
    )

    # -------------------------------------------------------------------------
    # STEP 2: Create ChipState with accurate φ_init
    # -------------------------------------------------------------------------
    print("\n### STEP 2: Create ChipState with measured φ_init ###")
    chip_state = ChipState.create_initial_state(
        chip_params=config.chip,
        p_fixed_watts=0.3,
        mzi_phi_init=phi_init_dict,  # Use the measured values!
    )

    print("ChipState created with accurate φ_init values:")
    for mzi_id, mzi in chip_state.mzis.items():
        print(f"  MZI {mzi_id}: φ_init = {mzi.phi_init_rad:+7.3f} rad")

    # -------------------------------------------------------------------------
    # STEP 3: Run main calibration loop
    # -------------------------------------------------------------------------
    print("\n### STEP 3: Ready for calibration ###")
    print("Now chip_state has accurate φ_init and can be used for calibration!")

    # from photonic_fir.calibration import run_calibration_loop
    # calibration_results = run_calibration_loop(chip_state, config)
