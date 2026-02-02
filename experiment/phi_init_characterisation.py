"""
phi_init_characterisation.py (REFACTORED)

Two-step method for determining initial phase offsets (φ_init) of MZIs.
This characterises the chip_state in place BEFORE running calibration.

Key improvements:
- Fixed voltage application to use set_voltages() API correctly
- Simplified logic: no need to store power dicts since we perturb one MZI at a time
- Batch voltage application for efficiency

Key principle: We create ChipState with default φ_init = 0.0, then measure
and populate the accurate values directly into the MZI objects.
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
from photonic_fir.core.data_structure import ExperimentConfig, ChipState


@dataclass
class MZICharacterisationResult:
    """Results from a single MZI characterisation measurement."""

    mzi_id: str
    phi_init_rad: float  # Measured initial phase offset
    psr_at_zero_db: float  # PSR when MZI at 0W
    psr_at_perturbation_db: float  # PSR when MZI at perturbation power
    delta_psr_db: float  # Change in PSR


def apply_single_mzi_perturbation(
    mzi_ids: list,
    perturb_mzi_id: str | None,
    perturbation_power_watts: float,
    config: ExperimentConfig,
    voltage_ctrl: VoltageController,
    v_max: float = 30.0,
) -> None:
    """
    Apply perturbation power to a single MZI, set all others to 0W.

    Args:
        mzi_ids: List of all MZI IDs on the chip
        perturb_mzi_id: MZI to apply perturbation to (None = all at 0W for baseline)
        perturbation_power_watts: Power to apply to perturbed MZI
        config: Experiment configuration
        voltage_ctrl: VoltageController instance
        v_max: Maximum allowed voltage (V)
    """
    R = config.chip.heater_resistance_ohm

    # Prepare channel and voltage lists for batch application
    channels = []
    voltages = []

    print("  Applying voltages:")
    for mzi_id in mzi_ids:
        # Apply perturbation to target MZI, 0V to all others
        power_watts = perturbation_power_watts if mzi_id == perturb_mzi_id else 0.0
        voltage = np.sqrt(power_watts * R)
        channel = config.channel_mapping.get_channel(f"MZI_{mzi_id}")

        channels.append(channel)
        voltages.append(voltage)

        if power_watts > 0:
            print(
                f"    MZI {mzi_id} (ch {channel}): {voltage:.4f} V ({power_watts:.4f} W)"
            )

    # Apply all voltages at once using correct API
    voltage_ctrl.set_voltages(channels=channels, voltages=voltages, v_max=v_max)

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
    chip_state: ChipState,
    config: ExperimentConfig,
    voltage_ctrl: VoltageController,
    perturbation_power_watts: float = 0.05,
) -> None:
    """
    Measure and populate φ_init for all MZIs directly in chip_state.

    This modifies chip_state.mzis[mzi_id].phi_init_rad in place.

    The two-step method:
    1. Baseline: All MZIs at 0W → measure PSRs
    2. Perturbations: Each MZI at perturbation power → measure its PSR
    3. From ΔPSR, determine branch and calculate φ_init

    Total measurements: N+1 (1 baseline + N individual perturbations)

    Args:
        chip_state: ChipState to populate with φ_init values
        config: Experiment configuration
        voltage_ctrl: VoltageController instance
        perturbation_power_watts: Perturbation power to apply (default 0.05W)

    Example:
        >>> config = load_config("config.yaml")
        >>> chip_state = ChipState(chip_params=config.chip, p_fixed_watts=0.3)
        >>> voltage_ctrl = VoltageController(...)
        >>> characterise_mzi_phi_init(chip_state, config, voltage_ctrl)
        >>> # Now chip_state.mzis have accurate phi_init_rad values!
    """

    print("\n" + "=" * 70)
    print("φ_init CHARACTERISATION - Two-Step Method")
    print("=" * 70)

    # Get MZI IDs and tree structure
    mzi_ids = config.chip.get_signal_mzi_ids()
    mzi_tree = config.signal_mzi_tree.tree

    print(f"MZIs to characterise: {mzi_ids}")
    print(f"Perturbation power: {perturbation_power_watts:.4f} W")
    print(f"Total measurements: {len(mzi_ids) + 1}")

    # Store characterisation results for logging
    char_results = {}

    # -------------------------------------------------------------------------
    # STEP 1: Baseline measurement (all MZIs at 0W)
    # -------------------------------------------------------------------------
    print("\n### STEP 1: Baseline (all MZIs at 0W) ###")
    apply_single_mzi_perturbation(
        mzi_ids=mzi_ids,
        perturb_mzi_id=None,  # None = all at 0W
        perturbation_power_watts=0.0,
        config=config,
        voltage_ctrl=voltage_ctrl,
    )

    psr_baseline = measure_and_extract_psrs(config, mzi_tree)
    print("\nBaseline PSRs (dB):")
    for mzi_id, psr in psr_baseline.items():
        print(f"  {mzi_id}: {psr:+7.2f} dB")

    # -------------------------------------------------------------------------
    # STEP 2: Perturbation measurements (one MZI at a time)
    # -------------------------------------------------------------------------
    print("\n### STEP 2: Individual perturbations ###")

    for mzi_id in mzi_ids:
        print(f"\n--- Characterising {mzi_id} ---")

        # Apply perturbation to this MZI only
        apply_single_mzi_perturbation(
            mzi_ids=mzi_ids,
            perturb_mzi_id=mzi_id,
            perturbation_power_watts=perturbation_power_watts,
            config=config,
            voltage_ctrl=voltage_ctrl,
        )

        # Measure PSRs
        psr_perturbed = measure_and_extract_psrs(config, mzi_tree)

        # Extract this MZI's PSR values
        psr_0 = psr_baseline[mzi_id]
        psr_1 = psr_perturbed[mzi_id]
        delta_psr = psr_1 - psr_0

        print(f"  PSR at 0W:     {psr_0:+7.2f} dB")
        print(f"  PSR at {perturbation_power_watts:.3f}W: {psr_1:+7.2f} dB")
        print(f"  ΔPSR:          {delta_psr:+7.2f} dB")

        # Calculate φ_init from ΔPSR
        # PSR = 20*log10(|cos(φ/2)|) where φ = φ_applied + φ_init
        # At 0W: φ = φ_init
        # At perturbation: φ = (P_pert / P_2π) * 2π + φ_init

        # φ_applied at perturbation
        phi_applied = (perturbation_power_watts / config.chip.p2pi_watts) * 2 * np.pi

        # Determine branch from sign of ΔPSR
        if delta_psr > 0:
            # Positive slope: moving away from π
            # φ_init is on [0, π/2] or [-π, -π/2]
            # Use positive slope formula
            phi_init = np.arccos(10 ** (psr_0 / 20)) - phi_applied / 2
            print(f"  → ΔPSR > 0: positive slope branch")
        else:
            # Negative slope: moving toward π
            # φ_init is on [π/2, π] or [-π/2, 0]
            # Use negative slope formula
            phi_init = -np.arccos(10 ** (psr_0 / 20)) - phi_applied / 2
            print(f"  → ΔPSR < 0: negative slope branch")

        # Wrap to [-π, π]
        phi_init = np.arctan2(np.sin(phi_init), np.cos(phi_init))

        print(f"  → φ_init = {phi_init:+7.3f} rad ({np.degrees(phi_init):+7.1f}°)")

        # *** DIRECTLY SET φ_init IN CHIP STATE ***
        chip_state.mzis[mzi_id].phi_init_rad = phi_init

        # Store for summary
        char_results[mzi_id] = MZICharacterisationResult(
            mzi_id=mzi_id,
            phi_init_rad=phi_init,
            psr_at_zero_db=psr_0,
            psr_at_perturbation_db=psr_1,
            delta_psr_db=delta_psr,
        )

    # -------------------------------------------------------------------------
    # STEP 3: Reset to baseline
    # -------------------------------------------------------------------------
    apply_single_mzi_perturbation(
        mzi_ids=mzi_ids,
        perturb_mzi_id=None,  # All at 0W
        perturbation_power_watts=0.0,
        config=config,
        voltage_ctrl=voltage_ctrl,
    )
    print("\nReset all MZIs to 0W")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CHARACTERISATION COMPLETE")
    print("=" * 70)
    print("\nφ_init values populated in chip_state:")
    for mzi_id, result in char_results.items():
        print(
            f"  {mzi_id}: φ_init = {result.phi_init_rad:+7.3f} rad "
            f"({np.degrees(result.phi_init_rad):+7.1f}°), "
            f"ΔPSR = {result.delta_psr_db:+6.2f} dB"
        )

    print("\n" + "=" * 70)
    print("ChipState ready for calibration!")
    print("=" * 70)


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Example showing the complete workflow:
    1. Create ChipState (with default φ_init = 0.0)
    2. Characterise φ_init (populates chip_state in place)
    3. Run calibration
    """
    from photonic_fir import load_config
    from photonic_fir.core.data_structure import ChipState
    from voltage_ctrl import VoltageController

    # Load configuration
    config = load_config("example_config.yaml")

    print("=" * 70)
    print("COMPLETE CALIBRATION WORKFLOW")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: Create ChipState with defaults
    # -------------------------------------------------------------------------
    print("\n### STEP 1: Create ChipState ###")
    chip_state = ChipState(
        chip_params=config.chip,
        p_fixed_watts=0.3,
    )
    print("ChipState created with default φ_init = 0.0 for all MZIs")

    # -------------------------------------------------------------------------
    # STEP 2: Characterise φ_init (mutates chip_state in place)
    # -------------------------------------------------------------------------
    print("\n### STEP 2: Characterise φ_init ###")

    # Initialize hardware
    voltage_ctrl = VoltageController(
        com_port=config.measurement.voltage_controller_port,
        baud_rate=config.measurement.voltage_controller_baudrate,
        zero_on_exit=True,
    )

    try:
        # Measure and populate φ_init
        characterise_mzi_phi_init(
            chip_state=chip_state,
            config=config,
            voltage_ctrl=voltage_ctrl,
            perturbation_power_watts=0.05,
        )

        print("\nVerifying φ_init values in chip_state:")
        for mzi_id, mzi in chip_state.mzis.items():
            print(f"  MZI {mzi_id}: φ_init = {mzi.phi_init_rad:+7.3f} rad")

    finally:
        voltage_ctrl.close()

    # -------------------------------------------------------------------------
    # STEP 3: Run main calibration loop
    # -------------------------------------------------------------------------
    print("\n### STEP 3: Ready for calibration ###")
    print("ChipState now has accurate φ_init values!")

    # from photonic_fir.calibration import run_calibration_loop
    # calibration_results = run_calibration_loop(chip_state, config)
