"""
phi_init_characterisation.py (CORRECTED)

Two-step method for determining initial phase offsets (φ_init) of MZIs.
This characterises the chip_state in place BEFORE running calibration.

Key improvements:
- Fixed voltage application to use set_voltages() API correctly
- Simplified logic: perturb one MZI at a time, no power dicts needed
- Correct mathematics: uses arctan (not arccos) per paper's tan² formula
- No P_2π assumption: determines φ_init from PSR_0 and ΔPSR sign only

Key principle: We create ChipState with default φ_init = 0.0, then measure
and populate the accurate values directly into the MZI objects.
"""

import logging

logger = logging.getLogger(__name__)

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

    logger.info("  Applying voltages:")
    for mzi_id in mzi_ids:
        # Apply perturbation to target MZI, 0V to all others
        power_watts = perturbation_power_watts if mzi_id == perturb_mzi_id else 0.0
        voltage = np.sqrt(power_watts * R)
        channel = config.channel_mapping.get_channel(f"MZI_{mzi_id}")

        channels.append(channel)
        voltages.append(voltage)

        if power_watts > 0:
            logger.info(
                f"    MZI {mzi_id} (ch {channel}): {voltage:.4f} V ({power_watts:.4f} W)"
            )

    # Apply all voltages at once using correct API
    voltage_ctrl.set_voltages(channels=channels, voltages=voltages, v_max=v_max)

    # Wait for thermal settling
    settling_time = config.chip.thermal_time_constant_s * 5  # 5 time constants
    logger.info(f"  Waiting {settling_time:.3f}s for thermal settling...")
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

    # 5. Calculate PSRs
    psr_dict = tap_coeffs_to_power_splitting_ratios(tap_coeffs, mzi_tree)

    return psr_dict


def characterise_mzi_phi_init(
    chip_state: ChipState,
    config: ExperimentConfig,
    voltage_ctrl: VoltageController,
    perturbation_power_watts: float = 0.05,
    mzi_ids: list = None,
) -> None:
    """
    Measure and populate φ_init for all MZIs directly in chip_state.

    This modifies chip_state.mzis[mzi_id].phi_init_rad in place.

    The two-step method (from Xu et al. 2022 paper):
    1. Baseline: All MZIs at 0W → measure PSRs
    2. Perturbations: Each MZI at perturbation power → measure its PSR
    3. From ΔPSR sign, determine branch and calculate φ_init

    Mathematics:
    -----------
    From the paper: PSR = 10*log₁₀(tan²(φ/2))

    At baseline: φ = φ_init
    Therefore: tan²(φ_init/2) = 10^(PSR_0/10)
               |tan(φ_init/2)| = 10^(PSR_0/20)

    The sign is determined by ΔPSR (slope direction):
    - ΔPSR > 0: sin(φ_init) > 0, so φ_init ∈ (0, π)  → use positive arctan
    - ΔPSR < 0: sin(φ_init) < 0, so φ_init ∈ (-π, 0) → use negative arctan

    Key advantage: No assumption about P_2π needed! We only use the perturbation
    to determine the slope direction, not the magnitude.

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

    logger.info("\n" + "=" * 70)
    logger.info("φ_init CHARACTERISATION - Two-Step Method")
    logger.info("=" * 70)

    # Get MZI IDs and tree structure
    if mzi_ids is None:
        mzi_ids = config.chip.get_all_mzi_ids()
    mzi_tree = config.full_mzi_tree.tree

    logger.info(f"MZIs to characterise: {mzi_ids}")
    logger.info(f"Perturbation power: {perturbation_power_watts:.4f} W")
    logger.info(f"Total measurements: {len(mzi_ids) + 1}")

    # Store characterisation results for logging
    char_results = {}

    # -------------------------------------------------------------------------
    # STEP 1: Baseline measurement (all MZIs at 0W)
    # -------------------------------------------------------------------------
    logger.info("\n### STEP 1: Baseline (all MZIs at 0W) ###")
    apply_single_mzi_perturbation(
        mzi_ids=mzi_ids,
        perturb_mzi_id=None,  # None = all at 0W
        perturbation_power_watts=0.0,
        config=config,
        voltage_ctrl=voltage_ctrl,
    )

    psr_baseline = measure_and_extract_psrs(config, mzi_tree)
    logger.info("\nBaseline PSRs (dB):")
    for mzi_id, psr in psr_baseline.items():
        logger.info(f"  {mzi_id}: {psr:+7.2f} dB")

    # -------------------------------------------------------------------------
    # STEP 2: Perturbation measurements (one MZI at a time)
    # -------------------------------------------------------------------------
    logger.info("\n### STEP 2: Individual perturbations ###")

    for mzi_id in mzi_ids:
        logger.info(f"\n--- Characterising {mzi_id} ---")

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

        logger.info(f"  PSR at 0W:     {psr_0:+7.2f} dB")
        logger.info(f"  PSR at {perturbation_power_watts:.3f}W: {psr_1:+7.2f} dB")
        logger.info(f"  ΔPSR:          {delta_psr:+7.2f} dB")

        # =====================================================================
        # Calculate φ_init from PSR_0 and ΔPSR sign
        # =====================================================================
        # From paper: PSR = 10*log₁₀(tan²(φ/2))
        # At baseline: φ = φ_init
        # Therefore: tan²(φ_init/2) = 10^(PSR_0/10)
        #            |tan(φ_init/2)| = 10^(PSR_0/20)
        #
        # The sign is determined by the slope direction:
        # dPSR/dφ = 20/(ln(10)*sin(φ))
        #
        # For small positive perturbation Δφ:
        # - If ΔPSR > 0: sin(φ_init) > 0, so 0 < φ_init < π
        # - If ΔPSR < 0: sin(φ_init) < 0, so -π < φ_init < 0

        if delta_psr > 0:
            # Positive slope: 0 < φ_init < π
            # Use positive arctan
            phi_init = 2 * np.arctan(10 ** (psr_0 / 20))
            logger.info(f"  → ΔPSR > 0: positive slope branch (0 < φ < π)")
        else:
            # Negative slope: -π < φ_init < 0
            # Use negative arctan
            phi_init = 2 * np.arctan(-(10 ** (psr_0 / 20)))
            logger.info(f"  → ΔPSR < 0: negative slope branch (-π < φ < 0)")

        # Result is already in [-π, π], no wrapping needed
        logger.info(
            f"  → φ_init = {phi_init:+7.3f} rad ({np.degrees(phi_init):+7.1f}°)"
        )

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
    logger.info("\nReset all MZIs to 0W")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("CHARACTERISATION COMPLETE")
    logger.info("=" * 70)
    logger.info("\nφ_init values populated in chip_state:")
    for mzi_id, result in char_results.items():
        logger.info(
            f"  {mzi_id}: φ_init = {result.phi_init_rad:+7.3f} rad "
            f"({np.degrees(result.phi_init_rad):+7.1f}°), "
            f"ΔPSR = {result.delta_psr_db:+6.2f} dB"
        )

    logger.info("\n" + "=" * 70)
    logger.info("ChipState ready for calibration!")
    logger.info("=" * 70)


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
    config = load_config(
        "measurements/experiment_config_shorter_range_reduce_num_avg.yaml"
    )

    logger.info("=" * 70)
    logger.info("COMPLETE CALIBRATION WORKFLOW")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: Create ChipState with defaults
    # -------------------------------------------------------------------------
    logger.info("\n### STEP 1: Create ChipState ###")
    chip_state = ChipState(
        chip_params=config.chip,
        p_fixed_watts=0.3,
    )
    logger.info("ChipState created with default φ_init = 0.0 for all MZIs")

    # -------------------------------------------------------------------------
    # STEP 2: Characterise φ_init (mutates chip_state in place)
    # -------------------------------------------------------------------------
    logger.info("\n### STEP 2: Characterise φ_init ###")

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

        logger.info("\nVerifying φ_init values in chip_state:")
        for mzi_id, mzi in chip_state.mzis.items():
            logger.info(f"  MZI {mzi_id}: φ_init = {mzi.phi_init_rad:+7.3f} rad")

    finally:
        voltage_ctrl._close_serial()

    # -------------------------------------------------------------------------
    # STEP 3: Run main calibration loop
    # -------------------------------------------------------------------------
    logger.info("\n### STEP 3: Ready for calibration ###")
    logger.info("ChipState now has accurate φ_init values!")

    # from photonic_fir.calibration import run_calibration_loop
    # calibration_results = run_calibration_loop(chip_state, config)
