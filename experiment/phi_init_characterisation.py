"""
phi_init_characterisation.py

Two-iteration method for determining initial phase offsets (φ_init) of MZIs
before the main self-calibration process, as described in Xu et al. (2022).

Integrates with existing photonic_fir package functions.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# Import from your existing package
from photonic_fir.hardware.measurement import measure_spectrum
from photonic_fir.processing.tap_recovery import (
    recover_impulse_response_from_df,
    detect_taps,
)
from photonic_fir.core.power_splitting_ratio import (
    tap_coeffs_to_power_splitting_ratios,
)
from photonic_fir.core.data_structure import (
    ChipState,
    ExperimentConfig,
)
from photonic_fir.hardware.voltage_adjustment import apply_voltages_to_hardware


@dataclass
class MZICharacterisation:
    """Results from MZI initial phase characterisation."""

    mzi_id: str
    phi_init_rad: float  # Initial phase offset
    psr_at_zero_db: float  # PSR when MZI at 0W
    psr_at_perturbation_db: float  # PSR when MZI at 0.05W
    delta_psr_db: float  # Change in PSR


def measure_and_extract_psrs(
    chip_state: ChipState,
    config: ExperimentConfig,
    mzi_tree: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Measure spectrum and extract PSRs for all MZIs.

    This wraps the existing measurement and processing pipeline to return
    PSRs for all MZIs from a single spectrum measurement.

    Args:
        chip_state: Current chip state (contains applied powers)
        config: Experiment configuration
        mzi_tree: MZI tree structure from build_mzi_tree_structure()

    Returns:
        Dictionary mapping MZI IDs to PSRs in dB
    """
    # 1. Measure spectrum using existing function
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

    # 2. Recover impulse response using existing function
    time_ps, h_time = recover_impulse_response_from_df(
        df=df,
        fsr_hz=config.chip.fsr_hz,
        wavelength_col=config.measurement.wavelength_col,
        freq_col=config.measurement.frequency_col,
        insertion_loss_col=config.measurement.insertion_loss_col,
    )

    # 3. Detect taps using existing function
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

    # 4. Extract signal processing taps (taps 9-16, indices 8-15)
    signal_tap_coeffs = tap_coeffs[list(config.chip.signal_tap_indices)]

    # 5. Calculate PSRs using existing function
    psr_dict = tap_coeffs_to_power_splitting_ratios(signal_tap_coeffs, mzi_tree)

    return psr_dict


def set_mzi_powers(
    mzi_powers: Dict[str, float],
    chip_state: ChipState,
    config: ExperimentConfig,
) -> None:
    """
    Set powers for all MZIs and apply to hardware.

    Args:
        mzi_powers: Dictionary mapping MZI IDs to powers in watts
        chip_state: Chip state to update
        config: Experiment configuration
    """
    # Update chip state with new powers
    chip_state.update_powers(
        new_mzi_powers=mzi_powers,
        new_ps_powers={},  # Don't change phase shifters during characterisation
    )

    # Apply to hardware using existing function
    apply_voltages_to_hardware(chip_state, config)

    # Wait for thermal settling
    import time

    settling_time = config.chip.thermal_time_constant_s * 5  # 5 time constants
    print(f"  Waiting {settling_time:.3f}s for thermal settling...")
    time.sleep(settling_time)


def two_step_phi_init_characterisation(
    chip_state: ChipState,
    config: ExperimentConfig,
    mzi_tree: Dict[str, Dict],
    perturbation_power_watts: float = 0.05,
) -> Dict[str, MZICharacterisation]:
    """
    Determine initial phase offsets (φ_init) for all MZIs using two-iteration method.

    Integrates with existing photonic_fir functions for measurement and processing.

    The method:
    1. Baseline measurement: All MZIs at 0W → measure PSRs for all MZIs
    2. Individual perturbations: For each MZI, set to 0.05W → measure its PSR
    3. From ΔPSR, determine branch and calculate φ_init

    Total measurements: 8 (1 baseline + 7 individual perturbations)

    Args:
        chip_state: Initial chip state
        config: Experiment configuration
        mzi_tree: MZI tree structure from build_mzi_tree_structure()
        perturbation_power_watts: Perturbation power to apply (default 0.05W)

    Returns:
        Dictionary mapping MZI IDs to MZICharacterisation objects
    """
    mzi_ids = list(mzi_tree.keys())
    p2pi_nominal_watts = config.chip.p2pi_watts

    print("\n" + "=" * 70)
    print("TWO-ITERATION φ_init CHARACTERISATION")
    print("=" * 70)
    print(f"MZIs to characterise: {mzi_ids}")
    print(f"Perturbation power: {perturbation_power_watts:.3f} W")
    print(f"P_2π: {p2pi_nominal_watts:.3f} W")

    # Calculate expected phase shift from perturbation
    phi_perturbation = (perturbation_power_watts / p2pi_nominal_watts) * 2 * np.pi
    print(
        f"Expected phase shift: {phi_perturbation:.3f} rad ({np.degrees(phi_perturbation):.1f}°)"
    )

    # Step 1: Baseline measurement (all MZIs at 0W)
    print("\n" + "-" * 70)
    print("STEP 1: Baseline measurement (all MZIs at 0W)")
    print("-" * 70)

    # Set all MZIs to 0W
    baseline_powers = {mzi_id: 0.0 for mzi_id in mzi_ids}
    set_mzi_powers(baseline_powers, chip_state, config)
    print("Set all MZIs to 0W")

    # Measure PSRs
    print("Measuring spectrum and recovering tap coefficients...")
    psr_baseline = measure_and_extract_psrs(chip_state, config, mzi_tree)

    print("\nBaseline PSRs:")
    for mzi_id in mzi_ids:
        psr = psr_baseline.get(mzi_id)
        if psr is not None:
            print(f"  {mzi_id}: {psr:7.2f} dB")
        else:
            print(f"  {mzi_id}: NOT FOUND")

    # Step 2: Individual perturbations
    print("\n" + "-" * 70)
    print("STEP 2: Individual perturbations (one MZI at a time)")
    print("-" * 70)

    results = {}

    for mzi_id in mzi_ids:
        print(f"\n--- Characterising MZI {mzi_id} ---")

        # Set only this MZI to perturbation power
        perturbed_powers = {mid: 0.0 for mid in mzi_ids}
        perturbed_powers[mzi_id] = perturbation_power_watts
        set_mzi_powers(perturbed_powers, chip_state, config)
        print(f"Applied {perturbation_power_watts:.3f} W to MZI {mzi_id}")

        # Measure PSRs
        psr_perturbed = measure_and_extract_psrs(chip_state, config, mzi_tree)

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

        # Determine branch and calculate φ_init
        # PSR = 10·log₁₀[tan²(φ_total/2)]
        # where φ_total = φ_MZI + φ_init

        # Convert PSR₀ from dB to linear
        tan_squared = 10 ** (psr_0 / 10)
        tan_value = np.sqrt(tan_squared)

        if delta_psr > 0:
            # PSR increased → positive slope branch
            # φ_init/2 is between minimum and maximum
            half_phi = np.arctan(tan_value)
            phi_init = 2 * half_phi
            print(f"  → ΔPSR > 0: positive slope branch")

        else:
            # PSR decreased → negative slope branch
            # φ_init/2 is between maximum and minimum
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
    set_mzi_powers(baseline_powers, chip_state, config)
    print("\nReset all MZIs to 0W")

    # Summary
    print("\n" + "=" * 70)
    print("CHARACTERISATION SUMMARY")
    print("=" * 70)
    for mzi_id, result in results.items():
        print(
            f"{mzi_id}: φ_init = {result.phi_init_rad:+7.3f} rad "
            f"({np.degrees(result.phi_init_rad):+7.1f}°), "
            f"ΔPSR = {result.delta_psr_db:+6.2f} dB"
        )

    return results


def apply_phi_init_to_chip_state(
    results: Dict[str, MZICharacterisation],
    chip_state: ChipState,
) -> None:
    """
    Apply measured φ_init values to chip state.

    Args:
        results: Results from two_step_phi_init_characterisation()
        chip_state: Chip state to update
    """
    print("\n" + "-" * 70)
    print("Updating chip state with measured φ_init values:")
    print("-" * 70)

    for mzi_id, result in results.items():
        if mzi_id in chip_state.mzis:
            chip_state.mzis[mzi_id].phi_init_rad = result.phi_init_rad
            print(f"  MZI {mzi_id}: φ_init = {result.phi_init_rad:+7.3f} rad")
        else:
            print(f"  WARNING: MZI {mzi_id} not found in chip state")

    print("\nChip state updated successfully!")


def extract_phi_init_dict(
    characterisation_results: Dict[str, MZICharacterisation],
) -> Dict[str, float]:
    """
    Extract just the phi_init values as a dictionary.

    Args:
        characterisation_results: Output from two_step_phi_init_characterisation()

    Returns:
        Dictionary mapping MZI IDs to phi_init in radians
    """
    return {
        mzi_id: result.phi_init_rad
        for mzi_id, result in characterisation_results.items()
    }


# Example usage
if __name__ == "__main__":
    """
    Example showing how to integrate φ_init characterisation into your workflow.
    """
    from photonic_fir import load_config

    # Load configuration
    config = load_config("example_config.yaml")

    # Create initial chip state
    chip_state = ChipState.create_initial_state(
        chip_params=config.chip,
        p_fixed_watts=0.3,
    )

    # Build MZI tree structure
    mzi_tree = config.signal_mzi_tree.tree

    print("=" * 70)
    print("INTEGRATED φ_init CHARACTERISATION WORKFLOW")
    print("=" * 70)

    # Run φ_init characterisation
    print("\nStep 1: Characterise φ_init for all MZIs")
    results = two_step_phi_init_characterisation(
        chip_state=chip_state,
        config=config,
        mzi_tree=mzi_tree,
        perturbation_power_watts=0.05,
    )

    # Apply to chip state
    print("\nStep 2: Apply φ_init to chip state")
    apply_phi_init_to_chip_state(results, chip_state)

    # Extract for saving/analysis
    phi_init_dict = extract_phi_init_dict(results)

    print("\n" + "=" * 70)
    print("READY FOR CALIBRATION")
    print("=" * 70)
    print("Chip state now contains accurate φ_init values.")
    print("Proceed with main calibration loop...")

    # Now you can run the main calibration with accurate φ_init
    # from photonic_fir.calibration import run_experiment
    # results = run_experiment(config)
