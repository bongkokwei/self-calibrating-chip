"""
main_expt.py

Main script for running FIR chip calibration experiments.
Simple implementation using the data structures.
"""

import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple

from data_structure import (
    ExperimentConfig,
    ChipState,
    IterationData,
    CalibrationResults,
    config_from_dict,
    config_to_dict,
)


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_from_dict(config_dict)


def save_config(config: ExperimentConfig, output_dir: str):
    """Save configuration to output directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config_dict = config_to_dict(config)
    output_path = Path(output_dir) / "experiment_config.yaml"

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to {output_path}")


def compute_target_tap_coefficients(
    config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute desired tap coefficients based on target filter specification.

    Returns:
        amplitudes: Array of length n_signal_taps
        phases_rad: Array of length n_signal_taps
    """
    n_taps = config.chip.n_signal_taps

    if config.target.filter_type == "sinc":
        # Equal amplitude sinc filter
        amplitudes = np.ones(n_taps)

        # Linear phase progression
        phases_rad = np.arange(n_taps) * config.target.phase_step_rad

    elif config.target.filter_type == "hilbert":
        # Hilbert transformer (90° phase shift)
        amplitudes = np.ones(n_taps)
        phases_rad = np.array([0, np.pi / 2, 0, np.pi / 2, 0, np.pi / 2, 0, np.pi / 2])

    elif config.target.filter_type == "custom":
        # Use provided custom coefficients
        amplitudes = np.array(config.target.custom_amplitudes)
        phases_rad = np.array(config.target.custom_phases_rad)

    else:
        raise ValueError(f"Unknown filter type: {config.target.filter_type}")

    return amplitudes, phases_rad


def measure_insertion_loss_spectrum(
    config: ExperimentConfig, chip_state: ChipState
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measure insertion loss spectrum of the chip.

    In real implementation, this would:
    - Configure voltage controller to set MZI and PS powers from chip_state
    - Trigger Luna OVA to measure spectrum
    - Return wavelengths and insertion loss

    For now, returns dummy data.
    """
    # Generate wavelength array
    center = config.measurement.center_wavelength_nm
    span = config.measurement.wavelength_span_nm
    n_points = config.measurement.n_points

    wavelengths = np.linspace(center - span / 2, center + span / 2, n_points)

    # TODO: Replace with actual measurement
    # For now, generate dummy spectrum
    insertion_loss_db = -10 * np.ones(n_points) + np.random.randn(n_points) * 0.1

    print(f"  Measured spectrum: {len(wavelengths)} points")

    return wavelengths, insertion_loss_db


def recover_phase_kramers_kronig(
    wavelengths_nm: np.ndarray, insertion_loss_db: np.ndarray, config: ExperimentConfig
) -> np.ndarray:
    """
    Recover phase response using Kramers-Kronig relationship.

    Returns:
        phase_rad: Phase response at each wavelength point
    """
    # TODO: Implement Kramers-Kronig phase recovery
    # 1. Convert insertion loss to amplitude
    # 2. Apply Hilbert transform
    # 3. Return phase

    # Dummy implementation
    phase_rad = np.zeros_like(wavelengths_nm)

    print(f"  Phase recovered via Kramers-Kronig")

    return phase_rad


def recover_tap_coefficients(
    wavelengths_nm: np.ndarray,
    insertion_loss_db: np.ndarray,
    phase_rad: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover tap coefficients from frequency response using inverse FFT.

    Returns:
        tap_amplitudes: Array of length n_taps (all 16 taps)
        tap_phases_rad: Array of length n_taps (all 16 taps)
    """
    # TODO: Implement inverse FFT to get impulse response
    # 1. Construct complex frequency response H(f)
    # 2. Apply inverse FFT
    # 3. Extract tap coefficients

    # Dummy implementation
    n_taps = config.chip.n_taps
    tap_amplitudes = np.random.rand(n_taps) * 0.5
    tap_phases_rad = np.random.randn(n_taps) * 0.1

    print(f"  Tap coefficients recovered: {n_taps} taps")

    return tap_amplitudes, tap_phases_rad


def calculate_errors(
    measured_amps: np.ndarray,
    measured_phases: np.ndarray,
    target_amps: np.ndarray,
    target_phases: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate errors between measured and target tap coefficients.

    Only considers signal processing taps (taps 9-16).

    Returns:
        amplitude_errors_db: Amplitude errors in dB
        phase_errors_rad: Phase errors in radians
    """
    # Extract signal processing taps (indices 8-15 for taps 9-16)
    signal_start = config.chip.n_taps - config.chip.n_signal_taps

    meas_amps_signal = measured_amps[signal_start:]
    meas_phases_signal = measured_phases[signal_start:]

    # Calculate errors
    amplitude_errors_db = 20 * np.log10(meas_amps_signal / target_amps)

    # Unwrap phase difference to handle 2π wrapping
    phase_diff = meas_phases_signal - target_phases
    phase_errors_rad = np.angle(np.exp(1j * phase_diff))

    return amplitude_errors_db, phase_errors_rad


def update_chip_state(
    chip_state: ChipState,
    amplitude_errors_db: np.ndarray,
    phase_errors_rad: np.ndarray,
    measured_amps: np.ndarray,
    config: ExperimentConfig,
) -> ChipState:
    """
    Update chip state (MZI and PS powers) based on errors.

    Returns:
        updated_state: New ChipState with updated powers
    """
    lr = config.calibration.learning_rate
    p2pi = config.chip.p2pi_watts

    # Update phase shifter powers
    for i, tap_num in enumerate(range(9, 17)):
        if tap_num in chip_state.phase_shifters:
            ps = chip_state.phase_shifters[tap_num]

            # Update phase based on phase error
            phase_correction = -lr * phase_errors_rad[i]
            power_delta = (phase_correction / (2 * np.pi)) * ps.p2pi_watts

            ps.applied_power_watts += power_delta
            ps.applied_power_watts = np.clip(ps.applied_power_watts, 0, 1.0)

    # Update MZI powers based on amplitude errors
    # TODO: Implement proper MZI power splitting ratio update
    # This requires mapping tap amplitudes to MZI settings via binary tree

    print(f"  Chip state updated (LR={lr})")

    return chip_state


def run_calibration_iteration(
    iteration: int,
    chip_state: ChipState,
    target_amps: np.ndarray,
    target_phases: np.ndarray,
    config: ExperimentConfig,
) -> IterationData:
    """
    Run a single calibration iteration.

    Returns:
        IterationData containing measurements and errors for this iteration
    """
    print(f"\nIteration {iteration}:")

    # 1. Measure insertion loss spectrum
    wavelengths, insertion_loss = measure_insertion_loss_spectrum(config, chip_state)

    # 2. Recover phase via Kramers-Kronig
    phase_response = recover_phase_kramers_kronig(wavelengths, insertion_loss, config)

    # 3. Recover tap coefficients via inverse FFT
    tap_amps, tap_phases = recover_tap_coefficients(
        wavelengths, insertion_loss, phase_response, config
    )

    # 4. Calculate errors (only for signal processing taps)
    amp_errors, phase_errors = calculate_errors(
        tap_amps, tap_phases, target_amps, target_phases, config
    )

    # Calculate RMS errors
    rms_amp_error = np.sqrt(np.mean(amp_errors**2))
    rms_phase_error = np.sqrt(np.mean(phase_errors**2))

    print(f"  RMS amplitude error: {rms_amp_error:.3f} dB")
    print(f"  RMS phase error: {rms_phase_error:.3f} rad")

    # 5. Update chip state
    chip_state = update_chip_state(
        chip_state, amp_errors, phase_errors, tap_amps, config
    )

    # Extract current powers
    mzi_powers = {
        mzi_id: mzi.applied_power_watts for mzi_id, mzi in chip_state.mzis.items()
    }
    ps_powers = {
        tap: ps.applied_power_watts for tap, ps in chip_state.phase_shifters.items()
    }

    # Create iteration data
    iter_data = IterationData(
        iteration=iteration,
        wavelengths_nm=wavelengths,
        insertion_loss_db=insertion_loss,
        tap_amplitudes=tap_amps,
        tap_phases_rad=tap_phases,
        amplitude_errors_db=amp_errors,
        phase_errors_rad=phase_errors,
        rms_amplitude_error_db=rms_amp_error,
        rms_phase_error_rad=rms_phase_error,
        mzi_powers=mzi_powers,
        ps_powers=ps_powers,
    )

    return iter_data


def check_convergence(iter_data: IterationData, config: ExperimentConfig) -> bool:
    """Check if calibration has converged."""

    amp_converged = (
        iter_data.rms_amplitude_error_db < config.calibration.amplitude_tolerance_db
    )
    phase_converged = (
        iter_data.rms_phase_error_rad < config.calibration.phase_tolerance_rad
    )

    return amp_converged and phase_converged


def save_results(results: CalibrationResults, output_dir: str):
    """Save calibration results to output directory."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save convergence summary
    summary_path = Path(output_dir) / "calibration_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Calibration Results\n")
        f.write(f"==================\n\n")
        f.write(f"Experiment: {results.config.name}\n")
        f.write(f"Filter type: {results.config.target.filter_type}\n")
        f.write(f"Converged: {results.converged}\n")
        f.write(f"Final iteration: {results.final_iteration}\n")
        f.write(f"Total iterations: {len(results.iterations)}\n")

        if results.iterations:
            last_iter = results.iterations[-1]
            f.write(f"\nFinal errors:\n")
            f.write(
                f"  RMS amplitude error: {last_iter.rms_amplitude_error_db:.4f} dB\n"
            )
            f.write(f"  RMS phase error: {last_iter.rms_phase_error_rad:.4f} rad\n")

    print(f"\nResults saved to {output_dir}")

    # TODO: Save detailed iteration data, plots, etc.


def run_experiment(config_path: str):
    """
    Main experiment function.

    Args:
        config_path: Path to YAML configuration file
    """

    print("=" * 60)
    print("FIR Chip Calibration Experiment")
    print("=" * 60)

    # Load configuration
    print(f"\nLoading configuration from: {config_path}")
    config = load_config(config_path)
    print(f"Experiment: {config.name}")
    print(f"Description: {config.description}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"{config.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save configuration
    save_config(config, str(output_dir))

    # Compute target tap coefficients
    print("\nComputing target filter response...")
    target_amps, target_phases = compute_target_tap_coefficients(config)
    print(f"Target filter: {config.target.filter_type}")
    print(f"  Number of taps: {len(target_amps)}")
    print(f"  Phase step: {config.target.phase_step_rad:.4f} rad")

    # Initialize chip state
    chip_state = config.initial_state

    # Run calibration iterations
    print("\n" + "=" * 60)
    print("Starting calibration...")
    print("=" * 60)

    iterations = []
    converged = False

    for i in range(config.calibration.max_iterations):
        # Run iteration
        iter_data = run_calibration_iteration(
            iteration=i + 1,
            chip_state=chip_state,
            target_amps=target_amps,
            target_phases=target_phases,
            config=config,
        )

        iterations.append(iter_data)

        # Check convergence
        if check_convergence(iter_data, config):
            print(f"\n*** Converged at iteration {i + 1} ***")
            converged = True
            break

    # Create results object
    results = CalibrationResults(
        config=config,
        iterations=iterations,
        converged=converged,
        final_iteration=len(iterations),
        final_amplitudes=iterations[-1].tap_amplitudes,
        final_phases_rad=iterations[-1].tap_phases_rad,
        final_state=chip_state,
    )

    # Save results
    print("\n" + "=" * 60)
    print("Calibration complete")
    print("=" * 60)
    save_results(results, str(output_dir))

    return results


if __name__ == "__main__":
    import sys

    # Get config path from command line or use default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "example_config.yaml"

    # Run experiment
    results = run_experiment(config_path)

    print(f"\nExperiment complete!")
    print(f"Converged: {results.converged}")
    print(f"Final iteration: {results.final_iteration}")
