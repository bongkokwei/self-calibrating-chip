import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys

from config import (
    ExperimentConfig,
    ChipState,
    IterationData,
    CalibrationResults,
    config_from_dict,
    config_to_dict,
    measure_spectrum,
    recover_impulse_response_from_df,
    detect_taps,
    calculate_all_errors,
    calculate_power_adjustments,
    apply_voltages_to_hardware,
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


def run_calibration_iteration(
    iteration: int,
    target_taps: Dict[int, complex],
    mzi_tree: Dict[str, Dict],  # TODO: incorporate this into data structures
    chip_state: ChipState,
    config: ExperimentConfig,
    prev_iter_data: Optional[IterationData] = None,
) -> IterationData:
    """
    Run a single calibration iteration.

    Returns:
        IterationData containing measurements and errors for this iteration
    """
    print(f"\nIteration {iteration}:")

    # 1. Measure spectrum
    df = measure_spectrum(
        center_wavelength_nm=config.measurement.center_wavelength_nm,
        wavelength_span_nm=config.measurement.wavelength_span_nm,
        num_averages=config.measurement.num_averages,
        edfa_port=config.measurement.edfa_port,
        edfa_baudrate=config.measurement.edfa_baudrate,
        edfa_output_power_dbm=config.measurement.edfa_output_power_dbm,
        ova_ip=config.measurement.ova_address,
    )

    # 2. Recover impulse response (using DataFrame wrapper)
    time_ps, h_time = recover_impulse_response_from_df(
        df=df,
        chip_params=config.chip,
        wavelength_col=config.measurement.wavelength_col,
        freq_col=config.measurement.frequency_col,
        insertion_loss_col=config.measurement.insertion_loss_col,
    )

    # 3. Detect taps
    tap_times, tap_coeffs = detect_taps(
        time_ps=time_ps,
        h_time=h_time,
        chip_params=config.chip,
        n_taps=config.chip.n_taps,
        use_db_scale=config.measurement.use_db_scale,
        prominence_factor_db=config.measurement.prominence_factor_db,
        height_threshold_db=config.measurement.height_threshold_db,
    )

    # 4. Calculate errors (only for signal processing taps)
    all_errors = calculate_all_errors(
        measured_taps=tap_coeffs,
        target_taps=target_taps,
        signal_tap_indices=config.chip.signal_tap_indices,
        signal_tap_numbers=config.chip.signal_tap_numbers,
        mzi_tree=mzi_tree,
        mzi_phi_init=chip_state.get_all_init_phase(),
        ps_phi_init=chip_state.get_all_init_phase(),
    )

    # 5. Calculate power adjustments
    (
        new_mzi_powers,
        new_ps_powers,
        phi_init_adjustments,
    ) = calculate_power_adjustments(
        mzi_phase_errors=all_errors["mzi_phase_errors"],
        ps_phase_errors=all_errors["ps_phase_errors"],
        mzi_psr_errors=all_errors["mzi_psr_errors"],
        prev_mzi_psr_errors=(
            prev_iter_data.mzi_psr_errors_db if prev_iter_data else None
        ),
        current_mzi_powers={
            mzi_id: mzi.applied_power_watts for mzi_id, mzi in chip_state.mzis.items()
        },
        current_ps_powers={
            tap: ps.applied_power_watts for tap, ps in chip_state.phase_shifters.items()
        },
        mzi_phi_init={
            mzi_id: mzi.phi_init_rad for mzi_id, mzi in chip_state.mzis.items()
        },
        ps_phi_init={
            tap: ps.phi_init_rad for tap, ps in chip_state.phase_shifters.items()
        },
        power_for_2pi=config.chip.p2pi_watts,
        learning_rate=config.calibration.learning_rate,
        min_power=config.calibration.min_power_watts,
        max_power=config.calibration.max_power_watts,
    )

    # 6. Update chip state (in-place)
    chip_state.update_powers(
        new_mzi_powers=new_mzi_powers,
        new_ps_powers=new_ps_powers,
        phi_init_adjustments=phi_init_adjustments,
    )

    # 7. Apply voltages to hardware
    apply_voltages_to_hardware(chip_state, config)

    # Create iteration data
    iter_data = IterationData(
        iteration=iteration,
        wavelengths_nm=df[config.measurement.wavelength_col],
        insertion_loss_db=df[config.measurement.insertion_loss_col],
        tap_amplitudes=np.abs(tap_coeffs),
        tap_phases_rad=np.angle(tap_coeffs),
        amplitude_errors_db=all_errors["amplitude_errors_db"],
        phase_errors_rad=all_errors["phase_errors_rad"],
        rms_amplitude_error_db=all_errors["rms_amplitude_error_db"],
        rms_phase_error_rad=all_errors["rms_phase_error_rad"],
        chip_state=chip_state.copy(),
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
    target_amps, target_phases = config.target.compute_target_taps(
        n_taps=config.chip.n_taps
    )
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
