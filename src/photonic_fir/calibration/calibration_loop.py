import pandas as pd
import numpy as np
import voltage_ctrl
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys
import time


import logging
from photonic_fir import setup_logging

logger = logging.getLogger(__name__)

from photonic_fir.core import (
    ExperimentConfig,
    ChipState,
    IterationData,
    CalibrationResults,
    calculate_all_errors,
    load_config,
    save_config,
)
from photonic_fir.hardware import (
    measure_spectrum,
    calculate_power_adjustments,
    apply_voltages_to_hardware,
    set_mzi_voltage,
)
from photonic_fir.processing import (
    recover_impulse_response_from_df,
    detect_taps,
    detect_taps_noise_tolerant,
)
from photonic_fir.utils import CalibrationPlotter, plot_impulse_response
from photonic_fir.calibration import characterise_mzi_phi_init, measure_and_detect_taps


from voltage_ctrl import VoltageController


def compute_target_taps(config: ExperimentConfig) -> np.ndarray:
    """
    Compute and pad target tap coefficients for use with the full MZI tree.

    Generates n_signal_taps coefficients then prepends zeros to align signal
    taps to their physical positions in the 16-tap tree (indices 8-15).

    Args:
        config: Experiment configuration

    Returns:
        Complex tap array of length n_taps (e.g. 16), zero-padded at the front.
    """
    signal_taps = config.target.get_target_taps(n_taps=config.chip.n_signal_taps)

    n_pad = config.chip.n_taps - len(signal_taps)  # 16 - 8 = 8
    target_taps = np.concatenate([np.zeros(n_pad, dtype=complex), signal_taps])

    logger.info("\nComputing target filter response...")
    logger.info(f"Target filter: {config.target.filter_type}")
    logger.info(f"  Signal taps: {len(signal_taps)}")
    logger.info(f"  Padded to:   {len(target_taps)} (prepended {n_pad} zeros)")
    logger.info(f"  Phase step:  {config.target.phase_step_rad:.4f} rad")

    return target_taps


def phi_init_measurement(config: ExperimentConfig, chip_state: ChipState):
    """Measure and populate φ_init values in chip_state."""

    # Initialize hardware
    with VoltageController(
        com_port=config.measurement.voltage_controller_port,
        baud_rate=config.measurement.voltage_controller_baudrate,
        zero_on_exit=True,
    ) as voltage_ctrl:
        # Measure and populate φ_init
        characterise_mzi_phi_init(
            chip_state=chip_state,
            config=config,
            voltage_ctrl=voltage_ctrl,
            perturbation_power_watts=0.05,
            mzi_ids=config.chip.get_signal_mzi_ids(),
        )

        logger.info("\nVerifying φ_init values in chip_state:")
        for mzi_id, mzi in chip_state.mzis.items():
            logger.info(f"  MZI {mzi_id}: φ_init = {mzi.phi_init_rad:+7.3f} rad")

        for ps_id, ps in chip_state.phase_shifters.items():
            logger.info(f"  PS {ps_id}: φ_init = {ps.phi_init_rad:+7.3f} rad")


def run_calibration_iteration(
    iteration: int,
    target_taps: np.ndarray,
    mzi_tree: Dict[str, Dict],
    chip_state: ChipState,
    config: ExperimentConfig,
    prev_iter_data: Optional[IterationData] = None,
    output_dir: str = "",
) -> IterationData:
    """
    Run a single calibration iteration.

    Returns:
        IterationData containing measurements and errors for this iteration
    """
    logger.info(f"\nIteration {iteration}:")

    with VoltageController(
        com_port=config.measurement.voltage_controller_port,
        baud_rate=config.measurement.voltage_controller_baudrate,
        zero_on_exit=True,
    ) as voltage_ctrl:

        logger.info("\nApplying initial voltages to hardware...")
        apply_voltages_to_hardware(chip_state, config, voltage_ctrl)

        time.sleep(config.measurement.settling_time_sec)

        df, tap_times, tap_coeffs, time_ps, h_time = measure_and_detect_taps(
            config=config,
            folder_dir=None,
            file_name=None,
        )

    plot_impulse_response(
        time_ps=time_ps,
        h_time=h_time,
        tap_times_ps=tap_times,
        tap_coeffs=tap_coeffs,
        save_fig=output_dir + f"/iteration_{iteration}_impulse_response.png",
    )

    # 4. Calculate errors (only for signal processing taps)
    all_errors = calculate_all_errors(
        measured_taps=tap_coeffs,
        target_taps=target_taps,
        signal_tap_indices=config.chip.signal_tap_indices,
        signal_tap_numbers=config.chip.signal_tap_numbers,
        mzi_tree=mzi_tree,
        mzi_phi_init=chip_state.get_mzi_init_phase(),
        ps_phi_init=chip_state.get_ps_init_phase(),
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
        current_mzi_powers=chip_state.get_mzi_applied_powers(),
        current_ps_powers=chip_state.get_ps_applied_powers(),
        mzi_phi_init=chip_state.get_mzi_init_phase(),
        ps_phi_init=chip_state.get_ps_init_phase(),
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

    # Create iteration data
    iter_data = IterationData(
        iteration=iteration,
        wavelengths_nm=df[config.measurement.wavelength_col],
        insertion_loss_db=df[config.measurement.insertion_loss_col],
        tap_amplitudes=np.abs(tap_coeffs),
        tap_phases_rad=np.angle(tap_coeffs),
        amplitude_errors_db=all_errors["tap_amplitude_errors"],
        phase_errors_rad=all_errors["tap_phase_errors"],
        rms_amplitude_error_db=all_errors["rms_amplitude_error"],
        rms_phase_error_rad=all_errors["rms_phase_error"],
        mzi_psr_errors_db=all_errors["mzi_psr_errors"],
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

    logger.info(f"\nResults saved to {output_dir}")


def run_experiment(config_path: str):
    """
    Main experiment function.

    Args:
        config_path: Path to YAML configuration file
    """

    # Load configuration
    config = load_config(config_path)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"{config.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to file in output directory
    setup_logging(
        log_file=str(output_dir / f"experiment_log_{timestamp}.log"),
        level="INFO",
    )

    logger.info("=" * 60)
    logger.info("FIR Chip Calibration Experiment")
    logger.info("=" * 60)

    logger.info(f"\nLoading configuration from: {config_path}")
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Description: {config.description}")

    # Create initial chip state
    chip_state = ChipState(
        chip_params=config.chip,
    )

    # Copy initial voltage settings from calibration to chip state (CHECKED)
    for mzi_id, mzi in chip_state.mzis.items():
        if mzi_id in config.calibration.initial_mzi_voltages:
            voltage = config.calibration.initial_mzi_voltages[mzi_id]
            resistance_ohms = config.chip.heater_resistance_ohm
            mzi.applied_power_watts = voltage**2 / resistance_ohms

    # Save configuration
    save_config(config, str(output_dir))

    # Compute target tap coefficients
    target_taps = compute_target_taps(config)

    # Measure phi_init, and populate chip_state
    logger.info("\nMeasuring initial MZI phases (φ_init)...")
    phi_init_measurement(config, chip_state)

    # Build MZI tree structure
    mzi_tree = config.signal_mzi_tree.tree

    # Initialise plotter after config is loaded
    plotter = CalibrationPlotter(
        num_taps=config.chip.n_signal_taps,  # 8 taps
        num_mzis=len(config.signal_mzi_tree.mzi_ids),  # Number of MZIs
    )

    # Run calibration iterations
    logger.info("\n" + "=" * 60)
    logger.info("Starting calibration...")
    logger.info("=" * 60)

    iterations = []
    converged = False
    prev_iter_data = None

    try:

        for i in range(config.calibration.max_iterations):
            # Run iteration
            iter_data = run_calibration_iteration(
                iteration=i + 1,
                target_taps=target_taps,
                mzi_tree=mzi_tree,
                chip_state=chip_state,
                config=config,
                prev_iter_data=prev_iter_data,
                output_dir=str(output_dir),
            )

            iterations.append(iter_data)
            prev_iter_data = iter_data

            plotter.update(iter_data)

            # Check convergence
            if check_convergence(iter_data, config):
                logger.info(f"\n*** Converged at iteration {i + 1} ***")
                converged = True
                break
    except Exception as e:
        logger.info(f"\nError during calibration at iteration {i + 1}: {e}")
        with VoltageController(
            com_port=config.measurement.voltage_controller_port,
            baud_rate=config.measurement.voltage_controller_baudrate,
            zero_on_exit=True,
        ) as voltage_ctrl:
            voltage_ctrl.set_voltages(
                channels=np.arange(1, 33),
                voltages=[0.0] * 32,
                v_max=config.measurement.voltage_controller_v_max,
            )
            logger.info("\nResetting voltages to zero...")
    finally:
        # Save plots before closing
        plotter.save_plots(str(output_dir))
        plotter.close()

    # Create results object
    results = CalibrationResults(
        config=config,
        iterations=iterations,
        converged=converged,
        final_iteration=len(iterations),
        final_amplitudes=iterations[-1].tap_amplitudes if iterations else None,
        final_phases_rad=iterations[-1].tap_phases_rad if iterations else None,
        final_state=chip_state,
    )

    # Save results
    logger.info("\n" + "=" * 60)
    logger.info("Calibration complete")
    logger.info("=" * 60)
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

    logger.info(f"\nExperiment complete!")
    logger.info(f"Converged: {results.converged}")
    logger.info(f"Final iteration: {results.final_iteration}")
