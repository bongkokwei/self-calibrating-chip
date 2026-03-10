import pandas as pd
import numpy as np
import voltage_ctrl
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys
import time
from pprint import pprint


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

try:
    from photonic_fir.hardware import (
        calculate_power_adjustments,
        apply_voltages_to_hardware,
    )

    _HARDWARE_AVAILABLE = True
except ImportError:
    _HARDWARE_AVAILABLE = False

from photonic_fir.utils import CalibrationPlotter, plot_impulse_response
from .phi_init_characterisation import characterise_mzi_phi_init
from .measurement_pipeline import measure_and_detect_taps


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
    calibration_mode: str = "simultaneous",
) -> IterationData:
    """
    Run a single calibration iteration.

    Args:
        iteration: Current iteration number (1-indexed).
        target_taps: Target complex tap coefficients, zero-padded to n_taps.
        mzi_tree: MZI tree structure from build_mzi_tree_structure().
        chip_state: Current chip state (modified in-place).
        config: Experiment configuration.
        prev_iter_data: Data from previous iteration (None for first iteration).
        output_dir: Directory to save per-iteration plots.
        calibration_mode: One of:
            "simultaneous"   – update MZIs and phase shifters every iteration (default).
            "amplitude_only" – MZI PSR corrections only; PS phase corrections zeroed.
            "phase_only"     – PS phase corrections only; MZI PSR corrections zeroed.

    Returns:
        IterationData containing measurements and errors for this iteration.
    """
    logger.info(f"\nIteration {iteration}:")

    with VoltageController(
        com_port=config.measurement.voltage_controller_port,
        baud_rate=config.measurement.voltage_controller_baudrate,
        zero_on_exit=True,
    ) as voltage_ctrl:

        logger.info("\nApplying initial voltages to hardware...")
        apply_voltages_to_hardware(chip_state, config, voltage_ctrl)

        logger.info(
            f"\nWaiting for system to settle for {config.measurement.settling_time_sec} seconds..."
        )
        time.sleep(config.measurement.settling_time_sec)

        logger.info("\nMeasuring impulse response and detecting taps...")
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
        # save_fig=output_dir + f"/iteration_{iteration}_impulse_response.png",
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

    # --- Sequential mode: suppress the inactive correction loop ---
    if calibration_mode == "amplitude_only":
        # Suppress PS phase corrections; MZI PSR + φ_init flip logic remain active
        logger.info("  [amplitude_only] Suppressing PS phase corrections")
        all_errors["ps_phase_errors"] = {k: 0.0 for k in all_errors["ps_phase_errors"]}
    elif calibration_mode == "phase_only":
        # Suppress MZI PSR corrections; PS phase corrections remain active
        logger.info("  [phase_only] Suppressing MZI PSR and phase corrections")
        all_errors["mzi_psr_errors"] = {k: 0.0 for k in all_errors["mzi_psr_errors"]}
        all_errors["mzi_phase_errors"] = {
            k: 0.0 for k in all_errors["mzi_phase_errors"]
        }
    # "simultaneous" → no changes to all_errors

    # Mask disabled PS taps (for crosstalk isolation experiments)
    disabled_ps_taps = [10, 11, 12, 13, 14, 15, 16]
    if disabled_ps_taps and False:
        for tap in disabled_ps_taps:
            if tap in all_errors["ps_phase_errors"]:
                all_errors["ps_phase_errors"][tap] = 0.0
        logger.info(f"  [crosstalk test] Disabled PS taps: {disabled_ps_taps}")

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
        prev_ps_phase_errors=(
            prev_iter_data.ps_phase_errors if prev_iter_data else None
        ),
        current_mzi_powers=chip_state.get_mzi_applied_powers(),
        current_ps_powers=chip_state.get_ps_applied_powers(),
        power_for_mzi_2pi=config.chip.p2pi_watts_mzi,
        power_for_ps_2pi=config.chip.p2pi_watts_ps,
        learning_rate=config.calibration.learning_rate,
        min_power=config.calibration.min_power_watts,
        max_power=config.calibration.max_power_watts,
        psr_increase_threshold_db=config.calibration.psr_increase_threshold_db,
        wrap_phase=config.calibration.wrap_phase,
        **config.calibration.adaptive_lr_kwargs(),
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
        ps_phase_errors=all_errors["ps_phase_errors"],
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
    if config.calibration.use_two_step_init:
        logger.info("\nMeasuring initial MZI phases (φ_init)...")
        phi_init_measurement(config, chip_state)

    # Build MZI tree structure
    mzi_tree = config.signal_mzi_tree.tree

    # Initialise plotter after config is loaded
    plotter = CalibrationPlotter(
        num_taps=config.chip.n_signal_taps,  # 8 taps
        num_mzis=len(config.signal_mzi_tree.mzi_ids),  # Number of MZIs
    )

    # Log calibration mode
    if config.calibration.sequential_mode:
        _stability_window = getattr(config.calibration, "amplitude_stability_window", 3)
        logger.info(
            "\nCalibration mode: SEQUENTIAL "
            f"(amplitude first, switch to phase after {_stability_window} consecutive "
            f"iterations with amp RMS < {config.calibration.amplitude_tolerance_db:.2f} dB)"
        )
    else:
        logger.info("\nCalibration mode: SIMULTANEOUS")

    # Run calibration iterations
    logger.info("\n" + "=" * 60)
    logger.info("Starting calibration...")
    logger.info("=" * 60)

    iterations = []
    converged = False
    prev_iter_data = None

    _amp_stable_count = 0  # consecutive iters below amp threshold
    _stability_window = getattr(config.calibration, "amplitude_stability_window", 3)
    _FALLBACK_MULTIPLIER = 2.0  # re-engage amp_only if amp exceeds tol * this
    _phase_mode_active = False  # latched True once stability window satisfied

    try:

        for i in range(config.calibration.max_iterations):

            # Determine calibration mode for this iteration
            if config.calibration.sequential_mode:
                if prev_iter_data is not None:
                    amp_rms = prev_iter_data.rms_amplitude_error_db
                    amp_tol = config.calibration.amplitude_tolerance_db

                    # Track consecutive iterations below threshold
                    if amp_rms < amp_tol:
                        _amp_stable_count += 1
                    else:
                        _amp_stable_count = 0

                    # Latch into phase_only once stability window is satisfied
                    if _amp_stable_count >= _stability_window:
                        _phase_mode_active = True

                    # Fallback: if amplitude degrades significantly during phase_only,
                    # return to amplitude_only to re-stabilise before continuing.
                    # This handles PS→MZI cross-coupling that can disturb amplitude
                    # after the loop switches to phase corrections.
                    if _phase_mode_active and amp_rms > _FALLBACK_MULTIPLIER * amp_tol:
                        logger.warning(
                            f"  Sequential mode: amplitude degraded "
                            f"({amp_rms:.2f} dB > {_FALLBACK_MULTIPLIER * amp_tol:.2f} dB), "
                            f"falling back to amplitude_only"
                        )
                        _phase_mode_active = False
                        _amp_stable_count = 0

                    calibration_mode = (
                        "phase_only" if _phase_mode_active else "amplitude_only"
                    )
                    logger.info(
                        f"  Sequential mode: {calibration_mode} "
                        f"(amp RMS = {amp_rms:.2f} dB, "
                        f"stable count = {_amp_stable_count}/{_stability_window}, "
                        f"tol = {amp_tol:.2f} dB)"
                    )
                else:
                    calibration_mode = "amplitude_only"
                    logger.info("  Sequential mode: amplitude_only (first iteration)")
            else:
                calibration_mode = "simultaneous"

            # Run iteration
            iter_data = run_calibration_iteration(
                iteration=i + 1,
                target_taps=target_taps,
                mzi_tree=mzi_tree,
                chip_state=chip_state,
                config=config,
                prev_iter_data=prev_iter_data,
                output_dir=str(output_dir),
                calibration_mode=calibration_mode,
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
