"""
error_calculation.py

Functions for calculating power splitting ratio and phase shifter errors
for the photonic FIR chip calibration process.
"""

import sys
import numpy as np
from typing import Dict, Tuple

sys.path.append("../..")
from simulation.power_splitting_ratio import PowerSplittingCalculator
from data_structure import ChipState, TargetFilter


def calculate_mzi_errors(
    measured_taps: np.ndarray,
    current_state: ChipState,
    target_power_ratios: Dict[str, float],
    psr_calculator: PowerSplittingCalculator,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate MZI power splitting ratio and phase errors.

    Args:
        measured_taps: Complex tap coefficients from Kramers-Kronig recovery (length n_taps)
        current_state: Current chip state with MZI information
        target_power_ratios: Desired power splitting ratios for each MZI
        psr_calculator: Calculator for power splitting ratio conversions

    Returns:
        Tuple of (power_ratio_errors, phase_errors) dictionaries keyed by MZI ID
    """
    # Extract signal processing taps (taps 9-16 for the 16-tap chip)
    signal_taps = measured_taps[8:16]  # Indices 8-15 correspond to taps 9-16

    # Calculate measured power splitting ratios from tap coefficients
    measured_psr = psr_calculator.tap_coeffs_to_power_splitting_ratios(signal_taps)

    # Convert power splitting ratios to MZI phases
    measured_mzi_phases = {
        mzi_id: psr_calculator.power_splitting_ratio_to_mzi_phase(ratio)
        for mzi_id, ratio in measured_psr.items()
    }

    target_mzi_phases = {
        mzi_id: psr_calculator.power_splitting_ratio_to_mzi_phase(ratio)
        for mzi_id, ratio in target_power_ratios.items()
    }

    # Calculate errors
    psr_errors = {}
    phase_errors = {}

    for mzi_id in target_power_ratios.keys():
        measured_ratio = measured_psr.get(mzi_id, 0.0)
        target_ratio = target_power_ratios[mzi_id]

        measured_phase = measured_mzi_phases.get(mzi_id, 0.0)
        target_phase = target_mzi_phases[mzi_id]

        # Error = target - measured
        psr_errors[mzi_id] = target_ratio - measured_ratio

        # Phase error accounting for initial phase offset
        mzi_state = current_state.mzis[mzi_id]
        phase_errors[mzi_id] = target_phase - (measured_phase - mzi_state.phi_init_rad)

    return psr_errors, phase_errors


def calculate_phase_shifter_errors(
    measured_taps: np.ndarray,
    current_state: ChipState,
    target_taps: np.ndarray,
) -> Dict[int, float]:
    """
    Calculate phase shifter errors for signal processing taps.

    Args:
        measured_taps: Measured complex tap coefficients (length n_taps)
        current_state: Current chip state with phase shifter information
        target_taps: Target complex tap coefficients (length n_signal_taps)

    Returns:
        Dictionary of phase errors (radians) keyed by tap number (9-16)
    """
    # Extract signal processing taps (taps 9-16)
    measured_signal_taps = measured_taps[8:16]

    # Calculate phases
    measured_phases = np.angle(measured_signal_taps)
    target_phases = np.angle(target_taps)

    # Calculate errors accounting for initial phase offsets
    phase_errors = {}
    for idx, tap_num in enumerate(range(9, 17)):
        ps_state = current_state.phase_shifters[tap_num]

        # Error = target - (measured - initial_offset)
        phase_errors[tap_num] = target_phases[idx] - (
            measured_phases[idx] - ps_state.phi_init_rad
        )

    return phase_errors


def calculate_rms_errors(
    amplitude_errors: np.ndarray,
    phase_errors: np.ndarray,
) -> Tuple[float, float]:
    """
    Calculate RMS errors for amplitude and phase.

    Args:
        amplitude_errors: Amplitude errors in dB
        phase_errors: Phase errors in radians

    Returns:
        Tuple of (rms_amplitude_error_db, rms_phase_error_rad)
    """
    rms_amp = np.sqrt(np.mean(amplitude_errors**2))
    rms_phase = np.sqrt(np.mean(phase_errors**2))

    return rms_amp, rms_phase


def wrap_phase_error(phase_error: float) -> float:
    """
    Wrap phase error to [-π, π] range.

    Args:
        phase_error: Phase error in radians

    Returns:
        Wrapped phase error in [-π, π]
    """
    return np.arctan2(np.sin(phase_error), np.cos(phase_error))


def calculate_all_errors(
    measured_taps: np.ndarray,
    current_state: ChipState,
    target_power_ratios: Dict[str, float],
    target_taps: np.ndarray,
    psr_calculator: PowerSplittingCalculator,
) -> Dict:
    """
    Calculate all errors (MZI and phase shifter) for calibration iteration.

    Args:
        measured_taps: Measured complex tap coefficients from KK recovery
        current_state: Current chip state
        target_power_ratios: Target power splitting ratios for MZIs
        target_taps: Target complex tap coefficients for signal processing
        psr_calculator: Power splitting ratio calculator

    Returns:
        Dictionary containing all error metrics:
        - 'mzi_psr_errors': Power splitting ratio errors (dB)
        - 'mzi_phase_errors': MZI phase errors (rad)
        - 'ps_phase_errors': Phase shifter errors (rad)
        - 'tap_amplitude_errors': Tap amplitude errors (dB)
        - 'tap_phase_errors': Tap phase errors (rad)
        - 'rms_amplitude_error': RMS amplitude error (dB)
        - 'rms_phase_error': RMS phase error (rad)
    """
    # Calculate MZI errors
    mzi_psr_errors, mzi_phase_errors = calculate_mzi_errors(
        measured_taps=measured_taps,
        current_state=current_state,
        target_power_ratios=target_power_ratios,
        psr_calculator=psr_calculator,
    )

    # Calculate phase shifter errors
    ps_phase_errors = calculate_phase_shifter_errors(
        measured_taps=measured_taps,
        current_state=current_state,
        target_taps=target_taps,
    )

    # Extract signal processing taps for amplitude error calculation
    measured_signal_taps = measured_taps[8:16]

    # Calculate tap amplitude errors (in dB)
    measured_amps = 20 * np.log10(np.abs(measured_signal_taps) + 1e-12)
    target_amps = 20 * np.log10(np.abs(target_taps) + 1e-12)
    tap_amp_errors = target_amps - measured_amps

    # Calculate tap phase errors
    tap_phase_errors = {
        tap_num: wrap_phase_error(err) for tap_num, err in ps_phase_errors.items()
    }

    # Calculate RMS errors
    rms_amp, rms_phase = calculate_rms_errors(
        amplitude_errors=tap_amp_errors,
        phase_errors=np.array(list(ps_phase_errors.values())),
    )

    return {
        "mzi_psr_errors": mzi_psr_errors,
        "mzi_phase_errors": mzi_phase_errors,
        "ps_phase_errors": ps_phase_errors,
        "tap_amplitude_errors": tap_amp_errors,
        "tap_phase_errors": tap_phase_errors,
        "rms_amplitude_error": rms_amp,
        "rms_phase_error": rms_phase,
    }
