"""
error_calculation.py

Functions for calculating power splitting ratio and phase shifter errors
for the photonic FIR chip calibration process.

Uses power_splitting_ratio.py for all PSR and phase conversions.
"""

import numpy as np
from typing import Dict, Tuple
from .power_splitting_ratio import (
    tap_coeffs_to_power_splitting_ratios,
    power_splitting_ratios_to_mzi_phases,
    extract_tap_phases,
)


def calculate_mzi_errors(
    measured_psr: Dict[str, float],
    target_psr: Dict[str, float],
    mzi_phi_init: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate MZI power splitting ratio and phase errors.

    Args:
        measured_psr: Measured power splitting ratios for each MZI in dB (dict keyed by MZI ID)
        target_psr: Target power splitting ratios for each MZI in dB (dict keyed by MZI ID)
        mzi_phi_init: Initial phase offsets for each MZI in radians (dict keyed by MZI ID)

    Returns:
        Tuple of (psr_errors, phase_errors) dictionaries keyed by MZI ID
        - psr_errors: Power splitting ratio errors in dB (target - measured)
        - phase_errors: MZI phase errors in radians (target - (measured - phi_init))
    """
    # Convert power splitting ratios to MZI phases
    measured_phases = power_splitting_ratios_to_mzi_phases(measured_psr)
    target_phases = power_splitting_ratios_to_mzi_phases(target_psr)

    # Calculate errors
    psr_errors = {}
    phase_errors = {}

    for mzi_id in target_psr.keys():
        measured_ratio = measured_psr.get(mzi_id, 0.0)
        target_ratio = target_psr[mzi_id]

        measured_phase = measured_phases.get(mzi_id, 0.0)
        target_phase = target_phases[mzi_id]
        phi_init = mzi_phi_init.get(mzi_id, 0.0)

        # Error = target - measured
        psr_errors[mzi_id] = target_ratio - measured_ratio

        # Phase error accounting for initial phase offset
        # phase_errors[mzi_id] = target_phase - (measured_phase - phi_init)
        phase_errors[mzi_id] = target_phase - measured_phase

    return psr_errors, phase_errors


def calculate_phase_shifter_errors(
    measured_phases: Dict[int, float],
    target_phases: Dict[int, float],
    ps_phi_init: Dict[int, float],
) -> Dict[int, float]:
    """
    Calculate phase shifter errors for signal processing taps.

    Args:
        measured_phases: Measured phases in radians (dict keyed by tap number)
        target_phases: Target phases in radians (dict keyed by tap number)
        ps_phi_init: Initial phase offsets in radians (dict keyed by tap number)

    Returns:
        Dictionary of phase errors in radians (target - (measured - phi_init)), keyed by tap number
    """
    phase_errors = {}

    for tap_num in target_phases.keys():
        measured_phase = measured_phases.get(tap_num, 0.0)
        target_phase = target_phases[tap_num]
        phi_init = ps_phi_init.get(tap_num, 0.0)

        # Error = target - (measured - initial_offset)
        # phase_errors[tap_num] = target_phase - (measured_phase - phi_init)
        phase_errors[tap_num] = target_phase - measured_phase

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
    target_taps: np.ndarray,
    signal_tap_indices: Tuple[int, ...],
    signal_tap_numbers: Tuple[int, ...],
    mzi_tree: Dict[str, Dict],
    mzi_phi_init: Dict[str, float],
    ps_phi_init: Dict[int, float],
) -> Dict:
    """
    Calculate all errors (MZI and phase shifter) for calibration iteration.

    Args:
        measured_taps: Measured complex tap coefficients from KK recovery (length n_taps=16)
        target_taps: Target complex tap coefficients, zero-padded to n_taps=16
                     (signal taps at indices signal_tap_indices, zeros elsewhere)
        signal_tap_indices: Indices of signal processing taps, e.g. (8, 9, ..., 15)
        signal_tap_numbers: Tap numbers for signal processing, e.g. (9, 10, ..., 16)
        mzi_tree: MZI tree structure from build_mzi_tree_structure()
        mzi_phi_init: Initial phase offsets for MZIs in radians (dict keyed by MZI ID)
        ps_phi_init: Initial phase offsets for phase shifters in radians (dict keyed by tap number)

    Returns:
        Dictionary containing all error metrics.
    """
    # =========================================================================
    # NORMALISE TO UNIT SIGNAL TAP POWER
    # Power sum is over signal taps only — excludes reference and unused taps.
    # Normalisation is applied to the full 16-element arrays so that
    # tap_coeffs_to_power_splitting_ratios receives correctly-aligned indices.
    # =========================================================================
    idx = list(signal_tap_indices)

    measured_power = np.sum(np.abs(measured_taps[idx]) ** 2)
    target_power = np.sum(np.abs(target_taps[idx]) ** 2)

    measured_norm = measured_taps / np.sqrt(measured_power)  # shape (16,)
    target_norm = target_taps / np.sqrt(target_power)  # shape (16,)

    # PSR: full 16-element arrays — tree indices 8-15 are now valid
    measured_psr = tap_coeffs_to_power_splitting_ratios(measured_norm, mzi_tree)
    target_psr = tap_coeffs_to_power_splitting_ratios(target_norm, mzi_tree)

    # MZI errors
    mzi_psr_errors, mzi_phase_errors = calculate_mzi_errors(
        measured_psr=measured_psr,
        target_psr=target_psr,
        mzi_phi_init=mzi_phi_init,
    )

    # =========================================================================
    # PHASE AND AMPLITUDE ERRORS
    # Slice to signal taps only for the remaining calculations.
    # Phase uses unnormalised taps — scaling doesn't affect angle.
    # Amplitude uses normalised taps — relative power distribution.
    # =========================================================================
    measured_signal = measured_taps[idx]  # (8,) unnormalised
    target_signal = target_taps[idx]  # (8,) unnormalised
    measured_signal_norm = measured_norm[idx]  # (8,) normalised
    target_signal_norm = target_norm[idx]  # (8,) normalised

    measured_phases = extract_tap_phases(measured_signal, signal_tap_numbers)
    target_phases = extract_tap_phases(target_signal, signal_tap_numbers)

    ps_phase_errors = calculate_phase_shifter_errors(
        measured_phases=measured_phases,
        target_phases=target_phases,
        ps_phi_init=ps_phi_init,
    )

    tap_amp_errors = 20 * np.log10(np.abs(target_signal_norm) + 1e-12) - 20 * np.log10(
        np.abs(measured_signal_norm) + 1e-12
    )

    tap_phase_errors = np.array(
        [
            wrap_phase_error(ps_phase_errors[tap_num])
            for tap_num in sorted(ps_phase_errors.keys())
        ]
    )

    rms_amp, rms_phase = calculate_rms_errors(
        amplitude_errors=tap_amp_errors,
        phase_errors=tap_phase_errors,
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
