"""
error_calculation.py

Functions for calculating power splitting ratio and phase shifter errors
for the photonic FIR chip calibration process.

Uses power_splitting_ratio.py for all PSR and phase conversions.
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

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


def _log_dict(label: str, d: dict, fmt: str = ".6f") -> None:
    logger.info(f"{label}:")
    for k, v in sorted(d.items()):
        logger.info(f"  {k}: {v:{fmt}}")


def _log_array(label: str, arr: np.ndarray, fmt: str = ".6f") -> None:
    logger.info(f"{label}:")
    for i, v in enumerate(arr):
        if np.iscomplexobj(v):
            logger.info(
                f"  [{i}]: {v.real:{fmt}} + {v.imag:{fmt}}j  |mag|={np.abs(v):{fmt}}  arg={np.degrees(np.angle(v)):+.2f}°"
            )
        else:
            logger.info(f"  [{i}]: {v:{fmt}}")


def calculate_all_errors(
    measured_taps: np.ndarray,
    target_taps: np.ndarray,
    signal_tap_indices: Tuple[int, ...],
    signal_tap_numbers: Tuple[int, ...],
    mzi_tree: Dict[str, Dict],
    mzi_phi_init: Dict[str, float],
    ps_phi_init: Dict[int, float],
) -> Dict:
    idx = list(signal_tap_indices)

    # --- Raw tap inputs ---
    logger.info("=== calculate_all_errors ===")
    logger.info(f"signal_tap_indices : {signal_tap_indices}")
    logger.info(f"signal_tap_numbers : {signal_tap_numbers}")
    _log_array("measured_taps (all 16)", measured_taps, fmt=".6f")
    _log_array("target_taps  (all 16)", target_taps, fmt=".6f")

    # --- Normalisation ---
    measured_power = np.sum(np.abs(measured_taps[idx]) ** 2)
    target_power = np.sum(np.abs(target_taps[idx]) ** 2)
    logger.info(f"measured_power (signal taps): {measured_power:.6f}")
    logger.info(f"target_power   (signal taps): {target_power:.6f}")

    measured_norm = measured_taps / np.sqrt(measured_power)
    target_norm = target_taps / np.sqrt(target_power)
    _log_array("measured_norm (all 16)", np.abs(measured_norm), fmt=".6f")
    _log_array("target_norm   (all 16)", np.abs(target_norm), fmt=".6f")

    # --- PSR ---
    measured_psr = tap_coeffs_to_power_splitting_ratios(measured_norm, mzi_tree)
    target_psr = tap_coeffs_to_power_splitting_ratios(target_norm, mzi_tree)
    _log_dict("measured_psr (dB)", measured_psr)
    _log_dict("target_psr (dB)", target_psr)

    # --- MZI errors ---
    mzi_psr_errors, mzi_phase_errors = calculate_mzi_errors(
        measured_psr=measured_psr,
        target_psr=target_psr,
        mzi_phi_init=mzi_phi_init,
    )
    _log_dict("mzi_psr_errors (dB)", mzi_psr_errors)
    _log_dict("mzi_phase_errors (rad)", mzi_phase_errors)

    # --- Signal-tap slices ---
    measured_signal = measured_taps[idx]
    target_signal = target_taps[idx]
    measured_signal_norm = measured_norm[idx]
    target_signal_norm = target_norm[idx]
    _log_array("measured_signal (signal taps)", np.abs(measured_signal), fmt=".6f")
    _log_array("target_signal   (signal taps)", np.abs(target_signal), fmt=".6f")
    _log_array(
        "measured_signal_norm (signal taps)", np.abs(measured_signal_norm), fmt=".6f"
    )
    _log_array(
        "target_signal_norm   (signal taps)", np.abs(target_signal_norm), fmt=".6f"
    )

    # --- Phases ---
    measured_phases = extract_tap_phases(measured_signal, signal_tap_numbers)
    target_phases = extract_tap_phases(target_signal, signal_tap_numbers)
    _log_dict("measured_phases (rad)", measured_phases)
    _log_dict("target_phases   (rad)", target_phases)

    # --- PS phase errors ---
    ps_phase_errors = calculate_phase_shifter_errors(
        measured_phases=measured_phases,
        target_phases=target_phases,
        ps_phi_init=ps_phi_init,
    )
    _log_dict("ps_phase_errors (rad)", ps_phase_errors)

    # --- Amplitude errors ---
    tap_amp_errors = 20 * np.log10(np.abs(target_signal_norm) + 1e-12) - 20 * np.log10(
        np.abs(measured_signal_norm) + 1e-12
    )
    _log_array("tap_amp_errors (dB, target - measured)", tap_amp_errors, fmt=".6f")

    tap_phase_errors = np.array(
        [ps_phase_errors[tap_num] for tap_num in sorted(ps_phase_errors.keys())]
    )
    _log_array("tap_phase_errors (rad, target - measured)", tap_phase_errors, fmt=".6f")

    # --- RMS ---
    rms_amp, rms_phase = calculate_rms_errors(
        amplitude_errors=tap_amp_errors,
        phase_errors=tap_phase_errors,
    )
    _log_array("rms_amplitude_error (dB)", [rms_amp], fmt=".6f")
    _log_array("rms_phase_error    (rad)", [rms_phase], fmt=".6f")

    return {
        "mzi_psr_errors": mzi_psr_errors,
        "mzi_phase_errors": mzi_phase_errors,
        "ps_phase_errors": ps_phase_errors,
        "tap_amplitude_errors": tap_amp_errors,
        "tap_phase_errors": tap_phase_errors,
        "rms_amplitude_error": rms_amp,
        "rms_phase_error": rms_phase,
    }
