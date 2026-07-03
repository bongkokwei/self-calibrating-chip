"""
probe_mode.py
=============

PS probe-mode correction for calculate_power_adjustments (see
voltage_adjustment.py).

Normally a PS's power step is driven directly by its phase error. That
breaks down once |phi_err| exceeds ps_probe_threshold_rad, because the
measured phase is ambiguous about how many multiples of the threshold
have been crossed. Probe mode instead accumulates a running probe target
in units of ps_probe_threshold_rad and drives the PS toward that target,
rather than the raw (possibly multi-cycle) phase error.
"""

from typing import Dict, Optional
import logging

import numpy as np

from ..core.data_structure import ChipState
from .power_math import compute_new_power

logger = logging.getLogger(__name__)


def apply_probe_correction(
    chip_state: ChipState,
    tap_num: int,
    phi_err: float,
    current_P: float,
    ps_phi_init: Dict[int, float],
    ps_measured_phases: Optional[Dict[int, float]],
    ps_probe_threshold_rad: float,
    ps_learning_rate: float,
    power_for_ps_2pi: float,
    min_power: float,
    max_power: float,
    wrap_phase: bool,
    ps_dead_zone_rad: float,
) -> float:
    """
    Probe-mode PS correction, used once |phi_err| exceeds ps_probe_threshold_rad.

    Rather than stepping on the raw (ambiguous, possibly multi-cycle) phase
    error, accumulate a probe target in units of ps_probe_threshold_rad and
    drive toward that instead.
    """
    phi_init = ps_phi_init.get(tap_num, 0.0)
    phi_measured = (
        ps_measured_phases[tap_num] if ps_measured_phases is not None else 0.0
    )

    if chip_state.phase_shifters[tap_num].target_probe_rad is None:
        chip_state.phase_shifters[tap_num].target_probe_rad = 0.0

    chip_state.phase_shifters[tap_num].target_probe_rad += (
        np.sign(phi_err) * ps_probe_threshold_rad
    )
    probe_target = chip_state.phase_shifters[tap_num].target_probe_rad
    probe_err = float(np.angle(np.exp(1j * (phi_measured - probe_target - phi_init))))

    new_P, delta_P = compute_new_power(
        probe_err,
        current_P,
        ps_learning_rate,
        power_for_ps_2pi,
        min_power,
        max_power,
        wrap=wrap_phase,
        dead_zone=ps_dead_zone_rad,
    )
    logger.warning(
        f"  PS {tap_num}: PROBE MODE triggered |φ_err|={abs(phi_err):.4f} > "
        f"{ps_probe_threshold_rad:.4f} rad → probe_target={probe_target:+.4f} rad, "
        f"ΔP={delta_P:.4f} W → P={new_P:.4f} W"
    )
    return new_P
