"""
power_math.py
=============

Low-level power-update primitives shared by voltage_adjustment.py,
probe_mode.py, and spsa_gradient.py:

    ΔP = (error / 2π) × P_2π × lr
    P_new = clip(P + ΔP,  wrap by P_2π)
"""

from typing import Optional
import numpy as np


def wrap_and_clip_power(
    new_P: float,
    power_for_2pi: float,
    min_power: float,
    max_power: float,
    wrap: bool = True,
) -> float:
    """Apply modulo-P_2π phase wrap (if enabled) then hard-clip to [min_power, max_power].

    Wrap margins are symmetric (±25% of P_2π) so a converged power sitting
    near either boundary (e.g. P≈0) doesn't get pushed across it and trigger
    a spurious full-P_2π jump.
    """
    if wrap:
        if new_P < -0.25 * power_for_2pi:
            new_P += power_for_2pi
        elif new_P > 1.25 * power_for_2pi:
            new_P -= power_for_2pi

    return float(np.clip(new_P, min_power, max_power))


def compute_new_power(
    error: float,
    current_P: float,
    lr: float,
    power_for_2pi: float,
    min_power: float,
    max_power: float,
    wrap: bool = True,
    dead_zone: float = 0.0,
    gate_err: Optional[float] = None,
) -> tuple[float, float]:
    """
    Compute updated heater power from an error signal.

    Parameters
    ----------
    error : float
        Error driving the power step (φ_err in rad for PS; φ_err in rad for MZI).
    current_P : float
        Current heater power (W).
    lr : float
        Learning rate (scalar).
    power_for_2pi : float
        Power corresponding to a 2π phase shift (W).
    min_power, max_power : float
        Hard clipping bounds (W).
    wrap : bool
        Apply modulo-P_2π phase wrap when P goes out of [0, 1.25·P_2π].
    dead_zone : float
        Skip update if |gate_err| < dead_zone.  Same units as gate_err.
    gate_err : float or None
        Signal used for the dead-zone check.  Defaults to error if None.
        Use this when the dead-zone signal differs from the step signal
        (e.g. MZI: gate on PSR dB, step on φ_err rad).

    Returns
    -------
    new_P : float
    delta_P : float
    """
    if abs(gate_err if gate_err is not None else error) < dead_zone:
        return current_P, 0.0

    delta_P = (error / (2 * np.pi)) * power_for_2pi * lr
    new_P = current_P + delta_P

    return wrap_and_clip_power(new_P, power_for_2pi, min_power, max_power, wrap), delta_P
