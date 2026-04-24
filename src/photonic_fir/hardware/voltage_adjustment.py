"""
voltage_adjustment.py
=====================

Power-adjustment helpers for the MZI and PS calibration loops.

Both actuators share the same update skeleton via _compute_new_power:

    ΔP = (error / 2π) × P_2π × lr
    P_new = clip(P + ΔP,  wrap by P_2π)

Differences:
  MZI  — dead zone gated on PSR error (dB); step driven by φ_err (rad).
  PS   — dead zone and step both driven by φ_err (rad); optional Rprop
         adaptive LR; optional crosstalk-decoupled update.

φ_init flipping on PSR divergence has been moved to calibration_loop.py.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

from ..core.data_structure import (
    ChipState,
    ExperimentConfig,
)
from voltage_ctrl import VoltageController


# ---------------------------------------------------------------------------
# Shared inner helpers
# ---------------------------------------------------------------------------


def _compute_new_power(
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

    if wrap:
        if new_P < 0:
            new_P += power_for_2pi
        elif new_P > 1.25 * power_for_2pi:
            new_P -= power_for_2pi

    return float(np.clip(new_P, min_power, max_power)), delta_P


def _effective_lr(
    error: float,
    prev_err: Optional[float],
    base_lr: float,
    adaptive: bool,
    **rprop_kw,
) -> float:
    """Return base_lr, or Rprop-adjusted LR if adaptive=True."""
    if not adaptive:
        return base_lr
    return adaptive_learning_rate(abs(error), prev_err, base_lr, **rprop_kw)


# ---------------------------------------------------------------------------
# Rprop adaptive learning rate
# ---------------------------------------------------------------------------


def adaptive_learning_rate(
    phi_err_rms: float,
    prev_phi_err_rms: Optional[float],
    prev_lr: float,
    lr_min: float = 1e-4,
    lr_max: float = 0.8,
    decay: float = 0.7,
    grow: float = 1.05,
    phi_scale: float = np.pi,
) -> float:
    """Rprop-style adaptive LR with magnitude-based ceiling."""
    if prev_phi_err_rms is None:
        lr_trend = prev_lr
    elif phi_err_rms > prev_phi_err_rms:
        lr_trend = prev_lr * decay
    else:
        lr_trend = prev_lr * grow

    lr_magnitude = lr_max * min(phi_err_rms / phi_scale, 1.0)
    return float(np.clip(min(lr_trend, lr_magnitude), lr_min, lr_max))


# ---------------------------------------------------------------------------
# PS crosstalk decoupling
# ---------------------------------------------------------------------------


def load_ps_crosstalk_matrix(csv_path: str) -> tuple[np.ndarray, list[int]]:
    """
    Load PS thermal crosstalk matrix from CSV.

    CSV format:
        tap\\swept_ps,9,10,...,16
        9, C_99, C_9_10, ...
        ...
    Rows = observed tap, columns = swept PS.
    C[i,j] = dφ_i/dP_j  (rad/W — must match power units in the control loop).

    Returns
    -------
    C : ndarray, shape (n, n)
    tap_order : list[int]
    """
    import pandas as pd

    df = pd.read_csv(csv_path, index_col=0)
    tap_order = [int(c) for c in df.columns]
    return df.values.astype(float), tap_order


def decouple_ps_delta_power(
    ps_phase_errors: dict[int, float],
    crosstalk_matrix: np.ndarray,
    tap_order: list[int],
    learning_rate: float,
) -> dict[int, float]:
    """
    Solve for decoupled PS power corrections via Δφ = C·ΔP → ΔP = lr·C⁻¹·Δφ.

    Uses lstsq for numerical stability.  Absent taps treated as zero error.
    """
    phi_err = np.array([ps_phase_errors.get(t, 0.0) for t in tap_order])
    delta_p, *_ = np.linalg.lstsq(crosstalk_matrix, phi_err * learning_rate, rcond=None)
    return {t: float(dp) for t, dp in zip(tap_order, delta_p)}


# ---------------------------------------------------------------------------
# Main power-adjustment function
# ---------------------------------------------------------------------------


def calculate_power_adjustments(
    mzi_phase_errors: Dict[str, float],
    ps_phase_errors: Dict[int, float],
    mzi_psr_errors: Dict[str, float],
    prev_mzi_psr_errors: Optional[Dict[str, float]],
    prev_ps_phase_errors: Optional[Dict[int, float]],
    current_mzi_powers: Dict[str, float],
    current_ps_powers: Dict[int, float],
    power_for_mzi_2pi: float,
    power_for_ps_2pi: float,
    mzi_learning_rate: float,
    ps_learning_rate: float,
    min_power: float,
    max_power: float,
    wrap_phase: bool = False,
    **kwargs,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate new heater powers for MZIs and phase shifters.

    φ_init flipping is handled upstream in calibration_loop.py.

    kwargs
    ------
    mzi_dead_zone_db : float        Dead zone threshold for MZI (PSR dB). Default 0.1.
    ps_dead_zone_rad : float        Dead zone threshold for PS (φ rad).   Default 0.0.
    mzi_adaptive_learning : bool    Enable Rprop LR for MZIs.  Default False.
    ps_adaptive_learning  : bool    Enable Rprop LR for PSs.   Default False.
    lr_min, lr_max, decay, grow, phi_scale : Rprop hyperparameters.
    ps_crosstalk_matrix : ndarray   Crosstalk matrix C (rad/W). Optional.
    ps_crosstalk_tap_order : list   Tap order matching C rows/cols. Optional.
    """

    rprop_kw = {
        k: kwargs.get(k, d)
        for k, d in [
            ("lr_min", 1e-4),
            ("lr_max", 0.8),
            ("decay", 0.7),
            ("grow", 1.05),
            ("phi_scale", np.pi),
        ]
    }
    mzi_dead_zone_db = kwargs.get("mzi_dead_zone_db", 0.1)
    ps_dead_zone_rad = kwargs.get("ps_dead_zone_rad", 0.0)
    mzi_adaptive = kwargs.get("mzi_adaptive_learning", False)
    ps_adaptive = kwargs.get("ps_adaptive_learning", False)

    ps_crosstalk_matrix = kwargs.get("ps_crosstalk_matrix", None)
    ps_crosstalk_tap_order = kwargs.get("ps_crosstalk_tap_order", None)

    probe_mode = kwargs.get("probe_mode", False)
    ps_probe_threshold_rad = kwargs.get("ps_probe_threshold_rad", np.pi / 2)
    ps_phi_init = kwargs.get("ps_phi_init", {})
    ps_measured_phases = kwargs.get("ps_measured_phases", None)

    new_mzi_powers: Dict[str, float] = {}
    new_ps_powers: Dict[str, float] = {}

    # -----------------------------------------------------------------------
    # MZIs — gate on PSR error (dB), step on φ_err (rad)
    # -----------------------------------------------------------------------
    for mzi_id, phi_err in mzi_phase_errors.items():
        psr_err = mzi_psr_errors.get(mzi_id, 0.0)
        prev_psr = prev_mzi_psr_errors.get(mzi_id) if prev_mzi_psr_errors else None

        lr = _effective_lr(
            psr_err,
            prev_psr,
            mzi_learning_rate,
            mzi_adaptive,
            **rprop_kw,
        )

        new_P, delta_P = _compute_new_power(
            phi_err,
            current_mzi_powers.get(mzi_id, 0.0),
            lr,
            power_for_mzi_2pi,
            min_power,
            max_power,
            dead_zone=mzi_dead_zone_db,
            gate_err=psr_err,
        )
        new_mzi_powers[mzi_id] = new_P
        logger.info(
            f"  MZI {mzi_id}: φ_err={phi_err:.4f} rad, PSR_err={psr_err:.3f} dB, "
            f"lr={lr:.4f}, ΔP={delta_P:.4f} W → P={new_P:.4f} W"
        )

    # -----------------------------------------------------------------------
    # Phase shifters — crosstalk-decoupled or per-tap
    # -----------------------------------------------------------------------
    if ps_crosstalk_matrix is not None and ps_crosstalk_tap_order is not None:
        decoupled = decouple_ps_delta_power(
            ps_phase_errors,
            ps_crosstalk_matrix,
            ps_crosstalk_tap_order,
            ps_learning_rate,
        )
        for tap_num, delta_P in decoupled.items():
            current_P = current_ps_powers.get(tap_num, 0.0)
            new_P = current_P + delta_P  # ← apply ΔP directly

            # Wrap and clip
            if wrap_phase:
                if new_P < 0:
                    new_P += power_for_ps_2pi
                elif new_P > 1.25 * power_for_ps_2pi:
                    new_P -= power_for_ps_2pi

            new_P = float(np.clip(new_P, min_power, max_power))
            new_ps_powers[tap_num] = new_P
            logger.info(
                f"  PS {tap_num}: ΔP={delta_P:.4f} W (decoupled) → P={new_P:.4f} W"
            )

    else:
        for tap_num, phi_err in ps_phase_errors.items():
            if wrap_phase:
                phi_err = float(np.angle(np.exp(1j * phi_err)))

            # --- Probe branch ---
            if probe_mode and abs(phi_err) > ps_probe_threshold_rad:
                phi_init = ps_phi_init.get(tap_num, 0.0)
                phi_measured = (
                    float(np.angle(ps_measured_phases[tap_num]))
                    if ps_measured_phases is not None
                    else 0.0
                )
                probe_target = phi_init + np.sign(phi_err) * ps_probe_threshold_rad
                probe_err = phi_measured - probe_target
                new_P, delta_P = _compute_new_power(
                    probe_err,
                    current_ps_powers.get(tap_num, 0.0),
                    ps_learning_rate,
                    power_for_ps_2pi,
                    min_power,
                    max_power,
                    wrap=wrap_phase,
                    dead_zone=ps_dead_zone_rad,
                )
                new_ps_powers[tap_num] = new_P
                logger.warning(
                    f"  PS {tap_num}: PROBE MODE triggered |φ_err|={abs(phi_err):.4f} > "
                    f"{ps_probe_threshold_rad:.4f} rad → probe_target={probe_target:+.4f} rad, "
                    f"ΔP={delta_P:.4f} W → P={new_P:.4f} W"
                )
                continue  # skip normal update

            prev_err = (
                abs(prev_ps_phase_errors[tap_num])
                if prev_ps_phase_errors and tap_num in prev_ps_phase_errors
                else None
            )
            lr = _effective_lr(
                phi_err, prev_err, ps_learning_rate, ps_adaptive, **rprop_kw
            )

            new_P, delta_P = _compute_new_power(
                phi_err,
                current_ps_powers.get(tap_num, 0.0),
                lr,
                power_for_ps_2pi,
                min_power,
                max_power,
                wrap=wrap_phase,
                dead_zone=ps_dead_zone_rad,
            )
            new_ps_powers[tap_num] = new_P
            logger.info(
                f"  PS {tap_num}: φ_err={phi_err:.4f} rad, lr={lr:.4f}, "
                f"ΔP={delta_P:.4f} W → P={new_P:.4f} W"
            )

    return new_mzi_powers, new_ps_powers


# ---------------------------------------------------------------------------
# Hardware helpers (unchanged)
# ---------------------------------------------------------------------------


def apply_voltages_to_hardware(
    chip_state: ChipState,
    config: ExperimentConfig,
    voltage_ctrl: VoltageController,
) -> None:
    """Apply all MZI and PS voltages to hardware in a single batch call."""
    R = config.chip.heater_resistance_ohm
    channels, voltages = [], []

    for mzi_id, mzi in chip_state.mzis.items():
        voltage = np.sqrt(mzi.applied_power_watts * R)
        channel = config.channel_mapping.get_channel(f"MZI_{mzi_id}")
        channels.append(channel)
        voltages.append(voltage)
        logger.info(
            f"    MZI {mzi_id} (ch {channel}): {voltage:.4f} V ({mzi.applied_power_watts:.4f} W)"
        )

    for tap_num, ps in chip_state.phase_shifters.items():
        voltage = np.sqrt(ps.applied_power_watts * R)
        channel = config.channel_mapping.get_channel(f"PS_{tap_num}")
        channels.append(channel)
        voltages.append(voltage)
        logger.info(
            f"    PS {tap_num} (ch {channel}): {voltage:.4f} V ({ps.applied_power_watts:.4f} W)"
        )

    logger.info("\n  Applying voltages to hardware:")
    voltage_ctrl.set_voltages(
        channels=channels,
        voltages=voltages,
        v_max=config.measurement.voltage_controller_v_max,
    )


def set_mzi_voltage(
    mzi_id: str,
    voltage: float,
    exp_config: ExperimentConfig,
    settling_time_sec: float = 2.0,
    v_max: float = 30.0,
) -> None:
    """Set voltage on a single MZI and wait for thermal settling."""
    mzi_channel = exp_config.channel_mapping.get_channel(f"MZI_{mzi_id}")
    logger.info(f"Setting MZI {mzi_id} (channel {mzi_channel}) to {voltage:.2f} V")

    with VoltageController(
        com_port=exp_config.measurement.voltage_controller_port,
        baud_rate=exp_config.measurement.voltage_controller_baudrate,
        zero_on_exit=False,
    ) as v_ctrl:
        v_ctrl.set_voltages([mzi_channel], [voltage], v_max=v_max)
        logger.info("✓ Voltage applied")

        if settling_time_sec > 0:
            logger.info(f"Waiting {settling_time_sec} s for thermal settling...")
            time.sleep(settling_time_sec)
            logger.info("✓ Settled")
