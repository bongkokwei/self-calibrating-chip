"""
voltage_adjustment.py
=====================

Power-adjustment helpers for the MZI and PS calibration loops.

Both actuators share the same update skeleton via compute_new_power:

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

logger = logging.getLogger(__name__)

from ..core.data_structure import (
    CalibrationConfig,
    ChipState,
    ExperimentConfig,
)
from .power_math import compute_new_power, wrap_and_clip_power
from .probe_mode import apply_probe_correction
from voltage_ctrl import VoltageController


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
    chip_state: ChipState,
    mzi_phase_errors: Dict[str, float],
    ps_phase_errors: Dict[int, float],
    mzi_psr_errors: Dict[str, float],
    prev_mzi_psr_errors: Optional[Dict[str, float]],
    prev_ps_phase_errors: Optional[Dict[int, float]],
    current_mzi_powers: Dict[str, float],
    current_ps_powers: Dict[int, float],
    power_for_mzi_2pi: float,
    power_for_ps_2pi: float,
    calibration_config: CalibrationConfig,
    ps_phi_init: Optional[Dict[int, float]] = None,
    ps_crosstalk_matrix: Optional[np.ndarray] = None,
    ps_crosstalk_tap_order: Optional[list] = None,
    ps_measured_phases: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate new heater powers for MZIs and phase shifters.

    φ_init flipping is handled upstream in calibration_loop.py.

    All algorithm hyperparameters (learning rates, dead zones, Rprop
    settings, probe mode) are read from calibration_config — see
    CalibrationConfig for defaults and docs.
    """
    ps_phi_init = ps_phi_init or {}

    mzi_learning_rate = calibration_config.mzi_learning_rate
    ps_learning_rate = calibration_config.ps_learning_rate
    min_power = calibration_config.min_power_watts
    max_power = calibration_config.max_power_watts
    wrap_phase = calibration_config.wrap_phase

    rprop_kw = {
        "lr_min": calibration_config.lr_min,
        "lr_max": calibration_config.lr_max,
        "decay": calibration_config.lr_decay,
        "grow": calibration_config.lr_grow,
        "phi_scale": calibration_config.lr_phi_scale,
    }
    mzi_dead_zone_db = calibration_config.mzi_dead_zone_db
    ps_dead_zone_rad = calibration_config.ps_dead_zone_rad
    mzi_adaptive = calibration_config.mzi_adaptive_learning
    ps_adaptive = calibration_config.ps_adaptive_learning

    probe_mode = calibration_config.probe_mode
    ps_probe_threshold_rad = calibration_config.ps_probe_threshold_rad

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

        new_P, delta_P = compute_new_power(
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

            new_P = wrap_and_clip_power(
                new_P, power_for_ps_2pi, min_power, max_power, wrap_phase
            )
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
                new_ps_powers[tap_num] = apply_probe_correction(
                    chip_state,
                    tap_num,
                    phi_err,
                    current_ps_powers.get(tap_num, 0.0),
                    ps_phi_init,
                    ps_measured_phases,
                    ps_probe_threshold_rad,
                    ps_learning_rate,
                    power_for_ps_2pi,
                    min_power,
                    max_power,
                    wrap_phase,
                    ps_dead_zone_rad,
                )
                continue  # skip normal update
            elif chip_state.phase_shifters[tap_num].target_probe_rad is not None:
                # Exiting probe-mode range: clear the accumulated probe target
                # so a future re-entry starts fresh instead of carrying a
                # stale offset forward.
                chip_state.phase_shifters[tap_num].target_probe_rad = None

            prev_err = (
                abs(prev_ps_phase_errors[tap_num])
                if prev_ps_phase_errors and tap_num in prev_ps_phase_errors
                else None
            )
            lr = _effective_lr(
                phi_err, prev_err, ps_learning_rate, ps_adaptive, **rprop_kw
            )

            new_P, delta_P = compute_new_power(
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


def voltage_range_uniform_v_squared(
    v_min: float, v_max: float, n_points: int
) -> np.ndarray:
    """
    Generate a voltage array with uniform spacing in V².

    Heater power (and thus phase shift) is proportional to V² for
    resistive heaters, so uniform V² spacing gives uniform power/phase
    steps across the sweep.
    """
    v_squared = np.linspace(v_min**2, v_max**2, n_points)
    return np.sqrt(v_squared)


def zero_all_heaters(
    exp_config: ExperimentConfig,
    n_channels: int = 32,
    v_max: float = 30.0,
) -> None:
    """Zero all heater channels — redundant safety measure at the end of a batch run."""
    with VoltageController(
        com_port=exp_config.measurement.voltage_controller_port,
        baud_rate=exp_config.measurement.voltage_controller_baudrate,
        zero_on_exit=True,
    ) as v_ctrl:
        v_ctrl.set_voltages(
            channels=np.arange(1, n_channels + 1).tolist(),
            voltages=[0.0] * n_channels,
            v_max=v_max,
        )


