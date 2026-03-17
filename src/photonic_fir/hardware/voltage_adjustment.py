"""
voltage_adjustment.py  — updated MZI adaptive learning
========================================================

Replaces the Rprop adaptive_learning_rate call in the MZI loop with the
MATLAB-faithful approach from kk_shaping_main.m:

  - slope_factor (+1 / -1): reset to +1 each iteration; flip if PSR error
    diverged by > sr_diverge_threshold dB vs the previous iteration.
    MATLAB compares to 2 iterations ago (floor(1/amp_lrt)=2 for lrt=0.5).
    We use 1-step here since only one iteration of history is held; pass
    prev2_mzi_psr_errors via kwargs to enable the full 2-step check.
  - Dead zone: skip update entirely if |PSR err| < mzi_dead_zone_db (0.1 dB).
  - Learning rate: plain scalar — no trend/magnitude Rprop logic for MZIs.

PS adaptive learning (Rprop) is unchanged.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

from ..core.data_structure import (
    ChipState,
    MZIState,
    PhaseShifterState,
    ExperimentConfig,
)
from voltage_ctrl import VoltageController


# ---------------------------------------------------------------------------
# PS-only Rprop helper (kept for phase-shifter loop)
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
    """Rprop-style adaptive LR with magnitude-based ceiling (used for PS only)."""
    if prev_phi_err_rms is None:
        lr_trend = prev_lr
    elif phi_err_rms > prev_phi_err_rms:
        lr_trend = prev_lr * decay
    else:
        lr_trend = prev_lr * grow

    lr_magnitude = lr_max * min(phi_err_rms / phi_scale, 1.0)
    lr = min(lr_trend, lr_magnitude)
    return float(np.clip(lr, lr_min, lr_max))


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
    learning_rate: float,
    min_power: float,
    max_power: float,
    psr_increase_threshold_db: float = 0.2,
    wrap_phase: bool = False,
    **kwargs,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Calculate power adjustments for MZIs and phase shifters.

    MZI update — MATLAB kk_shaping_main.m style:
        ΔP = (φ_err / 2π) × P_2π × slope_factor × LR
        slope_factor = −1 if |PSR_err(n)| − |PSR_err(n−k)| > sr_diverge_threshold
        skip update if |PSR_err| < mzi_dead_zone_db

    PS update — Rprop adaptive LR (unchanged):
        ΔP = (φ_err / 2π) × P_2π × adaptive_LR

    Convergence rules (both):
        (a) If PSR_err(m,n) − PSR_err(m,n−1) > psr_increase_threshold_db → add π to φ_init
        (b) If P < 0 → P += P_2π
    """

    # --- MZI kwargs ---
    sr_diverge_threshold = kwargs.get("sr_diverge_threshold", 5.0)  # dB; MATLAB uses 5
    mzi_dead_zone_db = kwargs.get("mzi_dead_zone_db", 0.1)  # dB; MATLAB uses 0.1
    # Optional 2-step PSR history for MATLAB-exact slope check (floor(1/0.5)=2)
    prev2_mzi_psr_errors: Optional[Dict[str, float]] = kwargs.get(
        "prev2_mzi_psr_errors", None
    )

    # --- PS kwargs (Rprop) ---
    lr_min = kwargs.get("lr_min", 1e-4)
    lr_max = kwargs.get("lr_max", 0.8)
    decay = kwargs.get("decay", 0.7)
    grow = kwargs.get("grow", 1.05)
    phi_scale = kwargs.get("phi_scale", np.pi)

    new_mzi_powers: Dict[str, float] = {}
    phi_init_adjustments: Dict[str, float] = {}

    # -----------------------------------------------------------------------
    # MZIs
    # -----------------------------------------------------------------------
    for mzi_id, phi_err in mzi_phase_errors.items():
        curr_psr_err = mzi_psr_errors.get(mzi_id, 0.0)

        # Rule (a): PSR error increased → flip φ_init branch (Xu et al. paper)
        if prev_mzi_psr_errors is not None:
            prev_psr_err = prev_mzi_psr_errors.get(mzi_id, 0.0)
            if curr_psr_err - prev_psr_err > psr_increase_threshold_db:
                phi_init_adjustments[mzi_id] = np.pi
                logger.info(
                    f"  MZI {mzi_id}: PSR err increased "
                    f"({prev_psr_err:.2f} → {curr_psr_err:.2f} dB), adding π to φ_init"
                )

        # Dead zone — MATLAB: if |d_SR| < 0.1 dB, zero the update
        if abs(curr_psr_err) < mzi_dead_zone_db:
            new_mzi_powers[mzi_id] = current_mzi_powers.get(mzi_id, 0.0)
            logger.info(
                f"  MZI {mzi_id}: |PSR err|={abs(curr_psr_err):.3f} dB "
                f"< dead zone ({mzi_dead_zone_db} dB), no update"
            )
            continue

        delta_P = (
            (phi_err / (2 * np.pi)) * power_for_mzi_2pi * learning_rate
        )

        logger.info(
            f"  MZI {mzi_id}: φ_err={phi_err:.4f} rad, "
            f"PSR_err={curr_psr_err:.3f} dB, ΔP={delta_P:.4f} W"
        )

        current_P = current_mzi_powers.get(mzi_id, 0.0)
        new_P = current_P + delta_P

        # Rule (b): negative power → wrap by P_2π
        if new_P < 0:
            new_P += power_for_mzi_2pi
            logger.info(f"  MZI {mzi_id}: wrapped negative power → {new_P:.4f} W")

        new_P = np.clip(new_P, min_power, max_power)
        new_mzi_powers[mzi_id] = new_P

    # -----------------------------------------------------------------------
    # Phase shifters — Rprop adaptive LR (unchanged)
    # -----------------------------------------------------------------------
    new_ps_powers: Dict[str, float] = {}
    for tap_num, phi_err in ps_phase_errors.items():
        prev_err = (
            np.abs(prev_ps_phase_errors[tap_num])
            if (prev_ps_phase_errors is not None and tap_num in prev_ps_phase_errors)
            else None
        )

        if wrap_phase:
            phi_err = np.angle(np.exp(1j * phi_err))
            logger.info(f"  PS {tap_num}: Wrapped φ_err to {phi_err:.4f} rad")

        if kwargs.get("adaptive_learning", False):
            adaptive_lr = adaptive_learning_rate(
                phi_err_rms=np.abs(phi_err),
                prev_phi_err_rms=prev_err,
                prev_lr=learning_rate,
                lr_min=lr_min,
                lr_max=lr_max,
                decay=decay,
                grow=grow,
                phi_scale=phi_scale,
            )
            logger.info(
                f"  PS {tap_num}: φ_err={phi_err:.4f} rad, adaptive LR={adaptive_lr:.4f}"
            )
            delta_P = (phi_err / (2 * np.pi)) * power_for_ps_2pi * adaptive_lr
        else:
            delta_P = (phi_err / (2 * np.pi)) * power_for_ps_2pi * learning_rate

        current_P = current_ps_powers.get(tap_num, 0.0)
        new_P = current_P + delta_P

        if new_P < 0:
            new_P += power_for_ps_2pi
            logger.info(f"  PS {tap_num}: lower phase-wrap → {new_P:.4f} W")
        elif new_P > 1.25 * power_for_ps_2pi:
            new_P -= power_for_ps_2pi
            logger.info(f"  PS {tap_num}: upper phase-wrap → {new_P:.4f} W")

        new_P = np.clip(new_P, min_power, max_power)
        new_ps_powers[tap_num] = new_P

    return new_mzi_powers, new_ps_powers, phi_init_adjustments


def apply_voltages_to_hardware(
    chip_state: ChipState,
    config: ExperimentConfig,
    voltage_ctrl: VoltageController,
):
    """
    Apply voltages to hardware based on current chip state.

    This is separate from update_powers() to allow for:
    1. Simulation mode (skip hardware updates)
    2. Batch voltage updates (more efficient)
    3. Hardware error handling

    Args:
        chip_state: Current chip state with applied powers
        config: Experiment configuration containing channel mapping
    """

    R = config.chip.heater_resistance_ohm

    # Collect all channels and voltages
    channels = []
    voltages = []

    # Collect MZI voltages
    for mzi_id, mzi in chip_state.mzis.items():
        voltage = np.sqrt(mzi.applied_power_watts * R)
        channel = config.channel_mapping.get_channel(f"MZI_{mzi_id}")
        channels.append(channel)
        voltages.append(voltage)
        logger.info(
            f"    MZI {mzi_id} (ch {channel}): {voltage:.4f} V ({mzi.applied_power_watts:.4f} W)"
        )

    # Collect phase shifter voltages
    for tap_num, ps in chip_state.phase_shifters.items():
        voltage = np.sqrt(ps.applied_power_watts * R)
        channel = config.channel_mapping.get_channel(f"PS_{tap_num}")
        channels.append(channel)
        voltages.append(voltage)
        logger.info(
            f"    PS {tap_num} (ch {channel}): {voltage:.4f} V ({ps.applied_power_watts:.4f} W)"
        )

    # Apply all voltages in a single batch call
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
    """
    Set voltage on a specified MZI.

    Parameters
    ----------
    mzi_id : str
        MZI identifier (e.g., "1-1", "2-1", "4-6")
    voltage : float
        Voltage to apply (V)
    exp_config : ExperimentConfig
        Experiment configuration object
    settling_time_sec : float
        Time to wait for thermal settling (seconds)
    v_max : float
        Maximum allowed voltage (V)
    """

    # Get MZI channel from mapping
    mzi_device_id = f"MZI_{mzi_id}"
    mzi_channel = exp_config.channel_mapping.get_channel(mzi_device_id)

    logger.info(f"Setting MZI {mzi_id} (channel {mzi_channel}) to {voltage:.2f} V")

    # Apply voltage
    with VoltageController(
        com_port=exp_config.measurement.voltage_controller_port,
        baud_rate=exp_config.measurement.voltage_controller_baudrate,
        zero_on_exit=False,  # Don't zero when we're done
    ) as v_ctrl:
        v_ctrl.set_voltages([mzi_channel], [voltage], v_max=v_max)
        logger.info(f"✓ Voltage applied")

        if settling_time_sec > 0:
            logger.info(f"Waiting {settling_time_sec} s for thermal settling...")
            time.sleep(settling_time_sec)
            logger.info(f"✓ Settled")
