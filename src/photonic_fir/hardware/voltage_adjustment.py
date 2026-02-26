from typing import Dict, Optional, Tuple
import numpy as np
import time

import logging

logger = logging.getLogger(__name__)


from ..core.data_structure import (
    ChipState,
    MZIState,
    PhaseShifterState,
    ExperimentConfig,
)

from voltage_ctrl import VoltageController


def adaptive_learning_rate(
    phi_err_rms: float,
    prev_phi_err_rms: Optional[float],
    prev_lr: float,
    lr_min: float = 1e-4,
    lr_max: float = 0.8,
    decay: float = 0.7,
    grow: float = 1.05,
    phi_scale: float = np.pi,  # error at which LR reaches lr_max
) -> float:
    """Rprop-style adaptive LR with magnitude-based ceiling.

    - Trend logic: decay if error worsened, grow if improving.
    - Magnitude ceiling: cap LR proportional to |φ_err| / phi_scale,
      so small residual errors automatically take smaller steps.
    """
    # --- Trend component (Rprop) ---
    if prev_phi_err_rms is None:
        lr_trend = prev_lr
    elif phi_err_rms > prev_phi_err_rms:
        lr_trend = prev_lr * decay
    else:
        lr_trend = prev_lr * grow

    # --- Magnitude ceiling ---
    # Scales 0 → 0 at zero error, lr_max at phi_scale (π rad)
    lr_magnitude = lr_max * min(phi_err_rms / phi_scale, 1.0)

    # Take the more conservative of the two
    lr = min(lr_trend, lr_magnitude)

    return float(np.clip(lr, lr_min, lr_max))


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
    Calculate power adjustments for MZIs and phase shifters based on calibration errors.

    Implements the algorithm from Xu et al. (2022) supplement:
        ΔP = (φ_err / 2π) × P_2π × LR

    With convergence rules:
        (a) If PSR_err(m,n) - PSR_err(m,n-1) > threshold, add π to φ_init
        (b) If P < 0, add P_2π (handle 2π phase wrapping)

    Args:
        mzi_phase_errors: Phase errors for each MZI in radians, e.g. {"2-1": 0.3, "3-3": -0.1}
        ps_phase_errors: Phase errors for each phase shifter in radians, e.g. {9: 0.2, 10: -0.15}
        mzi_psr_errors: Current power splitting ratio errors in dB, e.g. {"2-1": 1.5}
        prev_mzi_psr_errors: Previous iteration's PSR errors (None for first iteration)
        current_mzi_powers: Current applied powers for MZIs in watts, e.g. {"2-1": 0.3}
        current_ps_powers: Current applied powers for phase shifters in watts, e.g. {9: 0.4}
        mzi_phi_init: Initial phase offsets for MZIs in radians, e.g. {"2-1": -1.2}
        ps_phi_init: Initial phase offsets for phase shifters in radians, e.g. {9: 0.5}
        power_for_2pi: Nominal power for 2π phase shift (watts), typically 0.75
        learning_rate: Learning rate for power updates, typically 0.5
        min_power: Minimum allowed power (watts), typically 0.0
        max_power: Maximum allowed power (watts), typically 1.0
        psr_increase_threshold_db: PSR error increase threshold for rule (a), default 0.2 dB

    Returns:
        Tuple of (new_mzi_powers, new_ps_powers, phi_init_adjustments):
            - new_mzi_powers: Updated MZI powers in watts, e.g. {"2-1": 0.35}
            - new_ps_powers: Updated phase shifter powers in watts, e.g. {9: 0.42}
            - phi_init_adjustments: Phase offset corrections applied, e.g. {"2-1": π}

    Example:
        >>> mzi_phase_errors = {"2-1": 0.3, "3-3": -0.2}
        >>> ps_phase_errors = {9: 0.1, 10: -0.15}
        >>> mzi_psr_errors = {"2-1": 1.2, "3-3": -0.5}
        >>> prev_psr_errors = {"2-1": 0.8, "3-3": -0.3}  # PSR error increased for 2-1
        >>> current_mzi_powers = {"2-1": 0.3, "3-3": 0.4}
        >>> current_ps_powers = {9: 0.35, 10: 0.40}
        >>> mzi_phi_init = {"2-1": 0.0, "3-3": 0.0}
        >>> ps_phi_init = {9: 0.0, 10: 0.0}
        >>>
        >>> new_mzi, new_ps, adjusts = calculate_power_adjustments(
        ...     mzi_phase_errors, ps_phase_errors, mzi_psr_errors, prev_psr_errors,
        ...     current_mzi_powers, current_ps_powers, mzi_phi_init, ps_phi_init,
        ...     power_for_2pi=0.75, learning_rate=0.5, min_power=0.0, max_power=1.0
        ... )
    """

    new_mzi_powers = {}
    phi_init_adjustments = {}

    # Process each MZI
    for mzi_id, phi_err in mzi_phase_errors.items():
        # Rule (a): Check if PSR error increased > threshold
        # Indicates we're on wrong side of symmetric MZI transfer function
        if prev_mzi_psr_errors is not None:
            prev_err = prev_mzi_psr_errors.get(mzi_id, 0.0)
            curr_err = mzi_psr_errors.get(mzi_id, 0.0)

            if curr_err - prev_err > psr_increase_threshold_db:
                # Add π to initial phase to flip to correct branch
                phi_init_adjustments[mzi_id] = np.pi
                logger.info(
                    f"  MZI {mzi_id}: PSR error increased "
                    f"({prev_err:.2f} → {curr_err:.2f} dB), adding π to φ_init"
                )

        # Calculate power adjustment: ΔP = (φ_err / 2π) × P_2π × LR
        delta_P = ((phi_err) / (2 * np.pi)) * power_for_mzi_2pi * learning_rate

        # Get current power
        current_P = current_mzi_powers.get(mzi_id, 0.0)
        new_P = current_P + delta_P

        # Rule (b): Handle negative power (2π phase wrapping)
        if new_P < 0:
            new_P = new_P + power_for_mzi_2pi
            logger.info(
                f"  MZI {mzi_id}: Wrapped negative power "
                f"({current_P + delta_P:.4f} → {new_P:.4f} W)"
            )

        # Clamp to power limits
        new_P = np.clip(new_P, min_power, max_power)

        new_mzi_powers[mzi_id] = new_P

    # Process each phase shifter
    new_ps_powers = {}
    for tap_num, phi_err in ps_phase_errors.items():
        # For PS, we can also apply adaptive learning rate based on error trend
        prev_err = (
            np.abs(prev_ps_phase_errors[tap_num])
            if (prev_ps_phase_errors is not None and tap_num in prev_ps_phase_errors)
            else None
        )

        lr_min = kwargs.get("lr_min", 1e-4)
        lr_max = kwargs.get("lr_max", 0.8)
        decay = kwargs.get("decay", 0.7)
        grow = kwargs.get("grow", 1.05)
        phi_scale = kwargs.get("phi_scale", np.pi)
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
        # Calculate power adjustment
        if wrap_phase:
            phi_err = np.angle(np.exp(1j * phi_err))  # Wrap to [-π, π]

        delta_P = (
            ((phi_err) / (2 * np.pi)) * power_for_ps_2pi * adaptive_lr
        )  # Use smaller LR for PS to prevent overshooting

        # Get current power
        current_P = current_ps_powers.get(tap_num, 0.0)
        new_P = current_P + delta_P

        # Handle negative power
        if new_P < 0:
            new_P = new_P + power_for_ps_2pi
            logger.info(f"  PS {tap_num}: lower phase-wrap → {new_P:.4f} W")
        elif new_P > power_for_ps_2pi:
            new_P = new_P - power_for_ps_2pi
            logger.info(f"  PS {tap_num}: upper phase-wrap → {new_P:.4f} W")

        new_P %= power_for_ps_2pi  # Ensure within 0 to P_2π range
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
