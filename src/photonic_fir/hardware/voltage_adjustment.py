from typing import Dict, Optional, Tuple
import numpy as np

from ..core.data_structure import (
    ChipState,
    MZIState,
    PhaseShifterState,
    ExperimentConfig,
)

from voltage_ctrl import VoltageController


def calculate_power_adjustments(
    mzi_phase_errors: Dict[str, float],
    ps_phase_errors: Dict[int, float],
    mzi_psr_errors: Dict[str, float],
    prev_mzi_psr_errors: Optional[Dict[str, float]],
    current_mzi_powers: Dict[str, float],
    current_ps_powers: Dict[int, float],
    mzi_phi_init: Dict[str, float],
    ps_phi_init: Dict[int, float],
    power_for_2pi: float,
    learning_rate: float,
    min_power: float,
    max_power: float,
    psr_increase_threshold_db: float = 0.2,
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
                print(
                    f"  MZI {mzi_id}: PSR error increased "
                    f"({prev_err:.2f} → {curr_err:.2f} dB), adding π to φ_init"
                )

        # Calculate power adjustment: ΔP = (φ_err / 2π) × P_2π × LR
        delta_P = (
            ((phi_err - mzi_phi_init.get(mzi_id, 0.0)) / (2 * np.pi))
            * power_for_2pi
            * learning_rate
        )

        # Get current power
        current_P = current_mzi_powers.get(mzi_id, 0.0)
        new_P = current_P + delta_P

        # Rule (b): Handle negative power (2π phase wrapping)
        if new_P < 0:
            new_P = new_P + power_for_2pi
            print(
                f"  MZI {mzi_id}: Wrapped negative power "
                f"({current_P + delta_P:.4f} → {new_P:.4f} W)"
            )

        # Clamp to power limits
        new_P = np.clip(new_P, min_power, max_power)

        new_mzi_powers[mzi_id] = new_P

    # Process each phase shifter
    new_ps_powers = {}
    for tap_num, phi_err in ps_phase_errors.items():
        # Calculate power adjustment
        delta_P = (
            ((phi_err - ps_phi_init.get(tap_num, 0.0)) / (2 * np.pi))
            * power_for_2pi
            * learning_rate
        )

        # Get current power
        current_P = current_ps_powers.get(tap_num, 0.0)
        new_P = current_P + delta_P

        # Handle negative power
        if new_P < 0:
            new_P = new_P + power_for_2pi
            print(
                f"  PS {tap_num}: Wrapped negative power "
                f"({current_P + delta_P:.4f} → {new_P:.4f} W)"
            )

        # Clamp to power limits
        new_P = np.clip(new_P, min_power, max_power)

        new_ps_powers[tap_num] = new_P

    return new_mzi_powers, new_ps_powers, phi_init_adjustments


def apply_voltages_to_hardware(chip_state: ChipState, config: ExperimentConfig):
    """
    Apply voltages to hardware based on current chip state.

    This is separate from update_powers() to allow for:
    1. Simulation mode (skip hardware updates)
    2. Batch voltage updates (more efficient)
    3. Hardware error handling
    """

    voltage_ctrl = VoltageController(port=config.measurement.voltage_controller_port)
    R = config.chip.heater_resistance_ohms

    print("\n  Applying voltages to hardware:")

    # Apply MZI voltages
    for mzi_id, mzi in chip_state.mzis.items():
        voltage = np.sqrt(mzi.applied_power_watts * R)
        channel = chip_state.get_device_channel(f"MZI_{mzi_id}")
        voltage_ctrl.set_voltage(channel=channel, voltage=voltage)
        print(
            f"    MZI {mzi_id} (ch {channel}): {voltage:.4f} V ({mzi.applied_power_watts:.4f} W)"
        )

    # Apply phase shifter voltages
    for tap_num, ps in chip_state.phase_shifters.items():
        voltage = np.sqrt(ps.applied_power_watts * R)
        channel = chip_state.get_device_channel(f"PS_{tap_num}")
        voltage_ctrl.set_voltage(channel=channel, voltage=voltage)
        print(
            f"    PS {tap_num} (ch {channel}): {voltage:.4f} V ({ps.applied_power_watts:.4f} W)"
        )
