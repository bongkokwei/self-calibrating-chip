import numpy as np

from psr_error import calculate_psr_error, calculate_phase_shifter_error


def delta_power(resistance: float, power_2pi: float, phase_error: float) -> float:
    """
    Calculate the change in power required for a given phase error.
    Args:
        resistance (float): Load resistance in ohms.
        power_2pi (float): Power required for a 2π phase shift.
        phase_error (float): Phase error in radians.
    Returns:
        float: Change in power required.
    """
    delta_power = (phase_error / (2 * np.pi)) * power_2pi
    return delta_power


def calculate_voltage_adjustments(
    mzi_error: dict,
    phase_shifter_error: dict,
    prev_psr_error: dict,
    current_psr_error: dict,
    resistance: float,
    power_2pi: float,
) -> dict:
    """
    Calculate voltage adjustments based on current and previous power splitting ratio errors.
    Args:
        mzi_error (dict): Current power splitting ratio errors.
        phase_shifter_error (dict): Current phase shifter errors.
        prev_psr_error (dict): Previous power splitting ratio errors.
        current_psr_error (dict): Current power splitting ratio errors.
        resistance (float): Load resistance in ohms.
    Returns:
        dict: Voltage adjustments for each MZI.
    """
    voltage_adjustments = {}
    delta_power_mzi = {}
    for mzi in mzi_error.keys():
        delta_power_mzi[mzi] = delta_power(
            resistance,
            power_2pi,
            mzi_error[mzi],
        )

    for phase in phase_shifter_error.keys():
        delta_power_mzi[phase] = delta_power(
            resistance,
            power_2pi,
            phase_shifter_error[phase],
        )

    # Rules to guarantee convergence

    return voltage_adjustments
