import sys
import numpy as np

sys.path.append("../..")
from simulation.power_splitting_ratio import PowerSplittingCalculator


def calculate_psr_error(
    measured_taps: np.array,
    target_psc: PowerSplittingCalculator,
) -> dict:
    """
    Calculate the power splitting ratio error based on measured tap coefficients.
    Args:
        measured_taps (np.array): Measured tap coefficients.
        target_psr (PowerSplittingCalculator): Target power splitting ratios.
    Returns:
        dict: Power splitting ratio errors.
    """

    psr_calculator = PowerSplittingCalculator()

    measured_psr = psr_calculator.tap_coeffs_to_power_splitting_ratios(measured_taps)
    target_psr = target_psc.power_splitting_ratios

    measured_mzi_phase = {
        name: psr_calculator.power_splitting_ratio_to_mzi_phase(ratio)
        for name, ratio in measured_psr.items()
    }
    target_mzi_phase = {
        name: psr_calculator.power_splitting_ratio_to_mzi_phase(ratio)
        for name, ratio in target_psr.items()
    }

    psr_error = {}
    mzi_phase_error = {}
    for mzi, target_ratio in target_psr.items():
        measured_ratio = measured_psr.get(mzi, 0)
        mzi_phase_error = target_mzi_phase.get(mzi, 0) - measured_mzi_phase.get(mzi, 0)
        error = target_ratio - measured_ratio
        psr_error[mzi] = error

    return psr_error, mzi_phase_error


def calculate_phase_shifter_error(
    measured_taps: np.array,
    target_tap: np.array,
) -> np.array:
    """
    Calculate the phase shifter error based on measured tap coefficients.
    Args:
        measured_taps (np.array): Measured tap coefficients.
        target_tap (np.array): Target tap coefficients.
    Returns:
        np.array: Phase shifter errors.
    """

    measured_phases = np.angle(measured_taps)
    target_phases = np.angle(target_tap)

    phase_shifter_error = target_phases - measured_phases

    return phase_shifter_error
