import sys
import numpy as np

sys.path.append("../..")
from simulation.power_splitting_ratio import PowerSplittingCalculator


def calculate_psr_error(measured_taps: np.array, target_psr: dict) -> dict:
    """
    Calculate the power splitting ratio error based on measured tap coefficients.
    Args:
        measured_taps (np.array): Measured tap coefficients.
        target_psr (dict): Target power splitting ratios.
    Returns:
        dict: Power splitting ratio errors.
    """

    psr_calculator = PowerSplittingCalculator()
    measured_psr = psr_calculator.tap_coeffs_to_power_splitting_ratios(measured_taps)

    psr_error = {}
    for mzi, target_ratio in target_psr.items():
        measured_ratio = measured_psr.get(mzi, 0)
        error = target_ratio - measured_ratio
        psr_error[mzi] = error

    return psr_error
