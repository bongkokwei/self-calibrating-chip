"""
Configuration Package
====================

This package contains configuration modules for the photonic FIR chip simulation
and experimental setup.

Modules
-------
data_structure : Configuration data structures
    Contains dataclass definitions for chip configuration, MZI parameters,
    phase shifters, and measurement settings.
"""

from .data_structure import (
    ChipParameters,
    MZIState,
    PhaseShifterState,
    ChipState,
    MeasurementConfig,
    CalibrationConfig,
    TargetFilter,
    ExperimentConfig,
    IterationData,
    CalibrationResults,
    config_from_dict,
    config_to_dict,
)

from .measure_spectrum import (
    measure_spectrum,
    measure_spectrum_with_config,
)
from .recover_tap_coefficients import (
    recover_impulse_response,
    recover_impulse_response_from_df,
    detect_taps,
)

from .error_calculation import (
    calculate_mzi_errors,
    calculate_phase_shifter_errors,
    calculate_rms_errors,
    calculate_all_errors,
)

from power_splitting_ratio import (
    build_mzi_tree_structure,
    tap_coeffs_to_power_splitting_ratios,
    extract_tap_phases,
    power_splitting_ratio_to_mzi_phase,
    mzi_phase_to_power_splitting_ratio,
    power_splitting_ratios_to_mzi_phases,
)


__all__ = [
    # Data structures
    "ChipParameters",
    "MZIState",
    "PhaseShifterState",
    "ChipState",
    "MeasurementConfig",
    "CalibrationConfig",
    "TargetFilter",
    "ExperimentConfig",
    "IterationData",
    "CalibrationResults",
    "config_from_dict",
    "config_to_dict",
    # Helper functions for YAML compatibility
    "measure_spectrum",
    "measure_spectrum_with_config",
    "recover_impulse_response",
    "recover_impulse_response_from_df",
    "detect_taps",
    "calculate_mzi_errors",
    "calculate_phase_shifter_errors",
    "calculate_rms_errors",
    "calculate_all_errors",
    "build_mzi_tree_structure",
    "tap_coeffs_to_power_splitting_ratios",
    "extract_tap_phases",
    "power_splitting_ratio_to_mzi_phase",
    "mzi_phase_to_power_splitting_ratio",
    "power_splitting_ratios_to_mzi_phases",
]
