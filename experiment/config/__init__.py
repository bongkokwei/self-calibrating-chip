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
    TapDetectionConfig,
    PhaseRecoveryConfig,
    CalibrationConfig,
    TargetFilter,
    ExperimentConfig,
    IterationData,
    CalibrationResults,
    config_from_dict,
    config_to_dict,
)

from measure_spectrum import (
    measure_spectrum,
    measure_spectrum_with_config,
)
from recover_tap_coefficients import (
    recover_tap_coefficients,
    recover_tap_coefficients_from_dataframe,
    detect_taps,
)

__all__ = [
    "ChipParameters",
    "MZIState",
    "PhaseShifterState",
    "ChipState",
    "MeasurementConfig",
    "TapDetectionConfig",
    "PhaseRecoveryConfig",
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
    "recover_tap_coefficients",
    "recover_tap_coefficients_from_dataframe",
    "detect_taps",
]
