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
    # Core chip configuration
    ChipConfig,
    # Component configurations
    MZIConfig,
    PhaseShifterConfig,
    DelayLineConfig,
    TapConfig,
    # Measurement and calibration
    MeasurementConfig,
    CalibrationConfig,
    # Results storage
    CalibrationResults,
    MeasurementResults,
    # Helper functions
    measure_spectrum,
    measure_spectrum_with_config,
    recover_tap_coefficients,
    recover_tap_coefficients_from_dataframe,
    detect_taps,
)

__all__ = [
    # Core configuration
    "ChipConfig",
    # Component configurations
    "MZIConfig",
    "PhaseShifterConfig",
    "DelayLineConfig",
    "TapConfig",
    # Measurement and calibration
    "MeasurementConfig",
    "CalibrationConfig",
    # Results storage
    "CalibrationResults",
    "MeasurementResults",
]
