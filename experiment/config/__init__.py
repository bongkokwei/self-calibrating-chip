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
