"""
Core Module
===========

Contains fundamental data structures and algorithms for photonic FIR chip
calibration that don't depend on hardware.

Submodules:
    data_structure: Dataclass definitions for chip configuration
    power_splitting_ratio: PSR calculations for binary tree MZI networks
    error_calculation: Error metric calculations for calibration
"""

from .data_structure import (
    # Core data structures
    ChipParameters,
    MZIState,
    PhaseShifterState,
    ChipState,
    VoltageChannelMapping,
    # Configuration structures
    MeasurementConfig,
    CalibrationConfig,
    TargetFilter,
    ExperimentConfig,
    # Results structures
    IterationData,
    CalibrationResults,
    # Helper functions
    config_from_dict,
    config_to_dict,
)

from .power_splitting_ratio import (
    # Tree structure
    build_mzi_tree_structure,
    # Power splitting ratio calculations
    tap_coeffs_to_power_splitting_ratios,
    extract_tap_phases,
    # Conversion functions
    power_splitting_ratio_to_mzi_phase,
    mzi_phase_to_power_splitting_ratio,
    power_splitting_ratios_to_mzi_phases,
    # Utilities
    print_tree_structure,
    print_psr_summary,
    create_sinc_filter_taps,
)

from .error_calculation import (
    calculate_mzi_errors,
    calculate_phase_shifter_errors,
    calculate_rms_errors,
    calculate_all_errors,
    wrap_phase_error,
)

__all__ = [
    # Data structures
    "ChipParameters",
    "MZIState",
    "PhaseShifterState",
    "ChipState",
    "VoltageChannelMapping",
    "MeasurementConfig",
    "CalibrationConfig",
    "TargetFilter",
    "ExperimentConfig",
    "IterationData",
    "CalibrationResults",
    "config_from_dict",
    "config_to_dict",
    # Power splitting ratio
    "build_mzi_tree_structure",
    "tap_coeffs_to_power_splitting_ratios",
    "extract_tap_phases",
    "power_splitting_ratio_to_mzi_phase",
    "mzi_phase_to_power_splitting_ratio",
    "power_splitting_ratios_to_mzi_phases",
    "print_tree_structure",
    "print_psr_summary",
    "create_sinc_filter_taps",
    # Error calculation
    "calculate_mzi_errors",
    "calculate_phase_shifter_errors",
    "calculate_rms_errors",
    "calculate_all_errors",
    "wrap_phase_error",
]
