"""
Photonic FIR Filter Calibration Package
========================================

A Python package for self-calibrating programmable photonic integrated circuits
based on Xu et al. (2022) Nature Photonics research.

Main modules:
    core: Data structures and core algorithms
    hardware: Hardware control interfaces
    processing: Signal processing and tap recovery
    calibration: High-level calibration workflows
"""

__version__ = "0.1.0"
__author__ = "Kok-Wei"

# Import commonly-used items at package level for convenience
from .core.data_structure import (
    ChipParameters,
    ChipState,
    MZIState,
    PhaseShifterState,
    MeasurementConfig,
    CalibrationConfig,
    TargetFilter,
    ExperimentConfig,
    IterationData,
    CalibrationResults,
    VoltageChannelMapping,
    config_from_dict,
    config_to_dict,
)

from .core.power_splitting_ratio import (
    tap_coeffs_to_power_splitting_ratios,
    extract_tap_phases,
    power_splitting_ratio_to_mzi_phase,
    mzi_phase_to_power_splitting_ratio,
    power_splitting_ratios_to_mzi_phases,
)

from .core.error_calculation import (
    calculate_all_errors,
    calculate_mzi_errors,
    calculate_phase_shifter_errors,
    calculate_rms_errors,
)

from .core.config import (
    load_config,
    save_config,
    convert_numpy_types,
)

from .processing.tap_recovery import (
    recover_impulse_response,
    recover_impulse_response_from_df,
    detect_taps,
    kramers_kronig_phase_recovery,
)

from .processing.tap_recovery_robust import detect_taps_noise_tolerant

from .utils.file_utils import get_next_run_dir, ensure_dir
from .utils.plotting_utils import plot_insertion_loss, plot_impulse_response
from .utils.calibration_plotting import (
    CalibrationPlotter,
    plot_calibration_errors,
)

# Hardware imports are kept separate to avoid import errors
# if hardware libraries aren't installed
try:
    from .hardware.measurement import (
        measure_spectrum,
    )
    from .hardware.voltage_adjustment import (
        calculate_power_adjustments,
        apply_voltages_to_hardware,
        set_mzi_voltage,
    )

    _HARDWARE_AVAILABLE = True
except ImportError:
    _HARDWARE_AVAILABLE = False

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Config
    "load_config",
    "save_config",
    "convert_numpy_types",
    # Data structures
    "ChipParameters",
    "ChipState",
    "MZIState",
    "PhaseShifterState",
    "MeasurementConfig",
    "CalibrationConfig",
    "TargetFilter",
    "ExperimentConfig",
    "IterationData",
    "CalibrationResults",
    "VoltageChannelMapping",
    "config_from_dict",
    "config_to_dict",
    # Power splitting ratio calculations
    "tap_coeffs_to_power_splitting_ratios",
    "extract_tap_phases",
    "power_splitting_ratio_to_mzi_phase",
    "mzi_phase_to_power_splitting_ratio",
    "power_splitting_ratios_to_mzi_phases",
    # Error calculations
    "calculate_all_errors",
    "calculate_mzi_errors",
    "calculate_phase_shifter_errors",
    "calculate_rms_errors",
    # Signal processing
    "recover_impulse_response",
    "recover_impulse_response_from_df",
    "detect_taps",
    "kramers_kronig_phase_recovery",
    "detect_taps_noise_tolerant",
    # Plotting utilities
    "plot_insertion_loss",
    "plot_impulse_response",
    # Calibration plotting
    "CalibrationPlotter",
    "plot_calibration_errors",
    # File utilities
    "get_next_run_dir",
    "ensure_dir",
]

# Add hardware functions to __all__ if available
if _HARDWARE_AVAILABLE:
    __all__.extend(
        [
            "measure_spectrum",
            "measure_spectrum_with_config",
            "calculate_power_adjustments",
            "apply_voltages_to_hardware",
            "set_mzi_voltage",
        ]
    )

import logging
import sys


def setup_logging(log_file="app.log", level="INFO"):
    """
    Basic logging setup - console + file

    Args:
        log_file: Path to log file (default: 'app.log')
        level: Logging level - 'DEBUG', 'INFO', 'WARNING', 'ERROR' (default: 'INFO')
    """
    # Create handlers with explicit UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding="utf-8")

    # Force UTF-8 for console on Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.stream = open(
        sys.stdout.fileno(),
        mode="w",
        encoding="utf-8",
        buffering=1,
    )

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler, console_handler],
    )
