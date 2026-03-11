"""
Calibration Module
==================

High-level calibration workflows and experiment orchestration.

This module provides the main calibration loop that coordinates:
    1. Spectral measurements
    2. Tap coefficient recovery
    3. Error calculation
    4. Power adjustment
    5. Convergence checking

Submodules:
    calibration_loop: Main calibration iteration and experiment runner
"""

# Note: calibration_loop.py would contain refactored code from main_expt.py
# For now, this is a placeholder until you create that module
from .measurement_pipeline import measure_and_detect_taps

from .calibration_loop import (
    run_calibration_iteration,
    check_convergence,
    run_experiment,
    save_results,
    load_config,
)

from .trim_spectrum_to_fsr import trim_spectrum_to_fsr

__all__ = [
    "run_calibration_iteration",
    "check_convergence",
    "run_experiment",
    "save_results",
    "load_config",
    "measure_and_detect_taps",
    "trim_spectrum_to_fsr",
]
