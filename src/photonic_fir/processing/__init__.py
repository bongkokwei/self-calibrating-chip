"""
Processing Module
=================

Signal processing algorithms for extracting tap coefficients from measured
spectral data.

Key algorithms:
    - Kramers-Kronig phase recovery
    - Inverse Fourier transform
    - Peak detection for tap identification

Submodules:
    tap_recovery: Extract impulse response and detect taps
"""

from .tap_recovery import (
    # Core recovery functions
    kramers_kronig_phase_recovery,
    recover_impulse_response,
    recover_impulse_response_from_df,
    # Tap detection
    detect_taps,
    # Visualisation
    plot_impulse_response,
)

__all__ = [
    "kramers_kronig_phase_recovery",
    "recover_impulse_response",
    "recover_impulse_response_from_df",
    "detect_taps",
    "plot_impulse_response",
]
