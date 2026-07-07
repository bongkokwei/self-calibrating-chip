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

from .impulse_response import (
    # Core recovery functions
    kramers_kronig_phase_recovery,
    recover_impulse_response,
    recover_impulse_response_from_df,
)

from .tap_recovery_robust import (
    detect_taps_windowed,
    detect_taps,
)

__all__ = [
    "kramers_kronig_phase_recovery",
    "recover_impulse_response",
    "recover_impulse_response_from_df",
    "detect_taps",
    "detect_taps_windowed",
]
