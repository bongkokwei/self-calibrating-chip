"""
tap_extraction.py

Tap coefficient extraction from measured spectral data compatible with data_structure.py.
Uses Kramers-Kronig phase recovery and inverse Fourier transform.
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy.fft import ifft, fftshift
from scipy.signal import find_peaks, hilbert
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def kramers_kronig_phase_recovery(insertion_loss_db: np.ndarray) -> np.ndarray:
    ln_amplitude = insertion_loss_db * np.log(10) / 20
    phase_recovered_rad = -np.imag(hilbert(ln_amplitude))
    return phase_recovered_rad


def recover_impulse_response(
    freq_hz: np.ndarray,
    insertion_loss_db: np.ndarray,
    fsr_hz: float,
    n_tiles: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract impulse response from measured insertion loss spectra.

    Tiles the spectrum n_tiles times before KK phase recovery and IFFT
    to suppress edge artefacts, then extracts the central period.

    Parameters
    ----------
    freq_hz : np.ndarray
        Frequency data in Hz
    insertion_loss_db : np.ndarray
        Insertion loss data in dB
    fsr_hz : float
        Free spectral range in Hz
    n_tiles : int
        Number of periodic tiles for Hilbert transform (default: 20)

    Returns
    -------
    time_ps : np.ndarray
        Time axis in picoseconds
    h_time_shifted : np.ndarray
        Complex impulse response
    """
    # Sort by frequency
    sort_idx = np.argsort(freq_hz)
    freq_hz = freq_hz[sort_idx]
    insertion_loss_db = insertion_loss_db[sort_idx]

    logger.info(f"Frequency range: {freq_hz[0]/1e9:.2f} to {freq_hz[-1]/1e9:.2f} GHz")
    logger.info(f"Number of frequency points: {len(freq_hz)}")

    # === Interpolate to uniform frequency grid ===
    n_points = len(freq_hz)
    freq_uniform = np.linspace(freq_hz[0], freq_hz[-1], n_points)
    insertion_loss_uniform = np.interp(freq_uniform, freq_hz, insertion_loss_db)
    df_hz = freq_uniform[1] - freq_uniform[0]

    logger.info(f"Uniform df: {df_hz/1e6:.3f} MHz")

    # === Tile spectrum for periodic Hilbert transform ===
    il_tiled = np.tile(insertion_loss_uniform, n_tiles)

    logger.info(f"Recovering phase using Kramers-Kronig ({n_tiles}x tiled)...")
    phase_tiled = kramers_kronig_phase_recovery(il_tiled)

    # Build tiled complex transfer function
    amplitude_tiled = 10 ** (il_tiled / 20)
    H_tiled = amplitude_tiled * np.exp(1j * phase_tiled)

    # IFFT to time domain
    h_time = ifft(H_tiled)
    h_time_shifted = fftshift(h_time)

    # Time axis
    time_s = np.fft.fftfreq(len(H_tiled), d=df_hz)
    time_ps = fftshift(time_s) * 1e12

    logger.info(f"Time resolution: {np.mean(np.diff(time_ps)):.3f} ps")
    logger.info(f"Time span: {time_ps[0]:.1f} to {time_ps[-1]:.1f} ps")

    return time_ps, h_time_shifted


def recover_impulse_response_from_df(
    df: pd.DataFrame,
    fsr_hz: float,
    freq_col: str = "f_axis",
    insertion_loss_col: str = "IL",
    n_tiles: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract impulse response from measured insertion loss spectra (DataFrame interface).

    Convenience wrapper that extracts data from DataFrame and calls
    recover_tap_coefficients().

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing measurement data
    fsr_hz : float
        Free spectral range in Hz
    wavelength_col : str
        Column name for wavelength data (nm)
    freq_col : str
        Column name for frequency data (THz)
    insertion_loss_col : str
        Column name for insertion loss data (dB)

    Returns
    -------
    time_ps : np.ndarray
        Time axis in picoseconds
    h_time_shifted : np.ndarray
        Complex impulse response
    """
    # Extract data from DataFrame
    freq_thz = df[freq_col].values
    freq_hz = freq_thz * 1e12  # Convert THz to Hz
    insertion_loss_db = df[insertion_loss_col].values

    # Call core function
    return recover_impulse_response(
        freq_hz=freq_hz,
        insertion_loss_db=insertion_loss_db,
        fsr_hz=fsr_hz,
        n_tiles=n_tiles,
    )
