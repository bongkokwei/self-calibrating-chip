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
    """
    Recover phase response from insertion loss using Kramers-Kronig relationship.

    For a minimum-phase system:
    φ(ω) = -H[ln|H(ω)|]
    where H[·] is the Hilbert transform.

    Parameters
    ----------
    insertion_loss_db : np.ndarray
        Insertion loss in dB

    Returns
    -------
    phase_recovered_rad : np.ndarray
        Recovered phase in radians

    Notes
    -----
    The Kramers-Kronig relationship states that for a minimum-phase system:
    In terms of insertion loss:
    IL(dB) = 20*log10(|H(ω)|)
    ln|H(ω)| = IL(dB) * ln(10) / 20
    """
    # Convert IL (dB) to natural log of amplitude
    # IL = -20*log10|H| → ln|H| = -IL*ln(10)/20
    ln_amplitude = insertion_loss_db * np.log(10) / 20  # ← Added negative sign

    # Remove DC component for better Hilbert transform
    ln_amplitude_mean = np.mean(ln_amplitude)
    ln_amplitude_centered = ln_amplitude - ln_amplitude_mean

    # Apply Hilbert transform to get phase
    phase_recovered_rad = -np.imag(hilbert(ln_amplitude_centered))

    return phase_recovered_rad


def recover_impulse_response(
    freq_hz: np.ndarray,
    insertion_loss_db: np.ndarray,
    fsr_hz: float,
    zero_pad_factor: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract impulse response from measured insertion loss spectra.

    Implements the Kramers-Kronig phase recovery and inverse Fourier
    transform approach from Xu et al. (2022).

    Parameters
    ----------
    freq_hz : np.ndarray
        Frequency data in Hz
    insertion_loss_db : np.ndarray
        Insertion loss data in dB
    fsr_hz : float
        Free spectral range in Hz
    zero_pad_factor : int, optional
        Zero-padding factor for improved time resolution (default: 4)
        Set to 1 for no padding. Higher values give smoother plots.

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

    # Recover phase using Kramers-Kronig
    logger.info("Recovering phase using Kramers-Kronig...")
    phase_rad = kramers_kronig_phase_recovery(insertion_loss_uniform)

    # Convert to complex transfer function
    amplitude = 10 ** (insertion_loss_uniform / 20)
    H_complex = amplitude * np.exp(1j * phase_rad)

    # === Zero-pad for improved time resolution ===
    if zero_pad_factor > 1:
        n_padded = n_points * zero_pad_factor
        H_padded = np.zeros(n_padded, dtype=complex)
        H_padded[:n_points] = H_complex
        logger.info(
            f"Zero-padding: {n_points} → {n_padded} points ({zero_pad_factor}x)"
        )
    else:
        H_padded = H_complex
        n_padded = n_points

    # IFFT to time domain
    h_time = ifft(H_padded)
    h_time_shifted = fftshift(h_time)

    # Time axis
    time_s = np.fft.fftfreq(n_padded, d=df_hz)
    time_ps = fftshift(time_s) * 1e12

    logger.info(f"Time resolution: {np.mean(np.diff(time_ps)):.3f} ps")
    logger.info(f"Time span: {time_ps[0]:.1f} to {time_ps[-1]:.1f} ps")

    return time_ps, h_time_shifted


def recover_impulse_response_from_df(
    df: pd.DataFrame,
    fsr_hz: float,
    wavelength_col: str = "wl_axis",
    freq_col: str = "f_axis",
    insertion_loss_col: str = "IL",
    zero_pad_factor: int = 4,
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
        zero_pad_factor=zero_pad_factor,
    )


def detect_taps(
    time_ps: np.ndarray,
    h_time: np.ndarray,
    fsr_hz: float,
    delay_step_s: float,
    n_taps: Optional[int] = 16,
    prominence_factor_db: float = 3.0,
    min_distance_ps: Optional[float] = None,
    height_threshold_db: float = -40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect tap positions and coefficients from impulse response.

    Parameters
    ----------
    time_ps : np.ndarray
        Time axis in picoseconds
    h_time : np.ndarray
        Complex impulse response
    chip_params : ChipParameters
        Chip parameters (contains FSR)
    n_taps : int, optional
        Expected number of taps
    prominence_factor_db : float
        Prominence threshold in dB
    min_distance_ps : float, optional
        Minimum distance between peaks in picoseconds
    height_threshold_db : float
        Minimum height in dB relative to maximum
    use_db_scale : bool
        Use dB scale for peak detection

    Returns
    -------
    tap_times_ps : np.ndarray
        Time positions of detected taps
    tap_coefficients : np.ndarray
        Complex tap coefficients
    """
    # Calculate magnitude
    h_magnitude = np.abs(h_time)
    max_magnitude = np.max(h_magnitude)

    # Convert to dB scale if requested
    logger.info("\n=== Using dB scale for peak detection ===")
    h_magnitude_normalised = h_magnitude / max_magnitude
    h_magnitude_db = 20 * np.log10(h_magnitude_normalised + 1e-12)
    detection_signal = h_magnitude_db
    height_min = height_threshold_db
    prominence_min = prominence_factor_db

    logger.info(f"Maximum magnitude (linear): {max_magnitude:.6f}")
    logger.info(f"Height threshold: {height_threshold_db:.1f} dB")
    logger.info(f"Prominence threshold: {prominence_factor_db:.1f} dB")

    # Calculate time resolution
    dt_ps = np.mean(np.diff(time_ps))

    # Determine minimum distance between peaks
    if min_distance_ps is None:
        tap_spacing_s = delay_step_s
        tap_spacing_ps = tap_spacing_s * 1e12
        min_distance_ps = tap_spacing_ps * 0.8

        logger.info(f"\nFSR: {fsr_hz/1e9:.2f} GHz")
        logger.info(f"Expected tap spacing: {tap_spacing_ps:.3f} ps")
        logger.info(f"Using min_distance: {min_distance_ps:.3f} ps")

    # Convert to samples
    min_distance = max(int(min_distance_ps / dt_ps), 1)
    logger.info(
        f"Minimum distance: {min_distance} samples ({min_distance * dt_ps:.3f} ps)"
    )

    # Find peaks
    peak_indices, peak_properties = find_peaks(
        detection_signal,
        height=height_min,
        prominence=prominence_min,
        distance=min_distance,
    )

    logger.info(f"\nInitial peaks found: {len(peak_indices)}")

    # Filter to N largest if specified
    if n_taps is None:
        n_taps = n_taps

    if len(peak_indices) > n_taps:
        logger.info(f"Filtering to keep only the {n_taps} largest peaks...")
        peak_magnitudes = h_magnitude[peak_indices]
        top_n_indices = np.argsort(peak_magnitudes)[::-1][0:n_taps]
        peak_indices = peak_indices[top_n_indices]
        peak_indices = np.sort(peak_indices)

    # Extract tap coefficients
    tap_coefficients = h_time[peak_indices]
    tap_times_ps = time_ps[peak_indices]

    logger.info(f"\nFinal detected taps: {len(tap_coefficients)}")

    # Display tap information
    logger.info("\nTap Coefficients:")
    logger.info(
        f"{'Tap':<5} {'Time (ps)':<12} {'Magnitude':<12} {'dB':<10} "
        f"{'Phase (rad)':<12} {'Phase (deg)':<12}"
    )
    logger.info("-" * 75)

    for i, (t, coeff) in enumerate(zip(tap_times_ps, tap_coefficients)):
        mag = np.abs(coeff)
        mag_db = 20 * np.log10(mag / max_magnitude)
        phase_rad = np.angle(coeff)
        phase_deg = np.degrees(phase_rad)
        logger.info(
            f"{i+1:<5} {t:<12.3f} {mag:<12.6f} {mag_db:<10.2f} "
            f"{phase_rad:<12.4f} {phase_deg:<12.2f}"
        )

    return tap_times_ps, tap_coefficients
