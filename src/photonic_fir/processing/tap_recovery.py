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


def kramers_kronig_with_hanning(
    wavelength_nm, insertion_loss_db, fsr_hz=160e9, bw_multi=3
):
    """
    KK recovery with Hanning window as per Xu lab note

    Parameters
    ----------
    wavelength_nm : array
        Wavelength axis
    insertion_loss_db : array
        Insertion loss in dB
    fsr_hz : float
        Free spectral range in Hz (160 GHz for 16-tap)
    bw_multi : int
        Window width multiplier (odd integer ≥ 1, recommend 3-5)
    """
    # Convert wavelength to frequency
    c = 3e8  # m/s
    freq_hz = c / (wavelength_nm * 1e-9)

    # Compute effective window width in samples
    freq_span = np.max(freq_hz) - np.min(freq_hz)
    samples_per_fsr = len(freq_hz) * (fsr_hz / freq_span)
    window_width = int(samples_per_fsr * bw_multi)

    # Ensure window doesn't exceed data length
    window_width = min(window_width, len(insertion_loss_db))

    # Convert to LINEAR insertion loss
    il_linear = 10 ** (-insertion_loss_db / 10)

    # Create Hanning window centred on data
    hanning_window = np.hanning(window_width)

    # Pad to match data length
    pad_left = (len(il_linear) - window_width) // 2
    pad_right = len(il_linear) - window_width - pad_left
    hanning_full = np.pad(
        hanning_window, (pad_left, pad_right), mode="constant", constant_values=0
    )

    # Apply window to linear IL
    il_linear_windowed = il_linear * hanning_full

    # Convert back to dB for KK (but keep windowed)
    il_db_windowed = -10 * np.log10(il_linear_windowed + 1e-12)

    # Standard KK recovery
    ln_amplitude = il_db_windowed * np.log(10) / 20
    ln_amplitude_centered = ln_amplitude - np.mean(ln_amplitude)
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
    phase_rad = kramers_kronig_with_hanning(insertion_loss_uniform, fsr_hz=fsr_hz)

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
    search_start_time_ps: float = -10.0,
    max_search_time_ps: Optional[float] = None,
    max_time_margin: float = 1.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect tap positions and coefficients from impulse response.

    Restricts search to physically plausible region to exclude acausal
    artefacts (t < 0) and late-time artefacts (t >> expected last tap).

    Parameters
    ----------
    time_ps : np.ndarray
        Time axis in picoseconds
    h_time : np.ndarray
        Complex impulse response
    fsr_hz : float
        Free spectral range in Hz
    delay_step_s : float
        Delay between taps in seconds
    n_taps : int, optional
        Expected number of taps
    prominence_factor_db : float
        Prominence threshold in dB above local baseline
    min_distance_ps : float, optional
        Minimum distance between peaks in picoseconds.
        If None, calculated as 0.8 × delay_step_s
    height_threshold_db : float
        Minimum height in dB relative to maximum peak
    search_start_time_ps : float
        Start time for peak search (default: -10 ps)
    max_search_time_ps : float, optional
        Maximum time for peak search in picoseconds.
        If None, auto-calculated as (n_taps - 1) × delay_step_s × max_time_margin
    max_time_margin : float
        Safety margin for auto-calculated max_search_time_ps (default: 1.3)

    Returns
    -------
    tap_times_ps : np.ndarray
        Time positions of detected taps
    tap_coefficients : np.ndarray
        Complex tap coefficients
    """
    # Auto-calculate maximum search time if not provided
    if max_search_time_ps is None and n_taps is not None:
        expected_last_tap_ps = (n_taps - 1) * delay_step_s * 1e12
        max_search_time_ps = expected_last_tap_ps * max_time_margin
        logger.info(
            f"Expected last tap: {expected_last_tap_ps:.1f} ps, "
            f"max search: {max_search_time_ps:.1f} ps"
        )

    # Create search window mask
    search_mask = time_ps >= search_start_time_ps
    if max_search_time_ps is not None:
        search_mask &= time_ps <= max_search_time_ps

    time_ps_search = time_ps[search_mask]
    h_time_search = h_time[search_mask]

    logger.info(
        f"\n=== Search window: {time_ps_search[0]:.1f} to {time_ps_search[-1]:.1f} ps ==="
    )
    logger.info(f"Points in search region: {len(time_ps_search)}/{len(time_ps)}")

    # Calculate magnitude and convert to dB
    h_magnitude = np.abs(h_time_search)
    max_magnitude = np.max(h_magnitude)
    h_magnitude_normalised = h_magnitude / max_magnitude
    h_magnitude_db = 20 * np.log10(h_magnitude_normalised + 1e-12)

    logger.info(f"\n=== Using dB scale for peak detection ===")
    logger.info(f"Maximum magnitude (linear): {max_magnitude:.6f}")
    logger.info(f"Height threshold: {height_threshold_db:.1f} dB")
    logger.info(f"Prominence threshold: {prominence_factor_db:.1f} dB")

    # Calculate time resolution
    dt_ps = np.mean(np.diff(time_ps_search))

    # Determine minimum distance between peaks
    if min_distance_ps is None:
        tap_spacing_ps = delay_step_s * 1e12
        min_distance_ps = tap_spacing_ps * 0.8
        logger.info(f"\nFSR: {fsr_hz/1e9:.2f} GHz")
        logger.info(f"Expected tap spacing: {tap_spacing_ps:.3f} ps")
        logger.info(f"Using min_distance: {min_distance_ps:.3f} ps")

    min_distance = max(int(min_distance_ps / dt_ps), 1)
    logger.info(
        f"Minimum distance: {min_distance} samples ({min_distance * dt_ps:.3f} ps)"
    )

    # Find peaks
    peak_indices, peak_properties = find_peaks(
        h_magnitude_db,
        height=height_threshold_db,
        prominence=prominence_factor_db,
        distance=min_distance,
    )

    logger.info(f"\nInitial peaks found: {len(peak_indices)}")

    if len(peak_indices) == 0:
        logger.warning("No peaks detected! Try adjusting detection parameters.")
        raise ValueError("No taps detected in search window")

    # Filter to N largest if requested
    if n_taps is not None and len(peak_indices) > n_taps:
        logger.info(f"Filtering to keep only the {n_taps} largest peaks...")
        peak_magnitudes = h_magnitude[peak_indices]
        top_n_indices = np.argsort(peak_magnitudes)[::-1][:n_taps]
        peak_indices = peak_indices[top_n_indices]
        peak_indices = np.sort(peak_indices)

    # Extract tap coefficients
    tap_coefficients = h_time_search[peak_indices]
    tap_times_ps = time_ps_search[peak_indices]

    # Warn about negative times
    if np.any(tap_times_ps < 0):
        n_negative = np.sum(tap_times_ps < 0)
        logger.warning(
            f"{n_negative} tap(s) detected at t < 0 ps: {tap_times_ps[tap_times_ps < 0]}"
        )

    # Warn about large time gaps
    if len(tap_times_ps) > 1:
        tap_spacings = np.diff(tap_times_ps)
        expected_spacing = delay_step_s * 1e12
        large_gaps = tap_spacings > 2.0 * expected_spacing

        if np.any(large_gaps):
            gap_indices = np.where(large_gaps)[0]
            logger.warning(f"\nSuspicious large time gaps detected:")
            for idx in gap_indices:
                logger.warning(
                    f"  Between tap {idx+1} ({tap_times_ps[idx]:.1f} ps) "
                    f"and tap {idx+2} ({tap_times_ps[idx+1]:.1f} ps): "
                    f"{tap_spacings[idx]:.1f} ps gap (expected ~{expected_spacing:.1f} ps)"
                )

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
