"""
tap_extraction.py

Tap coefficient extraction from measured spectral data compatible with data_structure.py.
Uses Kramers-Kronig phase recovery and inverse Fourier transform.
"""

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
    ln_amplitude = insertion_loss_db * np.log(10) / 20

    # Apply Hilbert transform to get phase
    phase_recovered_rad = -np.imag(hilbert(ln_amplitude))

    return phase_recovered_rad


def recover_impulse_response(
    wavelength_nm: np.ndarray,
    freq_hz: np.ndarray,
    insertion_loss_db: np.ndarray,
    fsr_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract impulse response from measured insertion loss spectra.

    Implements the Kramers-Kronig phase recovery and inverse Fourier
    transform approach from Xu et al. (2022).

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Wavelength data in nm
    freq_hz : np.ndarray
        Frequency data in Hz
    insertion_loss_db : np.ndarray
        Insertion loss data in dB
    fsr_hz : float
        Free spectral range in Hz

    Returns
    -------
    time_ps : np.ndarray
        Time axis in picoseconds
    h_time_shifted : np.ndarray
        Complex impulse response

    Notes
    -----
    The function performs:
    1. Converts wavelength to frequency
    2. Recovers phase using Kramers-Kronig (Hilbert transform)
    3. Reconstructs complex transfer function H(f) = A(f) * exp(jφ(f))
    4. Applies inverse FFT to get impulse response h(t)
    """
    # Sort by frequency
    sort_idx = np.argsort(freq_hz)
    freq_hz = freq_hz[sort_idx]
    insertion_loss_db = insertion_loss_db[sort_idx]

    # Calculate frequency spacing
    df_hz = np.mean(np.diff(freq_hz))

    print(f"Frequency range: {freq_hz[0]/1e9:.2f} to {freq_hz[-1]/1e9:.2f} GHz")
    print(f"Number of frequency points: {len(freq_hz)}")
    print(f"Expected FSR: {fsr_hz/1e9:.2f} GHz")

    # Recover phase using Kramers-Kronig
    print("\nRecovering phase using Kramers-Kronig relationship...")
    phase_rad = kramers_kronig_phase_recovery(insertion_loss_db)
    print("Phase recovery complete!")

    # Convert insertion loss (dB) to amplitude (linear)
    amplitude = 10 ** (insertion_loss_db / 20)

    # Construct complex transfer function
    H_complex = amplitude * np.exp(1j * phase_rad)

    # Perform inverse FFT to get impulse response
    h_time = ifft(H_complex)
    h_time_shifted = fftshift(h_time)

    # Calculate time axis
    n_points = len(h_time)
    time_s = np.fft.fftfreq(n_points, d=df_hz)
    time_s_shifted = np.fft.fftshift(time_s)
    time_ps = time_s_shifted * 1e12  # Convert to picoseconds

    print(f"\nTime domain:")
    print(f"Time resolution: {np.mean(np.diff(time_ps)):.3f} ps")
    print(f"Time span: {time_ps[0]:.1f} to {time_ps[-1]:.1f} ps")

    return time_ps, h_time_shifted


def recover_impulse_response_from_df(
    df: pd.DataFrame,
    fsr_hz: float,
    wavelength_col: str = "wl_axis",
    freq_col: str = "f_axis",
    insertion_loss_col: str = "IL",
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
    wavelength_nm = df[wavelength_col].values
    freq_thz = df[freq_col].values
    freq_hz = freq_thz * 1e12  # Convert THz to Hz
    insertion_loss_db = df[insertion_loss_col].values

    # Call core function
    return recover_impulse_response(
        wavelength_nm=wavelength_nm,
        freq_hz=freq_hz,
        insertion_loss_db=insertion_loss_db,
        fsr_hz=fsr_hz,
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
    print("\n=== Using dB scale for peak detection ===")
    h_magnitude_normalised = h_magnitude / max_magnitude
    h_magnitude_db = 20 * np.log10(h_magnitude_normalised + 1e-12)
    detection_signal = h_magnitude_db
    height_min = height_threshold_db
    prominence_min = prominence_factor_db

    print(f"Maximum magnitude (linear): {max_magnitude:.6f}")
    print(f"Height threshold: {height_threshold_db:.1f} dB")
    print(f"Prominence threshold: {prominence_factor_db:.1f} dB")

    # Calculate time resolution
    dt_ps = np.mean(np.diff(time_ps))

    # Determine minimum distance between peaks
    if min_distance_ps is None:
        tap_spacing_s = delay_step_s
        tap_spacing_ps = tap_spacing_s * 1e12
        min_distance_ps = tap_spacing_ps * 0.8

        print(f"\nFSR: {fsr_hz/1e9:.2f} GHz")
        print(f"Expected tap spacing: {tap_spacing_ps:.3f} ps")
        print(f"Using min_distance: {min_distance_ps:.3f} ps")

    # Convert to samples
    min_distance = max(int(min_distance_ps / dt_ps), 1)
    print(f"Minimum distance: {min_distance} samples ({min_distance * dt_ps:.3f} ps)")

    # Find peaks
    peak_indices, peak_properties = find_peaks(
        detection_signal,
        height=height_min,
        prominence=prominence_min,
        distance=min_distance,
    )

    print(f"\nInitial peaks found: {len(peak_indices)}")

    # Filter to N largest if specified
    if n_taps is None:
        n_taps = n_taps

    if len(peak_indices) > n_taps:
        print(f"Filtering to keep only the {n_taps} largest peaks...")
        peak_magnitudes = h_magnitude[peak_indices]
        top_n_indices = np.argsort(peak_magnitudes)[::-1][
            1 : n_taps + 1
        ]  # Exclude main peak, change to [0:+n_taps] to include main peak
        peak_indices = peak_indices[top_n_indices]
        peak_indices = np.sort(peak_indices)

    # Extract tap coefficients
    tap_coefficients = h_time[peak_indices]
    tap_times_ps = time_ps[peak_indices]

    print(f"\nFinal detected taps: {len(tap_coefficients)}")

    # Display tap information
    print("\nTap Coefficients:")
    print(
        f"{'Tap':<5} {'Time (ps)':<12} {'Magnitude':<12} {'dB':<10} "
        f"{'Phase (rad)':<12} {'Phase (deg)':<12}"
    )
    print("-" * 75)

    for i, (t, coeff) in enumerate(zip(tap_times_ps, tap_coefficients)):
        mag = np.abs(coeff)
        mag_db = 20 * np.log10(mag / max_magnitude)
        phase_rad = np.angle(coeff)
        phase_deg = np.degrees(phase_rad)
        print(
            f"{i+1:<5} {t:<12.3f} {mag:<12.6f} {mag_db:<10.2f} "
            f"{phase_rad:<12.4f} {phase_deg:<12.2f}"
        )

    return tap_times_ps, tap_coefficients


def plot_impulse_response(
    time_ps: np.ndarray,
    h_time: np.ndarray,
    tap_times_ps: Optional[np.ndarray] = None,
    tap_coeffs: Optional[np.ndarray] = None,
):
    """
    Plot impulse response magnitude with detected taps.

    Parameters
    ----------
    time_ps : np.ndarray
        Time axis in picoseconds
    h_time : np.ndarray
        Complex impulse response
    tap_times_ps : np.ndarray, optional
        Detected tap positions
    tap_coeffs : np.ndarray, optional
        Detected tap coefficients
    """
    h_magnitude = np.abs(h_time)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot impulse response
    ax.plot(
        time_ps,
        10 * np.log10(h_magnitude) + 1e-12,
        "b-",
        linewidth=1.5,
        label="Impulse Response",
    )

    # Mark detected taps if provided
    if tap_times_ps is not None and tap_coeffs is not None:
        tap_magnitudes = np.abs(tap_coeffs)
        ax.plot(
            tap_times_ps,
            10 * np.log10(tap_magnitudes),
            "ro",
            markersize=8,
            label="Detected Taps",
        )

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.set_title("Impulse Response (Kramers-Kronig Phase Recovery)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-40)
    ax.set_xlim(
        left=-(1 / 160e9) * 1e12, right=16 * (1 / 160e9) * 1e12
    )  # Example: 16 taps at 160 GHz FSR

    plt.tight_layout()
    plt.show()
