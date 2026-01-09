"""
Tap coefficient extraction from measured spectral data.

This module extracts tap coefficients from insertion loss measurements only
using Kramers-Kronig phase recovery and inverse Fourier transform,
as described in Xu et al. (2022).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import ifft, fftshift
from scipy.signal import find_peaks, hilbert
from scipy.constants import c
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def kramers_kronig_phase_recovery(
    insertion_loss_db: np.ndarray,
) -> np.ndarray:
    """
    Recover phase response from insertion loss using Kramers-Kronig relationship.

    This uses the Hilbert transform to recover the phase from amplitude measurements,
    which works when the system satisfies the minimum-phase condition.

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
    φ(ω) = -H[ln|H(ω)|]
    where H[·] is the Hilbert transform and H(ω) is the transfer function.

    In terms of insertion loss:
    IL(dB) = 20*log10(|H(ω)|)
    ln|H(ω)| = IL(dB) * ln(10) / 20
    """

    # Convert IL (dB) to natural log of amplitude
    # IL(dB) = 20*log10(|H|)
    # ln|H| = IL(dB) * ln(10) / 20
    ln_amplitude = insertion_loss_db * np.log(10) / 20

    # Apply Hilbert transform to get phase
    # Note: scipy.signal.hilbert returns the analytic signal
    # We need just the imaginary part for the phase
    phase_recovered_rad = -np.imag(hilbert(ln_amplitude))

    return phase_recovered_rad


def extract_tap_coefficients(
    df: pd.DataFrame,
    wavelength_col: str = "wl_axis",
    freq_col: str = "f_axis",
    insertion_loss_col: str = "IL",
    center_wavelength_nm: float = 1550.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract impulse response from measured insertion loss spectra.

    This implements the Kramers-Kronig phase recovery and inverse Fourier
    transform approach from Xu et al. (2022) to recover the impulse response
    from amplitude-only measurements.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing wavelength and insertion loss data
    wavelength_col : str
        Column name for wavelength data (in nm)
    freq_col : str
        Column name for frequency data (in THz)
    insertion_loss_col : str
        Column name for insertion loss data (in dB)
    center_wavelength_nm : float
        Centre wavelength in nm (default 1550 nm)

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
    2. Recovers phase using Kramers-Kronig relationship (Hilbert transform)
    3. Reconstructs complex transfer function H(f) = √(10^(IL/10)) * exp(jφ)
    4. Applies inverse FFT to get impulse response h(t)
    """

    # Extract data from dataframe
    wavelength_nm = df[wavelength_col].values
    freq_hz = df[freq_col].values * 1e12
    insertion_loss_db = df[insertion_loss_col].values

    # Sort by frequency (should be monotonically increasing for FFT)
    sort_idx = np.argsort(freq_hz)
    freq_hz = freq_hz[sort_idx]
    insertion_loss_db = insertion_loss_db[sort_idx]

    # Calculate frequency spacing
    df_hz = np.mean(np.diff(freq_hz))

    print(f"Frequency range: {freq_hz[0]/1e9:.2f} to {freq_hz[-1]/1e9:.2f} GHz")
    print(f"Number of frequency points: {len(freq_hz)}")

    # Recover phase using Kramers-Kronig relationship
    print("\nRecovering phase using Kramers-Kronig relationship...")
    phase_rad = kramers_kronig_phase_recovery(insertion_loss_db)
    print("Phase recovery complete!")

    # Convert insertion loss (dB) to amplitude (linear)
    # IL(dB) = 10*log10(Power) = 20*log10(Amplitude)
    # Therefore: Amplitude = 10^(IL/20)
    amplitude = 10 ** (insertion_loss_db / 20)

    # Construct complex transfer function
    # H(f) = A(f) * exp(j*φ(f))
    H_complex = amplitude * np.exp(1j * phase_rad)

    # Perform inverse FFT to get impulse response
    h_time = ifft(H_complex)
    h_time_shifted = fftshift(h_time)  # Shift zero frequency to centre

    # Calculate time axis using fftfreq
    n_points = len(h_time)
    time_s = np.fft.fftfreq(n_points, d=df_hz)
    time_s_shifted = np.fft.fftshift(time_s)
    time_ps = time_s_shifted * 1e12  # Convert to picoseconds

    print(f"\nTime domain:")
    print(f"Time resolution: {np.mean(np.diff(time_ps)):.3f} ps")
    print(f"Time span: {time_ps[0]:.1f} to {time_ps[-1]:.1f} ps")

    return time_ps, h_time_shifted


def detect_taps(
    time_ps: np.ndarray,
    h_time: np.ndarray,
    fsr_hz: Optional[float] = None,
    n_taps: Optional[int] = None,
    prominence_factor_db: float = 3.0,  # 3 dB prominence
    min_distance_ps: Optional[float] = None,
    height_threshold_db: float = -40.0,  # Relative to max (in dB)
    use_db_scale: bool = True,  # New parameter!
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect tap positions and coefficients from impulse response.

    Parameters
    ----------
    time_ps : np.ndarray
        Time axis in picoseconds
    h_time : np.ndarray
        Complex impulse response
    fsr_hz : float, optional
        Free spectral range in Hz (chip design parameter)
    n_taps : int, optional
        Expected number of taps. If provided, only the N largest peaks are returned.
    prominence_factor_db : float
        Prominence threshold in dB (default 3.0 dB)
        Used only if use_db_scale=True
    min_distance_ps : float, optional
        Minimum distance between peaks in picoseconds
    height_threshold_db : float
        Minimum height in dB relative to maximum (default -40 dB)
        Used only if use_db_scale=True
    use_db_scale : bool
        If True, perform peak detection in dB scale (recommended for large dynamic range)
        If False, use linear scale

    Returns
    -------
    tap_times_ps : np.ndarray
        Time positions of detected taps in picoseconds
    tap_coefficients : np.ndarray
        Complex tap coefficients (amplitude and phase)
    """

    # Calculate magnitude
    h_magnitude = np.abs(h_time)
    max_magnitude = np.max(h_magnitude)

    # Convert to dB scale if requested
    if use_db_scale:
        print("\n=== Using dB scale for peak detection ===")

        # Normalize to max = 0 dB, add small offset to avoid log(0)
        h_magnitude_normalized = h_magnitude / max_magnitude
        h_magnitude_db = 20 * np.log10(h_magnitude_normalized + 1e-12)

        # Use dB scale for detection
        detection_signal = h_magnitude_db

        # Set thresholds in dB
        height_min = height_threshold_db  # e.g., -40 dB
        prominence_min = prominence_factor_db  # e.g., 3 dB

        print(f"Maximum magnitude (linear): {max_magnitude:.6f}")
        print(f"Maximum magnitude (dB): 0.00 dB (normalized)")
        print(f"Height threshold: {height_threshold_db:.1f} dB")
        print(f"Prominence threshold: {prominence_factor_db:.1f} dB")

    else:
        print("\n=== Using linear scale for peak detection ===")

        detection_signal = h_magnitude

        # Linear thresholds (original approach)
        height_threshold_linear = 0.05  # 5% of max
        prominence_factor_linear = 0.1  # 10% of max

        height_min = max_magnitude * height_threshold_linear
        prominence_min = max_magnitude * prominence_factor_linear

        print(f"Maximum magnitude: {max_magnitude:.6f}")
        print(
            f"Height threshold: {height_min:.6f} ({height_threshold_linear*100:.1f}% of max)"
        )
        print(
            f"Prominence threshold: {prominence_min:.6f} ({prominence_factor_linear*100:.1f}% of max)"
        )

    # Calculate time resolution
    dt_ps = np.mean(np.diff(time_ps))

    # Determine minimum distance between peaks
    if min_distance_ps is None and fsr_hz is not None:
        tap_spacing_s = 1.0 / fsr_hz
        tap_spacing_ps = tap_spacing_s * 1e12
        min_distance_ps = tap_spacing_ps * 0.8

        print(f"\nFSR provided: {fsr_hz/1e9:.2f} GHz")
        print(f"Expected tap spacing: {tap_spacing_ps:.3f} ps")
        print(f"Using min_distance: {min_distance_ps:.3f} ps (80% of expected)")
    elif min_distance_ps is None:
        min_distance_ps = 1.0
        print(f"\nUsing default min_distance: {min_distance_ps:.3f} ps")

    # Convert to samples
    min_distance = max(int(min_distance_ps / dt_ps), 1)
    print(f"Minimum distance: {min_distance} samples ({min_distance * dt_ps:.3f} ps)")

    # Find all peaks that meet the criteria
    peak_indices, peak_properties = find_peaks(
        detection_signal,
        height=height_min,
        prominence=prominence_min,
        distance=min_distance,
    )

    print(f"\nInitial peaks found: {len(peak_indices)}")

    # If n_taps is specified, keep only the N largest peaks
    if n_taps is not None and len(peak_indices) > n_taps:
        print(f"Filtering to keep only the {n_taps} largest peaks...")

        # Sort by the ORIGINAL linear magnitude (not dB)
        peak_magnitudes = h_magnitude[peak_indices]

        # Keep top N by magnitude
        top_n_indices = np.argsort(peak_magnitudes)[::-1][:n_taps]
        peak_indices = peak_indices[top_n_indices]
        peak_indices = np.sort(peak_indices)

    # Extract tap coefficients at peak positions
    tap_coefficients = h_time[peak_indices]
    tap_times_ps = time_ps[peak_indices]

    print(f"\nFinal detected taps: {len(tap_coefficients)}")

    # Validation
    if n_taps is not None and len(tap_coefficients) != n_taps:
        print(f"WARNING: Expected {n_taps} taps but found {len(tap_coefficients)}")

    # Display tap information
    print("\nTap Coefficients:")
    print(
        f"{'Tap':<5} {'Time (ps)':<12} {'Magnitude':<12} {'dB':<10} {'Phase (rad)':<12} {'Phase (deg)':<12}"
    )
    print("-" * 75)
    for i, (t, coeff) in enumerate(zip(tap_times_ps, tap_coefficients)):
        mag = np.abs(coeff)
        mag_db = 20 * np.log10(mag / max_magnitude)  # Relative to strongest tap
        phase_rad = np.angle(coeff)
        phase_deg = np.degrees(phase_rad)
        print(
            f"{i+1:<5} {t:<12.3f} {mag:<12.6f} {mag_db:<10.2f} {phase_rad:<12.4f} {phase_deg:<12.2f}"
        )

    return tap_times_ps, tap_coefficients


def plot_impulse_response(
    time_ps: np.ndarray,
    h_time: np.ndarray,
):
    """
    Plot impulse response magnitude.

    Parameters
    ----------
    time_ps : np.ndarray
        Time axis in picoseconds
    h_time : np.ndarray
        Complex impulse response
    """
    h_magnitude = np.abs(h_time)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot impulse response
    ax.plot(time_ps, h_magnitude, "b-", linewidth=1.5, label="Impulse Response")

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.set_title("Impulse Response (from Kramers-Kronig Phase Recovery)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def example_usage():
    """
    Example of how to use the tap extraction function with measured data.

    This workflow uses only insertion loss measurements to extract tap coefficients
    via Kramers-Kronig phase recovery.
    """
    # Load your measured data file
    data_file = "measurements/spectrum_test_20260108_150115.csv"  # Update this path

    try:
        df = pd.read_csv(data_file, comment="#")
        print(f"Loaded data from: {data_file}")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}\n")

    except FileNotFoundError:
        print(f"Error: Could not find file {data_file}")
        print("\nPlease update the data_file path in example_usage()")
        return None

    print("=" * 70)
    print("Tap Coefficient Extraction using Kramers-Kronig Phase Recovery")
    print("=" * 70 + "\n")

    # Step 1: Extract impulse response using Kramers-Kronig
    time_ps, h_time = extract_tap_coefficients(
        df,
        wavelength_col="wl_axis",
        freq_col="f_axis",
        insertion_loss_col="IL",
        center_wavelength_nm=1550.0,
    )

    # Step 2: Detect taps
    # If you know your chip's FSR, provide it here for better detection
    tap_times, tap_coeffs = detect_taps(
        time_ps=time_ps,
        h_time=h_time,
        fsr_hz=160e9,
        n_taps=16,
        use_db_scale=True,
        prominence_factor_db=0.3,  # 3 dB prominence
        height_threshold_db=-60.0,  # Detect taps down to -40 dB
    )

    # Step 3: Plot impulse response
    plot_impulse_response(
        time_ps=time_ps,
        h_time=h_time,
    )

    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)

    return time_ps, h_time, tap_times, tap_coeffs


if __name__ == "__main__":
    results = example_usage()
