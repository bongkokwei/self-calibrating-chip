"""
Tap coefficient extraction from measured spectral data.

This module extracts tap coefficients from insertion loss and phase measurements
using inverse Fourier transform, as described in Xu et al. (2022).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import ifft, fftshift
from scipy.signal import find_peaks
from scipy.constants import c
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def extract_tap_coefficients(
    df: pd.DataFrame,
    wavelength_col: str = "wl_axis",
    insertion_loss_col: str = "IL",
    phase_col: str = "LPD",
    center_wavelength_nm: float = 1550.0,
    n_taps: Optional[int] = None,
    time_window_ps: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract tap coefficients from measured insertion loss and phase spectra.

    This implements the inverse Fourier transform approach from Xu et al. (2022)
    to recover the impulse response (tap coefficients) from frequency domain data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing wavelength, insertion loss, and phase data
    wavelength_col : str
        Column name for wavelength data (in nm)
    insertion_loss_col : str
        Column name for insertion loss data (in dB)
    phase_col : str
        Column name for linear phase deviation data (in radians)
    center_wavelength_nm : float
        Center wavelength in nm (default 1550 nm)
    n_taps : int, optional
        Expected number of taps (for validation)
    time_window_ps : tuple, optional
        Time window (t_min, t_max) in ps to extract taps from

    Returns
    -------
    time_ps : np.ndarray
        Time axis in picoseconds (full range)
    h_time_shifted : np.ndarray
        Complex impulse response (full range)
    tap_times_ps : np.ndarray
        Time positions of detected taps in picoseconds
    tap_coefficients : np.ndarray
        Complex tap coefficients (amplitude and phase)

    Notes
    -----
    The function performs:
    1. Converts wavelength to frequency
    2. Reconstructs complex transfer function H(f) = √(10^(IL/10)) * exp(jφ)
    3. Applies inverse FFT to get impulse response h(t)
    4. Detects peaks corresponding to taps
    """

    # Extract data from dataframe
    wavelength_nm = df[wavelength_col].values
    insertion_loss_db = df[insertion_loss_col].values
    phase_rad = df[phase_col].values

    # Convert wavelength to frequency (relative to center)
    freq_hz = c / (wavelength_nm * 1e-9) - c / (center_wavelength_nm * 1e-9)

    # Sort by frequency (should be monotonically increasing for FFT)
    sort_idx = np.argsort(freq_hz)
    freq_hz = freq_hz[sort_idx]
    insertion_loss_db = insertion_loss_db[sort_idx]
    phase_rad = phase_rad[sort_idx]

    # Calculate frequency spacing and total bandwidth
    df_hz = np.mean(np.diff(freq_hz))
    total_bandwidth_hz = freq_hz[-1] - freq_hz[0]

    print(f"Frequency range: {freq_hz[0]/1e9:.2f} to {freq_hz[-1]/1e9:.2f} GHz")
    print(f"Total bandwidth: {total_bandwidth_hz/1e9:.2f} GHz")
    print(f"Frequency spacing: {df_hz/1e9:.3f} GHz")
    print(f"Number of frequency points: {len(freq_hz)}")

    # Convert insertion loss (dB) to amplitude (linear)
    # IL(dB) = 10*log10(Power) = 20*log10(Amplitude)
    # Therefore: Amplitude = 10^(IL/20)
    amplitude = 10 ** (insertion_loss_db / 20)

    # Construct complex transfer function
    # H(f) = A(f) * exp(j*φ(f))
    H_complex = amplitude * np.exp(1j * phase_rad)

    # Perform inverse FFT to get impulse response
    # Note: Using ifft which gives us h(t)
    h_time = ifft(H_complex)
    h_time_shifted = fftshift(h_time)  # Shift zero frequency to center

    # Calculate time axis
    # Time resolution = 1 / (total bandwidth)
    dt_s = 1 / total_bandwidth_hz
    dt_ps = dt_s * 1e12  # Convert to picoseconds

    n_points = len(h_time)
    time_ps = np.arange(n_points) * dt_ps
    time_ps = time_ps - time_ps[n_points // 2]  # Center at t=0

    print(f"\nTime domain:")
    print(f"Time resolution: {dt_ps:.3f} ps")
    print(f"Time span: {time_ps[0]:.1f} to {time_ps[-1]:.1f} ps")

    # Extract taps within specified time window
    if time_window_ps is not None:
        t_min, t_max = time_window_ps
        window_mask = (time_ps >= t_min) & (time_ps <= t_max)
        time_windowed = time_ps[window_mask]
        h_windowed = h_time_shifted[window_mask]
    else:
        time_windowed = time_ps
        h_windowed = h_time_shifted

    # Find peaks in the magnitude of impulse response
    h_magnitude = np.abs(h_windowed)

    # Detect peaks - use prominence to find significant taps
    # Prominence helps distinguish actual taps from noise
    mean_magnitude = np.mean(h_magnitude)

    # Calculate minimum distance between peaks
    # Use at least 1 ps spacing, or 1 sample point if resolution is lower
    min_distance = max(1, int(1.0 / dt_ps))

    print(f"\nPeak detection:")
    print(
        f"Minimum distance between peaks: {min_distance} samples ({min_distance * dt_ps:.3f} ps)"
    )

    peak_indices, peak_properties = find_peaks(
        h_magnitude,
        prominence=mean_magnitude * 0.1,  # Adjust threshold as needed
        distance=min_distance,  # Minimum distance between peaks
    )

    # Extract tap coefficients at peak positions
    tap_coefficients = h_windowed[peak_indices]
    tap_times_ps = time_windowed[peak_indices]

    print(f"\nDetected {len(tap_coefficients)} taps")
    if n_taps is not None and len(tap_coefficients) != n_taps:
        print(f"WARNING: Expected {n_taps} taps but found {len(tap_coefficients)}")

    # Display tap information
    print("\nTap Coefficients:")
    print(
        f"{'Tap':<5} {'Time (ps)':<12} {'Magnitude':<12} {'Phase (rad)':<12} {'Phase (deg)':<12}"
    )
    print("-" * 65)
    for i, (t, coeff) in enumerate(zip(tap_times_ps, tap_coefficients)):
        mag = np.abs(coeff)
        phase_rad_tap = np.angle(coeff)
        phase_deg = np.degrees(phase_rad_tap)
        print(
            f"{i+1:<5} {t:<12.3f} {mag:<12.6f} {phase_rad_tap:<12.4f} {phase_deg:<12.2f}"
        )

    return time_ps, h_time_shifted, tap_times_ps, tap_coefficients


def plot_time_domain_results(
    time_ps: np.ndarray,
    h_time_shifted: np.ndarray,
    tap_times_ps: np.ndarray,
    tap_coefficients: np.ndarray,
    time_window_ps: Optional[Tuple[float, float]] = None,
):
    """
    Plot time domain impulse response with detected taps.

    Parameters
    ----------
    time_ps : np.ndarray
        Full time axis
    h_time_shifted : np.ndarray
        Full impulse response
    tap_times_ps : np.ndarray
        Detected tap times
    tap_coefficients : np.ndarray
        Detected tap coefficients
    time_window_ps : tuple, optional
        Time window for zoomed view
    """
    # Apply time window for plot
    if time_window_ps is not None:
        t_min, t_max = time_window_ps
        window_mask = (time_ps >= t_min) & (time_ps <= t_max)
        time_windowed = time_ps[window_mask]
        h_windowed = h_time_shifted[window_mask]
    else:
        time_windowed = time_ps
        h_windowed = h_time_shifted

    h_magnitude = np.abs(h_windowed)

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot impulse response with detected taps
    ax.plot(time_windowed, h_magnitude, "b-", linewidth=1.5, label="Impulse Response")
    ax.plot(
        tap_times_ps,
        np.abs(tap_coefficients),
        "ro",
        markersize=10,
        label=f"Detected Taps (n={len(tap_coefficients)})",
    )

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.set_title(
        "Impulse Response with Detected Tap Coefficients",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def example_usage():
    """
    Example of how to use the tap extraction function with measured data.
    """
    # Load your measured data file
    data_file = ".\\data\\spectrum_test_20241212.csv"  # Update this path

    try:
        df = pd.read_csv(data_file, comment="#")
        print(f"Loaded data from: {data_file}")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}\n")

    except FileNotFoundError:
        print(f"Error: Could not find file {data_file}")
        print("\nPlease update the data_file path in example_usage()")
        return None

    # Extract tap coefficients
    time_ps, h_time, tap_times, tap_coeffs = extract_tap_coefficients(
        df,
        wavelength_col="wl_axis",  # Adjust to match your CSV column names
        insertion_loss_col="IL",
        phase_col="LPD",
        center_wavelength_nm=1550.0,  # Adjust to your measurement center wavelength
        n_taps=16,  # Expected number of taps
        time_window_ps=(0, 100),  # Focus on first 100 ps
    )

    # Plot time domain results
    plot_time_domain_results(
        time_ps=time_ps,
        h_time_shifted=h_time,
        tap_times_ps=tap_times,
        tap_coefficients=tap_coeffs,
        time_window_ps=(0, 100),
    )

    print(tap_coeffs.dtype)

    return time_ps, h_time, tap_times, tap_coeffs


if __name__ == "__main__":
    results = example_usage()
