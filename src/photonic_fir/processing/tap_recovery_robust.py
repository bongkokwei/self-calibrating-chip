"""
Refactored detect_taps function using windowed search approach.

Key changes:
- Each tap gets its own dedicated time window around expected delay
- Searches for peak within that window only
- Returns -50 dB with phase=0 if no peak above noise floor
- Prevents spurious peaks at wrong delays from being detected
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def detect_taps_noise_tolerant(
    time_ps: np.ndarray,
    h_time: np.ndarray,
    fsr_hz: float,
    delay_step_s: float,
    n_taps: int = 16,
    window_width_ps: float = 3.0,
    noise_floor_db: float = -60.0,
    return_threshold_db: float = -50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect tap positions and coefficients from impulse response using windowed search.

    Searches for peaks in time windows centred on expected tap delays. If no peak
    is found above the noise floor in a window, returns a coefficient at the noise
    threshold with zero phase.

    Parameters
    ----------
    time_ps : np.ndarray
        Time axis in picoseconds
    h_time : np.ndarray
        Complex impulse response
    fsr_hz : float
        Free spectral range in Hz (unused but kept for API compatibility)
    delay_step_s : float
        Delay between taps in seconds (6.25 ps typical)
    n_taps : int
        Number of taps to detect (default: 16)
    window_width_ps : float
        Width of search window around each expected tap delay in picoseconds
        (default: 3.0 ps, giving ±1.5 ps tolerance)
    noise_floor_db : float
        Noise floor threshold in dB relative to maximum (default: -60.0)
    return_threshold_db : float
        Value to return for taps below noise floor in dB (default: -50.0)

    Returns
    -------
    tap_times_ps : np.ndarray
        Time positions of detected taps (expected delay if below threshold)
    tap_coefficients : np.ndarray
        Complex tap coefficients

    Notes
    -----
    - Window spacing: delay_step_s * 1e12 ps
    - Window width: window_width_ps (should be < delay spacing to avoid overlap)
    - For 6.25 ps spacing and 3 ps windows: 3.25 ps gap between windows
    - Taps below noise floor return: magnitude = 10^(-50/20), phase = 0
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Windowed Tap Detection")
    logger.info(f"{'='*60}")
    logger.info(f"Number of taps:     {n_taps}")
    logger.info(f"Tap spacing:        {delay_step_s * 1e12:.3f} ps")
    logger.info(
        f"Window width:       {window_width_ps:.3f} ps (±{window_width_ps/2:.3f} ps)"
    )
    logger.info(f"Window gap:         {delay_step_s * 1e12 - window_width_ps:.3f} ps")
    logger.info(f"Noise floor:        {noise_floor_db:.1f} dB")
    logger.info(f"Return threshold:   {return_threshold_db:.1f} dB")

    # Calculate magnitude and normalise to dB
    h_magnitude = np.abs(h_time)
    max_magnitude = np.max(h_magnitude)
    h_magnitude_normalised = h_magnitude / max_magnitude
    h_magnitude_db = 20 * np.log10(h_magnitude_normalised + 1e-12)

    logger.info(f"Max magnitude:      {max_magnitude:.6e}")
    logger.info(f"Time range:         [{time_ps[0]:.2f}, {time_ps[-1]:.2f}] ps")
    logger.info(f"Time resolution:    {np.mean(np.diff(time_ps)):.4f} ps")

    # Initialise output arrays
    tap_times_ps = np.zeros(n_taps)
    tap_coefficients = np.zeros(n_taps, dtype=complex)

    # Threshold magnitude for weak taps (linear scale)
    threshold_magnitude = 10 ** (return_threshold_db / 20) * max_magnitude

    logger.info(f"\n{'Tap':<5} {'Window [ps]':<25} {'Status':<50}")
    logger.info(f"{'-'*80}")

    # Process each tap
    for tap_idx in range(n_taps):
        # Expected tap delay
        expected_delay_ps = tap_idx * delay_step_s * 1e12

        # Define window bounds
        window_start = expected_delay_ps - window_width_ps / 2
        window_end = expected_delay_ps + window_width_ps / 2

        # Create window mask
        window_mask = (time_ps >= window_start) & (time_ps <= window_end)
        n_samples = np.sum(window_mask)

        if n_samples == 0:
            logger.warning(
                f"{tap_idx:<5} [{window_start:6.2f}, {window_end:6.2f}]   "
                f"No samples in window - using threshold"
            )
            tap_times_ps[tap_idx] = expected_delay_ps
            tap_coefficients[tap_idx] = threshold_magnitude + 0j
            continue

        # Extract window data
        time_window = time_ps[window_mask]
        h_magnitude_window = h_magnitude[window_mask]
        h_magnitude_db_window = h_magnitude_db[window_mask]
        h_time_window = h_time[window_mask]

        # Find peaks in window (any local maxima)
        peak_indices, _ = find_peaks(h_magnitude_window)

        if len(peak_indices) == 0:
            # No peaks found - use maximum value in window as fallback
            max_idx = np.argmax(h_magnitude_window)
            peak_magnitude_db = h_magnitude_db_window[max_idx]

            if peak_magnitude_db < noise_floor_db:
                logger.info(
                    f"{tap_idx:<5} [{window_start:6.2f}, {window_end:6.2f}]   "
                    f"No peak, max={peak_magnitude_db:6.1f} dB → threshold"
                )
                tap_times_ps[tap_idx] = expected_delay_ps
                tap_coefficients[tap_idx] = threshold_magnitude + 0j
            else:
                # Weak signal but above noise floor
                peak_time = time_window[max_idx]
                logger.info(
                    f"{tap_idx:<5} [{window_start:6.2f}, {window_end:6.2f}]   "
                    f"No peak, max={peak_magnitude_db:6.1f} dB at {peak_time:.2f} ps"
                )
                tap_times_ps[tap_idx] = peak_time
                tap_coefficients[tap_idx] = h_time_window[max_idx]
        else:
            # Find the largest peak in window
            peak_magnitudes = h_magnitude_window[peak_indices]
            largest_peak_local_idx = peak_indices[np.argmax(peak_magnitudes)]

            peak_time = time_window[largest_peak_local_idx]
            peak_magnitude_db = h_magnitude_db_window[largest_peak_local_idx]
            peak_coeff = h_time_window[largest_peak_local_idx]

            # Check if peak is above noise floor
            if peak_magnitude_db < noise_floor_db:
                logger.info(
                    f"{tap_idx:<5} [{window_start:6.2f}, {window_end:6.2f}]   "
                    f"Peak at {peak_time:.2f} ps ({peak_magnitude_db:6.1f} dB) below noise → threshold"
                )
                tap_times_ps[tap_idx] = expected_delay_ps
                tap_coefficients[tap_idx] = threshold_magnitude + 0j
            else:
                offset = peak_time - expected_delay_ps
                n_peaks_info = (
                    f" ({len(peak_indices)} peaks)" if len(peak_indices) > 1 else ""
                )
                logger.info(
                    f"{tap_idx:<5} [{window_start:6.2f}, {window_end:6.2f}]   "
                    f"✓ {peak_magnitude_db:6.1f} dB at {peak_time:6.2f} ps (Δ={offset:+5.2f} ps){n_peaks_info}"
                )
                tap_times_ps[tap_idx] = peak_time
                tap_coefficients[tap_idx] = peak_coeff

    # Display final summary table
    logger.info(f"\n{'='*90}")
    logger.info(f"Tap Detection Summary")
    logger.info(f"{'='*90}")
    logger.info(
        f"{'Tap':<5} {'Time (ps)':<12} {'Expected (ps)':<15} "
        f"{'|Coeff|':<12} {'dB':<10} {'∠(rad)':<10} {'∠(°)':<10}"
    )
    logger.info(f"{'-'*90}")

    for i in range(n_taps):
        expected_delay_ps = i * delay_step_s * 1e12
        mag = np.abs(tap_coefficients[i])
        mag_db = 20 * np.log10(mag / max_magnitude)
        phase_rad = np.angle(tap_coefficients[i])
        phase_deg = np.degrees(phase_rad)

        # Flag if at threshold
        at_threshold = " WARNING " if np.isclose(mag, threshold_magnitude) else "  "

        logger.info(
            f"{at_threshold}{i:<3} {tap_times_ps[i]:<12.3f} {expected_delay_ps:<15.3f} "
            f"{mag:<12.6e} {mag_db:<10.2f} {phase_rad:<10.4f} {phase_deg:<10.2f}"
        )

    # Count taps below threshold
    n_below_threshold = np.sum(
        np.isclose(np.abs(tap_coefficients), threshold_magnitude)
    )
    if n_below_threshold > 0:
        logger.warning(
            f"\n{n_below_threshold}/{n_taps} taps below noise floor "
            f"(returned at {return_threshold_db} dB)"
        )

    return tap_times_ps, tap_coefficients
