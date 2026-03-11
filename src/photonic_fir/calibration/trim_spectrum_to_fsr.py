"""
trim_spectrum_to_fsr.py

Trim a measured spectrum to an integer number of FSRs, using detected
IL dips as alignment references. This ensures the Hilbert transform
(used in KK phase recovery) operates on a naturally periodic signal,
eliminating edge artefacts that corrupt the recovered phase.

Usage
-----
    df = measure_spectrum(...)
    df_trimmed, info = trim_spectrum_to_fsr(df, nominal_fsr_hz=160e9)
    time_ps, h_time = recover_impulse_response_from_df(
        df_trimmed, fsr_hz=info["measured_fsr_hz"], ...
    )
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from typing import Tuple, Optional, Dict

import logging

logger = logging.getLogger(__name__)


def trim_spectrum_to_fsr(
    df: pd.DataFrame,
    nominal_fsr_hz: float = 160e9,
    n_fsr: Optional[int] = None,
    freq_col: str = "f_axis",
    il_col: str = "IL",
    smooth_window_ghz: float = 5.0,
    dip_prominence_db: float = 1.0,
    fsr_tolerance: float = 0.3,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Trim measured spectrum to an integer number of FSRs using dip alignment.

    Detects periodic IL dips, measures the true FSR from their spacing,
    and trims dip-to-dip so the Hilbert transform sees a periodic signal.

    Parameters
    ----------
    df : pd.DataFrame
        Output of measure_spectrum(). Must contain frequency and IL columns.
    nominal_fsr_hz : float
        Expected FSR in Hz (guides dip detection spacing). Default 160 GHz.
    n_fsr : int, optional
        Number of FSRs to retain. If None, uses the maximum available.
    freq_col : str
        Column name for frequency axis (in THz). Default "f_axis".
    il_col : str
        Column name for insertion loss (in dB). Default "IL".
    smooth_window_ghz : float
        Savitzky-Golay smoothing window in GHz for dip detection. Default 5.0.
    dip_prominence_db : float
        Minimum dip prominence in dB. Default 1.0.
    fsr_tolerance : float
        Fractional tolerance for valid dip spacings. Default 0.3 (±30%).

    Returns
    -------
    df_trimmed : pd.DataFrame
        Trimmed DataFrame with same columns as input.
    info : dict
        Metadata: "measured_fsr_hz", "n_fsr", "n_dips", "dip_freqs_hz",
        "trim_start_hz", "trim_end_hz".
    """
    # --- Extract and sort by frequency ---
    freq_hz = df[freq_col].values.copy() * 1e12  # THz -> Hz
    il_db = df[il_col].values.copy()

    sort_idx = np.argsort(freq_hz)
    freq_hz_sorted = freq_hz[sort_idx]
    il_sorted = il_db[sort_idx]
    df_freq = np.median(np.diff(freq_hz_sorted))

    logger.info(
        f"Input: {len(freq_hz)} pts, "
        f"{freq_hz_sorted[0]/1e12:.4f}-{freq_hz_sorted[-1]/1e12:.4f} THz "
        f"(span {(freq_hz_sorted[-1] - freq_hz_sorted[0])/1e9:.1f} GHz)"
    )

    # --- Smooth for robust dip detection ---
    win = int(smooth_window_ghz * 1e9 / df_freq)
    win = win + 1 if win % 2 == 0 else win
    win = max(win, 5)
    il_smooth = savgol_filter(il_sorted, window_length=win, polyorder=3)

    # --- Detect dips (peaks in -IL) ---
    min_dist = int(0.5 * nominal_fsr_hz / df_freq)
    dip_idx, _ = find_peaks(-il_smooth, distance=min_dist, prominence=dip_prominence_db)

    if len(dip_idx) < 2:
        raise ValueError(
            f"Only {len(dip_idx)} dip(s) detected, need >=2. "
            f"Try reducing dip_prominence_db ({dip_prominence_db} dB)."
        )

    dip_freqs = freq_hz_sorted[dip_idx]
    logger.info(f"Detected {len(dip_freqs)} dips")

    # --- Measure FSR from dip spacings ---
    spacings = np.diff(dip_freqs)

    # Accept spacings within tolerance of nominal FSR
    valid = (spacings > nominal_fsr_hz * (1 - fsr_tolerance)) & (
        spacings < nominal_fsr_hz * (1 + fsr_tolerance)
    )

    if not np.any(valid):
        # Check for half-FSR spacings (reference tap at 0.5T offset)
        half_valid = (spacings > 0.5 * nominal_fsr_hz * (1 - fsr_tolerance)) & (
            spacings < 0.5 * nominal_fsr_hz * (1 + fsr_tolerance)
        )
        if np.any(half_valid):
            measured_fsr = 2.0 * np.median(spacings[half_valid])
            dip_freqs = dip_freqs[::2]  # keep every other dip
            logger.info(
                f"Dips at ~FSR/2 spacing - using every other dip. "
                f"Measured FSR: {measured_fsr/1e9:.2f} GHz"
            )
        else:
            raise ValueError(
                f"No dip spacings match nominal FSR ({nominal_fsr_hz/1e9:.1f} GHz). "
                f"Found spacings: {spacings/1e9} GHz"
            )
    else:
        measured_fsr = np.median(spacings[valid])
        logger.info(
            f"Measured FSR: {measured_fsr/1e9:.2f} GHz "
            f"(from {np.sum(valid)} spacings, "
            f"std {np.std(spacings[valid])/1e9:.2f} GHz)"
        )

    # --- Select trim boundaries ---
    n_available = len(dip_freqs) - 1
    if n_fsr is None:
        n_fsr = n_available
    n_fsr = min(n_fsr, n_available)

    if n_fsr < 1:
        raise ValueError("Cannot form even 1 complete FSR from detected dips.")

    # Centre the selection within available dips
    start_dip = (len(dip_freqs) - 1 - n_fsr) // 2
    end_dip = start_dip + n_fsr

    freq_start = dip_freqs[start_dip]
    freq_end = dip_freqs[end_dip]

    # --- Trim ---
    orig_freq_hz = df[freq_col].values * 1e12
    mask = (orig_freq_hz >= freq_start) & (orig_freq_hz <= freq_end)
    df_trimmed = df.loc[mask].copy().reset_index(drop=True)

    logger.info(
        f"Trimmed: {len(df)} -> {len(df_trimmed)} pts, "
        f"{n_fsr} FSRs, span {(freq_end - freq_start)/1e9:.1f} GHz"
    )

    info = {
        "measured_fsr_hz": measured_fsr,
        "n_fsr": n_fsr,
        "n_dips": len(dip_freqs),
        "dip_freqs_hz": dip_freqs,
        "trim_start_hz": freq_start,
        "trim_end_hz": freq_end,
    }

    return df_trimmed, info
