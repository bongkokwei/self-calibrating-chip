"""
gap_method.py

Gap-method impulse response recovery as an alternative to Kramers-Kronig.

Instead of requiring the minimum-phase condition (beta < 1), the gap method
exploits the delay gap between the reference tap and the signal processing
taps. When tau_gap > T_spc, the cross-correlation terms between reference
and signal taps are cleanly separated from the autocorrelation terms in the
Fourier domain, allowing direct extraction of complex tap coefficients.

Reference: Wang et al., Light: Sci. Appl. 15, 149 (2026)
           Wang et al., Laser Photonics Rev. 19, 2400942 (2025)

Drop-in replacement for recover_impulse_response() — same inputs/outputs.
"""

import numpy as np
from scipy.fft import ifft, fftshift
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def recover_impulse_response_gap(
    freq_hz: np.ndarray,
    insertion_loss_db: np.ndarray,
    fsr_hz: float,
    ref_tap_index: int = 0,
    first_signal_tap_index: int = 8,
    n_signal_taps: int = 8,
    n_tiles: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract impulse response using the gap method (no KK condition required).

    The power spectrum |H(f)|^2 is Fourier-transformed to give the
    autocorrelation R(tau). The cross-correlation peaks between the reference
    tap and signal taps appear at delays tau_gap + n*dT, separated from the
    DC/autocorrelation cluster by the gap. A rectangular window isolates
    these peaks to yield complex tap coefficients directly.

    Parameters
    ----------
    freq_hz : np.ndarray
        Frequency axis in Hz (should span integer number of FSRs).
    insertion_loss_db : np.ndarray
        Measured insertion loss in dB (power, not amplitude).
    fsr_hz : float
        Free spectral range in Hz.
    ref_tap_index : int
        Index of reference tap (default 0 = shortest path).
    first_signal_tap_index : int
        Index of first signal processing tap (default 8 for your chip).
    n_signal_taps : int
        Number of signal processing taps (default 8).
    n_tiles : int
        Number of times to tile the spectrum before FFT (default 1).
        Tiling here is optional zero-padding-like smoothing, not required
        as it is for KK. Keep at 1 unless you want interpolated time bins.

    Returns
    -------
    time_ps : np.ndarray
        Time axis in picoseconds.
    h_time : np.ndarray
        Complex impulse response (autocorrelation of full chip).

    Notes
    -----
    The returned h_time contains all three groups (DC, cross+, cross-).
    Use detect_taps_windowed() on the cross-correlation region to extract
    tap coefficients. The tap delays in the cross-correlation region are:

        tau_n = (first_signal_tap_index - ref_tap_index + n) * delay_step

    for n = 0, ..., n_signal_taps - 1.

    This means detect_taps_windowed needs its expected delays shifted to
    the cross-correlation region rather than starting from tau=0.
    """
    # Convert insertion loss from dB to linear power
    il_linear = 10 ** (insertion_loss_db / 10)

    # Tile if requested (optional, mainly for time-domain interpolation)
    if n_tiles > 1:
        il_tiled = np.tile(il_linear, n_tiles)
    else:
        il_tiled = il_linear

    # Compute autocorrelation via IFFT of power spectrum
    # |H(f)|^2 is real, its IFFT gives R(tau)
    R = fftshift(ifft(il_tiled))

    # Build time axis
    n_pts = len(il_tiled)
    df = freq_hz[1] - freq_hz[0]  # frequency spacing
    dt = 1.0 / (n_pts * df)  # time spacing
    time_s = (np.arange(n_pts) - n_pts // 2) * dt
    time_ps = time_s * 1e12

    logger.info(f"Gap method: {n_pts} pts, dt={dt*1e12:.4f} ps")
    logger.info(f"Time range: [{time_ps[0]:.1f}, {time_ps[-1]:.1f}] ps")

    delay_step_s = 1.0 / fsr_hz
    gap_delay_ps = (first_signal_tap_index - ref_tap_index) * delay_step_s * 1e12
    last_cross_ps = gap_delay_ps + (n_signal_taps - 1) * delay_step_s * 1e12
    spc_duration_ps = (n_signal_taps - 1) * delay_step_s * 1e12

    logger.info(f"Gap delay: {gap_delay_ps:.2f} ps")
    logger.info(
        f"Cross-correlation region: [{gap_delay_ps:.2f}, {last_cross_ps:.2f}] ps"
    )
    logger.info(f"SPC duration: {spc_duration_ps:.2f} ps, Gap: {gap_delay_ps:.2f} ps")

    if gap_delay_ps <= spc_duration_ps:
        logger.warning(
            f"Gap ({gap_delay_ps:.1f} ps) <= SPC duration ({spc_duration_ps:.1f} ps)! "
            f"Autocorrelation and cross-correlation may overlap."
        )

    return time_ps, R


def recover_impulse_response_gap_from_df(
    df,
    fsr_hz: float,
    freq_col: str = "f_axis",
    insertion_loss_col: str = "IL",
    ref_tap_index: int = 0,
    first_signal_tap_index: int = 8,
    n_signal_taps: int = 8,
    n_tiles: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """DataFrame wrapper for recover_impulse_response_gap."""
    freq_thz = df[freq_col].values
    il_db = df[insertion_loss_col].values
    freq_hz = freq_thz * 1e12

    return recover_impulse_response_gap(
        freq_hz=freq_hz,
        insertion_loss_db=il_db,
        fsr_hz=fsr_hz,
        ref_tap_index=ref_tap_index,
        first_signal_tap_index=first_signal_tap_index,
        n_signal_taps=n_signal_taps,
        n_tiles=n_tiles,
    )


def extract_taps_from_cross_correlation(
    time_ps: np.ndarray,
    R: np.ndarray,
    delay_step_s: float,
    ref_tap_index: int = 0,
    first_signal_tap_index: int = 8,
    n_signal_taps: int = 8,
    window_width_ps: float = 3.0,
    noise_floor_db: float = -60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract tap coefficients directly from cross-correlation peaks.

    This is a specialised alternative to detect_taps_windowed that looks
    specifically in the cross-correlation region (positive delays at the
    gap offset) rather than at tap delays starting from zero.

    Parameters
    ----------
    time_ps : np.ndarray
        Time axis from recover_impulse_response_gap.
    R : np.ndarray
        Complex autocorrelation from recover_impulse_response_gap.
    delay_step_s : float
        Delay between adjacent taps in seconds.
    ref_tap_index : int
        Reference tap index.
    first_signal_tap_index : int
        First signal tap index.
    n_signal_taps : int
        Number of signal taps to extract.
    window_width_ps : float
        Search window width around each expected peak.
    noise_floor_db : float
        Threshold below which taps are considered absent.

    Returns
    -------
    tap_amplitudes : np.ndarray
        Tap amplitudes (normalised, ref factored out).
    tap_phases : np.ndarray
        Tap phases in radians (relative to reference).
    """
    delay_step_ps = delay_step_s * 1e12
    gap_offset = first_signal_tap_index - ref_tap_index

    R_mag = np.abs(R)
    max_R = np.max(R_mag)

    amplitudes = np.zeros(n_signal_taps)
    phases = np.zeros(n_signal_taps)

    logger.info(f"\nGap method tap extraction:")
    logger.info(
        f"{'Tap':<5} {'Expected (ps)':<15} {'Mag':<12} {'dB':<10} {'Phase (rad)':<12}"
    )
    logger.info("-" * 60)

    for n in range(n_signal_taps):
        expected_delay_ps = (gap_offset + n) * delay_step_ps

        # Window around expected delay
        w_start = expected_delay_ps - window_width_ps / 2
        w_end = expected_delay_ps + window_width_ps / 2
        mask = (time_ps >= w_start) & (time_ps <= w_end)

        if not np.any(mask):
            logger.warning(
                f"Tap {n}: no samples in window [{w_start:.1f}, {w_end:.1f}] ps"
            )
            amplitudes[n] = 10 ** (noise_floor_db / 20) * max_R
            phases[n] = 0.0
            continue

        # Find peak in window
        R_window = R[mask]
        R_mag_window = np.abs(R_window)
        peak_idx = np.argmax(R_mag_window)

        peak_val = R_window[peak_idx]
        peak_mag = np.abs(peak_val)
        peak_phase = np.angle(peak_val)
        peak_db = 20 * np.log10(peak_mag / max_R + 1e-12)

        if peak_db < noise_floor_db:
            amplitudes[n] = 10 ** (noise_floor_db / 20) * max_R
            phases[n] = 0.0
            status = "(below noise floor)"
        else:
            amplitudes[n] = peak_mag
            phases[n] = peak_phase
            status = ""

        logger.info(
            f"{n:<5} {expected_delay_ps:<15.2f} {peak_mag:<12.6f} "
            f"{peak_db:<10.2f} {peak_phase:<12.4f} {status}"
        )

    # Normalise: divide out the reference amplitude (it's common to all peaks)
    # The reference amplitude is sqrt of the R(0) peak contributed by ref alone,
    # but in practice we just normalise to the strongest cross-correlation peak
    ref_amplitude = np.max(amplitudes)
    if ref_amplitude > 0:
        amplitudes_normalised = amplitudes / ref_amplitude
    else:
        amplitudes_normalised = amplitudes

    # Phase is already relative to phi_ref (common offset cancels in error calc)
    return amplitudes_normalised, phases


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ---- Config — edit these ----
    csv_path = "measurements.csv"
    freq_col = "f_axis"  # column name, frequency in THz
    il_col = "IL"  # column name, insertion loss in dB
    fsr_hz = 160e9
    ref_tap_index = 0
    first_signal_tap_index = 8
    n_signal_taps = 8
    n_taps_total = 16
    # -----------------------------

    delay_step_s = 1.0 / fsr_hz
    delay_step_ps = delay_step_s * 1e12

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    freq_hz = df[freq_col].values * 1e12
    il_db = df[il_col].values

    # Run gap method
    time_ps, R = recover_impulse_response_gap(
        freq_hz=freq_hz,
        insertion_loss_db=il_db,
        fsr_hz=fsr_hz,
        ref_tap_index=ref_tap_index,
        first_signal_tap_index=first_signal_tap_index,
        n_signal_taps=n_signal_taps,
    )

    # Extract taps
    tap_amps, tap_phases = extract_taps_from_cross_correlation(
        time_ps=time_ps,
        R=R,
        delay_step_s=delay_step_s,
        ref_tap_index=ref_tap_index,
        first_signal_tap_index=first_signal_tap_index,
        n_signal_taps=n_signal_taps,
    )

    # Annotation positions
    gap_offset = first_signal_tap_index - ref_tap_index
    autocorr_end_ps = (n_taps_total - 1) * delay_step_ps
    cross_start_ps = gap_offset * delay_step_ps
    cross_end_ps = (gap_offset + n_signal_taps - 1) * delay_step_ps

    # Plot
    R_mag = np.abs(R)
    R_mag_db = 20 * np.log10(R_mag / np.max(R_mag) + 1e-12)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14))

    # 1. Insertion loss spectrum
    ax = axes[0]
    freq_ghz_plot = freq_hz / 1e9
    ax.plot(freq_ghz_plot - freq_ghz_plot[0], il_db, linewidth=0.8)
    ax.set_xlabel("Frequency offset (GHz)")
    ax.set_ylabel("Insertion Loss (dB)")
    ax.set_title("Measured Power Spectrum (calibration port)")
    ax.grid(True, alpha=0.3)

    # 2. Autocorrelation magnitude (full)
    ax = axes[1]
    ax.plot(time_ps, R_mag_db, linewidth=0.8)
    ax.set_xlabel("Delay (ps)")
    ax.set_ylabel("|R(τ)| (dB)")
    ax.set_title("Autocorrelation — full view")
    ax.set_xlim(time_ps[0], time_ps[-1])
    ax.grid(True, alpha=0.3)
    ax.axvspan(
        -cross_end_ps, -cross_start_ps, alpha=0.15, color="blue", label="Cross (B−)"
    )
    ax.axvspan(
        -autocorr_end_ps,
        autocorr_end_ps,
        alpha=0.10,
        color="grey",
        label="Autocorr (A)",
    )
    ax.axvspan(
        cross_start_ps, cross_end_ps, alpha=0.15, color="red", label="Cross (B+)"
    )
    ax.legend(fontsize=9, loc="upper right")

    # 3. Zoomed cross-correlation region
    ax = axes[2]
    zoom_margin = 3 * delay_step_ps
    zoom_mask = (time_ps >= cross_start_ps - zoom_margin) & (
        time_ps <= cross_end_ps + zoom_margin
    )
    ax.plot(time_ps[zoom_mask], R_mag_db[zoom_mask], linewidth=1.0)
    ax.set_xlabel("Delay (ps)")
    ax.set_ylabel("|R(τ)| (dB)")
    ax.set_title("Cross-correlation region (B+) — tap amplitudes")
    ax.grid(True, alpha=0.3)
    for n in range(n_signal_taps):
        tap_delay = (gap_offset + n) * delay_step_ps
        ax.axvline(tap_delay, color="red", alpha=0.4, linestyle="--", linewidth=0.8)
        ax.text(
            tap_delay,
            ax.get_ylim()[1] - 3,
            f"T{n}",
            ha="center",
            fontsize=8,
            color="red",
        )

    # 4. Extracted tap coefficients
    ax = axes[3]
    tap_indices = np.arange(n_signal_taps)
    tap_amps_db = 20 * np.log10(tap_amps + 1e-12)

    markerline, stemlines, baseline = ax.stem(
        tap_indices, tap_amps_db, linefmt="tab:blue", markerfmt="o", basefmt="k-"
    )
    ax.set_xlabel("Signal tap index")
    ax.set_ylabel("Amplitude (dB)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xticks(tap_indices)
    ax.grid(True, alpha=0.3, axis="y")

    ax_ph = ax.twinx()
    ax_ph.plot(tap_indices, np.degrees(tap_phases), "rs", markersize=8, label="Phase")
    ax_ph.set_ylabel("Phase (deg)", color="tab:red")
    ax_ph.tick_params(axis="y", labelcolor="tab:red")

    ax.set_title("Extracted tap coefficients (gap method)")
    plt.tight_layout()
    plt.savefig("gap_method_results.png", dpi=300)

    # plt.show()
