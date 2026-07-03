"""
gap_method_demo.py

Standalone demo/visualisation for the gap-method impulse response recovery
(photonic_fir.processing.gap_method). Reads a saved insertion-loss spectrum
CSV, runs the gap method, extracts tap coefficients from the cross-correlation
region, and plots the intermediate stages (spectrum, autocorrelation,
zoomed cross-correlation region, extracted tap coefficients).

Usage:
    # Configure csv_path and column names in main() and run
    python gap_method_demo.py
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from photonic_fir.processing.gap_method import (
    recover_impulse_response_gap,
    extract_taps_from_cross_correlation,
)


def main():
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


if __name__ == "__main__":
    main()
