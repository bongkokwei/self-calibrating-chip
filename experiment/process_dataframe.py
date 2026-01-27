"""
Script to re-process the insertion loss spectrum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from photonic_fir import (
    load_config,
    recover_impulse_response_from_df,
    detect_taps,
    plot_impulse_response,
    plot_insertion_loss,
)


def main():
    # Load configuration
    config = load_config("config/photonic_fir_config.yaml")

    # 1. Load the raw data
    df = pd.read_csv("data/insertion_loss_raw.csv")

    # 2. Recover impulse response using existing function
    time_ps, h_time = recover_impulse_response_from_df(
        df=df,
        fsr_hz=config.chip.fsr_hz,
        wavelength_col=config.measurement.wavelength_col,
        freq_col=config.measurement.frequency_col,
        insertion_loss_col=config.measurement.insertion_loss_col,
    )

    # 3. Detect taps using existing function
    tap_times, tap_coeffs = detect_taps(
        time_ps=time_ps,
        h_time=h_time,
        fsr_hz=config.chip.fsr_hz,
        delay_step_s=config.chip.delay_step_s,
        n_taps=config.chip.n_taps,
        prominence_factor_db=config.measurement.prominence_factor_db,
        min_distance_ps=config.measurement.min_distance_ps,
        height_threshold_db=config.measurement.height_threshold_db,
    )

    plot_insertion_loss(
        df=df,
        title="Re-processed Insertion Loss Spectrum",
        save_dir="results",
        file_name_base="insertion_loss_reprocessed",
        show_plot=True,
    )

    plot_impulse_response(
        time_ps=time_ps,
        h_time=h_time,
        tap_times_ps=tap_times,
        tap_coeffs=tap_coeffs,
    )
