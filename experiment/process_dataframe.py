"""
Script to re-process the insertion loss spectrum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from photonic_fir import (
    load_config,
    recover_impulse_response_from_df,
    detect_taps,
    tap_coeffs_to_power_splitting_ratios,
    build_mzi_tree_structure,
)

from photonic_fir.utils import (
    plot_impulse_response,
    plot_insertion_loss,
)


def retrive_filenames_from_directory(directory_path: str):
    import os

    filenames = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            filenames.append(os.path.join(directory_path, filename))
    return filenames


def get_all_mzi_ids() -> List[str]:
    """Generate all MZI IDs based on stage configuration."""
    mzi_ids = []
    for stage in range(1, 5):
        for pos in range(1, 2 ** (stage - 1) + 1):
            mzi_ids.append(f"{stage}-{pos}")
    return mzi_ids


def main():
    # Get list of files to process
    file_list = retrive_filenames_from_directory(
        directory_path="measurements/v2pi_batch_scan_results_run_007/4-8"
    )

    # Load configuration
    config = load_config("measurements/experiment_config.yaml")

    # Prepare MZI tree structure
    mzi_id = "1-1"
    mzi_ids = get_all_mzi_ids()

    mzi_tree = build_mzi_tree_structure(
        n_signal_taps=16,
        mzi_ids=mzi_ids,
    )

    power_splitting_ratios = np.zeros(len(file_list))

    for i, filename in enumerate(file_list):
        print(f"Processing file: {filename}")
        # 1. Load the raw data
        df = pd.read_csv(filename)

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

        # Get all power splitting ratios from tap coefficients
        psr_dict = tap_coeffs_to_power_splitting_ratios(tap_coeffs, mzi_tree)

        # Extract the specific MZI's PSR
        psr_db = psr_dict.get(mzi_id, 0.0)
        power_splitting_ratios[i] = psr_db

    plt.plot(power_splitting_ratios)
    plt.show()


if __name__ == "__main__":
    main()
