"""
Script to re-process the insertion loss spectrum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import re

from photonic_fir import (
    ExperimentConfig,
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


def get_power_splitting_ratio_from_file(
    filename: str,
    config: ExperimentConfig,
    mzi_tree: Dict,
    mzi_id: str,
) -> float:
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

    tap_coeffs = np.abs(tap_coeffs) / np.max(np.abs(tap_coeffs))

    # Get all power splitting ratios from tap coefficients
    psr_dict = tap_coeffs_to_power_splitting_ratios(tap_coeffs, mzi_tree)

    # Extract the specific MZI's PSR
    psr_db = psr_dict.get(mzi_id, 0.0)

    # Extract voltage from filename for plotting
    voltage_match = re.search(r"(\d+\.\d+)v", filename)
    voltage = float(voltage_match.group(1)) if voltage_match else 999.0

    # plot_impulse_response(
    #     time_ps=time_ps,
    #     h_time=h_time / np.max(np.abs(h_time)),
    #     tap_times_ps=tap_times,
    #     tap_coeffs=tap_coeffs,
    # )

    return psr_db, voltage


def main():
    # Get list of files to process
    mzi_id = "1-1"
    file_list = retrive_filenames_from_directory(
        # directory_path=f"measurements/v2pi_batch_scan_results_run_009/mzi_{mzi_id}"
        directory_path=f"measurements/mzi_{mzi_id}"
    )

    # Load configuration
    config = load_config("measurements/experiment_config_shorter_range.yaml")

    # Prepare MZI tree structure
    mzi_ids = get_all_mzi_ids()
    mzi_tree = build_mzi_tree_structure(
        n_signal_taps=16,
        mzi_ids=mzi_ids,
    )

    power_splitting_ratios = np.zeros(len(file_list))
    voltages = np.zeros(len(file_list))
    for i, filename in enumerate(file_list):
        psr_db, voltage = get_power_splitting_ratio_from_file(
            filename=filename,
            config=config,
            mzi_tree=mzi_tree,
            mzi_id=mzi_id,
        )
        power_splitting_ratios[i] = psr_db
        voltages[i] = voltage

    plt.plot(voltages**2, power_splitting_ratios, ".")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Power Splitting Ratio (dB)")
    plt.title(f"Power Splitting Ratio vs Voltage for MZI {mzi_id}")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
