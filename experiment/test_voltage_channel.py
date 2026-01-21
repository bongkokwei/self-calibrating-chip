"""
Test voltage channel configuration
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


from photonic_fir import ExperimentConfig, MeasurementConfig
from voltage_ctrl import VoltageController
from photonic_fir import (
    measure_spectrum,
    recover_impulse_response_from_df,
    detect_taps,
)


def test_voltage_control():
    config = ExperimentConfig()

    # Test channel mapping
    print("\nVoltage Channel Mapping:")
    all_channels = config.chip.channel_mapping.get_all_channels()
    for device_id, channel in sorted(all_channels.items()):
        print(f"  {device_id:<15} → Channel {channel}")

    # Validate no duplicates
    try:
        config.chip.channel_mapping.validate_no_duplicates()
        print("\n✓ Channel mapping validated (no duplicates)")
    except ValueError as e:
        print(f"\n✗ Channel mapping error: {e}")

    channels = list(range(1, 32))
    voltages = np.arange(0, 31, dtype=float) / 10

    with VoltageController(
        com_port="COM3",
        baud_rate=9600,
        zero_on_exit=False,
    ) as vc:
        vc.set_voltages(channels, voltages, v_max=30)

    return 0


def test_measure_spectrum():

    config = ExperimentConfig(measurement=MeasurementConfig(wavelength_span_nm=5.0))

    # Generate filename with timestamp
    file_name_base = "spectrum_test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{file_name_base}_{timestamp}"

    df = measure_spectrum(
        center_wavelength_nm=config.measurement.center_wavelength_nm,
        wavelength_span_nm=config.measurement.wavelength_span_nm,
        num_averages=config.measurement.num_averages,
        edfa_port=config.measurement.edfa_port,
        edfa_baudrate=config.measurement.edfa_baudrate,
        edfa_output_power_dbm=config.measurement.edfa_output_power_dbm,
        ova_ip=config.measurement.ova_address,
        folder_dir="./measurements",
        file_name=file_name,
    )

    # Quick inspection
    print("\nFirst few rows:")
    print(df.head())
    print(f"\nData shape: {df.shape}")

    # Plot insertion loss and phase
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot insertion loss
    ax1.plot(df["wl_axis"], df["IL"], "b-", linewidth=1.5)
    ax1.set_ylabel("Insertion Loss (dB)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Luna OVA Measurement Results (Config-based)", fontsize=12)

    # Plot phase
    ax2.plot(df["wl_axis"], df["LPD"], "r-", linewidth=1.5)
    ax2.set_xlabel("Wavelength (nm)", fontsize=12)
    ax2.set_ylabel("Phase (rad)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate timestamp for plot filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_filename = f"./measurements/{file_name}.png"
    fig.savefig(fig_filename, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {fig_filename}")

    plt.close(fig)


def test_tap_recovery():
    """
    Example usage with data structures from data_structure.py
    """
    from pathlib import Path

    # Load measured data
    data_file = "measurements/spectrum_test_20260121_150848.csv"

    try:
        df = pd.read_csv(data_file, comment="#")
        print(f"Loaded data from: {data_file}")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}\n")
    except FileNotFoundError:
        print(f"Error: Could not find file {data_file}")
        return None

    print("=" * 70)
    print("Tap Coefficient Extraction using Kramers-Kronig Phase Recovery")
    print("=" * 70 + "\n")

    # Step 1: Recover impulse response (using DataFrame wrapper)
    time_ps, h_time = recover_impulse_response_from_df(
        df=df,
        fsr_hz=160e9,
        wavelength_col="wl_axis",
        freq_col="f_axis",
        insertion_loss_col="IL",
    )

    # Step 2: Detect taps
    tap_times, tap_coeffs = detect_taps(
        time_ps=time_ps,
        h_time=h_time,
        fsr_hz=160e9,
        delay_step_s=1 / 160e9,
        n_taps=16,
        prominence_factor_db=0.3,
        height_threshold_db=-60.0,
    )

    # Step 3: Plot results
    from photonic_fir.processing.tap_recovery import plot_impulse_response

    plot_impulse_response(
        time_ps=time_ps,
        h_time=h_time,
        tap_times_ps=tap_times,
        tap_coeffs=tap_coeffs,
    )

    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)

    return time_ps, h_time, tap_times, tap_coeffs


if __name__ == "__main__":
    # test_voltage_control()
    # test_measure_spectrum()
    test_tap_recovery()
