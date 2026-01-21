"""
Test scripts configuration
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
    voltages = np.arange(0, 31, dtype=float) / 1.2  # 0 to 6 V

    with VoltageController(
        com_port="COM3",
        baud_rate=9600,
        zero_on_exit=False,
    ) as vc:
        vc.set_voltages(channels, voltages, v_max=30)

    return 0


def test_measure_spectrum(config: ExperimentConfig):

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

    return file_name


def test_tap_recovery(config: ExperimentConfig, filename: str = None):
    """
    Example usage with data structures from data_structure.py
    """
    from pathlib import Path

    # Load measured data
    data_file = (
        f"measurements/{filename}.csv"
        if filename
        else "measurements/spectrum_test_20260121_154125.csv"
    )

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
        fsr_hz=config.chip.fsr_hz,
        wavelength_col=config.measurement.wavelength_col,
        freq_col=config.measurement.frequency_col,
        insertion_loss_col=config.measurement.insertion_loss_col,
    )

    # Step 2: Detect taps
    tap_times, tap_coeffs = detect_taps(
        time_ps=time_ps,
        h_time=h_time,
        fsr_hz=config.chip.fsr_hz,
        delay_step_s=1 / config.chip.fsr_hz,
        n_taps=config.chip.n_taps,
        prominence_factor_db=config.measurement.prominence_factor_db,
        height_threshold_db=config.measurement.height_threshold_db,
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

    config = ExperimentConfig(
        measurement=MeasurementConfig(
            wavelength_span_nm=86.0,
            num_averages=3,
        ),
    )

    test_voltage_control()
    filename = test_measure_spectrum(config)
    test_tap_recovery(config, filename)

    with VoltageController(
        com_port="COM3",
        baud_rate=9600,
        zero_on_exit=True,
    ) as vc:
        print("\nEnd of tests. Voltages reset to zero.")
