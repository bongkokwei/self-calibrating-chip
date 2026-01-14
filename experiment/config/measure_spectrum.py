"""
measure_spectrum_with_config.py

Spectral measurement script compatible with data_structure.py configuration system.
Uses MeasurementConfig dataclass for all settings.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

from fiberlabs_edfa import EDFAController, DrivingMode
from luna_ova import LunaOVA
from data_structure import MeasurementConfig, ChipParameters


def measure_spectrum(
    config: MeasurementConfig,
    folder_dir: str,
    file_name: str,
    chip_params: Optional[ChipParameters] = None,
) -> pd.DataFrame:
    """
    Measure insertion loss, phase, and other parameters using Luna OVA.
    Configuration comes from MeasurementConfig dataclass.

    Parameters
    ----------
    config : MeasurementConfig
        Measurement configuration from data_structure.py
    folder_dir : str
        Directory path where CSV file will be saved
    file_name : str
        Name of output CSV file (without .csv extension)
    chip_params : Optional[ChipParameters]
        Chip parameters for metadata (optional)
    edfa_port : str, optional
        COM port for EDFA controller (default: "COM6")
    edfa_output_level_dbm : float, optional
        EDFA output level in dBm (default: 13.0)
    num_averages : int, optional
        Number of averages for measurement (default: 1)

    Returns
    -------
    pd.DataFrame
        DataFrame containing all measurement data

    Raises
    ------
    IOError
        If unable to create output directory or write file
    """

    # Create output directory if it doesn't exist
    output_dir = Path(folder_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use instrument addresses from config if provided
    ova_ip = config.ova_address

    # Connect to EDFA and perform measurement
    with EDFAController(
        config.edfa_port,
        baudrate=config.edfa_baudrate,
    ) as edfa:

        # Get device info for metadata
        device_info = edfa.get_identification()
        print(f"EDFA connected: {device_info}")

        # Configure EDFA
        edfa.set_driving_mode(1, DrivingMode.ALC)
        edfa.set_alc_output_level(1, config.edfa_output_power_dbm)
        edfa.set_output_active(True)

        print(f"EDFA output active at {config.edfa_output_power_dbm} dBm")

        try:
            # Connect to OVA and measure
            with LunaOVA(ip=ova_ip) as ova:
                print("Performing OVA measurement...")

                data = ova.measure_full(
                    center_wavelength_nm=config.center_wavelength_nm,
                    wavelength_range_nm=config.wavelength_span_nm,
                    num_averages=config.num_averages,
                )

                print("Measurement complete")

        finally:
            # Always deactivate EDFA output
            edfa.set_output_active(False)
            print("EDFA output deactivated")

    # Create DataFrame from all returned data
    df = pd.DataFrame(data)

    # Save to CSV with metadata in header
    output_path = output_dir / f"{file_name}.csv"

    # Append DataFrame
    df.to_csv(output_path, mode="a", index=False)

    print(f"Data saved to: {output_path}")
    print(f"Columns saved: {list(df.columns)}")
    print(f"Number of data points: {len(df)}")

    return df


def measure_with_default_config(
    folder_dir: str = "./measurements",
    file_name_base: str = "spectrum_test",
    num_averages: int = 5,
) -> pd.DataFrame:
    """
    Convenience function using default MeasurementConfig.

    Parameters
    ----------
    folder_dir : str
        Output directory
    file_name_base : str
        Base name for output file (timestamp will be appended)
    num_averages : int
        Number of averages

    Returns
    -------
    pd.DataFrame
        Measurement data
    """
    # Create default config
    config = MeasurementConfig(
        center_wavelength_nm=1550.0,
        wavelength_span_nm=1.28,
        n_points=1000,
        chip_temperature_c=30.0,
    )

    # Create default chip parameters
    chip_params = ChipParameters()

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{file_name_base}_{timestamp}"

    # Perform measurement
    return measure_spectrum(
        config=config,
        folder_dir=folder_dir,
        file_name=file_name,
        chip_params=chip_params,
        num_averages=num_averages,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Perform measurement with default config
    df = measure_with_default_config(
        folder_dir="./measurements",
        file_name_base="config_based_test",
        num_averages=5,
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
    fig_filename = f"./measurements/config_based_test_plot_{timestamp}.png"
    fig.savefig(fig_filename, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {fig_filename}")

    plt.close(fig)
