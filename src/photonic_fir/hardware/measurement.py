"""
measure_spectrum_with_config.py

Spectral measurement script compatible with data_structure.py configuration system.
Uses MeasurementConfig dataclass for all settings.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from fiberlabs_edfa import EDFAController, DrivingMode
from luna_ova import LunaOVA


def measure_spectrum(
    center_wavelength_nm: float,
    wavelength_span_nm: float,
    num_averages: int,
    edfa_port: str,
    edfa_baudrate: int,
    edfa_output_power_dbm: float,
    ova_ip: Optional[str],
    folder_dir: Optional[str],
    file_name: Optional[str],
) -> pd.DataFrame:
    """
    Measure insertion loss, phase, and other parameters using Luna OVA.

    Parameters
    ----------
    center_wavelength_nm : float
        Centre wavelength in nm
    wavelength_span_nm : float
        Wavelength span in nm
    num_averages : int
        Number of averages for measurement
    edfa_port : str
        COM port for EDFA controller
    edfa_baudrate : int
        Baudrate for EDFA communication
    edfa_output_power_dbm : float
        EDFA output level in dBm
    ova_ip : Optional[str]
        IP address for OVA instrument
    folder_dir : str
        Directory path where CSV file will be saved
    file_name : str
        Name of output CSV file (without .csv extension)

    Returns
    -------
    pd.DataFrame
        DataFrame containing all measurement data

    Raises
    ------
    IOError
        If unable to create output directory or write file
    """

    # Connect to EDFA and perform measurement
    with EDFAController(
        edfa_port,
        baudrate=edfa_baudrate,
    ) as edfa:

        # Get device info for metadata
        device_info = edfa.get_identification()
        print(f"EDFA connected: {device_info}")

        # Configure EDFA
        edfa.set_driving_mode(1, DrivingMode.ALC)
        edfa.set_alc_output_level(1, edfa_output_power_dbm)
        edfa.set_output_active(True)

        print(f"EDFA output active at {edfa_output_power_dbm} dBm")

        try:
            # Connect to OVA and measure
            with LunaOVA(ip=ova_ip) as ova:
                print("Performing OVA measurement...")

                data = ova.measure_full(
                    center_wavelength_nm=center_wavelength_nm,
                    wavelength_range_nm=wavelength_span_nm,
                    num_averages=num_averages,
                )

                print("Measurement complete")

        finally:
            # Always deactivate EDFA output
            edfa.set_output_active(False)
            print("EDFA output deactivated")

    # Create DataFrame from all returned data
    df = pd.DataFrame(data)

    # Create output directory if it doesn't exist
    if folder_dir is not None:
        output_dir = Path(folder_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    if file_name is not None:
        output_path = output_dir / f"{file_name}.csv"
        df.to_csv(output_path, mode="a", index=False)

    print(f"Data saved to: {output_path}")
    print(f"Columns saved: {list(df.columns)}")
    print(f"Number of data points: {len(df)}")

    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate filename with timestamp
    file_name_base = "spectrum_test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{file_name_base}_{timestamp}"

    # Perform measurement with config
    df = measure_spectrum(
        center_wavelength_nm=1550.0,
        wavelength_span_nm=5.0,
        num_averages=5,
        edfa_port="COM6",
        edfa_baudrate=57600,
        edfa_output_power_dbm=13.0,
        ova_ip="130.194.137.122",
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
    fig_filename = f"./measurements/config_based_test_plot_{timestamp}.png"
    fig.savefig(fig_filename, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {fig_filename}")

    plt.close(fig)
