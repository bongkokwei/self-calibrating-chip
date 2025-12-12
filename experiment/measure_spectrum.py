import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fiberlabs_edfa import EDFAController, DrivingMode
from luna_ova import LunaOVA


def measure_and_save_spectrum(
    folder_dir: str,
    file_name: str,
    edfa_port: str = "COM6",
    edfa_output_level_dbm: float = 13.0,
    ova_ip: str = "130.194.137.122",
    center_wavelength_nm: float = 1550,
    wavelength_range_nm: float = 4,
    num_averages: int = 1,
) -> pd.DataFrame:
    """
    Measure insertion loss, phase, and other parameters vs wavelength using Luna OVA.

    Saves all data returned by measure_full() to a CSV file with metadata in the header.

    Parameters
    ----------
    folder_dir : str
        Directory path where the CSV file will be saved
    file_name : str
        Name of the output CSV file (without .csv extension)
    edfa_port : str, optional
        COM port for EDFA controller (default: "COM6")
    edfa_output_level_dbm : float, optional
        EDFA output level in dBm (default: 13.0)
    ova_ip : str, optional
        IP address of Luna OVA instrument (default: "130.194.137.122")
    center_wavelength_nm : float, optional
        Centre wavelength in nanometres (default: 1550)
    wavelength_range_nm : float, optional
        Wavelength scan range in nanometres (default: 4)
    num_averages : int, optional
        Number of averages for measurement (default: 1)

    Returns
    -------
    pd.DataFrame
        DataFrame containing all measurement data from measure_full()

    Raises
    ------
    IOError
        If unable to create output directory or write file
    """

    # Create output directory if it doesn't exist
    output_dir = Path(folder_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to EDFA and perform measurement
    with EDFAController(edfa_port, baudrate=57600) as edfa:

        # Get device info for metadata
        device_info = edfa.get_identification()
        print(f"EDFA connected: {device_info}")

        # Configure EDFA
        edfa.set_driving_mode(1, DrivingMode.ALC)
        edfa.set_alc_output_level(1, edfa_output_level_dbm)
        edfa.set_output_active(True)

        print(f"EDFA output active at {edfa_output_level_dbm} dBm")

        try:
            # Connect to OVA and measure
            with LunaOVA(ip=ova_ip) as ova:
                print("Performing OVA measurement...")

                data = ova.measure_full(
                    center_wavelength_nm=center_wavelength_nm,
                    wavelength_range_nm=wavelength_range_nm,
                    num_averages=num_averages,
                )

                print("Measurement complete")

        finally:
            # Always deactivate EDFA output
            edfa.set_output_active(False)
            print("EDFA output deactivated")

    # Create DataFrame from all returned data
    df = pd.DataFrame(data)

    # Prepare metadata
    metadata = {
        "measurement_timestamp": datetime.now().isoformat(),
        "center_wavelength_nm": center_wavelength_nm,
        "wavelength_range_nm": wavelength_range_nm,
        "num_averages": num_averages,
        "edfa_output_level_dbm": edfa_output_level_dbm,
        "edfa_port": edfa_port,
        "edfa_device_info": device_info,
        "ova_ip": ova_ip,
    }

    # Save to CSV with metadata in header
    output_path = output_dir / f"{file_name}.csv"

    with open(output_path, "w") as f:
        # Write metadata as comment lines
        f.write("# Luna OVA Measurement Data\n")
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        f.write("#\n")

    # Append DataFrame
    df.to_csv(output_path, mode="a", index=False)

    print(f"Data saved to: {output_path}")
    print(f"Columns saved: {list(df.columns)}")
    print(f"Number of data points: {len(df)}")

    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = measure_and_save_spectrum(
        folder_dir="./measurements",
        file_name="spectrum_test_20241212",
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
    ax1.set_title("Luna OVA Measurement Results", fontsize=12)

    # Plot phase
    ax2.plot(df["wl_axis"], df["LPD"], "r-", linewidth=1.5)
    ax2.set_xlabel("Wavelength (nm)", fontsize=12)
    ax2.set_ylabel("Phase (rad)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = Path("./measurements") / "spectrum_test_20241212_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")

    plt.show()
