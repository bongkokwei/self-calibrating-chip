"""
Script to process insertion loss spectra and extract MZI power splitting ratios.

This script reads voltage-swept measurement data, recovers impulse responses
via Kramers-Kronig phase recovery, and extracts power splitting ratios for
specified MZIs in a programmable photonic FIR filter.
"""

import argparse
import re
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from photonic_fir import (
    ExperimentConfig,
    load_config,
    tap_coeffs_to_power_splitting_ratios,
)

from photonic_fir.calibration import measure_and_detect_taps


class PSRResult(NamedTuple):
    """Result from power splitting ratio extraction."""

    psr_db: float
    voltage: float
    filename: str


def get_csv_files(directory: Path) -> list[Path]:
    """
    Retrieve all CSV files from a directory.

    Parameters
    ----------
    directory : Path
        Path to the directory containing measurement files.

    Returns
    -------
    list[Path]
        Sorted list of CSV file paths.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If no CSV files are found in the directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    csv_files = sorted(directory.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in: {directory}")

    return csv_files


def generate_mzi_ids(n_stages: int = 4) -> list[str]:
    """
    Generate all MZI IDs for a binary tree structure.

    For a tree with n_stages, stage k has 2^(k-1) MZIs.

    Parameters
    ----------
    n_stages : int
        Number of stages in the MZI tree (default: 4 for 16-tap filter).

    Returns
    -------
    list[str]
        List of MZI IDs in format "stage-position" (e.g., "1-1", "2-1", "2-2").
    """
    return [
        f"{stage}-{pos}"
        for stage in range(1, n_stages + 1)
        for pos in range(1, 2 ** (stage - 1) + 1)
    ]


def extract_voltage_from_filename(filename: Path) -> float | None:
    """
    Extract voltage value from filename.

    Expected format: contains pattern like "1.23v" or "0.50v".

    Parameters
    ----------
    filename : Path
        Path to the measurement file.

    Returns
    -------
    float | None
        Extracted voltage value, or None if not found.
    """
    match = re.search(r"(\d+\.?\d*)v", filename.name, re.IGNORECASE)
    return float(match.group(1)) if match else None


def extract_psr_from_file(
    filepath: Path,
    config: ExperimentConfig,
    mzi_tree: dict,
    mzi_id: str,
) -> PSRResult | None:
    """
    Extract power splitting ratio for a specific MZI from a measurement file.

    Parameters
    ----------
    filepath : Path
        Path to the CSV measurement file.
    config : ExperimentConfig
        Experiment configuration containing chip and measurement parameters.
    mzi_tree : dict
        MZI tree structure mapping tap indices to MZI IDs.
    mzi_id : str
        Target MZI identifier (e.g., "1-1").

    Returns
    -------
    PSRResult | None
        Named tuple containing (psr_db, voltage, filename), or None if extraction fails.
    """
    voltage = extract_voltage_from_filename(filepath)
    if voltage is None:
        return None

    try:
        df = pd.read_csv(filepath)
    except Exception:
        return None

    df, tap_times, tap_coeffs, _, _ = measure_and_detect_taps(config, df=df)

    tap_coeffs_normalised = np.abs(tap_coeffs) / np.max(np.abs(tap_coeffs))

    # Convert to power splitting ratios
    psr_dict = tap_coeffs_to_power_splitting_ratios(tap_coeffs_normalised, mzi_tree)
    psr_db = psr_dict.get(mzi_id, np.nan)

    return PSRResult(psr_db=psr_db, voltage=voltage, filename=str(filepath))


def process_voltage_scan(
    directory: Path,
    config: ExperimentConfig,
    mzi_id: str,
    n_signal_taps: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process all files in a voltage scan directory.

    Parameters
    ----------
    directory : Path
        Directory containing measurement CSV files.
    config : ExperimentConfig
        Experiment configuration.
    mzi_id : str
        Target MZI identifier.
    n_signal_taps : int
        Number of signal taps in the filter (default: 16).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of (voltages, power_splitting_ratios), sorted by voltage.
    """
    # Build MZI tree structure
    n_stages = int(np.log2(n_signal_taps))
    mzi_ids = generate_mzi_ids(n_stages)
    mzi_tree = config.signal_mzi_tree.tree

    # Get files and process
    csv_files = get_csv_files(directory)
    results: list[PSRResult] = []

    for filepath in csv_files:
        result = extract_psr_from_file(filepath, config, mzi_tree, mzi_id)
        if result is not None:
            results.append(result)

    if not results:
        raise ValueError(f"No valid results extracted from {directory}")

    # Sort by voltage and extract arrays
    results.sort(key=lambda r: r.voltage)
    voltages = np.array([r.voltage for r in results])
    psrs = np.array([r.psr_db for r in results])

    return voltages, psrs


def plot_psr_vs_voltage(
    voltages: np.ndarray,
    psrs: np.ndarray,
    mzi_id: str,
    output_path: Path | None = None,
) -> None:
    """
    Plot power splitting ratio versus voltage squared.

    The x-axis uses V² since heater power (and thus phase shift) is
    proportional to V² for resistive heaters.

    Parameters
    ----------
    voltages : np.ndarray
        Array of voltage values.
    psrs : np.ndarray
        Array of power splitting ratios in dB.
    mzi_id : str
        MZI identifier for plot title.
    output_path : Path | None
        If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(voltages**2, psrs, "o-", markersize=4, linewidth=1)
    ax.set_xlabel(r"Heater Power $\propto V^2$ (V²)")
    ax.set_ylabel("Power Splitting Ratio (dB)")
    ax.set_title(f"MZI {mzi_id}: Power Splitting Ratio vs Heater Power")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)

    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process voltage scan data to extract MZI power splitting ratios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mzi-id",
        type=str,
        default="1-1",
        help="MZI identifier (e.g., '1-1', '2-1')",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("measurements/mzi_1-1"),
        help="Directory containing measurement CSV files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("measurements/experiment_config_shorter_range.yaml"),
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--n-taps",
        type=int,
        default=16,
        help="Number of signal taps in the filter",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (optional)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(str(args.config))

    # Process voltage scan
    voltages, psrs = process_voltage_scan(
        directory=args.data_dir / Path(f"mzi_{args.mzi_id}"),
        config=config,
        mzi_id=args.mzi_id,
        n_signal_taps=args.n_taps,
    )

    # Plot results
    plot_psr_vs_voltage(voltages, psrs, args.mzi_id, args.output)


if __name__ == "__main__":
    main()
