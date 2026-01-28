"""
Analyse MZI crosstalk: how sweeping one MZI's voltage affects all MZI PSRs.

This script processes voltage scan measurements to show how changing the voltage
on one MZI affects the power splitting ratios of all MZIs in the photonic circuit.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from photonic_fir import (
    ExperimentConfig,
    build_mzi_tree_structure,
    detect_taps,
    load_config,
    recover_impulse_response_from_df,
    tap_coeffs_to_power_splitting_ratios,
)


def generate_mzi_ids(n_stages: int = 4) -> list[str]:
    """Generate all MZI IDs for a binary tree structure."""
    return [
        f"{stage}-{pos}"
        for stage in range(1, n_stages + 1)
        for pos in range(1, 2 ** (stage - 1) + 1)
    ]


def extract_voltage_from_filename(filename: str) -> float | None:
    """Extract voltage value from filename (e.g., '1.23v' -> 1.23)."""
    import re

    match = re.search(r"(\d+\.?\d*)v", filename, re.IGNORECASE)
    return float(match.group(1)) if match else None


def analyse_mzi_crosstalk(
    data_dir: Path,
    config: ExperimentConfig,
    swept_mzi_id: str,
    n_signal_taps: int = 16,
) -> pd.DataFrame:
    """
    Analyse how sweeping one MZI affects all MZI PSRs.

    Parameters
    ----------
    data_dir : Path
        Directory containing voltage scan CSV files.
    config : ExperimentConfig
        Experiment configuration.
    swept_mzi_id : str
        ID of the MZI being swept (for labelling).
    n_signal_taps : int
        Number of signal taps (default: 16).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: voltage, swept_mzi, and one column per MZI.
    """
    # Build MZI tree
    n_stages = int(np.log2(n_signal_taps))
    mzi_ids = generate_mzi_ids(n_stages)
    mzi_tree = build_mzi_tree_structure(n_signal_taps=n_signal_taps, mzi_ids=mzi_ids)

    # Get all CSV files
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    results = []

    for csv_file in csv_files:
        # Extract voltage
        voltage = extract_voltage_from_filename(csv_file.name)
        if voltage is None:
            print(f"Warning: Could not extract voltage from {csv_file.name}, skipping")
            continue

        try:
            # Load spectrum
            df = pd.read_csv(csv_file)

            # Recover impulse response
            time_ps, h_time = recover_impulse_response_from_df(
                df=df,
                fsr_hz=config.chip.fsr_hz,
                wavelength_col=config.measurement.wavelength_col,
                freq_col=config.measurement.frequency_col,
                insertion_loss_col=config.measurement.insertion_loss_col,
            )

            # Detect taps
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

            # Normalise tap coefficients
            tap_coeffs_norm = np.abs(tap_coeffs) / np.max(np.abs(tap_coeffs))

            # Extract PSR for all MZIs
            psr_dict = tap_coeffs_to_power_splitting_ratios(tap_coeffs_norm, mzi_tree)

            # Store results
            result = {"voltage": voltage, "swept_mzi": swept_mzi_id}
            result.update({mzi_id: psr_dict.get(mzi_id, np.nan) for mzi_id in mzi_ids})
            results.append(result)

        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue

    if not results:
        raise ValueError("No valid measurements processed")

    # Create DataFrame and sort by voltage
    df_results = pd.DataFrame(results).sort_values("voltage").reset_index(drop=True)

    return df_results


def plot_mzi_crosstalk(
    df: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple = (14, 10),
) -> None:
    """
    Plot PSR of all MZIs vs swept voltage.

    Parameters
    ----------
    df : pd.DataFrame
        Results from analyse_mzi_crosstalk().
    output_path : Path | None
        Save path for figure (optional).
    figsize : tuple
        Figure size in inches.
    """
    swept_mzi = df["swept_mzi"].iloc[0]
    mzi_cols = [col for col in df.columns if col not in ["voltage", "swept_mzi"]]
    n_mzis = len(mzi_cols)

    # Create subplot grid
    n_cols = 4
    n_rows = (n_mzis + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_mzis > 1 else [axes]

    for idx, mzi_id in enumerate(mzi_cols):
        ax = axes[idx]

        # Highlight the swept MZI
        if mzi_id == swept_mzi:
            ax.plot(
                df["voltage"] ** 2,
                df[mzi_id],
                "o-",
                markersize=6,
                linewidth=2.5,
                color="red",
                alpha=0.8,
            )
            title = f"MZI {mzi_id} (swept)"
            title_weight = "bold"
        else:
            ax.plot(
                df["voltage"] ** 2,
                df[mzi_id],
                "o-",
                markersize=4,
                linewidth=1.5,
                alpha=0.7,
            )
            title = f"MZI {mzi_id}"
            title_weight = "normal"

        ax.set_xlabel(r"$V^2$ (V²)", fontsize=10)
        ax.set_ylabel("PSR (dB)", fontsize=10)
        ax.set_title(title, fontweight=title_weight, fontsize=11)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_mzis, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"MZI Crosstalk Analysis: Voltage Sweep on MZI {swept_mzi}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyse MZI crosstalk from voltage scan measurements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--swept-mzi",
        type=str,
        required=True,
        help="ID of the MZI being swept (e.g., '1-1')",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing voltage scan CSV files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("measurements/experiment_config_shorter_range.yaml"),
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--n-taps",
        type=int,
        default=16,
        help="Number of signal taps in the filter",
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=None,
        help="Output path for figure (optional)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output path for CSV data (optional)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print(f"Analysing MZI {args.swept_mzi} crosstalk...")
    print(f"Data directory: {args.data_dir}")

    # Load configuration
    config = load_config(str(args.config))

    # Analyse crosstalk
    df_results = analyse_mzi_crosstalk(
        data_dir=args.data_dir / Path(f"mzi_{args.swept_mzi}"),
        config=config,
        swept_mzi_id=args.swept_mzi,
        n_signal_taps=args.n_taps,
    )

    print(f"\nProcessed {len(df_results)} voltage points")
    print(
        f"Voltage range: {df_results['voltage'].min():.3f} - {df_results['voltage'].max():.3f} V"
    )

    # Save CSV if requested
    if args.output_csv:
        df_results.to_csv(args.output_csv, index=False)
        print(f"Saved data to {args.output_csv}")

    # Plot results
    plot_mzi_crosstalk(df_results, args.output_fig)


if __name__ == "__main__":
    main()
