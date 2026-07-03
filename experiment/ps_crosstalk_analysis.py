"""
ps_crosstalk_analysis.py

Analyse phase-shifter thermal crosstalk: how sweeping one PS voltage affects
the KK-recovered phase of ALL signal taps (9–16).

Mirrors mzi_crosstalk_analysis.py exactly in structure, but the observable
changes from MZI power-splitting ratio (PSR, dB) to tap phase (rad).

Physical rationale
------------------
Each thermo-optic phase shifter (PS) heats the waveguide above it, but the
substrate also conducts heat to neighbouring heaters.  When PS k is driven,
tap k acquires the intended phase shift while every other tap acquires a
parasitic phase offset.  This script maps that cross-coupling matrix by
sweeping each PS independently and recording all 8 tap phases.

The resulting cross-coupling matrix C is:

    Δφ_j = C_jk · ΔP_k      (j = tap index,  k = swept PS index)

where C_kk is the primary sensitivity (rad/W) and C_jk (j≠k) are the
off-diagonal crosstalk terms.  A diagonal-dominant C with small off-diagonal
entries means independent PS control is a good approximation.

Usage
-----
    python ps_crosstalk_analysis.py \\
        --swept-ps 9 \\
        --data-dir measurements/ps_crosstalk_scans \\
        --config measurements/experiment_config_shorter_range.yaml

    # Run for all PSs and save results:
    for ps in 9 10 11 12 13 14 15 16; do
        python ps_crosstalk_analysis.py --swept-ps $ps \\
            --data-dir measurements/ps_crosstalk_scans \\
            --output-fig figures/ps_crosstalk_ps${ps}.png \\
            --output-csv results/ps_crosstalk_ps${ps}.csv
    done

Data directory layout
---------------------
    data_dir/
    └── ps_9/          <-- one sub-dir per swept PS (named ps_{tap_num})
        ├── 0.000v_....csv
        ├── 1.414v_....csv
        ├── 2.000v_....csv
        ...

This matches the output layout of batch_ps_scan.py when save_raw_data=True.

Output DataFrame columns
------------------------
    voltage      : swept voltage (V)
    swept_ps     : tap number of the swept PS (int)
    tap_9 … tap_16 : wrapped tap phase in radians for each signal tap

Cross-coupling matrix
---------------------
The matrix is assembled automatically in main() once all per-PS CSVs are
present alongside the current one.  It is written to ps_crosstalk_matrix.csv
in the same directory as the per-PS CSVs.

Format:
    tap\\swept_ps,9,10,11,...,16
    9,C_99,C_9_10,...
    10,C_10_9,C_10_10,...
    ...

Units: rad/W (slopes converted from rad/V² using heater resistance R).
Diagonal entries are primary sensitivities; off-diagonals are crosstalk.

To load:
    import numpy as np
    data = np.loadtxt("ps_crosstalk_matrix.csv", delimiter=",", skiprows=1)
    C = data[:, 1:]   # drop the tap-number column
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from photonic_fir import ExperimentConfig, load_config
from photonic_fir.calibration import measure_and_detect_taps
from photonic_fir.utils.file_utils import extract_voltage_from_filename


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal_tap_cols(n_signal_taps: int = 8, first_tap: int = 9) -> list[str]:
    """Column names for signal tap phases."""
    return [f"tap_{t}" for t in range(first_tap, first_tap + n_signal_taps)]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyse_ps_crosstalk(
    data_dir: Path,
    config: ExperimentConfig,
    swept_ps_tap: int,
    n_signal_taps: int = 8,
    first_tap: int = 9,
) -> pd.DataFrame:
    """
    Analyse how sweeping one PS voltage affects all signal-tap phases.

    Parameters
    ----------
    data_dir : Path
        Directory containing voltage-scan CSV files for the swept PS.
        Expected layout: data_dir/ps_{swept_ps_tap}/*.csv
        (matching batch_ps_scan.py save_raw_data=True output).
    config : ExperimentConfig
        Experiment configuration.
    swept_ps_tap : int
        Tap number (1-indexed) of the PS being swept.
    n_signal_taps : int
        Number of signal processing taps (default: 8, i.e. taps 9–16).
    first_tap : int
        Tap number of the first signal tap (default: 9).

    Returns
    -------
    pd.DataFrame
        Columns: voltage, swept_ps, tap_9, tap_10, …, tap_16.
        Phases are in radians, wrapped to [−π, π].
    """
    tap_cols = _signal_tap_cols(n_signal_taps, first_tap)
    ps_subdir = data_dir / f"ps_tap{swept_ps_tap}"

    if not ps_subdir.exists():
        # Fall back to flat data_dir if the sub-directory doesn't exist
        ps_subdir = data_dir
        print(
            f"  Sub-directory ps_{swept_ps_tap}/ not found – "
            f"reading CSV files directly from {data_dir}"
        )

    csv_files = sorted(ps_subdir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {ps_subdir}")

    results: list[dict] = []

    for csv_file in csv_files:
        voltage = extract_voltage_from_filename(csv_file.name)
        if voltage is None:
            print(
                f"  Warning: could not parse voltage from '{csv_file.name}', skipping"
            )
            continue

        try:
            df_raw = pd.read_csv(csv_file)

            _df, _tap_times, tap_coeffs, _a, _b = measure_and_detect_taps(
                config, df=df_raw
            )

            # Extract phase for every signal tap
            row: dict = {"voltage": voltage, "swept_ps": swept_ps_tap}
            for tap_num, col in zip(
                range(first_tap, first_tap + n_signal_taps), tap_cols
            ):
                tap_idx = tap_num - 1  # 0-indexed
                if tap_idx < len(tap_coeffs):
                    row[col] = float(np.angle(tap_coeffs[tap_idx]))
                else:
                    row[col] = float("nan")

            results.append(row)

        except Exception as exc:
            print(f"  Error processing {csv_file.name}: {exc}")
            continue

    if not results:
        raise ValueError("No valid measurements were processed — check data directory.")

    df_out = pd.DataFrame(results).sort_values("voltage").reset_index(drop=True)
    return df_out


# ---------------------------------------------------------------------------
# Crosstalk matrix
# ---------------------------------------------------------------------------


def save_crosstalk_matrix(
    per_ps_csv_paths: dict[int, Path],
    resistance_ohm: float,
    output_path: Path,
) -> np.ndarray | None:
    """
    Assemble and save the crosstalk matrix C from per-PS result CSVs.

    Fits dφ/dV² for every (tap, swept-PS) pair, converts to rad/W via R,
    and writes the matrix to a labelled CSV.

    Parameters
    ----------
    per_ps_csv_paths : dict[int, Path]
        Mapping from swept-PS tap number to its results CSV path.
        Only paths that exist are used; missing columns are left as NaN.
    resistance_ohm : float
        Nominal heater resistance (Ω).  Converts rad/V² → rad/W via R.
    output_path : Path
        Destination CSV path for the assembled matrix.

    Returns
    -------
    C : np.ndarray of shape (N, N) or None
        The crosstalk matrix in rad/W, or None if no CSVs could be loaded.
    """
    # Determine tap order from whichever CSVs are present
    available = {ps: p for ps, p in per_ps_csv_paths.items() if p.exists()}
    if not available:
        print("  No per-PS CSVs found — skipping matrix assembly.")
        return None

    # Infer full tap list from the first available CSV
    first_df = pd.read_csv(next(iter(available.values())))
    tap_cols = sorted(
        [c for c in first_df.columns if c.startswith("tap_")],
        key=lambda c: int(c.split("_")[1]),
    )
    tap_nums = [int(c.split("_")[1]) for c in tap_cols]
    N = len(tap_nums)
    tap_index = {t: i for i, t in enumerate(tap_nums)}

    C = np.full((N, N), np.nan)

    for swept_ps, csv_path in available.items():
        if swept_ps not in tap_index:
            print(f"  Warning: swept PS {swept_ps} not in tap list, skipping column.")
            continue
        col_idx = tap_index[swept_ps]

        df = pd.read_csv(csv_path)
        v2 = df["voltage"].values ** 2

        for tap_col in tap_cols:
            row_idx = tap_index[int(tap_col.split("_")[1])]
            phi = np.unwrap(df[tap_col].values)
            valid = ~np.isnan(phi)
            if valid.sum() < 2:
                continue
            slope_v2, _ = np.polyfit(v2[valid], phi[valid], 1)
            C[row_idx, col_idx] = slope_v2 * resistance_ohm  # rad/W

    # Write labelled CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "tap\\swept_ps," + ",".join(str(t) for t in tap_nums)
    lines = [header]
    for row_idx, tap_num in enumerate(tap_nums):
        vals = ",".join(
            f"{C[row_idx, col_idx]:.6f}" if not np.isnan(C[row_idx, col_idx]) else ""
            for col_idx in range(N)
        )
        lines.append(f"{tap_num},{vals}")

    output_path.write_text("\n".join(lines) + "\n")

    n_available = len(available)
    n_total = len(per_ps_csv_paths)
    print(f"\n✓ Crosstalk matrix saved: {output_path}")
    print(f"  Shape: {N}×{N}   Units: rad/W")
    print(f"  Columns populated: {n_available}/{n_total} PSs measured")
    if n_available < n_total:
        missing = sorted(set(per_ps_csv_paths) - set(available))
        print(f"  Missing PSs (NaN columns): {missing}")

    return C


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ps_crosstalk(
    df: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple = (16, 10),
) -> None:
    """
    Plot phase of every signal tap vs swept-PS heater power (V²).

    The swept PS is highlighted in red.  Off-diagonal traces reveal thermal
    crosstalk.  Unwrapped phase is shown so that continuous trends are visible
    even if a trace passes through ±π.

    Parameters
    ----------
    df : pd.DataFrame
        Output from analyse_ps_crosstalk().
    output_path : Path | None
        If given, save figure to this path.
    figsize : tuple
        Figure dimensions in inches.
    """
    swept_ps = int(df["swept_ps"].iloc[0])
    tap_cols = [c for c in df.columns if c.startswith("tap_")]
    n_taps = len(tap_cols)

    v2 = df["voltage"].values ** 2  # heater power proxy (V²)

    # ---- Subplot grid: 2 rows × 4 columns for taps 9–16 ----
    n_cols = 4
    n_rows = (n_taps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes_flat = axes.flatten()

    for idx, col in enumerate(tap_cols):
        ax = axes_flat[idx]
        tap_num = int(col.split("_")[1])
        is_swept = tap_num == swept_ps

        # Unwrap phase for a cleaner trend line
        phi_wrapped = df[col].values
        phi_uw = np.unwrap(phi_wrapped)

        # Estimate linear slope (dφ/dV²) for annotation
        valid = ~np.isnan(phi_uw)
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(v2[valid], phi_uw[valid], 1)
            phi_fit = slope * v2 + intercept
            r2 = 1.0 - np.var(phi_uw[valid] - phi_fit[valid]) / np.var(phi_uw[valid])
        else:
            slope = np.nan
            r2 = np.nan
            phi_fit = np.full_like(v2, np.nan)

        # --- Primary / crosstalk colouring ---
        if is_swept:
            colour = "crimson"
            lw = 2.5
            ms = 6
            title_weight = "bold"
            title_str = f"Tap {tap_num}  ← swept PS"
        else:
            colour = "steelblue"
            lw = 1.5
            ms = 4
            title_weight = "normal"
            title_str = f"Tap {tap_num}"

        # Unwrapped phase (dots + line)
        ax.plot(
            v2,
            phi_uw,
            "o-",
            color=colour,
            markersize=ms,
            linewidth=lw,
            alpha=0.85,
            label="Measured (unwrapped)",
        )

        # Linear fit overlay
        if not np.isnan(slope):
            ax.plot(
                v2,
                phi_fit,
                "--",
                color="orange",
                linewidth=1.2,
                alpha=0.8,
                label=f"Fit: {slope*1e3:.2f} mrad/V²\nR²={r2:.3f}",
            )

        ax.set_xlabel(r"$V^2_{\mathrm{swept}}$  (V²)", fontsize=9)
        ax.set_ylabel("Phase (rad)", fontsize=9)
        ax.set_title(title_str, fontweight=title_weight, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_taps, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        f"Phase-Shifter Crosstalk Analysis — Swept PS: Tap {swept_ps}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Crosstalk summary table
# ---------------------------------------------------------------------------


def print_crosstalk_summary(df: pd.DataFrame, resistance_ohm: float = 600.0) -> None:
    """
    Print a summary of primary sensitivity and crosstalk coefficients.

    For each tap, fits dφ/dV² and converts to dφ/dW using R:
        dP/dV² = 1/R  →  dφ/dW = R · (dφ/dV²)

    Parameters
    ----------
    df : pd.DataFrame
        Output from analyse_ps_crosstalk().
    resistance_ohm : float
        Nominal heater resistance (Ω).  Used only for unit conversion.
    """
    swept_ps = int(df["swept_ps"].iloc[0])
    tap_cols = [c for c in df.columns if c.startswith("tap_")]
    v2 = df["voltage"].values ** 2

    print(f"\n{'='*65}")
    print(f"  Crosstalk Summary — Swept PS: Tap {swept_ps}")
    print(f"  Heater resistance (nominal): {resistance_ohm:.0f} Ω")
    print(f"{'='*65}")
    print(
        f"  {'Tap':>5}  {'dφ/dV² (mrad/V²)':>18}  "
        f"{'dφ/dW (rad/W)':>14}  {'R²':>7}  {'Type':>12}"
    )
    print(f"  {'-'*60}")

    for col in tap_cols:
        tap_num = int(col.split("_")[1])
        phi_uw = np.unwrap(df[col].values)
        valid = ~np.isnan(phi_uw)

        if valid.sum() >= 2:
            slope, _ = np.polyfit(v2[valid], phi_uw[valid], 1)
            phi_fit = slope * v2[valid] + _
            r2 = 1.0 - (np.var(phi_uw[valid] - phi_fit) / np.var(phi_uw[valid]))
            slope_per_watt = slope * resistance_ohm  # dφ/dW
        else:
            slope = np.nan
            slope_per_watt = np.nan
            r2 = np.nan

        tap_type = "PRIMARY" if tap_num == swept_ps else "crosstalk"
        print(
            f"  {tap_num:>5}  {slope*1e3:>18.3f}  "
            f"{slope_per_watt:>14.4f}  {r2:>7.4f}  {tap_type:>12}"
        )

    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse phase-shifter thermal crosstalk by processing voltage-scan "
            "CSV files and plotting tap phase vs swept-PS heater power."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--swept-ps",
        type=int,
        required=True,
        metavar="TAP_NUM",
        help="Tap number of the PS being swept (e.g. 9, 10, …, 16)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Root directory containing voltage-scan CSV files. "
            "Expected sub-directory: ps_{tap_num}/"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/calibration_config.yaml"),
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--n-signal-taps",
        type=int,
        default=8,
        help="Number of signal processing taps",
    )
    parser.add_argument(
        "--first-tap",
        type=int,
        default=9,
        help="Tap number of the first signal tap (1-indexed)",
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=None,
        help="Output path for the figure (optional)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output path for the per-PS results CSV (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"PS Crosstalk Analysis — Swept PS: Tap {args.swept_ps}")
    print(f"Data directory : {args.data_dir}")
    print(f"Config         : {args.config}")

    config = load_config(str(args.config))

    df = analyse_ps_crosstalk(
        data_dir=args.data_dir,
        config=config,
        swept_ps_tap=args.swept_ps,
        n_signal_taps=args.n_signal_taps,
        first_tap=args.first_tap,
    )

    print(f"\nProcessed {len(df)} voltage points")
    print(f"Voltage range: {df['voltage'].min():.3f} – {df['voltage'].max():.3f} V")
    print(f"(V² range: {df['voltage'].min()**2:.2f} – {df['voltage'].max()**2:.2f} V²)")

    print_crosstalk_summary(df, resistance_ohm=config.chip.heater_resistance_ohm)

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Data saved to {args.output_csv}")

        # # After saving, attempt to assemble the full crosstalk matrix.
        # # Assumes sibling CSVs are named ps_crosstalk_ps{tap}.csv in the same directory.
        # output_dir = args.output_csv.parent
        # all_ps_taps = list(range(args.first_tap, args.first_tap + args.n_signal_taps))
        # per_ps_csvs = {
        #     tap: output_dir / f"ps_crosstalk_ps{tap}.csv" for tap in all_ps_taps
        # }
        # matrix_path = output_dir / "ps_crosstalk_matrix.csv"
        # save_crosstalk_matrix(
        #     per_ps_csv_paths=per_ps_csvs,
        #     resistance_ohm=config.chip.heater_resistance_ohm,
        #     output_path=matrix_path,
        # )

    plot_ps_crosstalk(df, output_path=args.output_fig)


if __name__ == "__main__":
    main()
