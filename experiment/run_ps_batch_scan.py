"""
run_ps_crosstalk_batch.py

Runs ps_crosstalk_analysis.py sequentially for all signal PS taps (9–16).
Equivalent to the bash loop:

    for ps in 9 10 11 12 13 14 15 16; do
        python ps_crosstalk_analysis.py --swept-ps $ps ...
    done

Directory layout (all derived from --data-dir):
    <data-dir>/
    ├── ps_crosstalk_csv/   <- per-PS result CSVs + final matrix
    └── figures/            <- per-PS crosstalk figures

Usage
-----
    python run_ps_crosstalk_batch.py --data-dir measurements/ps_crosstalk_scans
"""

import argparse
import subprocess
import sys
from pathlib import Path

from photonic_fir import load_config
from ps_crosstalk_analysis import save_crosstalk_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run ps_crosstalk_analysis.py for all signal PS taps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Root directory for this crosstalk run. "
            "CSVs are written to <data-dir>/ps_crosstalk_csv/; "
            "figures are written to <data-dir>/figures/."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/calibration_config.yaml"),
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--first-tap",
        type=int,
        default=9,
        help="Tap number of the first signal PS (1-indexed)",
    )
    parser.add_argument(
        "--n-signal-taps",
        type=int,
        default=8,
        help="Number of signal PS taps to scan",
    )
    parser.add_argument(
        "--analysis-script",
        type=Path,
        default=Path("experiment/ps_crosstalk_analysis.py"),
        help="Path to ps_crosstalk_analysis.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(str(args.config))

    output_dir = args.data_dir / "ps_crosstalk_csv"
    fig_dir = args.data_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ps_taps = list(range(args.first_tap, args.first_tap + args.n_signal_taps))

    print("Batch PS crosstalk analysis")
    print(f"  PS taps    : {ps_taps}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Fig dir    : {fig_dir}")
    print(f"  Config     : {args.config}")
    print()

    failed = []

    for i, tap in enumerate(ps_taps):
        output_csv = output_dir / f"ps_crosstalk_ps{tap}.csv"
        output_fig = fig_dir / f"ps_crosstalk_ps{tap}.png"

        cmd = [
            sys.executable,
            str(args.analysis_script),
            "--swept-ps",
            str(tap),
            "--data-dir",
            str(args.data_dir),
            "--config",
            str(args.config),
            "--first-tap",
            str(args.first_tap),
            "--n-signal-taps",
            str(args.n_signal_taps),
            "--output-csv",
            str(output_csv),
            "--output-fig",
            str(output_fig),
        ]

        print(f"[{i+1}/{len(ps_taps)}] Swept PS: tap {tap}")
        print(f"  {' '.join(cmd)}\n")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"  WARNING: FAILED (exit code {result.returncode}) -- continuing\n")
            failed.append(tap)
        else:
            print(f"  Done\n")

    print("=" * 60)
    print(
        f"Batch complete.  {len(ps_taps) - len(failed)}/{len(ps_taps)} PSs succeeded."
    )

    if failed:
        print(f"Failed PSs: {failed}")
        print("Skipping matrix assembly — not all CSVs present.")
        return

    per_ps_csvs = {tap: output_dir / f"ps_crosstalk_ps{tap}.csv" for tap in ps_taps}
    missing = [tap for tap, p in per_ps_csvs.items() if not p.exists()]
    if missing:
        print(f"Skipping matrix assembly — missing CSVs for PSs: {missing}")
        return

    save_crosstalk_matrix(
        per_ps_csv_paths=per_ps_csvs,
        resistance_ohm=config.chip.heater_resistance_ohm,
        output_path=output_dir / "ps_crosstalk_matrix.csv",
    )


if __name__ == "__main__":
    main()
