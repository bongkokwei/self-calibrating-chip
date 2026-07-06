"""
Compare the ideal calibration-port spectrum (from the FIR frequency-response
equation) against a measured insertion-loss spectrum.

Usage
-----
    python -m simulation.compare_spectrum config/calibration_config.yaml \
        measurements/run_001/spectrum.csv
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from photonic_fir.calibration.trim_spectrum_to_fsr import trim_spectrum_to_fsr
from photonic_fir.core.config import load_config
from photonic_fir.core.data_structure import ExperimentConfig

from .fir_response import FIRFrequencyResponse


def load_measured_spectrum(config: ExperimentConfig, csv_path: str) -> pd.DataFrame:
    """Load a measured spectrum CSV and trim it to the configured FSR window."""
    df = pd.read_csv(csv_path)
    df_trimmed, _ = trim_spectrum_to_fsr(
        df=df,
        fsr_hz=config.chip.fsr_hz,
        n_fsr=config.calibration.trim_n_fsr,
        freq_col=config.measurement.frequency_col,
        il_col=config.measurement.insertion_loss_col,
    )
    return df_trimmed


def compare_to_measured(
    config: ExperimentConfig,
    measured_csv_path: str,
    include_reference: bool = True,
    output_path: str | None = None,
):
    """Plot ideal vs measured calibration-port insertion-loss spectra.

    Both traces are normalised to their own peak (0 dB) so that filter
    *shape* can be compared independently of absolute calibration offsets
    between the ideal model and the OVA measurement.
    """
    df = load_measured_spectrum(config, measured_csv_path)

    # frequency column is stored in THz (see trim_spectrum_to_fsr)
    frequencies_hz = df[config.measurement.frequency_col].to_numpy() * 1e12
    measured_il_db = df[config.measurement.insertion_loss_col].to_numpy()

    fir = FIRFrequencyResponse.from_config(config, include_reference=include_reference)
    ideal_il_db = fir.magnitude_db(frequencies_hz)

    ideal_il_db = ideal_il_db - np.max(ideal_il_db)
    measured_il_db = measured_il_db - np.max(measured_il_db)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frequencies_hz / 1e9, measured_il_db, label="Measured", alpha=0.8)
    ax.plot(
        frequencies_hz / 1e9,
        ideal_il_db,
        label="Ideal (FIR equation)",
        linestyle="--",
    )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Insertion loss (dB, normalised to peak)")
    ax.set_title("Calibration-port spectrum: ideal vs measured")
    ax.legend()
    ax.grid(alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Path to calibration YAML config")
    parser.add_argument(
        "measured_csv_path", help="Path to measured spectrum CSV (frequency + IL columns)"
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Exclude the reference tap from the ideal response (signal core only)",
    )
    parser.add_argument("--output", default=None, help="Save plot to this path")
    args = parser.parse_args()

    config = load_config(args.config_path)
    compare_to_measured(
        config,
        args.measured_csv_path,
        include_reference=not args.no_reference,
        output_path=args.output,
    )
    plt.show()


if __name__ == "__main__":
    main()
