import logging

logger = logging.getLogger(__name__)

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pathlib import Path

from .style_utils import apply_calibration_style

apply_calibration_style(dark=True)


def plot_insertion_loss(
    df: pd.DataFrame,
    title: str = "Insertion Loss",
    save_dir: str | Path = "./measurements",
    file_name_base: str = "spectrum_test",
    dpi: int = 300,
    figsize: Tuple[float, float] = (10, 8),
    wl_range: Optional[Tuple[float, float]] = None,
    il_range: Optional[Tuple[float, float]] = None,
    phase_range: Optional[Tuple[float, float]] = None,
    plot_phase: bool = True,
    save_fig: bool = False,
    show_plot: bool = True,
) -> str:
    """
    Plot insertion loss and optionally phase from optical spectrum measurement.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'wl_axis', 'IL', and optionally 'LPD'
    title : str
        Base title for the plot
    save_dir : str
        Directory to save the figure
    file_name_base : str
        Base name for saved files
    show_plot : bool
        Whether to display the plot interactively
    dpi : int
        Resolution for saved figure
    figsize : tuple
        Figure size (width, height) in inches
    wl_range : tuple or None
        Wavelength axis limits (min, max) in nm
    il_range : tuple or None
        Insertion loss axis limits (min, max) in dB
    phase_range : tuple or None
        Phase axis limits (min, max) in rad
    plot_phase : bool
        Whether to plot phase subplot (requires 'LPD' column)

    Returns
    -------
    fig_filename : str
        Path to saved figure
    """

    # Quick inspection
    logger.info("\nFirst few rows:")
    logger.info(df.head())
    logger.info(f"\nData shape: {df.shape}")

    # Determine subplot layout
    if plot_phase and "LPD" in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] / 2))
        ax2 = None

    # Plot insertion loss
    ax1.plot(df["wl_axis"], df["IL"])
    ax1.set_ylabel("Insertion Loss (dB)", fontsize=12)

    # Set title with optional timestamp
    plot_title = title
    ax1.set_title(plot_title, fontsize=12)

    # Set axis ranges if specified
    if wl_range is not None:
        ax1.set_xlim(wl_range)
    if il_range is not None:
        ax1.set_ylim(il_range)

    # Plot phase if requested
    if ax2 is not None:
        ax2.plot(df["wl_axis"], df["LPD"])
        ax2.set_xlabel("Wavelength (nm)", fontsize=12)
        ax2.set_ylabel("Phase (rad)", fontsize=12)

        if wl_range is not None:
            ax2.set_xlim(wl_range)
        if phase_range is not None:
            ax2.set_ylim(phase_range)
    else:
        ax1.set_xlabel("Wavelength (nm)", fontsize=12)

    plt.tight_layout()

    # Save figure
    fig_filename = os.path.join(save_dir, f"{file_name_base}.png")

    if save_fig:
        fig.savefig(fig_filename, dpi=dpi, bbox_inches="tight")
        logger.info(f"\nFigure saved to: {fig_filename}")

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig_filename


def plot_impulse_response(
    time_ps: np.ndarray,
    h_time: np.ndarray,
    tap_times_ps: Optional[np.ndarray] = None,
    tap_coeffs: Optional[np.ndarray] = None,
    save_fig: Optional[str] = None,
    show_plot: bool = False,
):
    """
    Plot impulse response magnitude with detected taps.

    Parameters
    ----------
    time_ps : np.ndarray
        Time axis in picoseconds
    h_time : np.ndarray
        Complex impulse response
    tap_times_ps : np.ndarray, optional
        Detected tap positions
    tap_coeffs : np.ndarray, optional
        Detected tap coefficients
    save_fig : str, optional
        File path to save the figure. If None, the figure is not saved.
    show_plot : bool
        Whether to display the plot interactively. Set to False when called
        during calibration to avoid disrupting CalibrationPlotter's interactive mode.
    """
    h_magnitude = np.abs(h_time)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(
        time_ps,
        10 * np.log10(h_magnitude) + 1e-12,
        "b-",
        linewidth=1.5,
        label="Impulse Response",
    )

    if tap_times_ps is not None and tap_coeffs is not None:
        tap_magnitudes = np.abs(tap_coeffs)
        ax.plot(
            tap_times_ps,
            10 * np.log10(tap_magnitudes),
            "ro",
            markersize=8,
            label="Detected Taps",
        )

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.set_title("Impulse Response (Kramers-Kronig Phase Recovery)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=-40)
    ax.set_xlim(left=-(1 / 160e9) * 1e12, right=16 * (1 / 160e9) * 1e12)

    plt.tight_layout()

    if save_fig is not None:
        fig.savefig(save_fig, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
