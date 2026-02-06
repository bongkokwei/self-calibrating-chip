"""
MZI Characterisation Plotting Utilities
========================================

Plotting functions for visualising MZI voltage scans and fitted parameters.
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict


def plot_mzi_characterisation(
    voltages: np.ndarray,
    psr_db: np.ndarray,
    mzi_id: str,
    v_2pi: Optional[float] = None,
    phi_init: Optional[float] = None,
    fit_info: Optional[Dict] = None,
    resistance_ohm: Optional[float] = None,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Plot MZI characterisation data with optional fitted model overlay.

    Creates a single clean plot showing:
    - Measured PSR vs V² data points
    - Fitted curve (if fit parameters provided)

    Parameters
    ----------
    voltages : np.ndarray
        Applied voltage values (V)
    psr_db : np.ndarray
        Measured power splitting ratios (dB)
    mzi_id : str
        MZI identifier (e.g., "4-5")
    v_2pi : float, optional
        Fitted voltage for 2π phase shift (V)
    phi_init : float, optional
        Fitted initial phase offset (radians)
    fit_info : Dict, optional
        Fit information from fit_mzi_v2pi_and_phi_init()
    resistance_ohm : float, optional
        Heater resistance (Ω)
    output_dir : str, optional
        Directory to save figure. If None, figure is not saved

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object

    Examples
    --------
    >>> # Plot data only
    >>> fig = plot_mzi_characterisation(voltages, psr_db, mzi_id="4-5")
    >>>
    >>> # Plot with fitted curve
    >>> v_2pi, phi_init, r2, info = fit_mzi_v2pi_and_phi_init(voltages, psr_db, 600)
    >>> fig = plot_mzi_characterisation(
    ...     voltages, psr_db, mzi_id="4-5",
    ...     v_2pi=v_2pi, phi_init=phi_init, fit_info=info,
    ...     output_dir="./results"
    ... )
    """
    voltage_squared = voltages**2

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot measured data
    ax.plot(
        voltage_squared,
        psr_db,
        "o",
        markersize=7,
        markerfacecolor="steelblue",
        markeredgecolor="darkblue",
        markeredgewidth=1.5,
        label="Measured",
        zorder=3,
    )

    # Plot fitted curve if parameters provided
    if (
        fit_info is not None
        and "fitted_psr" in fit_info
        and fit_info["fitted_psr"] is not None
    ):
        psr_fit = fit_info["fitted_psr"]
        ax.plot(
            voltage_squared,
            psr_fit,
            "-",
            color="crimson",
            linewidth=2.5,
            label="Fitted",
            zorder=2,
        )
    elif v_2pi is not None or phi_init is not None:
        # User provided fit parameters but no fitted data - likely fit failed
        logger.info(
            "⚠ Warning: Fit parameters provided but fitted curve unavailable (fit may have failed)"
        )

    # Reference line
    ax.axhline(
        y=0, color="grey", linestyle="--", alpha=0.5, linewidth=1.5, label="50:50 split"
    )

    # Formatting
    ax.set_xlabel("Voltage² (V²)", fontsize=13)
    ax.set_ylabel("Power Splitting Ratio (dB)", fontsize=13)
    ax.set_title(
        f"MZI {mzi_id} Characterisation",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()

    # Save if output directory specified
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mzi_{mzi_id}_characterisation_{timestamp}.png"
        fig.savefig(output_path / filename, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Figure saved: {output_path / filename}")

    return fig
