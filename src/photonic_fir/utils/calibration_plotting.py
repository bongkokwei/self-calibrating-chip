"""
Live plotting utilities for calibration loop.

Uses in-place line updates (set_data) instead of clearing and redrawing axes
each iteration. This keeps the GUI event loop responsive so the plot window
can be moved and resized in real time.
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Union

from ..core import IterationData
from .style_utils import apply_calibration_style

apply_calibration_style(dark=True)


class CalibrationPlotter:
    """Live plotting for calibration convergence.

    Performance notes
    -----------------
    All static axis furniture (labels, titles, grids, legends) is created once
    in ``__init__`` / ``add_mzi_plot``.  The ``update`` and ``update_mzi_plot``
    methods only call ``Line2D.set_data`` followed by ``relim`` /
    ``autoscale_view`` / ``draw_idle`` / ``flush_events``, avoiding the
    expensive full-canvas redraw that ``ax.clear()`` + ``fig.canvas.draw()``
    would cause.
    """

    # ------------------------------------------------------------------ #
    #  Colour palette                                                      #
    # ------------------------------------------------------------------ #
    _RMS_AMP_COLOUR = "#2E86AB"
    _RMS_PHASE_COLOUR = "#A23B72"
    _TAP_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def __init__(self, num_taps: int, num_mzis: int):
        """
        Initialise plotter with figure and axes.

        Parameters
        ----------
        num_taps : int
            Number of signal processing taps.
        num_mzis : int
            Number of MZIs in the routing tree.
        """
        self.num_taps = num_taps
        self.num_mzis = num_mzis

        # ----- History storage -----
        self.iterations: List[int] = []
        self.rms_amp_history: List[float] = []
        self.rms_phase_history: List[float] = []
        self.tap_amp_errors_history: List[np.ndarray] = []
        self.tap_phase_errors_history: List[np.ndarray] = []
        self.mzi_psr_errors_history: List[np.ndarray] = []

        # ----- Create figure and static axis furniture -----
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle("Calibration Convergence", fontsize=14, fontweight="bold")

        # -- Axes [0, 0]: RMS amplitude error --
        ax1 = self.axes[0, 0]
        (self._line_rms_amp,) = ax1.plot(
            [], [], "o-", color=self._RMS_AMP_COLOUR, linewidth=2, markersize=6
        )
        ax1.set_xlabel("Iteration", fontsize=11)
        ax1.set_ylabel("RMS Amplitude Error (dB)", fontsize=11)
        ax1.set_title("RMS Amplitude Error Convergence", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # -- Axes [0, 1]: RMS phase error --
        ax2 = self.axes[0, 1]
        (self._line_rms_phase,) = ax2.plot(
            [], [], "o-", color=self._RMS_PHASE_COLOUR, linewidth=2, markersize=6
        )
        ax2.set_xlabel("Iteration", fontsize=11)
        ax2.set_ylabel("RMS Phase Error (rad)", fontsize=11)
        ax2.set_title("RMS Phase Error Convergence", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # -- Axes [1, 0]: Individual tap amplitude errors --
        ax3 = self.axes[1, 0]
        self._lines_tap_amp: List[plt.Line2D] = []
        for i in range(num_taps):
            colour = self._TAP_COLOURS[i % len(self._TAP_COLOURS)]
            (line,) = ax3.plot(
                [],
                [],
                "o-",
                label=f"Tap {i + 1}",
                alpha=0.7,
                markersize=4,
                color=colour,
            )
            self._lines_tap_amp.append(line)
        ax3.set_xlabel("Iteration", fontsize=11)
        ax3.set_ylabel("Amplitude Error (dB)", fontsize=11)
        ax3.set_title("Individual Tap Amplitude Errors", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best", fontsize=8, ncol=2)

        # -- Axes [1, 1]: Individual tap phase errors --
        ax4 = self.axes[1, 1]
        self._lines_tap_phase: List[plt.Line2D] = []
        for i in range(num_taps):
            colour = self._TAP_COLOURS[i % len(self._TAP_COLOURS)]
            (line,) = ax4.plot(
                [],
                [],
                "o-",
                label=f"Tap {i + 1}",
                alpha=0.7,
                markersize=4,
                color=colour,
            )
            self._lines_tap_phase.append(line)
        ax4.set_xlabel("Iteration", fontsize=11)
        ax4.set_ylabel("Phase Error (rad)", fontsize=11)
        ax4.set_title("Individual Tap Phase Errors", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="best", fontsize=8, ncol=2)

        # Initial layout pass
        self.fig.tight_layout()

        # Turn on interactive mode and show
        plt.ion()
        self.fig.show()

    # ------------------------------------------------------------------ #
    #  Main convergence update                                             #
    # ------------------------------------------------------------------ #
    def update(self, iter_data: IterationData):
        """
        Update plots with new iteration data.

        Only the line data is changed; static axis furniture (labels, titles,
        grids, legends) is left untouched for performance.

        Parameters
        ----------
        iter_data : IterationData
            Data from current calibration iteration.
        """
        # ----- Store data -----
        self.iterations.append(iter_data.iteration)
        self.rms_amp_history.append(iter_data.rms_amplitude_error_db)
        self.rms_phase_history.append(iter_data.rms_phase_error_rad)
        self.tap_amp_errors_history.append(iter_data.amplitude_errors_db.copy())
        self.tap_phase_errors_history.append(iter_data.phase_errors_rad.copy())

        # Convert MZI errors dict → array
        if isinstance(iter_data.mzi_psr_errors_db, dict):
            sorted_mzis = sorted(iter_data.mzi_psr_errors_db.keys())
            mzi_errors_array = np.array(
                [iter_data.mzi_psr_errors_db[mzi_id] for mzi_id in sorted_mzis]
            )
            self.mzi_psr_errors_history.append(mzi_errors_array)
            if not hasattr(self, "mzi_ids"):
                self.mzi_ids = sorted_mzis
        else:
            self.mzi_psr_errors_history.append(iter_data.mzi_psr_errors_db.copy())

        iters = self.iterations  # shorthand

        # ----- Update RMS amplitude line -----
        self._line_rms_amp.set_data(iters, self.rms_amp_history)
        ax1 = self.axes[0, 0]
        ax1.relim()
        ax1.autoscale_view()

        # ----- Update RMS phase line -----
        self._line_rms_phase.set_data(iters, self.rms_phase_history)
        ax2 = self.axes[0, 1]
        ax2.relim()
        ax2.autoscale_view()

        # ----- Update per-tap amplitude error lines -----
        tap_amp_arr = np.array(self.tap_amp_errors_history)  # (n_iter, n_taps)
        ax3 = self.axes[1, 0]
        for i, line in enumerate(self._lines_tap_amp):
            line.set_data(iters, tap_amp_arr[:, i])
        ax3.relim()
        ax3.autoscale_view()

        # ----- Update per-tap phase error lines -----
        tap_phase_arr = np.array(self.tap_phase_errors_history)
        ax4 = self.axes[1, 1]
        for i, line in enumerate(self._lines_tap_phase):
            line.set_data(iters, tap_phase_arr[:, i])
        ax4.relim()
        ax4.autoscale_view()

        # ----- Non-blocking redraw -----
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ------------------------------------------------------------------ #
    #  MZI PSR error figure                                                #
    # ------------------------------------------------------------------ #
    def add_mzi_plot(self):
        """
        Add a separate figure for MZI PSR errors.

        Call this *before* the first ``update_mzi_plot()``.  MZI line objects
        are created lazily on the first call to ``update_mzi_plot`` because the
        number of MZIs (and their IDs) may not be known until the first
        ``IterationData`` arrives.
        """
        self.fig_mzi, self.ax_mzi = plt.subplots(figsize=(10, 4))
        self.fig_mzi.suptitle(
            "MZI Power Splitting Ratio Errors", fontsize=14, fontweight="bold"
        )

        # Static axis furniture
        self.ax_mzi.set_xlabel("Iteration", fontsize=11)
        self.ax_mzi.set_ylabel("PSR Error (dB)", fontsize=11)
        self.ax_mzi.grid(True, alpha=0.3)

        # Line objects created lazily in update_mzi_plot
        self._lines_mzi: Optional[List[plt.Line2D]] = None

        self.fig_mzi.tight_layout()
        plt.ion()
        self.fig_mzi.show()

    def update_mzi_plot(self):
        """Update MZI error plot if it exists.

        On the first call the line objects and legend are created (since the
        MZI IDs are not known until data arrives).  Subsequent calls only
        update the line data.
        """
        if not hasattr(self, "ax_mzi"):
            return

        if not self.mzi_psr_errors_history:
            return

        mzi_psr_errors = np.array(self.mzi_psr_errors_history)
        n_mzis = mzi_psr_errors.shape[1]

        # Lazy initialisation of line objects on first data arrival
        if self._lines_mzi is None:
            labels = (
                self.mzi_ids
                if hasattr(self, "mzi_ids")
                else [f"MZI {i + 1}" for i in range(n_mzis)]
            )
            self._lines_mzi = []
            for i, label in enumerate(labels):
                colour = self._TAP_COLOURS[i % len(self._TAP_COLOURS)]
                (line,) = self.ax_mzi.plot(
                    [],
                    [],
                    "o-",
                    label=label,
                    alpha=0.7,
                    markersize=4,
                    color=colour,
                )
                self._lines_mzi.append(line)
            self.ax_mzi.legend(loc="best", fontsize=8, ncol=3)

        # Update line data
        for i, line in enumerate(self._lines_mzi):
            line.set_data(self.iterations, mzi_psr_errors[:, i])

        self.ax_mzi.relim()
        self.ax_mzi.autoscale_view()

        # Non-blocking redraw
        self.fig_mzi.canvas.draw_idle()
        self.fig_mzi.canvas.flush_events()

    # ------------------------------------------------------------------ #
    #  Save / close                                                        #
    # ------------------------------------------------------------------ #
    def save_plots(self, output_dir: str):
        """
        Save final plots to output directory.

        Parameters
        ----------
        output_dir : str
            Directory to save plots.
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Force a full render before saving so the file is up to date
        self.fig.canvas.draw()
        self.fig.savefig(
            output_path / "calibration_convergence.png", dpi=300, bbox_inches="tight"
        )

        if hasattr(self, "fig_mzi"):
            self.fig_mzi.canvas.draw()
            self.fig_mzi.savefig(
                output_path / "mzi_psr_errors.png", dpi=300, bbox_inches="tight"
            )

        logger.info(f"Plots saved to {output_dir}")

    def close(self):
        """Close all plot windows."""
        plt.ioff()
        plt.close("all")


# ====================================================================== #
#  Post-processing (unchanged)                                             #
# ====================================================================== #
def plot_calibration_errors(
    iterations: List[IterationData], output_dir: Optional[str] = None
):
    """
    Plot final calibration results (post-processing version).

    Parameters
    ----------
    iterations : List[IterationData]
        List of all iteration data.
    output_dir : Optional[str]
        If provided, save plots to this directory.
    """
    if not iterations:
        logger.info("No iteration data to plot")
        return

    # Extract data
    iter_nums = [d.iteration for d in iterations]
    rms_amp = [d.rms_amplitude_error_db for d in iterations]
    rms_phase = [d.rms_phase_error_rad for d in iterations]

    num_taps = len(iterations[0].amplitude_errors_db)
    tap_amp_errors = np.array([d.amplitude_errors_db for d in iterations])
    tap_phase_errors = np.array([d.phase_errors_rad for d in iterations])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Calibration Results", fontsize=14, fontweight="bold")

    # Plot 1: RMS amplitude error
    axes[0, 0].plot(iter_nums, rms_amp, "o-", color="#2E86AB", linewidth=2)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("RMS Amplitude Error (dB)")
    axes[0, 0].set_title("RMS Amplitude Error Convergence")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: RMS phase error
    axes[0, 1].plot(iter_nums, rms_phase, "o-", color="#A23B72", linewidth=2)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("RMS Phase Error (rad)")
    axes[0, 1].set_title("RMS Phase Error Convergence")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Individual tap amplitude errors
    for i in range(num_taps):
        axes[1, 0].plot(
            iter_nums, tap_amp_errors[:, i], "o-", label=f"Tap {i+1}", alpha=0.7
        )
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Amplitude Error (dB)")
    axes[1, 0].set_title("Individual Tap Amplitude Errors")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="best", fontsize=8)

    # Plot 4: Individual tap phase errors
    for i in range(num_taps):
        axes[1, 1].plot(
            iter_nums, tap_phase_errors[:, i], "o-", label=f"Tap {i+1}", alpha=0.7
        )
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Phase Error (rad)")
    axes[1, 1].set_title("Individual Tap Phase Errors")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="best", fontsize=8)

    plt.tight_layout()

    if output_dir:
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path / "calibration_results.png", dpi=300, bbox_inches="tight"
        )
        logger.info(f"Plot saved to {output_dir}")

    plt.show()
