"""
Live plotting utilities for calibration loop.
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Union

from ..core import IterationData


class CalibrationPlotter:
    """Live plotting for calibration convergence."""

    def __init__(self, num_taps: int, num_mzis: int):
        """
        Initialise plotter with figure and axes.

        Parameters
        ----------
        num_taps : int
            Number of signal processing taps
        num_mzis : int
            Number of MZIs in the routing tree
        """
        self.num_taps = num_taps
        self.num_mzis = num_mzis

        # Storage for history
        self.iterations = []
        self.rms_amp_history = []
        self.rms_phase_history = []
        self.tap_amp_errors_history = []
        self.tap_phase_errors_history = []
        self.mzi_psr_errors_history = []

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle("Calibration Convergence", fontsize=14, fontweight="bold")

        # Turn on interactive mode
        plt.ion()
        plt.show()

    def update(self, iter_data: IterationData):
        """
        Update plots with new iteration data.

        Parameters
        ----------
        iter_data : IterationData
            Data from current calibration iteration
        """
        # Store data
        self.iterations.append(iter_data.iteration)
        self.rms_amp_history.append(iter_data.rms_amplitude_error_db)
        self.rms_phase_history.append(iter_data.rms_phase_error_rad)
        self.tap_amp_errors_history.append(iter_data.amplitude_errors_db.copy())
        self.tap_phase_errors_history.append(iter_data.phase_errors_rad.copy())

        # Convert MZI errors dict to array for plotting
        if isinstance(iter_data.mzi_psr_errors_db, dict):
            # Sort by MZI ID to maintain consistent ordering
            sorted_mzis = sorted(iter_data.mzi_psr_errors_db.keys())
            mzi_errors_array = np.array(
                [iter_data.mzi_psr_errors_db[mzi_id] for mzi_id in sorted_mzis]
            )
            self.mzi_psr_errors_history.append(mzi_errors_array)
            # Store MZI IDs on first iteration
            if not hasattr(self, "mzi_ids"):
                self.mzi_ids = sorted_mzis
        else:
            # Fallback for array input
            self.mzi_psr_errors_history.append(iter_data.mzi_psr_errors_db.copy())

        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Plot 1: RMS amplitude error convergence
        ax1 = self.axes[0, 0]
        ax1.plot(
            self.iterations,
            self.rms_amp_history,
            "o-",
            color="#2E86AB",
            linewidth=2,
            markersize=6,
        )
        ax1.set_xlabel("Iteration", fontsize=11)
        ax1.set_ylabel("RMS Amplitude Error (dB)", fontsize=11)
        ax1.set_title("RMS Amplitude Error Convergence", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=-0.5)

        # Plot 2: RMS phase error convergence
        ax2 = self.axes[0, 1]
        ax2.plot(
            self.iterations,
            self.rms_phase_history,
            "o-",
            color="#A23B72",
            linewidth=2,
            markersize=6,
        )
        ax2.set_xlabel("Iteration", fontsize=11)
        ax2.set_ylabel("RMS Phase Error (rad)", fontsize=11)
        ax2.set_title("RMS Phase Error Convergence", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=-0.5)

        # Plot 3: Individual tap amplitude errors
        ax3 = self.axes[1, 0]
        tap_amp_errors = np.array(self.tap_amp_errors_history)

        # Plot each tap's error over iterations
        tap_numbers = np.arange(1, self.num_taps + 1)
        for i in range(self.num_taps):
            ax3.plot(
                self.iterations,
                tap_amp_errors[:, i],
                "o-",
                label=f"Tap {tap_numbers[i]}",
                alpha=0.7,
                markersize=4,
            )

        ax3.set_xlabel("Iteration", fontsize=11)
        ax3.set_ylabel("Amplitude Error (dB)", fontsize=11)
        ax3.set_title("Individual Tap Amplitude Errors", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best", fontsize=8, ncol=2)
        ax3.set_xlim(left=-0.5)

        # Plot 4: Individual tap phase errors
        ax4 = self.axes[1, 1]
        tap_phase_errors = np.array(self.tap_phase_errors_history)

        # Plot each tap's error over iterations
        for i in range(self.num_taps):
            ax4.plot(
                self.iterations,
                tap_phase_errors[:, i],
                "o-",
                label=f"Tap {tap_numbers[i]}",
                alpha=0.7,
                markersize=4,
            )

        ax4.set_xlabel("Iteration", fontsize=11)
        ax4.set_ylabel("Phase Error (rad)", fontsize=11)
        ax4.set_title("Individual Tap Phase Errors", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="best", fontsize=8, ncol=2)
        ax4.set_xlim(left=-0.5)

        # Adjust layout and refresh
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Brief pause to allow display update

    def add_mzi_plot(self):
        """
        Add a fifth subplot for MZI PSR errors if needed.
        This creates a new figure specifically for MZI errors.
        """
        self.fig_mzi, self.ax_mzi = plt.subplots(figsize=(10, 4))
        self.fig_mzi.suptitle(
            "MZI Power Splitting Ratio Errors", fontsize=14, fontweight="bold"
        )
        plt.ion()
        plt.show()

    def update_mzi_plot(self):
        """Update MZI error plot if it exists."""
        if not hasattr(self, "ax_mzi"):
            return

        self.ax_mzi.clear()

        mzi_psr_errors = np.array(self.mzi_psr_errors_history)

        # Use actual MZI IDs if available, otherwise use indices
        if hasattr(self, "mzi_ids"):
            labels = self.mzi_ids
        else:
            labels = [f"MZI {i+1}" for i in range(mzi_psr_errors.shape[1])]

        # Plot each MZI's error over iterations
        for i, label in enumerate(labels):
            self.ax_mzi.plot(
                self.iterations,
                mzi_psr_errors[:, i],
                "o-",
                label=label,
                alpha=0.7,
                markersize=4,
            )

        self.ax_mzi.set_xlabel("Iteration", fontsize=11)
        self.ax_mzi.set_ylabel("PSR Error (dB)", fontsize=11)
        self.ax_mzi.grid(True, alpha=0.3)
        self.ax_mzi.legend(loc="best", fontsize=8, ncol=3)
        self.ax_mzi.set_xlim(left=-0.5)

        self.fig_mzi.tight_layout()
        self.fig_mzi.canvas.draw()
        self.fig_mzi.canvas.flush_events()
        plt.pause(0.01)

    def save_plots(self, output_dir: str):
        """
        Save final plots to output directory.

        Parameters
        ----------
        output_dir : str
            Directory to save plots
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main convergence plot
        self.fig.savefig(
            output_path / "calibration_convergence.png", dpi=300, bbox_inches="tight"
        )

        # Save MZI plot if it exists
        if hasattr(self, "fig_mzi"):
            self.fig_mzi.savefig(
                output_path / "mzi_psr_errors.png", dpi=300, bbox_inches="tight"
            )

        logger.info(f"Plots saved to {output_dir}")

    def close(self):
        """Close all plot windows."""
        plt.ioff()
        plt.close("all")


def plot_calibration_errors(
    iterations: List[IterationData], output_dir: Optional[str] = None
):
    """
    Plot final calibration results (post-processing version).

    Parameters
    ----------
    iterations : List[IterationData]
        List of all iteration data
    output_dir : Optional[str]
        If provided, save plots to this directory
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


if __name__ == "__main__":
    # Example usage
    logger.info("Example: Live plotting during calibration")

    # Initialise plotter (8 taps, 15 MZIs)
    plotter = CalibrationPlotter(num_taps=8, num_mzis=15)

    # Optionally add MZI plot
    plotter.add_mzi_plot()

    # Simulate calibration iterations
    mzi_ids = [
        "2-1",
        "3-1",
        "3-2",
        "3-3",
        "4-1",
        "4-2",
        "4-3",
        "4-4",
        "4-5",
        "4-6",
        "4-7",
        "4-8",
        "5-1",
        "5-2",
        "5-3",
    ]

    for i in range(10):
        # Create dummy data with dict for MZI errors
        mzi_errors_dict = {
            mzi_id: np.random.randn() * np.exp(-i / 4) for mzi_id in mzi_ids
        }

        iter_data = IterationData(
            iteration=i,
            amplitude_errors_db=np.random.randn(8) * np.exp(-i / 3),
            phase_errors_rad=np.random.randn(8) * np.exp(-i / 3),
            rms_amplitude_error_db=np.exp(-i / 3),
            rms_phase_error_rad=np.exp(-i / 3),
            mzi_psr_errors_db=mzi_errors_dict,
        )

        # Update plots
        plotter.update(iter_data)
        plotter.update_mzi_plot()

        # Simulate processing time
        import time

        time.sleep(0.5)

    # Save final plots
    plotter.save_plots("/mnt/user-data/outputs")

    logger.info("\nPress Enter to close plots...")
    input()
    plotter.close()
