"""
Power Splitting Ratio Calculator for 16-tap FIR Photonic Chip
Based on: Xu et al. (2022) "Self-calibrating programmable photonic integrated circuits"
Nature Photonics, Vol 16, August 2022, 595-602
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from typing import Dict


class PowerSplittingCalculator:
    """
    Calculate MZI power splitting ratios from desired tap coefficients
    for a 16-tap FIR chip with binary tree architecture.

    Architecture:
    - 1 reference tap (tap 0)
    - 7 unused taps (taps 1-7)
    - 8 signal processing taps (taps 8-15)
    - 7 MZIs in binary tree (3 stages)
    """

    def __init__(self):
        self.n_taps = 16
        self.n_signal_taps = 8
        self.n_unused = 7
        self.n_mzis = 7

        # MZI names for clarity
        self.mzi_names = [
            "MZI_2-1",  # Stage 1: splits into [9-12] and [13-16]
            "MZI_3-3",  # Stage 2: splits [9-12] into [9-10] and [11-12]
            "MZI_3-4",  # Stage 2: splits [13-16] into [13-14] and [15-16]
            "MZI_4-5",  # Stage 3: splits [9-10] into 9 and 10
            "MZI_4-6",  # Stage 3: splits [11-12] into 11 and 12
            "MZI_4-7",  # Stage 3: splits [13-14] into 13 and 14
            "MZI_4-8",  # Stage 3: splits [15-16] into 15 and 16
        ]

        self.power_splitting_ratios: Dict[str, float] = {}

    def tap_coeffs_to_power_splitting_ratios(
        self, tap_coeffs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate MZI power splitting ratios from desired tap coefficients.

        The binary tree architecture means:
        - Stage 1 (MZI 2-1): Controls ratio between taps [9-12] vs [13-16]
        - Stage 2 (MZI 3-3, 3-4): Controls ratios within each group of 4
        - Stage 3 (MZI 4-5 to 4-8): Controls ratios within each pair

        Parameters:
        -----------
        tap_coeffs : np.ndarray
            Complex tap coefficients (length 16)
            tap_coeffs[0]: reference tap
            tap_coeffs[1-7]: unused taps
            tap_coeffs[8-15]: signal processing taps

        Returns:
        --------
        power_splitting_ratios : np.ndarray
            Power splitting ratios for 7 MZIs in dB
        tap_phases : np.ndarray
            Phase shifts for 8 signal processing taps in radians
        """
        if len(tap_coeffs) != self.n_taps:
            raise ValueError(f"Expected {self.n_taps} tap coefficients")

        # Extract signal processing core taps (indices 8-15)
        signal_taps = tap_coeffs[8:16]

        # Get tap powers (intensities)
        tap_powers = np.abs(signal_taps) ** 2

        # Get tap phases
        tap_phases = np.angle(signal_taps)

        # Calculate power splitting ratios for each MZI
        power_splitting_ratios = np.zeros(self.n_mzis)

        # Stage 1: MZI 2-1
        # Controls ratio between [taps 9-12] and [taps 13-16]
        power_9_12 = np.sum(tap_powers[0:4])  # Taps 9-12 (indices 0-3)
        power_13_16 = np.sum(tap_powers[4:8])  # Taps 13-16 (indices 4-7)

        if power_9_12 + power_13_16 > 0:
            # Power splitting ratio in dB (bar port / cross port)
            power_splitting_ratios[0] = 10 * np.log10(power_9_12 / power_13_16)

        # Stage 2: MZI 3-3
        # Controls ratio between [taps 9-10] and [taps 11-12]
        power_9_10 = np.sum(tap_powers[0:2])
        power_11_12 = np.sum(tap_powers[2:4])

        if power_9_10 + power_11_12 > 0:
            power_splitting_ratios[1] = 10 * np.log10(power_9_10 / power_11_12)

        # Stage 2: MZI 3-4
        # Controls ratio between [taps 13-14] and [taps 15-16]
        power_13_14 = np.sum(tap_powers[4:6])
        power_15_16 = np.sum(tap_powers[6:8])

        if power_13_14 + power_15_16 > 0:
            power_splitting_ratios[2] = 10 * np.log10(power_13_14 / power_15_16)

        # Stage 3: MZI 4-5, 4-6, 4-7, 4-8
        # Each controls ratio between individual taps
        tap_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]  # Pairs in signal_taps indices

        for i, (idx1, idx2) in enumerate(tap_pairs):
            if tap_powers[idx1] + tap_powers[idx2] > 0:
                power_splitting_ratios[3 + i] = 10 * np.log10(
                    tap_powers[idx1] / tap_powers[idx2]
                )
        # Store as dictionary with MZI names as keys
        self.power_splitting_ratios = {
            name: psr for name, psr in zip(self.mzi_names, power_splitting_ratios)
        }

        return power_splitting_ratios, tap_phases

    def power_splitting_ratio_to_mzi_phase(
        self, power_splitting_ratio_db: float
    ) -> float:
        """
        Convert power splitting ratio (dB) to MZI phase shift (radians).

        For an MZI with phase φ:
        - Bar port power = cos²(φ/2)
        - Cross port power = sin²(φ/2)
        - Power splitting ratio (dB) = 10 log10(cos²(φ/2) / sin²(φ/2))

        Solving for φ:
        φ = 2 * arctan(10^(-PSR_dB/20))

        Parameters:
        -----------
        power_splitting_ratio_db : float
            Power splitting ratio in dB

        Returns:
        --------
        phase : float
            MZI phase shift in radians
        """
        # Convert dB to linear ratio
        ratio_linear = 10 ** (power_splitting_ratio_db / 10)

        # Calculate phase
        # tan²(φ/2) = 1/ratio_linear
        # tan(φ/2) = 1/sqrt(ratio_linear)
        phase = 2 * np.arctan(1 / np.sqrt(ratio_linear))

        return phase

    def print_splitting_ratios(
        self, power_splitting_ratios: np.ndarray, tap_phases: np.ndarray
    ):
        """Print power splitting ratios and tap phases in readable format."""
        print("\n" + "=" * 70)
        print("Power Splitting Ratios for Binary Tree MZIs")
        print("=" * 70)

        for i, (name, psr) in enumerate(zip(self.mzi_names, power_splitting_ratios)):
            mzi_phase = self.power_splitting_ratio_to_mzi_phase(psr)
            print(
                f"{name:12s}: {psr:8.3f} dB  "
                f"(MZI phase: {mzi_phase:6.3f} rad = {np.degrees(mzi_phase):6.2f}°)"
            )

        print("\n" + "=" * 70)
        print("Tap Phases for Signal Processing Core (Taps 9-16)")
        print("=" * 70)

        for i in range(self.n_signal_taps):
            tap_num = i + 9
            print(
                f"Tap {tap_num:2d} Phase: {tap_phases[i]:7.4f} rad = "
                f"{np.degrees(tap_phases[i]):7.2f}°"
            )


def create_sinc_filter_taps(
    n_taps: int = 8, phase_step: float = 0.0, normalize: bool = True
) -> np.ndarray:
    """
    Create sinc filter tap coefficients with optional linear phase progression.

    Parameters:
    -----------
    n_taps : int
        Number of filter taps
    phase_step : float
        Phase step between taps in radians
    normalize : bool
        Whether to normalize amplitudes

    Returns:
    --------
    taps : np.ndarray
        Complex tap coefficients
    """
    n = np.arange(n_taps)
    center = (n_taps - 1) / 2
    x = n - center

    # Sinc amplitudes
    amplitudes = np.sinc(x / 2)

    if normalize:
        amplitudes = amplitudes / np.max(amplitudes) * 0.5

    # Linear phase progression
    phases = n * phase_step

    # Complex tap coefficients
    taps = amplitudes * np.exp(1j * phases)

    return taps


def main():
    """Demonstrate power splitting ratio calculation."""

    calculator = PowerSplittingCalculator()

    # Example 1: Sinc filter with 0 phase step
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Sinc Filter (0 phase step)")
    print("=" * 70)

    # Create full 16-tap coefficients
    signal_taps = create_sinc_filter_taps(n_taps=8, phase_step=0.0)

    # Full tap array: [reference, unused (7), signal processing (8)]
    tap_coeffs = np.zeros(16, dtype=complex)
    tap_coeffs[0] = 1.0  # Reference tap
    tap_coeffs[1:8] = 0.01  # Unused taps (very small)
    tap_coeffs[8:16] = signal_taps  # Signal processing taps

    # Calculate power splitting ratios
    psr, phases = calculator.tap_coeffs_to_power_splitting_ratios(tap_coeffs)
    calculator.print_splitting_ratios(psr, phases)

    # Example 2: Sinc filter with 2π/7 phase step
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Sinc Filter (2π/7 phase step)")
    print("=" * 70)

    signal_taps = create_sinc_filter_taps(n_taps=8, phase_step=2 * np.pi / 7)
    tap_coeffs[8:16] = signal_taps

    psr, phases = calculator.tap_coeffs_to_power_splitting_ratios(tap_coeffs)
    calculator.print_splitting_ratios(psr, phases)

    # Example 3: Custom tap coefficients
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Custom Tap Coefficients")
    print("=" * 70)

    # Create custom taps with specific amplitudes and phases
    custom_amplitudes = np.array([0.3, 0.5, 0.7, 0.5, 0.5, 0.7, 0.5, 0.3])
    custom_phases = np.array(
        [
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4,
            np.pi,
            5 * np.pi / 4,
            3 * np.pi / 2,
            7 * np.pi / 4,
        ]
    )

    signal_taps = custom_amplitudes * np.exp(1j * custom_phases)
    tap_coeffs[8:16] = signal_taps

    psr, phases = calculator.tap_coeffs_to_power_splitting_ratios(tap_coeffs)
    calculator.print_splitting_ratios(psr, phases)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot tap amplitudes
    ax = axes[0, 0]
    tap_nums = np.arange(16) + 1
    tap_amps = np.abs(tap_coeffs)
    ax.bar(tap_nums, 20 * np.log10(tap_amps + 1e-12), alpha=0.6)
    ax.set_xlabel("Tap Number")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title("Tap Amplitudes (All 16 taps)")
    ax.grid(True, alpha=0.3)
    ax.axvspan(0.5, 1.5, alpha=0.2, color="green", label="Reference")
    ax.axvspan(1.5, 8.5, alpha=0.2, color="gray", label="Unused")
    ax.axvspan(8.5, 16.5, alpha=0.2, color="yellow", label="Signal Core")
    ax.legend()

    # Plot tap phases
    ax = axes[0, 1]
    ax.plot(tap_nums[8:], phases, "ro", markersize=8)
    ax.set_xlabel("Tap Number")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Tap Phases (Signal Processing Core)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-np.pi - 0.5, np.pi + 0.5])

    # Plot MZI power splitting ratios
    ax = axes[1, 0]
    mzi_indices = np.arange(len(psr)) + 1
    bars = ax.bar(mzi_indices, psr, alpha=0.6, color="blue")
    ax.set_xlabel("MZI Index")
    ax.set_ylabel("Power Splitting Ratio (dB)")
    ax.set_title("MZI Power Splitting Ratios")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(mzi_indices)
    ax.set_xticklabels([f"MZI\n{i+1}" for i in range(len(psr))])

    # Plot binary tree structure visualization
    ax = axes[1, 1]
    ax.text(
        0.5,
        0.95,
        "Binary Tree Architecture",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Draw tree structure
    tree_info = [
        (0.5, 0.80, "MZI 2-1\nStage 1"),
        (0.25, 0.55, "MZI 3-3\nStage 2"),
        (0.75, 0.55, "MZI 3-4\nStage 2"),
        (0.12, 0.30, "MZI 4-5\nStage 3"),
        (0.37, 0.30, "MZI 4-6\nStage 3"),
        (0.62, 0.30, "MZI 4-7\nStage 3"),
        (0.87, 0.30, "MZI 4-8\nStage 3"),
    ]

    for x, y, label in tree_info:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            transform=ax.transAxes,
            fontsize=8,
        )

    # Draw tap labels at bottom
    tap_positions = [0.06, 0.18, 0.31, 0.43, 0.56, 0.68, 0.81, 0.93]
    for i, x in enumerate(tap_positions):
        ax.text(
            x,
            0.05,
            f"Tap\n{i+9}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
        )

    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        "measurements/power_splitting_ratios.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("\nPlot saved: power_splitting_ratios.png")


if __name__ == "__main__":
    main()
