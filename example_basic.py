"""
Photonic FIR Chip Simulation - Basic Examples

This script demonstrates basic functionality of the photonic FIR chip:
1. Basic sinc filter with 0 phase step
2. Sinc filters with different phase steps

Based on: Xu et al. (2022) "Self-calibrating programmable photonic integrated circuits"
"""

from photonic_fir_chip import PhotonicFIRChip, ChipParameters
from ideal_response_generator import create_sinc_filter
import numpy as np
import matplotlib.pyplot as plt


def example_1_basic_sinc_filter():
    """Example 1: Basic sinc filter demonstration."""

    print("=" * 70)
    print("Example 1: Basic Sinc Filter (0 phase step)")
    print("=" * 70)

    # Create chip instance
    chip = PhotonicFIRChip()

    print(f"\nChip Configuration:")
    print(f"  Total taps: {chip.params.n_taps}")
    print(f"  Signal processing taps: {chip.params.n_signal_taps}")
    print(f"  FSR: {chip.params.fsr / 1e9:.1f} GHz")
    print(f"  Delay step: {chip.params.delay_step * 1e12:.2f} ps")

    # Create a sinc filter with 0 phase step
    amplitudes, phases = create_sinc_filter(n_taps=8, phase_step=0)

    # Set MZIs for approximately equal power distribution
    # For equal splitting, MZI phase = π/2 gives 50:50 split
    equal_split_phases = np.ones(7) * np.pi / 2
    chip.set_mzi_phases(equal_split_phases)

    # Set tap phases
    chip.set_signal_tap_phases(phases)

    # Compute and display tap coefficients
    tap_coeffs = chip.compute_tap_coefficients()
    print("\nTap Coefficients (Signal Processing Core):")
    for i in range(9, 17):  # Taps 9-16
        print(
            f"  Tap {i}: Amplitude = {np.abs(tap_coeffs[i-1]):.4f}, "
            f"Phase = {np.angle(tap_coeffs[i-1]):.4f} rad"
        )

    # Compute frequency response
    freq_range = np.linspace(-chip.params.fsr / 2, chip.params.fsr / 2, 1000)
    H_cal = chip.compute_frequency_response(freq_range, port="calibration")
    H_sig = chip.compute_frequency_response(freq_range, port="signal")

    # Compute impulse response
    time_cal, h_cal = chip.compute_impulse_response(port="calibration")
    time_sig, h_sig = chip.compute_impulse_response(port="signal")

    # Compute insertion loss spectrum
    wavelength_center = chip.params.center_wavelength * 1e9  # Convert to nm
    wavelengths = wavelength_center + np.linspace(-0.4, 0.4, 1000)
    insertion_loss_cal = chip.get_insertion_loss_spectrum(
        wavelengths, port="calibration"
    )
    insertion_loss_sig = chip.get_insertion_loss_spectrum(wavelengths, port="signal")

    # Create comprehensive plots
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Insertion Loss Spectrum (Calibration Port)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(wavelengths - wavelength_center, insertion_loss_cal, "b-", linewidth=1.5)
    ax1.set_xlabel("Wavelength offset (nm)")
    ax1.set_ylabel("Insertion Loss (dB)")
    ax1.set_title("Insertion Loss - Calibration Port")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-40, 5])

    # Plot 2: Insertion Loss Spectrum (Signal Port)
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(wavelengths - wavelength_center, insertion_loss_sig, "r-", linewidth=1.5)
    ax2.set_xlabel("Wavelength offset (nm)")
    ax2.set_ylabel("Insertion Loss (dB)")
    ax2.set_title("Insertion Loss - Signal Port")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-40, 5])

    # Plot 3: Frequency Response Amplitude
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(
        freq_range / 1e9,
        20 * np.log10(np.abs(H_cal) + 1e-12),
        "b-",
        linewidth=1.5,
        label="Calibration",
    )
    ax3.plot(
        freq_range / 1e9,
        20 * np.log10(np.abs(H_sig) + 1e-12),
        "r-",
        linewidth=1.5,
        label="Signal",
        alpha=0.7,
    )
    ax3.set_xlabel("Frequency (GHz)")
    ax3.set_ylabel("Magnitude (dB)")
    ax3.set_title("Frequency Response - Amplitude")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([-40, 5])

    # Plot 4: Frequency Response Phase
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(
        freq_range / 1e9,
        np.angle(H_cal),
        "b-",
        linewidth=1.5,
        label="Calibration",
    )
    ax4.plot(
        freq_range / 1e9,
        np.angle(H_sig),
        "r-",
        linewidth=1.5,
        label="Signal",
        alpha=0.7,
    )
    ax4.set_xlabel("Frequency (GHz)")
    ax4.set_ylabel("Phase (rad)")
    ax4.set_title("Frequency Response - Phase")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Impulse Response Amplitude (Calibration)
    ax5 = plt.subplot(3, 3, 5)
    ax5.stem(time_cal * 1e12, np.abs(h_cal), basefmt=" ")
    ax5.set_xlabel("Time (ps)")
    ax5.set_ylabel("Amplitude")
    ax5.set_title("Impulse Response Amplitude - Calibration Port")
    ax5.grid(True, alpha=0.3)

    # Plot 6: Impulse Response Phase (Calibration)
    ax6 = plt.subplot(3, 3, 6)
    ax6.stem(
        time_cal * 1e12, np.angle(h_cal), basefmt=" ", linefmt="r-", markerfmt="ro"
    )
    ax6.set_xlabel("Time (ps)")
    ax6.set_ylabel("Phase (rad)")
    ax6.set_title("Impulse Response Phase - Calibration Port")
    ax6.grid(True, alpha=0.3)

    # Plot 7: Impulse Response Amplitude (Signal)
    ax7 = plt.subplot(3, 3, 7)
    ax7.stem(time_sig * 1e12, np.abs(h_sig), basefmt=" ")
    ax7.set_xlabel("Time (ps)")
    ax7.set_ylabel("Amplitude")
    ax7.set_title("Impulse Response Amplitude - Signal Port")
    ax7.grid(True, alpha=0.3)

    # Plot 8: Impulse Response Phase (Signal)
    ax8 = plt.subplot(3, 3, 8)
    ax8.stem(
        time_sig * 1e12, np.angle(h_sig), basefmt=" ", linefmt="r-", markerfmt="ro"
    )
    ax8.set_xlabel("Time (ps)")
    ax8.set_ylabel("Phase (rad)")
    ax8.set_title("Impulse Response Phase - Signal Port")
    ax8.grid(True, alpha=0.3)

    # Plot 9: Tap Coefficients
    ax9 = plt.subplot(3, 3, 9)
    tap_indices = np.arange(chip.params.n_taps) + 1
    tap_amplitudes = np.abs(tap_coeffs)
    tap_phases = np.angle(tap_coeffs)

    ax9_twin = ax9.twinx()
    bars = ax9.bar(
        tap_indices,
        20 * np.log10(tap_amplitudes + 1e-12),
        alpha=0.6,
        color="blue",
        label="Amplitude",
    )
    dots = ax9_twin.plot(tap_indices, tap_phases, "ro", markersize=8, label="Phase")

    ax9.set_xlabel("Tap Number")
    ax9.set_ylabel("Amplitude (dB)", color="blue")
    ax9_twin.set_ylabel("Phase (rad)", color="red")
    ax9.set_title("Tap Coefficients (All 16 taps)")
    ax9.grid(True, alpha=0.3)
    ax9.set_xticks(tap_indices)

    # Highlight different regions
    ax9.axvspan(0.5, 1.5, alpha=0.2, color="green", label="Reference")
    ax9.axvspan(1.5, 8.5, alpha=0.2, color="gray", label="Unused")
    ax9.axvspan(8.5, 16.5, alpha=0.2, color="yellow", label="Signal Core")

    plt.tight_layout()
    plt.savefig("output/example1_basic_sinc_filter.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved: example1_basic_sinc_filter.png")

    plt.show()

    print("\n" + "=" * 70)
    print("Example 1 Complete!")
    print("=" * 70)


def example_2_phase_steps():
    """Example 2: Sinc filters with different phase steps."""

    print("\n" + "=" * 70)
    print("Example 2: Sinc Filters with Phase Steps")
    print("=" * 70)

    # Create chip instance
    chip = PhotonicFIRChip()

    # Set MZIs for equal power distribution
    equal_split_phases = np.ones(7) * np.pi / 2
    chip.set_mzi_phases(equal_split_phases)

    phase_steps = [0, 2 * np.pi / 7, 4 * np.pi / 7, 6 * np.pi / 7]

    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    wavelength_center = chip.params.center_wavelength * 1e9

    for idx, phase_step in enumerate(phase_steps):
        amplitudes, phases = create_sinc_filter(n_taps=8, phase_step=phase_step)
        chip.set_signal_tap_phases(phases)

        wavelengths_plot = wavelength_center + np.linspace(-0.4, 0.4, 1000)
        insertion_loss = chip.get_insertion_loss_spectrum(
            wavelengths_plot, port="signal"
        )

        axes[idx].plot(
            wavelengths_plot - wavelength_center, insertion_loss, linewidth=1.5
        )
        axes[idx].set_xlabel("Wavelength offset (nm)")
        axes[idx].set_ylabel("Insertion Loss (dB)")
        axes[idx].set_title(
            f"Phase step = {phase_step:.3f} rad ({phase_step*180/np.pi:.1f}°)"
        )
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([-40, 5])

        print(f"\nPhase step = {phase_step:.3f} rad ({phase_step*180/np.pi:.1f}°):")
        print(f"  Max insertion loss: {np.max(insertion_loss):.2f} dB")
        print(f"  Min insertion loss: {np.min(insertion_loss):.2f} dB")

    plt.tight_layout()
    plt.savefig("output/example2_phase_steps.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved: example2_phase_steps.png")

    plt.show()

    print("\n" + "=" * 70)
    print("Example 2 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    """Run basic examples."""
    example_1_basic_sinc_filter()
    example_2_phase_steps()
