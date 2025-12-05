"""
Self-Calibration Simulation for Photonic FIR Chip

This script simulates the self-calibration process described in Xu et al. (2022).
It demonstrates:
1. Defining ideal frequency responses for various filter types
2. Computing ideal tap coefficients via inverse Fourier transform
3. Simulating the calibration process with thermal cross-talk effects
"""

import sys

sys.path.append("/mnt/project")

from photonic_fir_chip import PhotonicFIRChip, ChipParameters, create_sinc_filter
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from typing import Tuple, Dict, Callable


class CalibrationSimulator:
    """Simulate the self-calibration process for the photonic FIR chip."""

    def __init__(self, chip: PhotonicFIRChip):
        """
        Initialize the calibration simulator.

        Parameters:
        -----------
        chip : PhotonicFIRChip
            The chip instance to calibrate
        """
        self.chip = chip
        self.params = chip.params

        # Storage for calibration history
        self.amplitude_history = []
        self.phase_history = []
        self.mzi_error_history = []
        self.phase_error_history = []

    def define_ideal_frequency_response(
        self,
        filter_type: str,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Define the ideal frequency response for various filter types.

        Parameters:
        -----------
        filter_type : str
            Type of filter: 'sinc', 'hilbert', 'lowpass', 'highpass', 'differentiator'
        **kwargs : dict
            Additional parameters specific to each filter type

        Returns:
        --------
        (frequencies, H_ideal) : Tuple[np.ndarray, np.ndarray]
            Frequency array and ideal complex frequency response
        """
        # Frequency array spanning one FSR
        n_points = kwargs.get("n_points", 1000)
        frequencies = np.linspace(-self.params.fsr / 2, self.params.fsr / 2, n_points)
        omega = 2 * np.pi * frequencies

        if filter_type == "sinc":
            # Complex sinc filter with phase step
            phase_step = kwargs.get("phase_step", 0)
            amplitudes, phases = create_sinc_filter(
                n_taps=self.params.n_signal_taps, phase_step=phase_step
            )

            # Construct frequency response from tap coefficients
            H_ideal = np.zeros(len(frequencies), dtype=complex)
            start_idx = self.params.n_unused + 1

            for i in range(self.params.n_signal_taps):
                tap_coeff = amplitudes[i] * np.exp(1j * phases[i])
                delay = (start_idx + i) * self.params.delay_step
                H_ideal += tap_coeff * np.exp(-1j * omega * delay)

        elif filter_type == "hilbert":
            # Hilbert transformer: -j*sign(ω)
            H_ideal = -1j * np.sign(frequencies)

            # Window to match 8-tap implementation
            # Apply smooth window to avoid Gibbs phenomenon
            window = np.hanning(n_points)
            H_ideal = H_ideal * window

        elif filter_type == "lowpass":
            # Half-band lowpass filter
            cutoff = self.params.fsr / 4  # Half-band
            H_ideal = np.where(np.abs(frequencies) <= cutoff, 1.0, 0.0)

            # Smooth transition
            transition_width = cutoff * 0.2
            for i, f in enumerate(frequencies):
                if cutoff < np.abs(f) < cutoff + transition_width:
                    H_ideal[i] = 0.5 * (
                        1 + np.cos(np.pi * (np.abs(f) - cutoff) / transition_width)
                    )

        elif filter_type == "highpass":
            # Half-band highpass filter
            cutoff = self.params.fsr / 4
            H_ideal = np.where(np.abs(frequencies) >= cutoff, 1.0, 0.0)

            # Smooth transition
            transition_width = cutoff * 0.2
            for i, f in enumerate(frequencies):
                if cutoff - transition_width < np.abs(f) < cutoff:
                    H_ideal[i] = 0.5 * (
                        1 - np.cos(np.pi * (cutoff - np.abs(f)) / transition_width)
                    )

        elif filter_type == "differentiator":
            # Differentiator: jω (scaled appropriately)
            omega_normalized = omega / (self.params.fsr * np.pi)
            H_ideal = 1j * omega_normalized

            # Window to match 8-tap implementation
            window = np.hanning(n_points)
            H_ideal = H_ideal * window

        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        # Ensure H_ideal is complex array
        H_ideal = H_ideal.astype(complex)

        return frequencies, H_ideal

    def compute_ideal_tap_coefficients(
        self,
        H_ideal: np.ndarray,
        frequencies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ideal tap coefficients from frequency response via inverse FFT.

        The key insight from Xu et al. (2022) is that the tap coefficients
        can be recovered by:
        1. Taking the measured/ideal frequency response H(ω)
        2. Performing inverse Fourier transform to get impulse response h(t)
        3. Sampling at the tap delays to get tap coefficients

        Parameters:
        -----------
        H_ideal : np.ndarray
            Ideal complex frequency response
        frequencies : np.ndarray
            Frequency array (Hz)

        Returns:
        --------
        (tap_amplitudes, tap_phases) : Tuple[np.ndarray, np.ndarray]
            Ideal tap amplitudes and phases for the 8 signal processing taps
        """
        # Perform inverse FFT to get impulse response
        # Need to ensure proper frequency sampling for IFFT
        df = frequencies[1] - frequencies[0]
        n_fft = len(frequencies)

        # Inverse FFT
        h_time = ifft(ifftshift(H_ideal)) * n_fft * df

        # Time array corresponding to the IFFT output
        t_max = 1 / (2 * df)  # Maximum time from Nyquist
        time_array = np.linspace(-t_max, t_max, n_fft, endpoint=False)

        # Extract tap coefficients at the appropriate delays
        tap_coeffs = np.zeros(self.params.n_signal_taps, dtype=complex)
        start_idx = self.params.n_unused + 1

        for i in range(self.params.n_signal_taps):
            tap_delay = (start_idx + i) * self.params.delay_step

            # Find closest time index
            idx = np.argmin(np.abs(time_array - tap_delay))
            tap_coeffs[i] = h_time[idx]

        # Normalize tap coefficients
        max_amplitude = np.max(np.abs(tap_coeffs))
        if max_amplitude > 0:
            tap_coeffs = tap_coeffs / max_amplitude * 0.5  # Scale to 0.5 max

        # Extract amplitudes and phases
        tap_amplitudes = np.abs(tap_coeffs)
        tap_phases = np.angle(tap_coeffs)

        return tap_amplitudes, tap_phases

    def compute_tap_coefficients_from_frequency_response_direct(
        self,
        H_ideal: np.ndarray,
        frequencies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alternative method: directly compute tap coefficients by evaluating
        H(ω) at specific frequencies and solving the linear system.

        This is more accurate for FIR filters as it directly inverts the
        frequency response equation:
        H(ω) = Σ h_n * exp(-jωτ_n)

        Parameters:
        -----------
        H_ideal : np.ndarray
            Ideal complex frequency response
        frequencies : np.ndarray
            Frequency array (Hz)

        Returns:
        --------
        (tap_amplitudes, tap_phases) : Tuple[np.ndarray, np.ndarray]
            Ideal tap amplitudes and phases
        """
        # Build the system matrix A where:
        # H(ω_k) = Σ h_n * exp(-jω_k*τ_n)
        # This becomes: H = A * h, where h is the tap coefficient vector

        n_taps = self.params.n_signal_taps
        n_freqs = len(frequencies)
        start_idx = self.params.n_unused + 1

        # We need at least n_taps frequency points
        if n_freqs < n_taps:
            raise ValueError("Need at least as many frequency points as taps")

        # Select n_taps frequency points uniformly
        freq_indices = np.linspace(0, n_freqs - 1, n_taps, dtype=int)
        selected_freqs = frequencies[freq_indices]
        selected_H = H_ideal[freq_indices]

        # Build matrix A
        A = np.zeros((n_taps, n_taps), dtype=complex)
        for k, freq in enumerate(selected_freqs):
            omega = 2 * np.pi * freq
            for n in range(n_taps):
                tap_delay = (start_idx + n) * self.params.delay_step
                A[k, n] = np.exp(-1j * omega * tap_delay)

        # Solve for tap coefficients: h = A^(-1) * H
        try:
            tap_coeffs = np.linalg.solve(A, selected_H)
        except np.linalg.LinAlgError:
            # If matrix is singular, use least squares
            tap_coeffs, _, _, _ = np.linalg.lstsq(A, selected_H, rcond=None)

        # Extract amplitudes and phases
        tap_amplitudes = np.abs(tap_coeffs)
        tap_phases = np.angle(tap_coeffs)

        return tap_amplitudes, tap_phases

    def plot_ideal_response(
        self,
        frequencies: np.ndarray,
        H_ideal: np.ndarray,
        tap_amplitudes: np.ndarray,
        tap_phases: np.ndarray,
        filter_type: str,
    ):
        """
        Plot the ideal frequency response and tap coefficients.

        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency array (Hz)
        H_ideal : np.ndarray
            Ideal complex frequency response
        tap_amplitudes : np.ndarray
            Ideal tap amplitudes
        tap_phases : np.ndarray
            Ideal tap phases
        filter_type : str
            Type of filter (for title)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Frequency response amplitude
        ax = axes[0, 0]
        amplitude_db = 20 * np.log10(np.abs(H_ideal) + 1e-12)
        ax.plot(frequencies / 1e9, amplitude_db, "b-", linewidth=1.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"Ideal Frequency Response - {filter_type}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-40, 5])

        # Plot 2: Frequency response phase
        ax = axes[0, 1]
        ax.plot(frequencies / 1e9, np.angle(H_ideal), "r-", linewidth=1.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Phase (rad)")
        ax.set_title("Ideal Phase Response")
        ax.grid(True, alpha=0.3)

        # Plot 3: Tap amplitudes
        ax = axes[1, 0]
        tap_indices = np.arange(self.params.n_signal_taps) + self.params.n_unused + 2
        ax.stem(tap_indices, tap_amplitudes, basefmt=" ")
        ax.set_xlabel("Tap Number")
        ax.set_ylabel("Amplitude")
        ax.set_title("Ideal Tap Amplitudes")
        ax.grid(True, alpha=0.3)

        # Plot 4: Tap phases
        ax = axes[1, 1]
        ax.stem(tap_indices, tap_phases, basefmt=" ", linefmt="r-", markerfmt="ro")
        ax.set_xlabel("Tap Number")
        ax.set_ylabel("Phase (rad)")
        ax.set_title("Ideal Tap Phases")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-np.pi - 0.5, np.pi + 0.5])

        plt.tight_layout()
        plt.savefig(
            f"/mnt/user-data/outputs/ideal_{filter_type}_response.png",
            dpi=150,
            bbox_inches="tight",
        )
        print(f"\nPlot saved: ideal_{filter_type}_response.png")
        plt.close()


def demonstrate_ideal_responses():
    """Demonstrate computing ideal tap coefficients for various filter types."""

    print("=" * 70)
    print("Ideal Response Computation for Self-Calibration")
    print("=" * 70)

    # Create chip instance
    chip = PhotonicFIRChip()
    simulator = CalibrationSimulator(chip)

    # Filter types to demonstrate
    filter_types = [
        ("sinc", {"phase_step": 0}),
        ("sinc", {"phase_step": 2 * np.pi / 7}),
        ("hilbert", {}),
        ("lowpass", {}),
        ("highpass", {}),
        ("differentiator", {}),
    ]

    results = {}

    for filter_type, kwargs in filter_types:
        filter_name = (
            filter_type
            if not kwargs.get("phase_step")
            else f"{filter_type}_{kwargs['phase_step']:.3f}rad"
        )

        print(f"\n{'-'*70}")
        print(f"Filter type: {filter_name}")
        print(f"{'-'*70}")

        # Define ideal frequency response
        frequencies, H_ideal = simulator.define_ideal_frequency_response(
            filter_type, **kwargs
        )

        # Method 1: Inverse FFT
        tap_amps_ifft, tap_phases_ifft = simulator.compute_ideal_tap_coefficients(
            H_ideal, frequencies
        )

        # Method 2: Direct matrix inversion (more accurate for FIR)
        tap_amps_direct, tap_phases_direct = (
            simulator.compute_tap_coefficients_from_frequency_response_direct(
                H_ideal, frequencies
            )
        )

        print("\nMethod 1 (IFFT):")
        print(f"  Tap amplitudes: {tap_amps_ifft}")
        print(f"  Tap phases (rad): {tap_phases_ifft}")

        print("\nMethod 2 (Direct):")
        print(f"  Tap amplitudes: {tap_amps_direct}")
        print(f"  Tap phases (rad): {tap_phases_direct}")

        # Use direct method for better accuracy
        tap_amplitudes = tap_amps_direct
        tap_phases = tap_phases_direct

        # Store results
        results[filter_name] = {
            "frequencies": frequencies,
            "H_ideal": H_ideal,
            "tap_amplitudes": tap_amplitudes,
            "tap_phases": tap_phases,
        }

        # Plot
        simulator.plot_ideal_response(
            frequencies, H_ideal, tap_amplitudes, tap_phases, filter_name
        )

        # Verify: compute frequency response from tap coefficients
        H_reconstructed = np.zeros(len(frequencies), dtype=complex)
        omega = 2 * np.pi * frequencies
        start_idx = chip.params.n_unused + 1

        for i in range(chip.params.n_signal_taps):
            tap_coeff = tap_amplitudes[i] * np.exp(1j * tap_phases[i])
            delay = (start_idx + i) * chip.params.delay_step
            H_reconstructed += tap_coeff * np.exp(-1j * omega * delay)

        # Compare
        error = np.mean(np.abs(H_ideal - H_reconstructed) ** 2)
        print(f"\nReconstruction MSE: {error:.6e}")

    print("\n" + "=" * 70)
    print("Ideal response computation complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = demonstrate_ideal_responses()
    print("\nAll ideal responses computed and saved.")
