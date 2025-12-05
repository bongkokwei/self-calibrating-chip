"""
Ideal Response Generator for Photonic FIR Chip

This module provides functionality to define ideal frequency responses
for various filter types and compute the corresponding ideal tap coefficients.

Based on: Xu et al. (2022) "Self-calibrating programmable photonic integrated circuits"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from photonic_fir_chip import PhotonicFIRChip, ChipParameters, create_sinc_filter


class IdealResponseGenerator:
    """
    Generate ideal frequency responses and compute corresponding tap coefficients
    for various filter types.
    """

    def __init__(self, chip_params: ChipParameters):
        """
        Initialize the ideal response generator.

        Parameters:
        -----------
        chip_params : ChipParameters
            Physical parameters of the photonic chip
        """
        self.params = chip_params
        # Calculate number of unused taps
        self.n_unused = (
            self.params.n_taps - self.params.n_signal_taps - 1
        )  # -1 for reference

    def define_frequency_response(
        self, filter_type: str, n_points: int = 1000, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Define the ideal frequency response for various filter types.

        Parameters:
        -----------
        filter_type : str
            Type of filter: 'sinc', 'hilbert', 'lowpass', 'highpass', 'differentiator'
        n_points : int
            Number of frequency points
        **kwargs : dict
            Additional parameters specific to each filter type

        Returns:
        --------
        (frequencies, H_ideal) : Tuple[np.ndarray, np.ndarray]
            Frequency array (Hz) and ideal complex frequency response
        """
        # Frequency array spanning one FSR
        frequencies = np.linspace(-self.params.fsr / 2, self.params.fsr / 2, n_points)
        omega = 2 * np.pi * frequencies

        if filter_type == "sinc":
            H_ideal = self._sinc_filter(frequencies, omega, **kwargs)

        elif filter_type == "hilbert":
            H_ideal = self._hilbert_transformer(frequencies, omega, **kwargs)

        elif filter_type == "lowpass":
            H_ideal = self._lowpass_filter(frequencies, omega, **kwargs)

        elif filter_type == "highpass":
            H_ideal = self._highpass_filter(frequencies, omega, **kwargs)

        elif filter_type == "differentiator":
            H_ideal = self._differentiator(frequencies, omega, **kwargs)

        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        # Ensure H_ideal is complex array
        H_ideal = H_ideal.astype(complex)

        return frequencies, H_ideal

    def _sinc_filter(
        self, frequencies: np.ndarray, omega: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Generate complex sinc filter frequency response.

        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency array (Hz)
        omega : np.ndarray
            Angular frequency array (rad/s)
        **kwargs : dict
            phase_step : float - Phase step between taps (default: 0)

        Returns:
        --------
        H_ideal : np.ndarray
            Complex frequency response
        """
        phase_step = kwargs.get("phase_step", 0)

        # Get sinc tap coefficients
        amplitudes, phases = create_sinc_filter(
            n_taps=self.params.n_signal_taps, phase_step=phase_step
        )

        # Construct frequency response from tap coefficients
        H_ideal = np.zeros(len(frequencies), dtype=complex)
        start_idx = self.n_unused + 1

        for i in range(self.params.n_signal_taps):
            tap_coeff = amplitudes[i] * np.exp(1j * phases[i])
            delay = (start_idx + i) * self.params.delay_step
            H_ideal += tap_coeff * np.exp(-1j * omega * delay)

        return H_ideal

    def _hilbert_transformer(
        self, frequencies: np.ndarray, omega: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Generate Hilbert transformer frequency response: -j*sign(ω)

        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency array (Hz)
        omega : np.ndarray
            Angular frequency array (rad/s)
        **kwargs : dict
            window : bool - Apply windowing (default: True)

        Returns:
        --------
        H_ideal : np.ndarray
            Complex frequency response
        """
        apply_window = kwargs.get("window", True)

        # Hilbert transformer: -j*sign(ω)
        H_ideal = -1j * np.sign(frequencies)

        # Apply smooth window to avoid Gibbs phenomenon and match 8-tap implementation
        if apply_window:
            window = np.hanning(len(frequencies))
            H_ideal = H_ideal * window

        return H_ideal

    def _lowpass_filter(
        self, frequencies: np.ndarray, omega: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Generate half-band lowpass filter frequency response.

        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency array (Hz)
        omega : np.ndarray
            Angular frequency array (rad/s)
        **kwargs : dict
            cutoff_fraction : float - Cutoff frequency as fraction of FSR (default: 0.25)
            transition_width : float - Transition width as fraction of cutoff (default: 0.2)

        Returns:
        --------
        H_ideal : np.ndarray
            Complex frequency response
        """
        cutoff_fraction = kwargs.get("cutoff_fraction", 0.25)
        transition_width_frac = kwargs.get("transition_width", 0.2)

        cutoff = self.params.fsr * cutoff_fraction
        transition_width = cutoff * transition_width_frac

        H_ideal = np.zeros(len(frequencies), dtype=complex)

        for i, f in enumerate(frequencies):
            abs_f = np.abs(f)
            if abs_f <= cutoff:
                H_ideal[i] = 1.0
            elif cutoff < abs_f < cutoff + transition_width:
                # Smooth cosine transition
                H_ideal[i] = 0.5 * (
                    1 + np.cos(np.pi * (abs_f - cutoff) / transition_width)
                )
            else:
                H_ideal[i] = 0.0

        return H_ideal

    def _highpass_filter(
        self, frequencies: np.ndarray, omega: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Generate half-band highpass filter frequency response.

        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency array (Hz)
        omega : np.ndarray
            Angular frequency array (rad/s)
        **kwargs : dict
            cutoff_fraction : float - Cutoff frequency as fraction of FSR (default: 0.25)
            transition_width : float - Transition width as fraction of cutoff (default: 0.2)

        Returns:
        --------
        H_ideal : np.ndarray
            Complex frequency response
        """
        cutoff_fraction = kwargs.get("cutoff_fraction", 0.25)
        transition_width_frac = kwargs.get("transition_width", 0.2)

        cutoff = self.params.fsr * cutoff_fraction
        transition_width = cutoff * transition_width_frac

        H_ideal = np.zeros(len(frequencies), dtype=complex)

        for i, f in enumerate(frequencies):
            abs_f = np.abs(f)
            if abs_f >= cutoff:
                H_ideal[i] = 1.0
            elif cutoff - transition_width < abs_f < cutoff:
                # Smooth cosine transition
                H_ideal[i] = 0.5 * (
                    1 - np.cos(np.pi * (cutoff - abs_f) / transition_width)
                )
            else:
                H_ideal[i] = 0.0

        return H_ideal

    def _differentiator(
        self, frequencies: np.ndarray, omega: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Generate differentiator frequency response: jω (normalized)

        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency array (Hz)
        omega : np.ndarray
            Angular frequency array (rad/s)
        **kwargs : dict
            window : bool - Apply windowing (default: True)

        Returns:
        --------
        H_ideal : np.ndarray
            Complex frequency response
        """
        apply_window = kwargs.get("window", True)

        # Differentiator: jω (normalized to avoid very large values)
        omega_normalized = omega / (self.params.fsr * np.pi)
        H_ideal = 1j * omega_normalized

        # Apply window to match 8-tap implementation
        if apply_window:
            window = np.hanning(len(frequencies))
            H_ideal = H_ideal * window

        return H_ideal

    def compute_tap_coefficients_ifft(
        self, H_ideal: np.ndarray, frequencies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ideal tap coefficients from frequency response via inverse FFT.

        This method uses the inverse Fourier transform to convert the frequency
        response to the time domain impulse response, then samples at the tap delays.

        Parameters:
        -----------
        H_ideal : np.ndarray
            Ideal complex frequency response
        frequencies : np.ndarray
            Frequency array (Hz)

        Returns:
        --------
        (tap_amplitudes, tap_phases) : Tuple[np.ndarray, np.ndarray]
            Ideal tap amplitudes and phases for the signal processing taps
        """
        from scipy.fft import ifft, ifftshift

        # Frequency spacing
        df = frequencies[1] - frequencies[0]
        n_fft = len(frequencies)

        # Inverse FFT
        h_time = ifft(ifftshift(H_ideal)) * n_fft * df

        # Time array corresponding to the IFFT output
        t_max = 1 / (2 * df)  # Maximum time from Nyquist
        time_array = np.linspace(-t_max, t_max, n_fft, endpoint=False)

        # Extract tap coefficients at the appropriate delays
        tap_coeffs = np.zeros(self.params.n_signal_taps, dtype=complex)
        start_idx = self.n_unused + 1

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

    def compute_tap_coefficients_direct(
        self, H_ideal: np.ndarray, frequencies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute tap coefficients by directly solving the FIR equation.

        This method directly inverts the frequency response equation:
        H(ω) = Σ h_n * exp(-jωτ_n)

        This is more accurate for FIR filters as it directly solves the linear system.

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
        n_taps = self.params.n_signal_taps
        n_freqs = len(frequencies)
        start_idx = self.n_unused + 1

        # We need at least n_taps frequency points
        if n_freqs < n_taps:
            raise ValueError("Need at least as many frequency points as taps")

        # Select n_taps frequency points uniformly
        freq_indices = np.linspace(0, n_freqs - 1, n_taps, dtype=int)
        selected_freqs = frequencies[freq_indices]
        selected_H = H_ideal[freq_indices]

        # Build system matrix A where H(ω_k) = Σ h_n * exp(-jω_k*τ_n)
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

    def verify_reconstruction(
        self,
        tap_amplitudes: np.ndarray,
        tap_phases: np.ndarray,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        """
        Verify tap coefficients by reconstructing the frequency response.

        Parameters:
        -----------
        tap_amplitudes : np.ndarray
            Tap amplitudes
        tap_phases : np.ndarray
            Tap phases (radians)
        frequencies : np.ndarray
            Frequency array (Hz)

        Returns:
        --------
        H_reconstructed : np.ndarray
            Reconstructed complex frequency response
        """
        H_reconstructed = np.zeros(len(frequencies), dtype=complex)
        omega = 2 * np.pi * frequencies
        start_idx = self.n_unused + 1

        for i in range(self.params.n_signal_taps):
            tap_coeff = tap_amplitudes[i] * np.exp(1j * tap_phases[i])
            delay = (start_idx + i) * self.params.delay_step
            H_reconstructed += tap_coeff * np.exp(-1j * omega * delay)

        return H_reconstructed

    def compute_reconstruction_error(
        self,
        H_ideal: np.ndarray,
        tap_amplitudes: np.ndarray,
        tap_phases: np.ndarray,
        frequencies: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute reconstruction error metrics.

        Parameters:
        -----------
        H_ideal : np.ndarray
            Ideal frequency response
        tap_amplitudes : np.ndarray
            Computed tap amplitudes
        tap_phases : np.ndarray
            Computed tap phases
        frequencies : np.ndarray
            Frequency array (Hz)

        Returns:
        --------
        errors : Dict[str, float]
            Dictionary containing various error metrics:
            - 'mse': Mean squared error
            - 'rmse': Root mean squared error
            - 'max_error': Maximum absolute error
            - 'mean_abs_error': Mean absolute error
        """
        H_reconstructed = self.verify_reconstruction(
            tap_amplitudes, tap_phases, frequencies
        )

        error = H_ideal - H_reconstructed

        errors = {
            "mse": np.mean(np.abs(error) ** 2),
            "rmse": np.sqrt(np.mean(np.abs(error) ** 2)),
            "max_error": np.max(np.abs(error)),
            "mean_abs_error": np.mean(np.abs(error)),
        }

        return errors

    def plot_ideal_response(
        self,
        frequencies: np.ndarray,
        H_ideal: np.ndarray,
        tap_amplitudes: np.ndarray,
        tap_phases: np.ndarray,
        filter_name: str,
        output_path: Optional[str] = None,
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
        filter_name : str
            Name of filter (for title and filename)
        output_path : str, optional
            Output directory path (default: /mnt/user-data/outputs/)
        """
        if output_path is None:
            output_path = "/mnt/user-data/outputs"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Frequency response amplitude
        ax = axes[0, 0]
        amplitude_db = 20 * np.log10(np.abs(H_ideal) + 1e-12)
        ax.plot(frequencies / 1e9, amplitude_db, "b-", linewidth=1.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"Ideal Frequency Response - {filter_name}")
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
        tap_indices = np.arange(self.params.n_signal_taps) + self.n_unused + 2
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

        filename = f"{output_path}/ideal_{filter_name}_response.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {filename}")
        plt.close()
