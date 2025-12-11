"""
Calibration Simulator for Photonic FIR Chip

This module simulates the self-calibration process including:
- Kramers-Kronig phase recovery
- Iterative parameter updates
- Thermal cross-talk effects
- Measurement noise

Based on: Xu et al. (2022) "Self-calibrating programmable photonic integrated circuits"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from photonic_fir_chip import PhotonicFIRChip, ChipParameters


class CalibrationSimulator:
    """
    Simulate the self-calibration process for the photonic FIR chip.

    This class handles:
    - Measurement simulation (with noise)
    - Kramers-Kronig phase recovery
    - Tap coefficient recovery via inverse Fourier transform
    - Error calculation and parameter updates
    - Convergence monitoring
    """

    def __init__(
        self,
        chip: PhotonicFIRChip,
        learning_rate: float = 0.5,
        add_noise: bool = True,
        noise_level_db: float = 0.1,
    ):
        """
        Initialize the calibration simulator.

        Parameters:
        -----------
        chip : PhotonicFIRChip
            The chip instance to calibrate
        learning_rate : float
            Learning rate for parameter updates (default: 0.5)
        add_noise : bool
            Whether to add measurement noise (default: True)
        noise_level_db : float
            Noise level in dB for insertion loss measurements (default: 0.1)
        """
        self.chip = chip
        self.params = chip.params
        self.learning_rate = learning_rate
        self.add_noise = add_noise
        self.noise_level_db = noise_level_db

        # Assumed chip parameters (may differ from actual)
        self.P_2pi_assumed = 0.75  # Assumed power for 2π phase shift (W)
        self.heater_resistance_assumed = 600  # Assumed heater resistance (Ω)

        # Storage for calibration history
        self.iteration_history = {
            "tap_amplitudes": [],
            "tap_phases": [],
            "mzi_phases": [],
            "ps_phases": [],
            "tap_amplitude_errors": [],
            "tap_phase_errors": [],
            "mzi_power_ratio_errors": [],
            "insertion_loss_spectra": [],
        }

        # Target tap coefficients
        self.target_tap_amplitudes = None
        self.target_tap_phases = None

    def set_target(self, tap_amplitudes: np.ndarray, tap_phases: np.ndarray):
        """
        Set the target tap coefficients for calibration.

        Parameters:
        -----------
        tap_amplitudes : np.ndarray
            Target tap amplitudes (length: n_signal_taps)
        tap_phases : np.ndarray
            Target tap phases in radians (length: n_signal_taps)
        """
        if len(tap_amplitudes) != self.params.n_signal_taps:
            raise ValueError(f"Expected {self.params.n_signal_taps} tap amplitudes")
        if len(tap_phases) != self.params.n_signal_taps:
            raise ValueError(f"Expected {self.params.n_signal_taps} tap phases")

        self.target_tap_amplitudes = tap_amplitudes.copy()
        self.target_tap_phases = tap_phases.copy()

    def measure_insertion_loss(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        """
        Simulate measuring the insertion loss spectrum of the chip.

        This simulates the measurement process with an optical vector analyzer
        or wavelength-swept laser + power meter.

        Parameters:
        -----------
        wavelengths_nm : np.ndarray
            Wavelength array in nanometres

        Returns:
        --------
        insertion_loss_db : np.ndarray
            Measured insertion loss in dB (with optional noise)
        """
        # Get true insertion loss from chip
        insertion_loss_db = self.chip.get_insertion_loss_spectrum(
            wavelengths_nm, port="calibration"
        )

        # Add measurement noise if enabled
        if self.add_noise:
            noise = np.random.uniform(
                -self.noise_level_db, self.noise_level_db, len(insertion_loss_db)
            )
            insertion_loss_db = insertion_loss_db + noise

        return insertion_loss_db

    def kramers_kronig_phase_recovery(
        self, insertion_loss_db: np.ndarray, frequencies: np.ndarray
    ) -> np.ndarray:
        """
        Recover phase response from insertion loss using Kramers-Kronig relationship.

        The Kramers-Kronig relationship allows recovery of phase from amplitude
        for minimum-phase systems. The chip is made minimum-phase by the reference
        path.

        Parameters:
        -----------
        insertion_loss_db : np.ndarray
            Measured insertion loss spectrum (dB)
        frequencies : np.ndarray
            Frequency array (Hz)

        Returns:
        --------
        phase_response : np.ndarray
            Recovered phase response (radians)
        """
        # Convert insertion loss (dB) to amplitude
        amplitude = 10 ** (insertion_loss_db / 20)

        # Take log of amplitude
        log_amplitude = np.log(amplitude + 1e-12)  # Add small value to avoid log(0)

        # Apply Hilbert transform to get phase
        # Phase = -Hilbert{log|H(ω)|}
        from scipy.signal import hilbert

        # The Hilbert transform in frequency domain
        # We need to be careful about the implementation
        analytic_signal = hilbert(log_amplitude)
        phase_response = -np.imag(analytic_signal)

        return phase_response

    def recover_tap_coefficients(
        self, insertion_loss_db: np.ndarray, frequencies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recover tap coefficients from insertion loss measurement.

        Process:
        1. Recover phase from amplitude using Kramers-Kronig
        2. Construct complex frequency response
        3. Inverse Fourier transform to get impulse response
        4. Extract tap coefficients

        Parameters:
        -----------
        insertion_loss_db : np.ndarray
            Measured insertion loss spectrum (dB)
        frequencies : np.ndarray
            Frequency array (Hz)

        Returns:
        --------
        (tap_amplitudes, tap_phases) : Tuple[np.ndarray, np.ndarray]
            Recovered tap amplitudes and phases for signal processing core
        """
        from scipy.fft import ifft, ifftshift

        # Step 1: Recover phase
        phase_response = self.kramers_kronig_phase_recovery(
            insertion_loss_db, frequencies
        )

        # Step 2: Construct complex frequency response
        amplitude = 10 ** (insertion_loss_db / 20)
        H_chip = amplitude * np.exp(1j * phase_response)

        # Step 3: Inverse Fourier transform to get impulse response
        df = frequencies[1] - frequencies[0]
        n_fft = len(frequencies)
        h_time = ifft(ifftshift(H_chip)) * n_fft * df

        # Time array
        t_max = 1 / (2 * df)
        time_array = np.linspace(-t_max, t_max, n_fft, endpoint=False)

        # Step 4: Extract tap coefficients at tap delays
        # For calibration port: includes all 16 taps
        all_tap_coeffs = np.zeros(self.params.n_taps, dtype=complex)

        for i in range(self.params.n_taps):
            tap_delay = i * self.params.delay_step
            idx = np.argmin(np.abs(time_array - tap_delay))
            all_tap_coeffs[i] = h_time[idx]

        # Extract signal processing core taps (taps 8-15, indices 8-15)
        start_idx = self.params.n_unused + 1
        signal_tap_coeffs = all_tap_coeffs[
            start_idx : start_idx + self.params.n_signal_taps
        ]

        # Extract amplitudes and phases
        tap_amplitudes = np.abs(signal_tap_coeffs)
        tap_phases = np.angle(signal_tap_coeffs)

        return tap_amplitudes, tap_phases

    def compute_errors(
        self, measured_tap_amplitudes: np.ndarray, measured_tap_phases: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute errors between measured and target tap coefficients.

        Parameters:
        -----------
        measured_tap_amplitudes : np.ndarray
            Measured tap amplitudes
        measured_tap_phases : np.ndarray
            Measured tap phases (radians)

        Returns:
        --------
        errors : Dict[str, np.ndarray]
            Dictionary containing:
            - 'amplitude_errors': Amplitude errors
            - 'phase_errors': Phase errors (radians)
            - 'mzi_power_ratio_errors': Power splitting ratio errors for MZIs
        """
        if self.target_tap_amplitudes is None or self.target_tap_phases is None:
            raise ValueError(
                "Target tap coefficients not set. Call set_target() first."
            )

        # Amplitude errors
        amplitude_errors = self.target_tap_amplitudes - measured_tap_amplitudes

        # Phase errors (handle wraparound)
        phase_errors = self.target_tap_phases - measured_tap_phases
        # Wrap to [-π, π]
        phase_errors = np.arctan2(np.sin(phase_errors), np.cos(phase_errors))

        # MZI power ratio errors (derived from tap amplitudes)
        # This requires understanding the binary tree structure
        mzi_power_ratio_errors = self._compute_mzi_errors(measured_tap_amplitudes)

        errors = {
            "amplitude_errors": amplitude_errors,
            "phase_errors": phase_errors,
            "mzi_power_ratio_errors": mzi_power_ratio_errors,
        }

        return errors

    def _compute_mzi_errors(self, measured_tap_amplitudes: np.ndarray) -> np.ndarray:
        """
        Compute MZI power splitting ratio errors from tap amplitudes.

        The binary tree structure determines how tap amplitudes relate to
        MZI power splitting ratios.

        Parameters:
        -----------
        measured_tap_amplitudes : np.ndarray
            Measured tap amplitudes

        Returns:
        --------
        mzi_errors : np.ndarray
            Power splitting ratio errors for 7 MZIs (in dB)
        """
        # Target and measured powers
        target_powers = self.target_tap_amplitudes**2
        measured_powers = measured_tap_amplitudes**2

        # Compute power splitting ratios for each stage
        # Stage 1: MZI 2-1 controls [taps 0-3] vs [taps 4-7]
        target_ratio_1 = np.sum(target_powers[0:4]) / np.sum(target_powers[4:8])
        measured_ratio_1 = np.sum(measured_powers[0:4]) / np.sum(measured_powers[4:8])
        error_1 = 10 * np.log10(target_ratio_1) - 10 * np.log10(measured_ratio_1)

        # Stage 2: MZI 3-3 controls [taps 0-1] vs [taps 2-3]
        target_ratio_2 = np.sum(target_powers[0:2]) / np.sum(target_powers[2:4])
        measured_ratio_2 = np.sum(measured_powers[0:2]) / np.sum(measured_powers[2:4])
        error_2 = 10 * np.log10(target_ratio_2 + 1e-12) - 10 * np.log10(
            measured_ratio_2 + 1e-12
        )

        # Stage 2: MZI 3-4 controls [taps 4-5] vs [taps 6-7]
        target_ratio_3 = np.sum(target_powers[4:6]) / np.sum(target_powers[6:8])
        measured_ratio_3 = np.sum(measured_powers[4:6]) / np.sum(measured_powers[6:8])
        error_3 = 10 * np.log10(target_ratio_3 + 1e-12) - 10 * np.log10(
            measured_ratio_3 + 1e-12
        )

        # Stage 3: MZI 4-5, 4-6, 4-7, 4-8 control individual tap pairs
        error_4 = 10 * np.log10(
            target_powers[0] / (target_powers[1] + 1e-12)
        ) - 10 * np.log10(measured_powers[0] / (measured_powers[1] + 1e-12))
        error_5 = 10 * np.log10(
            target_powers[2] / (target_powers[3] + 1e-12)
        ) - 10 * np.log10(measured_powers[2] / (measured_powers[3] + 1e-12))
        error_6 = 10 * np.log10(
            target_powers[4] / (target_powers[5] + 1e-12)
        ) - 10 * np.log10(measured_powers[4] / (measured_powers[5] + 1e-12))
        error_7 = 10 * np.log10(
            target_powers[6] / (target_powers[7] + 1e-12)
        ) - 10 * np.log10(measured_powers[6] / (measured_powers[7] + 1e-12))

        mzi_errors = np.array(
            [error_1, error_2, error_3, error_4, error_5, error_6, error_7]
        )

        return mzi_errors

    def update_parameters(self, errors: Dict[str, np.ndarray]):
        """
        Update chip parameters based on computed errors.

        This simulates the control algorithm that adjusts MZI and phase shifter
        settings to minimize errors.

        Parameters:
        -----------
        errors : Dict[str, np.ndarray]
            Dictionary containing amplitude, phase, and MZI errors
        """
        # Update MZI phases (controls tap amplitudes)
        # TODO: Implement MZI update logic

        # Update phase shifter phases (controls tap phases)
        phase_updates = self.learning_rate * errors["phase_errors"]

        # Get current phase shifter phases
        current_phases = np.array(
            [
                self.chip.phase_shifters[i + self.params.n_unused + 1].phase
                for i in range(self.params.n_signal_taps)
            ]
        )

        # Apply updates
        new_phases = current_phases + phase_updates
        self.chip.set_signal_tap_phases(new_phases)

        # Update MZIs (simplified - in practice this is more complex)
        # TODO: Implement proper MZI update based on power splitting ratios

    def run_calibration(
        self, max_iterations: int = 25, convergence_threshold: float = 0.01
    ) -> Dict:
        """
        Run the complete calibration process.

        Parameters:
        -----------
        max_iterations : int
            Maximum number of calibration iterations
        convergence_threshold : float
            Convergence threshold for phase errors (radians)

        Returns:
        --------
        results : Dict
            Dictionary containing calibration results and history
        """
        if self.target_tap_amplitudes is None:
            raise ValueError("Target not set. Call set_target() first.")

        # Measurement wavelengths (one FSR)
        wavelength_center = self.params.center_wavelength * 1e9  # nm
        wavelengths = wavelength_center + np.linspace(-0.4, 0.4, 1000)
        frequencies = 3e8 / (wavelengths * 1e-9)

        print(f"\nStarting calibration...")
        print(f"Max iterations: {max_iterations}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Convergence threshold: {convergence_threshold} rad")

        for iteration in range(max_iterations):
            # Step 1: Measure insertion loss
            insertion_loss = self.measure_insertion_loss(wavelengths)

            # Step 2: Recover tap coefficients
            tap_amps, tap_phases = self.recover_tap_coefficients(
                insertion_loss, frequencies
            )

            # Step 3: Compute errors
            errors = self.compute_errors(tap_amps, tap_phases)

            # Step 4: Store history
            self._store_iteration(
                iteration, tap_amps, tap_phases, errors, insertion_loss
            )

            # Step 5: Check convergence
            phase_error_rms = np.sqrt(np.mean(errors["phase_errors"] ** 2))

            print(
                f"Iteration {iteration+1:2d}: Phase RMS error = {phase_error_rms:.6f} rad"
            )

            if phase_error_rms < convergence_threshold:
                print(f"\nConverged after {iteration+1} iterations!")
                break

            # Step 6: Update parameters
            self.update_parameters(errors)

        results = {
            "converged": phase_error_rms < convergence_threshold,
            "final_iteration": iteration + 1,
            "final_phase_error": phase_error_rms,
            "history": self.iteration_history,
        }

        return results

    def _store_iteration(
        self,
        iteration: int,
        tap_amplitudes: np.ndarray,
        tap_phases: np.ndarray,
        errors: Dict[str, np.ndarray],
        insertion_loss: np.ndarray,
    ):
        """Store iteration data in history."""
        self.iteration_history["tap_amplitudes"].append(tap_amplitudes.copy())
        self.iteration_history["tap_phases"].append(tap_phases.copy())
        self.iteration_history["tap_amplitude_errors"].append(
            errors["amplitude_errors"].copy()
        )
        self.iteration_history["tap_phase_errors"].append(errors["phase_errors"].copy())
        self.iteration_history["mzi_power_ratio_errors"].append(
            errors["mzi_power_ratio_errors"].copy()
        )
        self.iteration_history["insertion_loss_spectra"].append(insertion_loss.copy())

    def plot_calibration_history(self, output_path: str = "/mnt/user-data/outputs"):
        """
        Plot the calibration history showing convergence.

        Parameters:
        -----------
        output_path : str
            Directory to save plots
        """
        history = self.iteration_history
        n_iterations = len(history["tap_phases"])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Tap amplitude convergence
        ax = axes[0, 0]
        tap_amps = np.array(history["tap_amplitudes"])
        for i in range(self.params.n_signal_taps):
            ax.plot(
                range(n_iterations),
                tap_amps[:, i],
                "-o",
                markersize=3,
                label=f"Tap {i+9}",
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Tap Amplitude")
        ax.set_title("Tap Amplitude Convergence")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Plot 2: Tap phase convergence
        ax = axes[0, 1]
        tap_phases = np.array(history["tap_phases"])
        for i in range(self.params.n_signal_taps):
            ax.plot(
                range(n_iterations),
                tap_phases[:, i],
                "-o",
                markersize=3,
                label=f"Tap {i+9}",
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Tap Phase (rad)")
        ax.set_title("Tap Phase Convergence")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Plot 3: Phase errors
        ax = axes[1, 0]
        phase_errors = np.array(history["tap_phase_errors"])
        phase_error_rms = np.sqrt(np.mean(phase_errors**2, axis=1))
        ax.semilogy(range(n_iterations), phase_error_rms, "b-o", markersize=5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Phase RMS Error (rad)")
        ax.set_title("Phase Error Convergence")
        ax.grid(True, alpha=0.3)

        # Plot 4: MZI power ratio errors
        ax = axes[1, 1]
        mzi_errors = np.array(history["mzi_power_ratio_errors"])
        for i in range(7):
            ax.plot(
                range(n_iterations),
                mzi_errors[:, i],
                "-o",
                markersize=3,
                label=f"MZI {i+1}",
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Power Ratio Error (dB)")
        ax.set_title("MZI Power Ratio Errors")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        plt.tight_layout()
        filename = f"{output_path}/calibration_convergence.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved: {filename}")
        plt.close()
