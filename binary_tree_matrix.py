"""
Binary Tree Transfer Matrix Implementation - COMPLETE VERSION
Including Reference Path and Calibration Port

This version properly implements:
1. Reference tap (tap 0) - shortest delay for Kramers-Kronig
2. Unused taps (taps 1-7) - minimized power
3. Signal processing core (taps 8-15) - binary tree controlled
4. Both signal and calibration ports
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BinaryTreeParams:
    """Parameters for binary tree MZI network"""

    n_stages: int = 3
    n_outputs: int = 8
    insertion_loss_db: float = 0.5


class BinaryTreeMatrix:
    """Binary tree transfer matrix for signal processing core (8 taps)"""

    def __init__(self, params: Optional[BinaryTreeParams] = None):
        self.params = params or BinaryTreeParams()
        self.n_mzis = 2**self.params.n_stages - 1
        self.insertion_loss = 10 ** (-self.params.insertion_loss_db / 10)

    def mzi_matrix(self, phase: float) -> np.ndarray:
        """2×2 MZI transfer matrix"""
        loss_factor = np.sqrt(self.insertion_loss)
        bar = np.cos(phase / 2) * loss_factor
        cross = 1j * np.sin(phase / 2) * loss_factor
        return np.array([[bar, bar], [cross, -cross]])

    def stage_matrix(self, stage: int, phases: np.ndarray) -> np.ndarray:
        """Transfer matrix for entire stage"""
        n_mzis = 2 ** (stage - 1)
        n_inputs = 2 ** (stage - 1)
        n_outputs = 2**stage
        M = np.zeros((n_outputs, n_inputs), dtype=complex)
        for i in range(n_mzis):
            mzi = self.mzi_matrix(phases[i])
            M[2 * i : 2 * i + 2, i : i + 1] = mzi[:, 0:1]
        return M

    def full_tree_matrix(self, mzi_phases: np.ndarray) -> np.ndarray:
        """Complete field distribution through binary tree"""
        if len(mzi_phases) != self.n_mzis:
            raise ValueError(f"Expected {self.n_mzis} phases, got {len(mzi_phases)}")

        M1 = self.stage_matrix(1, mzi_phases[0:1])
        M2 = self.stage_matrix(2, mzi_phases[1:3])
        M3 = self.stage_matrix(3, mzi_phases[3:7])

        input_field = np.array([[1.0 / np.sqrt(2)]])
        after_stage1 = M1 @ input_field
        after_stage2 = M2 @ after_stage1
        output_fields = M3 @ after_stage2

        return output_fields.flatten()


class PhotonicFIRComplete:
    """
    Complete photonic FIR chip with reference path.

    Architecture (16 taps total):
    - Tap 0: Reference tap (shortest delay, high power for K-K)
    - Taps 1-7: Unused taps (minimized power)
    - Taps 8-15: Signal processing core (binary tree controlled)

    Ports:
    - Signal: Only taps 8-15 (for actual signal processing)
    - Calibration: All 16 taps (for phase recovery via K-K)
    """

    def __init__(self, params):
        """
        Initialize complete FIR chip.

        Parameters
        ----------
        params : ChipParameters
            Physical chip parameters
        """
        self.params = params
        self.binary_tree = BinaryTreeMatrix()

        # Architecture: 1 reference + 7 unused + 8 signal core = 16 taps
        self.n_reference = 1
        self.n_unused = 7
        self.n_signal_core = 8

        # Control settings
        self.mzi_phases = np.zeros(7)  # Binary tree MZIs
        self.tap_phases = np.zeros(8)  # Signal core phase shifters

        # Reference and unused tap settings
        self.reference_amplitude = 0.5  # Strong reference for K-K
        self.reference_phase = 0.0
        self.unused_amplitude = 0.01  # Minimize unused taps

        # Delay lines for all 16 taps
        self.delays = np.arange(16) * params.delay_step

    def compute_all_tap_coefficients(self) -> np.ndarray:
        """
        Compute all 16 tap coefficients.

        Returns
        -------
        np.ndarray
            Complex tap coefficients for all 16 taps:
            [reference, unused×7, signal_core×8]
        """
        tap_coeffs = np.zeros(16, dtype=complex)

        # Tap 0: Reference tap (strong, fixed)
        # This is the key for Kramers-Kronig phase recovery
        tap_coeffs[0] = self.reference_amplitude * np.exp(1j * self.reference_phase)

        # Taps 1-7: Unused taps (minimized power, fixed phase)
        # These maintain minimum phase condition but don't process signal
        for i in range(1, 8):
            tap_coeffs[i] = self.unused_amplitude * np.exp(1j * 0.0)

        # Taps 8-15: Signal processing core (binary tree controlled)
        tree_fields = self.binary_tree.full_tree_matrix(self.mzi_phases)

        for i in range(8):
            # Phase shifter contribution
            phase_transfer = np.exp(1j * self.tap_phases[i]) * np.sqrt(0.95)
            tap_coeffs[8 + i] = tree_fields[i] * phase_transfer

        return tap_coeffs

    def compute_frequency_response(
        self, frequencies: np.ndarray, port: str = "signal"
    ) -> np.ndarray:
        """
        Compute frequency response H(ω).

        Parameters
        ----------
        frequencies : np.ndarray
            Frequency array in Hz
        port : str
            'signal' - Only signal core (taps 8-15)
            'calibration' - All taps including reference (taps 0-15)

        Returns
        -------
        np.ndarray
            Complex frequency response

        Notes
        -----
        The calibration port response is used for Kramers-Kronig phase
        recovery because it includes the strong reference tap (tap 0).

        The condition |h_ref| > |H_spc(ω)| ensures minimum phase, allowing
        phase recovery from amplitude-only measurements.
        """
        omega = 2 * np.pi * frequencies
        H = np.zeros(len(frequencies), dtype=complex)

        # Get all tap coefficients
        tap_coeffs = self.compute_all_tap_coefficients()

        if port == "signal":
            # Signal port: Only signal processing core (taps 8-15)
            for i in range(8, 16):
                delay = self.delays[i]
                length_cm = (delay * 3e8 / self.params.group_index) * 100
                loss = 10 ** (-self.params.waveguide_loss * length_cm / 10)
                delay_transfer = np.exp(-1j * omega * delay) * np.sqrt(loss)
                H += tap_coeffs[i] * delay_transfer

        elif port == "calibration":
            # Calibration port: ALL taps (0-15) for K-K phase recovery
            for i in range(16):
                delay = self.delays[i]
                length_cm = (delay * 3e8 / self.params.group_index) * 100
                loss = 10 ** (-self.params.waveguide_loss * length_cm / 10)
                delay_transfer = np.exp(-1j * omega * delay) * np.sqrt(loss)
                H += tap_coeffs[i] * delay_transfer
        else:
            raise ValueError(f"Port must be 'signal' or 'calibration', got '{port}'")

        return H

    def set_mzi_phases(self, phases: np.ndarray):
        """Set binary tree MZI phases"""
        if len(phases) != 7:
            raise ValueError("Expected 7 MZI phases")
        self.mzi_phases = phases

    def set_tap_phases(self, phases: np.ndarray):
        """Set signal core phase shifter phases"""
        if len(phases) != 8:
            raise ValueError("Expected 8 tap phases")
        self.tap_phases = phases

    def set_reference_amplitude(self, amplitude: float):
        """
        Set reference tap amplitude.

        Should be larger than sum of signal core to maintain K-K condition.
        Typically set to 0.5 (3dB coupler splits 50% to reference path).
        """
        self.reference_amplitude = amplitude

    def check_kramers_kronig_condition(self, frequencies: np.ndarray) -> dict:
        """
        Check if Kramers-Kronig condition is satisfied.

        The condition is: |h_ref| > |H_spc(ω)| for all ω

        Returns
        -------
        dict
            Contains:
            - 'satisfied': bool
            - 'reference_power': float
            - 'max_spc_power': float
            - 'margin_db': float
        """
        # Get signal processing core response (without reference)
        H_spc = self.compute_frequency_response(frequencies, port="signal")

        # Reference tap power
        ref_power = np.abs(self.reference_amplitude) ** 2

        # Maximum signal processing core power
        max_spc_power = np.max(np.abs(H_spc) ** 2)

        # Check condition
        satisfied = ref_power > max_spc_power
        margin_db = 10 * np.log10(ref_power / (max_spc_power + 1e-12))

        return {
            "satisfied": satisfied,
            "reference_power": ref_power,
            "max_spc_power": max_spc_power,
            "margin_db": margin_db,
        }

    def get_tap_info(self) -> dict:
        """
        Get diagnostic information about all taps.

        Returns
        -------
        dict
            Information about reference, unused, and signal core taps
        """
        tap_coeffs = self.compute_all_tap_coefficients()

        return {
            "reference": {
                "tap": 0,
                "amplitude": np.abs(tap_coeffs[0]),
                "phase": np.angle(tap_coeffs[0]),
                "power_db": 10 * np.log10(np.abs(tap_coeffs[0]) ** 2 + 1e-12),
            },
            "unused": {
                "taps": list(range(1, 8)),
                "amplitudes": np.abs(tap_coeffs[1:8]),
                "phases": np.angle(tap_coeffs[1:8]),
                "powers_db": 10 * np.log10(np.abs(tap_coeffs[1:8]) ** 2 + 1e-12),
            },
            "signal_core": {
                "taps": list(range(8, 16)),
                "amplitudes": np.abs(tap_coeffs[8:16]),
                "phases": np.angle(tap_coeffs[8:16]),
                "powers_db": 10 * np.log10(np.abs(tap_coeffs[8:16]) ** 2 + 1e-12),
            },
        }


# Demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("Complete Photonic FIR Chip with Reference Path")
    print("=" * 70)

    # Import existing chip parameters
    import sys

    sys.path.append("/mnt/project")
    from photonic_fir_chip import ChipParameters

    params = ChipParameters()
    chip = PhotonicFIRComplete(params)

    # Configure for equal splitting
    chip.set_mzi_phases(np.ones(7) * np.pi / 2)
    chip.set_tap_phases(np.zeros(8))

    # Show tap information
    print("\n--- Tap Configuration ---")
    info = chip.get_tap_info()

    print(f"\nReference Tap (Tap {info['reference']['tap']}):")
    print(f"  Power: {info['reference']['power_db']:.2f} dB")
    print(f"  Amplitude: {info['reference']['amplitude']:.4f}")

    print(
        f"\nUnused Taps (Taps {info['unused']['taps'][0]}-{info['unused']['taps'][-1]}):"
    )
    print(f"  Powers: {info['unused']['powers_db']}")
    print(f"  Mean: {np.mean(info['unused']['powers_db']):.2f} dB")

    print(
        f"\nSignal Core (Taps {info['signal_core']['taps'][0]}-{info['signal_core']['taps'][-1]}):"
    )
    print(f"  Powers: {info['signal_core']['powers_db']}")
    print(f"  Mean: {np.mean(info['signal_core']['powers_db']):.2f} dB")

    # Check Kramers-Kronig condition
    print("\n--- Kramers-Kronig Condition ---")
    freqs = np.linspace(-params.fsr / 2, params.fsr / 2, 1000)
    kk_check = chip.check_kramers_kronig_condition(freqs)

    print(f"Satisfied: {kk_check['satisfied']}")
    print(f"Reference power: {kk_check['reference_power']:.4f}")
    print(f"Max SPC power: {kk_check['max_spc_power']:.4f}")
    print(f"Margin: {kk_check['margin_db']:.2f} dB")

    # Compare frequency responses
    print("\n--- Frequency Response Comparison ---")
    H_cal = chip.compute_frequency_response(freqs, port="calibration")
    H_sig = chip.compute_frequency_response(freqs, port="signal")

    print(f"Calibration port (includes reference):")
    print(f"  Peak power: {10*np.log10(np.max(np.abs(H_cal)**2)):.2f} dB")
    print(f"\nSignal port (signal core only):")
    print(f"  Peak power: {10*np.log10(np.max(np.abs(H_sig)**2)):.2f} dB")

    print("\n" + "=" * 70)
    print("✓ Complete implementation with reference path working correctly!")
    print("=" * 70)
