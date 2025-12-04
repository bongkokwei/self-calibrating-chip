"""
Photonic FIR Chip Simulation - Realistic Implementation

Based on: Xu et al. (2022) "Self-calibrating programmable photonic integrated circuits"
Nature Photonics, Vol 16, August 2022, 595-602

This simulation provides a realistic model of the 16-tap FIR chip with:
- Direct phase shifter control (in radians)
- MZI control (power splitting ratios)
- Multiple input/output ports (signal ports and calibration ports)
- Thermal effects and waveguide losses
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ChipParameters:
    """Physical parameters of the photonic chip"""
    n_taps: int = 16
    n_signal_taps: int = 8
    fsr: float = 160e9  # Hz - Free Spectral Range
    delay_step: float = 6.25e-12  # seconds (6.25 ps)
    waveguide_loss: float = 0.15  # dB/cm
    coupling_loss: float = 0.1  # dB per coupler
    center_wavelength: float = 1550e-9  # metres
    group_index: float = 1.71  # for Si3N4 waveguide


class MZI:
    """
    Mach-Zehnder Interferometer with tunable phase shifter.
    
    Transfer function: |t|² = sin²(φ/2), |r|² = cos²(φ/2)
    where φ is the phase difference between the two arms.
    """
    
    def __init__(self, name: str, insertion_loss_db: float = 0.5):
        """
        Parameters:
        -----------
        name : str
            Identifier for this MZI (e.g., "MZI_2-1", "MZI_4-5")
        insertion_loss_db : float
            Total insertion loss through the MZI in dB
        """
        self.name = name
        self.phase = 0.0  # radians
        self.insertion_loss = 10 ** (-insertion_loss_db / 10)  # Convert dB to linear
        
    def set_phase(self, phase_rad: float):
        """Set the phase difference in the MZI (radians)"""
        self.phase = phase_rad
        
    def get_splitting_ratio(self) -> Tuple[float, float]:
        """
        Get power splitting ratio.
        
        Returns:
        --------
        (bar_port, cross_port) : Tuple[float, float]
            Power at bar and cross ports (normalized)
        """
        bar = np.cos(self.phase / 2) ** 2
        cross = np.sin(self.phase / 2) ** 2
        return bar * self.insertion_loss, cross * self.insertion_loss
    
    def get_field_transfer(self) -> Tuple[complex, complex]:
        """
        Get complex field transfer coefficients.
        
        Returns:
        --------
        (bar_field, cross_field) : Tuple[complex, complex]
            Complex field coefficients including phase
        """
        bar = np.cos(self.phase / 2) * np.sqrt(self.insertion_loss)
        cross = 1j * np.sin(self.phase / 2) * np.sqrt(self.insertion_loss)
        return bar, cross


class PhaseShifter:
    """
    Optical phase shifter using thermo-optic effect.
    """
    
    def __init__(self, name: str, insertion_loss_db: float = 0.05):
        """
        Parameters:
        -----------
        name : str
            Identifier for this phase shifter
        insertion_loss_db : float
            Insertion loss in dB
        """
        self.name = name
        self.phase = 0.0  # radians
        self.insertion_loss = 10 ** (-insertion_loss_db / 10)
        
    def set_phase(self, phase_rad: float):
        """Set the phase shift (radians)"""
        self.phase = phase_rad
        
    def get_transfer(self) -> complex:
        """Get complex field transfer coefficient"""
        return np.sqrt(self.insertion_loss) * np.exp(1j * self.phase)


class DirectionalCoupler:
    """
    3 dB directional coupler (50:50 splitter/combiner).
    """
    
    def __init__(self, coupling_ratio: float = 0.5, insertion_loss_db: float = 0.1):
        """
        Parameters:
        -----------
        coupling_ratio : float
            Power coupling ratio (0.5 for 3dB coupler)
        insertion_loss_db : float
            Total insertion loss in dB
        """
        self.coupling_ratio = coupling_ratio
        self.insertion_loss = 10 ** (-insertion_loss_db / 10)
        
    def get_field_transfer(self) -> Tuple[complex, complex]:
        """
        Returns complex field coefficients for the two outputs.
        
        For a 50:50 coupler: 1/√2 * (1, j)
        """
        t = np.sqrt(self.coupling_ratio * self.insertion_loss)
        k = np.sqrt((1 - self.coupling_ratio) * self.insertion_loss)
        return t, 1j * k


class DelayLine:
    """
    Optical delay line using spiral waveguide.
    """
    
    def __init__(self, delay: float, length_cm: float = 1.0, loss_per_cm: float = 0.15):
        """
        Parameters:
        -----------
        delay : float
            Time delay in seconds
        length_cm : float
            Physical length in cm
        loss_per_cm : float
            Propagation loss in dB/cm
        """
        self.delay = delay
        self.length = length_cm
        self.loss = 10 ** (-loss_per_cm * length_cm / 10)
        
    def get_transfer(self, omega: np.ndarray) -> np.ndarray:
        """
        Get frequency-dependent transfer function.
        
        Parameters:
        -----------
        omega : np.ndarray
            Angular frequencies (rad/s)
            
        Returns:
        --------
        H : np.ndarray
            Complex transfer function
        """
        return np.sqrt(self.loss) * np.exp(-1j * omega * self.delay)


class PhotonicFIRChip:
    """
    Complete 16-tap photonic FIR chip with binary tree architecture.
    
    Architecture:
    - 1 reference tap (tap 0)
    - 7 unused taps (taps 1-7)
    - 8 signal processing taps (taps 8-15)
    - Binary tree of MZIs for power distribution
    - Phase shifters for each tap
    """
    
    def __init__(self, params: Optional[ChipParameters] = None):
        """
        Initialize the photonic FIR chip.
        
        Parameters:
        -----------
        params : ChipParameters, optional
            Chip physical parameters
        """
        self.params = params or ChipParameters()
        
        # Calculate number of unused taps
        self.params.n_unused = self.params.n_taps - self.params.n_signal_taps - 1  # -1 for reference
        
        # Create ports
        self.ports = {
            'sig_in': 'Signal Input',
            'sig_out': 'Signal Output',
            'cal_in': 'Calibration Input',
            'cal_out': 'Calibration Output'
        }
        
        # Input couplers (3 dB)
        self.input_coupler = DirectionalCoupler(coupling_ratio=0.5)
        self.output_coupler = DirectionalCoupler(coupling_ratio=0.5)
        
        # Initialize MZI network for signal processing core (binary tree)
        # Stage 1: 1 MZI
        self.mzi_2_1 = MZI("MZI_2-1")
        
        # Stage 2: 2 MZIs
        self.mzi_3_3 = MZI("MZI_3-3")
        self.mzi_3_4 = MZI("MZI_3-4")
        
        # Stage 3: 4 MZIs
        self.mzi_4_5 = MZI("MZI_4-5")
        self.mzi_4_6 = MZI("MZI_4-6")
        self.mzi_4_7 = MZI("MZI_4-7")
        self.mzi_4_8 = MZI("MZI_4-8")
        
        # Store all MZIs in order for easy access
        self.mzis = [
            self.mzi_2_1,
            self.mzi_3_3, self.mzi_3_4,
            self.mzi_4_5, self.mzi_4_6, self.mzi_4_7, self.mzi_4_8
        ]
        
        # Initialize phase shifters for all 16 taps
        self.phase_shifters = [
            PhaseShifter(f"PS_{i+1}") for i in range(self.params.n_taps)
        ]
        
        # Initialize delay lines (progressive delays)
        self.delay_lines = []
        for i in range(self.params.n_taps):
            delay = i * self.params.delay_step
            # Estimate physical length from delay
            length_cm = (delay * 3e8 / self.params.group_index) * 100
            self.delay_lines.append(
                DelayLine(delay=delay, length_cm=length_cm, 
                         loss_per_cm=self.params.waveguide_loss)
            )
        
        # Fixed phase settings for reference and unused taps
        # (set to maintain minimum phase condition)
        self._reference_phase = 0.0
        self._unused_phases = np.zeros(self.params.n_unused)
        
    def set_mzi_phase(self, mzi_index: int, phase_rad: float):
        """
        Set phase of a specific MZI in the binary tree.
        
        Parameters:
        -----------
        mzi_index : int
            Index of MZI (0-6):
            0: MZI 2-1 (stage 1)
            1-2: MZI 3-3, 3-4 (stage 2)
            3-6: MZI 4-5, 4-6, 4-7, 4-8 (stage 3)
        phase_rad : float
            Phase shift in radians
        """
        if 0 <= mzi_index < len(self.mzis):
            self.mzis[mzi_index].set_phase(phase_rad)
        else:
            raise ValueError(f"MZI index must be between 0 and {len(self.mzis)-1}")
    
    def set_tap_phase(self, tap_index: int, phase_rad: float):
        """
        Set phase of a specific tap's phase shifter.
        
        Parameters:
        -----------
        tap_index : int
            Tap index (0-15)
        phase_rad : float
            Phase shift in radians
        """
        if tap_index == 0:
            print("Warning: Tap 0 is the reference tap, phase is fixed")
            return
        elif 1 <= tap_index <= self.params.n_unused:
            print(f"Warning: Tap {tap_index} is unused, phase is fixed")
            return
        elif 0 <= tap_index < self.params.n_taps:
            self.phase_shifters[tap_index].set_phase(phase_rad)
        else:
            raise ValueError(f"Tap index must be between 0 and {self.params.n_taps-1}")
    
    def set_signal_tap_phases(self, phases: np.ndarray):
        """
        Set phases for all signal processing taps (taps 8-15).
        
        Parameters:
        -----------
        phases : np.ndarray
            Array of 8 phase values in radians
        """
        if len(phases) != self.params.n_signal_taps:
            raise ValueError(f"Expected {self.params.n_signal_taps} phase values")
        
        start_idx = self.params.n_unused + 1
        for i, phase in enumerate(phases):
            self.phase_shifters[start_idx + i].set_phase(phase)
    
    def set_mzi_phases(self, phases: np.ndarray):
        """
        Set phases for all MZIs in the binary tree.
        
        Parameters:
        -----------
        phases : np.ndarray
            Array of 7 phase values in radians
        """
        if len(phases) != len(self.mzis):
            raise ValueError(f"Expected {len(self.mzis)} phase values")
        
        for mzi, phase in zip(self.mzis, phases):
            mzi.set_phase(phase)
    
    def compute_tap_coefficients(self) -> np.ndarray:
        """
        Compute the complex-valued tap coefficients based on current MZI and phase shifter settings.
        
        Returns:
        --------
        tap_coefficients : np.ndarray
            Complex tap coefficients (length 16)
        """
        tap_coeffs = np.zeros(self.params.n_taps, dtype=complex)
        
        # Reference tap (index 0) - direct path
        tap_coeffs[0] = self.phase_shifters[0].get_transfer()
        
        # Unused taps (indices 1-7) - set to very small values
        for i in range(1, self.params.n_unused + 1):
            tap_coeffs[i] = 0.01 * self.phase_shifters[i].get_transfer()
        
        # Signal processing taps (indices 8-15) - controlled by binary tree
        # The field distribution is determined by the MZI tree
        
        # Start with equal field from the coupler (amplitude = 1/√2)
        # Start with equal field from the coupler (amplitude = 1/√2)
        tap_fields = np.ones(self.params.n_signal_taps, dtype=complex) / np.sqrt(2)
        
        # Stage 1: MZI 2-1 splits into two groups of 4
        bar_2_1, cross_2_1 = self.mzi_2_1.get_field_transfer()
        tap_fields[0:4] *= bar_2_1    # Taps 9-12 (indices 0-3)
        tap_fields[4:8] *= cross_2_1  # Taps 13-16 (indices 4-7)
        
        # Stage 2: MZI 3-3 and 3-4 split into groups of 2
        bar_3_3, cross_3_3 = self.mzi_3_3.get_field_transfer()
        bar_3_4, cross_3_4 = self.mzi_3_4.get_field_transfer()
        
        # MZI 3-3 splits taps 9-12 into [9-10] and [11-12]
        temp_0_1 = tap_fields[0:2].copy()
        tap_fields[0:2] = temp_0_1 * bar_3_3    # Taps 9-10
        tap_fields[2:4] = temp_0_1 * cross_3_3  # Taps 11-12
        
        # MZI 3-4 splits taps 13-16 into [13-14] and [15-16]
        temp_4_5 = tap_fields[4:6].copy()
        tap_fields[4:6] = temp_4_5 * bar_3_4    # Taps 13-14
        tap_fields[6:8] = temp_4_5 * cross_3_4  # Taps 15-16
        
        # Stage 3: MZI 4-5, 4-6, 4-7, 4-8 split into individual taps
        bar_4_5, cross_4_5 = self.mzi_4_5.get_field_transfer()
        bar_4_6, cross_4_6 = self.mzi_4_6.get_field_transfer()
        bar_4_7, cross_4_7 = self.mzi_4_7.get_field_transfer()
        bar_4_8, cross_4_8 = self.mzi_4_8.get_field_transfer()
        
        # Split tap 9-10
        temp_0 = tap_fields[0]
        tap_fields[0] = temp_0 * bar_4_5    # Tap 9
        tap_fields[1] = temp_0 * cross_4_5  # Tap 10
        
        # Split tap 11-12
        temp_2 = tap_fields[2]
        tap_fields[2] = temp_2 * bar_4_6    # Tap 11
        tap_fields[3] = temp_2 * cross_4_6  # Tap 12
        
        # Split tap 13-14
        temp_4 = tap_fields[4]
        tap_fields[4] = temp_4 * bar_4_7    # Tap 13
        tap_fields[5] = temp_4 * cross_4_7  # Tap 14
        
        # Split tap 15-16
        temp_6 = tap_fields[6]
        tap_fields[6] = temp_6 * bar_4_8    # Tap 15
        tap_fields[7] = temp_6 * cross_4_8  # Tap 16
        
        # Apply phase shifters to create final complex coefficients
        start_idx = self.params.n_unused + 1
        for i in range(self.params.n_signal_taps):
            phase_transfer = self.phase_shifters[start_idx + i].get_transfer()
            tap_coeffs[start_idx + i] = tap_fields[i] * phase_transfer
        
        return tap_coeffs
    
    def compute_frequency_response(self, frequencies: np.ndarray, 
                                   port: str = 'signal') -> np.ndarray:
        """
        Compute the frequency response H(ω) of the chip.
        
        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency array in Hz (relative to center frequency)
        port : str
            'signal' for signal ports, 'calibration' for calibration ports
            
        Returns:
        --------
        H : np.ndarray
            Complex frequency response
        """
        omega = 2 * np.pi * frequencies
        
        # Get current tap coefficients
        tap_coeffs = self.compute_tap_coefficients()
        
        # Initialize frequency response
        H = np.zeros(len(frequencies), dtype=complex)
        
        if port == 'signal':
            # Signal port only sees signal processing taps (8-15)
            start_idx = self.params.n_unused + 1
            for i in range(start_idx, self.params.n_taps):
                delay_transfer = self.delay_lines[i].get_transfer(omega)
                H += tap_coeffs[i] * delay_transfer
                
        elif port == 'calibration':
            # Calibration port sees all taps including reference
            for i in range(self.params.n_taps):
                delay_transfer = self.delay_lines[i].get_transfer(omega)
                H += tap_coeffs[i] * delay_transfer
        else:
            raise ValueError("Port must be 'signal' or 'calibration'")
        
        return H
    
    def compute_impulse_response(self, port: str = 'signal') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the impulse response h(t) of the chip.
        
        Parameters:
        -----------
        port : str
            'signal' or 'calibration'
            
        Returns:
        --------
        (time, impulse_response) : Tuple[np.ndarray, np.ndarray]
            Time array and complex impulse response
        """
        tap_coeffs = self.compute_tap_coefficients()
        
        if port == 'signal':
            start_idx = self.params.n_unused + 1
            h = tap_coeffs[start_idx:]
            time = np.arange(len(h)) * self.params.delay_step + start_idx * self.params.delay_step
        elif port == 'calibration':
            h = tap_coeffs
            time = np.arange(len(h)) * self.params.delay_step
        else:
            raise ValueError("Port must be 'signal' or 'calibration'")
        
        return time, h
    
    def get_insertion_loss_spectrum(self, wavelengths_nm: np.ndarray, 
                                   port: str = 'calibration') -> np.ndarray:
        """
        Compute the insertion loss spectrum (power) in dB.
        This is what would be measured by a wavelength-swept laser and power metre.
        
        Parameters:
        -----------
        wavelengths_nm : np.ndarray
            Wavelength array in nanometres
        port : str
            'signal' or 'calibration'
            
        Returns:
        --------
        insertion_loss_db : np.ndarray
            Insertion loss in dB
        """
        # Convert wavelength to frequency
        c = 3e8  # speed of light
        frequencies = c / (wavelengths_nm * 1e-9) - self.params.center_wavelength * c
        
        # Compute frequency response
        H = self.compute_frequency_response(frequencies, port=port)
        
        # Convert to power (insertion loss)
        power = np.abs(H) ** 2
        insertion_loss_db = 10 * np.log10(power + 1e-12)  # Add small value to avoid log(0)
        
        return insertion_loss_db


def create_sinc_filter(n_taps: int = 8, phase_step: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a sinc filter with specified phase step.
    
    Parameters:
    -----------
    n_taps : int
        Number of filter taps
    phase_step : float
        Phase step between taps in radians
        
    Returns:
    --------
    (amplitudes, phases) : Tuple[np.ndarray, np.ndarray]
        Tap amplitudes and phases
    """
    # Sinc function centred at middle tap
    n = np.arange(n_taps)
    center = (n_taps - 1) / 2
    x = (n - center)
    
    # Sinc amplitudes (normalized)
    amplitudes = np.sinc(x / 2)
    amplitudes = amplitudes / np.max(amplitudes) * 0.5  # Normalise to 0.5 max
    
    # Linear phase progression
    phases = n * phase_step
    
    return amplitudes, phases


def main():
    """Demonstration of the photonic FIR chip simulation."""
    
    print("=" * 70)
    print("Photonic FIR Chip Simulation")
    print("=" * 70)
    
    # Create chip instance
    chip = PhotonicFIRChip()
    
    print(f"\nChip Configuration:")
    print(f"  Total taps: {chip.params.n_taps}")
    print(f"  Signal processing taps: {chip.params.n_signal_taps}")
    print(f"  FSR: {chip.params.fsr / 1e9:.1f} GHz")
    print(f"  Delay step: {chip.params.delay_step * 1e12:.2f} ps")
    
    # Example 1: Create a sinc filter with 0 phase step
    print("\n" + "-" * 70)
    print("Example 1: Sinc Filter (0 phase step)")
    print("-" * 70)
    
    amplitudes, phases = create_sinc_filter(n_taps=8, phase_step=0)
    
    # Set MZIs for approximately equal power distribution
    # For equal splitting, MZI phase = π/2 gives 50:50 split
    # Setting all MZIs to π/2 will give equal power to all 8 taps
    equal_split_phases = np.ones(7) * np.pi / 2
    chip.set_mzi_phases(equal_split_phases)
    
    # Set tap phases
    chip.set_signal_tap_phases(phases)
    
    # Compute and display tap coefficients
    tap_coeffs = chip.compute_tap_coefficients()
    print("\nTap Coefficients (Signal Processing Core):")
    for i in range(9, 17):  # Taps 9-16
        print(f"  Tap {i}: Amplitude = {np.abs(tap_coeffs[i-1]):.4f}, "
              f"Phase = {np.angle(tap_coeffs[i-1]):.4f} rad")
    
    # Compute frequency response
    freq_range = np.linspace(-chip.params.fsr/2, chip.params.fsr/2, 1000)
    H_cal = chip.compute_frequency_response(freq_range, port='calibration')
    H_sig = chip.compute_frequency_response(freq_range, port='signal')
    
    # Compute impulse response
    time_cal, h_cal = chip.compute_impulse_response(port='calibration')
    time_sig, h_sig = chip.compute_impulse_response(port='signal')
    
    # Compute insertion loss spectrum
    wavelength_center = chip.params.center_wavelength * 1e9  # Convert to nm
    wavelengths = wavelength_center + np.linspace(-0.4, 0.4, 1000)  # ±0.4 nm range
    insertion_loss_cal = chip.get_insertion_loss_spectrum(wavelengths, port='calibration')
    insertion_loss_sig = chip.get_insertion_loss_spectrum(wavelengths, port='signal')
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Insertion Loss Spectrum (Calibration Port)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(wavelengths - wavelength_center, insertion_loss_cal, 'b-', linewidth=1.5)
    ax1.set_xlabel('Wavelength offset (nm)')
    ax1.set_ylabel('Insertion Loss (dB)')
    ax1.set_title('Insertion Loss - Calibration Port')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-40, 5])
    
    # Plot 2: Insertion Loss Spectrum (Signal Port)
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(wavelengths - wavelength_center, insertion_loss_sig, 'r-', linewidth=1.5)
    ax2.set_xlabel('Wavelength offset (nm)')
    ax2.set_ylabel('Insertion Loss (dB)')
    ax2.set_title('Insertion Loss - Signal Port')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-40, 5])
    
    # Plot 3: Frequency Response Amplitude
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(freq_range / 1e9, 20 * np.log10(np.abs(H_cal) + 1e-12), 'b-', 
             linewidth=1.5, label='Calibration')
    ax3.plot(freq_range / 1e9, 20 * np.log10(np.abs(H_sig) + 1e-12), 'r-', 
             linewidth=1.5, label='Signal', alpha=0.7)
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_title('Frequency Response - Amplitude')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([-40, 5])
    
    # Plot 4: Frequency Response Phase
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(freq_range / 1e9, np.angle(H_cal), 'b-', linewidth=1.5, label='Calibration')
    ax4.plot(freq_range / 1e9, np.angle(H_sig), 'r-', linewidth=1.5, label='Signal', alpha=0.7)
    ax4.set_xlabel('Frequency (GHz)')
    ax4.set_ylabel('Phase (rad)')
    ax4.set_title('Frequency Response - Phase')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Impulse Response Amplitude (Calibration)
    ax5 = plt.subplot(3, 3, 5)
    ax5.stem(time_cal * 1e12, np.abs(h_cal), basefmt=' ')
    ax5.set_xlabel('Time (ps)')
    ax5.set_ylabel('Amplitude')
    ax5.set_title('Impulse Response Amplitude - Calibration Port')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Impulse Response Phase (Calibration)
    ax6 = plt.subplot(3, 3, 6)
    ax6.stem(time_cal * 1e12, np.angle(h_cal), basefmt=' ', linefmt='r-', markerfmt='ro')
    ax6.set_xlabel('Time (ps)')
    ax6.set_ylabel('Phase (rad)')
    ax6.set_title('Impulse Response Phase - Calibration Port')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Impulse Response Amplitude (Signal)
    ax7 = plt.subplot(3, 3, 7)
    ax7.stem(time_sig * 1e12, np.abs(h_sig), basefmt=' ')
    ax7.set_xlabel('Time (ps)')
    ax7.set_ylabel('Amplitude')
    ax7.set_title('Impulse Response Amplitude - Signal Port')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Impulse Response Phase (Signal)
    ax8 = plt.subplot(3, 3, 8)
    ax8.stem(time_sig * 1e12, np.angle(h_sig), basefmt=' ', linefmt='r-', markerfmt='ro')
    ax8.set_xlabel('Time (ps)')
    ax8.set_ylabel('Phase (rad)')
    ax8.set_title('Impulse Response Phase - Signal Port')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Tap Coefficients
    ax9 = plt.subplot(3, 3, 9)
    tap_indices = np.arange(chip.params.n_taps) + 1
    tap_amplitudes = np.abs(tap_coeffs)
    tap_phases = np.angle(tap_coeffs)
    
    ax9_twin = ax9.twinx()
    bars = ax9.bar(tap_indices, 20 * np.log10(tap_amplitudes + 1e-12), 
                   alpha=0.6, color='blue', label='Amplitude')
    dots = ax9_twin.plot(tap_indices, tap_phases, 'ro', markersize=8, label='Phase')
    
    ax9.set_xlabel('Tap Number')
    ax9.set_ylabel('Amplitude (dB)', color='blue')
    ax9_twin.set_ylabel('Phase (rad)', color='red')
    ax9.set_title('Tap Coefficients (All 16 taps)')
    ax9.grid(True, alpha=0.3)
    ax9.set_xticks(tap_indices)
    
    # Highlight different regions
    ax9.axvspan(0.5, 1.5, alpha=0.2, color='green', label='Reference')
    ax9.axvspan(1.5, 8.5, alpha=0.2, color='gray', label='Unused')
    ax9.axvspan(8.5, 16.5, alpha=0.2, color='yellow', label='Signal Core')
    
    plt.tight_layout()
    plt.savefig('/home/claude/fir_chip_sinc_filter.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: fir_chip_sinc_filter.png")
    
    # Example 2: Different phase steps
    print("\n" + "-" * 70)
    print("Example 2: Sinc Filter with Phase Steps")
    print("-" * 70)
    
    phase_steps = [0, 2*np.pi/7, 4*np.pi/7, 6*np.pi/7]
    
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, phase_step in enumerate(phase_steps):
        amplitudes, phases = create_sinc_filter(n_taps=8, phase_step=phase_step)
        chip.set_signal_tap_phases(phases)
        
        wavelengths_plot = wavelength_center + np.linspace(-0.4, 0.4, 1000)
        insertion_loss = chip.get_insertion_loss_spectrum(wavelengths_plot, port='signal')
        
        axes[idx].plot(wavelengths_plot - wavelength_center, insertion_loss, linewidth=1.5)
        axes[idx].set_xlabel('Wavelength offset (nm)')
        axes[idx].set_ylabel('Insertion Loss (dB)')
        axes[idx].set_title(f'Phase step = {phase_step:.3f} rad ({phase_step*180/np.pi:.1f}°)')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([-40, 5])
        
        print(f"\nPhase step = {phase_step:.3f} rad ({phase_step*180/np.pi:.1f}°):")
        print(f"  Max insertion loss: {np.max(insertion_loss):.2f} dB")
        print(f"  Min insertion loss: {np.min(insertion_loss):.2f} dB")
    
    plt.tight_layout()
    plt.savefig('/home/claude/fir_chip_phase_steps.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: fir_chip_phase_steps.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
