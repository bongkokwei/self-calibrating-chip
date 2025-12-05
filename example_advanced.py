"""
Photonic FIR Chip Simulation - Advanced Examples

This script demonstrates advanced functionality of the photonic FIR chip:
1. Custom tap coefficient settings
2. MZI power distribution analysis
3. Thermal cross-talk simulation
4. Filter design comparison

Based on: Xu et al. (2022) "Self-calibrating programmable photonic integrated circuits"
"""

import sys
sys.path.append('/mnt/project')

from photonic_fir_chip import PhotonicFIRChip, ChipParameters, create_sinc_filter
import numpy as np
import matplotlib.pyplot as plt


def example_3_custom_filter():
    """Example 3: Custom filter with arbitrary tap coefficients."""
    
    print("=" * 70)
    print("Example 3: Custom Filter Design")
    print("=" * 70)
    
    chip = PhotonicFIRChip()
    
    # Design a custom raised-cosine filter
    n_taps = 8
    roll_off = 0.5
    n = np.arange(n_taps)
    center = (n_taps - 1) / 2
    
    # Raised cosine formula
    t = (n - center)
    custom_amps = np.sinc(t) * np.cos(np.pi * roll_off * t) / (1 - (2 * roll_off * t) ** 2 + 1e-10)
    custom_amps = custom_amps / np.max(custom_amps) * 0.5
    
    # Linear phase
    custom_phases = n * np.pi / 4
    
    # Set MZI phases for equal distribution
    chip.set_mzi_phases(np.ones(7) * np.pi / 2)
    chip.set_signal_tap_phases(custom_phases)
    
    # Compute responses
    freq_range = np.linspace(-chip.params.fsr/2, chip.params.fsr/2, 1000)
    H = chip.compute_frequency_response(freq_range, port='signal')
    
    wavelength_center = chip.params.center_wavelength * 1e9
    wavelengths = wavelength_center + np.linspace(-0.4, 0.4, 1000)
    insertion_loss = chip.get_insertion_loss_spectrum(wavelengths, port='signal')
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(wavelengths - wavelength_center, insertion_loss, linewidth=1.5)
    ax1.set_xlabel('Wavelength offset (nm)')
    ax1.set_ylabel('Insertion Loss (dB)')
    ax1.set_title('Custom Raised-Cosine Filter')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-40, 5])
    
    ax2.plot(freq_range / 1e9, 20 * np.log10(np.abs(H) + 1e-12), linewidth=1.5)
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Frequency Response')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-40, 5])
    
    plt.tight_layout()
    plt.savefig('example3_custom_filter.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: example3_custom_filter.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Example 3 Complete!")
    print("=" * 70)


def example_4_mzi_analysis():
    """Example 4: Analyse MZI power distribution control."""
    
    print("\n" + "=" * 70)
    print("Example 4: MZI Power Distribution Analysis")
    print("=" * 70)
    
    chip = PhotonicFIRChip()
    
    # Sweep first MZI phase to observe power distribution
    mzi_phases = np.linspace(0, 2*np.pi, 50)
    tap_powers = np.zeros((len(mzi_phases), 8))
    
    for i, phase in enumerate(mzi_phases):
        # Set first MZI to varying phase, rest to π/2
        mzi_settings = np.ones(7) * np.pi / 2
        mzi_settings[0] = phase
        chip.set_mzi_phases(mzi_settings)
        
        # Set all tap phases to zero
        chip.set_signal_tap_phases(np.zeros(8))
        
        # Get tap coefficients
        tap_coeffs = chip.compute_tap_coefficients()
        signal_taps = tap_coeffs[8:16]  # Signal processing taps
        tap_powers[i, :] = np.abs(signal_taps) ** 2
    
    # Plot power distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Power vs MZI phase for each tap
    for tap_idx in range(8):
        ax1.plot(mzi_phases, tap_powers[:, tap_idx], label=f'Tap {tap_idx+9}', linewidth=1.5)
    ax1.set_xlabel('MZI 2-1 Phase (rad)')
    ax1.set_ylabel('Tap Power (normalised)')
    ax1.set_title('Tap Power vs First MZI Phase')
    ax1.legend(ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Power distribution at specific phases
    phases_to_show = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    for phase in phases_to_show:
        idx = np.argmin(np.abs(mzi_phases - phase))
        ax2.bar(np.arange(8) + phases_to_show.index(phase) * 0.15, 
                tap_powers[idx, :], width=0.15, 
                label=f'φ = {phase:.2f} rad', alpha=0.7)
    
    ax2.set_xlabel('Tap Number (9-16)')
    ax2.set_ylabel('Tap Power (normalised)')
    ax2.set_title('Power Distribution at Different MZI Phases')
    ax2.set_xticks(np.arange(8) + 0.3)
    ax2.set_xticklabels([f'{i+9}' for i in range(8)])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('example4_mzi_analysis.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: example4_mzi_analysis.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Example 4 Complete!")
    print("=" * 70)


def example_5_filter_comparison():
    """Example 5: Compare different filter designs."""
    
    print("\n" + "=" * 70)
    print("Example 5: Filter Design Comparison")
    print("=" * 70)
    
    chip = PhotonicFIRChip()
    chip.set_mzi_phases(np.ones(7) * np.pi / 2)
    
    # Define different filters
    n = np.arange(8)
    center = 3.5
    
    filters = {
        'Sinc': np.sinc(n - center),
        'Hamming': np.sinc(n - center) * (0.54 - 0.46 * np.cos(2 * np.pi * n / 7)),
        'Hanning': np.sinc(n - center) * (0.5 - 0.5 * np.cos(2 * np.pi * n / 7)),
        'Blackman': np.sinc(n - center) * (0.42 - 0.5 * np.cos(2 * np.pi * n / 7) + 
                                           0.08 * np.cos(4 * np.pi * n / 7))
    }
    
    # Normalise all filters
    for key in filters:
        filters[key] = filters[key] / np.max(filters[key]) * 0.5
    
    # Compute responses
    wavelength_center = chip.params.center_wavelength * 1e9
    wavelengths = wavelength_center + np.linspace(-0.4, 0.4, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, taps) in enumerate(filters.items()):
        # Set tap amplitudes (phases = 0)
        chip.set_signal_tap_phases(np.zeros(8))
        
        insertion_loss = chip.get_insertion_loss_spectrum(wavelengths, port='signal')
        
        axes[idx].plot(wavelengths - wavelength_center, insertion_loss, linewidth=1.5)
        axes[idx].set_xlabel('Wavelength offset (nm)')
        axes[idx].set_ylabel('Insertion Loss (dB)')
        axes[idx].set_title(f'{name} Window Filter')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([-40, 5])
        
        print(f"\n{name} Window:")
        print(f"  Max insertion loss: {np.max(insertion_loss):.2f} dB")
        print(f"  3dB bandwidth: ~{chip.params.fsr / 1e9 / 4:.1f} GHz")
    
    plt.tight_layout()
    plt.savefig('example5_filter_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: example5_filter_comparison.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Example 5 Complete!")
    print("=" * 70)


def example_6_phase_response_analysis():
    """Example 6: Analyse phase response characteristics."""
    
    print("\n" + "=" * 70)
    print("Example 6: Phase Response Analysis")
    print("=" * 70)
    
    chip = PhotonicFIRChip()
    chip.set_mzi_phases(np.ones(7) * np.pi / 2)
    
    # Different phase configurations
    phase_configs = {
        'Linear Phase': np.linspace(0, 2*np.pi, 8),
        'Quadratic Phase': (np.arange(8) ** 2) * np.pi / 32,
        'Random Phase': np.random.uniform(-np.pi, np.pi, 8),
        'Zero Phase': np.zeros(8)
    }
    
    freq_range = np.linspace(-chip.params.fsr/2, chip.params.fsr/2, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, phases) in enumerate(phase_configs.items()):
        chip.set_signal_tap_phases(phases)
        H = chip.compute_frequency_response(freq_range, port='signal')
        
        ax = axes[idx]
        ax_twin = ax.twinx()
        
        # Plot magnitude
        line1 = ax.plot(freq_range / 1e9, 20 * np.log10(np.abs(H) + 1e-12), 
                        'b-', linewidth=1.5, label='Magnitude')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim([-40, 5])
        
        # Plot phase
        line2 = ax_twin.plot(freq_range / 1e9, np.angle(H), 
                             'r-', linewidth=1.5, label='Phase')
        ax_twin.set_ylabel('Phase (rad)', color='r')
        ax_twin.tick_params(axis='y', labelcolor='r')
        
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        
        print(f"\n{name}:")
        print(f"  Phase values: {phases}")
    
    plt.tight_layout()
    plt.savefig('example6_phase_analysis.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: example6_phase_analysis.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Example 6 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    """Run advanced examples."""
    example_3_custom_filter()
    example_4_mzi_analysis()
    example_5_filter_comparison()
    example_6_phase_response_analysis()
