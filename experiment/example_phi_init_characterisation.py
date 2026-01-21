"""
example_phi_init_characterisation.py

Example script demonstrating how to integrate the two-step φ_init characterisation
with the existing photonic_fir package and hardware control.
"""

import numpy as np
from pathlib import Path
from typing import Dict

# Imports from your existing package structure
# from photonic_fir.core.data_structure import ChipState, ExperimentConfig
# from photonic_fir.hardware.voltage_control import VoltageController
# from photonic_fir.measurement.spectrum_acquisition import measure_spectrum
# from photonic_fir.core.power_splitting_ratio import calculate_measured_psr

# Import the two-step characterisation method
from mzi_init_phase_characterisation import (
    two_step_phi_init_characterisation,
    characterise_all_mzis,
    extract_phi_init_dict,
)


def example_single_mzi_characterisation():
    """
    Example: Characterise a single MZI's φ_init.
    
    This demonstrates the core two-step method for one MZI.
    """
    
    print("EXAMPLE 1: Single MZI Characterisation")
    print("=" * 70)
    
    # In practice, you would connect to real hardware:
    # voltage_controller = VoltageController(port="COM3", baudrate=9600)
    # ova = OVA5000(address="130.194.137.122")
    # edfa = FiberlabsEDFA(port="COM6", baudrate=57600)
    
    # For this example, we'll use mock functions
    mzi_id = "2-1"
    
    def measure_psr() -> float:
        """
        Measure spectrum and extract PSR for the MZI being characterised.
        
        In practice:
        1. Acquire spectrum with OVA
        2. Apply Kramers-Kronig phase recovery
        3. Inverse Fourier transform to get tap coefficients
        4. Calculate PSR from tap powers using binary tree relationships
        """
        # Mock: simulate PSR measurement based on current power
        # In reality: return calculate_measured_psr(spectrum_data, mzi_id)
        return -15.0  # Example PSR in dB
    
    def apply_power(mzi_id: str, power_watts: float) -> None:
        """
        Apply power to the MZI's thermal heater.
        
        In practice:
        1. Look up voltage channel for this MZI from channel_mapping
        2. Convert power to voltage: V = sqrt(P * R_heater)
        3. Set voltage controller
        """
        # Mock implementation
        # In reality:
        # channel = chip_state.get_device_channel(f"MZI_{mzi_id}")
        # voltage = power_to_voltage(power_watts, r_heater=100.0)
        # voltage_controller.set_voltage(channel, voltage)
        print(f"    Applied {power_watts:.3f} W to MZI {mzi_id}")
    
    # Run characterisation
    result = two_step_phi_init_characterisation(
        mzi_id=mzi_id,
        measure_psr_callback=measure_psr,
        apply_power_callback=apply_power,
        power_sweep_step=0.05,
        max_power=1.0,
        p2pi_nominal=0.75,
    )
    
    print(f"\nResult:")
    print(f"  φ_init = {result.phi_init_rad:.3f} rad ({np.degrees(result.phi_init_rad):.1f}°)")
    print(f"  P_null = {result.power_at_null_watts:.3f} W")
    print(f"  PSR_null = {result.psr_at_null_db:.2f} dB")
    
    return result


def example_full_chip_characterisation():
    """
    Example: Characterise all MZIs in the 8-tap signal processing core.
    
    This would be run once before starting the main calibration loop.
    """
    
    print("\n\nEXAMPLE 2: Full Chip Characterisation")
    print("=" * 70)
    
    # MZI IDs in 8-tap signal processing core (3 stages, 7 MZIs)
    # Binary tree structure: stage 2 (1 MZI), stage 3 (2 MZIs), stage 4 (4 MZIs)
    mzi_ids = [
        "2-1",  # Stage 2
        "3-3", "3-4",  # Stage 3
        "4-5", "4-6", "4-7", "4-8",  # Stage 4
    ]
    
    def measure_psr_for_mzi(mzi_id: str) -> float:
        """
        Measure and extract PSR for a specific MZI.
        
        In practice, this function:
        1. Acquires full spectrum
        2. Performs phase recovery
        3. Gets tap coefficients
        4. Calculates PSR for the requested MZI using binary tree relationships
        
        For example, MZI 3-4 controls:
            PSR_3-4 = (|h[13]|² + |h[14]|²) / (|h[15]|² + |h[16]|²)
        """
        # Mock implementation
        # In reality:
        # spectrum = ova.measure_spectrum()
        # taps = kramers_kronig_recovery(spectrum)
        # psr = calculate_measured_psr(taps, mzi_tree, mzi_id)
        # return psr[mzi_id]
        
        return np.random.uniform(-20, -5)  # Mock PSR in dB
    
    def apply_power(mzi_id: str, power_watts: float) -> None:
        """Apply power to specific MZI."""
        # Implementation same as single MZI example
        pass
    
    # Run characterisation for all MZIs
    results = characterise_all_mzis(
        mzi_ids=mzi_ids,
        measure_psr_for_mzi_callback=measure_psr_for_mzi,
        apply_power_callback=apply_power,
        power_sweep_step=0.05,
        max_power=1.0,
        p2pi_nominal=0.75,
    )
    
    # Extract just the phi_init values for use in calibration
    phi_init_dict = extract_phi_init_dict(results)
    
    print("\n\nInitial Phase Offsets for Calibration:")
    print("-" * 70)
    for mzi_id, phi_init in phi_init_dict.items():
        print(f"  MZI {mzi_id}: {phi_init:6.3f} rad ({np.degrees(phi_init):6.1f}°)")
    
    return phi_init_dict


def example_integration_with_chipstate():
    """
    Example: How to integrate characterisation results with ChipState.
    
    This shows how to update your ChipState dataclass with the measured
    φ_init values before starting calibration.
    """
    
    print("\n\nEXAMPLE 3: Integration with ChipState")
    print("=" * 70)
    
    # Assume we've run characterisation
    phi_init_dict = {
        "2-1": 0.523,
        "3-3": -0.785,
        "3-4": 1.047,
        "4-5": -0.314,
        "4-6": 0.872,
        "4-7": -1.152,
        "4-8": 0.628,
    }
    
    print("Updating ChipState with characterised φ_init values:")
    print()
    
    # In practice, you would do:
    # for mzi_id, phi_init in phi_init_dict.items():
    #     chip_state.mzis[mzi_id].phi_init_rad = phi_init
    #     print(f"  Updated MZI {mzi_id}: φ_init = {phi_init:.3f} rad")
    
    # Mock for demonstration
    for mzi_id, phi_init in phi_init_dict.items():
        print(f"  chip_state.mzis['{mzi_id}'].phi_init_rad = {phi_init:.3f} rad")
    
    print("\nChipState is now ready for calibration with accurate φ_init values!")
    
    # The calibration loop can now proceed:
    # for iteration in range(max_iterations):
    #     iter_data = calibration_iteration(chip_state, config, ...)
    #     if check_convergence(iter_data, config):
    #         break


def example_integration_workflow():
    """
    Complete workflow showing when and how to use φ_init characterisation.
    """
    
    print("\n\nEXAMPLE 4: Complete Calibration Workflow")
    print("=" * 70)
    
    print("""
Complete Self-Calibration Workflow
-----------------------------------

1. INITIALISE HARDWARE
   - Connect to voltage controller, OVA, EDFA
   - Set EDFA output power
   - Stabilise chip temperature
   
2. CHARACTERISE φ_init (TWO-STEP METHOD) ← THIS NEW CODE
   - For each MZI in signal processing core:
     a) Sweep power to find intensity null
     b) Apply small offset to determine branch
     c) Calculate φ_init from null position
   - Update ChipState with measured φ_init values
   - Store results for reference
   
3. SET TARGET FILTER
   - Define desired tap coefficients (sinc, Hilbert, etc.)
   - Calculate target PSRs and phases
   
4. INITIALISE CHIP STATE
   - Set all MZIs and phase shifters to zero power
   - Use characterised φ_init values
   
5. MAIN CALIBRATION LOOP
   - Measure spectrum
   - Recover phase (Kramers-Kronig)
   - Calculate tap coefficients (inverse FFT)
   - Calculate errors (MZI PSR & phase shifter phase)
   - Update powers with convergence rules:
     a) If PSR_err increases > 0.2 dB: add π to φ_init
     b) If P < 0: wrap by adding P_2π
   - Apply new voltages
   - Check convergence
   - Repeat until converged or max iterations
   
6. VALIDATE RESULTS
   - Measure final spectrum
   - Verify tap amplitudes and phases
   - Calculate RMS errors
   - Generate calibration report
""")
    
    print("Key Advantage of Two-Step Method:")
    print("-" * 70)
    print("""
By characterising φ_init BEFORE calibration:
  ✓ Calibration loop converges faster (fewer iterations)
  ✓ Power adjustments are more accurate
  ✓ Avoids oscillations from incorrect phase predictions
  ✓ Reduces need for π-flipping convergence rules
  
Without φ_init characterisation:
  ✗ Calibration must discover φ_init implicitly
  ✗ More iterations required (potentially 20-30 vs 5-10)
  ✗ Higher chance of getting stuck in local minima
""")


def example_data_structure_compatibility():
    """
    Show how the characterisation results fit into your existing data structures.
    """
    
    print("\n\nEXAMPLE 5: Data Structure Compatibility")
    print("=" * 70)
    
    print("""
Your existing MZIState dataclass:

    @dataclass
    class MZIState:
        stage: int
        position: int
        phi_init_rad: float = 0.0  ← UPDATE THIS
        p2pi_watts: float = 0.75
        applied_power_watts: float = 0.0
        phase_shift_rad: float = 0.0

The two-step characterisation provides:
    - phi_init_rad: Initial phase offset (rad)
    - p2pi_watts: Verified power for 2π (often close to nominal 0.75 W)
    - power_at_null_watts: Power where PSR is minimum

Integration:
    
    # Run characterisation
    char_results = characterise_all_mzis(...)
    
    # Update ChipState
    for mzi_id, result in char_results.items():
        chip_state.mzis[mzi_id].phi_init_rad = result.phi_init_rad
        chip_state.mzis[mzi_id].p2pi_watts = result.p2pi_watts
    
    # Now ready for calibration_loop.py
    history = []
    for iteration in range(config.calibration.max_iterations):
        iter_data = calibration_iteration(
            chip_state=chip_state,
            config=config,
            prev_iter_data=history[-1] if history else None,
        )
        history.append(iter_data)
        
        if check_convergence(iter_data, config):
            print(f"Converged at iteration {iteration}")
            break
""")


if __name__ == "__main__":
    """Run all examples."""
    
    # Example 1: Single MZI
    example_single_mzi_characterisation()
    
    # Example 2: Full chip
    example_full_chip_characterisation()
    
    # Example 3: ChipState integration
    example_integration_with_chipstate()
    
    # Example 4: Complete workflow
    example_integration_workflow()
    
    # Example 5: Data structure compatibility
    example_data_structure_compatibility()
    
    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Adapt measure_psr_callback to your spectrum acquisition system")
    print("2. Adapt apply_power_callback to your voltage controller")
    print("3. Run characterisation before main calibration")
    print("4. Store results in your experiment configuration")
