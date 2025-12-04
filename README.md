# Photonic FIR Chip Simulation

A realistic Python simulation of a 16-tap programmable photonic finite impulse response (FIR) chip based on the research by Xu et al. (2022) published in *Nature Photonics*.

## Overview

This simulation models a self-calibrating programmable photonic integrated circuit (PIC) with the following features:

- **16-tap FIR filter architecture** with binary tree power distribution
- **8 signal processing taps** (taps 9-16) plus 1 reference tap and 7 unused taps
- **Realistic optical components**: Mach-Zehnder Interferometers (MZIs), phase shifters, directional couplers, and delay lines
- **Physical parameters**: waveguide losses, coupling losses, thermal effects
- **160 GHz free spectral range** with 6.25 ps delay steps

## Key Components

- `MZI`: Tunable Mach-Zehnder Interferometer with phase control
- `PhaseShifter`: Thermo-optic phase shifter
- `DirectionalCoupler`: 3 dB optical coupler (50:50 splitter)
- `DelayLine`: Spiral waveguide with progressive delays
- `PhotonicFIRChip`: Complete 16-tap programmable filter

## Quick Start

```python
from photonic_fir_chip import PhotonicFIRChip
import numpy as np

# Create chip instance
chip = PhotonicFIRChip()

# Set MZI phases for equal power distribution (π/2 for 50:50 split)
chip.set_mzi_phases(np.ones(7) * np.pi / 2)

# Set tap phases for signal processing
phases = np.linspace(0, 2*np.pi, 8)
chip.set_signal_tap_phases(phases)

# Compute frequency response
frequencies = np.linspace(-80e9, 80e9, 1000)
H = chip.compute_frequency_response(frequencies, port='signal')
```

## Main Features

**Analysis capabilities:**
- Frequency response (amplitude and phase)
- Impulse response
- Tap coefficient calculation
- Insertion loss spectrum measurements

**Applications:**
- Optical signal processing
- Programmable photonic filters
- Telecommunications
- Microwave photonics

## Example Output

The `main()` function demonstrates:
1. Creating sinc filters with various phase steps
2. Plotting comprehensive frequency/time domain responses
3. Visualising tap coefficients and insertion loss spectra

Run with: `python photonic_fir_chip.py`

## Reference

Based on: Xu, X., Ren, G., Feleppa, T., Liu, X., Boes, A., Mitchell, A., & Lowery, A.J. (2022). Self-calibrating programmable photonic integrated circuits. *Nature Photonics*, 16, 595-602.

## Requirements

- NumPy
- Matplotlib
- SciPy
