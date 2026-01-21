"""
Hardware Module
===============

Interfaces for controlling photonic chip hardware and measurement instruments.

This module requires hardware-specific libraries:
    - fiberlabs-edfa: EDFA controller
    - luna-ova: Luna OVA optical spectrum analyser
    - voltage-ctrl: Voltage controller for chip heaters

Submodules:
    measurement: Spectral measurement with Luna OVA
    voltage_control: Voltage adjustment and application to chip
"""

try:
    from .measurement import measure_spectrum

    from .voltage_adjustment import (
        calculate_power_adjustments,
        apply_voltages_to_hardware,
    )

    _HARDWARE_AVAILABLE = True

except ImportError as e:
    _HARDWARE_AVAILABLE = False
    _IMPORT_ERROR = str(e)

    # Provide helpful error message
    def _missing_hardware(*args, **kwargs):
        raise ImportError(
            f"Hardware libraries not available: {_IMPORT_ERROR}\n"
            "Install with: pip install fiberlabs-edfa luna-ova voltage-ctrl"
        )

    measure_spectrum = _missing_hardware
    calculate_power_adjustments = _missing_hardware
    apply_voltages_to_hardware = _missing_hardware

__all__ = [
    "measure_spectrum",
    "calculate_power_adjustments",
    "apply_voltages_to_hardware",
]


# Expose hardware availability status
def is_hardware_available() -> bool:
    """Check if hardware libraries are available."""
    return _HARDWARE_AVAILABLE


__all__.append("is_hardware_available")
