"""
Useful scripts that controls the experiment's instruments
"""

from .measure_spectrum import measure_and_save_spectrum
from .extract_tap_coefficients import extract_tap_coefficients
from .test_instruments import (
    configure_edfa,
    configure_voltage_controller,
    measure_with_ova,
)
