from .fir_response import FIRFrequencyResponse

# Note: compare_spectrum is not imported here. It pulls in
# photonic_fir.calibration, which requires the hardware control libraries
# (voltage_ctrl, luna_ova, fiberlabs_edfa) to be installed. Import it
# explicitly (`from simulation.compare_spectrum import compare_to_measured`)
# when you actually need to compare against a measured spectrum.

__all__ = [
    "FIRFrequencyResponse",
]
