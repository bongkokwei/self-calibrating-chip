from fiberlabs_edfa import EDFAController, DrivingMode
from voltage_ctrl import VoltageController
from luna_ova import LunaOVA

with LunaOVA(ip="130.194.137.122") as ova:
    # Perform full measurement
    data = ova.measure_full(
        center_wavelength_nm=1550,
        wavelength_range_nm=4,
        num_averages=1,
    )

# Connect to EDFA
with EDFAController("COM6", baudrate=9600) as edfa:
    # Get device info
    print(edfa.get_identification())

    # Monitor parameters
    output_levels = edfa.get_output_level()
    print(f"Output levels: {output_levels} dBm")

    # Set to ALC mode
    edfa.set_driving_mode(1, DrivingMode.ALC)
    edfa.set_alc_output_level(1, 13.0)  # dBm

    # Activate output
    edfa.set_output_active(True)

    # Deactivate output
    edfa.set_output_active(False)

# Define your channel numbers
channels = [8, 9, 10, 11, 12, 13, 14, 15]

# Create controller instance
controller = VoltageController(
    channels=channels,
    com_port="COM3",
    baud_rate=9600,
)

# Set voltages for all channels
voltages = [5.0, 3.3, 2.5, 1.8, 4.2, 3.0, 2.8, 3.5]  # Volts
resistance = 50.0  # Load resistance in ohms
v_max = 10.0  # Maximum voltage limit

controller.set_voltages(voltages, resistance, v_max)
