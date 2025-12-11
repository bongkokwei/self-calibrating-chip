from fiberlabs_edfa import EDFAController, DrivingMode
from voltage_ctrl import VoltageController
from luna_ova import LunaOVA

# # Define your channel numbers
# channels = [8, 9, 10, 11, 12, 13, 14, 15]
# with VoltageController(channels=channels, com_port="COM3") as controller:
#     # Display channel information
#     info = controller.get_channel_info()
#     print("\n=== Power Supply Configuration ===")
#     print(f"Channels: {info['channels']}")
#     print(f"Number of channels: {info['num_channels']}")
#     print(f"DAC resolution: {info['dac_resolution']} bits")
#     print(f"Voltage full scale: {info['voltage_full_scale']} V")
#     print(f"Voltage per bit: {info['voltage_per_bit']:.4f} V")

#     # Set voltages
#     voltages = [5.0, 3.3, 2.5, 1.8, 4.2, 3.0, 2.8, 3.5]
#     resistance = 50.0  # ohms
#     v_max = 10.0  # volts
#     controller.set_voltages(voltages, resistance, v_max)
# # Voltages automatically zeroed and connection closed here

# Connect to EDFA
with EDFAController("COM6", baudrate=57600) as edfa:
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

    with LunaOVA(ip="130.194.137.122") as ova:
        # Perform full measurement
        data = ova.measure_full(
            center_wavelength_nm=1550,
            wavelength_range_nm=4,
            num_averages=1,
        )

    # Deactivate output
    edfa.set_output_active(False)
