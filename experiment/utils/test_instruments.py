from fiberlabs_edfa import EDFAController, DrivingMode
from voltage_ctrl import VoltageController
from luna_ova import LunaOVA


def configure_voltage_controller(
    channels,
    voltages,
    resistance=50.0,
    v_max=10.0,
    com_port="COM3",
    zero_on_exit=True,
):
    """
    Configure voltage controller channels with specified voltages.

    Parameters:
    -----------
    channels : list
        List of channel numbers to configure
    voltages : list
        List of voltage values to set for each channel
    resistance : float
        Load resistance in ohms (default: 50.0)
    v_max : float
        Maximum voltage limit in volts (default: 10.0)
    com_port : str
        COM port for voltage controller (default: "COM3")
    """
    with VoltageController(
        channels=channels,
        com_port=com_port,
        zero_on_exit=zero_on_exit,
    ) as controller:
        # Display channel information
        info = controller.get_channel_info()
        print("\n=== Voltage Controller Configuration ===")
        print(f"Channels: {info['channels']}")
        print(f"Number of channels: {info['num_channels']}")
        print(f"DAC resolution: {info['dac_resolution']} bits")
        print(f"Voltage full scale: {info['voltage_full_scale']} V")
        print(f"Voltage per bit: {info['voltage_per_bit']:.4f} V")

        # Set voltages
        controller.set_voltages(voltages, resistance, v_max)
        print(f"✓ Voltages set successfully")


def configure_edfa(output_power_dbm=13.0, activate=True, com_port="COM6"):
    """
    Configure EDFA settings.

    Parameters:
    -----------
    output_power_dbm : float
        EDFA output power in dBm (default: 13.0)
    activate : bool
        Whether to activate the output (default: True)
    com_port : str
        COM port for EDFA (default: "COM6")

    Returns:
    --------
    EDFAController : Context manager for EDFA control
    """
    edfa = EDFAController(com_port, baudrate=57600)
    edfa.__enter__()  # Enter context manager

    try:
        # Get device info
        print("\n=== EDFA Configuration ===")
        print(edfa.get_identification())

        # Monitor parameters
        output_levels = edfa.get_output_level()
        print(f"Current output levels: {output_levels} dBm")

        # Set to ALC mode
        edfa.set_driving_mode(1, DrivingMode.ALC)
        edfa.set_alc_output_level(1, output_power_dbm)
        print(f"Set to ALC mode at {output_power_dbm} dBm")

        # Activate/deactivate output
        edfa.set_output_active(activate)
        status = "activated" if activate else "deactivated"
        print(f"✓ EDFA output {status}")

        return edfa

    except Exception as e:
        edfa.__exit__(None, None, None)
        raise e


def measure_with_ova(
    center_wavelength_nm=1550,
    wavelength_range_nm=4,
    num_averages=1,
    ip="130.194.137.122",
):
    """
    Perform optical spectrum measurement with Luna OVA.

    Parameters:
    -----------
    center_wavelength_nm : float
        Center wavelength for measurement in nm (default: 1550)
    wavelength_range_nm : float
        Wavelength range in nm (default: 4)
    num_averages : int
        Number of measurement averages (default: 1)
    ip : str
        IP address for Luna OVA (default: "130.194.137.122")

    Returns:
    --------
    dict : Measurement data from OVA
    """
    with LunaOVA(ip=ip) as ova:
        print("\n=== OVA Measurement ===")
        print(f"Center wavelength: {center_wavelength_nm} nm")
        print(f"Wavelength range: {wavelength_range_nm} nm")
        print(f"Number of averages: {num_averages}")

        data = ova.measure_full(
            center_wavelength_nm=center_wavelength_nm,
            wavelength_range_nm=wavelength_range_nm,
            num_averages=num_averages,
        )

        print(f"✓ Measurement complete")
        return data


# Main execution
if __name__ == "__main__":
    # Define channels and voltages
    channels = [8, 9, 10, 11, 12, 13, 14, 15]
    voltages = [5.0, 3.3, 2.5, 1.8, 4.2, 3.0, 2.8, 3.5]

    # Step 1: Configure voltage controller
    configure_voltage_controller(
        channels=channels,
        voltages=voltages,
        resistance=50.0,
        v_max=10.0,
        zero_on_exit=True,
    )

    # Step 2: Configure and activate EDFA
    edfa = configure_edfa(output_power_dbm=13.0, activate=True)

    try:
        # Step 3: Perform OVA measurement
        measurement_data = measure_with_ova(
            center_wavelength_nm=1550, wavelength_range_nm=4, num_averages=1
        )

        print("\n=== All Operations Complete ===")

    finally:
        # Always deactivate EDFA output and close connection
        if edfa:
            edfa.set_output_active(False)
            edfa.__exit__(None, None, None)
            print("✓ EDFA output deactivated and connection closed")
