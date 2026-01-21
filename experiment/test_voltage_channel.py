"""
Test voltage channel configuration
"""

from photonic_fir import ExperimentConfig
from voltage_ctrl import VoltageController


def main():
    config = ExperimentConfig()

    # Test channel mapping
    print("\nVoltage Channel Mapping:")
    all_channels = config.chip.channel_mapping.get_all_channels()
    for device_id, channel in sorted(all_channels.items()):
        print(f"  {device_id:<15} → Channel {channel}")

    # Validate no duplicates
    try:
        config.chip.channel_mapping.validate_no_duplicates()
        print("\n✓ Channel mapping validated (no duplicates)")
    except ValueError as e:
        print(f"\n✗ Channel mapping error: {e}")

    channels = [1, 2, 3, 4, 5, 6, 7, 8]
    voltages = [1.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    with VoltageController(
        com_port="COM3",
        baud_rate=9600,
        zero_on_exit=True,
    ) as vc:
        vc.set_voltages(channels, voltages)

    return 0


if __name__ == "__main__":
    main()
