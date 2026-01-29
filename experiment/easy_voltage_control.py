"""
Easy Voltage Control CLI
"""

import argparse
import sys
import numpy as np
from voltage_ctrl import VoltageController


def parse_channel_range(channel_str):
    """Parse channel specification like '1-16' or '1,2,3' or '1-5,8,10-12'"""
    channels = []
    for part in channel_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            channels.extend(range(start, end + 1))
        else:
            channels.append(int(part))
    return channels


def main():
    parser = argparse.ArgumentParser(
        description="Control voltages on specified channels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Set channel 5 to 10V
            python easy_voltage_control.py -c 5 -v 10
            
            # Set channels 1-16 to 5V each
            python easy_voltage_control.py -c 1-16 -v 5
            
            # Set channels 1,2,3 to 5V, 6V, 7V respectively
            python easy_voltage_control.py -c 1,2,3 -v 5,6,7
            
            # Set channels 17-32 to linearly spaced voltages from 1 to 16V
            python easy_voltage_control.py -c 17-32 -v 1:16
        """,
    )

    parser.add_argument(
        "-c",
        "--channels",
        required=True,
        help="Channel(s): single (5), range (1-16), list (1,2,3), or mixed (1-5,8,10-12)",
    )
    parser.add_argument(
        "-v",
        "--voltages",
        required=True,
        help="Voltage(s): single (5.0), list (5.0,6.0,7.0), or range (1:16 for linspace)",
    )
    parser.add_argument("--com-port", default="COM3", help="COM port (default: COM3)")
    parser.add_argument(
        "--baud-rate", type=int, default=9700, help="Baud rate (default: 9700)"
    )
    parser.add_argument(
        "--v-max", type=float, default=30.0, help="Maximum voltage limit (default: 30V)"
    )
    parser.add_argument(
        "--no-zero-on-exit", action="store_true", help="Don't zero voltages on exit"
    )

    args = parser.parse_args()

    # Parse channels
    channels = parse_channel_range(args.channels)

    # Parse voltages
    if ":" in args.voltages:
        # Linspace notation: "1:16" means np.linspace(1, 16, len(channels))
        start, end = map(float, args.voltages.split(":"))
        voltages = np.linspace(start, end, len(channels))
    elif "," in args.voltages:
        # List of voltages
        voltages = [float(v) for v in args.voltages.split(",")]
    else:
        # Single voltage - apply to all channels
        voltages = [float(args.voltages)] * len(channels)

    # Validate
    if len(voltages) != len(channels) and len(voltages) != 1:
        print(
            f"Error: Number of voltages ({len(voltages)}) must match number of channels ({len(channels)}) or be 1"
        )
        sys.exit(1)

    if len(voltages) == 1:
        voltages = voltages * len(channels)

    print(f"Setting voltages on {len(channels)} channel(s):")
    for ch, v in zip(channels, voltages):
        print(f"  Channel {ch}: {v:.2f}V")

    # Apply voltages
    with VoltageController(
        com_port=args.com_port,
        baud_rate=args.baud_rate,
        zero_on_exit=not args.no_zero_on_exit,
    ) as vc:
        vc.set_voltages(
            channels=channels,
            voltages=voltages,
            v_max=args.v_max,
        )

    print("Done.")


if __name__ == "__main__":
    main()
