"""
Run calibration experiment
"""

import sys
from pathlib import Path

# Import from installed package (absolute imports)
from photonic_fir import (
    ExperimentConfig,
    ChipParameters,
)

from photonic_fir.calibration import (
    run_experiment,
    load_config,
)

from photonic_fir.hardware import is_hardware_available


def main():
    # Check hardware availability
    if not is_hardware_available():
        print("WARNING: Hardware libraries not available. Cannot run experiment.")
        sys.exit(1)

    # Get config path from command line or use default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default to config in same directory
        config_path ="measurements/calibration_config.yaml"

    print(f"Loading config from: {config_path}")

    # Run experiment
    results = run_experiment(str(config_path))

    print(f"\nExperiment complete!")
    print(f"Converged: {results.converged}")
    print(f"Final iteration: {results.final_iteration}")


if __name__ == "__main__":
    main()
