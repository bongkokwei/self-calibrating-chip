import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys
from pprint import pprint

from config import (
    ExperimentConfig,
    ChipState,
    IterationData,
    CalibrationResults,
    config_from_dict,
    config_to_dict,
    measure_spectrum,
    recover_impulse_response_from_df,
    detect_taps,
    calculate_all_errors,
    calculate_power_adjustments,
    apply_voltages_to_hardware,
)


def save_config(config: ExperimentConfig, output_dir: str):
    """Save configuration to output directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config_dict = config_to_dict(config)
    output_path = Path(output_dir) / "experiment_config.yaml"

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to {output_path}")


def main():

    config = ExperimentConfig()
    save_config(config, "")
    pprint(config)

    return 0


if __name__ == "__main__":
    main()
