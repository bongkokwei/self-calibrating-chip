import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys

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


def main():

    config = ExperimentConfig()
    print(config)

    return 0


if __name__ == "__main__":
    main()
