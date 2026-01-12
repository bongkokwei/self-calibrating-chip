import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fiberlabs_edfa import EDFAController, DrivingMode
from voltage_ctrl import VoltageController
from luna_ova import LunaOVA

from utils import measure_and_save_spectrum, extract_tap_coefficients


def main():

    # Load instrument configurations
    # Initialise instruments with config files

    # For Loop:
    # 1. measure spectrum
    # 2. extract tap coefficients
    # 3. calculate power splitting ratios error
    # 4. Calculate voltage adjustments
    # 5. Apply voltage adjustments
    # 6. break if error < threshold

    return
