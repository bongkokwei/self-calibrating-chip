"""
scan_common.py

Shared voltage-sweep scaffolding for the batch V_2pi characterisation scripts
(batch_mzi_scan.py, batch_ps_scan.py).

Handles the parts that are identical between MZI and PS scans: output
directory setup, OVA DUT-length calibration, per-voltage hardware control
and measurement, and the batch retry loop. The physics (nonlinear tan² fit
for MZIs vs linear regression for PSs), plotting, and per-device result
extraction differ per device type and stay in each script.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple, TypeVar

import numpy as np
import pandas as pd

from voltage_ctrl import VoltageController
from luna_ova import LunaOVA

from photonic_fir import ExperimentConfig
from photonic_fir.calibration import measure_and_detect_taps

T = TypeVar("T")


def sweep_device_voltage(
    device_id: str,
    scan_config,
    exp_config: ExperimentConfig,
    file_prefix: str,
    extract_result: Callable[[int, float, np.ndarray], T],
) -> Tuple[np.ndarray, List[T], List[pd.DataFrame]]:
    """
    Step device_id's channel through scan_config's voltage range, measuring
    and detecting taps at each point.

    Reference MZIs are held at exp_config.calibration.initial_mzi_voltages
    for the duration of the sweep; the VoltageController zeroes all heaters
    on each context exit.

    extract_result(i, voltage, tap_coeffs) is called after each measurement
    to compute (and print) whatever per-device metric(s) the caller needs;
    its return value is collected into the returned results list.

    scan_config must provide: get_voltage_range(), v_max, settling_time_sec,
    output_dir, save_raw_data.
    """
    output_path = Path(scan_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    voltage_range = scan_config.get_voltage_range()
    n_voltages = len(voltage_range)
    channel = exp_config.channel_mapping.get_channel(device_id)

    results: List[T] = []
    dataframes: List[pd.DataFrame] = []

    with LunaOVA(ip=exp_config.measurement.ova_address) as ova:
        ova.set_dut_length()

    for i, voltage in enumerate(voltage_range):
        print(f"[{i+1}/{n_voltages}] Voltage: {voltage:.3f} V (channel {channel})")

        with VoltageController(
            com_port=exp_config.measurement.voltage_controller_port,
            baud_rate=exp_config.measurement.voltage_controller_baudrate,
            zero_on_exit=True,
        ) as v_ctrl:
            init_mzi_channels = list(
                exp_config.calibration.initial_mzi_voltages.keys()
            )
            init_psu_channels = [
                exp_config.channel_mapping.get_channel(f"MZI_{mzi_id}")
                for mzi_id in init_mzi_channels
            ]
            init_mzi_voltages = list(
                exp_config.calibration.initial_mzi_voltages.values()
            )

            v_ctrl.set_voltages(
                init_psu_channels + [channel],
                init_mzi_voltages + [voltage],
                v_max=scan_config.v_max,
            )

            time.sleep(scan_config.settling_time_sec)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if scan_config.save_raw_data:
                file_name = f"{file_prefix}_{voltage:.3f}v_{timestamp}"
                folder_dir = str(output_path)
            else:
                file_name = None
                folder_dir = None

            df, tap_times, tap_coeffs, _, _ = measure_and_detect_taps(
                config=exp_config,
                file_name=file_name,
                folder_dir=folder_dir,
            )

            time.sleep(scan_config.settling_time_sec)

        dataframes.append(df)
        results.append(extract_result(i, voltage, tap_coeffs))

    print(f"✓ Sweep complete — voltage controller channels zeroed\n")
    return voltage_range, results, dataframes


def run_batch_characterisation(
    item_ids: List,
    item_label: str,
    characterise_fn: Callable[..., None],
    id_kwarg: str,
    characterise_kwargs: dict,
    settle_delay_sec: float = 5.0,
    format_item: Callable[..., str] = str,
) -> None:
    """
    Run characterise_fn once per item in item_ids, catching and logging
    failures so one bad device doesn't abort the whole batch, and inserting
    a thermal-settling delay between successful items.

    format_item(item_id) controls how the item is displayed in progress
    messages (e.g. "Tap 9" instead of the bare tap number).
    """
    for i, item_id in enumerate(item_ids):
        print(f"\n{'#'*70}")
        print(
            f"# CHARACTERISING {item_label} {i+1}/{len(item_ids)}: "
            f"{format_item(item_id)}"
        )
        print(f"{'#'*70}\n")

        try:
            characterise_fn(**{id_kwarg: item_id}, **characterise_kwargs)
        except Exception as e:
            print(f"⚠ FAILED to characterise {item_label} {format_item(item_id)}: {e}")
            print("Continuing with next...\n")
            continue

        if i < len(item_ids) - 1:
            print(f"\nWaiting {settle_delay_sec:.0f} seconds before next scan...\n")
            time.sleep(settle_delay_sec)
