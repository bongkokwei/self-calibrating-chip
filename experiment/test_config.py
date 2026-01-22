import yaml
from pprint import pprint

from pathlib import Path

from photonic_fir import (
    ChipParameters,
    ChipState,
    MZIState,
    PhaseShifterState,
    MeasurementConfig,
    CalibrationConfig,
    TargetFilter,
    ExperimentConfig,
    IterationData,
    CalibrationResults,
    VoltageChannelMapping,
    config_from_dict,
    config_to_dict,
    save_config,
    load_config,
    measure_spectrum,
)

config = ExperimentConfig(
    measurement=MeasurementConfig(
        num_averages=3,
        center_wavelength_nm=1550,
        wavelength_span_nm=42,
    ),
)
output_path = save_config(config, "measurements")
config_from_disk = load_config(output_path)

df = measure_spectrum(
    center_wavelength_nm=config_from_disk.measurement.center_wavelength_nm,
    wavelength_span_nm=config_from_disk.measurement.wavelength_span_nm,
    num_averages=config_from_disk.measurement.num_averages,
    edfa_port=config_from_disk.measurement.edfa_port,
    edfa_baudrate=config_from_disk.measurement.edfa_baudrate,
    edfa_output_power_dbm=config_from_disk.measurement.edfa_output_power_dbm,
    ova_ip=config_from_disk.measurement.ova_address,
    folder_dir="./measurements",
    file_name="_delete_me",
)
