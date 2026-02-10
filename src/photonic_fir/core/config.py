import logging

logger = logging.getLogger(__name__)

"""Configuration loading and saving utilities."""

from pathlib import Path
import yaml

from .data_structure import ExperimentConfig, config_from_dict, config_to_dict


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    ExperimentConfig
        Loaded experiment configuration.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_from_dict(config_dict)


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types.

    Parameters
    ----------
    obj : any
        Object to convert (dict, list, numpy type, etc.)

    Returns
    -------
    any
        Object with numpy types converted to Python native types.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_config(
    config: ExperimentConfig,
    output_dir: str,
    config_filename: str = "experiment_config",
) -> Path:
    """Save configuration to output directory.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to save.
    output_dir : str
        Directory to save configuration to. Will be created if it doesn't exist.

    Returns
    -------
    Path
        Path to the saved configuration file.

    Raises
    ------
    ValueError
        If output_dir exists as a file rather than a directory.
    """
    output_path_obj = Path(output_dir)

    # Check if path exists as a file
    if output_path_obj.exists() and output_path_obj.is_file():
        raise ValueError(f"'{output_dir}' exists as a file, not a directory")

    output_path_obj.mkdir(parents=True, exist_ok=True)

    config_dict = convert_numpy_types(config_to_dict(config))
    output_path = output_path_obj / f"{config_filename}.yaml"

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to {output_path}")
    return output_path
