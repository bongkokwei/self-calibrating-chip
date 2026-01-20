"""
example_usage.py

Example of how to use the data structures.
"""

import yaml
from .data_structure import (
    ExperimentConfig,
    ChipParameters,
    TargetFilter,
    config_to_dict,
    config_from_dict,
)


def create_default_config():
    """Create a default experiment configuration."""

    config = ExperimentConfig(
        name="test_sinc_filter", description="Test sinc filter calibration"
    )

    # Modify target filter
    config.target.filter_type = "sinc"
    config.target.phase_step_rad = 2 * 3.14159 / 7  # 2π/7

    return config


def save_config_to_yaml(config: ExperimentConfig, filepath: str):
    """Save configuration to YAML file."""

    config_dict = config_to_dict(config)

    with open(filepath, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to {filepath}")


def load_config_from_yaml(filepath: str) -> ExperimentConfig:
    """Load configuration from YAML file."""

    with open(filepath, "r") as f:
        config_dict = yaml.safe_load(f)

    config = config_from_dict(config_dict)

    print(f"Configuration loaded from {filepath}")
    return config


def example_function_using_config(config: ExperimentConfig):
    """Example function that uses the configuration."""

    print(f"\nExperiment: {config.name}")
    print(f"Number of taps: {config.chip.n_taps}")
    print(f"FSR: {config.chip.fsr_hz / 1e9:.1f} GHz")
    print(f"Target filter: {config.target.filter_type}")
    print(f"Phase step: {config.target.phase_step_rad:.4f} rad")
    print(f"Learning rate: {config.calibration.learning_rate}")
    print(f"Max iterations: {config.calibration.max_iterations}")

    # Access chip state
    print(f"\nNumber of MZIs: {len(config.initial_state.mzis)}")
    print(f"Number of phase shifters: {len(config.initial_state.phase_shifters)}")

    # Example: iterate over MZIs
    for mzi_id, mzi in config.initial_state.mzis.items():
        print(f"  MZI {mzi_id}: P2π = {mzi.p2pi_watts:.3f} W")


if __name__ == "__main__":
    # Create and save a default configuration
    config = create_default_config()
    save_config_to_yaml(config, "test_config.yaml")

    # Load configuration
    loaded_config = load_config_from_yaml("test_config.yaml")

    # Use configuration in a function
    example_function_using_config(loaded_config)
