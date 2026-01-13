"""
data_structures.py

Data structures for photonic FIR chip parameters and experiment configuration.
Compatible with YAML serialization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class ChipParameters:
    """Physical parameters of the 16-tap FIR chip."""

    # Basic architecture
    n_taps: int = 16
    n_signal_taps: int = 8  # Taps 9-16
    reference_tap_idx: int = 0  # Tap 1

    # Timing
    fsr_hz: float = 160e9  # Free spectral range (Hz)
    delay_step_s: float = 6.25e-12  # 6.25 ps between taps

    # Waveguide properties
    waveguide_loss_db_per_cm: float = 0.15
    group_index: float = 1.71

    # Heater/Phase shifter properties
    p2pi_watts: float = 0.75  # Power for 2π phase shift (nominal)
    heater_resistance_ohm: float = 600.0

    # Loss parameters
    coupling_loss_db: float = 0.2  # Per 3dB coupler
    base_insertion_loss_db: float = 3.0  # Minimum chip insertion loss


@dataclass
class MZIState:
    """State of a single MZI."""

    mzi_id: str  # e.g., "2-1", "3-3", "4-5"
    applied_power_watts: float = 0.0
    phase_shift_rad: float = 0.0

    # Characterisation (measured)
    phi_init_rad: float = 0.0  # Initial phase offset
    p2pi_watts: float = 0.75  # Specific P2π for this MZI


@dataclass
class PhaseShifterState:
    """State of a single phase shifter."""

    tap_number: int  # 9-16 for signal processing taps
    applied_power_watts: float = 0.0
    phase_shift_rad: float = 0.0

    # Characterization (measured)
    phi_init_rad: float = 0.0
    p2pi_watts: float = 0.75


@dataclass
class ChipState:
    """Complete state of the FIR chip."""

    # MZI states (7 MZIs for binary tree)
    mzis: Dict[str, MZIState] = field(default_factory=dict)

    # Phase shifter states (8 for signal processing taps)
    phase_shifters: Dict[int, PhaseShifterState] = field(default_factory=dict)

    # Fixed power for reference and unused taps
    p_fixed_watts: float = 0.3

    def __post_init__(self):
        """Initialize default MZIs and phase shifters if empty."""
        if not self.mzis:
            # Binary tree structure: stage-position
            mzi_ids = ["2-1", "3-3", "3-4", "4-5", "4-6", "4-7", "4-8"]
            self.mzis = {mzi_id: MZIState(mzi_id=mzi_id) for mzi_id in mzi_ids}

        if not self.phase_shifters:
            # Signal processing taps: 9-16
            self.phase_shifters = {
                i: PhaseShifterState(tap_number=i) for i in range(9, 17)
            }


@dataclass
class MeasurementConfig:
    """Configuration for spectral measurements."""

    # Wavelength range (single FSR)
    center_wavelength_nm: float = 1550.0
    wavelength_span_nm: float = 1.0
    n_points: int = 1000

    # Temperature control
    chip_temperature_c: float = 30.0

    # Instrument addresses (optional, for real hardware)
    ova_address: Optional[str] = None
    voltage_controller_port: Optional[str] = None


@dataclass
class CalibrationConfig:
    """Configuration for self-calibration algorithm."""

    # Learning parameters
    learning_rate: float = 0.5
    max_iterations: int = 25

    # Convergence criteria
    amplitude_tolerance_db: float = 0.5
    phase_tolerance_rad: float = 0.1

    # Initial MZI characterization
    use_two_step_init: bool = True
    init_power_sweep_watts: float = 0.05


@dataclass
class TargetFilter:
    """Specification of desired filter response."""

    filter_type: str = (
        "sinc"  # "sinc", "hilbert", "lowpass", "highpass", "differentiator"
    )

    # For sinc filters with linear phase
    phase_step_rad: float = 0.0  # e.g., 0, 2π/7, 4π/7, etc.

    # Custom tap weights (if filter_type == "custom")
    custom_amplitudes: Optional[List[float]] = None
    custom_phases_rad: Optional[List[float]] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Metadata
    name: str = "fir_calibration"
    description: str = ""

    # Chip and state
    chip: ChipParameters = field(default_factory=ChipParameters)
    initial_state: ChipState = field(default_factory=ChipState)

    # Target filter
    target: TargetFilter = field(default_factory=TargetFilter)

    # Measurement settings
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)

    # Calibration settings
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # Output paths
    output_dir: str = "./results/"
    save_iterations: bool = True


@dataclass
class IterationData:
    """Data from a single calibration iteration."""

    iteration: int

    # Measured spectrum
    wavelengths_nm: np.ndarray
    insertion_loss_db: np.ndarray

    # Recovered tap coefficients (complex)
    tap_amplitudes: np.ndarray  # Length n_taps
    tap_phases_rad: np.ndarray  # Length n_taps

    # Errors for signal processing taps
    amplitude_errors_db: np.ndarray  # Length n_signal_taps
    phase_errors_rad: np.ndarray  # Length n_signal_taps

    # RMS errors
    rms_amplitude_error_db: float
    rms_phase_error_rad: float

    # Updated powers
    mzi_powers: Dict[str, float]
    ps_powers: Dict[int, float]


@dataclass
class CalibrationResults:
    """Results from complete calibration experiment."""

    config: ExperimentConfig
    iterations: List[IterationData]

    converged: bool
    final_iteration: int

    # Final calibrated tap coefficients
    final_amplitudes: np.ndarray
    final_phases_rad: np.ndarray

    # Final chip state
    final_state: ChipState


# Helper functions for YAML compatibility


def config_to_dict(config: ExperimentConfig) -> dict:
    """Convert ExperimentConfig to dictionary for YAML export."""
    from dataclasses import asdict

    return asdict(config)


def config_from_dict(config_dict: dict) -> ExperimentConfig:
    """Create ExperimentConfig from dictionary loaded from YAML."""

    # Create chip parameters
    chip = ChipParameters(**config_dict.get("chip", {}))

    # Create initial state
    state_dict = config_dict.get("initial_state", {})

    # Reconstruct MZI states
    mzis = {}
    for mzi_id, mzi_data in state_dict.get("mzis", {}).items():
        mzis[mzi_id] = MZIState(**mzi_data)

    # Reconstruct phase shifter states
    phase_shifters = {}
    for tap_num, ps_data in state_dict.get("phase_shifters", {}).items():
        phase_shifters[int(tap_num)] = PhaseShifterState(**ps_data)

    initial_state = ChipState(
        mzis=mzis,
        phase_shifters=phase_shifters,
        p_fixed_watts=state_dict.get("p_fixed_watts", 0.3),
    )

    # Create target filter
    target = TargetFilter(**config_dict.get("target", {}))

    # Create measurement config
    measurement = MeasurementConfig(**config_dict.get("measurement", {}))

    # Create calibration config
    calibration = CalibrationConfig(**config_dict.get("calibration", {}))

    # Create complete config
    return ExperimentConfig(
        name=config_dict.get("name", "fir_calibration"),
        description=config_dict.get("description", ""),
        chip=chip,
        initial_state=initial_state,
        target=target,
        measurement=measurement,
        calibration=calibration,
        output_dir=config_dict.get("output_dir", "./results/"),
        save_iterations=config_dict.get("save_iterations", True),
    )
