"""
data_structures.py

Data structures for photonic FIR chip parameters and experiment configuration.
Compatible with YAML serialization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np


@dataclass
class ChipParameters:
    """Physical parameters of the 16-tap FIR chip."""

    # Basic architecture
    n_taps: int = 16
    n_signal_taps: int = 8  # Taps 9-16
    reference_tap_idx: int = 0  # Tap 1
    signal_tap_start: int = 9  # First signal processing tap

    # Timing
    fsr_hz: float = 160e9  # Free spectral range (Hz)
    delay_step_s: float = 6.25e-12  # 6.25 ps between taps

    # Waveguide properties
    waveguide_loss_db_per_cm: float = 0.15
    group_index: float = 1.71
    dispersion_ps_per_nm_per_km: float = 0.0  # Chromatic dispersion

    # Heater/Phase shifter properties
    p2pi_watts: float = 0.75  # Power for 2π phase shift (nominal)
    p2pi_tolerance: float = 0.05  # Expected variation in P2π
    heater_resistance_ohm: float = 600.0
    heater_resistance_tolerance: float = 30.0

    # Loss parameters
    coupling_loss_db: float = 0.2  # Per 3dB coupler
    base_insertion_loss_db: float = 3.0  # Minimum chip insertion loss

    # Thermal parameters
    thermal_time_constant_s: float = 1e-3  # Time for thermal equilibrium

    # Derived properties (computed from above)
    @property
    def signal_tap_indices(self) -> Tuple[int, ...]:
        """Indices for signal processing taps (e.g., taps 9-16 → indices 8-15)."""
        return tuple(
            range(
                self.signal_tap_start - 1,  # -1 for 0-based indexing
                self.signal_tap_start - 1 + self.n_signal_taps,
            )
        )

    @property
    def signal_tap_numbers(self) -> Tuple[int, ...]:
        """Tap numbers for signal processing (e.g., 9, 10, 11, ..., 16)."""
        return tuple(
            range(self.signal_tap_start, self.signal_tap_start + self.n_signal_taps)
        )

    @property
    def n_mzis(self) -> int:
        """Number of MZIs in binary tree (n_signal_taps - 1)."""
        return self.n_signal_taps - 1

    def get_mzi_ids(self) -> List[str]:
        """
        Generate MZI IDs for signal processing core following Xu et al. (2022) convention.

        The signal processing MZIs control the upper half of taps (9-16 for 16-tap chip).
        The MZI position numbers follow the FULL binary tree structure where:
        - Stage 2: position 1 (signal processing root)
        - Stage 3+: positions in the SECOND HALF (because signal processing
                    descends from the upper branch of the tree)

        For 8 signal taps (n_signal_taps=8):
        - Stage 2: 1 MZI  → "2-1"
        - Stage 3: 2 MZIs → "3-3", "3-4"  (positions 3-4, second half of 4)
        - Stage 4: 4 MZIs → "4-5", "4-6", "4-7", "4-8" (positions 5-8, second half of 8)

        Returns:
            List of MZI IDs in order, e.g., ["2-1", "3-3", "3-4", "4-5", "4-6", "4-7", "4-8"]
        """
        mzi_ids = []
        stage = 2  # Start from stage 2 (first MZI in signal processing)

        while len(mzi_ids) < self.n_mzis:
            n_mzis_in_stage = 2 ** (stage - 2)

            if stage == 2:
                # Stage 2: signal processing starts at position 1
                position_start = 1
            else:
                # Stages 3+: signal processing uses second half of positions
                # Total positions at stage s = 2^(s-1)
                # Second half starts at position 2^(s-2) + 1
                position_start = 2 ** (stage - 2) + 1

            for i in range(n_mzis_in_stage):
                if len(mzi_ids) >= self.n_mzis:
                    break
                position = position_start + i
                mzi_ids.append(f"{stage}-{position}")

            stage += 1

        return mzi_ids


@dataclass
class MZIState:
    """State of a single MZI."""

    mzi_id: str  # e.g., "2-1", "3-3", "4-5"
    applied_power_watts: float = 0.0
    phase_shift_rad: float = 0.0

    # Characterisation (measured or estimated)
    phi_init_rad: float = 0.0  # Initial phase offset
    p2pi_watts: float = 0.75  # Specific P2π for this MZI

    # Target value (for calibration)
    target_power_ratio_db: Optional[float] = None

    def __post_init__(self):
        """Wrap phase to [-π, π]."""
        self.phase_shift_rad = np.arctan2(
            np.sin(self.phase_shift_rad),
            np.cos(self.phase_shift_rad),
        )


@dataclass
class PhaseShifterState:
    """State of a single phase shifter."""

    tap_number: int  # e.g., 9-16 for signal processing taps
    applied_power_watts: float = 0.0
    phase_shift_rad: float = 0.0

    # Characterisation (measured or estimated)
    phi_init_rad: float = 0.0
    p2pi_watts: float = 0.75

    # Target value (for calibration)
    target_phase_rad: Optional[float] = None

    def __post_init__(self):
        """Wrap phase to [-π, π]."""
        self.phase_shift_rad = np.arctan2(
            np.sin(self.phase_shift_rad),
            np.cos(self.phase_shift_rad),
        )


@dataclass
class ChipState:
    """Complete state of the FIR chip."""

    # Reference to chip parameters (REQUIRED for initialization)
    chip_params: ChipParameters = field(default_factory=ChipParameters)

    # MZI states (generated automatically from chip_params)
    mzis: Dict[str, MZIState] = field(default_factory=dict)

    # Phase shifter states (generated automatically from chip_params)
    phase_shifters: Dict[int, PhaseShifterState] = field(default_factory=dict)

    # Fixed power for reference and unused taps
    p_fixed_watts: float = 0.3

    def __post_init__(self):
        """Initialize MZIs and phase shifters from chip_params."""
        # Initialize MZIs if empty
        if not self.mzis:
            mzi_ids = self.chip_params.get_mzi_ids()
            self.mzis = {
                mzi_id: MZIState(mzi_id=mzi_id, p2pi_watts=self.chip_params.p2pi_watts)
                for mzi_id in mzi_ids
            }

        # Initialize phase shifters if empty
        if not self.phase_shifters:
            self.phase_shifters = {
                tap_num: PhaseShifterState(
                    tap_number=tap_num, p2pi_watts=self.chip_params.p2pi_watts
                )
                for tap_num in self.chip_params.signal_tap_numbers
            }

    def get_all_applied_powers(self) -> Dict[str, float]:
        """Get dictionary of all applied powers for monitoring."""
        powers = {}
        for mzi_id, mzi in self.mzis.items():
            powers[f"MZI_{mzi_id}"] = mzi.applied_power_watts
        for tap_num, ps in self.phase_shifters.items():
            powers[f"PS_{tap_num}"] = ps.applied_power_watts
        return powers


@dataclass
class MeasurementConfig:
    """Configuration for spectral measurements."""

    # Wavelength range (single FSR)
    center_wavelength_nm: float = 1550.0
    wavelength_span_nm: float = 1.0
    n_points: int = 1000

    # Temperature control
    chip_temperature_c: float = 30.0
    temperature_tolerance_c: float = 0.1

    # Instrument addresses (optional, for real hardware)
    ova_address: Optional[str] = "130.194.137.122"
    voltage_controller_port: Optional[str] = "COM3"
    voltage_controller_baudrate: Optional[int] = 9600
    edfa_port: Optional[str] = "COM6"
    edfa_baudrate: Optional[int] = 57600
    edfa_output_power_dbm: float = 13.0  # Output power of EDFA

    # Measurement settings
    integration_time_s: float = 0.1
    num_averages: int = 1


@dataclass
class TapDetectionConfig:
    """Configuration for tap detection from impulse response."""

    # Peak detection method
    use_db_scale: bool = True  # Use dB scale for detection

    # Detection thresholds
    prominence_factor_db: float = 0.3  # Prominence in dB (if use_db_scale=True)
    height_threshold_db: float = -60.0  # Minimum height in dB
    prominence_factor_linear: float = 0.1  # Prominence ratio (if use_db_scale=False)
    height_threshold_linear: float = 0.05  # Minimum height ratio

    # Spacing constraints
    min_distance_ps: Optional[float] = None  # If None, use 80% of delay_step_s
    min_distance_factor: float = 0.8  # Factor of delay_step_s to use

    # Expected number of taps (if None, use chip.n_taps)
    expected_n_taps: Optional[int] = None


@dataclass
class PhaseRecoveryConfig:
    """Configuration for Kramers-Kronig phase recovery."""

    # Column names in measurement CSV
    wavelength_col: str = "wl_axis"
    frequency_col: str = "f_axis"
    insertion_loss_col: str = "IL"

    # Hilbert transform parameters
    add_noise_floor_db: float = -120.0  # Add small offset to avoid log(0)

    # Validation
    check_minimum_phase: bool = True  # Verify minimum phase condition
    reference_tap_margin_db: float = 3.0  # Min margin for ref tap


@dataclass
class CalibrationConfig:
    """Configuration for self-calibration algorithm."""

    # Learning parameters
    learning_rate: float = 0.5
    max_iterations: int = 25

    # Convergence criteria
    amplitude_tolerance_db: float = 0.5
    phase_tolerance_rad: float = 0.1

    # Initial MZI characterisation
    use_two_step_init: bool = True
    init_power_sweep_watts: float = 0.05

    # Power update constraints
    max_power_step_watts: float = 0.1
    min_power_watts: float = 0.0
    max_power_watts: float = 1.0

    # Phase wrap handling
    phase_wrap_threshold_rad: float = np.pi / 2


@dataclass
class TargetFilter:
    """Specification of desired filter response."""

    filter_type: str = (
        "sinc"  # "sinc", "hilbert", "lowpass", "highpass", "differentiator", "custom"
    )

    # For sinc filters with linear phase
    phase_step_rad: float = 0.0  # e.g., 0, 2π/7, 4π/7, etc.

    # Custom tap weights (if filter_type == "custom")
    custom_amplitudes: Optional[np.ndarray] = None
    custom_phases_rad: Optional[np.ndarray] = None

    def get_target_taps(self, n_taps: int = 8) -> np.ndarray:
        """
        Generate target tap coefficients based on filter specification.

        Args:
            n_taps: Number of taps in signal processing core

        Returns:
            Complex-valued tap coefficients
        """
        if self.filter_type == "custom":
            if self.custom_amplitudes is None or self.custom_phases_rad is None:
                raise ValueError("Custom filter requires amplitudes and phases")
            return self.custom_amplitudes * np.exp(1j * self.custom_phases_rad)

        # Generate standard filter types
        return self._generate_standard_filter(n_taps)

    def _generate_standard_filter(self, n_taps: int) -> np.ndarray:
        """Generate standard filter tap coefficients."""
        n = np.arange(n_taps)

        if self.filter_type == "sinc":
            # Sinc filter: EQUAL tap amplitudes with linear phase
            # Equal amplitudes in time domain → sinc-shaped frequency response
            # This matches Figure 3 of Xu et al. (2022) where all 8 taps
            # converge to the same amplitude value
            amplitudes = np.ones(n_taps)
            phases = self.phase_step_rad * n

        elif self.filter_type == "hilbert":
            # Hilbert transformer
            # Antisymmetric impulse response: h[n] = 2/(π*n) for odd n, 0 for even n
            amplitudes = np.where(
                n % 2 == 0, 0.0, 2.0 / (np.pi * (n - (n_taps - 1) / 2))
            )
            phases = np.zeros(n_taps)

        elif self.filter_type == "lowpass":
            # Half-band lowpass filter
            # Sinc-shaped impulse response → rectangular (brick-wall) frequency response
            fc = 0.25  # Normalised cutoff frequency
            amplitudes = 2 * fc * np.sinc(2 * fc * (n - (n_taps - 1) / 2))
            phases = np.zeros(n_taps)

        elif self.filter_type == "highpass":
            # Half-band highpass filter
            # All-pass minus lowpass = highpass
            fc = 0.25
            amplitudes = np.sinc(n - (n_taps - 1) / 2) - 2 * fc * np.sinc(
                2 * fc * (n - (n_taps - 1) / 2)
            )
            phases = np.zeros(n_taps)

        elif self.filter_type == "differentiator":
            # Differentiator
            # h[n] = (-1)^n / n for n ≠ 0, h[0] = 0
            centre = (n_taps - 1) / 2
            amplitudes = np.where(n == centre, 0.0, (-1) ** (n - centre) / (n - centre))
            phases = np.zeros(n_taps)

        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        # Normalise amplitudes
        amplitudes /= np.max(np.abs(amplitudes))

        return amplitudes * np.exp(1j * phases)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Metadata
    name: str = "fir_calibration"
    description: str = ""
    timestamp: Optional[str] = None

    # Chip parameters (defines architecture)
    chip: ChipParameters = field(default_factory=ChipParameters)

    # Initial state (will be auto-populated from chip)
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

    def __post_init__(self):
        """Ensure initial_state has reference to chip parameters."""
        if self.initial_state.chip_params is None:
            self.initial_state.chip_params = self.chip


@dataclass
class IterationData:
    """Data from a single calibration iteration."""

    iteration: int

    # Measured spectrum
    wavelengths_nm: np.ndarray
    insertion_loss_db: np.ndarray

    # Recovered tap coefficients (complex, all 16 taps)
    tap_amplitudes: np.ndarray
    tap_phases_rad: np.ndarray

    # Errors for signal processing taps only (8 taps)
    amplitude_errors_db: np.ndarray
    phase_errors_rad: np.ndarray

    # RMS errors
    rms_amplitude_error_db: float
    rms_phase_error_rad: float

    # Updated powers
    mzi_powers: Dict[str, float]
    ps_powers: Dict[int, float]

    # Additional metrics
    max_amplitude_error_db: float = field(init=False)
    max_phase_error_rad: float = field(init=False)

    def __post_init__(self):
        """Calculate derived metrics."""
        self.max_amplitude_error_db = np.max(np.abs(self.amplitude_errors_db))
        self.max_phase_error_rad = np.max(np.abs(self.phase_errors_rad))


@dataclass
class CalibrationResults:
    """Results from complete calibration experiment."""

    config: ExperimentConfig
    iterations: List[IterationData]

    converged: bool
    final_iteration: int

    # Final calibrated tap coefficients (8 signal processing taps)
    final_amplitudes: np.ndarray
    final_phases_rad: np.ndarray

    # Final chip state
    final_state: ChipState

    # Convergence metrics
    convergence_history: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        """Extract convergence history."""
        self.convergence_history = {
            "rms_amplitude_error": [
                it.rms_amplitude_error_db for it in self.iterations
            ],
            "rms_phase_error": [it.rms_phase_error_rad for it in self.iterations],
            "max_amplitude_error": [
                it.max_amplitude_error_db for it in self.iterations
            ],
            "max_phase_error": [it.max_phase_error_rad for it in self.iterations],
        }


# Helper functions for YAML compatibility


def config_to_dict(config: ExperimentConfig) -> dict:
    """Convert ExperimentConfig to dictionary for YAML export."""
    from dataclasses import asdict

    config_dict = asdict(config)

    # Convert numpy arrays to lists for YAML compatibility
    if config.target.custom_amplitudes is not None:
        config_dict["target"][
            "custom_amplitudes"
        ] = config.target.custom_amplitudes.tolist()
    if config.target.custom_phases_rad is not None:
        config_dict["target"][
            "custom_phases_rad"
        ] = config.target.custom_phases_rad.tolist()

    return config_dict


def config_from_dict(config_dict: dict) -> ExperimentConfig:
    """Create ExperimentConfig from dictionary loaded from YAML."""

    # Create chip parameters
    chip = ChipParameters(**config_dict.get("chip", {}))

    # Create initial state WITH chip parameters reference
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
        chip_params=chip,  # Pass chip parameters
        mzis=mzis,
        phase_shifters=phase_shifters,
        p_fixed_watts=state_dict.get("p_fixed_watts", 0.3),
    )

    # Create target filter
    target_dict = config_dict.get("target", {})
    if target_dict.get("custom_amplitudes") is not None:
        target_dict["custom_amplitudes"] = np.array(target_dict["custom_amplitudes"])
    if target_dict.get("custom_phases_rad") is not None:
        target_dict["custom_phases_rad"] = np.array(target_dict["custom_phases_rad"])
    target = TargetFilter(**target_dict)

    # Create measurement config
    measurement = MeasurementConfig(**config_dict.get("measurement", {}))

    # Create calibration config
    calibration = CalibrationConfig(**config_dict.get("calibration", {}))

    # Create complete config
    return ExperimentConfig(
        name=config_dict.get("name", "fir_calibration"),
        description=config_dict.get("description", ""),
        timestamp=config_dict.get("timestamp"),
        chip=chip,
        initial_state=initial_state,
        target=target,
        measurement=measurement,
        calibration=calibration,
        output_dir=config_dict.get("output_dir", "./results/"),
        save_iterations=config_dict.get("save_iterations", True),
    )
