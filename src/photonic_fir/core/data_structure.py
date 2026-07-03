"""
data_structures.py

Data structures for photonic FIR chip parameters and experiment configuration.
Compatible with YAML serialization.
"""

from dataclasses import dataclass, field
import sys
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np


@dataclass
class VoltageChannelMapping:
    """
    Maps voltage controller hardware channels to MZI/phase shifter IDs.

    This mapping depends on your physical chip layout and how the voltage
    controller is wired to the chip's wire bonds.
    """

    # MZI channel assignments
    mzi_channels: Dict[str, int] = field(default_factory=dict)

    # Phase shifter channel assignments
    ps_channels: Dict[int, int] = field(default_factory=dict)

    # Reference and unused tap channels (if controlled)
    reference_channel: Optional[int] = None
    unused_channels: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default channel mapping if empty."""
        if not self.mzi_channels:
            # Example mapping - adjust based on your actual wiring
            self.mzi_channels = {
                "1-1": 4,
                "2-1": 18,
                "2-2": 2,
                "3-1": 17,
                "3-2": 19,
                "3-3": 3,
                "3-4": 1,
                "4-1": 23,
                "4-2": 22,
                "4-3": 21,
                "4-4": 20,
                "4-5": 5,
                "4-6": 6,
                "4-7": 7,
                "4-8": 8,
            }

        if not self.ps_channels:
            # Signal processing taps 9-16 mapped to channels 7-14
            self.ps_channels = {
                1: 24,
                2: 25,
                3: 26,
                4: 27,
                5: 28,
                6: 29,
                7: 30,
                8: 31,
                9: 16,
                10: 15,
                11: 14,
                12: 13,
                13: 12,
                14: 11,
                15: 10,
                16: 9,
            }

    def get_channel(self, device_id: str) -> int:
        """
        Get voltage controller channel for a device.

        Args:
            device_id: Device identifier (e.g., "MZI_2-1", "PS_9", "REF", "UNUSED_3")

        Returns:
            int: Channel number for voltage controller

        Raises:
            ValueError: If device_id not found in mapping
        """
        if device_id.startswith("MZI_"):
            mzi_id = device_id[4:]  # Remove "MZI_" prefix
            if mzi_id not in self.mzi_channels:
                raise ValueError(f"MZI '{mzi_id}' not in channel mapping")
            return self.mzi_channels[mzi_id]

        elif device_id.startswith("PS_"):
            tap_num = int(device_id[3:])  # Remove "PS_" prefix
            if tap_num not in self.ps_channels:
                raise ValueError(f"Phase shifter tap {tap_num} not in channel mapping")
            return self.ps_channels[tap_num]

        elif device_id == "REF":
            if self.reference_channel is None:
                raise ValueError("Reference channel not configured")
            return self.reference_channel

        elif device_id.startswith("UNUSED_"):
            tap_num = int(device_id[7:])  # Remove "UNUSED_" prefix
            if tap_num not in self.unused_channels:
                raise ValueError(f"Unused tap {tap_num} not in channel mapping")
            return self.unused_channels[tap_num]

        else:
            raise ValueError(f"Unknown device ID format: {device_id}")

    def get_all_channels(self) -> Dict[str, int]:
        """Get complete mapping of all devices to channels."""
        all_channels = {}

        # MZIs
        for mzi_id, channel in self.mzi_channels.items():
            all_channels[f"MZI_{mzi_id}"] = channel

        # Phase shifters
        for tap_num, channel in self.ps_channels.items():
            all_channels[f"PS_{tap_num}"] = channel

        # Reference
        if self.reference_channel is not None:
            all_channels["REF"] = self.reference_channel

        # Unused
        for tap_num, channel in self.unused_channels.items():
            all_channels[f"UNUSED_{tap_num}"] = channel

        return all_channels

    def validate_no_duplicates(self) -> bool:
        """
        Verify no channel is assigned to multiple devices.

        Returns:
            bool: True if mapping is valid

        Raises:
            ValueError: If duplicate channel assignments found
        """
        all_channels = list(self.mzi_channels.values())
        all_channels.extend(self.ps_channels.values())
        all_channels.extend(self.unused_channels.values())
        if self.reference_channel is not None:
            all_channels.append(self.reference_channel)

        if len(all_channels) != len(set(all_channels)):
            duplicates = [ch for ch in all_channels if all_channels.count(ch) > 1]
            raise ValueError(f"Duplicate channel assignments: {set(duplicates)}")

        return True


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
    p2pi_watts_mzi: float = 0.76  # Power for 2π phase shift (nominal)
    p2pi_watts_ps: float = 0.56  # Power for 2π phase shift (nominal)
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
        """
        mzi_ids = []
        stage = 2  # Start from stage 2 (first MZI in signal processing)

        while len(mzi_ids) < self.n_mzis:
            n_mzis_in_stage = 2 ** (stage - 2)

            if stage == 2:
                # Stage 2: signal processing starts at position 1
                position_start = 2
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

    def get_signal_mzi_ids(self) -> List[str]:
        """
        Generate MZI IDs for signal processing core (stages 2-4).
        Returns the same list as the existing get_mzi_ids() method.
        """
        return self.get_mzi_ids()

    def get_all_mzi_ids(self) -> List[str]:
        """
        Get ALL MZI IDs on chip including stage 1.
        Returns all MZIs from stages 1 to 4.
        """
        mzi_ids = []
        stage = np.log2(self.n_signal_taps) + 1

        for s in range(1, int(stage) + 1):
            n_mzis_in_stage = 2 ** (s - 1)
            for position in range(1, n_mzis_in_stage + 1):
                mzi_ids.append(f"{s}-{position}")

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
    phi_measured_rad: float = 0.0
    p2pi_watts: float = 0.56

    # Target value (for calibration)
    target_phase_rad: Optional[float] = None
    target_probe_rad: Optional[float] = None  # For probe mode

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
    p_fixed_watts: float = 0.01

    def __post_init__(self):
        """Initialize MZIs and phase shifters from chip_params."""
        # Initialize MZIs if empty
        if not self.mzis:
            mzi_ids = self.chip_params.get_all_mzi_ids()
            self.mzis = {
                mzi_id: MZIState(
                    mzi_id=mzi_id,
                    p2pi_watts=self.chip_params.p2pi_watts_mzi,
                )
                for mzi_id in mzi_ids
            }

        # Initialize phase shifters if empty
        if not self.phase_shifters:
            self.phase_shifters = {
                tap_num: PhaseShifterState(
                    tap_number=tap_num,
                    p2pi_watts=self.chip_params.p2pi_watts_ps,
                )
                for tap_num in self.chip_params.signal_tap_numbers
            }

    def get_mzi_applied_powers(self) -> Dict[str, float]:
        """Get dictionary of MZI applied powers for monitoring."""
        return {mzi_id: mzi.applied_power_watts for mzi_id, mzi in self.mzis.items()}

    def get_ps_applied_powers(self) -> Dict[int, float]:
        """Get dictionary of phase shifter applied powers for monitoring."""
        return {
            tap_num: ps.applied_power_watts
            for tap_num, ps in self.phase_shifters.items()
        }

    def get_mzi_init_phase(self) -> Dict[str, float]:
        """Get dictionary of MZI initial phase offsets."""
        return {mzi_id: mzi.phi_init_rad for mzi_id, mzi in self.mzis.items()}

    def get_ps_init_phase(self) -> Dict[int, float]:
        """Get dictionary of phase shifter initial phase offsets."""
        return {tap_num: ps.phi_init_rad for tap_num, ps in self.phase_shifters.items()}

    def get_ps_measured_phase(self) -> Dict[int, float]:
        """Get dictionary of phase shifter measured phases."""
        return {
            tap_num: ps.phi_measured_rad for tap_num, ps in self.phase_shifters.items()
        }

    def get_all_applied_powers(self) -> Dict[str, float]:
        """Get dictionary of all applied powers for monitoring."""
        powers = {}
        for mzi_id, mzi in self.mzis.items():
            powers[mzi_id] = mzi.applied_power_watts
        for tap_num, ps in self.phase_shifters.items():
            powers[tap_num] = ps.applied_power_watts
        return powers

    def get_all_init_phase(self) -> Dict[str, float]:
        """Get dictionary of all initial phase offsets."""
        init_phase = {}
        for mzi_id, mzi in self.mzis.items():
            init_phase[mzi_id] = mzi.phi_init_rad
        for tap_num, ps in self.phase_shifters.items():
            init_phase[tap_num] = ps.phi_init_rad
        return init_phase

    def update_powers(
        self,
        new_mzi_powers: Dict[str, float],
        new_ps_powers: Dict[int, float],
        prev_mzi_psr_errors: Optional[Dict[str, float]] = None,
        curr_mzi_psr_errors: Optional[Dict[str, float]] = None,
        psr_increase_threshold_db: float = 0.2,
    ) -> None:
        """
        Update applied powers and initial phase offsets for MZIs and phase shifters.

        Updates chip state in-place. Recalculates phase shifts via φ = (P / P_2π) × 2π.
        If PSR error worsened since the last iteration, flips φ_init by π for that MZI.

        Args:
            new_mzi_powers: New power settings for MZIs in watts, e.g. {"2-1": 0.35}
            new_ps_powers: New power settings for phase shifters in watts, e.g. {9: 0.42}
            prev_mzi_psr_errors: PSR errors from the previous iteration (dB). Optional.
            curr_mzi_psr_errors: PSR errors from the current iteration (dB). Optional.
            psr_increase_threshold_db: Threshold above which φ_init is flipped (dB).
        """
        # φ_init flip — PSR error worsened since last iteration
        if prev_mzi_psr_errors is not None and curr_mzi_psr_errors is not None:
            for mzi_id, curr_err in curr_mzi_psr_errors.items():
                prev_err = prev_mzi_psr_errors.get(mzi_id, 0.0)
                if curr_err - prev_err > psr_increase_threshold_db:
                    self.mzis[mzi_id].phi_init_rad += np.pi
                    self.mzis[mzi_id].phi_init_rad = float(
                        np.angle(np.exp(1j * self.mzis[mzi_id].phi_init_rad))
                    )

        # Update MZI states
        for mzi_id, new_power in new_mzi_powers.items():
            if mzi_id not in self.mzis:
                raise ValueError(f"MZI '{mzi_id}' not found in chip state")
            mzi = self.mzis[mzi_id]
            mzi.applied_power_watts = new_power
            mzi.phase_shift_rad = float(
                np.angle(np.exp(1j * (new_power / mzi.p2pi_watts) * 2 * np.pi))
            )

        # Update phase shifter states
        for tap_num, new_power in new_ps_powers.items():
            if tap_num not in self.phase_shifters:
                raise ValueError(f"Phase shifter tap {tap_num} not found in chip state")
            ps = self.phase_shifters[tap_num]
            ps.applied_power_watts = new_power
            ps.phase_shift_rad = float(
                np.angle(np.exp(1j * (new_power / ps.p2pi_watts) * 2 * np.pi))
            )

    def copy(self) -> "ChipState":
        """Create a deep copy for history tracking."""
        import copy

        return copy.deepcopy(self)


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
    voltage_controller_v_max: float = 30.0  # Max voltage (V)
    edfa_port: Optional[str] = "COM6"
    edfa_baudrate: Optional[int] = 57600
    edfa_output_power_dbm: float = 13.0  # Output power of EDFA

    # Measurement settings
    integration_time_s: float = 0.1
    num_averages: int = 1
    settling_time_sec: float = 2.0

    # Column names in measurement CSV
    wavelength_col: str = "wl_axis"
    frequency_col: str = "f_axis"
    insertion_loss_col: str = "IL"

    # Hilbert transform parameters
    add_noise_floor_db: float = -120.0  # Add small offset to avoid log(0)

    # Validation
    check_minimum_phase: bool = True  # Verify minimum phase condition
    reference_tap_margin_db: float = 3.0  # Min margin for ref tap

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
class CalibrationConfig:
    """Configuration for self-calibration algorithm."""

    # Learning parameters
    mzi_learning_rate: float = 0.5
    ps_learning_rate: float = 0.5
    max_iterations: int = 25
    trim_n_fsr: int = (
        10  # Number of FSRs to keep when trimming spectrum for impulse response recovery
    )

    # Convergence criteria
    mzi_dead_zone_db: float = 0.1  # Minimum power step to consider MZI "active"
    ps_dead_zone_rad: float = 0.0
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

    # Sequential calibration mode
    # If True: run amplitude-only loop until amp RMS < amplitude_tolerance_db,
    # then switch to phase-only. If False (default): update both simultaneously.
    sequential_mode: bool = False

    # --- calculate_power_adjustments hyperparameters ---
    # Minimum PSR improvement required to accept an MZI update (dB).
    # Prevents noisy small steps from being applied.
    psr_increase_threshold_db: float = 0.2
    # Whether to apply modulo-2π wrapping when computing power corrections.
    # Set True when crosstalk causes multi-cycle phase accumulation.
    wrap_phase: bool = False

    # --- adaptive_learning_rate hyperparameters ---
    lr_min: float = 1e-4
    lr_max: float = 0.8
    lr_decay: float = 0.7
    lr_grow: float = 1.05
    lr_phi_scale: float = float(np.pi)
    mzi_adaptive_learning: bool = False
    ps_adaptive_learning: bool = False

    # ---Diaagnostic settings---
    disable_taps_ps_taps: list[int] = field(
        default_factory=lambda: [10, 11, 12, 13, 14, 15, 16]
    )
    disable_taps: bool = False

    # --- Probe mode settings ---
    probe_mode: bool = False
    ps_probe_threshold_rad: float = float(np.pi / 2)

    # --- Number of FSRs to tile ---
    num_tiles_for_kk_recovery: int = 20
    tap_detection_window_width_ps: float = 3.0

    # --- Paths for loading characterisation data ---
    ps_crosstalk_matrix_path: Optional[str] = None

    # --- Gap method settings ---
    use_gap_method: bool = False

    # Optional initial power settings
    initial_mzi_voltages: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "1-1": 13.0,
            "2-1": 6.0,
            "2-2": 0.0,
            "3-1": 6.0,
            "3-2": 13.0,
            "3-3": 0.0,
            "3-4": 0.0,
            "4-1": 20.0,
            "4-2": 9.0,
            "4-3": 10.0,
            "4-4": 11.0,
            "4-5": 0.0,
            "4-6": 0.0,
            "4-7": 0.0,
            "4-8": 0.0,
        }
    )
    initial_mzi_powers: Optional[Dict[str, float]] = None  # e.g. {"2-1": 0.3}
    initial_ps_powers: Optional[Dict[int, float]] = None  # e.g. {9: 0.4}

    # --- Save settings ---
    save_spectrum: bool = True
    save_impulse_response: bool = True


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
                n % 2 == 0, 1e-2, 2.0 / (np.pi * (n - (n_taps - 1) / 2))
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
            amplitudes = np.where(
                n == centre, 1e-2, (-1) ** (n - centre) / (n - centre)
            )
            phases = np.zeros(n_taps)

        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        # Normalise amplitudes
        amplitudes /= np.max(np.abs(amplitudes))

        return amplitudes * np.exp(1j * phases)


# ==================== Tree Structure Building ====================
def build_mzi_tree_structure(n_signal_taps: int, mzi_ids: List[str]) -> Dict[str, Dict]:
    """
    Build binary tree structure mapping for MZI network.

    Args:
        n_signal_taps: Number of signal processing taps (must be power of 2)
        mzi_ids: List of MZI identifiers in hierarchical order

    Returns:
        Dict mapping MZI ID to structure info containing:
            - 'stage': Stage number in tree
            - 'position': Position within stage
            - 'lower_taps': (start, end) indices for bar port
            - 'upper_taps': (start, end) indices for cross port

    Raises:
        ValueError: If n_signal_taps is not a power of 2

    Example:
        >>> mzi_ids = ["2-1", "3-3", "3-4", "4-5", "4-6", "4-7", "4-8"]
        >>> tree = build_mzi_tree_structure(8, mzi_ids)
        >>> tree["2-1"]
        {'stage': 1, 'position': 0, 'lower_taps': (0, 4), 'upper_taps': (4, 8)}
    """
    # Validate power of 2
    if n_signal_taps <= 0 or (n_signal_taps & (n_signal_taps - 1)) != 0:
        raise ValueError(f"n_signal_taps must be a power of 2, got {n_signal_taps}")

    n_stages = int(np.log2(n_signal_taps))
    expected_mzis = n_signal_taps - 1

    if len(mzi_ids) != expected_mzis:
        raise ValueError(
            f"Expected {expected_mzis} MZI IDs for {n_signal_taps} taps, "
            f"got {len(mzi_ids)}"
        )

    tree_structure = {}
    mzi_index = 0

    # Process each stage
    for stage in range(1, n_stages + 1):
        n_mzis_in_stage = 2 ** (stage - 1)
        group_size = n_signal_taps // (2**stage)  # Taps per MZI output

        for position in range(n_mzis_in_stage):
            mzi_id = mzi_ids[mzi_index]

            # Calculate tap ranges this MZI controls
            start_tap = position * group_size * 2
            mid_tap = start_tap + group_size
            end_tap = start_tap + group_size * 2

            tree_structure[mzi_id] = {
                "stage": stage,
                "position": position,
                "lower_taps": (start_tap, mid_tap),  # Bar port
                "upper_taps": (mid_tap, end_tap),  # Cross port
            }

            mzi_index += 1

    return tree_structure


@dataclass
class MZITreeStructure:
    """Binary tree structure for MZI network."""

    chip: ChipParameters  # Need full chip to build complete tree
    mzi_ids: List[str]  # Which MZIs to include after filtering
    tree: Dict[str, Dict] = field(init=False)

    def __post_init__(self):
        """Build full tree and filter to requested MZIs."""
        # Build full tree with all MZIs
        full_tree = build_mzi_tree_structure(
            n_signal_taps=16, mzi_ids=self.chip.get_all_mzi_ids()
        )

        # Filter to requested MZIs
        self.tree = {mzi_id: full_tree[mzi_id] for mzi_id in self.mzi_ids}

    @classmethod
    def from_chip_signal_processing(cls, chip: ChipParameters) -> "MZITreeStructure":
        """Create tree for signal processing core (stages 2-4)."""
        return cls(chip=chip, mzi_ids=chip.get_signal_mzi_ids())

    @classmethod
    def from_chip_full(cls, chip: ChipParameters) -> "MZITreeStructure":
        """Create full tree for characterisation (stages 1-4)."""
        return cls(chip=chip, mzi_ids=chip.get_all_mzi_ids())


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    Contains all parameters needed to run a calibration experiment.
    Does NOT contain runtime state - use ChipState.create_initial_state() for that.
    """

    # Metadata
    name: str = "fir_calibration"
    description: str = ""
    timestamp: Optional[str] = None

    # Chip parameters (physics only)
    chip: ChipParameters = field(default_factory=ChipParameters)

    # Hardware channel mapping (lab-specific wiring)
    channel_mapping: VoltageChannelMapping = field(
        default_factory=VoltageChannelMapping
    )

    # Target filter
    target: TargetFilter = field(default_factory=TargetFilter)

    # Measurement settings
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)

    # Calibration settings
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # MZI tree structures
    signal_mzi_tree: MZITreeStructure = field(init=False)
    full_mzi_tree: MZITreeStructure = field(init=False)

    # Output paths
    output_dir: str = "./measurements/"
    save_iterations: bool = True

    def __post_init__(self):
        """Build MZI trees from chip parameters."""
        self.signal_mzi_tree = MZITreeStructure.from_chip_signal_processing(self.chip)
        self.full_mzi_tree = MZITreeStructure.from_chip_full(self.chip)


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

    # Updated chip state
    chip_state: ChipState

    # Additional metrics
    max_amplitude_error_db: float = field(init=False)
    max_phase_error_rad: float = field(init=False)

    mzi_psr_errors_db: Dict[str, float] = field(default_factory=dict)
    ps_phase_errors: Dict[int, float] = field(default_factory=dict)

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

    # Create chip parameters (no nested channel_mapping anymore)
    chip = ChipParameters(**config_dict.get("chip", {}))

    # Create channel mapping separately at top level
    channel_mapping = VoltageChannelMapping(**config_dict.get("channel_mapping", {}))

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
        channel_mapping=channel_mapping,
        target=target,
        measurement=measurement,
        calibration=calibration,
        output_dir=config_dict.get("output_dir", "./results/"),
        save_iterations=config_dict.get("save_iterations", True),
    )
