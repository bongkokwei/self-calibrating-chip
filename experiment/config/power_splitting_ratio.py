"""
power_splitting_ratio.py

Functional power splitting ratio calculations for binary tree MZI architectures.
"""

import numpy as np
from typing import Dict, Tuple, List


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


# ==================== Power Splitting Ratio Calculations ====================


def tap_coeffs_to_power_splitting_ratios(
    tap_coeffs: np.ndarray, mzi_tree: Dict[str, Dict]
) -> Dict[str, float]:
    """
    Calculate MZI power splitting ratios from tap coefficients.

    Args:
        tap_coeffs: Complex tap coefficients (length must match tree structure)
        mzi_tree: Tree structure from build_mzi_tree_structure()

    Returns:
        Dict[str, float]: Power splitting ratios in dB, keyed by MZI ID

    Example:
        >>> taps = np.ones(8) * np.exp(1j * np.linspace(0, 2*np.pi, 8))
        >>> mzi_ids = ["2-1", "3-3", "3-4", "4-5", "4-6", "4-7", "4-8"]
        >>> tree = build_mzi_tree_structure(8, mzi_ids)
        >>> psr = tap_coeffs_to_power_splitting_ratios(taps, tree)
    """
    # Extract tap powers
    tap_powers = np.abs(tap_coeffs) ** 2

    # Calculate PSR for each MZI
    psr_dict = {}

    for mzi_id, mzi_info in mzi_tree.items():
        lower_range = mzi_info["lower_taps"]
        upper_range = mzi_info["upper_taps"]

        # Sum power in each branch
        power_lower = np.sum(tap_powers[lower_range[0] : lower_range[1]])
        power_upper = np.sum(tap_powers[upper_range[0] : upper_range[1]])

        # Calculate power splitting ratio (bar/cross) in dB
        psr_dict[mzi_id] = 10 * np.log10(power_lower / (power_upper + 1e-12))

    return psr_dict


def extract_tap_phases(
    tap_coeffs: np.ndarray, tap_numbers: Tuple[int, ...]
) -> Dict[int, float]:
    """
    Extract phases from tap coefficients.

    Args:
        tap_coeffs: Complex tap coefficients
        tap_numbers: Tuple of tap numbers corresponding to coefficients

    Returns:
        Dict[int, float]: Phases in radians, keyed by tap number

    Example:
        >>> taps = np.array([1+0j, 0+1j, -1+0j, 0-1j])
        >>> phases = extract_tap_phases(taps, (9, 10, 11, 12))
        >>> phases[9]  # Should be 0
        0.0
    """
    return {
        tap_num: np.angle(tap_coeffs[idx]) for idx, tap_num in enumerate(tap_numbers)
    }


# ==================== Conversion Functions ====================


def power_splitting_ratio_to_mzi_phase(psr_db: float) -> float:
    """
    Convert power splitting ratio (dB) to MZI phase (radians).

    For MZI with phase φ:
    PSR_dB = 10*log₁₀(cos²(φ/2) / sin²(φ/2))

    Solving: φ = 2*arctan(10^(-PSR_dB/20))

    Args:
        psr_db: Power splitting ratio in dB

    Returns:
        MZI phase shift in radians

    Example:
        >>> phase = power_splitting_ratio_to_mzi_phase(0.0)  # Equal split
        >>> np.isclose(phase, np.pi/2)
        True
    """
    ratio_linear = 10 ** (psr_db / 10)
    phase = 2 * np.arctan(1 / np.sqrt(ratio_linear + 1e-12))
    return phase


def mzi_phase_to_power_splitting_ratio(phase_rad: float) -> float:
    """
    Convert MZI phase to power splitting ratio.

    Args:
        phase_rad: MZI phase in radians

    Returns:
        Power splitting ratio in dB

    Example:
        >>> psr = mzi_phase_to_power_splitting_ratio(np.pi/2)  # Equal split
        >>> np.isclose(psr, 0.0)
        True
    """
    bar_power = np.cos(phase_rad / 2) ** 2
    cross_power = np.sin(phase_rad / 2) ** 2
    return 10 * np.log10(bar_power / (cross_power + 1e-12))


def power_splitting_ratios_to_mzi_phases(
    psr_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Convert dictionary of PSRs to MZI phases.

    Args:
        psr_dict: Power splitting ratios in dB, keyed by MZI ID

    Returns:
        Dict[str, float]: MZI phases in radians, keyed by MZI ID
    """
    return {
        mzi_id: power_splitting_ratio_to_mzi_phase(psr)
        for mzi_id, psr in psr_dict.items()
    }


# ==================== Error Calculations ====================


def compute_psr_errors(
    measured_taps: np.ndarray, target_psr: Dict[str, float], mzi_tree: Dict[str, Dict]
) -> Dict[str, float]:
    """
    Compute power splitting ratio errors.

    Args:
        measured_taps: Measured tap coefficients
        target_psr: Target power splitting ratios in dB
        mzi_tree: MZI tree structure

    Returns:
        Dict[str, float]: PSR errors in dB (target - measured)
    """
    measured_psr = tap_coeffs_to_power_splitting_ratios(measured_taps, mzi_tree)

    return {
        mzi_id: target_psr[mzi_id] - measured_psr.get(mzi_id, 0.0)
        for mzi_id in target_psr.keys()
    }


def compute_phase_errors(
    measured_taps: np.ndarray,
    target_phases: Dict[int, float],
    tap_numbers: Tuple[int, ...],
) -> Dict[int, float]:
    """
    Compute phase shifter errors.

    Args:
        measured_taps: Measured tap coefficients
        target_phases: Target phases in radians
        tap_numbers: Tap numbers corresponding to coefficients

    Returns:
        Dict[int, float]: Phase errors in radians (target - measured), wrapped to [-π, π]
    """
    measured_phases = extract_tap_phases(measured_taps, tap_numbers)

    phase_errors = {}
    for tap_num in target_phases.keys():
        error = target_phases[tap_num] - measured_phases.get(tap_num, 0.0)
        # Wrap to [-π, π]
        phase_errors[tap_num] = np.arctan2(np.sin(error), np.cos(error))

    return phase_errors


# ==================== Visualization ====================


def print_tree_structure(
    n_signal_taps: int, mzi_tree: Dict[str, Dict], signal_tap_start: int = 9
):
    """
    Print the binary tree structure.

    Args:
        n_signal_taps: Number of signal processing taps
        mzi_tree: Tree structure
        signal_tap_start: Starting tap number for signal processing
    """
    n_stages = int(np.log2(n_signal_taps))

    print("\n" + "=" * 80)
    print(f"Binary Tree Structure for {n_signal_taps} Taps")
    print(f"Total Stages: {n_stages}")
    print(f"Total MZIs: {len(mzi_tree)}")
    print("=" * 80)

    # Group by stage
    stages = {}
    for mzi_id, info in mzi_tree.items():
        stage = info["stage"]
        if stage not in stages:
            stages[stage] = []
        stages[stage].append(mzi_id)

    # Print each stage
    for stage in sorted(stages.keys()):
        group_size = n_signal_taps // (2**stage)
        print(
            f"\nStage {stage}: {len(stages[stage])} MZIs, group size = {group_size} taps"
        )
        print("-" * 80)

        for mzi_id in stages[stage]:
            info = mzi_tree[mzi_id]
            lower = info["lower_taps"]
            upper = info["upper_taps"]

            # Convert to actual tap numbers
            lower_nums = f"{lower[0]+signal_tap_start}-{lower[1]+signal_tap_start-1}"
            upper_nums = f"{upper[0]+signal_tap_start}-{upper[1]+signal_tap_start-1}"

            print(f"  {mzi_id:<8} controls taps [{lower_nums}] vs [{upper_nums}]")


def print_psr_summary(psr_dict: Dict[str, float], mzi_tree: Dict[str, Dict]):
    """
    Print power splitting ratios in readable format.

    Args:
        psr_dict: Power splitting ratios in dB
        mzi_tree: Tree structure (for stage grouping)
    """
    print("\n" + "=" * 80)
    print("Power Splitting Ratios")
    print("=" * 80)
    print(
        f"{'MZI ID':<10} {'Stage':<8} {'PSR (dB)':<12} {'Phase (rad)':<12} {'Phase (deg)':<12}"
    )
    print("-" * 80)

    for mzi_id, psr in psr_dict.items():
        stage = mzi_tree[mzi_id]["stage"]
        phase = power_splitting_ratio_to_mzi_phase(psr)
        print(
            f"{mzi_id:<10} {stage:<8} {psr:+8.3f}     {phase:8.4f}      {np.degrees(phase):8.2f}"
        )


# ==================== Integration with ChipState ====================


def update_chip_state_with_psr(
    chip_state,  # ChipState type
    target_psr: Dict[str, float],
) -> None:
    """
    Update chip state MZI phases to achieve target PSRs.

    Args:
        chip_state: ChipState object to update (modified in place)
        target_psr: Target power splitting ratios in dB
    """
    target_phases = power_splitting_ratios_to_mzi_phases(target_psr)

    for mzi_id, target_phase in target_phases.items():
        if mzi_id in chip_state.mzis:
            chip_state.mzis[mzi_id].phase_shift_rad = target_phase
            chip_state.mzis[mzi_id].target_power_ratio_db = target_psr[mzi_id]


def update_chip_state_with_phases(
    chip_state,  # ChipState type
    target_phases: Dict[int, float],
) -> None:
    """
    Update chip state phase shifter phases.

    Args:
        chip_state: ChipState object to update (modified in place)
        target_phases: Target phases in radians, keyed by tap number
    """
    for tap_num, target_phase in target_phases.items():
        if tap_num in chip_state.phase_shifters:
            chip_state.phase_shifters[tap_num].phase_shift_rad = target_phase
            chip_state.phase_shifters[tap_num].target_phase_rad = target_phase


def compute_target_state_from_filter(
    chip_state,  # ChipState type
    target_filter,  # TargetFilter type
    mzi_tree: Dict[str, Dict],
) -> None:
    """
    Compute and set complete target state from filter specification.

    Args:
        chip_state: ChipState object to update (modified in place)
        target_filter: TargetFilter specification
        mzi_tree: MZI tree structure
    """
    # Get target tap coefficients
    target_taps = target_filter.get_target_taps(chip_state.chip_params.n_signal_taps)

    # Compute target PSRs and phases
    target_psr = tap_coeffs_to_power_splitting_ratios(target_taps, mzi_tree)
    target_phases = extract_tap_phases(
        target_taps, chip_state.chip_params.signal_tap_numbers
    )

    # Update chip state
    update_chip_state_with_psr(chip_state, target_psr)
    update_chip_state_with_phases(chip_state, target_phases)


# ==================== Convenience Functions ====================


def create_sinc_filter_taps(
    n_taps: int, phase_step: float = 0.0, normalize: bool = True
) -> np.ndarray:
    """
    Create sinc filter tap coefficients.

    Args:
        n_taps: Number of filter taps (must be power of 2)
        phase_step: Phase step between taps in radians
        normalize: Whether to normalize amplitudes to 0.5 max

    Returns:
        Complex tap coefficients
    """
    # Validate power of 2
    if n_taps <= 0 or (n_taps & (n_taps - 1)) != 0:
        raise ValueError(f"n_taps must be a power of 2, got {n_taps}")

    n = np.arange(n_taps)

    # Equal amplitudes (creates sinc-shaped frequency response)
    amplitudes = np.ones(n_taps)

    if normalize:
        amplitudes = amplitudes / np.max(amplitudes) * 0.5

    # Linear phase progression
    phases = n * phase_step

    return amplitudes * np.exp(1j * phases)
