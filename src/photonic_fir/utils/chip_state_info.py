"""
chip_state_info.py

Utility functions for printing a human-readable summary of a ChipState,
including applied electrical power and initial phase offset (φ_init) for
every MZI and phase shifter on the chip.

Typical usage
-------------
>>> from photonic_fir.utils.chip_state_info import print_chip_state
>>> print_chip_state(chip_state)

Or with a logger instead of stdout:

>>> print_chip_state(chip_state, use_logger=True)
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Separator widths
_WIDE = 70
_NARROW = 40


def _emit(msg: str, use_logger: bool) -> None:
    """Write *msg* to either the module logger or stdout."""
    if use_logger:
        logger.info(msg)
    else:
        print(msg)


def print_chip_state(
    chip_state,
    title: Optional[str] = None,
    use_logger: bool = False,
    show_phase_shift: bool = True,
    show_target: bool = True,
) -> None:
    """
    Print a formatted summary of the current ChipState.

    Displays applied electrical power (W), initial phase offset φ_init (rad
    and as a multiple of π), and optionally the current phase shift and any
    stored calibration targets for every MZI and phase shifter.

    Parameters
    ----------
    chip_state : ChipState
        The chip state object to summarise.
    title : str, optional
        Custom heading.  Defaults to "CHIP STATE SUMMARY".
    use_logger : bool
        If True, emit via ``logger.info()``; otherwise print to stdout.
    show_phase_shift : bool
        Include the computed phase shift column (default True).
    show_target : bool
        Include target values when they have been set (default True).

    Examples
    --------
    >>> print_chip_state(chip_state)
    >>> print_chip_state(chip_state, title="After φ_init characterisation", use_logger=True)
    """
    emit = lambda msg: _emit(msg, use_logger)

    heading = title or "CHIP STATE SUMMARY"
    emit("=" * _WIDE)
    emit(f"  {heading}")
    emit("=" * _WIDE)

    # ------------------------------------------------------------------ #
    # Fixed power reference                                                #
    # ------------------------------------------------------------------ #
    emit(f"  Fixed reference power : {chip_state.p_fixed_watts:.4f} W")
    emit("")

    # ------------------------------------------------------------------ #
    # MZI section                                                          #
    # ------------------------------------------------------------------ #
    emit("-" * _WIDE)
    emit("  MZI STATES")
    emit("-" * _WIDE)

    # Build header
    col_headers = ["MZI ID", "Power (W)", "φ_init (rad)", "φ_init / π"]
    if show_phase_shift:
        col_headers.append("φ_shift (rad)")
    col_headers.append("P_2π (W)")
    if show_target:
        col_headers.append("Target PSR (dB)")

    _print_mzi_table(chip_state, emit, show_phase_shift, show_target)

    emit("")

    # ------------------------------------------------------------------ #
    # Phase shifter section                                                #
    # ------------------------------------------------------------------ #
    emit("-" * _WIDE)
    emit("  PHASE SHIFTER STATES")
    emit("-" * _WIDE)

    _print_ps_table(chip_state, emit, show_phase_shift, show_target)

    emit("")
    emit("=" * _WIDE)


def _print_mzi_table(
    chip_state, emit, show_phase_shift: bool, show_target: bool
) -> None:
    """Print MZI state table."""
    # Column widths
    w = {
        "id": 8,
        "power": 10,
        "phi_init_r": 14,
        "phi_init_pi": 12,
        "phi_shift": 14,
        "p2pi": 9,
        "target": 16,
    }

    # Header row
    header = (
        f"  {'MZI':>{w['id']}}  "
        f"{'Power(W)':>{w['power']}}  "
        f"{'φ_init(rad)':>{w['phi_init_r']}}  "
        f"{'φ_init/π':>{w['phi_init_pi']}}"
    )
    if show_phase_shift:
        header += f"  {'φ_shift(rad)':>{w['phi_shift']}}"
    header += f"  {'P_2π(W)':>{w['p2pi']}}"
    if show_target:
        header += f"  {'Target PSR(dB)':>{w['target']}}"

    emit(header)
    emit("  " + "-" * (_WIDE - 2))

    if not chip_state.mzis:
        emit("  (no MZIs registered)")
        return

    for mzi_id in sorted(chip_state.mzis.keys(), key=_mzi_sort_key):
        mzi = chip_state.mzis[mzi_id]
        phi_pi = mzi.phi_init_rad / math.pi

        row = (
            f"  {mzi_id:>{w['id']}}  "
            f"{mzi.applied_power_watts:>{w['power']}.4f}  "
            f"{mzi.phi_init_rad:>+{w['phi_init_r']}.4f}  "
            f"{phi_pi:>+{w['phi_init_pi']}.4f}π"
        )
        if show_phase_shift:
            row += f"  {mzi.phase_shift_rad:>+{w['phi_shift']}.4f}"
        row += f"  {mzi.p2pi_watts:>{w['p2pi']}.4f}"
        if show_target:
            target_str = (
                f"{mzi.target_power_ratio_db:.2f}"
                if mzi.target_power_ratio_db is not None
                else "—"
            )
            row += f"  {target_str:>{w['target']}}"

        emit(row)


def _print_ps_table(
    chip_state, emit, show_phase_shift: bool, show_target: bool
) -> None:
    """Print phase shifter state table."""
    w = {
        "tap": 8,
        "power": 10,
        "phi_init_r": 14,
        "phi_init_pi": 12,
        "phi_shift": 14,
        "p2pi": 9,
        "target": 16,
    }

    header = (
        f"  {'Tap':>{w['tap']}}  "
        f"{'Power(W)':>{w['power']}}  "
        f"{'φ_init(rad)':>{w['phi_init_r']}}  "
        f"{'φ_init/π':>{w['phi_init_pi']}}"
    )
    if show_phase_shift:
        header += f"  {'φ_shift(rad)':>{w['phi_shift']}}"
    header += f"  {'P_2π(W)':>{w['p2pi']}}"
    if show_target:
        header += f"  {'Target φ(rad)':>{w['target']}}"

    emit(header)
    emit("  " + "-" * (_WIDE - 2))

    if not chip_state.phase_shifters:
        emit("  (no phase shifters registered)")
        return

    for tap_num in sorted(chip_state.phase_shifters.keys()):
        ps = chip_state.phase_shifters[tap_num]
        phi_pi = ps.phi_init_rad / math.pi

        row = (
            f"  {tap_num:>{w['tap']}}  "
            f"{ps.applied_power_watts:>{w['power']}.4f}  "
            f"{ps.phi_init_rad:>+{w['phi_init_r']}.4f}  "
            f"{phi_pi:>+{w['phi_init_pi']}.4f}π"
        )
        if show_phase_shift:
            row += f"  {ps.phase_shift_rad:>+{w['phi_shift']}.4f}"
        row += f"  {ps.p2pi_watts:>{w['p2pi']}.4f}"
        if show_target:
            target_str = (
                f"{ps.target_phase_rad:+.4f}"
                if ps.target_phase_rad is not None
                else "—"
            )
            row += f"  {target_str:>{w['target']}}"

        emit(row)


def _mzi_sort_key(mzi_id: str):
    """Sort MZI IDs numerically by stage then position (e.g. '2-1' < '3-2')."""
    try:
        stage, pos = mzi_id.split("-")
        return (int(stage), int(pos))
    except ValueError:
        return (999, 999)
