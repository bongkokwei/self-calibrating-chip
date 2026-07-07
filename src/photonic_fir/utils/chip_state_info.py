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
from typing import Callable, Optional

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

    _print_state_table(
        emit,
        chip_state.mzis,
        id_label="MZI",
        sort_key=_mzi_sort_key,
        target_label="Target PSR(dB)",
        target_value=lambda mzi: (
            f"{mzi.target_power_ratio_db:.2f}"
            if mzi.target_power_ratio_db is not None
            else "—"
        ),
        empty_message="no MZIs registered",
        show_phase_shift=show_phase_shift,
        show_target=show_target,
    )

    emit("")

    # ------------------------------------------------------------------ #
    # Phase shifter section                                                #
    # ------------------------------------------------------------------ #
    emit("-" * _WIDE)
    emit("  PHASE SHIFTER STATES")
    emit("-" * _WIDE)

    _print_state_table(
        emit,
        chip_state.phase_shifters,
        id_label="Tap",
        sort_key=None,
        target_label="Target φ(rad)",
        target_value=lambda ps: (
            f"{ps.target_phase_rad:+.4f}" if ps.target_phase_rad is not None else "—"
        ),
        empty_message="no phase shifters registered",
        show_phase_shift=show_phase_shift,
        show_target=show_target,
    )

    emit("")
    emit("=" * _WIDE)


def _print_state_table(
    emit,
    items: dict,
    id_label: str,
    sort_key: Optional[Callable],
    target_label: str,
    target_value: Callable,
    empty_message: str,
    show_phase_shift: bool,
    show_target: bool,
) -> None:
    """Print a formatted state table for MZIs or phase shifters."""
    w = {
        "id": 8,
        "power": 10,
        "phi_init_r": 14,
        "phi_init_pi": 12,
        "phi_shift": 14,
        "p2pi": 9,
        "target": 16,
    }

    header = (
        f"  {id_label:>{w['id']}}  "
        f"{'Power(W)':>{w['power']}}  "
        f"{'φ_init(rad)':>{w['phi_init_r']}}  "
        f"{'φ_init/π':>{w['phi_init_pi']}}"
    )
    if show_phase_shift:
        header += f"  {'φ_shift(rad)':>{w['phi_shift']}}"
    header += f"  {'P_2π(W)':>{w['p2pi']}}"
    if show_target:
        header += f"  {target_label:>{w['target']}}"

    emit(header)
    emit("  " + "-" * (_WIDE - 2))

    if not items:
        emit(f"  ({empty_message})")
        return

    for key in sorted(items.keys(), key=sort_key):
        item = items[key]
        phi_pi = item.phi_init_rad / math.pi

        row = (
            f"  {key:>{w['id']}}  "
            f"{item.applied_power_watts:>{w['power']}.4f}  "
            f"{item.phi_init_rad:>+{w['phi_init_r']}.4f}  "
            f"{phi_pi:>+{w['phi_init_pi']}.4f}π"
        )
        if show_phase_shift:
            row += f"  {item.phase_shift_rad:>+{w['phi_shift']}.4f}"
        row += f"  {item.p2pi_watts:>{w['p2pi']}.4f}"
        if show_target:
            row += f"  {target_value(item):>{w['target']}}"

        emit(row)


def _mzi_sort_key(mzi_id: str):
    """Sort MZI IDs numerically by stage then position (e.g. '2-1' < '3-2')."""
    try:
        stage, pos = mzi_id.split("-")
        return (int(stage), int(pos))
    except ValueError:
        return (999, 999)
