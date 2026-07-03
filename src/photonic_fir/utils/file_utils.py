"""
Utility functions for photonic FIR filter experiments.

General-purpose helper functions that don't fit into specific modules.
"""

import re
from pathlib import Path
from typing import Optional, Union


def get_next_run_dir(base_dir: str = "measurements", prefix: str = "run") -> str:
    """
    Get the next available run directory by auto-incrementing run number.

    Checks existing directories and returns the next available run number.
    For example, if run_001 and run_002 exist, returns run_003.

    Parameters
    ----------
    base_dir : str
        Base directory to search in
    prefix : str
        Prefix for run directories (e.g., "v2pi_batch_scan_results")

    Returns
    -------
    output_dir : str
        Full path to next run directory (e.g., "measurements/run_003")

    Examples
    --------
    >>> get_next_run_dir("measurements", "v2pi_scan")
    'measurements/v2pi_scan_run_001'

    >>> get_next_run_dir("data", "calibration")
    'data/calibration_run_005'  # if run_001 through run_004 exist
    """
    base_path = Path(base_dir)

    # Create base directory if it doesn't exist
    base_path.mkdir(parents=True, exist_ok=True)

    # Find all existing run directories matching the pattern
    pattern = f"{prefix}_run_*"
    existing_runs = list(base_path.glob(pattern))

    if not existing_runs:
        # No existing runs, start with run_001
        next_run = 1
    else:
        # Extract run numbers from directory names
        run_numbers = []
        for run_dir in existing_runs:
            # Extract number from pattern like "v2pi_batch_scan_results_run_003"
            match = run_dir.name.split("_run_")
            if len(match) == 2:
                try:
                    run_num = int(match[1])
                    run_numbers.append(run_num)
                except ValueError:
                    # Skip directories that don't have valid run numbers
                    continue

        # Get next run number
        if run_numbers:
            next_run = max(run_numbers) + 1
        else:
            next_run = 1

    # Format with leading zeros (e.g., run_001, run_002, ...)
    output_dir = str(base_path / f"{prefix}_run_{next_run:03d}")

    return output_dir


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        Directory path to create

    Returns
    -------
    path : Path
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def generate_mzi_ids(n_stages: int = 4) -> list[str]:
    """
    Generate all MZI IDs for a binary tree structure.

    For a tree with n_stages, stage k has 2^(k-1) MZIs.

    Parameters
    ----------
    n_stages : int
        Number of stages in the MZI tree (default: 4 for 16-tap filter).

    Returns
    -------
    list[str]
        List of MZI IDs in format "stage-position" (e.g., "1-1", "2-1", "2-2").
    """
    return [
        f"{stage}-{pos}"
        for stage in range(1, n_stages + 1)
        for pos in range(1, 2 ** (stage - 1) + 1)
    ]


def extract_voltage_from_filename(filename: Union[str, Path]) -> Optional[float]:
    """
    Extract voltage value from a filename.

    Expected format: contains a pattern like "1.23v" or "0.50v".

    Parameters
    ----------
    filename : str | Path
        Filename or path to the measurement file.

    Returns
    -------
    float | None
        Extracted voltage value, or None if not found.
    """
    match = re.search(r"(\d+\.?\d*)v", Path(filename).name, re.IGNORECASE)
    return float(match.group(1)) if match else None


# Add more general utilities as needed
