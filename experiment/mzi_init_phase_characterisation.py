"""
mzi_init_phase_characterisation.py

Two-step method for determining initial phase offsets (φ_init) of MZIs
before the main self-calibration process, as described in Xu et al. (2022).

The method characterises the phase-to-power relationship and inherent phase
offsets through a simple two-iteration measurement process.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class MZICharacterisation:
    """Results from MZI initial phase characterisation."""

    mzi_id: str
    phi_init_rad: float  # Initial phase offset
    p2pi_watts: float  # Power for 2π phase shift
    power_at_null_watts: float  # Power at intensity null
    psr_at_null_db: float  # Power splitting ratio at null


def two_step_phi_init_characterisation(
    mzi_id: str,
    measure_psr_callback,
    apply_power_callback,
    power_sweep_step: float = 0.05,
    max_power: float = 1.0,
    p2pi_nominal: float = 0.75,
) -> MZICharacterisation:
    """
    Determine initial phase offset (φ_init) using two-step method.

    The method works as follows:

    **Step 1: Find the intensity null**
    - Sweep power P from 0 to max_power in small steps
    - Measure power splitting ratio PSR(P) at each step
    - Find power P_null where PSR is minimum (intensity null)
    - At this null: φ_MZI(P_null) + φ_init = π/2 + nπ

    **Step 2: Apply small power offset**
    - Apply power P_null + ΔP (e.g., ΔP = 0.05 W)
    - Measure new PSR
    - Determine which branch of arcsin we're on from PSR change
    - Calculate φ_init from the relationship:

      φ_init = π/2 + nπ - φ_MZI(P_null)

    where φ_MZI(P) = (P / P_2π) × 2π

    The sign ambiguity is resolved by observing whether PSR increases
    or decreases when power increases from P_null.

    Args:
        mzi_id: MZI identifier (e.g., "2-1")
        measure_psr_callback: Function that measures PSR when called,
            returns float in dB
        apply_power_callback: Function that sets power P in watts,
            signature: apply_power_callback(mzi_id, power_watts)
        power_sweep_step: Step size for power sweep in watts
        max_power: Maximum power to sweep to in watts
        p2pi_nominal: Nominal power for 2π phase shift in watts

    Returns:
        MZICharacterisation object with phi_init and related parameters

    Example:
        >>> def measure_psr():
        ...     # Measure chip, extract PSR for this MZI
        ...     return measured_psr_db
        >>>
        >>> def apply_power(mzi_id, power_watts):
        ...     # Set voltage controller for this MZI
        ...     controller.set_voltage(channel, power_to_voltage(power_watts))
        >>>
        >>> result = two_step_phi_init_characterisation(
        ...     mzi_id="2-1",
        ...     measure_psr_callback=measure_psr,
        ...     apply_power_callback=apply_power,
        ...     power_sweep_step=0.05,
        ...     max_power=1.0,
        ...     p2pi_nominal=0.75
        ... )
        >>> print(f"φ_init = {result.phi_init_rad:.3f} rad")
    """

    # Step 1: Find intensity null by power sweep
    print(f"\n=== Step 1: Finding intensity null for MZI {mzi_id} ===")

    powers = np.arange(0, max_power + power_sweep_step, power_sweep_step)
    psr_values = []

    for power in powers:
        # Apply power
        apply_power_callback(mzi_id, power)

        # Measure PSR
        psr_db = measure_psr_callback()
        psr_values.append(psr_db)

        print(f"  P = {power:.3f} W → PSR = {psr_db:.2f} dB")

    # Find minimum PSR (intensity null)
    psr_values = np.array(psr_values)
    null_idx = np.argmin(psr_values)
    power_at_null = powers[null_idx]
    psr_at_null = psr_values[null_idx]

    print(f"\n  → Null found at P = {power_at_null:.3f} W, PSR = {psr_at_null:.2f} dB")

    # At null: φ_MZI(P_null) + φ_init = π/2 + nπ
    # where φ_MZI(P) = (P / P_2π) × 2π
    phi_mzi_at_null = (power_at_null / p2pi_nominal) * 2 * np.pi

    # Step 2: Apply small offset to determine which branch
    print(f"\n=== Step 2: Determining branch (sign of φ_init) ===")

    power_offset = power_at_null + power_sweep_step

    # Ensure we don't exceed max power
    if power_offset > max_power:
        power_offset = power_at_null - power_sweep_step
        if power_offset < 0:
            raise ValueError(
                f"Cannot determine branch: null at P={power_at_null:.3f} W "
                f"with no room for ±{power_sweep_step:.3f} W offset"
            )

    # Apply offset power
    apply_power_callback(mzi_id, power_offset)
    psr_at_offset = measure_psr_callback()

    print(f"  P = {power_offset:.3f} W → PSR = {psr_at_offset:.2f} dB")

    # Determine branch from PSR change
    # If PSR increases when power increases from null, we're on upper branch (n=0)
    # If PSR decreases, we're on lower branch (n=1)

    # The MZI transfer function for power splitting ratio is:
    # PSR(φ) ∝ sin²(φ/2)
    # At null: φ = π/2 + nπ
    #
    # For n=0 (upper branch, φ = π/2):
    #   - Increasing φ → PSR increases (moving away from null towards π)
    # For n=1 (lower branch, φ = 3π/2):
    #   - Increasing φ → PSR decreases (moving towards 2π)

    power_increased = power_offset > power_at_null
    psr_increased = psr_at_offset > psr_at_null

    if power_increased and psr_increased:
        # On upper branch: φ = π/2 + φ_init
        n = 0
        phi_init = np.pi / 2 - phi_mzi_at_null
        print(f"  → PSR increased: on upper branch (n=0)")
    elif power_increased and not psr_increased:
        # On lower branch: φ = 3π/2 + φ_init
        n = 1
        phi_init = 3 * np.pi / 2 - phi_mzi_at_null
        print(f"  → PSR decreased: on lower branch (n=1)")
    else:
        # Power decreased case
        if psr_increased:
            # On lower branch
            n = 1
            phi_init = 3 * np.pi / 2 - phi_mzi_at_null
            print(f"  → PSR increased: on lower branch (n=1)")
        else:
            # On upper branch
            n = 0
            phi_init = np.pi / 2 - phi_mzi_at_null
            print(f"  → PSR decreased: on upper branch (n=0)")

    # Wrap to [-π, π]
    phi_init = np.arctan2(np.sin(phi_init), np.cos(phi_init))

    print(f"\n  → φ_init = {phi_init:.3f} rad ({np.degrees(phi_init):.1f}°)")
    print(f"  → φ_MZI(P_null) = {phi_mzi_at_null:.3f} rad")
    print(f"  → Total phase at null: {phi_mzi_at_null + phi_init:.3f} rad")
    print(f"     (should be ≈ {np.pi/2 + n*np.pi:.3f} rad)")

    return MZICharacterisation(
        mzi_id=mzi_id,
        phi_init_rad=phi_init,
        p2pi_watts=p2pi_nominal,
        power_at_null_watts=power_at_null,
        psr_at_null_db=psr_at_null,
    )


def characterise_all_mzis(
    mzi_ids: list,
    measure_psr_for_mzi_callback,
    apply_power_callback,
    power_sweep_step: float = 0.05,
    max_power: float = 1.0,
    p2pi_nominal: float = 0.75,
) -> Dict[str, MZICharacterisation]:
    """
    Characterise initial phase offsets for all MZIs.

    Args:
        mzi_ids: List of MZI identifiers to characterise
        measure_psr_for_mzi_callback: Function that measures PSR for a given MZI,
            signature: measure_psr_for_mzi_callback(mzi_id) -> float (dB)
        apply_power_callback: Function that sets power,
            signature: apply_power_callback(mzi_id, power_watts)
        power_sweep_step: Step size for power sweep in watts
        max_power: Maximum power in watts
        p2pi_nominal: Nominal power for 2π phase shift in watts

    Returns:
        Dictionary mapping MZI IDs to MZICharacterisation objects

    Example:
        >>> mzi_ids = ["2-1", "3-3", "3-4", "4-5", "4-6", "4-7", "4-8"]
        >>> results = characterise_all_mzis(
        ...     mzi_ids=mzi_ids,
        ...     measure_psr_for_mzi_callback=lambda mzi_id: measure_and_extract_psr(mzi_id),
        ...     apply_power_callback=lambda mzi_id, p: controller.set_power(mzi_id, p)
        ... )
    """

    results = {}

    for mzi_id in mzi_ids:
        print(f"\n{'='*70}")
        print(f"Characterising MZI {mzi_id}")
        print(f"{'='*70}")

        # Create callback that measures PSR for this specific MZI
        def measure_psr():
            return measure_psr_for_mzi_callback(mzi_id)

        result = two_step_phi_init_characterisation(
            mzi_id=mzi_id,
            measure_psr_callback=measure_psr,
            apply_power_callback=apply_power_callback,
            power_sweep_step=power_sweep_step,
            max_power=max_power,
            p2pi_nominal=p2pi_nominal,
        )

        results[mzi_id] = result

    # Summary
    print(f"\n{'='*70}")
    print("CHARACTERISATION SUMMARY")
    print(f"{'='*70}")
    for mzi_id, result in results.items():
        print(
            f"{mzi_id}: φ_init = {result.phi_init_rad:6.3f} rad "
            f"({np.degrees(result.phi_init_rad):6.1f}°), "
            f"P_null = {result.power_at_null_watts:.3f} W"
        )

    return results


def extract_phi_init_dict(
    characterisation_results: Dict[str, MZICharacterisation],
) -> Dict[str, float]:
    """
    Extract just the phi_init values as a dictionary.

    Args:
        characterisation_results: Output from characterise_all_mzis()

    Returns:
        Dictionary mapping MZI IDs to phi_init in radians
    """
    return {
        mzi_id: result.phi_init_rad
        for mzi_id, result in characterisation_results.items()
    }
