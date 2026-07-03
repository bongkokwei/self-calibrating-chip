# spsa_gradient.py

import numpy as np
import logging
import time
from typing import Dict, Callable, Tuple

from .voltage_adjustment import _wrap_and_clip_power

logger = logging.getLogger(__name__)


def estimate_ps_gradients_spsa(
    tap_nums: list[int],
    current_ps_powers: Dict[int, float],
    current_ps_phases: Dict[int, float],
    measure_fn: Callable[[Dict[int, float]], Dict[int, float]],
    perturbation_power: float = 0.002,  # δP in watts (~10% of typical P_2pi)
    settle_time_s: float = 5.0,
    n_averages: int = 1,
    rng: np.random.Generator = None,
) -> Dict[int, float]:
    """
    Estimate per-tap PS phase gradients dφ/dP using SPSA.

    Requires exactly 2 OVA sweeps regardless of tap count.

    Parameters
    ----------
    tap_nums : list[int]
        Active PS tap numbers to estimate gradients for.
    current_ps_powers : dict
        Current heater powers {tap_num: P_watts}.
    current_ps_phases : dict
        Current measured phases {tap_num: phi_rad}. Used as baseline.
    measure_fn : callable
        Function that accepts {tap_num: P_watts} and returns
        {tap_num: phi_rad} — wraps your OVA sweep + KK recovery.
    perturbation_power : float
        Perturbation magnitude c (W). Should be small but above noise floor.
        Typical P_2pi ~ 50–100 mW, so 2–5 mW is reasonable.
    settle_time_s : float
        Thermal settle time between perturbation and measurement (s).
    n_averages : int
        Number of SPSA pairs to average (reduces variance, costs 2n sweeps).
    rng : np.random.Generator
        Optional RNG for reproducibility.

    Returns
    -------
    gradients : dict {tap_num: dφ/dP (rad/W)}
        Positive values expected for a well-behaved thermo-optic PS.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(tap_nums)
    grad_accumulator = np.zeros(n)

    for avg_idx in range(n_averages):
        # --- Draw Rademacher vector: each element ±1 with equal probability ---
        delta = rng.choice([-1.0, 1.0], size=n)

        # --- Positive perturbation: P + c*Δ ---
        p_plus = {
            tap: current_ps_powers[tap] + perturbation_power * delta[i]
            for i, tap in enumerate(tap_nums)
        }
        logger.info(
            f"SPSA avg {avg_idx+1}/{n_averages}: applying +perturbation, settling {settle_time_s}s"
        )
        time.sleep(settle_time_s)
        phi_plus = measure_fn(p_plus)

        # --- Negative perturbation: P - c*Δ ---
        p_minus = {
            tap: current_ps_powers[tap] - perturbation_power * delta[i]
            for i, tap in enumerate(tap_nums)
        }
        logger.info(
            f"SPSA avg {avg_idx+1}/{n_averages}: applying -perturbation, settling {settle_time_s}s"
        )
        time.sleep(settle_time_s)
        phi_minus = measure_fn(p_minus)

        # --- SPSA gradient estimate ---
        # g_k = (φ(P+cΔ) - φ(P-cΔ)) / (2c * Δ_k)
        for i, tap in enumerate(tap_nums):
            dphi = phi_plus[tap] - phi_minus[tap]
            grad_accumulator[i] += dphi / (2 * perturbation_power * delta[i])

    gradients = {
        tap: grad_accumulator[i] / n_averages for i, tap in enumerate(tap_nums)
    }

    # Sanity check — thermo-optic gradient should be positive
    for tap, g in gradients.items():
        if g < 0:
            logger.warning(
                f"PS {tap}: negative gradient {g:.4f} rad/W — "
                f"possible crosstalk dominance or noisy estimate"
            )

    logger.info("SPSA gradient estimates (rad/W):")
    for tap, g in gradients.items():
        p2pi_implied = 2 * np.pi / g if abs(g) > 1e-6 else float("inf")
        logger.info(
            f"  PS {tap}: {g:.4f} rad/W  (implied P_2π = {p2pi_implied*1e3:.1f} mW)"
        )

    return gradients


def apply_ps_corrections_with_spsa_gradients(
    ps_phase_errors: Dict[int, float],
    current_ps_powers: Dict[int, float],
    gradients: Dict[int, float],
    learning_rate: float,
    min_power: float,
    max_power: float,
    power_for_2pi: float,  # fallback only
    wrap_phase: bool = True,
    ps_dead_zone_rad: float = 0.0,
    fallback_if_bad_grad: bool = True,
) -> Dict[int, float]:
    """
    Compute new PS powers using empirical SPSA gradients.

    ΔP = (φ_err / g_hat) * lr
    P_new = clip(P + ΔP)

    Falls back to nominal P_2pi if gradient is degenerate.
    """
    new_ps_powers = {}

    for tap, phi_err in ps_phase_errors.items():
        # Wrap error to [-π, π]
        phi_err_wrapped = float(np.angle(np.exp(1j * phi_err)))

        if abs(phi_err_wrapped) < ps_dead_zone_rad:
            new_ps_powers[tap] = current_ps_powers[tap]
            continue

        g_hat = gradients.get(tap, None)

        # Fallback to P_2pi-based step if gradient is missing or degenerate
        if g_hat is None or abs(g_hat) < 1e-6:
            if fallback_if_bad_grad:
                logger.warning(f"PS {tap}: degenerate gradient, falling back to P_2π")
                delta_P = (
                    (phi_err_wrapped / (2 * np.pi)) * power_for_2pi * learning_rate
                )
            else:
                delta_P = 0.0
        else:
            delta_P = (phi_err_wrapped / g_hat) * learning_rate

        current_P = current_ps_powers.get(tap, 0.0)
        new_P = current_P + delta_P

        new_ps_powers[tap] = _wrap_and_clip_power(
            new_P, power_for_2pi, min_power, max_power, wrap_phase
        )
        logger.info(
            f"  PS {tap}: φ_err={phi_err_wrapped:.4f} rad, "
            f"g_hat={g_hat:.4f} rad/W, ΔP={delta_P:.4f} W → P={new_ps_powers[tap]:.4f} W"
        )

    return new_ps_powers
