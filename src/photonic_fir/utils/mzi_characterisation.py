"""
MZI Characterisation Utilities
==============================

Functions for fitting and analysing MZI voltage-power-phase relationships.

Theory:
-------
For a thermo-optic phase shifter in an MZI:
    
    Power: P = V² / R
    Phase: φ = (P / P_2π) × 2π + φ_init
    PSR: PSR_dB = 10*log₁₀[tan²(φ/2)]

Where:
    V: Applied voltage
    R: Heater resistance
    P_2π: Power required for 2π phase shift
    φ_init: Initial phase offset
    PSR: Power splitting ratio (bar/cross ports)

The PSR vs voltage² relationship is:
    PSR_dB = 10*log₁₀[tan²(π*V²/(V_2π²*R) + φ_init/2)]

Where V_2π = √(R * P_2π)
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Dict


def mzi_psr_model(
    voltage_squared: np.ndarray,
    v_2pi_squared: float,
    phi_init: float,
    resistance: float,
) -> np.ndarray:
    """
    Model for MZI power splitting ratio as a function of voltage squared.
    
    The power splitting ratio (PSR) in an MZI follows:
        
        PSR_dB = 10*log₁₀[cos²(φ/2) / sin²(φ/2)] = 20*log₁₀|cot(φ/2)|
    
    where the phase φ is related to voltage by:
        
        φ = 2π * V² / V_2π² + φ_init
    
    Combining:
        
        PSR_dB = 20*log₁₀|cot(π*V²/V_2π² + φ_init/2)|
    
    Parameters
    ----------
    voltage_squared : np.ndarray
        Voltage squared values (V²)
    v_2pi_squared : float
        Square of voltage for 2π phase shift (V²)
    phi_init : float
        Initial phase offset (radians)
    resistance : float
        Heater resistance (Ω) - included for completeness but cancels in the ratio
    
    Returns
    -------
    psr_db : np.ndarray
        Power splitting ratio in dB
    
    Notes
    -----
    The resistance parameter is included to match the physical relationship
    P = V²/R, but it cancels when computing φ = (P/P_2π)*2π since both
    the applied power and P_2π scale with 1/R.
    
    Physical interpretation:
    - φ = 0: PSR → +∞ dB (all power in bar port)
    - φ = π/2: PSR = 0 dB (equal 50:50 split)
    - φ = π: PSR → -∞ dB (all power in cross port)
    """
    # Phase shift: φ = 2π * V² / V_2π² + φ_init
    phase = 2 * np.pi * voltage_squared / v_2pi_squared + phi_init
    
    # PSR = 20*log₁₀|cot(φ/2)|
    # To avoid singularities at φ = 0, π, 2π, add small epsilon
    epsilon = 1e-12
    psr_db = 20 * np.log10(np.abs(1.0 / (np.tan(phase / 2) + epsilon)) + epsilon)
    
    return psr_db


def fit_mzi_v2pi_and_phi_init(
    voltages: np.ndarray,
    psr_db: np.ndarray,
    resistance_ohm: float,
    v_2pi_initial_guess: float = 25.0,
    phi_init_guess: float = 0.0,
) -> Tuple[float, float, float, Dict]:
    """
    Fit MZI V_2π and φ_init from PSR vs voltage data using nonlinear least squares.
    
    Fits the relationship:
        PSR_dB = 20*log₁₀|cot(π*V²/V_2π² + φ_init/2)|
    
    Parameters
    ----------
    voltages : np.ndarray
        Applied voltage values (V)
    psr_db : np.ndarray
        Measured power splitting ratios (dB)
    resistance_ohm : float
        Heater resistance (Ω)
    v_2pi_initial_guess : float
        Initial guess for V_2π (V), default 25.0
    phi_init_guess : float
        Initial guess for φ_init (radians), default 0.0
    
    Returns
    -------
    v_2pi : float
        Fitted voltage for 2π phase shift (V)
    phi_init : float
        Fitted initial phase offset (radians)
    r_squared : float
        Coefficient of determination (goodness of fit)
    fit_info : Dict
        Additional fitting information including:
        - p_2pi: Power for 2π phase shift (W)
        - covariance: Covariance matrix from fit
        - residuals: Fit residuals
        - rmse: Root mean square error
    
    Examples
    --------
    >>> voltages = np.linspace(0, 30, 50)
    >>> psr_measured = mzi_psr_model(voltages**2, 25**2, 0.1, 600)
    >>> v_2pi, phi_init, r2, info = fit_mzi_v2pi_and_phi_init(
    ...     voltages, psr_measured, resistance_ohm=600)
    >>> np.isclose(v_2pi, 25.0, rtol=0.01)
    True
    >>> np.isclose(phi_init, 0.1, rtol=0.1)
    True
    """
    # Convert to voltage squared for fitting
    voltage_squared = voltages ** 2
    
    # Estimate initial guess from data if not provided
    # V_2π estimation: find voltage range that covers approximately 2π phase
    psr_range = np.max(psr_db) - np.min(psr_db)
    v_range = np.max(voltages) - np.min(voltages)
    
    # If PSR range is large, data likely covers significant phase range
    # Use data-driven estimate if provided guess seems wrong
    if psr_range > 10:  # dB, indicates significant phase modulation
        # Estimate V_2π from voltage range that gives ~2π phase
        # Assume we're covering roughly 1-2π in the data
        v_2pi_estimate = v_range * np.sqrt(2)  # Rough estimate
        if abs(v_2pi_estimate - v_2pi_initial_guess) / v_2pi_initial_guess > 0.5:
            v_2pi_initial_guess = v_2pi_estimate
    
    # Initial parameter guess: [V_2π², φ_init]
    p0 = [v_2pi_initial_guess ** 2, phi_init_guess]
    
    # More relaxed bounds: V_2π from 10-50V, φ_init in [-π, π]
    bounds = ([100.0, -np.pi], [2500.0, np.pi])  # V_2π² from 10²to 50²
    
    try:
        # Perform nonlinear least squares fit
        # Note: We pass resistance as a fixed parameter
        popt, pcov = curve_fit(
            lambda v2, v2pi2, phi: mzi_psr_model(v2, v2pi2, phi, resistance_ohm),
            voltage_squared,
            psr_db,
            p0=p0,
            bounds=bounds,
            max_nfev=20000,  # Increased max iterations
            ftol=1e-10,  # Tighter tolerance
            xtol=1e-10,
        )
        
        v_2pi_squared_fit, phi_init_fit = popt
        v_2pi = np.sqrt(v_2pi_squared_fit)
        phi_init = phi_init_fit
        
        # Calculate goodness of fit (R²)
        psr_fit = mzi_psr_model(voltage_squared, v_2pi_squared_fit, phi_init_fit, resistance_ohm)
        ss_res = np.sum((psr_db - psr_fit) ** 2)
        ss_tot = np.sum((psr_db - np.mean(psr_db)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate P_2π from fitted V_2π
        p_2pi = v_2pi ** 2 / resistance_ohm
        
        # Additional fit statistics
        residuals = psr_db - psr_fit
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        fit_info = {
            "p_2pi_watts": p_2pi,
            "covariance": pcov,
            "residuals": residuals,
            "rmse_db": rmse,
            "fitted_psr": psr_fit,
            "voltage_squared": voltage_squared,
        }
        
    except RuntimeError as e:
        print(f"Fit failed: {e}")
        print("Returning initial guesses with R² = 0")
        v_2pi = v_2pi_initial_guess
        phi_init = phi_init_guess
        r_squared = 0.0
        fit_info = {
            "p_2pi_watts": v_2pi ** 2 / resistance_ohm,
            "covariance": None,
            "residuals": None,
            "rmse_db": np.nan,
            "fitted_psr": None,
            "voltage_squared": voltage_squared,
        }
    
    return v_2pi, phi_init, r_squared, fit_info


def print_fit_results(
    mzi_id: str,
    v_2pi: float,
    phi_init: float,
    r_squared: float,
    fit_info: Dict,
    resistance_ohm: float,
) -> None:
    """
    Print formatted fitting results.
    
    Parameters
    ----------
    mzi_id : str
        MZI identifier (e.g., "4-5")
    v_2pi : float
        Fitted V_2π (V)
    phi_init : float
        Fitted φ_init (radians)
    r_squared : float
        R² goodness of fit
    fit_info : Dict
        Additional fit information from fit_mzi_v2pi_and_phi_init()
    resistance_ohm : float
        Heater resistance (Ω)
    """
    print(f"\n{'='*70}")
    print(f"MZI {mzi_id} - Fitted Parameters (Nonlinear Least Squares)")
    print(f"{'='*70}")
    print(f"V_2π:        {v_2pi:.3f} V")
    print(f"φ_init:      {phi_init:.4f} rad ({np.degrees(phi_init):.2f}°)")
    print(f"P_2π:        {fit_info['p_2pi_watts']:.4f} W ({fit_info['p_2pi_watts']*1000:.2f} mW)")
    print(f"R²:          {r_squared:.6f}")
    print(f"RMSE:        {fit_info['rmse_db']:.3f} dB")
    print(f"Resistance:  {resistance_ohm:.1f} Ω")
    print(f"{'='*70}\n")
