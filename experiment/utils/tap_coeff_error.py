import numpy as np


def compute_tap_error(measured_taps, ideal_taps):
    """
    Compute error metrics between measured and ideal tap coefficients.

    Based on Xu et al. (2022) "Self-calibrating programmable photonic
    integrated circuits" - uses RMSE for amplitude and phase errors.

    Parameters:
    -----------
    measured_taps : np.ndarray
        Complex-valued array of measured/experimental tap coefficients
    ideal_taps : np.ndarray
        Complex-valued array of ideal/desired tap coefficients

    Returns:
    --------
    dict : Dictionary containing error metrics:
        - 'rmse_amplitude_db': RMSE of tap amplitudes in dB
        - 'rmse_phase_rad': RMSE of tap phases in radians
        - 'mse_complex': Mean square error of complex coefficients
        - 'amplitude_errors_db': Individual amplitude errors in dB (array)
        - 'phase_errors_rad': Individual phase errors in radians (array)

    Example:
    --------
    >>> measured = np.array([0.5+0.2j, 0.3-0.1j, 0.7+0.4j])
    >>> ideal = np.array([0.6+0.1j, 0.3-0.2j, 0.65+0.45j])
    >>> errors = compute_tap_error(measured, ideal)
    >>> print(f"Amplitude RMSE: {errors['rmse_amplitude_db']:.3f} dB")
    >>> print(f"Phase RMSE: {errors['rmse_phase_rad']:.3f} rad")
    """
    # Ensure inputs are numpy arrays
    measured_taps = np.array(measured_taps, dtype=complex)
    ideal_taps = np.array(ideal_taps, dtype=complex)

    # Validate inputs
    if len(measured_taps) != len(ideal_taps):
        raise ValueError(
            f"Array length mismatch: measured has {len(measured_taps)} taps, "
            f"ideal has {len(ideal_taps)} taps"
        )

    # Extract amplitudes and phases
    measured_amp = np.abs(measured_taps)
    ideal_amp = np.abs(ideal_taps)
    measured_phase = np.angle(measured_taps)
    ideal_phase = np.angle(ideal_taps)

    # Convert amplitudes to dB (add small value to avoid log(0))
    measured_amp_db = 20 * np.log10(measured_amp + 1e-12)
    ideal_amp_db = 20 * np.log10(ideal_amp + 1e-12)

    # Compute amplitude errors in dB
    amplitude_errors_db = measured_amp_db - ideal_amp_db
    rmse_amplitude_db = np.sqrt(np.mean(amplitude_errors_db**2))

    # Compute phase errors with proper wrapping to [-π, π]
    phase_diff = measured_phase - ideal_phase
    phase_errors_rad = np.angle(np.exp(1j * phase_diff))  # Wrap to [-π, π]
    rmse_phase_rad = np.sqrt(np.mean(phase_errors_rad**2))

    # Compute complex MSE (useful for overall error metric)
    complex_diff = measured_taps - ideal_taps
    mse_complex = np.mean(np.abs(complex_diff) ** 2)

    return {
        "rmse_amplitude_db": rmse_amplitude_db,
        "rmse_phase_rad": rmse_phase_rad,
        "mse_complex": mse_complex,
        "amplitude_errors_db": amplitude_errors_db,
        "phase_errors_rad": phase_errors_rad,
    }


def compute_tap_rmse_simple(measured_taps, ideal_taps):
    """
    Compute simple RMSE between measured and ideal complex tap coefficients.

    Parameters:
    -----------
    measured_taps : np.ndarray
        Complex-valued array of measured tap coefficients
    ideal_taps : np.ndarray
        Complex-valued array of ideal tap coefficients

    Returns:
    --------
    float : Root mean square error (unitless)

    Example:
    --------
    >>> measured = np.array([0.5+0.2j, 0.3-0.1j, 0.7+0.4j])
    >>> ideal = np.array([0.6+0.1j, 0.3-0.2j, 0.65+0.45j])
    >>> rmse = compute_tap_rmse_simple(measured, ideal)
    >>> print(f"RMSE: {rmse:.4f}")
    """
    measured_taps = np.array(measured_taps, dtype=complex)
    ideal_taps = np.array(ideal_taps, dtype=complex)

    if len(measured_taps) != len(ideal_taps):
        raise ValueError("Arrays must have the same length")

    # Compute complex difference and RMSE
    diff = measured_taps - ideal_taps
    rmse = np.sqrt(np.mean(np.abs(diff) ** 2))

    return rmse
