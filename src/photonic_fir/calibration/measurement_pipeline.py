from typing import Optional, Tuple
import time
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)

from photonic_fir.calibration import trim_spectrum_to_fsr
from photonic_fir.core import ExperimentConfig, ChipState
from photonic_fir.processing.gap_method import (
    recover_impulse_response_gap_from_df,
    extract_taps_from_cross_correlation,
)


try:
    from photonic_fir.hardware import measure_spectrum

    _HARDWARE_AVAILABLE = True
except ImportError:
    _HARDWARE_AVAILABLE = False


from photonic_fir.processing import (
    recover_impulse_response_from_df,
    detect_taps_windowed,
)

from .trim_spectrum_to_fsr import trim_spectrum_to_fsr


def measure_and_detect_taps(
    config: ExperimentConfig,
    df: Optional[pd.DataFrame] = None,
    file_name: Optional[str] = None,
    folder_dir: Optional[str] = None,
    use_gap_method: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure the optical spectrum and recover tap coefficients.

    Optionally accepts a pre-measured DataFrame to skip hardware measurement,
    useful for reprocessing saved data or offline analysis.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration (chip + measurement parameters).
    df : pd.DataFrame, optional
        Pre-measured spectrum DataFrame. If provided, skips hardware
        measurement entirely.

    Returns
    -------
    df : pd.DataFrame
        Measured (or passed-through) spectrum DataFrame.
    tap_times : np.ndarray
        Time positions of detected taps in picoseconds.
    tap_coeffs : np.ndarray
        Complex tap coefficients.
    """

    if not _HARDWARE_AVAILABLE:
        raise RuntimeError(
            "Hardware libraries not available. Cannot call measure_and_detect_taps."
        )

    if df is None:
        # 1. Measure spectrum
        df = measure_spectrum(
            center_wavelength_nm=config.measurement.center_wavelength_nm,
            wavelength_span_nm=config.measurement.wavelength_span_nm,
            num_averages=config.measurement.num_averages,
            edfa_port=config.measurement.edfa_port,
            edfa_baudrate=config.measurement.edfa_baudrate,
            edfa_output_power_dbm=config.measurement.edfa_output_power_dbm,
            ova_ip=config.measurement.ova_address,
            folder_dir=folder_dir,
            file_name=file_name,
        )
    else:
        logger.info("Using provided DataFrame, skipping spectrum measurement.")

    # 3. Trim spectrum to FSR
    df_trimmed, trim_info = trim_spectrum_to_fsr(
        df=df,
        fsr_hz=config.chip.fsr_hz,
        n_fsr=config.calibration.trim_n_fsr,
        freq_col=config.measurement.frequency_col,
        il_col=config.measurement.insertion_loss_col,
    )

    if use_gap_method:
        time_ps, h_time = recover_impulse_response_gap_from_df(
            df=df_trimmed,
            fsr_hz=config.chip.fsr_hz,
            freq_col=config.measurement.frequency_col,
            insertion_loss_col=config.measurement.insertion_loss_col,
        )

        tap_amps, tap_phases = extract_taps_from_cross_correlation(
            time_ps=time_ps,
            R=h_time,
            delay_step_s=config.chip.delay_step_s,
        )

        tap_coeffs = tap_amps * np.exp(1j * tap_phases)

        gap_offset = config.chip.n_taps - config.chip.n_signal_taps
        tap_times = np.array([
            (gap_offset + n) * config.chip.delay_step_s * 1e12
            for n in range(len(tap_amps))
        ])
    else:
        time_ps, h_time = recover_impulse_response_from_df(
            df=df_trimmed,
            fsr_hz=config.chip.fsr_hz,
            freq_col=config.measurement.frequency_col,
            insertion_loss_col=config.measurement.insertion_loss_col,
            n_tiles=config.calibration.num_tiles_for_kk_recovery,
        )

        tap_times, tap_coeffs = detect_taps_windowed(
            time_ps=time_ps,
            h_time=h_time,
            fsr_hz=config.chip.fsr_hz,
            delay_step_s=config.chip.delay_step_s,
            n_taps=config.chip.n_taps,
            window_width_ps=config.calibration.tap_detection_window_width_ps,
        )

    return df, tap_times, tap_coeffs, time_ps, h_time