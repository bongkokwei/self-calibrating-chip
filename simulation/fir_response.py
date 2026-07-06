"""
Closed-form frequency response of the photonic FIR chip.

Rather than simulating individual MZIs, couplers and delay lines, this uses
the frequency-response equation given in the Supplementary Material of
Xu et al. (2022), "Self-calibrating programmable photonic integrated
circuits" (Nature Photonics 16, 595-602):

    H_chip(omega) = sum_{n=0}^{N} h(n) * exp(j * omega * n * T)

where h(n) = |h(n)| * exp(j*phi(n)) is the complex coefficient of the n-th
tap and T is the delay step between adjacent taps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from photonic_fir.core.config import load_config
from photonic_fir.core.data_structure import ExperimentConfig


@dataclass
class FIRFrequencyResponse:
    """Frequency response of an N-tap photonic FIR filter.

    Attributes
    ----------
    tap_weights : np.ndarray
        Complex tap coefficients h(n).
    tap_indices : np.ndarray
        Physical (0-based) delay index n for each entry in `tap_weights`.
    delay_step_s : float
        Delay T between adjacent tap indices, in seconds.
    """

    tap_weights: np.ndarray
    tap_indices: np.ndarray
    delay_step_s: float

    def __post_init__(self):
        self.tap_weights = np.asarray(self.tap_weights, dtype=complex)
        self.tap_indices = np.asarray(self.tap_indices, dtype=float)
        if self.tap_weights.shape != self.tap_indices.shape:
            raise ValueError("tap_weights and tap_indices must have the same shape")

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        include_reference: bool = False,
        reference_amplitude: complex = 1.0 + 0.0j,
    ) -> "FIRFrequencyResponse":
        """Build tap weights and delays from an experiment config.

        Tap weights are read directly from `config.target` (the calibration
        target) and delays from `config.chip` -- no tap-weight computation
        happens here.

        Parameters
        ----------
        config : ExperimentConfig
            Loaded experiment configuration.
        include_reference : bool
            If False (default), model only the 8-tap signal-processing core
            (the part of the filter you design via `config.target`).
            If True, also place a reference tap at `config.chip.reference_tap_idx`
            with amplitude `reference_amplitude`, approximating the
            calibration-port response H_chip = H_ref + H_spc described in the
            paper.
        reference_amplitude : complex
            Complex weight of the reference tap when `include_reference=True`.

        Returns
        -------
        FIRFrequencyResponse
        """
        signal_weights = config.target.get_target_taps(config.chip.n_signal_taps)
        signal_indices = np.array(config.chip.signal_tap_indices)

        if not include_reference:
            return cls(signal_weights, signal_indices, config.chip.delay_step_s)

        tap_weights = np.zeros(config.chip.n_taps, dtype=complex)
        tap_weights[signal_indices] = signal_weights
        tap_weights[config.chip.reference_tap_idx] = reference_amplitude
        tap_indices = np.arange(config.chip.n_taps)
        return cls(tap_weights, tap_indices, config.chip.delay_step_s)

    @classmethod
    def from_yaml(cls, config_path: str, **kwargs) -> "FIRFrequencyResponse":
        """Build directly from a YAML config file path. See `from_config`."""
        return cls.from_config(load_config(config_path), **kwargs)

    def response(self, frequencies_hz: np.ndarray) -> np.ndarray:
        """Complex frequency response H(f) = sum_n h(n) * exp(j*2*pi*f*n*T)."""
        frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        omega = 2 * np.pi * frequencies_hz
        phase = np.exp(1j * np.outer(omega, self.tap_indices) * self.delay_step_s)
        return phase @ self.tap_weights

    def magnitude_db(self, frequencies_hz: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
        """Insertion-loss-style magnitude response, 10*log10(|H(f)|^2)."""
        power = np.abs(self.response(frequencies_hz)) ** 2
        return 10 * np.log10(np.maximum(power, 10 ** (floor_db / 10)))

    def phase_rad(self, frequencies_hz: np.ndarray) -> np.ndarray:
        """Phase response arg(H(f)), in radians."""
        return np.angle(self.response(frequencies_hz))

    def impulse_response(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (delay_s, tap_weights), sorted by increasing delay."""
        order = np.argsort(self.tap_indices)
        return self.tap_indices[order] * self.delay_step_s, self.tap_weights[order]
