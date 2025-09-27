from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np

from .models import FeaturePacket


class DirectionEstimator:
    """Estimate coarse azimuth using GCC-PHAT between stereo channels."""

    def __init__(
        self,
        samplerate: int,
        max_tau: float = 0.0007,
        num_angles: int = 36,
        smoothing_alpha: float = 0.4,
    ) -> None:
        self.samplerate = samplerate
        self.max_tau = max_tau
        self.num_angles = num_angles
        self.smoothing_alpha = smoothing_alpha
        self._smoothed_angle: float | None = None

    def estimate(self, left: np.ndarray, right: np.ndarray) -> float:
        """Return azimuth in degrees where 0 is forward, positive to the right."""
        if left.size == 0 or right.size == 0:
            return 0.0
        left = left - np.mean(left)
        right = right - np.mean(right)
        n = left.size + right.size
        interp = 8
        left_fft = np.fft.rfft(left, n=n)
        right_fft = np.fft.rfft(right, n=n)
        cross_power = left_fft * np.conj(right_fft)
        cross_power /= np.abs(cross_power) + 1e-12
        corr = np.fft.irfft(cross_power, n=interp * n)
        corr = np.fft.fftshift(corr)
        lags = np.arange(-corr.size // 2, corr.size // 2)
        if self.max_tau:
            max_shift = min(int(self.max_tau * self.samplerate * interp), corr.size // 2 - 1)
            center = corr.size // 2
            start = center - max_shift
            end = center + max_shift + 1
            corr = corr[start:end]
            lags = lags[start:end]
        shift_index = int(np.argmax(np.abs(corr)))
        tau = lags[shift_index] / (self.samplerate * interp)
        theta = math.degrees(math.asin(np.clip(tau / self.max_tau, -1.0, 1.0)))
        theta = float(np.clip(theta, -90.0, 90.0))
        if not (0.0 < self.smoothing_alpha < 1.0):
            return theta
        if self._smoothed_angle is None:
            self._smoothed_angle = theta
        else:
            self._smoothed_angle = (
                self.smoothing_alpha * theta + (1 - self.smoothing_alpha) * self._smoothed_angle
            )
        return float(self._smoothed_angle)


def compute_feature_packet(
    frame: np.ndarray,
    samplerate: int,
    estimator: DirectionEstimator | None = None,
) -> FeaturePacket:
    if frame.ndim != 2 or frame.shape[1] < 2:
        raise ValueError("Frame must have at least two channels for ILD/IPD analysis")

    left = frame[:, 0]
    right = frame[:, 1]
    energy = float(np.sqrt(np.mean(frame**2)))
    window = np.hanning(frame.shape[0])[:, None]
    windowed = frame * window
    mix = windowed.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(mix))
    if spectrum.size == 0:
        spectrum = np.zeros(1)

    num_bands = 32
    band_chunks = np.array_split(spectrum, num_bands)
    band_energies = [float(chunk.mean()) for chunk in band_chunks]

    freqs = np.linspace(0, samplerate / 2, spectrum.size)
    spectral_centroid = float(
        np.sum(freqs * spectrum) / (spectrum.sum() + 1e-9)
    )
    onset_strength = float(
        np.maximum(0.0, np.max(np.diff(windowed[:, 0])) + np.max(np.diff(windowed[:, 1])))
    )

    eps = 1e-9
    low_band_energy = float(spectrum[freqs < 250].mean()) if np.any(freqs < 250) else 0.0
    mid_band_energy = float(
        spectrum[(freqs >= 250) & (freqs < 2000)].mean()
    ) if np.any((freqs >= 250) & (freqs < 2000)) else 0.0
    high_band_energy = float(
        spectrum[freqs >= 2000].mean()
    ) if np.any(freqs >= 2000) else 0.0
    geometric_mean = np.exp(np.mean(np.log(spectrum + eps)))
    arithmetic_mean = np.mean(spectrum + eps)
    spectral_flatness = float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))

    estimator = estimator or DirectionEstimator(samplerate)
    azimuth = estimator.estimate(left, right)

    return FeaturePacket(
        timestamp=time.time(),
        azimuth_deg=azimuth,
        energy=energy,
        band_energies=band_energies,
        onset_strength=onset_strength,
        spectral_centroid=spectral_centroid,
        low_band_energy=low_band_energy,
        mid_band_energy=mid_band_energy,
        high_band_energy=high_band_energy,
        spectral_flatness=spectral_flatness,
    )


def feature_stream(frames: Iterable[np.ndarray], samplerate: int) -> Iterable[FeaturePacket]:
    estimator = DirectionEstimator(samplerate)
    for frame in frames:
        yield compute_feature_packet(frame, samplerate, estimator)
