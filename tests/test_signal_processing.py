import numpy as np

from spatial_hud.signal_processing import DirectionEstimator, compute_feature_packet


def test_direction_estimator_right_side():
    samplerate = 48_000
    estimator = DirectionEstimator(samplerate)
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(int(0.02 * samplerate))
    delay_samples = int(0.0003 * samplerate)
    padded = np.pad(signal, (delay_samples, 0))[: signal.size]
    frame = np.stack([signal, padded], axis=-1)
    feature = compute_feature_packet(frame, samplerate, estimator)
    assert feature.azimuth_deg < 0  # right ear receives later, so source on left


def test_direction_estimator_left_side():
    samplerate = 48_000
    estimator = DirectionEstimator(samplerate)
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(int(0.02 * samplerate))
    delay_samples = int(0.0003 * samplerate)
    padded = np.pad(signal, (delay_samples, 0))[: signal.size]
    frame = np.stack([padded, signal], axis=-1)
    feature = compute_feature_packet(frame, samplerate, estimator)
    assert feature.azimuth_deg > 0
