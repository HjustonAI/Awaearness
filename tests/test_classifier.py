import numpy as np

from spatial_hud.event_classifier import EventClassifier
from spatial_hud.models import DistanceBucket, FeaturePacket


def make_packet(
    energy: float,
    onset: float,
    centroid: float,
    bands: list[float],
    azimuth: float = 0.0,
) -> FeaturePacket:
    band_arr = np.array(bands, dtype=float)
    third = max(1, band_arr.size // 3)
    low_band = float(band_arr[:third].mean())
    mid_band = float(band_arr[third : 2 * third].mean()) if band_arr.size >= 2 * third else low_band
    high_band = float(band_arr[2 * third :].mean()) if band_arr.size > 2 * third else mid_band
    eps = 1e-9
    spectral_flatness = float(
        np.exp(np.mean(np.log(band_arr + eps))) / (np.mean(band_arr + eps) + eps)
    )

    return FeaturePacket(
        timestamp=0.0,
        azimuth_deg=azimuth,
        energy=energy,
        band_energies=bands,
        onset_strength=onset,
        spectral_centroid=centroid,
        low_band_energy=low_band,
        mid_band_energy=mid_band,
        high_band_energy=high_band,
        spectral_flatness=spectral_flatness,
    )


def test_footstep_detection():
    classifier = EventClassifier()
    bands = [0.01] * 32
    bands[10:22] = [0.02] * 12
    bands[-12:] = [0.035] * 12
    packet = make_packet(
        energy=0.07,
        onset=0.09,
        centroid=1200.0,
        bands=bands,
    )
    event = classifier.classify(packet)
    assert event.kind == "footstep"
    assert event.distance_bucket is DistanceBucket.MID


def test_vehicle_detection():
    classifier = EventClassifier()
    bands = [0.01] * 32
    for idx in range(0, 10):
        bands[idx] = 0.09
    packet = make_packet(
        energy=0.09,
        onset=0.01,
        centroid=600.0,
        bands=bands,
    )
    event = classifier.classify(packet)
    assert event.kind == "vehicle"
    assert event.distance_bucket is DistanceBucket.NEAR


def test_gunfire_detection():
    classifier = EventClassifier()
    packet = make_packet(
        energy=0.2,
        onset=0.06,
        centroid=5000.0,
        bands=[0.08] * 32,
    )
    event = classifier.classify(packet)
    assert event.kind == "gunfire"
    assert event.distance_bucket is DistanceBucket.NEAR
