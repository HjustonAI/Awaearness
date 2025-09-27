from spatial_hud.event_classifier import EventClassifier, ClassifierConfig
from spatial_hud.models import DistanceBucket, FeaturePacket


def make_packet(
    energy: float,
    onset: float,
    centroid: float,
    bands: list[float],
    azimuth: float = 0.0,
) -> FeaturePacket:
    return FeaturePacket(
        timestamp=0.0,
        azimuth_deg=azimuth,
        energy=energy,
        band_energies=bands,
        onset_strength=onset,
        spectral_centroid=centroid,
    )


def test_footstep_detection():
    classifier = EventClassifier()
    packet = make_packet(
        energy=0.07,
        onset=0.09,
        centroid=1200.0,
        bands=[0.01] * 32,
    )
    event = classifier.classify(packet)
    assert event.kind == "footstep"
    assert event.distance_bucket is DistanceBucket.MID


def test_vehicle_detection():
    classifier = EventClassifier()
    bands = [0.01] * 32
    for idx in range(2, 6):
        bands[idx] = 0.08
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
        bands=[0.05] * 32,
    )
    event = classifier.classify(packet)
    assert event.kind == "gunfire"
    assert event.distance_bucket is DistanceBucket.NEAR
