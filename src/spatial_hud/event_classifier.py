from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from .models import DistanceBucket, Event, FeaturePacket


@dataclass
class ClassifierConfig:
    footstep_onset_threshold: float = 0.05
    footstep_energy_threshold: float = 0.02
    gunfire_energy_threshold: float = 0.12
    gunfire_centroid_threshold: float = 3000.0
    vehicle_band_energy_threshold: float = 0.04
    vehicle_low_band_index: int = 2
    vehicle_high_band_index: int = 5

    near_energy: float = 0.08
    mid_energy: float = 0.04


class EventClassifier:
    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()

    def classify_distance(self, energy: float) -> DistanceBucket:
        if energy >= self.config.near_energy:
            return DistanceBucket.NEAR
        if energy >= self.config.mid_energy:
            return DistanceBucket.MID
        return DistanceBucket.FAR

    def classify(self, feature: FeaturePacket) -> Event:
        band_arr = np.array(feature.band_energies, dtype=float)
        low_band_power = float(
            band_arr[self.config.vehicle_low_band_index : self.config.vehicle_high_band_index + 1].mean()
        )

        if feature.energy >= self.config.gunfire_energy_threshold and feature.spectral_centroid >= self.config.gunfire_centroid_threshold:
            kind = "gunfire"
            confidence = 0.75 + min(0.2, (feature.spectral_centroid - self.config.gunfire_centroid_threshold) / 10)
        elif low_band_power >= self.config.vehicle_band_energy_threshold:
            kind = "vehicle"
            confidence = 0.5 + min(0.3, low_band_power)
        elif feature.onset_strength >= self.config.footstep_onset_threshold and feature.energy >= self.config.footstep_energy_threshold:
            kind = "footstep"
            confidence = 0.6 + min(0.35, feature.energy * 2)
        else:
            kind = "ambient"
            confidence = 0.3

        distance = self.classify_distance(feature.energy)

        return Event(
            kind=kind,
            azimuth_deg=feature.azimuth_deg,
            distance_bucket=distance,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    def stream(self, features: Iterable[FeaturePacket]) -> Iterator[Event]:
        for feat in features:
            event = self.classify(feat)
            if event.kind == "ambient":
                continue
            yield event
