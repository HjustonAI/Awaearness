from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from .models import DistanceBucket, Event, FeaturePacket


logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    # Footstep heuristics
    footstep_onset_threshold: float = 0.05
    footstep_mid_band_min: float = 0.015
    footstep_high_mid_ratio: float = 1.15
    footstep_energy_min: float = 0.04
    footstep_onset_ratio: float = 1.8
    footstep_mid_ratio: float = 1.4
    footstep_energy_ratio: float = 1.5
    footstep_score_threshold: float = 0.55

    # Gunfire heuristics
    gunfire_energy_threshold: float = 0.12
    gunfire_centroid_threshold: float = 3200.0
    gunfire_high_band_threshold: float = 0.06
    gunfire_energy_ratio: float = 2.0
    gunfire_high_ratio: float = 1.5
    gunfire_onset_ratio: float = 1.8
    gunfire_score_threshold: float = 0.55

    # Vehicle heuristics
    vehicle_low_band_threshold: float = 0.045
    vehicle_low_mid_ratio: float = 1.25
    vehicle_flatness_max: float = 0.65
    vehicle_low_dynamic_ratio: float = 1.4
    vehicle_score_threshold: float = 0.6

    # Ambient / adaptive thresholds
    ambient_energy_floor: float = 0.012
    dynamic_energy_multiplier: float = 1.5
    ambient_decay_alpha: float = 0.05
    ambient_update_margin: float = 1.25
    front_back_positive_threshold: float = 0.15
    front_back_negative_threshold: float = -0.15
    front_back_ambiguous_scale: float = 0.8

    # Distance estimation
    near_energy: float = 0.08
    mid_energy: float = 0.04
    distance_near_ratio: float = 2.2
    distance_mid_ratio: float = 1.4


@dataclass
class AmbientTracker:
    alpha: float
    energy: float | None = None
    onset: float | None = None
    low: float | None = None
    mid: float | None = None
    high: float | None = None

    def observe(self, feature: FeaturePacket) -> None:
        values = {
            "energy": feature.energy,
            "onset": feature.onset_strength,
            "low": feature.low_band_energy,
            "mid": feature.mid_band_energy,
            "high": feature.high_band_energy,
        }
        for attr, new_value in values.items():
            current = getattr(self, attr)
            if current is None:
                setattr(self, attr, new_value)
            else:
                setattr(self, attr, (1 - self.alpha) * current + self.alpha * new_value)

    def baseline(self, attr: str, fallback: float) -> float:
        value = getattr(self, attr)
        return fallback if value is None else value


class EventClassifier:
    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self._ambient = AmbientTracker(alpha=self.config.ambient_decay_alpha)

    def classify_distance(self, energy: float, baseline_energy: float) -> DistanceBucket:
        near_threshold = max(self.config.near_energy, baseline_energy * self.config.distance_near_ratio)
        mid_threshold = max(self.config.mid_energy, baseline_energy * self.config.distance_mid_ratio)
        if energy >= near_threshold:
            return DistanceBucket.NEAR
        if energy >= mid_threshold:
            return DistanceBucket.MID
        return DistanceBucket.FAR

    def classify(self, feature: FeaturePacket) -> Event:
        cfg = self.config
        eps = 1e-6

        baseline_energy = self._ambient.baseline("energy", cfg.ambient_energy_floor)
        baseline_onset = self._ambient.baseline("onset", cfg.footstep_onset_threshold * 0.5)
        baseline_mid = self._ambient.baseline("mid", cfg.footstep_mid_band_min * 0.5)
        baseline_low = self._ambient.baseline("low", cfg.vehicle_low_band_threshold * 0.5)
        baseline_high = self._ambient.baseline("high", cfg.gunfire_high_band_threshold * 0.5)

        dynamic_energy_floor = max(cfg.ambient_energy_floor, baseline_energy * cfg.dynamic_energy_multiplier)
        high_mid_ratio = feature.high_band_energy / (feature.mid_band_energy + eps)
        low_mid_ratio = feature.low_band_energy / (feature.mid_band_energy + eps)

        footstep_components = [
            (feature.energy >= max(cfg.footstep_energy_min, baseline_energy * cfg.footstep_energy_ratio), 0.2),
            (feature.onset_strength >= max(cfg.footstep_onset_threshold, baseline_onset * cfg.footstep_onset_ratio), 0.35),
            (feature.mid_band_energy >= max(cfg.footstep_mid_band_min, baseline_mid * cfg.footstep_mid_ratio), 0.3),
            (high_mid_ratio >= cfg.footstep_high_mid_ratio, 0.15),
        ]
        footstep_score = sum(weight for condition, weight in footstep_components if condition)

        vehicle_components = [
            (feature.low_band_energy >= max(cfg.vehicle_low_band_threshold, baseline_low * cfg.vehicle_low_dynamic_ratio), 0.45),
            (low_mid_ratio >= cfg.vehicle_low_mid_ratio, 0.25),
            (feature.spectral_flatness <= cfg.vehicle_flatness_max, 0.2),
            (feature.energy >= dynamic_energy_floor, 0.1),
        ]
        vehicle_score = sum(weight for condition, weight in vehicle_components if condition)

        gunfire_components = [
            (feature.energy >= max(cfg.gunfire_energy_threshold, baseline_energy * cfg.gunfire_energy_ratio), 0.4),
            (feature.spectral_centroid >= cfg.gunfire_centroid_threshold, 0.3),
            (feature.high_band_energy >= max(cfg.gunfire_high_band_threshold, baseline_high * cfg.gunfire_high_ratio), 0.2),
            (feature.onset_strength >= max(cfg.footstep_onset_threshold, baseline_onset * cfg.gunfire_onset_ratio), 0.1),
        ]
        gunfire_score = sum(weight for condition, weight in gunfire_components if condition)

        scores = {
            "footstep": (footstep_score, cfg.footstep_score_threshold),
            "vehicle": (vehicle_score, cfg.vehicle_score_threshold),
            "gunfire": (gunfire_score, cfg.gunfire_score_threshold),
        }

        best_kind, (best_score, threshold) = max(scores.items(), key=lambda item: item[1][0])
        if best_score >= threshold:
            kind = best_kind
            confidence = min(1.0, best_score)
        else:
            kind = "ambient"
            confidence = 0.0

        if feature.front_back_score <= cfg.front_back_negative_threshold:
            orientation = "back"
        elif feature.front_back_score >= cfg.front_back_positive_threshold:
            orientation = "front"
        else:
            orientation = "ambiguous"

        distance = self.classify_distance(feature.energy, baseline_energy)

        adjusted_azimuth = _project_azimuth(feature.azimuth_deg, orientation)

        logger.debug(
            "Classified event: kind=%s score=%.2f energy=%.3f centroid=%.1f onset=%.3f low=%.3f mid=%.3f high=%.3f flat=%.2f ratio_high_mid=%.2f ratio_low_mid=%.2f fb_score=%.2f orientation=%s distance=%s",
            kind,
            best_score,
            feature.energy,
            feature.spectral_centroid,
            feature.onset_strength,
            feature.low_band_energy,
            feature.mid_band_energy,
            feature.high_band_energy,
            feature.spectral_flatness,
            high_mid_ratio,
            low_mid_ratio,
            feature.front_back_score,
            orientation,
            distance.value,
        )

        if kind == "ambient" or feature.energy <= cfg.ambient_energy_floor * cfg.ambient_update_margin:
            self._ambient.observe(feature)

        if orientation == "ambiguous" and kind != "ambient":
            confidence *= cfg.front_back_ambiguous_scale

        return Event(
            kind=kind,
            azimuth_deg=adjusted_azimuth,
            distance_bucket=distance,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            orientation=orientation,
            raw_azimuth_deg=feature.azimuth_deg,
        )

    def stream(self, features: Iterable[FeaturePacket]) -> Iterator[Event]:
        for feat in features:
            event = self.classify(feat)
            if event.kind == "ambient":
                continue
            yield event


def _project_azimuth(raw_angle: float, orientation: str) -> float:
    angle_rad = math.radians(raw_angle)
    if orientation == "back":
        front_sign = -1.0
    else:
        front_sign = 1.0
    x = math.sin(angle_rad)
    y = math.cos(angle_rad) * front_sign
    adjusted = math.degrees(math.atan2(x, y))
    if adjusted < 0:
        adjusted += 360.0
    return adjusted
