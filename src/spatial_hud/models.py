from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class DistanceBucket(str, Enum):
    NEAR = "near"
    MID = "mid"
    FAR = "far"


@dataclass
class FeaturePacket:
    timestamp: float
    azimuth_deg: float
    energy: float
    band_energies: list[float]
    onset_strength: float
    spectral_centroid: float


@dataclass
class Event:
    kind: Literal["footstep", "vehicle", "gunfire", "ambient"]
    azimuth_deg: float
    distance_bucket: DistanceBucket
    confidence: float
    ttl_ms: int = 1200


@dataclass
class HudState:
    events: list[Event]
    compass_offset_deg: float = 0.0
