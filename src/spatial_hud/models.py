from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class DistanceBucket(str, Enum):
    NEAR = "near"
    MID = "mid"
    FAR = "far"


@dataclass(slots=True)
class FeaturePacket:
    """Unified feature packet for HRTF audio analysis."""
    timestamp: float
    azimuth_deg: float
    energy: float
    band_energies: list[float]
    onset_strength: float
    spectral_centroid: float
    low_band_energy: float
    mid_band_energy: float
    high_band_energy: float
    spectral_flatness: float
    direction_confidence: float = 0.0
    front_back_score: float = 0.0
    # Extended HRTF features (optional, for ML)
    spectral_spread: float = 0.0
    spectral_rolloff: float = 0.0
    pinna_notch_ratio: float = 0.0
    high_freq_rolloff: float = 0.0
    interaural_coherence: float = 0.0


@dataclass
class Event:
    kind: Literal["footstep", "vehicle", "gunfire", "ambient"]
    azimuth_deg: float
    distance_bucket: DistanceBucket
    confidence: float
    ttl_ms: int = 1200
    orientation: Literal["front", "back", "ambiguous"] = "front"
    raw_azimuth_deg: float = 0.0


@dataclass
class HudState:
    events: list[Event]
    compass_offset_deg: float = 0.0
