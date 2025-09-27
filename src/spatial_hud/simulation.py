from __future__ import annotations

import math
import random
import time
from typing import Iterator

import numpy as np

from .models import FeaturePacket


def _synthetic_feature(
    azimuth_deg: float,
    energy: float,
    onset_strength: float,
    centroid: float,
    bands: list[float],
) -> FeaturePacket:
    return FeaturePacket(
        timestamp=time.time(),
        azimuth_deg=azimuth_deg,
        energy=energy,
        band_energies=bands,
        onset_strength=onset_strength,
        spectral_centroid=centroid,
    )


def offline_feature_stream(duration_s: float = 30.0, seed: int | None = None) -> Iterator[FeaturePacket]:
    """Yield synthetic feature packets for demo/testing."""
    rng = random.Random(seed)
    start = time.time()
    while time.time() - start < duration_s:
        mode = rng.choice(["footstep", "vehicle", "gunfire", "ambient", "ambient"])
        azimuth = rng.uniform(-90, 90)
        if mode == "footstep":
            energy = rng.uniform(0.05, 0.1)
            onset = rng.uniform(0.08, 0.2)
            centroid = rng.uniform(10, 20)
            bands = [rng.uniform(0.01, 0.03) for _ in range(32)]
        elif mode == "vehicle":
            energy = rng.uniform(0.07, 0.15)
            onset = rng.uniform(0.01, 0.04)
            centroid = rng.uniform(5, 12)
            bands = [rng.uniform(0.02, 0.06) if 2 <= i <= 5 else rng.uniform(0.005, 0.03) for i in range(32)]
        elif mode == "gunfire":
            energy = rng.uniform(0.12, 0.25)
            onset = rng.uniform(0.05, 0.11)
            centroid = rng.uniform(20, 32)
            bands = [rng.uniform(0.02, 0.08) for _ in range(32)]
        else:
            energy = rng.uniform(0.005, 0.03)
            onset = rng.uniform(0.0, 0.01)
            centroid = rng.uniform(5, 15)
            bands = [rng.uniform(0.001, 0.01) for _ in range(32)]

        yield _synthetic_feature(azimuth, energy, onset, centroid, bands)
        time.sleep(0.05)
