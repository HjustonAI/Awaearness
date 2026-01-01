"""
Spatial HUD - Audio radar for hearing-impaired gamers.

Uses HRTF (Head-Related Transfer Function) analysis to decode
spatial audio from games like PUBG and display directional
information visually.
"""

from .main import Pipeline, PipelineConfig, run_pipeline
from .hrtf_processing import (
    OptimizedHRTFProcessor,
    feature_stream,
    HRTF_BANDS,
)

__all__ = [
    'Pipeline',
    'PipelineConfig', 
    'run_pipeline',
    'OptimizedHRTFProcessor',
    'feature_stream',
    'HRTF_BANDS',
]
