"""
Test optimized HRTF processing.
"""

import time
import numpy as np
import pytest

from spatial_hud.hrtf_processing import (
    OptimizedHRTFProcessor,
    StreamingBandpassBank,
    HRTF_BANDS,
    feature_stream,
)
from spatial_hud.models import FeaturePacket


def create_test_stereo(
    samplerate: int,
    duration: float,
    frequency: float,
    azimuth_deg: float,
) -> np.ndarray:
    """Create synthetic stereo test signal with ITD/ILD."""
    n_samples = int(samplerate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    
    # Base tone with harmonics
    tone = np.sin(2 * np.pi * frequency * t) * 0.5
    tone += np.sin(2 * np.pi * frequency * 2 * t) * 0.2
    
    # ITD
    max_itd = int(0.0007 * samplerate)
    itd = int(max_itd * np.sin(np.radians(azimuth_deg)))
    
    # ILD
    ild_db = 10 * np.sin(np.radians(azimuth_deg))
    ild_ratio = 10 ** (ild_db / 20)
    
    if azimuth_deg >= 0:
        left = np.roll(tone, abs(itd)) / ild_ratio
        right = tone * ild_ratio
    else:
        left = tone * (1 / ild_ratio)
        right = np.roll(tone, abs(itd)) * ild_ratio
    
    return np.stack([left, right], axis=1)


class TestOptimizedProcessor:
    """Tests for OptimizedHRTFProcessor."""
    
    def test_process_returns_feature_packet(self):
        proc = OptimizedHRTFProcessor(samplerate=48000)
        frame = create_test_stereo(48000, 0.02, 1000, 30)
        
        result = proc.process(frame)
        
        assert isinstance(result, FeaturePacket)
        assert result.timestamp > 0
        assert len(result.band_energies) == len(HRTF_BANDS)
    
    def test_direction_estimation_right(self):
        proc = OptimizedHRTFProcessor(samplerate=48000)
        frame = create_test_stereo(48000, 0.05, 1000, 45)
        
        # Run multiple times to stabilize
        for _ in range(5):
            result = proc.process(frame)
        
        # Should detect sound from right (positive angle)
        assert result.azimuth_deg > 0
    
    def test_direction_estimation_left(self):
        proc = OptimizedHRTFProcessor(samplerate=48000)
        frame = create_test_stereo(48000, 0.05, 1000, -45)
        
        for _ in range(5):
            result = proc.process(frame)
        
        # Should detect sound from left (negative angle)
        assert result.azimuth_deg < 0
    
    def test_direction_estimation_center(self):
        proc = OptimizedHRTFProcessor(samplerate=48000)
        frame = create_test_stereo(48000, 0.05, 1000, 0)
        
        for _ in range(5):
            result = proc.process(frame)
        
        # Should be near center
        assert abs(result.azimuth_deg) < 15
    
    def test_silent_frame_handling(self):
        proc = OptimizedHRTFProcessor(samplerate=48000)
        frame = np.zeros((1024, 2), dtype=np.float32)
        
        result = proc.process(frame)
        
        assert result.direction_confidence == 0.0
        assert result.energy < 0.001
    
    def test_mono_input_handling(self):
        proc = OptimizedHRTFProcessor(samplerate=48000)
        frame = np.sin(np.linspace(0, 100, 1024)).astype(np.float32)
        
        result = proc.process(frame)
        
        assert isinstance(result, FeaturePacket)
    
    def test_reset_clears_state(self):
        proc = OptimizedHRTFProcessor(samplerate=48000)
        frame = create_test_stereo(48000, 0.02, 1000, 45)
        
        # Process some frames
        for _ in range(10):
            proc.process(frame)
        
        # Reset
        proc.reset()
        
        assert proc._smoothed_angle == 0.0
        assert proc._smoothed_fb_score == 0.0


class TestStreamingBandpassBank:
    """Tests for streaming filter bank."""
    
    def test_filter_stereo_returns_correct_bands(self):
        bank = StreamingBandpassBank(48000, HRTF_BANDS)
        left = np.random.randn(1024).astype(np.float32)
        right = np.random.randn(1024).astype(np.float32)
        
        results = bank.filter_stereo(left, right)
        
        assert len(results) == len(HRTF_BANDS)
        for left_filt, right_filt in results:
            assert left_filt.shape == left.shape
            assert right_filt.shape == right.shape
    
    def test_filter_maintains_state(self):
        bank = StreamingBandpassBank(48000, HRTF_BANDS)
        chunk1 = np.random.randn(1024).astype(np.float32)
        chunk2 = np.random.randn(1024).astype(np.float32)
        
        # Process two consecutive chunks
        bank.filter_stereo(chunk1, chunk1)
        result2 = bank.filter_stereo(chunk2, chunk2)
        
        # State should be maintained
        assert bank._zi_left[0] is not None
    
    def test_reset_clears_state(self):
        bank = StreamingBandpassBank(48000, HRTF_BANDS)
        left = np.random.randn(1024).astype(np.float32)
        
        bank.filter_stereo(left, left)
        bank.reset()
        
        assert bank._zi_left[0] is None


class TestFeatureStream:
    """Tests for feature_stream generator."""
    
    def test_yields_feature_packets(self):
        frames = [
            create_test_stereo(48000, 0.02, 1000, 30)
            for _ in range(5)
        ]
        
        results = list(feature_stream(frames, 48000))
        
        assert len(results) == 5
        assert all(isinstance(r, FeaturePacket) for r in results)


class TestPerformance:
    """Performance benchmarks."""
    
    def test_processing_speed(self):
        """Verify processing is fast enough for real-time."""
        proc = OptimizedHRTFProcessor(samplerate=48000, blocksize=1024)
        frame = create_test_stereo(48000, 1024/48000, 1000, 30)
        
        # Warmup
        for _ in range(10):
            proc.process(frame)
        
        # Benchmark
        n_iterations = 100
        start = time.perf_counter()
        
        for _ in range(n_iterations):
            proc.process(frame)
        
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / n_iterations) * 1000
        
        # At 48kHz with 1024 samples, we have ~21ms per frame
        # Processing should be well under that
        print(f"\nAverage processing time: {avg_ms:.2f} ms per frame")
        print(f"Real-time budget: ~21 ms")
        print(f"CPU usage: {avg_ms/21*100:.1f}%")
        
        assert avg_ms < 15, f"Processing too slow: {avg_ms:.2f} ms > 15 ms target"
