# Spatial Audio Assist Architecture

## Objectives
- Provide real-time directional audio cues for players with single-sided hearing while gaming on Windows PCs.
- Operate without modifying or injecting code into the game; rely entirely on desktop audio capture and a layered HUD.
- Keep latency under 100 ms end-to-end to preserve competitive viability.
- Allow extensibility for new event types, output modalities (e.g., haptics), and ML model upgrades.

## System Overview
```
┌─────────────┐    WASAPI loopback    ┌────────────────┐    feature vectors    ┌─────────────────┐
│ Audio Mixer │ ─────────────────────▶ │ Capture Service │ ─────────────────────▶ │ Signal Processor │
└─────────────┘                        └────────────────┘                        └─────────────────┘
                                                                                          │
                                                                                 classified events
                                                                                          ▼
                                                                                ┌────────────────┐
                                                                                │  Event Queue   │
                                                                                └────────────────┘
                                                                                          │
                                                                         HUD updates / telemetry
                                                                                          ▼
                                                                                ┌────────────────┐
                                                                                │ Visual Overlay │
                                                                                └────────────────┘
```

## Module Breakdown

### 1. Audio Capture Service
- **Role**: Tap the system playback mix via WASAPI loopback on Windows.
- **Tech**: `sounddevice` (PortAudio) or `pyaudio` for prototyping; a future C++/Rust loopback capture for low-latency release.
- **Output**: Ring buffer of multichannel frames (float32 PCM) at 48 kHz.
- **Considerations**: Channel layout detection (stereo vs. 5.1/7.1), buffer underrun monitoring, optional down-mix to stereo for processing.

### 2. Signal Processor
- **Pipeline**:
  1. Short-time Fourier transform (STFT) on overlapping windows (e.g., 1024 samples, 50% overlap).
  2. Inter-channel level difference (ILD) and phase difference (IPD) extraction between left/right or surround pairs.
  3. Beamforming-style energy estimation across azimuth bins via simple generalized cross-correlation with phase transform (GCC-PHAT).
  4. Temporal smoothing using exponential moving averages to stabilize direction estimates.
- **Outputs**: Direction hypotheses (front-left/right/back), broadband energy, spectral centroid, transient metrics.

### 3. Event Classifier
- **Approach**: Hybrid heuristic + lightweight ML.
  - Rule-based trigger on envelope/transient profiles for footsteps.
  - Lower-frequency energy with sustained duration for vehicles.
  - Broadband, high-amplitude bursts for gunfire.
  - Optional shallow classifier (e.g., scikit-learn RandomForest) trained offline on extracted features.
- **Interface**: Accepts feature vectors, returns `Event(type, azimuth_deg, confidence, distance_bucket, timestamp)`.

### 4. Event Queue & Smoothing
- **Purpose**: Debounce classifier outputs, merge duplicates, maintain rolling state for HUD.
- **Implementation**: Thread-safe queue with time-based decay (e.g., remove events older than 1.5 s).

### 5. Visual Overlay (HUD)
- **Prototype**: Python `pygame` or `pyqtgraph` window in borderless, click-through mode using Win32 APIs for transparency.
- **Production**: C++/C# Direct2D overlay or integration with tools like Overwolf.
- **Features**: Compass ring, event icons, color-coded distance (near/mid/far), text labels, optional minimap-style radial pulses.

## Data Contracts
- `AudioFrame`: ndarray shape `(num_channels, frame_size)`.
- `FeaturePacket`: dataclass containing timestamp, azimuth (deg), energy, band energies, onset metrics.
- `Event`: dataclass {`kind`, `azimuth_deg`, `distance_bucket` ("near" | "mid" | "far"), `confidence` in [0,1], `ttl_ms`}.
- `HudState`: list of active `Event`s + global compass orientation.

## Latency Budget
- Capture buffer: 20 ms
- STFT + feature extraction: 10 ms (vectorized NumPy)
- Classification + smoothing: 5 ms
- HUD render loop: 16 ms (60 FPS)
- **Total target**: ≈51 ms (headroom to stay <100 ms).

## Threading Model
- Capture thread writes into lock-free ring buffer.
- Processing thread consumes frames, produces events.
- HUD thread polls event queue at 60 Hz, interpolates positions.
- Inter-thread communication via `queue.Queue` or `asyncio` channels.

## Extensibility Hooks
- Plug-in interface for new classifiers, using entry points or config-driven pipeline definitions.
- Telemetry publisher (WebSocket/OSC) for external devices (e.g., haptic vest).
- Config file for sensitivity thresholds, color palette, hotkeys.

## Risks & Mitigations
- **Anti-cheat**: Use non-intrusive overlay APIs, avoid code injection; consider whitelisting or partnership for popular titles.
- **Performance**: Provide GPU-accelerated path (PyTorch, ONNX Runtime) if ML models grow; allow CPU-only fallback.
- **False positives**: Implement per-class hysteresis, allow user calibration sessions.
- **User comfort**: Provide colorblind-safe palettes, adjustable opacity, optional audio cue mirroring.
