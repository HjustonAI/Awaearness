# Spatial HUD Prototype

Assistive heads-up display that visualizes directional game audio for players with single-sided hearing. The system taps the Windows system audio mix, estimates coarse azimuth and distance of salient sound events (footsteps, vehicles, gunfire), and renders a lightweight overlay HUD with clear icons.

## Features
- **Loopback capture** using WASAPI to ingest the live stereo mix without modifying games.
- **Signal processing** for interaural level/phase differences and GCC-PHAT direction estimates.
- **Heuristic classifier** to tag footsteps, vehicles, and gunfire with confidence and distance buckets.
- **Visual HUD** implemented with `pygame`, creating a transparent, click-through compass overlay.
- **Mock simulator** that generates synthetic feature packets for development without live audio.

## Getting Started

### Prerequisites
- Windows 10/11 PC with Python 3.10+
- Microsoft Visual C++ Build Tools (required by `pyaudio`)
- Stereo or surround sound playback device enabled
- [`soundcard`](https://github.com/bastibe/SoundCard) Python package for WASAPI loopback capture (installed automatically via `pip install -e .[dev]`)

### Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]
```

### Running the HUD with Mock Data
```bash
python -m spatial_hud.main --mock
```
Press `Ctrl+C` in the terminal to stop the demo. The HUD displays synthetic events sweeping around the compass so you can validate rendering without launching a game.

### Running with Live Game Audio
```bash
python -m spatial_hud.main
```
This starts the loopback capture pipeline. Launch your game and ensure Windows is outputting stereo/5.1. You should see directional markers appear as footsteps, weapon fire, and vehicles are detected. Latency and accuracy depend on the game mix; adjust classifier thresholds in `spatial_hud/event_classifier.py` if needed.
If you encounter a message about the `soundcard` package or missing loopback devices, check that the dependency installed correctly and that your default speaker exposes a loopback endpoint (enable "Stereo Mix" in Windows Sound Control Panel if available).

**HUD controls**
- Drag the compass window with the left mouse button to reposition it.
- Press `T` to toggle click-through mode once it’s in place.
- Press `Esc` to close the HUD quickly.

### Testing with PUBG
Follow the quick-start steps below, then see [`docs/testing_pubg.md`](docs/testing_pubg.md) for a full field-test checklist, calibration routine, and troubleshooting table.

1. **Activate the virtual environment** (or run via `.venv\Scripts\python.exe`) before launching the HUD to avoid `ModuleNotFoundError`.
2. **Match Windows playback to PUBG**: set the same headset/speakers as the default device, disable Dolby Atmos/DTS/Windows Sonic, and confirm a 48 kHz sample rate.
3. **Configure PUBG audio**:
   - Output mode: `Stereo`
   - Master volume: ~80% so the classifier can pick up quieter footsteps
   - Lower music/UI sliders to reduce false positives
4. **Place the HUD**: Start the HUD first, drag it into position, press `T` for click-through, then launch PUBG in `Borderless` or `Windowed` mode. `Alt+Tab` if you need to reposition mid-game.
5. **Calibrate**: In the training range, verify azimuth tracking with footsteps, gunfire, and vehicles. Tweak `ClassifierConfig` thresholds if detections are weak or noisy.
6. **Monitor latency**: Updates should trail sounds by <100 ms. Close heavy background apps or confirm the sample rate if lag is noticeable.
7. **Squad awareness**: The HUD highlights both teammates and enemies—use PUBG’s quick mute (`Ctrl+T`) during calibration to establish a baseline.

Wrap up by pressing `Ctrl+C` in the terminal (or `Esc` on the HUD) before closing PUBG.

### Tests
```bash
pytest
```

## Project Structure
```
src/spatial_hud/
  audio_capture.py    # WASAPI loopback capture wrapper
  signal_processing.py# Feature extraction + direction estimation
  event_classifier.py # Heuristic classifier for key sound types
  hud.py              # Compass overlay renderer (pygame)
  simulation.py       # Synthetic feature generator for mock mode
  main.py             # Pipeline orchestration entrypoint
```

## Calibration & Tuning
- **Sensitivity**: Adjust `ClassifierConfig` thresholds to match your audio device output. Lower thresholds capture quieter sounds but may increase false positives.
- **Direction smoothing**: Increase the FFT window size or apply exponential smoothing in `DirectionEstimator` if the HUD jitter is too high.
- **Distance buckets**: Tune `near_energy` and `mid_energy` in `ClassifierConfig` to suit different volume levels or surround mixes.
- **Opacity & colors**: Customize color palettes and font in `spatial_hud/hud.py` for accessibility preferences (e.g., colorblind-safe schemes).

## Roadmap
1. **ML Upgrade**: Replace heuristics with a lightweight neural classifier trained on labeled in-game audio datasets.
2. **Game Overlay Integration**: Port HUD to DirectX overlay or Overwolf app for anti-cheat-safe distribution.
3. **User Calibration Wizard**: Guide players through a 90-second calibration to auto-tune thresholds and head orientation.
4. **Haptics/Voice Output**: Optional haptic vest or TTS callouts for players who prefer tactile or spoken cues.

## Limitations
- Prototype accuracy is approximate and may drift with complex reverberations or dynamic range compression.
- Requires WASAPI loopback support; some USB headsets may need alternate drivers.
- Overlay is a separate window; it may not stay on top in full-screen exclusive mode without additional tooling.

## Troubleshooting
- **`ModuleNotFoundError: spatial_hud`** – Activate the local virtual environment (`.venv\Scripts\activate`) or run commands via `.venv\Scripts\python.exe` so Python resolves the editable install.
- **No HUD window appears** – Check that Windows hasn’t sent the transparent window behind PUBG; use `Alt+Tab`, or temporarily disable `WS_EX_TRANSPARENT` by commenting out the Win32 block in `hud.py` to bring it to the front.
- **Audio not detected** – Confirm the default playback device is the one PUBG uses and that loopback capture is allowed; some capture cards/headsets need “Listen to this device” disabled to expose loopback.
- **High CPU usage** – Increase `blocksize` in `PipelineConfig` to 2048 or reduce refresh rate in `HudRenderer` to 45 FPS.
- **False positives** – Raise `footstep_onset_threshold` / `vehicle_band_energy_threshold` or lower master volume slightly; run calibration in a quiet lobby to tune thresholds per device.
