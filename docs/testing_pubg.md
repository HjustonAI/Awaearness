# PUBG Field Test Guide

This guide walks you through verifying the Spatial HUD against live PUBG audio on Windows. It assumes you have already completed the project installation steps from the root `README.md`.

## 1. Pre-flight Checklist

Before launching anything, make sure the following items are in place:

- **System**: Windows 10/11 with a stable GPU driver and two-channel (stereo) output.
- **Python environment**: Local virtual environment created with `python -m venv .venv` and dependencies installed via `pip install -e .[dev]`.
- **Audio device**: Same headset or speakers that PUBG will use must be set as the Windows *Default* playback device. Disable spatial audio enhancements (Dolby Atmos, DTS, Windows Sonic) to keep the stereo cues predictable.
- **Loopback endpoint**: Confirm that the device exposes a loopback stream—look for "Stereo Mix" or "<device name> (Loopback)" in the Windows Sound Control Panel (`mmsys.cpl`). If it is disabled, right-click to *Show Disabled Devices* and enable it.
- **Sample rate**: Set the device to 48 kHz in the *Advanced* tab of the playback device properties. This matches PUBG's default output and reduces resampling noise.

## 2. Launch the HUD

1. Open a terminal and activate your virtual environment:
   ```bash
   .venv\Scripts\activate
   ```
2. Start the HUD with live capture:
   ```bash
   python -m spatial_hud.main
   ```
3. When the window appears, drag it to your preferred corner and press `T` to toggle click-through so it stays interactive-free on top of PUBG. Press `Esc` any time to close the HUD.

> **Tip:** If you do not see the HUD, `Alt+Tab` to cycle through windows or temporarily comment out `set_click_through` in `hud.py` to debug positioning.

## 3. Configure PUBG

1. Launch PUBG **after** the HUD is running so the transparent window remains on top.
2. Set the video mode to **Borderless** or **Windowed**; full-screen exclusive can hide all overlays.
3. Open *Settings → Audio* and apply the following:
   - Output: `Stereo`
   - Master Volume: 75–85%
   - In-Game Voice Volume: adjust to taste (squads can mask footsteps)
   - Music/UI Volume: lower to reduce false positives
4. Optionally bind a quick mute key (`Ctrl+T`) for squad chatter to isolate footsteps during calibration.

## 4. Calibrate in the Training Range

1. Enter the PUBG Training Mode.
2. Walk in small circles while watching the compass. The azimuth marker should follow your rotation within ~10°.
3. Fire several weapon types at different distances. Gunfire icons should appear within 100 ms of the muzzle flash.
4. Drive a vehicle across the range. Vehicle markers should track direction and fade as you pass the observer.
5. If detections are inconsistent:
   - Lower `ClassifierConfig` thresholds in `spatial_hud/event_classifier.py` (e.g., `footstep_energy_threshold`, `vehicle_band_energy_threshold`).
   - Ensure Windows playback volume is not capped by hardware mixers.
   - Reduce background noise (voice chat, music).

## 5. Live Match Etiquette

- Expect the HUD to highlight both teammates and enemies; use context to differentiate.
- Brief audio spikes (grenades, UI cues) may generate short-lived markers—tune thresholds if they are distracting.
- Monitor PC performance; the overlay should consume <5% CPU. If it spikes, close other capture or audio apps.

## 6. Shutting Down

1. Exit the match or return to the lobby.
2. Press `Esc` inside the HUD window or hit `Ctrl+C` in the terminal to stop the pipeline.
3. Deactivate the virtual environment with `deactivate` if you are done testing.

## 7. Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| HUD not visible in-game | PUBG in exclusive full screen | Switch to Borderless/Windowed, re-run HUD first |
| No detections | Wrong default device or loopback disabled | Re-check Windows sound settings, enable Stereo Mix |
| High latency (>150 ms) | System under load or resampling | Close heavy apps, ensure 48 kHz sample rate |
| False positives | Thresholds too low | Raise classifier thresholds, lower Master Volume |
| HUD steals focus | Click-through not toggled | Press `T` once the HUD is positioned |

With these steps, you should be able to validate the Spatial HUD against real PUBG matches and iteratively tune thresholds for your hardware.
