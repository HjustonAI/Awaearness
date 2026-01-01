# Comprehensive Research: Audio Radar for Deaf/Hard-of-Hearing PUBG Gamers

## Executive Summary

This document provides detailed technical research for building a real-time audio visualization system that decodes HRTF-encoded stereo audio from PUBG back into directional information, enabling deaf/hard-of-hearing players to perceive spatial audio through visual cues.

---

## 1. HRTF Decoding Techniques

### 1.1 How HRTF Encodes Direction

HRTF (Head-Related Transfer Function) encodes 3D spatial information into stereo audio using three primary mechanisms:

#### **Interaural Time Difference (ITD)**
- **Definition**: The time difference between when a sound reaches the left and right ears
- **Physics**: Sound from the right arrives at the right ear before the left ear
- **Range**: Maximum ITD ≈ 0.6-0.7ms (based on head width of ~21.5cm)
- **Formula**: 
  ```
  ITD = (d/c) × sin(θ)
  where:
    d = ear distance (~21.5cm for humans)
    c = speed of sound (~343 m/s)
    θ = azimuth angle
  ```
- **Effective Range**: Works well for frequencies below 1500 Hz
- **Maximum Delay**: ~626 μs for sounds at 90° azimuth

#### **Interaural Level Difference (ILD)**
- **Definition**: The amplitude/intensity difference between ears due to head shadowing
- **Physics**: The head blocks high-frequency sounds, creating a "shadow"
- **Formula**: `ILD ≈ 1.0 + (f/1000)^0.8 × sin(θ)` (in dB)
- **Effective Range**: Works well for frequencies above 1500 Hz
- **Typical Values**: 0-20 dB depending on frequency and angle

#### **Spectral Cues (Pinna Effects)**
- **Definition**: Frequency-dependent filtering by the outer ear (pinna), head, and torso
- **Key Feature**: Different directions create unique spectral "fingerprints"
- **Primary Resonance**: HRTF typically boosts 2-5 kHz with +17 dB peak at ~2700 Hz
- **Elevation Encoding**: High-frequency notches (6-10 kHz) encode vertical angle
- **Front/Back Encoding**: High-frequency rolloff differences distinguish front from back

### 1.2 The Cone of Confusion Problem

**The fundamental limitation of HRTF-based localization:**

```
                    FRONT
                      |
                     /|\
                    / | \
                   /  |  \
           LEFT  /   |   \  RIGHT
                 \   |   /
                  \  |  /
                   \ | /
                    \|/
                     |
                    BACK

All points on a cone sharing the same ITD and ILD
produce identical binaural cues!
```

**Key Points:**
- ITD and ILD are identical for all points on a cone centered on the interaural axis
- This creates **front/back ambiguity** and **elevation ambiguity**
- The cone passes through the ears and extends in both directions
- **Resolution Methods:**
  1. Head movement (not available in recorded audio)
  2. Spectral cue analysis (pinna filtering)
  3. Prior knowledge of expected sound locations
  4. Monaural spectral analysis

### 1.3 Reverse-Engineering Direction from HRTF Stereo

**Critical Insight**: Games like PUBG use **generic HRTF** (not personalized), which actually helps our cause because:
1. The HRTF is consistent across all players
2. We can potentially characterize the specific HRTF used
3. Machine learning can learn the inverse mapping

#### **Approach 1: Dual-Cue Extraction (ITD + ILD)**

```python
# Your current implementation in DirectionEstimator
class DirectionEstimator:
    """
    Combines GCC-PHAT for ITD and RMS ratio for ILD
    """
    def estimate(self, left, right):
        # GCC-PHAT for time delay
        tau = gcc_phat(left, right)  # Returns delay in samples
        theta_itd = arcsin(tau / max_tau) * 90  # Map to degrees
        
        # ILD from RMS ratio
        ild_db = 20 * log10(rms_left / rms_right)
        theta_ild = -(ild_db / ild_max_db) * 90
        
        # Blend both cues
        theta = (1 - blend) * theta_itd + blend * theta_ild
        return theta
```

**Limitations:**
- Only resolves left/right, not front/back
- Confidence degrades with low-energy signals
- Doesn't utilize spectral cues

#### **Approach 2: Spectral Cue Analysis for Front/Back**

```python
# Your FrontBackDisambiguator approach
def estimate_front_back(low_band, mid_band, high_band, interaural_corr):
    """
    Front sounds: Higher high/low ratio (more high frequencies)
    Back sounds: Lower high/low ratio (more low frequencies due to head shadow)
    """
    ratio = (high_band + eps) / (low_band + eps)
    deviation = ratio - baseline_ratio
    ratio_score = tanh(deviation * gain)
    
    # Interaural correlation:
    # High correlation = sound from front/back (symmetric)
    # Low correlation = sound from side (asymmetric reflections)
    
    return weighted_combination(ratio_score, interaural_corr)
```

### 1.4 Academic Papers & Algorithms

#### **Key Papers:**

1. **"Binaural Source Localization"** - Dietz et al., 2011
   - Introduces the **diotic-dichotic** framework
   - Models ITD extraction in frequency bands

2. **"A Probabilistic Model for Binaural Sound Localization"** - May et al., 2011
   - Bayesian framework for combining ITD/ILD cues
   - Handles the cone of confusion with priors

3. **"Deep Learning for Binaural Sound Source Localization"** - Adavanne et al., 2018
   - CNN-based approach directly on binaural spectrograms
   - Achieves 11° MAE on synthetic data

4. **"Sound Event Localization and Detection using CRNN"** - Adavanne et al., 2018 (DCASE)
   - SELDnet architecture for joint localization + classification
   - Works with binaural/Ambisonics input

5. **CIPIC HRTF Database** - UC Davis
   - 45 subjects × 1250 directions
   - Standard reference for HRTF research
   - URL: https://www.ece.ucdavis.edu/cipic/

---

## 2. Audio Localization Algorithms

### 2.1 GCC-PHAT Limitations with HRTF Audio

**Your current implementation uses GCC-PHAT (Generalized Cross-Correlation with Phase Transform):**

```python
# Standard GCC-PHAT
cross_power = left_fft * conj(right_fft)
cross_power /= abs(cross_power) + eps  # Phase transform (whitening)
corr = irfft(cross_power)
tau = argmax(corr)
```

**Limitations for HRTF-encoded audio:**

| Issue | Description | Impact |
|-------|-------------|--------|
| **Frequency-dependent ITD** | HRTF applies different delays at different frequencies | GCC-PHAT assumes uniform delay |
| **Phase wrapping** | High frequencies have ambiguous phase | Multiple peaks in correlation |
| **Wideband nature** | Game sounds are broadband | Different ITDs in different bands |
| **Reverberation** | In-game reverb creates false peaks | Reduces localization accuracy |
| **Low SNR** | Distant sounds are quiet | Noise dominates correlation |

### 2.2 Better Alternatives for Binaural Audio

#### **Alternative 1: Sub-band GCC-PHAT**

```python
def subband_gcc_phat(left, right, num_bands=8):
    """
    Compute GCC-PHAT in frequency sub-bands and combine
    """
    angles = []
    confidences = []
    
    for band in split_into_bands(left, right, num_bands):
        angle, conf = gcc_phat_single_band(*band)
        angles.append(angle)
        confidences.append(conf)
    
    # Weight by confidence and band reliability
    # Low bands (< 1500 Hz) more reliable for ITD
    return weighted_circular_mean(angles, confidences)
```

#### **Alternative 2: ILD-Weighted Localization**

```python
def ild_weighted_localization(left, right):
    """
    For high frequencies, use ILD directly
    For low frequencies, use ITD from GCC-PHAT
    """
    spectrum_left = stft(left)
    spectrum_right = stft(right)
    
    # Per-frequency ILD
    ild = 20 * log10(abs(spectrum_left) / abs(spectrum_right))
    
    # Per-frequency ITD from phase difference
    phase_diff = angle(spectrum_left) - angle(spectrum_right)
    itd = phase_diff / (2 * pi * frequencies)
    
    # Combine: ITD below 1.5kHz, ILD above
    crossover = 1500  # Hz
    angles_low = itd_to_angle(itd[freq < crossover])
    angles_high = ild_to_angle(ild[freq >= crossover])
    
    return combine_estimates(angles_low, angles_high)
```

#### **Alternative 3: Learned Binaural Features (Neural Network)**

```python
class BiauralLocalizer(nn.Module):
    """
    CNN that learns to map binaural spectrograms to direction
    """
    def __init__(self):
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),  # 2 channels: L and R
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * H * W, 256),
            nn.ReLU(),
            nn.Linear(256, 360)  # 360 degrees output
        )
    
    def forward(self, spec_left, spec_right):
        x = torch.stack([spec_left, spec_right], dim=1)
        features = self.conv_layers(x)
        return self.fc(features.flatten(1))
```

### 2.3 Solving Front/Back Disambiguation

#### **Method 1: Spectral Tilt Analysis (Your Current Approach)**

```python
# Front sounds have more high-frequency energy (direct path)
# Back sounds have more low-frequency energy (head shadow)
ratio = high_band / low_band
if ratio > baseline * 1.2:
    orientation = "front"
elif ratio < baseline * 0.8:
    orientation = "back"
else:
    orientation = "ambiguous"
```

**Improvements:**
```python
def improved_front_back(left, right, samplerate):
    """
    Use multiple spectral features for disambiguation
    """
    # 1. High-frequency rolloff difference
    left_rolloff = spectral_rolloff(left, samplerate)
    right_rolloff = spectral_rolloff(right, samplerate)
    
    # 2. Spectral centroid in 4-8kHz band
    # Front sounds: higher centroid in this band
    left_centroid = band_centroid(left, 4000, 8000)
    right_centroid = band_centroid(right, 4000, 8000)
    
    # 3. Pinna notch detection (~8kHz for front sources)
    left_notch = detect_notch(left, 7000, 9000)
    right_notch = detect_notch(right, 7000, 9000)
    
    # Combine features
    features = [left_rolloff, right_rolloff, 
                left_centroid, right_centroid,
                left_notch, right_notch]
    
    return classifier.predict(features)  # ML classifier
```

#### **Method 2: Interaural Coherence**

```python
def interaural_coherence(left, right):
    """
    Front/back have different coherence patterns
    - Front: High coherence, clear peaks
    - Back: Lower coherence, smeared patterns
    """
    # Compute coherence in frequency bands
    coherence = mscohere(left, right, fs=samplerate, nperseg=512)
    
    # Back sources have lower coherence in 2-6kHz band
    mid_coherence = np.mean(coherence[(freq > 2000) & (freq < 6000)])
    
    return mid_coherence  # High = front, Low = back
```

#### **Method 3: HRTF Template Matching**

```python
def hrtf_template_match(left, right, hrtf_database):
    """
    Compare observed binaural spectrum to HRTF templates
    """
    # Compute binaural transfer function
    observed_btf = stft(left) / (stft(right) + eps)
    
    best_match = None
    best_score = -inf
    
    for direction, hrtf in hrtf_database.items():
        # Template is ratio of left/right HRTF
        template_btf = hrtf['left'] / hrtf['right']
        
        # Compare using spectral correlation
        score = spectral_correlation(observed_btf, template_btf)
        
        if score > best_score:
            best_score = score
            best_match = direction
    
    return best_match  # Contains azimuth AND front/back info
```

### 2.4 Real-Time Algorithm Recommendations

For your **<100ms latency requirement**, here's a pipeline:

```
Input Audio (48kHz, 20ms chunks = 960 samples)
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Pre-processing (2ms)                     │
│ - DC removal, normalization             │
│ - Overlap-add with previous chunk       │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Parallel Feature Extraction (8ms)        │
│ ┌─────────────┐ ┌─────────────────────┐ │
│ │ GCC-PHAT    │ │ Band Energies       │ │
│ │ (ITD)       │ │ (ILD, spectral)     │ │
│ └─────────────┘ └─────────────────────┘ │
│ ┌─────────────┐ ┌─────────────────────┐ │
│ │ Onset       │ │ Interaural          │ │
│ │ Detection   │ │ Coherence           │ │
│ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Direction Estimation (3ms)               │
│ - Combine ITD + ILD                     │
│ - Front/back from spectral cues         │
│ - Temporal smoothing                    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ Classification (5ms)                     │
│ - Footstep / Gunfire / Vehicle          │
│ - Confidence scoring                    │
│ - Distance estimation                   │
└─────────────────────────────────────────┘
                    │
                    ▼
        HUD Update (next frame)
        
Total: ~18ms processing + 20ms buffer = 38ms latency
```

---

## 3. Game Audio Classification

### 3.1 Spectral Signatures of PUBG Sounds

#### **Footsteps**
```
Frequency Range: 200-2000 Hz (dominant)
Pattern: Rhythmic, 0.3-0.5s intervals (walking), 0.2-0.3s (running)
Spectral Features:
  - Mid-band emphasis (500-1500 Hz)
  - Sharp onset, quick decay
  - Spectral centroid: 800-1200 Hz
  - High-mid ratio: 1.0-1.5
Surface Variations:
  - Grass: More low-frequency, muffled
  - Wood: Pronounced mids, "clicky"
  - Concrete: Brighter, more highs
  - Water: Splashy, noise-like components
```

#### **Gunshots**
```
Frequency Range: 100-10000 Hz (broadband)
Pattern: Impulsive, instantaneous onset
Spectral Features:
  - Very high onset strength
  - High energy across all bands
  - Spectral centroid: 2000-4000 Hz
  - High spectral flatness (noise-like)
Weapon Variations:
  - Pistol: Sharper, more mids
  - Rifle: Fuller spectrum, more bass
  - Sniper: Very strong low-end, "crack" at high end
  - Suppressed: Reduced highs, more "puff" sound
```

#### **Vehicles**
```
Frequency Range: 50-500 Hz (engine), 500-2000 Hz (road noise)
Pattern: Continuous, doppler shifting
Spectral Features:
  - Very strong low-band energy
  - Low spectral flatness (tonal)
  - Low spectral centroid: 200-500 Hz
  - Low/mid ratio: 2.0-5.0
Vehicle Types:
  - Car: 80-150 Hz fundamental
  - Motorcycle: 100-200 Hz, more variation
  - Boat: Very low, 50-100 Hz
  - UAZ/Truck: 60-120 Hz, heavier
```

### 3.2 Feature Extraction for Classification

```python
def extract_classification_features(frame, samplerate):
    """
    Extract features optimized for footstep/gunfire/vehicle classification
    """
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), 1/samplerate)
    
    features = {
        # Energy features
        'energy': np.mean(frame**2),
        'peak_energy': np.max(frame**2),
        
        # Band energies (Hz ranges)
        'sub_bass': band_energy(spectrum, freqs, 20, 100),      # Vehicles
        'bass': band_energy(spectrum, freqs, 100, 300),         # Vehicles, explosions
        'low_mid': band_energy(spectrum, freqs, 300, 800),      # Footsteps
        'mid': band_energy(spectrum, freqs, 800, 2000),         # Footsteps, gunshots
        'high_mid': band_energy(spectrum, freqs, 2000, 5000),   # Gunshots
        'highs': band_energy(spectrum, freqs, 5000, 10000),     # Gunshots (transient)
        
        # Spectral features
        'centroid': spectral_centroid(spectrum, freqs),
        'bandwidth': spectral_bandwidth(spectrum, freqs),
        'rolloff': spectral_rolloff(spectrum, freqs, 0.85),
        'flatness': spectral_flatness(spectrum),
        'slope': spectral_slope(spectrum, freqs),
        
        # Temporal features
        'onset_strength': onset_strength(frame),
        'zero_crossing_rate': zero_crossing_rate(frame),
        'attack_time': envelope_attack_time(frame, samplerate),
        'decay_time': envelope_decay_time(frame, samplerate),
        
        # Ratios (discriminative)
        'low_mid_ratio': band_energy(spectrum, freqs, 100, 500) / 
                        (band_energy(spectrum, freqs, 500, 2000) + 1e-9),
        'high_mid_ratio': band_energy(spectrum, freqs, 2000, 8000) / 
                         (band_energy(spectrum, freqs, 500, 2000) + 1e-9),
    }
    return features
```

### 3.3 ML Approaches for Game Sound Classification

#### **Approach 1: Random Forest (Your Current Direction)**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class GameSoundClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced'
        )
    
    def train(self, features, labels):
        X = self.scaler.fit_transform(features)
        self.clf.fit(X, labels)
    
    def predict(self, features):
        X = self.scaler.transform(features)
        probs = self.clf.predict_proba(X)
        return self.clf.classes_[np.argmax(probs)], np.max(probs)
```

**Pros**: Fast inference (<1ms), interpretable, no GPU needed  
**Cons**: Limited capacity for complex patterns

#### **Approach 2: 1D CNN for Audio**

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    """
    Lightweight 1D CNN for real-time audio classification
    """
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Input: 1024 samples at 48kHz = 21ms
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=4),  # -> 241
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),  # -> 60
            
            nn.Conv1d(16, 32, kernel_size=16, stride=2),  # -> 23
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> 11
            
            nn.Conv1d(32, 64, kernel_size=4),  # -> 8
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> 1
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

**Inference Time**: ~2ms on CPU, <1ms on GPU

#### **Approach 3: Mel-Spectrogram + 2D CNN**

```python
class MelSpectrogramCNN(nn.Module):
    """
    2D CNN on mel-spectrograms for more robust classification
    """
    def __init__(self, num_classes=4, n_mels=64, time_frames=32):
        super().__init__()
        
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, waveform):
        mel = self.mel(waveform).unsqueeze(1)  # Add channel dim
        mel = torch.log(mel + 1e-9)  # Log scale
        features = self.conv(mel)
        features = features.view(features.size(0), -1)
        return self.fc(features)
```

### 3.4 Pre-trained Models and Datasets

#### **Available Audio Datasets**

| Dataset | Sounds | Size | Use Case |
|---------|--------|------|----------|
| **AudioSet** | General | 2M clips | Pre-training, transfer learning |
| **FSDKaggle2019** | Diverse | 18K clips | Environmental sounds |
| **ESC-50** | Environmental | 2K clips | Quick prototyping |
| **UrbanSound8K** | Urban | 8K clips | Outdoor sounds |
| **DCASE** | Sound events | Varies | Localization + detection |

#### **Pre-trained Models to Consider**

1. **YAMNet** (Google) - MobileNetV1 on AudioSet
   - 521 classes, lightweight
   - TensorFlow Hub / ONNX available

2. **PANNs** (Pretrained Audio Neural Networks)
   - CNN14, ResNet38, Wavegram-Logmel
   - PyTorch models available
   - Good for transfer learning

3. **OpenL3** (NYU)
   - Audio embeddings
   - 512-dim feature vectors
   - Useful for custom classifiers

#### **Creating Your Own Game Audio Dataset**

```python
"""
Strategy for collecting PUBG audio data:
1. Play game with audio recording enabled
2. Manually annotate segments
3. Extract clips with timestamps
"""

import json
from pathlib import Path

class PUBGAudioDataset:
    def __init__(self, audio_dir, annotations_file):
        self.audio_dir = Path(audio_dir)
        with open(annotations_file) as f:
            self.annotations = json.load(f)
    
    def extract_clips(self, output_dir, clip_length_ms=500):
        """
        Extract labeled clips from full recordings
        """
        for recording in self.annotations:
            audio = load_audio(self.audio_dir / recording['file'])
            
            for event in recording['events']:
                start = event['start_ms']
                end = min(event['end_ms'], start + clip_length_ms)
                clip = audio[start:end]
                
                # Save with label in filename
                label = event['type']  # footstep, gunshot, vehicle
                clip_name = f"{label}_{recording['file']}_{start}.wav"
                save_audio(clip, output_dir / label / clip_name)

# Annotation format example
annotations = {
    "file": "pubg_match_001.wav",
    "events": [
        {"start_ms": 1500, "end_ms": 1800, "type": "footstep", "direction": 45},
        {"start_ms": 3200, "end_ms": 3400, "type": "gunshot", "direction": -30},
        {"start_ms": 5000, "end_ms": 8000, "type": "vehicle", "direction": 180},
    ]
}
```

---

## 4. Existing Solutions & Research

### 4.1 Academic Research on Gaming Accessibility

#### **Key Papers**

1. **"Game Accessibility: A Survey"** - Yuan et al., 2011
   - Comprehensive taxonomy of game accessibility
   - Categories: visual, auditory, motor, cognitive

2. **"Sonification and Accessibility in Video Games"** - Collins, 2013
   - How games use audio for accessibility
   - Sonic icons and audio maps

3. **"Designing Audio Games for Deaf and Hard of Hearing Players"**
   - Visual alternatives to audio cues
   - Haptic feedback options

4. **"Sound Visualization for Deaf Players"** - UIST Workshop 2019
   - Real-time audio-to-visual translation
   - User studies with deaf gamers

### 4.2 Existing Tools & Software

#### **Commercial Solutions**

| Tool | Features | Platform |
|------|----------|----------|
| **Sound Lock** | Shows audio direction | Windows overlay |
| **Pedalboard Ear** | Frequency visualizer | Browser-based |
| **NVIDIA Broadcast** | Noise removal (not localization) | Windows |

#### **Game-Specific Features**

| Game | Accessibility Feature | How It Works |
|------|----------------------|--------------|
| **Fortnite** | Sound Visualizer | Radar-style icons for sounds |
| **Apex Legends** | Visualize Sound Effects | Directional indicators |
| **The Last of Us Part II** | Audio Cues | Visual ping for sounds |
| **Rainbow Six Siege** | No built-in | Community requests exist |
| **PUBG** | Limited captions | No spatial audio visualization |

### 4.3 How Other Games Handle Visual Audio Cues

#### **Fortnite's Approach (Best-in-Class)**

```
┌──────────────────────────────────────────────┐
│                    Screen                     │
│                                              │
│                 ┌─────────┐                  │
│        ●        │ Player  │        ●         │  ● = Footstep
│                 └─────────┘                  │
│    ▲                                    ▲    │  ▲ = Gunshot
│                                              │
│  ●   Distance indicated by:                  │
│      - Icon size (larger = closer)           │
│      - Opacity (brighter = closer)           │
│      - Icon type (footstep, gunshot, etc.)   │
│                                              │
└──────────────────────────────────────────────┘
```

**Key Design Principles:**
1. **Radial placement** around player center
2. **Icon differentiation** by sound type
3. **Distance encoding** via size/opacity
4. **Minimal clutter** - only important sounds
5. **Customizable** - toggle on/off, adjust sensitivity

#### **Implementation for PUBG-style HUD**

```python
class VisualSoundRadar:
    """
    Fortnite-style visual sound radar
    """
    def __init__(self, screen_center, radar_radius=150):
        self.center = screen_center
        self.radius = radar_radius
        
        self.icons = {
            'footstep': pygame.image.load('icons/footstep.png'),
            'gunshot': pygame.image.load('icons/gunshot.png'),
            'vehicle': pygame.image.load('icons/vehicle.png'),
        }
        
        self.distance_scale = {
            'near': 1.0,
            'mid': 0.7,
            'far': 0.4
        }
    
    def render_event(self, event, surface):
        # Convert azimuth to screen position
        angle_rad = math.radians(event.azimuth_deg - 90)  # 0° = up
        
        # Distance affects how close to center
        dist_factor = self.distance_scale[event.distance_bucket]
        r = self.radius * (0.3 + 0.7 * (1 - dist_factor))
        
        x = self.center[0] + r * math.cos(angle_rad)
        y = self.center[1] + r * math.sin(angle_rad)
        
        # Get and scale icon
        icon = self.icons[event.kind]
        scale = 0.5 + 0.5 * dist_factor  # Larger = closer
        icon = pygame.transform.scale(icon, 
            (int(32 * scale), int(32 * scale)))
        
        # Opacity based on distance and confidence
        icon.set_alpha(int(255 * event.confidence * dist_factor))
        
        surface.blit(icon, (x - icon.get_width()//2, 
                           y - icon.get_height()//2))
```

### 4.4 Relevant Patents

1. **EA Patent: "Generating Subtitles for Audio"** (2019)
   - Audio event detection for subtitles
   - Part of EA's accessibility patent pledge

2. **Sony Patent: "Audio Visualization in Games"** (2020)
   - Real-time audio to visual conversion
   - Directional sound indicators

3. **Microsoft Patent: "Spatial Audio Accessibility"** (2021)
   - Haptic feedback for spatial audio
   - Visual sound representation

---

## 5. Technical Implementation

### 5.1 Best Python Libraries

#### **Audio Capture & Processing**

| Library | Purpose | Notes |
|---------|---------|-------|
| **sounddevice** | WASAPI loopback | Low latency, cross-platform |
| **pyaudio** | Alternative capture | More setup, same results |
| **soundcard** | Loopback recording | Simpler API |
| **numpy** | Array operations | Essential for DSP |
| **scipy.signal** | FFT, filtering | Reliable, fast |

#### **Feature Extraction**

| Library | Purpose | Notes |
|---------|---------|-------|
| **librosa** | Audio features | Comprehensive, but slow |
| **python-aubio** | Onset detection | C-based, fast |
| **essentia** | Audio analysis | Industrial-grade |
| **torchaudio** | ML-ready features | GPU accelerated |

#### **Machine Learning**

| Library | Purpose | Notes |
|---------|---------|-------|
| **scikit-learn** | Classical ML | Fast, CPU-based |
| **PyTorch** | Deep learning | Flexible, ONNX export |
| **ONNX Runtime** | Inference | Fastest CPU inference |
| **TensorFlow Lite** | Edge inference | Mobile/embedded |

#### **Visualization**

| Library | Purpose | Notes |
|---------|---------|-------|
| **pygame** | Prototype overlay | Simple, fast |
| **pyqtgraph** | Real-time plots | Good for debugging |
| **Dear PyGui** | Modern UI | Fast rendering |
| **Win32 APIs** | Click-through overlay | Production quality |

### 5.2 Neural Network Architectures

#### **Architecture 1: Separate Localization + Classification**

```python
class SpatialAudioNet(nn.Module):
    """
    Two-headed network:
    - Head 1: Direction estimation (azimuth, front/back)
    - Head 2: Sound classification (footstep, gunshot, vehicle)
    """
    def __init__(self):
        super().__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 64, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 32, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 16, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )
        
        # Direction head
        self.direction_head = nn.Sequential(
            nn.Linear(128 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [sin(θ), cos(θ)] for angle
        )
        
        # Front/back head
        self.fb_head = nn.Sequential(
            nn.Linear(128 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Sigmoid -> 0=back, 1=front
            nn.Sigmoid()
        )
        
        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(128 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4)  # 4 classes
        )
    
    def forward(self, left, right):
        x = torch.stack([left, right], dim=1)  # [B, 2, T]
        features = self.features(x).flatten(1)  # [B, 512]
        
        direction = self.direction_head(features)
        azimuth = torch.atan2(direction[:, 0], direction[:, 1])
        
        front_back = self.fb_head(features)
        
        class_logits = self.class_head(features)
        
        return azimuth, front_back, class_logits
```

#### **Architecture 2: SELDnet-style (Joint Localization + Detection)**

```python
class SELDNet(nn.Module):
    """
    Sound Event Localization and Detection network
    Based on DCASE challenge architecture
    """
    def __init__(self, num_classes=4, num_directions=72):  # 5° resolution
        super().__init__()
        
        # CNN for spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((8, 2)),
            
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((8, 2)),
            
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        
        # Bi-directional GRU for temporal context
        self.gru = nn.GRU(
            input_size=256 * 4,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Output heads
        self.sed_head = nn.Linear(256, num_classes)  # Detection
        self.doa_head = nn.Linear(256, num_classes * 3)  # Direction (x,y,z per class)
    
    def forward(self, mel_left, mel_right):
        # Input: [B, T, F] spectrograms for each channel
        x = torch.stack([mel_left, mel_right], dim=1)  # [B, 2, T, F]
        
        x = self.cnn(x)  # [B, 256, T', F']
        x = x.permute(0, 2, 1, 3).flatten(2)  # [B, T', 256*F']
        
        x, _ = self.gru(x)  # [B, T', 256]
        
        sed = torch.sigmoid(self.sed_head(x))  # [B, T', C]
        doa = torch.tanh(self.doa_head(x))  # [B, T', C*3]
        doa = doa.view(*doa.shape[:2], -1, 3)  # [B, T', C, 3]
        
        return sed, doa
```

### 5.3 Achieving <100ms Latency

#### **Latency Budget Breakdown**

```
┌────────────────────────────────────────────────────────────┐
│ Component              │ Target  │ Optimization           │
├────────────────────────┼─────────┼────────────────────────┤
│ Audio capture buffer   │ 20 ms   │ WASAPI exclusive mode  │
│ FFT/feature extraction │ 5 ms    │ Vectorized NumPy       │
│ Direction estimation   │ 3 ms    │ GCC-PHAT optimized     │
│ Classification         │ 5 ms    │ ONNX Runtime / RF      │
│ Event smoothing        │ 2 ms    │ Exponential average    │
│ HUD rendering          │ 16 ms   │ 60 FPS, GPU accelerated│
│ OS scheduling jitter   │ 5 ms    │ Thread priority        │
├────────────────────────┼─────────┼────────────────────────┤
│ TOTAL                  │ 56 ms   │ Well under 100ms! ✓    │
└────────────────────────────────────────────────────────────┘
```

#### **Optimization Techniques**

```python
# 1. Use overlap-add for smooth processing
class AudioProcessor:
    def __init__(self, chunk_size=1024, hop_size=512):
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.buffer = np.zeros(chunk_size - hop_size)
    
    def process_chunk(self, new_audio):
        # Overlap with previous chunk
        full_chunk = np.concatenate([self.buffer, new_audio])
        self.buffer = full_chunk[-self.hop_size:]
        return full_chunk

# 2. Pre-compute FFT parameters
class OptimizedFFT:
    def __init__(self, n):
        self.n = n
        self.window = np.hanning(n)
        # Pre-allocate output arrays
        self.spectrum = np.empty(n//2 + 1, dtype=np.complex128)
    
    def compute(self, signal):
        np.multiply(signal, self.window, out=signal)
        np.fft.rfft(signal, out=self.spectrum)
        return self.spectrum

# 3. Use ONNX Runtime for neural networks
import onnxruntime as ort

class FastClassifier:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # or CUDAExecutionProvider
        )
        self.input_name = self.session.get_inputs()[0].name
    
    def predict(self, features):
        return self.session.run(None, {self.input_name: features})[0]

# 4. Thread pool for parallel processing
from concurrent.futures import ThreadPoolExecutor

class ParallelProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def process(self, frame):
        # Submit parallel tasks
        futures = [
            self.executor.submit(compute_gcc_phat, frame),
            self.executor.submit(compute_band_energies, frame),
            self.executor.submit(compute_spectral_features, frame),
        ]
        # Collect results
        return [f.result() for f in futures]
```

### 5.4 Feature Extraction Techniques

#### **Recommended Feature Set**

```python
def extract_all_features(frame, samplerate=48000):
    """
    Complete feature extraction for spatial audio HUD
    """
    left, right = frame[:, 0], frame[:, 1]
    
    # Stereo features
    features = {
        # Direction features
        'itd': compute_itd_gcc_phat(left, right, samplerate),
        'ild': compute_ild_db(left, right),
        'interaural_coherence': compute_coherence(left, right),
        'correlation': compute_correlation(left, right),
        
        # Energy features
        'energy': np.mean(frame**2),
        'energy_ratio': np.mean(left**2) / (np.mean(right**2) + 1e-9),
        
        # Spectral features (mono mix)
        'spectral_centroid': librosa.feature.spectral_centroid(
            y=mix, sr=samplerate)[0, 0],
        'spectral_bandwidth': librosa.feature.spectral_bandwidth(
            y=mix, sr=samplerate)[0, 0],
        'spectral_rolloff': librosa.feature.spectral_rolloff(
            y=mix, sr=samplerate, roll_percent=0.85)[0, 0],
        'spectral_flatness': librosa.feature.spectral_flatness(y=mix)[0, 0],
        'zcr': librosa.feature.zero_crossing_rate(mix)[0, 0],
        
        # Band energies
        'sub_bass': band_energy(mix, samplerate, 20, 100),
        'bass': band_energy(mix, samplerate, 100, 300),
        'low_mid': band_energy(mix, samplerate, 300, 800),
        'mid': band_energy(mix, samplerate, 800, 2000),
        'high_mid': band_energy(mix, samplerate, 2000, 5000),
        'highs': band_energy(mix, samplerate, 5000, 10000),
        
        # Temporal features
        'onset_strength': librosa.onset.onset_strength(
            y=mix, sr=samplerate).max(),
        'rms_delta': compute_rms_delta(frame),
    }
    
    return features
```

---

## 6. Implementation Recommendations

### 6.1 Immediate Improvements to Your Codebase

Based on reviewing your [signal_processing.py](src/spatial_hud/signal_processing.py) and [event_classifier.py](src/spatial_hud/event_classifier.py):

#### **1. Enhanced Front/Back Disambiguation**

```python
# Replace FrontBackDisambiguator with multi-feature approach
class EnhancedFrontBackDisambiguator:
    def __init__(self):
        self.history = deque(maxlen=5)
        self.baseline_ratio = None
    
    def estimate(self, spectrum, freqs, left, right, samplerate):
        # Feature 1: High/low ratio (your current approach)
        low = np.mean(spectrum[freqs < 300])
        high = np.mean(spectrum[(freqs > 4000) & (freqs < 8000)])
        ratio_score = (high / (low + 1e-9)) - 1.0
        
        # Feature 2: Interaural coherence in 2-6kHz
        coherence = compute_band_coherence(left, right, 2000, 6000, samplerate)
        # Front sounds have higher coherence
        coherence_score = (coherence - 0.5) * 2
        
        # Feature 3: Spectral rolloff comparison
        left_rolloff = spectral_rolloff(left, samplerate)
        right_rolloff = spectral_rolloff(right, samplerate)
        # Front sounds: higher rolloff on both channels
        rolloff_score = (left_rolloff + right_rolloff) / 2 / samplerate - 0.4
        
        # Combine with weights
        raw_score = 0.4 * ratio_score + 0.35 * coherence_score + 0.25 * rolloff_score
        
        # Temporal smoothing with hysteresis
        self.history.append(raw_score)
        smoothed = np.mean(self.history)
        
        return np.clip(smoothed, -1, 1)
```

#### **2. Improved Classification with Adaptive Thresholds**

```python
# Add environmental adaptation
class AdaptiveClassifier:
    def __init__(self):
        self.ambient_tracker = AmbientTracker()
        self.scene_detector = SceneDetector()  # Indoor/outdoor/vehicle
    
    def classify(self, features):
        scene = self.scene_detector.current_scene
        
        # Adjust thresholds based on scene
        thresholds = self.get_scene_thresholds(scene)
        
        # Your classification logic with adaptive thresholds
        ...
```

### 6.2 Next Steps

1. **Data Collection**
   - Record 10+ hours of PUBG gameplay audio
   - Manually annotate sound events with timestamps and directions
   - Split into train/validation/test sets

2. **Model Training**
   - Start with Random Forest on hand-crafted features
   - Experiment with lightweight CNNs
   - Export best model to ONNX for fast inference

3. **Direction Estimation**
   - Implement sub-band GCC-PHAT
   - Add spectral-based front/back disambiguation
   - Consider learning-based approach with enough data

4. **HUD Design**
   - Study Fortnite's sound visualizer
   - User testing with deaf gamers
   - Iterate on icon design and placement

5. **Production Quality**
   - Move overlay to C++/C# with Direct2D
   - Anti-cheat compatibility testing
   - Performance profiling

---

## References

### Academic Papers
1. Dietz, M. et al. (2011). "Auditory Model Based Direction Estimation of Concurrent Speakers from Binaural Signals"
2. May, T. et al. (2011). "A Probabilistic Model for Robust Localization Based on a Binaural Auditory Front-End"
3. Adavanne, S. et al. (2018). "Sound Event Localization and Detection of Overlapping Sources Using Convolutional Recurrent Neural Networks"
4. Blauert, J. (1997). "Spatial Hearing: The Psychophysics of Human Sound Localization" (MIT Press)

### HRTF Databases
- CIPIC: https://www.ece.ucdavis.edu/cipic/
- Listen: http://recherche.ircam.fr/equipes/salles/listen/
- SOFA: https://www.sofaconventions.org/

### Audio ML Resources
- PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn
- AudioSet: https://research.google.com/audioset/
- ESC-50: https://github.com/karolpiczak/ESC-50

### Game Accessibility
- Can I Play That: https://caniplaythat.com/
- Game Accessibility Guidelines: http://gameaccessibilityguidelines.com/
