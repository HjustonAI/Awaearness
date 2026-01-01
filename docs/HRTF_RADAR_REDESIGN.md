# 🎯 Projekt: HRTF Audio Radar dla Graczy z Niepełnosprawnością Słuchu

## Misja
Stworzyć narzędzie, które przywraca graczom niedosłyszącym i głuchym pełne doświadczenie gry w PUBG poprzez wizualizację dźwięków przestrzennych w czasie rzeczywistym.

---

## 1. Analiza Problemu

### 1.1 Jak PUBG koduje dźwięk przestrzenny (HRTF)

HRTF (Head-Related Transfer Function) symuluje jak ludzkie ucho słyszy dźwięki z różnych kierunków. Koduje 3 typy informacji w stereo:

| Wskaźnik | Co koduje | Zakres częstotliwości | Wyzwanie |
|----------|-----------|----------------------|----------|
| **ITD** (Interaural Time Difference) | Opóźnienie czasowe między uszami | < 1500 Hz | Łatwe do wykrycia |
| **ILD** (Interaural Level Difference) | Różnica głośności (cień głowy) | > 1500 Hz | Łatwe do wykrycia |
| **Spectral Cues** (Pinna filtering) | Przód/tył, góra/dół | 4-16 kHz | **Najtrudniejsze** |

### 1.2 Cone of Confusion (Stożek Konfuzji)

```
        PRZÓD (0°)
           │
      ╭────┼────╮
     ╱     │     ╲   ← Wszystkie punkty na tym stożku
    ╱      │      ╲     mają IDENTYCZNE ITD i ILD!
   ╱       │       ╲
  ╱        ●        ╲
  ╲      (głowa)    ╱
   ╲       │       ╱
    ╲      │      ╱
     ╲     │     ╱
      ╰────┼────╯
           │
        TYŁ (180°)
```

**Problem**: Dźwięk z przodu (0°) i z tyłu (180°) ma takie samo ITD i ILD!
**Rozwiązanie**: Musimy analizować **spectral cues** - HRTF dodaje charakterystyczne "notche" (wycięcia) w wysokich częstotliwościach dla dźwięków z tyłu.

---

## 2. Architektura Rozwiązania

### 2.1 Pipeline Przetwarzania

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HRTF AUDIO RADAR PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │ WASAPI  │───▶│ Multi-Band   │───▶│  Direction  │───▶│   Event    │  │
│  │ Capture │    │ Analyzer     │    │  Estimator  │    │ Classifier │  │
│  │ (Stereo)│    │ (6 bands)    │    │  (Enhanced) │    │    (ML)    │  │
│  └─────────┘    └──────────────┘    └─────────────┘    └────────────┘  │
│       │                                    │                  │         │
│       │              ┌─────────────────────┘                  │         │
│       │              │                                        │         │
│       │         ┌────▼────────┐    ┌─────────────────┐       │         │
│       │         │ Front/Back  │───▶│ Confidence      │       │         │
│       │         │Disambiguator│    │ Fusion          │◀──────┘         │
│       │         │ (Spectral)  │    │                 │                 │
│       │         └─────────────┘    └────────┬────────┘                 │
│       │                                     │                          │
│       │                              ┌──────▼──────┐                   │
│       │                              │    HUD      │                   │
│       └─────────────────────────────▶│   Render    │                   │
│              (audio passthrough)     │  (Visual)   │                   │
│                                      └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Moduły do Przeprojektowania

| Moduł | Stan | Priorytet | Zmiany |
|-------|------|-----------|--------|
| `DirectionEstimator` | ✅ Działa | P1 | Multi-band GCC-PHAT |
| `FrontBackDisambiguator` | ⚠️ Słaby | **P0** | Spectral notch detection |
| `EventClassifier` | ⚠️ Heurystyczny | P1 | ML classifier (PUBG-specific) |
| `AudioCapture` | ✅ Działa | P2 | Optymalizacja latencji |
| `HUD` | ✅ Działa | P2 | Fortnite-style icons |

---

## 3. Rozwiązania Techniczne

### 3.1 Enhanced Direction Estimation (Multi-Band GCC-PHAT)

**Problem**: Standardowy GCC-PHAT używa całego spektrum. HRTF koduje różne informacje w różnych pasmach.

**Rozwiązanie**: Osobna estymacja kierunku dla każdego pasma, potem fuzja wyników.

```python
class MultiBandDirectionEstimator:
    """
    Estymacja kierunku w 6 pasmach częstotliwości:
    - Band 1 (100-500 Hz):   ITD dominant, footsteps low
    - Band 2 (500-1500 Hz):  ITD+ILD, footsteps main
    - Band 3 (1500-3000 Hz): ILD dominant, voice, mid gunshots
    - Band 4 (3000-6000 Hz): ILD + spectral, transients
    - Band 5 (6000-12000 Hz): Spectral cues, front/back
    - Band 6 (12000-20000 Hz): High spectral cues
    """
    
    BANDS = [
        (100, 500, 0.15),    # (low_hz, high_hz, weight)
        (500, 1500, 0.25),   # Main footstep range
        (1500, 3000, 0.25),  # Transition zone
        (3000, 6000, 0.20),  # Transient details
        (6000, 12000, 0.10), # Front/back cues
        (12000, 20000, 0.05) # Air absorption cues
    ]
    
    def estimate(self, left, right, samplerate):
        band_estimates = []
        
        for low, high, weight in self.BANDS:
            # Bandpass filter
            left_band = bandpass_filter(left, low, high, samplerate)
            right_band = bandpass_filter(right, low, high, samplerate)
            
            # GCC-PHAT for this band
            azimuth, confidence = gcc_phat(left_band, right_band, samplerate)
            
            # Weight by band energy (ignore silent bands)
            energy = np.sqrt(np.mean(left_band**2) + np.mean(right_band**2))
            band_estimates.append((azimuth, confidence * weight * energy))
        
        # Weighted fusion
        total_weight = sum(c for _, c in band_estimates)
        final_azimuth = sum(a * c for a, c in band_estimates) / (total_weight + 1e-9)
        
        return final_azimuth, total_weight
```

### 3.2 Front/Back Disambiguation (Spectral Notch Detection)

**Kluczowe odkrycie**: HRTF dodaje charakterystyczne "notche" (wycięcia) w paśmie 8-12 kHz dla dźwięków z tyłu głowy.

```python
class SpectralFrontBackDetector:
    """
    Wykrywa charakterystyki HRTF wskazujące przód/tył:
    
    1. Pinna Notch Detection (8-12 kHz)
       - Dźwięki z tyłu mają głębsze wycięcie w tym paśmie
       
    2. High-Frequency Rolloff
       - Dźwięki z tyłu mają szybszy spadek wysokich częstotliwości
       
    3. Interaural Coherence w paśmie 2-6 kHz
       - Dźwięki z przodu mają wyższą koherencję
    """
    
    def estimate(self, left, right, samplerate):
        # 1. Spectral analysis
        left_spec = np.abs(np.fft.rfft(left * np.hanning(len(left))))
        right_spec = np.abs(np.fft.rfft(right * np.hanning(len(right))))
        freqs = np.fft.rfftfreq(len(left), 1/samplerate)
        
        # 2. Pinna notch detection (8-12 kHz)
        notch_band = (freqs >= 8000) & (freqs <= 12000)
        reference_band = (freqs >= 4000) & (freqs <= 8000)
        
        notch_energy = (left_spec[notch_band].mean() + right_spec[notch_band].mean()) / 2
        ref_energy = (left_spec[reference_band].mean() + right_spec[reference_band].mean()) / 2
        
        notch_ratio = notch_energy / (ref_energy + 1e-9)
        # Low ratio = deep notch = BACK
        # High ratio = no notch = FRONT
        
        # 3. High-frequency rolloff
        high_band = freqs >= 10000
        mid_band = (freqs >= 2000) & (freqs < 10000)
        
        high_energy = (left_spec[high_band].mean() + right_spec[high_band].mean()) / 2
        mid_energy = (left_spec[mid_band].mean() + right_spec[mid_band].mean()) / 2
        
        rolloff_ratio = high_energy / (mid_energy + 1e-9)
        # Low ratio = steep rolloff = BACK
        
        # 4. Interaural coherence (2-6 kHz)
        coherence_band = (freqs >= 2000) & (freqs <= 6000)
        left_coh = left_spec[coherence_band]
        right_coh = right_spec[coherence_band]
        
        coherence = np.corrcoef(left_coh, right_coh)[0, 1]
        # High coherence = FRONT
        # Low coherence = BACK (more diffuse)
        
        # 5. Combine features
        # Scale to [-1, 1] where +1 = FRONT, -1 = BACK
        notch_score = (notch_ratio - 0.5) * 2  # Assumes 0.5 is neutral
        rolloff_score = (rolloff_ratio - 0.3) * 3
        coherence_score = coherence
        
        # Weighted combination
        front_back_score = (
            0.4 * notch_score +
            0.3 * rolloff_score +
            0.3 * coherence_score
        )
        
        return np.clip(front_back_score, -1, 1)
```

### 3.3 Event Classification (ML-Enhanced)

**Obecny system**: Heurystyki oparte na progach
**Nowy system**: Lekki model ML + heurystyki jako fallback

```python
class HybridEventClassifier:
    """
    Dwupoziomowy klasyfikator:
    1. Fast-path: Heurystyki dla oczywistych przypadków
    2. ML-path: Random Forest dla niejednoznacznych
    """
    
    def __init__(self, model_path=None):
        self.heuristic = HeuristicClassifier()
        self.ml_model = self._load_model(model_path)
        
    def classify(self, features: FeaturePacket) -> Event:
        # Fast-path: Oczywiste przypadki
        if features.energy > 0.3 and features.onset_strength > 0.15:
            # Bardzo głośne, nagłe = gunfire
            return self._create_event("gunfire", features, confidence=0.9)
            
        if features.low_band_energy > 0.1 and features.spectral_flatness < 0.3:
            # Mocne basy, niski szum = vehicle
            return self._create_event("vehicle", features, confidence=0.85)
        
        # ML-path: Niejednoznaczne przypadki
        if self.ml_model:
            feature_vector = self._extract_features(features)
            prediction = self.ml_model.predict_proba([feature_vector])[0]
            
            classes = ["ambient", "footstep", "gunfire", "vehicle"]
            best_idx = np.argmax(prediction)
            
            return self._create_event(
                classes[best_idx], 
                features, 
                confidence=prediction[best_idx]
            )
        
        # Fallback: Heurystyki
        return self.heuristic.classify(features)
    
    def _extract_features(self, fp: FeaturePacket) -> np.ndarray:
        """Feature vector for ML model (13 features)"""
        return np.array([
            fp.energy,
            fp.onset_strength,
            fp.spectral_centroid / 10000,  # Normalize
            fp.low_band_energy,
            fp.mid_band_energy,
            fp.high_band_energy,
            fp.spectral_flatness,
            fp.high_band_energy / (fp.mid_band_energy + 1e-9),  # high/mid ratio
            fp.low_band_energy / (fp.mid_band_energy + 1e-9),  # low/mid ratio
            fp.direction_confidence,
            abs(fp.front_back_score),
            np.std(fp.band_energies),  # Spectral variance
            np.max(fp.band_energies) / (np.mean(fp.band_energies) + 1e-9),  # Peak ratio
        ])
```

---

## 4. PUBG-Specific Audio Signatures

### 4.1 Charakterystyki Dźwięków PUBG

Na podstawie analizy (do weryfikacji z prawdziwymi nagraniami):

| Dźwięk | Częstotliwości | Onset | Czas trwania | Charakterystyka |
|--------|----------------|-------|--------------|-----------------|
| **Kroki (trawa)** | 400-1200 Hz | Średni | 100-200ms | Rytmiczne, miękki atak |
| **Kroki (beton)** | 500-2500 Hz | Wysoki | 50-150ms | Ostrzejszy, wyższy pitch |
| **AKM** | 200-8000 Hz | Bardzo wysoki | 150-300ms | Broadband, mocne basy |
| **M416** | 300-10000 Hz | Bardzo wysoki | 100-250ms | Wyższy pitch niż AKM |
| **AWM** | 100-12000 Hz | Ekstremalny | 400-600ms | Najgłośniejszy, długi decay |
| **UAZ** | 50-800 Hz | Niski | Ciągły | Silnik, niskie częstotliwości |
| **Buggy** | 100-1500 Hz | Niski | Ciągły | Wyższy pitch niż UAZ |

### 4.2 Distance Estimation

PUBG używa attenuation i low-pass filter dla odległości:

```python
def estimate_distance(features: FeaturePacket) -> DistanceBucket:
    """
    PUBG distance cues:
    1. Overall energy (amplitude falloff)
    2. High-frequency rolloff (air absorption)
    3. Reverb ratio (distant = more reverb)
    """
    
    # High/low ratio - distant sounds lose highs
    high_low_ratio = features.high_band_energy / (features.low_band_energy + 1e-9)
    
    # Spectral centroid drops with distance
    centroid_norm = features.spectral_centroid / 5000
    
    # Energy threshold (calibrated for PUBG)
    if features.energy > 0.15 and high_low_ratio > 0.8:
        return DistanceBucket.NEAR  # < 20m
    elif features.energy > 0.05 and high_low_ratio > 0.4:
        return DistanceBucket.MID   # 20-50m
    else:
        return DistanceBucket.FAR   # > 50m
```

---

## 5. Plan Implementacji

### Faza 1: Core Improvements (Tydzień 1-2)

1. **[P0] SpectralFrontBackDetector**
   - Implementacja pinna notch detection
   - Testy z syntetycznym HRTF
   - A/B test vs obecny system

2. **[P1] MultiBandDirectionEstimator**
   - 6-band GCC-PHAT
   - Weighted fusion
   - Benchmark latencji

### Faza 2: ML Integration (Tydzień 3-4)

3. **[P1] Data Collection Tool**
   - Nagrywanie PUBG audio z labelami
   - Annotator dla kroków/strzałów/pojazdów
   - Eksport do formatu ML

4. **[P1] Train Classifier**
   - Random Forest jako baseline
   - Optionally: lightweight CNN
   - Validation na unseen data

### Faza 3: Polish & Test (Tydzień 5-6)

5. **[P2] HUD Improvements**
   - Fortnite-style icons
   - Smooth animations
   - Confidence-based opacity

6. **[P2] Performance Optimization**
   - Profiling
   - SIMD for FFT
   - Target: <50ms latency

---

## 6. Metryki Sukcesu

| Metryka | Obecna wartość | Cel | Jak mierzyć |
|---------|----------------|-----|-------------|
| Direction accuracy (L/R) | ~80%? | >95% | Test z syntetycznym HRTF |
| Front/Back accuracy | ~50%? | >80% | Test z HRTF database |
| Classification accuracy | ~70%? | >85% | Labeled PUBG recordings |
| End-to-end latency | ~50ms | <80ms | Timestamp diff |
| False positive rate | Unknown | <10% | User testing |

---

## 7. Potencjalne Rozszerzenia

1. **Per-game profiles**: Różne gry używają różnych silników dźwiękowych
2. **User calibration**: Pozwól użytkownikowi dostroić czułość
3. **ML personalization**: Adaptacja do indywidualnego HRTF użytkownika
4. **Haptic feedback**: Wibracje dla graczy z głuchotą
5. **Integration z Discord/OBS**: Overlay dla streamerów

---

## 8. Zasoby

### Bazy danych HRTF
- CIPIC HRTF Database (UC Davis)
- MIT KEMAR HRTF
- SADIE II Database

### Biblioteki
- `scipy.signal` - filtrowanie
- `librosa` - analiza audio (opcjonalnie)
- `scikit-learn` - ML classifier
- `onnxruntime` - inference (produkcja)

### Artykuły naukowe
- Algazi et al., "Approximating the head-related transfer function using simple geometric models"
- Keyrouz & Diepold, "A new HRTF interpolation approach"
- Middlebrooks & Green, "Sound localization by human listeners"

---

*Dokument stworzony: 31.12.2025*
*Autor: Spatial HUD Team*
*Cel: Accessibility for deaf/HoH gamers*
