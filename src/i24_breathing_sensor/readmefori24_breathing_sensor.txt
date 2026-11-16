# I24 Breathing Sensor Module

A Python library for acquiring, detecting, and analysing respiratory signals using the HASOMED I24 measurement system.

The library is organised into three independent modules:

1. **Acquisition** – acquires the raw breathing signal from I24 Channel 4 and stores it in a CSV file.
2. **Event Detection** – detects respiratory waveform landmarks, segments respiratory cycles, and validates the detected cycles.
3. **Feature Extraction** – calculates temporal and frequency-domain respiratory features.

---

## 1. Project Structure

```text
i24_breathing_sensor/
│
├── __init__.py
│
├── acquisition/
│   ├── __init__.py
│   ├── breathing_acquisition.py
│   └── csv_writer.py
│
├── event_detection/
│   ├── __init__.py
│   ├── models.py
│   ├── peak_detection.py
│   ├── cycle_segmentation.py
│   ├── event_validation.py
│   ├── visualization.py
│   └── pipeline.py
│
└── feature_extraction/
    ├── __init__.py
    ├── models.py
    ├── temporal_features.py
    ├── frequency_features.py
    └── pipeline.py
```

The library separates hardware acquisition from respiratory signal processing. This allows event detection and feature extraction to be used independently with previously recorded signals.

---

## 2. Acquisition

The acquisition module communicates with the HASOMED I24 through the ScienceMode4 Python library.

It configures the Dyscom measurement module, starts the measurement, receives live-data packets, extracts the breathing signal from Channel 4, and writes the received samples to a CSV file.

### Signal Configuration

The configured signal order is:

| Channel index | Signal |
|---:|---|
| 0 | BI |
| 1 | EMG 1 |
| 2 | EMG 2 |
| 3 | BREATHING |

The breathing signal is extracted using index `3` from the received live-data packet.

The ADS129x configuration uses high-resolution mode and the configured output data rate `HR_MODE_4_KSPS__LP_MODE_2_KSPS`.

### Main Function

```python
async def acquire_breathing_signal(
    com_port: str,
    duration_s: float,
    output_file: str | Path,
) -> int:
```

**Parameters**

- `com_port`: Serial port connected to the I24, e.g. `"COM5"`.
- `duration_s`: Requested acquisition duration in seconds.
- `output_file`: Explicit destination path for the output CSV file.

**Returns**

- Integer representing the number of Channel 4 samples written to the CSV file.

The function does not generate filenames. The calling application is responsible for providing the output path.

### CSV Output

The acquisition module records the following columns:

| Column | Description |
|---|---|
| `sample_index` | Sequential index assigned to each recorded sample |
| `packet_number` | Packet number received from the I24 |
| `device_time_offset_raw` | Raw device time offset provided by the live-data packet |
| `host_elapsed_s` | Host-side elapsed time when the packet was received |
| `breathing_ch4_raw` | Raw breathing signal value from Channel 4 |
| `breathing_ch4_status` | Status flags associated with the received sample |

CSV writing is handled by `BreathingCsvWriter`, which uses a background thread to avoid blocking the acquisition loop.

**Note:** `host_elapsed_s` represents host-side packet reception time. It should not be treated as the device's sampling clock. The configured ADC output data rate and the observed live-packet reception rate are also distinct quantities.

---

## 3. Event Detection

The event detection module processes a one-dimensional respiratory waveform and identifies important landmarks in the breathing signal.

The processing pipeline consists of:

1. Signal validation
2. Positive peak detection
3. Negative trough detection
4. Respiratory cycle segmentation
5. Respiratory cycle validation

### Main Function

```python
def detect_respiratory_events(
    signal,
    sampling_rate,
    min_event_distance_s=1.0,
    prominence_factor=0.35,
    positive_prominence=None,
    negative_prominence=None,
    min_cycle_duration_s=1.0,
    max_cycle_duration_s=10.0,
    min_peak_to_peak_amplitude=None,
    min_positive_peak_prominence=None,
):
```

**Parameters**

| Parameter | Description |
|---|---|
| `signal` | Input respiratory waveform |
| `sampling_rate` | Sampling frequency in Hz |
| `min_event_distance_s` | Minimum separation between events of the same type |
| `prominence_factor` | Factor used for automatic prominence estimation |
| `positive_prominence` | Optional custom positive peak prominence |
| `negative_prominence` | Optional custom negative trough prominence |
| `min_cycle_duration_s` | Minimum acceptable respiratory cycle duration |
| `max_cycle_duration_s` | Maximum acceptable respiratory cycle duration |
| `min_peak_to_peak_amplitude` | Optional minimum cycle amplitude |
| `min_positive_peak_prominence` | Optional minimum positive peak prominence |

### Detected Events

The module identifies two types of waveform landmarks:

- **Positive peaks:** Local maxima in the respiratory waveform.
- **Negative troughs:** Local minima in the respiratory waveform.

Each detected event contains:

- Event type
- Sample index
- Time in seconds
- Amplitude
- Prominence

### Respiratory Cycle Segmentation

Respiratory cycles are segmented from one negative trough to the next negative trough.

A positive peak is associated with the cycle when available.

Each cycle contains:

- Cycle index
- Start trough
- Positive peak
- End trough
- Cycle duration
- Phase 1 duration
- Phase 2 duration
- Peak-to-peak amplitude
- Validation status
- Rejection reasons

### Cycle Validation

The validation stage evaluates the segmented cycles using the configured duration and optional amplitude and prominence criteria.

The resulting `EventDetectionResult` provides:

- `positive_peaks`
- `negative_troughs`
- `cycles`
- `valid_cycles`
- `rejected_cycles`

### Visualization

The event detection module also provides:

```python
plot_breathing_curve(...)
plot_detected_events(...)
```

These functions support visual inspection of the breathing waveform and detected events.

---

## 4. Feature Extraction

The feature extraction module combines temporal and frequency-domain analysis.

It uses the raw respiratory waveform, sampling frequency, and the `EventDetectionResult` produced by the event detection module.

### Main Function

```python
def extract_respiratory_features(
    signal,
    sampling_rate,
    detection_result,
    inhalation_phase=None,
    pause_threshold_s=None,
    include_rejected=False,
    respiratory_band_hz=(0.05, 1.0),
    spectral_bands_hz=None,
    welch_window_s=16.0,
    stft_window_s=16.0,
    stft_hop_s=2.0,
):
```

### 4.1 Temporal Features

Temporal feature extraction uses the detected respiratory cycles.

The implementation calculates:

| Feature | Description |
|---|---|
| Respiratory rate | Respiratory rate derived from cycle duration |
| Inter-breath interval | Interval calculated from consecutive trough landmarks within a cycle |
| Respiratory cycle duration | Duration from the starting trough to the ending trough |
| Phase 1 duration | Duration from the starting trough to the positive peak |
| Phase 2 duration | Duration from the positive peak to the ending trough |
| Inhalation duration | Phase duration assigned as inhalation |
| Exhalation duration | Phase duration assigned as exhalation |
| I:E ratio | Ratio between inhalation and exhalation duration |
| Inspiratory duty cycle | Fraction of the respiratory cycle assigned to inhalation |

The phase-dependent features require an explicit inhalation phase assignment.

**Inhalation phase parameter**

```python
inhalation_phase="phase_1"
```

or

```python
inhalation_phase="phase_2"
```

The parameter determines which detected phase is interpreted as inhalation. If it is `None`, inhalation-dependent features such as I:E ratio and inspiratory duty cycle are not assigned.

The physiological interpretation of Phase 1 and Phase 2 should be established for the sensor and signal configuration before assigning inhalation and exhalation.

**Pause candidates**

An optional `pause_threshold_s` can be supplied to identify prolonged inter-trough intervals.

These are reported as pause candidates. They should not be interpreted as confirmed apnea events.

### 4.2 Frequency-Domain Features

Frequency-domain analysis operates directly on the raw respiratory waveform.

The implementation includes:

| Feature | Method | Description |
|---|---|---|
| Dominant respiratory frequency | FFT | Dominant frequency within the configured respiratory band |
| Dominant respiratory rate | FFT | Dominant frequency converted to breaths per minute |
| FFT frequency resolution | FFT | Theoretical frequency-bin resolution |
| Respiratory band power | Welch PSD | Integrated power within the respiratory frequency band |
| Spectral centroid | Welch PSD | Power-weighted mean frequency |
| Spectral entropy | Welch PSD | Normalised measure of spectral distribution |
| Spectral band power | Welch PSD | Integrated power within user-defined frequency bands |
| Time-varying respiratory frequency | STFT | Dominant respiratory frequency over successive time windows |

### Default Frequency Settings

The default respiratory frequency band is:

```python
respiratory_band_hz = (0.05, 1.0)
```

The default exploratory spectral bands are:

```python
{
    "slow": (0.05, 0.15),
    "intermediate": (0.15, 0.40),
    "fast": (0.40, 1.0),
}
```

These bands are intended for exploratory signal analysis and are not clinical classifications.

The default Welch and STFT window durations are 16 seconds. The default STFT hop duration is 2 seconds.

### Frequency Analysis Considerations

- The sampling rate supplied to feature extraction must represent the actual signal sampling rate.
- FFT frequency resolution depends on the number of samples and the sampling rate.
- Short recordings may not provide sufficient frequency resolution for the requested respiratory band.
- Spectral power values are calculated from the supplied signal units. They should not be interpreted as calibrated physical quantities unless the sensor signal has been calibrated.

---

## 5. Combined Processing Workflow

The intended processing sequence is:

```text
        HASOMED I24
             |
             v
    Breathing Sensor
             |
             v
       Acquisition
             |
             v
       Raw CSV Data
             |
             v
      Signal Loading
             |
             v
      Event Detection
             |
             v
    Respiratory Cycles
             |
             v
     Feature Extraction
             |
       +-----+-----+
       |           |
       v           v
    Temporal   Frequency
    Features   Features
       |           |
       +-----+-----+
             |
             v
      Analysis Results
```

Event detection and feature extraction are separate processing stages. Feature extraction requires the event detection result for its temporal analysis, while its frequency-domain analysis uses the raw signal directly.

---

## 6. Dependencies

The source code uses the following Python libraries and components:

| Dependency | Purpose |
|---|---|
| NumPy | Numerical operations and signal arrays |
| SciPy | Peak detection and frequency-domain signal processing |
| ScienceMode4 | Communication with the HASOMED I24 and Dyscom measurement module |
| Python standard library | CSV handling, threading, queues, paths, and data models |

The ScienceMode4 package must be available in the Python environment used for hardware acquisition.

---

## 7. Current Scope

The library currently provides three implemented processing areas:

- I24 breathing signal acquisition and CSV recording.
- Respiratory peak and trough detection, cycle segmentation, validation, and visualization.
- Temporal and frequency-domain feature extraction.

The library is intended as a signal acquisition and processing module. The calculated features and detected events are not, by themselves, clinical diagnoses or validated medical measurements.

---

## 8. Development

The library is developed as part of the I24 Measurement Module project.

The package is designed to keep hardware acquisition, event detection, and feature extraction modular, allowing each processing stage to be maintained and extended independently.
