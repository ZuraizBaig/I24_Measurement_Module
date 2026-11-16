# I24 EMG Processing

A Python library for acquiring, recording, visualizing, and preprocessing electromyography (EMG) signals from the HASOMED I24 system.

The library provides an I24-specific abstraction layer on top of the ScienceMode 4 Python library. It converts incoming I24 live-data packets into structured EMG data and provides tools for recording, CSV export, visualization, and offline signal preprocessing.

## Features

### Acquisition
- Communication with the HASOMED I24 through ScienceMode 4.
- Configuration of I24 measurement signals.
- Structured EMG frame acquisition.
- I24-specific channel mapping.
- Buffered acquisition and asynchronous frame reading.

### EMG Tools
- Record EMG1, EMG2, or both channels.
- Save recordings as CSV files.
- Record and save in one operation.
- Plot acquired EMG signals.
- Load and plot previously recorded CSV files.

### Preprocessing
- Read saved I24 EMG recordings from CSV.
- Automatic identification of common EMG column names.
- Support for configurable channel mappings.
- 50 Hz notch filtering.
- Butterworth bandpass filtering.
- Full-wave rectification.
- RMS envelope calculation.
- Structured output containing raw and processed signals.

## Architecture

The library is organized into three main modules:

```text
i24_emg/
├── acquisition/
│   ├── backend.py
│   ├── buffer.py
│   ├── channel.py
│   ├── config.py
│   ├── i24_acquisition.py
│   └── models.py
│
├── emg_tools/
│   └── emg_tools.py
│
└── preprocessing/
    ├── csv_reader.py
    ├── filters.py
    ├── models.py
    └── pipeline.py
```

| Module | Responsibility |
|---|---|
| `acquisition` | I24 communication abstraction, configuration, buffering, and frame acquisition |
| `emg_tools` | Recording, plotting, and CSV export |
| `preprocessing` | Offline CSV loading, filtering, rectification, and envelope extraction |

## Requirements

- Python
- HASOMED I24 hardware for live acquisition
- ScienceMode 4 Python library
- Dependencies specified by the project package configuration

The preprocessing functionality can be used without connecting an I24 device, provided that a compatible recording CSV is available.

## Installation

### Clone the repository

```powershell
git clone <I24_EMG_REPOSITORY_URL>
cd <I24_EMG_REPOSITORY_DIRECTORY>
```

### Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install the package in development mode

```powershell
python -m pip install --upgrade pip
pip install -e .
```

Install the project's declared dependencies through its package configuration.

## 1. Acquisition

The acquisition module provides a structured interface for retrieving EMG frames from the I24.

It is built on top of ScienceMode 4 and does not replace the underlying device communication protocol.

### I24 channel mapping

The application uses the following I24 live-data mapping:

| I24 packet sample index | Application channel |
|---:|---|
| 2 | EMG1 |
| 3 | EMG2 |

This mapping is specific to the I24 configuration used by this library.

### Acquisition workflow

The general acquisition workflow is:

1. Open the serial connection to the I24.
2. Initialize the I24 device through ScienceMode 4.
3. Obtain the Dyscom layer.
4. Create the acquisition backend.
5. Configure the required signals and sampling settings.
6. Initialize the acquisition instance.
7. Start acquisition.
8. Read structured EMG frames.
9. Stop acquisition and release the hardware resources.

See the `examples/` directory for the complete hardware acquisition implementation.

## 2. EMG Tools

`I24EMGTools` provides higher-level functionality using an initialized and running `I24Acquisition` instance.

### Recording

The recording interface supports:

- `EMG1`
- `EMG2`
- `both`

Example:

```python
data = await tools.record(
    duration=5,
    channel="both",
)
```

The returned dictionary contains a time vector and the requested EMG channel data.

For example:

```python
{
    "time": [...],
    "EMG1": [...],
    "EMG2": [...]
}
```

Only the requested channels are included.

### CSV export

The tools module provides:

- `save_csv()`
- `record_and_save_csv()`

`record_and_save_csv()` performs the recording and saves the resulting data in one operation.

### Plotting

The module also provides:

- `plot()` — visualize acquired EMG data.
- `plot_csv()` — visualize an existing CSV recording.

`plot_csv()` is an offline operation and does not require an active I24 connection.

## 3. Preprocessing

The preprocessing module operates on saved recordings. It does not communicate with the I24 hardware.

### Supported operations

| Operation | Description | Default |
|---|---|---|
| Notch filter | Suppresses power-line interference | Disabled |
| Bandpass filter | Butterworth bandpass filtering | Enabled |
| Full-wave rectification | Converts negative signal amplitudes to positive values | Disabled |
| RMS envelope | Calculates a moving RMS envelope | Disabled |

### Default preprocessing configuration

```python
from i24_emg.preprocessing import PreprocessingConfig

config = PreprocessingConfig(
    apply_notch=False,
    notch_hz=50.0,
    notch_quality_factor=30.0,

    apply_bandpass=True,
    lowcut_hz=20.0,
    highcut_hz=450.0,
    filter_order=4,

    calculate_rectified=False,
    calculate_rms_envelope=False,
    rms_window_ms=50.0,
)
```

These are configurable defaults, not mandatory settings for every EMG application.

### Reading a CSV recording

```python
from i24_emg.preprocessing import read_i24_csv

recording = read_i24_csv(
    "data/emg_recording.csv",
)
```

The CSV reader supports common EMG column naming conventions and can accept explicit column mappings for non-standard files.

### Preprocessing a CSV file

```python
from i24_emg.preprocessing import (
    PreprocessingConfig,
    preprocess_csv,
)

config = PreprocessingConfig(
    apply_notch=True,
    apply_bandpass=True,
    calculate_rectified=True,
    calculate_rms_envelope=True,
)

processed = preprocess_csv(
    "data/emg_recording.csv",
    config=config,
)
```

The returned recording contains structured raw and derived signal data together with processing metadata.

### Public preprocessing API

The preprocessing package exposes:

```python
from i24_emg.preprocessing import (
    I24Signal,
    I24EMGRecording,
    read_i24_csv,
    PreprocessingConfig,
    ProcessedEMGSignal,
    PreprocessedEMGRecording,
    preprocess_recording,
    preprocess_csv,
    bandpass_filter,
    notch_filter,
    full_wave_rectify,
    rms_envelope,
)
```

## Examples

The GitHub repository includes an `examples/` directory containing practical demonstrations of the library.

Examples are intended to show how to:
- Initialize and acquire data from the I24.
- Record individual EMG channels or both channels.
- Export recordings to CSV.
- Plot live-recorded data.
- Load and visualize existing recordings.
- Apply offline preprocessing.

Run the examples from the repository root so that package imports resolve correctly.

Example execution patterns are documented alongside the individual scripts.

## Tests

The repository includes tests for the library's functionality.

Preprocessing tests can be performed using synthetic signals and CSV files without connecting I24 hardware.

Acquisition tests require a compatible I24 device and serial connection.

Run the relevant test scripts from the repository root using the commands documented in the test directory.

## Repository and distribution

The GitHub repository contains the complete development project:

- Source code
- Examples
- Tests
- Documentation
- Package configuration

The PyPI distribution is intended to contain the installable library and its runtime dependencies. Development examples and tests are maintained in GitHub rather than included in the published package.

## Limitations

- Live acquisition requires compatible HASOMED I24 hardware and ScienceMode 4 support.
- Channel mapping is specific to the configured I24 signal arrangement.
- Preprocessing operates on saved recordings and does not control hardware.
- Filter settings should be selected according to the signal characteristics and intended application.

## License

See the `LICENSE` file in this repository.

## Maintainer

Developed for the HASOMED I24 platform.
