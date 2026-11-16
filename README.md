# I24 Measurement Modules

*A collection of Python-based libraries for biomedical signal acquisition, processing, feature extraction, and analysis using the I24 measurement platform.*

---

## Overview

The **I24 Measurement Modules** project provides a collection of modular Python libraries designed to acquire, process, and analyse physiological signals through the I24 measurement platform.

The project currently consists of two independent modules:

1. **I24 EMG Module** – A library for electromyography (EMG) signal acquisition, preprocessing, feature extraction, and analysis.
2. **I24 Respiration Sensor Module** – A library for respiratory signal acquisition, event detection, and feature extraction.

Both modules follow a modular software architecture, separating signal acquisition from subsequent processing and analysis functionalities.

The goal is to provide reusable, maintainable, and extensible software components for biomedical measurement applications.

For detailed technical documentation, installation instructions, API information, and practical usage examples, please refer to the individual README files of each module.

---

## Modules

### 1. I24 EMG Module

The I24 EMG Module provides functionalities for acquiring, preprocessing, extracting features from, and analysing electromyography signals using the I24 measurement platform.

**Main components:**

- **Acquisition**
  - EMG signal acquisition
  - Hardware communication through ScienceMode4
  - Channel configuration and management
  - Frame-based signal acquisition

- **Preprocessing**
  - Signal filtering
  - Digital signal preprocessing
  - Configurable processing pipelines
  - CSV-based signal loading

- **Feature Extraction**
  - Extraction of EMG signal features
  - Time-domain and frequency-domain feature analysis
  - Feature-based signal representation

- **Analysis**
  - EMG signal analysis
  - Analysis of extracted signal features
  - EMG-specific analytical utilities

- **EMG Tools**
  - EMG-specific signal processing and analysis utilities

### 2. I24 Respiration Sensor Module

The I24 Respiration Sensor Module provides functionalities for acquiring and analysing respiratory signals using the I24 measurement platform.

**Main components:**

- **Acquisition**
  - Respiratory signal acquisition
  - Sensor channel configuration
  - Signal recording and data handling

- **Event Detection**
  - Respiratory peak and trough detection
  - Respiratory cycle segmentation
  - Respiratory cycle validation
  - Identification of respiratory waveform phases

- **Feature Extraction**

  **Time-domain features**
  - Respiratory rate
  - Inhalation and exhalation duration
  - Inspiratory-to-expiratory (I:E) ratio
  - Inter-breath interval
  - Respiratory cycle duration
  - Inspiratory duty cycle
  - Respiratory pause candidates

  **Frequency-domain features**
  - Dominant respiratory frequency
  - Respiratory power using Welch's PSD
  - Spectral centroid
  - Spectral band power
  - Spectral entropy
  - Time-varying respiratory frequency using STFT

---

## Repository Structure

```text
I24_Measurement_Modules/
│
├── I24_EMG_Module/
│   ├── src/
│   │   └── i24_emg/
│   ├── tests/
│   ├── examples/
│   ├── README.md
│   └── pyproject.toml
│
├── I24_Respiration_Sensor/
│   ├── src/
│   │   └── i24_breathing_sensor/
│   ├── tests/
│   ├── examples/
│   ├── README.md
│   └── pyproject.toml
│
├── README.md
├── LICENSE
└── .gitignore
```

Each module maintains its own source code, dependencies, installation configuration, and technical documentation.

---

## Requirements

- Python 3.10 or newer, subject to individual module requirements.
- Compatible I24 measurement hardware for hardware-dependent acquisition.
- ScienceMode4 Python library for communication with the I24 measurement platform.

Additional dependencies are defined separately within each module.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/I24_Measurement_Modules.git

cd I24_Measurement_Modules
```

Install the desired module independently.

**I24 EMG Module:**

```bash
pip install -e ./I24_EMG_Module
```

**I24 Respiration Sensor Module:**

```bash
pip install -e ./I24_Respiration_Sensor
```

For detailed installation instructions and usage examples, refer to the README file of the respective module.

---

## Usage

Both modules are designed to be integrated into Python applications and biomedical measurement workflows.

Their functionalities are accessible through their respective Python packages.

The I24 EMG Module supports EMG acquisition, preprocessing, feature extraction, and analysis, while the I24 Respiration Sensor Module provides respiratory acquisition, event detection, and feature extraction.

For detailed API documentation, configuration options, and practical examples, please refer to the individual module documentation.

---

## Development Philosophy

The I24 Measurement Modules follow these software development principles:

- **Modularity:** Independent components for acquisition, processing, and analysis.
- **Reusability:** Functionalities designed for integration into different applications.
- **Maintainability:** Clear separation of responsibilities and structured source code.
- **Extensibility:** Architecture designed to accommodate additional measurement modules and signal-processing functionalities.
- **Consistency:** A common software approach across different physiological measurement applications.

---

## Future Development

The repository is intended to serve as a foundation for additional I24-based measurement and signal-processing modules.

Future extensions may include additional physiological sensors, advanced signal-processing algorithms, feature extraction methods, and further analysis capabilities.

---

## License

See the [LICENSE](LICENSE) file for licensing and usage conditions.

---

*I24 Measurement Modules*  
*Biomedical Signal Acquisition, Processing, Feature Extraction, and Analysis with Python*
