
"""
Extract time-domain and frequency-domain features from an I24 EMG CSV.
Save feature values and their units to feature_results.csv.
"""

from pathlib import Path
import csv
import re

from i24_emg.preprocessing.csv_reader import (
    read_emg_csv,
    extract_channel,
)
from i24_emg.analysis.time_domain import (
    extract_time_domain_features,
)
from i24_emg.analysis.frequency_domain import (
    extract_frequency_domain_features,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = (
    PROJECT_ROOT
    / "output"
    / "acquisitions"
    / "i24_emg_recording.csv"
)

CHANNEL_NAME = "EMG_1"
SAMPLING_RATE_HZ = 4000.0
NPERSEG = 1024
FREQUENCY_RANGE = None

# Set this to the actual unit used in your CSV.
# Examples: "V", "mV", "µV", or "ADC counts"
SIGNAL_UNIT = "V"

OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "feature_results.csv"
)


# ============================================================
# FEATURE UNIT MAPPING
# ============================================================

def normalize_name(name):
    """Normalize feature names for unit lookup."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def get_feature_unit(name, domain, signal_unit="V"):
    """Return units based on the implemented feature formulas."""

    time_units = {
        "iemg": f"{signal_unit}·samples",
        "mav": signal_unit,
        "mmav1": signal_unit,
        "mmav2": signal_unit,
        "ssi": f"{signal_unit}²",
        "variance": f"{signal_unit}²",
        "vorder": signal_unit,
        "rms": signal_unit,
        "waveform_length": signal_unit,
        "willison_amplitude": "count",
        "zero_crossings": "count",
        "slope_sign_changes": "count",
        "log_detector": signal_unit,
    }

    frequency_units = {
        "total_power": f"{signal_unit}²",
        "mean_frequency": "Hz",
        "median_frequency": "Hz",
        "peak_frequency": "Hz",
        "frequency_variance": "Hz²",
        "frequency_std": "Hz",
        "spectral_entropy": "dimensionless",
        "spectral_edge_frequency": "Hz",
    }

    units = time_units if domain == "time" else frequency_units

    if name not in units:
        raise ValueError(
            f"No unit defined for feature '{name}' in domain '{domain}'."
        )

    return units[name]


# ============================================================
# MAIN
# ============================================================

def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"EMG CSV not found: {CSV_FILE}")

    print("=" * 60)
    print("I24 EMG — FEATURE EXTRACTION")
    print("=" * 60)

    data = read_emg_csv(CSV_FILE)
    signal = extract_channel(data, CHANNEL_NAME)

    print(f"Input file: {CSV_FILE.name}")
    print(f"Channel: {CHANNEL_NAME}")
    print(f"Samples: {len(signal)}")
    print(f"Sampling rate: {SAMPLING_RATE_HZ:g} Hz")
    print(f"Duration: {len(signal) / SAMPLING_RATE_HZ:.3f} s")
    print(f"Assumed signal unit: {SIGNAL_UNIT}")

    # Extract features
    time_features = extract_time_domain_features(signal)

    frequency_features = extract_frequency_domain_features(
        signal,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        nperseg=min(NPERSEG, len(signal)),
        frequency_range=FREQUENCY_RANGE,
    )

    # Combine values and units
    results = []

    for domain, features in (
        ("time", time_features),
        ("frequency", frequency_features),
    ):
        for name, value in features.items():
            results.append({
                "domain": domain,
                "feature": name,
                "value": value,
                "unit": get_feature_unit(name, domain),
            })

    # Print results
    print("\n" + "-" * 75)
    print(f"{'Domain':<12} {'Feature':<28} {'Value':>14}  Unit")
    print("-" * 75)

    for item in results:
        print(
            f"{item['domain']:<12} "
            f"{item['feature']:<28} "
            f"{item['value']:>14.8g}  "
            f"{item['unit']}"
        )

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["domain", "feature", "value", "unit"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED")
    print(f"Time-domain features: {len(time_features)}")
    print(f"Frequency-domain features: {len(frequency_features)}")
    print(f"Total features: {len(results)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
