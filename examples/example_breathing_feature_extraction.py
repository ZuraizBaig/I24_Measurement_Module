
"""
Example: I24 Breathing Sensor Feature Extraction
=================================================

Loads a previously recorded Channel 4 signal and calculates
temporal and frequency-domain respiratory features.
"""

import argparse
import asyncio
import csv
import json

import numpy as np

from pathlib import Path

from i24_breathing_sensor.event_detection import (
    detect_respiratory_events,
)

from i24_breathing_sensor.feature_extraction import (
    extract_respiratory_features,
)


def load_breathing_csv(
    filename,
    signal_column="breathing_ch4_raw",
):
    """Load raw Channel 4 samples from the acquisition CSV."""

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"Recording not found: {filename}"
        )

    values = []

    with filename.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames or signal_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{signal_column}' not found in recording."
            )

        for row in reader:

            value = float(row[signal_column])

            if not np.isfinite(value):
                raise ValueError(
                    "Recording contains non-finite samples."
                )

            values.append(value)

    if not values:
        raise ValueError("No respiratory samples were loaded.")

    return np.asarray(values, dtype=np.float64)


def save_temporal_features(result, output_file):
    """Save per-cycle temporal features to CSV."""

    rows = result.temporal.per_cycle

    if not rows:
        print("No cycle features available for CSV export.")
        return

    fieldnames = list(rows[0].keys())

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_frequency_summary(result, output_file):
    """Save scalar frequency-domain features as JSON."""

    frequency = result.frequency

    data = {
        "dominant_frequency_hz": (
            frequency.dominant_frequency_hz
        ),

        "dominant_respiratory_rate_bpm": (
            frequency.dominant_respiratory_rate_bpm
        ),

        "fft_resolution_hz": frequency.fft_resolution_hz,

        "respiratory_band_power": (
            frequency.respiratory_band_power
        ),

        "spectral_centroid_hz": (
            frequency.spectral_centroid_hz
        ),

        "spectral_entropy": frequency.spectral_entropy,

        "spectral_band_power": frequency.spectral_band_power,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


def save_stft_frequency(result, output_file):
    """Save time-varying respiratory frequency to CSV."""

    frequency = result.frequency

    times = frequency.stft_times_s
    frequencies = frequency.time_varying_frequency_hz

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "window_center_time_s",
            "dominant_respiratory_frequency_hz",
            "dominant_respiratory_rate_bpm",
        ])

        for time_s, frequency_hz in zip(times, frequencies):

            rate = (
                frequency_hz * 60.0
                if np.isfinite(frequency_hz)
                else None
            )

            writer.writerow([
                float(time_s),
                float(frequency_hz)
                if np.isfinite(frequency_hz)
                else "",
                rate if rate is not None else "",
            ])


def main():

    parser = argparse.ArgumentParser(
        description="I24 Respiratory Feature Extraction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to Channel 4 acquisition CSV",
    )

    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=4000.0,
        help="Validated acquisition sampling rate in Hz",
    )

    parser.add_argument(
        "--inhalation-phase",
        choices=["phase_1", "phase_2"],
        default=None,
        help="Experimentally established inhalation phase",
    )

    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=None,
        help="Optional prolonged inter-trough interval threshold in seconds",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Load raw signal
    # ---------------------------------------------------------

    print("\nLoading respiratory recording...")

    signal = load_breathing_csv(args.input)

    duration_s = len(signal) / args.sampling_rate

    print(f"Samples: {len(signal)}")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print(f"Duration: {duration_s:.2f} seconds")

    # ---------------------------------------------------------
    # 2. Event detection
    # ---------------------------------------------------------

    print("\nRunning event detection...")

    detection_result = detect_respiratory_events(
        signal=signal,
        sampling_rate=args.sampling_rate,
    )

    print(f"Positive peaks: {len(detection_result.positive_peaks)}")
    print(f"Negative troughs: {len(detection_result.negative_troughs)}")
    print(f"Valid cycles: {len(detection_result.valid_cycles)}")
    print(f"Rejected cycles: {len(detection_result.rejected_cycles)}")

    # ---------------------------------------------------------
    # 3. Feature extraction
    # ---------------------------------------------------------

    print("\nExtracting respiratory features...")

    result = extract_respiratory_features(
        signal=signal,
        sampling_rate=args.sampling_rate,
        detection_result=detection_result,
        inhalation_phase=args.inhalation_phase,
        pause_threshold_s=args.pause_threshold,
        respiratory_band_hz=(0.05, 1.0),
        welch_window_s=16.0,
        stft_window_s=16.0,
        stft_hop_s=2.0,
    )

    # ---------------------------------------------------------
    # 4. Print temporal results
    # ---------------------------------------------------------

    print("\n========== TEMPORAL FEATURES ==========")

    for feature, stats in result.temporal.summary.items():

        if isinstance(stats, dict):

            print(f"\n{feature}")

            for name, value in stats.items():

                print(f"    {name}: {value}")

        else:

            print(f"{feature}: {stats}")

    # ---------------------------------------------------------
    # 5. Print frequency results
    # ---------------------------------------------------------

    frequency = result.frequency

    print("\n========== FREQUENCY FEATURES ==========")

    print(
        "Dominant respiratory frequency:",
        frequency.dominant_frequency_hz,
        "Hz"
    )

    print(
        "Spectral respiratory rate:",
        frequency.dominant_respiratory_rate_bpm,
        "breaths/min"
    )

    print(
        "Respiratory band power:",
        frequency.respiratory_band_power
    )

    print(
        "Spectral centroid:",
        frequency.spectral_centroid_hz,
        "Hz"
    )

    print(
        "Spectral entropy:",
        frequency.spectral_entropy
    )

    print(
        "Spectral band power:",
        frequency.spectral_band_power
    )

    # ---------------------------------------------------------
    # 6. Save feature results
    # ---------------------------------------------------------

    input_path = Path(args.input)

    output_dir = input_path.parent

    prefix = input_path.stem

    temporal_file = (
        output_dir / f"{prefix}_temporal_features.csv"
    )

    frequency_file = (
        output_dir / f"{prefix}_frequency_features.json"
    )

    stft_file = (
        output_dir / f"{prefix}_stft_frequency.csv"
    )

    save_temporal_features(
        result,
        temporal_file,
    )

    save_frequency_summary(
        result,
        frequency_file,
    )

    save_stft_frequency(
        result,
        stft_file,
    )

    print("\n========== FEATURE EXPORT ==========")

    print(f"Temporal features: {temporal_file}")
    print(f"Frequency features: {frequency_file}")
    print(f"STFT frequency: {stft_file}")

    print("\nFeature extraction completed successfully.")


if __name__ == "__main__":
    main()
