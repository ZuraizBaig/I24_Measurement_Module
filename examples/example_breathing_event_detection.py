
"""
Example: I24 Breathing Sensor Event Detection

Reads an acquired breathing CSV and performs:

    1. Breathing curve visualization
    2. Positive peak detection
    3. Negative trough detection
    4. Respiratory cycle segmentation
    5. Event validation
    6. Event visualization and manual verification
"""

import argparse
import csv

import numpy as np

from pathlib import Path

from i24_breathing_sensor.event_detection import (
    detect_respiratory_events,
    plot_breathing_curve,
    plot_detected_events,
)


def load_breathing_csv(
    filename,
    signal_column="breathing_ch4_raw",
):
    """Load raw Channel 4 signal values from acquisition CSV."""

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"CSV file not found: {filename}"
        )

    values = []

    with filename.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")

        if signal_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{signal_column}' not found. "
                f"Available columns: {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):

            try:
                value = float(row[signal_column])

            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid signal value at CSV line {line_number}."
                )

            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite signal value at line {line_number}."
                )

            values.append(value)

    if not values:
        raise ValueError("No breathing samples found.")

    return np.asarray(values, dtype=np.float64)


def main():

    parser = argparse.ArgumentParser(
        description="I24 Breathing Sensor Event Detection Example"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the breathing acquisition CSV"
    )

    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=4000.0,
        help="Validated acquisition sample rate in Hz (default: 4000)"
    )

    parser.add_argument(
        "--min-event-distance",
        type=float,
        default=1.0,
        help="Minimum distance between same-type events in seconds"
    )

    parser.add_argument(
        "--prominence-factor",
        type=float,
        default=0.35,
        help="Automatic prominence factor"
    )

    parser.add_argument(
        "--min-cycle-duration",
        type=float,
        default=1.0,
        help="Minimum accepted cycle duration in seconds"
    )

    parser.add_argument(
        "--max-cycle-duration",
        type=float,
        default=10.0,
        help="Maximum accepted cycle duration in seconds"
    )

    parser.add_argument(
        "--min-peak-to-peak",
        type=float,
        default=None,
        help="Optional minimum peak-to-peak amplitude"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Load raw breathing signal
    # ---------------------------------------------------------

    print("\nLoading breathing signal...")

    signal = load_breathing_csv(args.input)

    print(f"Samples loaded: {len(signal)}")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print(
        f"Recording duration: "
        f"{len(signal) / args.sampling_rate:.2f} seconds"
    )

    # ---------------------------------------------------------
    # 2. Run event detection
    # ---------------------------------------------------------

    print("\nRunning respiratory event detection...")

    result = detect_respiratory_events(
        signal=signal,
        sampling_rate=args.sampling_rate,
        min_event_distance_s=args.min_event_distance,
        prominence_factor=args.prominence_factor,
        min_cycle_duration_s=args.min_cycle_duration,
        max_cycle_duration_s=args.max_cycle_duration,
        min_peak_to_peak_amplitude=args.min_peak_to_peak,
    )

    # ---------------------------------------------------------
    # 3. Print detection summary
    # ---------------------------------------------------------

    print("\n----------------------------------------")
    print("EVENT DETECTION SUMMARY")
    print("----------------------------------------")

    print(f"Positive peaks: {len(result.positive_peaks)}")
    print(f"Negative troughs: {len(result.negative_troughs)}")
    print(f"Candidate cycles: {len(result.cycles)}")
    print(f"Valid cycles: {len(result.valid_cycles)}")
    print(f"Rejected cycles: {len(result.rejected_cycles)}")

    # ---------------------------------------------------------
    # 4. Print individual cycle information
    # ---------------------------------------------------------

    print("\nCycle details:")
    print(
        f"{'Cycle':<8}"
        f"{'Duration':<14}"
        f"{'Peak-to-peak':<18}"
        f"{'Status':<12}"
        f"Rejection reason"
    )

    for cycle in result.cycles:

        status = (
            "ACCEPTED"
            if cycle.is_valid
            else "REJECTED"
        )

        reasons = ", ".join(cycle.rejection_reasons)

        print(
            f"{cycle.cycle_index:<8}"
            f"{cycle.duration_s:<14.3f}"
            f"{cycle.peak_to_peak_amplitude:<18.6f}"
            f"{status:<12}"
            f"{reasons}"
        )

    # ---------------------------------------------------------
    # 5. Generate visualization files
    # ---------------------------------------------------------

    input_path = Path(args.input)

    curve_path = input_path.with_name(
        input_path.stem + "_breathing_curve.png"
    )

    event_path = input_path.with_name(
        input_path.stem + "_event_detection.png"
    )

    # Raw breathing curve.
    plot_breathing_curve(
        signal=signal,
        sampling_rate=args.sampling_rate,
        save_path=curve_path,
        show=False,
    )

    # Detected event overlay.
    plot_detected_events(
        signal=signal,
        sampling_rate=args.sampling_rate,
        result=result,
        save_path=event_path,
        show=True,
    )

    print("\nVisualization files:")
    print(f"Raw curve: {curve_path}")
    print(f"Event detection: {event_path}")

    print("\nEvent detection example completed.")


if __name__ == "__main__":
    main()
