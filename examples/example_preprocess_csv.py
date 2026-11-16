"""
I24 EMG preprocessing example.

Opens two separate figure windows:
    Window 1: EMG1 (Channel 2)
    Window 2: EMG2 (Channel 3)

Each window shows:
    1. Raw EMG
    2. Detrended EMG (if enabled)
    3. Notch-filtered EMG (if enabled)
    4. Bandpass-filtered EMG
    5. Full-wave rectified EMG
    6. RMS envelope

Run from the project root:
    python examples/example_preprocess_csv.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from i24_emg.preprocessing import (
    read_emg_csv,
    extract_channels,
    FilterConfig,
    EnvelopeConfig,
    EMGPreprocessingPipeline,
)

from i24_emg.preprocessing.filters import (
    detrend_signal,
    notch_filter,
    bandpass_filter,
    full_wave_rectify,
    rms_envelope,
)


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "output" / "acquisitions" / "i24_emg_recording.csv"

OUTPUT_DIR = PROJECT_ROOT / "output" / "preprocessed"

SAMPLING_RATE_HZ = 4000.0

# Physical CSV columns mapped to logical I24 EMG channels.
CHANNEL_MAPPING = {
    "EMG1": "EMG_1",
    "EMG2": "EMG_2",
}

# Filter settings
FILTER_CONFIG = FilterConfig(
    sampling_rate_hz=SAMPLING_RATE_HZ,
    lowcut_hz=20.0,
    highcut_hz=450.0,
    notch_hz=50.0,
    filter_order=4,
)

# Envelope settings
ENVELOPE_CONFIG = EnvelopeConfig(
    method="rms",
    window_ms=500.0,
)

# Choose which processing stages to apply.
DETREND_METHOD = "constant"     # None, "constant", or "linear"
APPLY_NOTCH = True        # Set True to apply the 50 Hz notch filter
APPLY_BANDPASS = True
CALCULATE_ENVELOPE = True


# ============================================================
# 2. LOAD RECORDING
# ============================================================

def load_recording():
    print("=" * 65)
    print("I24 EMG PREPROCESSING")
    print("=" * 65)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Recording not found:\n{CSV_PATH}\n"
            "Update CSV_PATH to point to your EMG recording."
        )

    data = read_emg_csv(CSV_PATH)

    print(f"\nInput file: {CSV_PATH}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {list(data.columns)}")

    missing_columns = [
        column
        for column in CHANNEL_MAPPING.values()
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing EMG columns: {missing_columns}\n"
            f"Available columns: {list(data.columns)}"
        )

    extracted = extract_channels(
        data,
        list(CHANNEL_MAPPING.values()),
    )

    # Map physical CSV column names to logical EMG1 / EMG2 names.
    signals = {
        logical_name: extracted[physical_column]
        for logical_name, physical_column in CHANNEL_MAPPING.items()
    }

    return signals


# ============================================================
# 3. PROCESS ONE CHANNEL AND RETAIN EACH STAGE
# ============================================================

def process_channel(channel_name, raw_signal):
    print(f"\n{'-' * 65}")
    print(f"Processing {channel_name}")
    print(f"{'-' * 65}")

    # Keep a copy of the original signal.
    raw = np.asarray(raw_signal, dtype=float).copy()

    # Store each stage separately for plotting and export.
    stages = {
        "Raw EMG": raw,
    }

    current = raw.copy()

    # Stage 1: detrending (optional)
    if DETREND_METHOD is not None:
        current = detrend_signal(
            current,
            method=DETREND_METHOD,
        )
        stages[f"Detrended ({DETREND_METHOD})"] = current.copy()

    # Stage 2: notch filtering (optional)
    if APPLY_NOTCH:
        current = notch_filter(
            current,
            sampling_rate_hz=FILTER_CONFIG.sampling_rate_hz,
            notch_hz=FILTER_CONFIG.notch_hz,
        )
        stages[f"Notch Filtered ({FILTER_CONFIG.notch_hz:g} Hz)"] = (
            current.copy()
        )

    # Stage 3: bandpass filtering
    if APPLY_BANDPASS:
        current = bandpass_filter(
            current,
            sampling_rate_hz=FILTER_CONFIG.sampling_rate_hz,
            lowcut_hz=FILTER_CONFIG.lowcut_hz,
            highcut_hz=FILTER_CONFIG.highcut_hz,
            order=FILTER_CONFIG.filter_order,
        )
        stages[
            f"Bandpass Filtered "
            f"({FILTER_CONFIG.lowcut_hz:g}-"
            f"{FILTER_CONFIG.highcut_hz:g} Hz)"
        ] = current.copy()

    filtered_signal = current.copy()

    # Stage 4: full-wave rectification for visualization
    rectified = full_wave_rectify(filtered_signal)
    stages["Full-Wave Rectified"] = rectified.copy()

    # Stage 5: RMS envelope, calculated from the filtered EMG.
    # RMS is equivalent whether the input is signed or full-wave
    # rectified because squaring removes the sign.
    if CALCULATE_ENVELOPE:
        envelope = rms_envelope(
            filtered_signal,
            sampling_rate_hz=FILTER_CONFIG.sampling_rate_hz,
            window_ms=ENVELOPE_CONFIG.window_ms,
        )
        stages[
            f"RMS Envelope ({ENVELOPE_CONFIG.window_ms:g} ms)"
        ] = envelope.copy()

    # Run the high-level pipeline as the primary final-result API.
    pipeline = EMGPreprocessingPipeline(
        filter_config=FILTER_CONFIG,
        envelope_config=ENVELOPE_CONFIG,
        detrend_method=DETREND_METHOD,
        apply_notch=APPLY_NOTCH,
        apply_bandpass=APPLY_BANDPASS,
        calculate_envelope=CALCULATE_ENVELOPE,
        channel=channel_name,
    )

    result = pipeline.process(
        raw,
        metadata={
            "source_file": CSV_PATH.name,
            "source_column": CHANNEL_MAPPING[channel_name],
        },
    )

    print(f"Samples: {result.sample_count}")
    print(f"Duration: {result.duration_seconds:.3f} seconds")
    print(f"Pipeline stages: {result.stages}")

    # Use the pipeline's final output for the final plotted/exported stage.
    if CALCULATE_ENVELOPE:
        final_label = (
            f"RMS Envelope ({ENVELOPE_CONFIG.window_ms:g} ms)"
        )
        stages[final_label] = result.signal.copy()
    else:
        final_label = list(stages.keys())[-1]
        stages[final_label] = result.signal.copy()

    return stages, result


# ============================================================
# 4. PLOT ONE CHANNEL IN ITS OWN WINDOW
# ============================================================

def plot_channel(channel_name, stages):
    stage_items = list(stages.items())
    number_of_stages = len(stage_items)

    fig, axes = plt.subplots(
        number_of_stages,
        1,
        figsize=(14, 3 * number_of_stages),
        sharex=True,
        squeeze=False,
    )

    axes = axes.ravel()

    for ax, (stage_name, signal) in zip(axes, stage_items):
        signal = np.asarray(signal)
        time_seconds = (
            np.arange(signal.size) / FILTER_CONFIG.sampling_rate_hz
        )

        ax.plot(time_seconds, signal, linewidth=0.7)
        ax.set_title(stage_name, loc="left")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")

    fig.suptitle(
        f"{channel_name} — EMG Processing Stages",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    # Keep a separate window for this channel.
    return fig


# ============================================================
# 5. SAVE PROCESSING RESULTS
# ============================================================

def save_channel_results(channel_name, stages):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save all stages as separate CSV columns when they have equal length.
    lengths = {len(signal) for signal in stages.values()}

    if len(lengths) != 1:
        print(
            f"Warning: {channel_name} stage lengths differ. "
            "Skipping combined stage CSV export."
        )
        return

    output_data = {
        "time_seconds": (
            np.arange(len(next(iter(stages.values()))))
            / FILTER_CONFIG.sampling_rate_hz
        ),
    }

    for stage_name, signal in stages.items():
        output_data[stage_name] = signal

    output_path = OUTPUT_DIR / f"{channel_name}_processing_stages.csv"
    pd.DataFrame(output_data).to_csv(output_path, index=False)

    print(f"Saved processing stages: {output_path}")


# ============================================================
# 6. MAIN
# ============================================================

def main():
    signals = load_recording()

    figures = []

    for channel_name, raw_signal in signals.items():
        stages, result = process_channel(
            channel_name,
            raw_signal,
        )

        save_channel_results(channel_name, stages)

        fig = plot_channel(channel_name, stages)
        figures.append(fig)

    print("\nProcessing finished.")
    print("Two separate channel windows should now be displayed.")

    # Display all figures together in separate windows.
    plt.show()


if __name__ == "__main__":
    main()
