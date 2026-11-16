
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from i24_emg.preprocessing import read_emg_csv, extract_channels
from i24_emg.preprocessing.filters import (
    detrend_signal,
    notch_filter,
    bandpass_filter,
    full_wave_rectify,
    rms_envelope,
)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = (
    PROJECT_ROOT
    / "output"
    / "acquisitions"
    / "i24_emg_recording.csv"
)

FS = 4000.0
LOWCUT_HZ = 20.0
HIGHCUT_HZ = 450.0
NOTCH_HZ = 50.0
RMS_WINDOW_MS = 100.0


def process_channel(signal):
    raw = np.asarray(signal, dtype=float)

    detrended = detrend_signal(raw, method="constant")

    notched = notch_filter(
        detrended,
        sampling_rate_hz=FS,
        notch_hz=NOTCH_HZ,
    )

    filtered = bandpass_filter(
        notched,
        sampling_rate_hz=FS,
        lowcut_hz=LOWCUT_HZ,
        highcut_hz=HIGHCUT_HZ,
        order=4,
    )

    rectified = full_wave_rectify(filtered)

    envelope = rms_envelope(
        rectified,
        sampling_rate_hz=FS,
        window_ms=RMS_WINDOW_MS,
    )

    return raw, filtered, rectified, envelope


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    data = read_emg_csv(CSV_PATH)
    channels = extract_channels(data, ["EMG_1", "EMG_2"])

    for channel_name, column in [ ("EMG1", "EMG_1"),("EMG2", "EMG_2"),]:

        raw, filtered, rectified, envelope = process_channel(channels[column])

        time = np.arange(len(raw)) / FS

        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(time, raw)
        axes[0].set_title("Raw EMG")

        axes[1].plot(time, filtered)
        axes[1].set_title("Filtered EMG")

        axes[2].plot(time, rectified)
        axes[2].set_title("Rectified EMG")

        axes[3].plot(time, envelope)
        axes[3].set_title("RMS Envelope")

        for ax in axes:
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (seconds)")
        fig.suptitle(f"{channel_name} — Preprocessing")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
