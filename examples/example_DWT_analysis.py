
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt

from i24_emg.preprocessing.csv_reader import (
    read_emg_csv,
    extract_channel,
)
from i24_emg.preprocessing.filters import (
    detrend_signal,
    notch_filter,
    bandpass_filter,
)

# This is is a toy implementation changing of parameters and Wavelets with purposeful and meaninggful intuition is advised.
# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = (
    PROJECT_ROOT
    / "output"
    / "acquisitions"
    / "i24_emg_recording.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "dwt_preprocessed"

FS = 4000.0
LOWCUT_HZ = 20.0
HIGHCUT_HZ = 450.0
NOTCH_HZ = 50.0

WAVELET = "db2"
DWT_LEVEL = 6

CHANNELS = {
    "EMG1": "EMG_1",
    "EMG2": "EMG_2",
}


# ============================================================
# Preprocessing
# ============================================================

def preprocess_signal(signal):
    signal = np.asarray(signal, dtype=float)

    signal = detrend_signal(signal, method="constant")

    signal = notch_filter(
        signal,
        sampling_rate_hz=FS,
        notch_hz=NOTCH_HZ,
    )

    signal = bandpass_filter(
        signal,
        sampling_rate_hz=FS,
        lowcut_hz=LOWCUT_HZ,
        highcut_hz=HIGHCUT_HZ,
        order=4,
    )

    return signal


# ============================================================
# DWT analysis and plotting
# ============================================================

def analyze_channel(signal, channel_name):
    # 1. Preprocess the EMG signal
    filtered = preprocess_signal(signal)

    # 2. Configure DWT
    wavelet = pywt.Wavelet(WAVELET)
    max_level = pywt.dwt_max_level(
        len(filtered), wavelet.dec_len
    )
    level = min(DWT_LEVEL, max_level)

    if level < 3:
        raise ValueError(
            f"{channel_name}: signal is too short for level-3 DWT."
        )

    # 3. Perform DWT
    coeffs = pywt.wavedec(
        filtered,
        wavelet=WAVELET,
        level=level,
        mode="symmetric",
    )

    # 4. Keep A-level and detail bands D3–Dlevel only.
    # For level 6: A6, D6, D5, D4, D3.
    components = [(f"A{level}", coeffs[0])]

    for i, coeff in zip(range(level, 0, -1), coeffs[1:]):
        if i >= 2:
            components.append((f"D{i}", coeff))

    # 5. Plot filtered signal and selected components
    time = np.arange(len(filtered)) / FS

    fig, axes = plt.subplots(
        len(components) + 1,
        1,
        figsize=(13, 2.2 * (len(components) + 1)),
    )

    axes[0].plot(time, filtered, linewidth=0.6)
    axes[0].set_title(
        f"{channel_name} — Filtered EMG ({LOWCUT_HZ}–{HIGHCUT_HZ} Hz)"
    )
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    for ax, (name, coeff) in zip(axes[1:], components):
        if name.startswith("D"):
            i = int(name[1:])
            low = FS / (2 ** (i + 1))
            high = FS / (2 ** i)
        else:
            low = 0.0
            high = FS / (2 ** (level + 1))

        ax.plot(np.abs(coeff), linewidth=0.6)
        ax.set_title(
            f"{name} — Approx. {low:.1f}–{high:.1f} Hz"
        )
        ax.set_ylabel("|Coefficient|")
        ax.set_xlabel("Coefficient index")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / f"{channel_name}_dwt.png",
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"{channel_name}: preprocessing and DWT completed "
        f"(level {level})."
    )



# ============================================================
# Main
# ============================================================

def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

    data = read_emg_csv(CSV_FILE)

    for channel_name, column in CHANNELS.items():
        print(f"\nProcessing {channel_name} ({column})...")
        signal = extract_channel(data, column)
        analyze_channel(signal, channel_name)

    print(f"\nResults saved in: {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
