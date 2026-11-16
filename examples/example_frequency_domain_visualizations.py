
"""
Example: I24 EMG frequency-domain visualizations.

Demonstrates:
    1. Power Spectral Density (PSD)
    2. Fast Fourier Transform (FFT)
    3. Spectrogram

Input:
    examples/muskelaufnahme.csv

Channel mapping:
    EMG1 -> Channel 2
    EMG2 -> Channel 3
"""

from pathlib import Path

import matplotlib.pyplot as plt

from i24_emg.emg_tools.emg_tools import I24EMGTools


# --------------------------------------------------
# Configuration
# --------------------------------------------------

#DATA_FILE = Path("examples") / "muskelaufnahme.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "output" / "acquisitions" / "i24_emg_recording.csv"
SAMPLING_RATE_HZ = 4000.0
CHANNEL = "both"  # "EMG1", "EMG2", or "both"

OUTPUT_DIR = Path("examples") / "frequency_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"CSV file not found: {DATA_FILE.resolve()}"
        )

    # Create the utility object without initializing hardware.
    # These visualization methods operate on existing CSV files.
    tools = I24EMGTools.__new__(I24EMGTools)

    print("=" * 60)
    print("I24 EMG FREQUENCY VISUALIZATION EXAMPLE")
    print("=" * 60)
    print(f"Input file: {DATA_FILE}")
    print(f"Channels: {CHANNEL}")
    print(f"Sampling rate: {SAMPLING_RATE_HZ} Hz")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    # --------------------------------------------------
    # 1. Power Spectral Density (PSD)
    # --------------------------------------------------

    print("\n[1/3] Generating PSD plot...")

    tools.plot_psd_csv(
        filename=DATA_FILE,
        channel=CHANNEL,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        nperseg=1024,
        frequency_range=(0, 500),
        save_path=OUTPUT_DIR / "emg_psd.png",
        show=False,
    )

    print("Saved: emg_psd.png")

    # --------------------------------------------------
    # 2. Fast Fourier Transform (FFT)
    # --------------------------------------------------

    print("\n[2/3] Generating FFT plot...")

    tools.plot_fft_csv(
        filename=DATA_FILE,
        channel=CHANNEL,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        frequency_range=(0, 500),
        save_path=OUTPUT_DIR / "emg_fft.png",
        show=False,
    )

    print("Saved: emg_fft.png")

    # --------------------------------------------------
    # 3. Spectrogram
    # --------------------------------------------------

    print("\n[3/3] Generating spectrogram...")

    tools.plot_spectrogram_csv(
        filename=DATA_FILE,
        channel=CHANNEL,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        nperseg=256,
        noverlap=128,
        frequency_range=(0, 500),
        save_path=OUTPUT_DIR / "emg_spectrogram.png",
        show=False,
    )

    print("Saved: emg_spectrogram.png")

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------

    plt.close("all")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Check the plots in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()