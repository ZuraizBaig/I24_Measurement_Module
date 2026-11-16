from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, spectrogram
from ..acquisition.i24_acquisition import I24Acquisition


# Functionality:
# - record(): Acquires EMG data and returns it to Python without saving a file.
# - save_csv(): Acquires EMG data and saves it directly to a CSV file.
# - record_and_save_csv(): Acquires EMG data, saves it to CSV, and also returns
#   the acquired data to Python.
# - plot(): Acquires EMG data from the I24 and plots the recorded signal.
# - plot_csv(): Reads an already recorded CSV file and plots the selected signal.
#
# Note:
# save_csv() and record_and_save_csv() perform the acquisition internally;
# record() does not need to be called separately for these two functions.
#
# plot_csv() does NOT communicate with the I24. It only reads an existing CSV.


EMGChannel = Literal["EMG1", "EMG2", "both"]


class I24EMGTools:
    """
    Tools for recording, plotting, and exporting I24 EMG signals.

    This class is intentionally I24-specific.

    It uses an already initialized and running I24Acquisition
    instance for live acquisition functionality and provides
    higher-level functionality for:

        - recording EMG frames
        - selecting EMG1, EMG2, or both
        - plotting live-recorded EMG signals
        - plotting previously recorded CSV files
        - saving EMG data to CSV

    I24-specific channel mapping:

        packet index 2 -> EMG1
        packet index 3 -> EMG2
    """

    def __init__(self, acquisition: I24Acquisition) -> None:
        self._acquisition = acquisition

    async def record(
        self,
        duration: float,
        channel: EMGChannel = "both",
        timeout: float = 1.0,
    ) -> dict[str, list[float]]:
        """
        Record I24 EMG data for a specified duration.

        Parameters
        ----------
        duration:
            Recording duration in seconds.

        channel:
            Which signal to record:

                "EMG1"
                "EMG2"
                "both"

        timeout:
            Maximum time in seconds to wait for each live packet.

        Returns
        -------
        dict[str, list[float]]
            Dictionary containing the requested EMG samples.

        Example
        -------
        data = await tools.record(
            duration=5,
            channel="both",
        )
        """

        self._validate_channel(channel)

        if duration <= 0:
            raise ValueError(
                "Recording duration must be greater than 0 seconds."
            )

        if not self._acquisition.is_running:
            raise RuntimeError(
                "I24 acquisition is not running."
            )

        sampling_rate = (
            self._acquisition.sampling_info.sampling_rate_hz
        )

        expected_samples = int(
            duration * sampling_rate
        )

        data: dict[str, list[float]] = {
            "time": [],
        }

        if channel in ("EMG1", "both"):
            data["EMG1"] = []

        if channel in ("EMG2", "both"):
            data["EMG2"] = []

        for sample_index in range(expected_samples):
            frame = await self._acquisition.read_frame(
                timeout=timeout
            )

            sample_map = {
                sample.signal_type: sample.value
                for sample in frame.samples
            }

            time_seconds = (
                sample_index / sampling_rate
            )

            data["time"].append(
                time_seconds
            )

            if "EMG1" in data:
                if "EMG_1" not in sample_map:
                    raise RuntimeError(
                        "EMG1 was not present in the "
                        "I24 acquisition frame."
                    )

                data["EMG1"].append(
                    sample_map["EMG_1"]
                )

            if "EMG2" in data:
                if "EMG_2" not in sample_map:
                    raise RuntimeError(
                        "EMG2 was not present in the "
                        "I24 acquisition frame."
                    )

                data["EMG2"].append(
                    sample_map["EMG_2"]
                )

        return data

    async def plot(
        self,
        duration: float,
        channel: EMGChannel = "both",
        timeout: float = 1.0,
    ) -> dict[str, list[float]]:
        """
        Record and plot I24 EMG data.

        This function communicates with the I24,
        records the requested duration, and then
        displays the recorded signal.

        Parameters
        ----------
        duration:
            Recording duration in seconds.

        channel:
            "EMG1", "EMG2", or "both".

        timeout:
            Maximum time to wait for each live packet.

        Returns
        -------
        dict[str, list[float]]
            The recorded data used for the plot.
        """

        data = await self.record(
            duration=duration,
            channel=channel,
            timeout=timeout,
        )

        plt.figure(figsize=(12, 6))

        if "EMG1" in data:
            plt.plot(
                data["time"],
                data["EMG1"],
                label="EMG1",
            )

        if "EMG2" in data:
            plt.plot(
                data["time"],
                data["EMG2"],
                label="EMG2",
            )

        plt.xlabel("Time (s)")
        plt.ylabel("EMG amplitude")
        plt.title(
            f"I24 EMG Signal - {channel}"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show()

        return data

    async def plot_csv(
        self,
        filename: str | Path,
        channel: EMGChannel = "both",
    ) -> dict[str, list[float]]:
        """
        Read an existing I24 EMG CSV file and plot the selected signal.

        This function does NOT communicate with the I24 and does
        NOT perform any new acquisition.

        Parameters
        ----------
        filename:
            Path to an existing I24 EMG CSV file.

        channel:
            Which signal to plot:

                "EMG1"
                "EMG2"
                "both"

        Returns
        -------
        dict[str, list[float]]
            Dictionary containing the time and requested EMG data.

        Expected CSV format
        -------------------
        time_seconds,EMG_1,EMG_2

        Examples
        --------
        data = await tools.plot_csv(
            "test_output/emg_both.csv",
            channel="both",
        )

        data = await tools.plot_csv(
            "test_output/emg1.csv",
            channel="EMG1",
        )
        """

        self._validate_channel(channel)

        input_path = Path(filename)

        if not input_path.exists():
            raise FileNotFoundError(
                f"CSV file was not found: {input_path}"
            )

        if not input_path.is_file():
            raise ValueError(
                f"CSV path is not a file: {input_path}"
            )

        data: dict[str, list[float]] = {
            "time": [],
        }

        if channel in ("EMG1", "both"):
            data["EMG1"] = []

        if channel in ("EMG2", "both"):
            data["EMG2"] = []

        with input_path.open(
            mode="r",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            if reader.fieldnames is None:
                raise ValueError(
                    "CSV file does not contain a header."
                )

            required_columns = {
                "time_seconds",
            }

            if channel in ("EMG1", "both"):
                required_columns.add("EMG_1")

            if channel in ("EMG2", "both"):
                required_columns.add("EMG_2")

            missing_columns = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing_columns:
                raise ValueError(
                    "CSV file is missing required "
                    f"columns: {', '.join(sorted(missing_columns))}"
                )

            for row_number, row in enumerate(
                reader,
                start=2,
            ):

                try:
                    data["time"].append(
                        float(row["time_seconds"])
                    )

                    if "EMG1" in data:
                        data["EMG1"].append(
                            float(row["EMG_1"])
                        )

                    if "EMG2" in data:
                        data["EMG2"].append(
                            float(row["EMG_2"])
                        )

                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid numeric value in CSV "
                        f"at row {row_number}."
                    ) from error

        if not data["time"]:
            raise ValueError(
                "CSV file does not contain any data rows."
            )

        plt.figure(figsize=(12, 6))

        if "EMG1" in data:
            plt.plot(
                data["time"],
                data["EMG1"],
                label="EMG1",
            )

        if "EMG2" in data:
            plt.plot(
                data["time"],
                data["EMG2"],
                label="EMG2",
            )

        plt.xlabel("Time (s)")
        plt.ylabel("EMG amplitude")
        plt.title(
            f"I24 EMG CSV Signal - {channel}"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show()

        return data

    async def save_csv(
        self,
        duration: float,
        filename: str | Path,
        channel: EMGChannel = "both",
        timeout: float = 1.0,
    ) -> Path:
        """
        Record I24 EMG data and save it to CSV.

        Parameters
        ----------
        duration:
            Recording duration in seconds.

        filename:
            Output CSV file path.

        channel:
            "EMG1", "EMG2", or "both".

        timeout:
            Maximum time to wait for each live packet.

        Returns
        -------
        pathlib.Path
            Path to the saved CSV file.
        """

        data = await self.record(
            duration=duration,
            channel=channel,
            timeout=timeout,
        )

        output_path = Path(filename)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = ["time_seconds"]

        if "EMG1" in data:
            fieldnames.append("EMG_1")

        if "EMG2" in data:
            fieldnames.append("EMG_2")

        with output_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow(fieldnames)

            number_of_rows = len(
                data["time"]
            )

            for index in range(number_of_rows):

                row = [
                    data["time"][index]
                ]

                if "EMG1" in data:
                    row.append(
                        data["EMG1"][index]
                    )

                if "EMG2" in data:
                    row.append(
                        data["EMG2"][index]
                    )

                writer.writerow(row)

        return output_path

    async def record_and_save_csv(
        self,
        duration: float,
        filename: str | Path,
        channel: EMGChannel = "both",
        timeout: float = 1.0,
    ) -> dict[str, list[float]]:
        """
        Record EMG data and save it to CSV.

        This is useful when the caller also wants the
        recorded data in memory.

        Returns
        -------
        dict[str, list[float]]
            The recorded EMG data.
        """

        data = await self.record(
            duration=duration,
            channel=channel,
            timeout=timeout,
        )

        output_path = Path(filename)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = ["time_seconds"]

        if "EMG1" in data:
            fieldnames.append("EMG_1")

        if "EMG2" in data:
            fieldnames.append("EMG_2")

        with output_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow(fieldnames)

            for index in range(
                len(data["time"])
            ):

                row = [
                    data["time"][index]
                ]

                if "EMG1" in data:
                    row.append(
                        data["EMG1"][index]
                    )

                if "EMG2" in data:
                    row.append(
                        data["EMG2"][index]
                    )

                writer.writerow(row)

        return data

    @staticmethod
    def _validate_channel(
        channel: EMGChannel,
    ) -> None:
        """
        Validate the requested I24 EMG channel.
        """

        valid_channels = {
            "EMG1",
            "EMG2",
            "both",
        }

        if channel not in valid_channels:
            raise ValueError(
                f"Invalid EMG channel '{channel}'. "
                f"Expected one of: "
                f"{', '.join(sorted(valid_channels))}."
            )
    @staticmethod
    def _read_frequency_csv(
        filename: str | Path,
        channel: EMGChannel,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Read selected EMG channels from supported recorded CSV formats."""
        I24EMGTools._validate_channel(channel)

        input_path = Path(filename)

        if not input_path.exists():
            raise FileNotFoundError(
                f"CSV file was not found: {input_path}"
            )

        if not input_path.is_file():
            raise ValueError(f"CSV path is not a file: {input_path}")

        with input_path.open(
            mode="r",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            if not reader.fieldnames:
                raise ValueError("CSV file does not contain a header.")

            columns = set(reader.fieldnames)

            # Support the I24EMGTools CSV format and the
            # muskelaufnahme.csv format used in the examples.
            if "EMG_1" in columns or "EMG_2" in columns:
                aliases = {
                    "EMG1": "EMG_1",
                    "EMG2": "EMG_2",
                }
                time_column = (
                    "time_seconds"
                    if "time_seconds" in columns
                    else None
                )
            elif "Channel 2" in columns or "Channel 3" in columns:
                aliases = {
                    "EMG1": "Channel 2",
                    "EMG2": "Channel 3",
                }
                time_column = None
            else:
                raise ValueError(
                    "Unsupported CSV format. Expected EMG_1/EMG_2 "
                    "or Channel 2/Channel 3 columns."
                )

            requested = (
                ["EMG1", "EMG2"]
                if channel == "both"
                else [channel]
            )

            missing = [
                aliases[name]
                for name in requested
                if aliases[name] not in columns
            ]
            if missing:
                raise ValueError(
                    f"CSV file is missing required columns: {missing}"
                )

            times: list[float] = []
            samples = {name: [] for name in requested}

            for row_number, row in enumerate(reader, start=2):
                try:
                    if time_column:
                        times.append(float(row[time_column]))

                    for name in requested:
                        samples[name].append(
                            float(row[aliases[name]])
                        )
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid numeric value in CSV row {row_number}."
                    ) from exc

        signals = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in samples.items()
        }

        if not signals or any(x.size == 0 for x in signals.values()):
            raise ValueError("CSV contains no EMG samples.")

        for name, signal in signals.items():
            if signal.ndim != 1 or not np.all(np.isfinite(signal)):
                raise ValueError(
                    f"{name} must contain finite numeric samples."
                )

        if len({x.size for x in signals.values()}) != 1:
            raise ValueError("Selected EMG channels have unequal lengths.")

        if time_column:
            time = np.asarray(times, dtype=np.float64)
            if time.size != next(iter(signals.values())).size:
                raise ValueError("Time and EMG arrays have unequal lengths.")
            if not np.all(np.isfinite(time)):
                raise ValueError("Time values must be finite.")
        else:
            # Construct the time axis in each plotting method using fs.
            time = np.array([], dtype=np.float64)

        return time, signals

    @staticmethod
    def _finish_frequency_plot(
        fig,
        save_path: str | Path | None,
        show: bool,
    ) -> None:
        """Save and/or display a frequency-domain figure."""
        fig.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

    def plot_psd_csv(
        self,
        filename: str | Path,
        channel: EMGChannel = "both",
        sampling_rate_hz: float = 4000.0,
        nperseg: int = 1024,
        frequency_range: tuple[float, float] | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ):
        """Plot Welch power spectral density for an existing EMG CSV."""
        fs = float(sampling_rate_hz)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("sampling_rate_hz must be positive.")

        _, signals = self._read_frequency_csv(filename, channel)

        fig, ax = plt.subplots(figsize=(11, 6))

        for name, signal in signals.items():
            if signal.size < 2:
                raise ValueError("At least two samples are required.")

            segment_length = min(nperseg, signal.size)
            if segment_length < 2:
                raise ValueError("nperseg must be at least 2.")

            frequencies, psd = welch(
                signal,
                fs=fs,
                window="hann",
                nperseg=segment_length,
                scaling="density",
            )

            if frequency_range is not None:
                low, high = frequency_range
                if (
                    not np.isfinite(low)
                    or not np.isfinite(high)
                    or low < 0
                    or high <= low
                    or high > fs / 2
                ):
                    raise ValueError(
                        "frequency_range must satisfy "
                        "0 <= low < high <= Nyquist."
                    )
                mask = (frequencies >= low) & (frequencies <= high)
                frequencies, psd = frequencies[mask], psd[mask]

            ax.plot(frequencies, psd, label=name)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (amplitude units²/Hz)")
        ax.set_title(f"EMG Power Spectral Density — {channel}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        self._finish_frequency_plot(fig, save_path, show)
        return fig, ax

    def plot_fft_csv(
        self,
        filename: str | Path,
        channel: EMGChannel = "both",
        sampling_rate_hz: float = 4000.0,
        frequency_range: tuple[float, float] | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ):
        """Plot the single-sided FFT amplitude spectrum of an EMG CSV."""
        fs = float(sampling_rate_hz)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("sampling_rate_hz must be positive.")

        _, signals = self._read_frequency_csv(filename, channel)

        fig, ax = plt.subplots(figsize=(11, 6))

        for name, signal in signals.items():
            n = signal.size
            if n < 2:
                raise ValueError("At least two samples are required.")

            # Remove DC offset and reduce spectral leakage.
            centered = signal - np.mean(signal)
            window = np.hanning(n)
            window_sum = np.sum(window)

            if window_sum == 0:
                raise ValueError("Cannot calculate FFT for this signal.")

            spectrum = np.fft.rfft(centered * window)
            amplitude = np.abs(spectrum) / window_sum

            if n % 2 == 0:
                amplitude[1:-1] *= 2.0
            else:
                amplitude[1:] *= 2.0

            frequencies = np.fft.rfftfreq(n, d=1.0 / fs)

            if frequency_range is not None:
                low, high = frequency_range
                if (
                    not np.isfinite(low)
                    or not np.isfinite(high)
                    or low < 0
                    or high <= low
                    or high > fs / 2
                ):
                    raise ValueError(
                        "frequency_range must satisfy "
                        "0 <= low < high <= Nyquist."
                    )
                mask = (frequencies >= low) & (frequencies <= high)
                frequencies, amplitude = (
                    frequencies[mask],
                    amplitude[mask],
                )

            ax.plot(frequencies, amplitude, label=name)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (signal units)")
        ax.set_title(f"EMG FFT Amplitude Spectrum — {channel}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        self._finish_frequency_plot(fig, save_path, show)
        return fig, ax

    def plot_spectrogram_csv(
        self,
        filename: str | Path,
        channel: EMGChannel = "both",
        sampling_rate_hz: float = 4000.0,
        nperseg: int = 256,
        noverlap: int | None = None,
        frequency_range: tuple[float, float] | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ):
        """Plot an STFT spectrogram for an existing EMG CSV."""
        fs = float(sampling_rate_hz)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("sampling_rate_hz must be positive.")

        if not isinstance(nperseg, int) or isinstance(nperseg, bool) or nperseg < 2:
            raise ValueError("nperseg must be an integer >= 2.")

        _, signals = self._read_frequency_csv(filename, channel)

        fig, axes = plt.subplots(
            len(signals),
            1,
            figsize=(11, 5 * len(signals)),
            squeeze=False,
        )

        for ax, (name, signal) in zip(axes[:, 0], signals.items()):
            if signal.size < 2:
                raise ValueError("At least two samples are required.")

            segment_length = min(nperseg, signal.size)

            if noverlap is not None and not (
                isinstance(noverlap, int)
                and not isinstance(noverlap, bool)
                and 0 <= noverlap < segment_length
            ):
                raise ValueError(
                    "noverlap must be an integer from 0 to nperseg - 1."
                )

            frequencies, times, power = spectrogram(
                signal,
                fs=fs,
                window="hann",
                nperseg=segment_length,
                noverlap=(
                    min(segment_length // 2, segment_length - 1)
                    if noverlap is None
                    else noverlap
                ),
                scaling="density",
                mode="psd",
            )

            if frequency_range is not None:
                low, high = frequency_range
                if (
                    not np.isfinite(low)
                    or not np.isfinite(high)
                    or low < 0
                    or high <= low
                    or high > fs / 2
                ):
                    raise ValueError(
                        "frequency_range must satisfy "
                        "0 <= low < high <= Nyquist."
                    )
                mask = (frequencies >= low) & (frequencies <= high)
                frequencies, power = frequencies[mask], power[mask]

            if frequencies.size == 0:
                raise ValueError(
                    "No spectrogram bins remain in frequency_range."
                )

            # Avoid log10(0) while displaying power in dB.
            power_db = 10.0 * np.log10(
                np.maximum(power, np.finfo(float).tiny)
            )

            mesh = ax.pcolormesh(
                times,
                frequencies,
                power_db,
                shading="auto",
                cmap="viridis",
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f"EMG Spectrogram — {name}")
            fig.colorbar(mesh, ax=ax, label="PSD (dB/Hz)")

        self._finish_frequency_plot(fig, save_path, show)
        return fig, axes
