
"""
Frequency-Domain Feature Extraction
====================================

Implements:

1. Dominant respiratory frequency using FFT.
2. Respiratory-band power using Welch PSD.
3. Spectral centroid.
4. Spectral band power.
5. Normalized spectral entropy.
6. Time-varying respiratory frequency using STFT.
"""

import numpy as np

from scipy.integrate import trapezoid
from scipy.signal import detrend, welch, stft

from .models import FrequencyFeatureResult


DEFAULT_RESPIRATORY_BAND_HZ = (0.05, 1.0)

# Exploratory analysis bands, not clinical classifications.
DEFAULT_SPECTRAL_BANDS = {
    "slow": (0.05, 0.15),
    "intermediate": (0.15, 0.40),
    "fast": (0.40, 1.0),
}


def _validate_signal(signal, sampling_rate):

    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(x) < 16:
        raise ValueError(
            "At least 16 samples are required for frequency analysis."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    return x


def _validate_frequency_band(band, sampling_rate):

    low, high = band

    nyquist = sampling_rate / 2.0

    if low < 0:
        raise ValueError("Lower frequency cannot be negative.")

    if high <= low:
        raise ValueError("Upper frequency must exceed lower frequency.")

    if high > nyquist:
        raise ValueError(
            f"Upper frequency must not exceed Nyquist frequency "
            f"({nyquist} Hz)."
        )

    return low, high


def _integrate_band(frequencies, power, low, high):

    mask = (
        (frequencies >= low)
        & (frequencies <= high)
    )

    if np.count_nonzero(mask) < 2:
        return None

    return float(
        trapezoid(
            power[mask],
            frequencies[mask]
        )
    )


def extract_frequency_features(
    signal,
    sampling_rate,
    respiratory_band_hz=DEFAULT_RESPIRATORY_BAND_HZ,
    spectral_bands_hz=None,
    welch_window_s=16.0,
    stft_window_s=16.0,
    stft_hop_s=2.0,
):
    """
    Calculate respiratory frequency-domain features.

    Parameters
    ----------
    signal:
        Raw Channel 4 breathing waveform.

    sampling_rate:
        Actual acquisition sampling rate in Hz.

    respiratory_band_hz:
        Frequency range used for respiratory analysis.

    spectral_bands_hz:
        Dictionary of named frequency bands.

    welch_window_s:
        Welch segment duration in seconds.

    stft_window_s:
        STFT window duration in seconds.

    stft_hop_s:
        Time step between STFT windows in seconds.

    Returns
    -------
    FrequencyFeatureResult
    """

    if welch_window_s <= 0 or stft_window_s <= 0:
        raise ValueError("Window durations must be positive.")

    if stft_hop_s <= 0:
        raise ValueError("STFT hop duration must be positive.")

    x = _validate_signal(signal, sampling_rate)

    low, high = _validate_frequency_band(
        respiratory_band_hz,
        sampling_rate
    )

    # ---------------------------------------------------------
    # 1. Remove DC offset and linear baseline trend
    # ---------------------------------------------------------

    x_detrended = detrend(
        x,
        type="linear"
    )

    # ---------------------------------------------------------
    # 2. FFT - Dominant respiratory frequency
    # ---------------------------------------------------------

    n_samples = len(x_detrended)

    window = np.hanning(n_samples)

    fft_values = np.fft.rfft(
        x_detrended * window
    )

    fft_frequencies = np.fft.rfftfreq(
        n_samples,
        d=1.0 / sampling_rate
    )

    # Single-sided amplitude spectrum.
    fft_amplitude = (
        2.0 * np.abs(fft_values)
        / np.sum(window)
    )

    fft_amplitude[0] *= 0.5

    fft_mask = (
        (fft_frequencies >= low)
        & (fft_frequencies <= high)
    )

    if not np.any(fft_mask):
        raise ValueError(
            "No FFT frequency bins found in the respiratory band. "
            "The recording may be too short for the requested band."
        )

    band_fft_amplitude = fft_amplitude[fft_mask]

    band_fft_frequencies = fft_frequencies[fft_mask]

    dominant_index = np.argmax(band_fft_amplitude)

    dominant_frequency = float(
        band_fft_frequencies[dominant_index]
    )

    dominant_respiratory_rate = (
        dominant_frequency * 60.0
    )

    # Theoretical FFT frequency resolution.
    fft_resolution = sampling_rate / n_samples

    # ---------------------------------------------------------
    # 3. Welch PSD
    # ---------------------------------------------------------

    requested_nperseg = int(
        sampling_rate * welch_window_s
    )

    nperseg = min(
        n_samples,
        max(8, requested_nperseg)
    )

    psd_frequencies, psd_values = welch(
        x_detrended,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    psd_mask = (
        (psd_frequencies >= low)
        & (psd_frequencies <= high)
    )

    if np.count_nonzero(psd_mask) < 2:
        raise ValueError(
            "Insufficient Welch PSD frequency bins in the "
            "requested respiratory band. Increase the recording "
            "duration or adjust the frequency band."
        )

    band_frequencies = psd_frequencies[psd_mask]
    band_power_values = psd_values[psd_mask]

    # ---------------------------------------------------------
    # 4. Total respiratory band power
    # ---------------------------------------------------------

    respiratory_band_power = float(
        trapezoid(
            band_power_values,
            band_frequencies
        )
    )

    # ---------------------------------------------------------
    # 5. Spectral centroid
    # ---------------------------------------------------------

    total_power = np.sum(band_power_values)

    if total_power > 0:

        spectral_centroid = float(
            np.sum(
                band_frequencies * band_power_values
            ) / total_power
        )

    else:
        spectral_centroid = None

    # ---------------------------------------------------------
    # 6. Spectral entropy
    # ---------------------------------------------------------

    if total_power > 0:

        probability = (
            band_power_values / total_power
        )

        probability = probability[
            probability > 0
        ]

        if len(probability) > 1:

            entropy = -np.sum(
                probability * np.log(probability)
            )

            spectral_entropy = float(
                entropy / np.log(len(probability))
            )

        else:
            spectral_entropy = 0.0

    else:
        spectral_entropy = None

    # ---------------------------------------------------------
    # 7. Spectral band power
    # ---------------------------------------------------------

    if spectral_bands_hz is None:
        spectral_bands_hz = DEFAULT_SPECTRAL_BANDS

    spectral_band_power = {}

    for band_name, limits in spectral_bands_hz.items():

        band_low, band_high = _validate_frequency_band(
            limits,
            sampling_rate
        )

        # Keep band inside the requested respiratory range.
        band_low = max(band_low, low)
        band_high = min(band_high, high)

        if band_high <= band_low:
            spectral_band_power[band_name] = None
            continue

        spectral_band_power[band_name] = _integrate_band(
            psd_frequencies,
            psd_values,
            band_low,
            band_high,
        )

    # ---------------------------------------------------------
    # 8. STFT - Time-varying respiratory frequency
    # ---------------------------------------------------------

    stft_nperseg = min(
        n_samples,
        max(8, int(sampling_rate * stft_window_s))
    )

    hop_samples = max(
        1,
        int(round(sampling_rate * stft_hop_s))
    )

    stft_noverlap = max(
        0,
        stft_nperseg - hop_samples
    )

    stft_noverlap = min(
        stft_noverlap,
        stft_nperseg - 1
    )

    stft_frequencies, stft_times, stft_values = stft(
        x_detrended,
        fs=sampling_rate,
        window="hann",
        nperseg=stft_nperseg,
        noverlap=stft_noverlap,
        boundary=None,
        padded=False,
    )

    stft_magnitude = np.abs(stft_values)

    stft_band_mask = (
        (stft_frequencies >= low)
        & (stft_frequencies <= high)
    )

    frequencies_in_band = stft_frequencies[stft_band_mask]

    magnitude_in_band = stft_magnitude[
        stft_band_mask,
        :
    ]

    time_varying_frequency = np.full(
        len(stft_times),
        np.nan,
        dtype=float
    )

    if len(frequencies_in_band) > 0:

        for i in range(len(stft_times)):

            current_magnitude = magnitude_in_band[:, i]

            if np.max(current_magnitude) > 0:

                dominant_idx = np.argmax(current_magnitude)

                time_varying_frequency[i] = (
                    frequencies_in_band[dominant_idx]
                )

    # ---------------------------------------------------------
    # 9. Return all calculated features
    # ---------------------------------------------------------

    return FrequencyFeatureResult(

        dominant_frequency_hz=dominant_frequency,

        dominant_respiratory_rate_bpm=(
            dominant_respiratory_rate
        ),

        fft_resolution_hz=float(fft_resolution),

        respiratory_band_power=respiratory_band_power,

        spectral_centroid_hz=spectral_centroid,

        spectral_entropy=spectral_entropy,

        spectral_band_power=spectral_band_power,

        fft_frequencies_hz=fft_frequencies,

        fft_amplitude=fft_amplitude,

        psd_frequencies_hz=psd_frequencies,

        psd_values=psd_values,

        stft_times_s=stft_times,

        stft_frequencies_hz=stft_frequencies,

        stft_magnitude=stft_magnitude,

        time_varying_frequency_hz=time_varying_frequency,
    )
