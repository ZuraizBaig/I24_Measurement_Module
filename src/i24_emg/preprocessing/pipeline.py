
"""Preprocessing pipeline for I24 EMG signals."""

from __future__ import annotations

import numpy as np

from .filters import (
    bandpass_filter,
    notch_filter,
    detrend_signal,
    rms_envelope,
    moving_average_envelope,
    detect_artifacts,
    signal_quality_metrics,
)
from .models import (
    FilterConfig,
    EnvelopeConfig,
    PreprocessingResult,
)


class EMGPreprocessingPipeline:
    """Configurable preprocessing pipeline for a single EMG channel.

    The default workflow is:
        1. Remove the constant or linear trend.
        2. Apply a notch filter if configured.
        3. Apply a band-pass filter.
        4. Optionally calculate an amplitude envelope.

    The input signal is never modified.
    """

    def __init__(
        self,
        filter_config: FilterConfig | None = None,
        envelope_config: EnvelopeConfig | None = None,
        *,
        detrend_method: str | None = None,
        apply_notch: bool = True,
        apply_bandpass: bool = True,
        calculate_envelope: bool = False,
        channel: str | None = None,
    ) -> None:
        self.filter_config = filter_config or FilterConfig()
        self.envelope_config = envelope_config or EnvelopeConfig()

        if detrend_method not in {None, "constant", "linear"}:
            raise ValueError(
                "detrend_method must be None, 'constant', or 'linear'."
            )

        if not isinstance(apply_notch, bool):
            raise ValueError("apply_notch must be a boolean.")

        if not isinstance(apply_bandpass, bool):
            raise ValueError("apply_bandpass must be a boolean.")

        if not isinstance(calculate_envelope, bool):
            raise ValueError("calculate_envelope must be a boolean.")

        if apply_notch and self.filter_config.notch_hz is None:
            raise ValueError(
                "A notch frequency is required when apply_notch=True."
            )

        self.detrend_method = detrend_method
        self.apply_notch = apply_notch
        self.apply_bandpass = apply_bandpass
        self.calculate_envelope = calculate_envelope
        self.channel = channel

    def process(
        self,
        signal: np.ndarray,
        *,
        channel: str | None = None,
        metadata: dict | None = None,
    ) -> PreprocessingResult:
        """Process one EMG signal and return a structured result."""
        values = np.asarray(signal, dtype=float)

        if values.ndim != 1:
            raise ValueError("Signal must be one-dimensional.")

        if values.size == 0:
            raise ValueError("Signal must not be empty.")

        if not np.all(np.isfinite(values)):
            raise ValueError("Signal must contain only finite values.")

        # Work on a copy so the caller's data is not modified.
        processed = values.copy()
        stages: list[str] = []
        fs = self.filter_config.sampling_rate_hz

        if self.detrend_method is not None:
            processed = detrend_signal(
                processed,
                method=self.detrend_method,
            )
            stages.append(f"detrend_{self.detrend_method}")

        if self.apply_notch:
            processed = notch_filter(
                processed,
                sampling_rate_hz=fs,
                notch_hz=self.filter_config.notch_hz,
            )
            stages.append(
                f"notch_{self.filter_config.notch_hz:g}Hz"
            )

        if self.apply_bandpass:
            processed = bandpass_filter(
                processed,
                sampling_rate_hz=fs,
                lowcut_hz=self.filter_config.lowcut_hz,
                highcut_hz=self.filter_config.highcut_hz,
                order=self.filter_config.filter_order,
            )
            stages.append(
                "bandpass_"
                f"{self.filter_config.lowcut_hz:g}_"
                f"{self.filter_config.highcut_hz:g}Hz"
            )

        if self.calculate_envelope:
            if self.envelope_config.method == "rms":
                processed = rms_envelope(
                    processed,
                    sampling_rate_hz=fs,
                    window_ms=self.envelope_config.window_ms,
                )
                stages.append(
                    f"rms_envelope_{self.envelope_config.window_ms:g}ms"
                )
            else:
                processed = moving_average_envelope(
                    processed,
                    sampling_rate_hz=fs,
                    window_ms=self.envelope_config.window_ms,
                    rectify=self.envelope_config.rectify,
                )
                stages.append(
                    "moving_average_envelope_"
                    f"{self.envelope_config.window_ms:g}ms"
                )

        selected_channel = (
            channel if channel is not None else self.channel
        )

        return PreprocessingResult(
            signal=processed,
            sampling_rate_hz=fs,
            channel=selected_channel,
            stages=stages,
            metadata={} if metadata is None else metadata,
        )

    def quality_report(
        self,
        signal: np.ndarray,
        *,
        clipping_threshold: float | None = None,
        flat_tolerance: float = 0.0,
    ) -> dict:
        """Calculate descriptive quality metrics for a raw signal."""
        return signal_quality_metrics(
            signal,
            sampling_rate_hz=self.filter_config.sampling_rate_hz,
            clipping_threshold=clipping_threshold,
            flat_tolerance=flat_tolerance,
        )

    def artifact_mask(
        self,
        signal: np.ndarray,
        *,
        threshold_sigma: float = 6.0,
        method: str = "mad",
        absolute_threshold: float | None = None,
    ) -> np.ndarray:
        """Return a Boolean mask for potentially abnormal raw samples."""
        return detect_artifacts(
            signal,
            threshold_sigma=threshold_sigma,
            method=method,
            absolute_threshold=absolute_threshold,
        )
