
"""Data models for I24 EMG preprocessing.

This module defines lightweight data structures for configuring
preprocessing operations and representing their results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for digital filtering of EMG signals."""

    sampling_rate_hz: float = 4000.0
    lowcut_hz: float = 20.0
    highcut_hz: float = 450.0
    notch_hz: float | None = 50.0
    filter_order: int = 4

    def __post_init__(self) -> None:
        """Validate filter configuration values."""
        if (
            not np.isfinite(self.sampling_rate_hz)
            or self.sampling_rate_hz <= 0
        ):
            raise ValueError(
                "sampling_rate_hz must be finite and positive."
            )

        nyquist = self.sampling_rate_hz / 2.0

        if (
            not np.isfinite(self.lowcut_hz)
            or self.lowcut_hz <= 0
            or self.lowcut_hz >= nyquist
        ):
            raise ValueError(
                "lowcut_hz must be positive and below Nyquist."
            )

        if (
            not np.isfinite(self.highcut_hz)
            or self.highcut_hz <= self.lowcut_hz
            or self.highcut_hz >= nyquist
        ):
            raise ValueError(
                "highcut_hz must exceed lowcut_hz and be below Nyquist."
            )

        if self.notch_hz is not None:
            if (
                not np.isfinite(self.notch_hz)
                or self.notch_hz <= 0
                or self.notch_hz >= nyquist
            ):
                raise ValueError(
                    "notch_hz must be positive and below Nyquist."
                )

        if (
            isinstance(self.filter_order, bool)
            or not isinstance(self.filter_order, int)
            or self.filter_order < 1
        ):
            raise ValueError(
                "filter_order must be a positive integer."
            )


@dataclass(frozen=True)
class EnvelopeConfig:
    """Configuration for EMG amplitude-envelope extraction."""

    method: str = "rms"
    window_ms: float = 50.0
    rectify: bool = True

    def __post_init__(self) -> None:
        """Validate envelope configuration values."""
        if self.method not in {"rms", "moving_average"}:
            raise ValueError(
                "method must be 'rms' or 'moving_average'."
            )

        if not np.isfinite(self.window_ms) or self.window_ms <= 0:
            raise ValueError(
                "window_ms must be finite and positive."
            )

        if not isinstance(self.rectify, bool):
            raise ValueError("rectify must be a boolean.")


@dataclass
class PreprocessingResult:
    """Result of preprocessing one EMG channel.

    Attributes:
        signal: Processed one-dimensional signal.
        sampling_rate_hz: Sampling rate in Hz.
        channel: Optional channel identifier, such as EMG1 or EMG2.
        stages: Names of preprocessing operations applied in order.
        metadata: Additional user-defined information.
    """

    signal: np.ndarray
    sampling_rate_hz: float
    channel: str | None = None
    stages: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the result's signal and sampling rate."""
        values = np.asarray(self.signal, dtype=float)

        if values.ndim != 1:
            raise ValueError(
                "Processed signal must be one-dimensional."
            )

        if values.size == 0:
            raise ValueError(
                "Processed signal must not be empty."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Processed signal must contain only finite values."
            )

        fs = float(self.sampling_rate_hz)

        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(
                "sampling_rate_hz must be finite and positive."
            )

        # Store a copy to avoid accidental modification through
        # the array originally passed to this object.
        self.signal = values.copy()
        self.sampling_rate_hz = fs
        self.stages = list(self.stages)
        self.metadata = dict(self.metadata)

    @property
    def sample_count(self) -> int:
        """Return the number of samples in the processed signal."""
        return int(self.signal.size)

    @property
    def duration_seconds(self) -> float:
        """Return sample-count duration in seconds."""
        return float(self.sample_count / self.sampling_rate_hz)
