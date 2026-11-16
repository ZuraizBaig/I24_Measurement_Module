"""
Data models for I24 respiratory event detection.
"""

from dataclasses import dataclass, field
from typing import Literal


EventType = Literal["positive_peak", "negative_trough"]


@dataclass(frozen=True)
class DetectedEvent:
    """Represents one detected respiratory waveform landmark."""

    event_type: EventType
    sample_index: int
    time_s: float
    amplitude: float
    prominence: float


@dataclass
class RespiratoryCycle:
    """Represents a candidate trough-to-trough respiratory cycle."""

    cycle_index: int

    start_trough: DetectedEvent
    positive_peak: DetectedEvent | None
    end_trough: DetectedEvent

    duration_s: float
    phase_1_duration_s: float | None
    phase_2_duration_s: float | None

    peak_to_peak_amplitude: float

    is_valid: bool = False
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class EventDetectionResult:
    """Complete output of the event detection pipeline."""

    positive_peaks: list[DetectedEvent]
    negative_troughs: list[DetectedEvent]
    cycles: list[RespiratoryCycle]

    @property
    def valid_cycles(self) -> list[RespiratoryCycle]:
        return [cycle for cycle in self.cycles if cycle.is_valid]

    @property
    def rejected_cycles(self) -> list[RespiratoryCycle]:
        return [cycle for cycle in self.cycles if not cycle.is_valid]
