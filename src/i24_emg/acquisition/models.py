from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class I24Channel(str, Enum):
    """
    Logical EMG channels exposed by the I24 acquisition module.

    These names represent the logical channels used by the
    acquisition library. They are not ADS1294R channel numbers.
    """

    EMG1 = "EMG1"
    EMG2 = "EMG2"


@dataclass(frozen=True)
class ChannelSample:
    """
    One decoded EMG sample from an I24 live-data frame.

    This is the library-level representation and is intentionally
    independent of ScienceMode4 classes.
    """

    channel: I24Channel
    value: float

    # Original entry position inside the I24 live-data packet.
    #
    # For the current I24 configuration:
    #   1 -> EMG1
    #   2 -> EMG2
    packet_index: int

    # Status information from the ScienceMode4 packet.
    # Kept generic so this layer does not depend directly on
    # ScienceMode4's status-flag enum.
    status: frozenset[Any] = field(
        default_factory=frozenset
    )

    # ScienceMode4 signal type, when available.
    #
    # Note:
    # The observed I24 live-data packets report UNUSED here,
    # so this field should not be used to determine the
    # logical EMG channel.
    signal_type: int | None = None


@dataclass(frozen=True)
class EMGFrame:
    """
    One simultaneous EMG acquisition frame from the I24.

    A frame represents the values contained in one
    DL_SEND_LIVE_DATA packet.

    EMG1 and EMG2 are therefore parallel measurements
    belonging to the same acquisition frame.
    """

    samples: tuple[ChannelSample, ...]

    # Raw protocol timing field.
    # It is preserved but not interpreted by this model.
    time_offset: int | None = None

    @property
    def emg_samples(
        self,
    ) -> tuple[ChannelSample, ...]:
        """
        Return all EMG samples contained in this frame.
        """

        return self.samples

    def get_channel(
        self,
        channel: I24Channel,
    ) -> ChannelSample | None:
        """
        Return the sample belonging to a specific EMG channel.
        """

        for sample in self.samples:
            if sample.channel == channel:
                return sample

        return None

    @property
    def emg1(self) -> ChannelSample | None:
        """Return the EMG1 sample."""

        return self.get_channel(I24Channel.EMG1)

    @property
    def emg2(self) -> ChannelSample | None:
        """Return the EMG2 sample."""

        return self.get_channel(I24Channel.EMG2)


@dataclass
class SamplingInfo:
    """
    Runtime information about the I24 acquisition.
    """

    sampling_rate_hz: float

    # Number of live-data frames acquired.
    number_of_samples: int = 0

    @property
    def duration_seconds(self) -> float:
        """
        Calculate acquisition duration from the number of
        acquired frames and the device sampling rate.
        """

        if self.sampling_rate_hz <= 0:
            raise ValueError(
                "Sampling rate must be greater than zero."
            )

        return (
            self.number_of_samples
            / self.sampling_rate_hz
        )