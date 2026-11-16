from __future__ import annotations

from dataclasses import dataclass

from science_mode_4.dyscom.dyscom_init import DyscomInitParams
from science_mode_4.dyscom.dyscom_types import DyscomSignalType

from .models import I24Channel


# =========================================================
# Fixed I24 signal configuration
# =========================================================
#
# This library is specifically designed for the HASOMED I24.
#
# The I24 signal configuration used by this acquisition
# module is fixed:
#
#   Channel 1 -> BI
#   Channel 2 -> EMG_1
#   Channel 3 -> EMG_2
#   Channel 4 -> BREATHING
#
# The order below must not be changed.
#


I24_SIGNAL_TYPES: tuple[DyscomSignalType, ...] = (
    DyscomSignalType.BI,
    DyscomSignalType.EMG_1,
    DyscomSignalType.EMG_2,
    DyscomSignalType.BREATHING,
)


# Mapping between our public I24 EMG channels and the
# corresponding fixed ScienceMode signal types.

CHANNEL_TO_SIGNAL_TYPE = {
    I24Channel.EMG1: DyscomSignalType.EMG_1,
    I24Channel.EMG2: DyscomSignalType.EMG_2,
}


@dataclass
class I24AcquisitionConfig:
    """
    Fixed configuration for I24 EMG acquisition.

    The I24 signal configuration is controlled by this
    library and cannot be reordered through the public
    configuration.

    enabled_channels determines which of the two fixed
    I24 EMG channels are exposed by the acquisition layer.
    """

    buffer_size: int = 10_000

    enabled_channels: tuple[I24Channel, ...] = (
        I24Channel.EMG1,
        I24Channel.EMG2,
    )

    def __post_init__(self) -> None:

        # --------------------------------------------------------------
        # Basic validation
        # --------------------------------------------------------------

        if self.buffer_size <= 0:
            raise ValueError(
                "buffer_size must be greater than zero."
            )

        if not self.enabled_channels:
            raise ValueError(
                "At least one EMG channel must be enabled."
            )

        # --------------------------------------------------------------
        # Validate supported channels
        # --------------------------------------------------------------

        unsupported = [
            channel
            for channel in self.enabled_channels
            if channel not in CHANNEL_TO_SIGNAL_TYPE
        ]

        if unsupported:
            raise ValueError(
                "Unsupported I24 acquisition channel(s): "
                + ", ".join(
                    channel.value
                    for channel in unsupported
                )
            )

        # --------------------------------------------------------------
        # Prevent duplicate channels
        # --------------------------------------------------------------

        if len(set(self.enabled_channels)) != len(
            self.enabled_channels
        ):
            raise ValueError(
                "enabled_channels must not contain duplicates."
            )

    @property
    def init_params(self) -> DyscomInitParams:
        """
        Create the fixed I24 DL_INIT configuration.

        The signal order is intentionally fixed:

            BI
            EMG_1
            EMG_2
            BREATHING

        This prevents external code from changing the
        I24 channel/signal mapping.
        """

        params = DyscomInitParams()

        params.signal_type = list(
            I24_SIGNAL_TYPES
        )

        return params

    @property
    def configured_signal_types(
        self,
    ) -> tuple[DyscomSignalType, ...]:
        """
        Return the fixed signal configuration used by
        this I24 acquisition module.
        """

        return I24_SIGNAL_TYPES

    @property
    def enabled_signal_types(
        self,
    ) -> tuple[DyscomSignalType, ...]:
        """
        Return the ScienceMode signal types corresponding
        to the enabled application EMG channels.
        """

        return tuple(
            CHANNEL_TO_SIGNAL_TYPE[channel]
            for channel in self.enabled_channels
        )