from __future__ import annotations

from .backend import DyscomBackend
from .buffer import EMGBuffer
from .channel import (
    PACKET_TO_CHANNEL,
    get_channel,
)
from .config import I24AcquisitionConfig
from .models import (
    ChannelSample,
    EMGFrame,
    SamplingInfo,
)


class I24Acquisition:
    """
    High-level EMG acquisition interface for the HASOMED I24.

    ScienceMode4/Dyscom performs the actual hardware
    communication.

    This class converts the raw I24 live-data packets into
    our own I24-specific EMG models.

    Fixed I24 configuration:

        Channel 1 -> BI
        Channel 2 -> EMG1
        Channel 3 -> EMG2
        Channel 4 -> BREATHING

    Fixed I24 live-packet mapping:

        packet index 2 -> EMG1
        packet index 3 -> EMG2

    This mapping is intentionally I24-specific.
    """

    def __init__(
        self,
        backend: DyscomBackend,
        config: I24AcquisitionConfig,
    ) -> None:

        self._backend = backend
        self._config = config

        self._buffer = EMGBuffer(
            max_size=config.buffer_size
        )

        self._sampling: SamplingInfo | None = None

        self._running = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize the I24 through the Dyscom backend.

        The configuration is generated internally by
        I24AcquisitionConfig and therefore always uses the
        fixed I24 signal order.
        """

        await self._backend.initialize(
            self._config.init_params
        )

        sampling_rate = (
            self._backend.sampling_rate_hz
        )

        if sampling_rate is None:
            raise RuntimeError(
                "I24 sampling rate is not available."
            )

        self._sampling = SamplingInfo(
            sampling_rate_hz=sampling_rate
        )

    # ------------------------------------------------------------------
    # Acquisition control
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start I24 live acquisition.
        """

        if self._sampling is None:
            raise RuntimeError(
                "I24 must be initialized before starting."
            )

        if self._running:
            return

        await self._backend.start()

        self._running = True

    async def stop(self) -> None:
        """
        Stop I24 live acquisition.
        """

        if not self._running:
            return

        await self._backend.stop()

        self._running = False

    # ------------------------------------------------------------------
    # Live data
    # ------------------------------------------------------------------

    async def read_frame(
        self,
        timeout: float = 1.0,
    ) -> EMGFrame:
        """
        Read one I24 EMG frame.

        Fixed I24 mapping:

            packet index 2 -> EMG1
            packet index 3 -> EMG2

        Only these two channels are exposed by this
        acquisition layer.
        """

        if not self._running:
            raise RuntimeError(
                "Acquisition is not running."
            )

        packet = await self._backend.get_live_packet(
            timeout
        )

        samples = []

        # --------------------------------------------------------------
        # Fixed I24 EMG packet mapping
        # --------------------------------------------------------------

        for packet_index in PACKET_TO_CHANNEL:

            if packet_index >= len(
                packet.samples
            ):
                raise ValueError(
                    f"I24 live packet does not contain "
                    f"packet index {packet_index}. "
                    f"Packet contains "
                    f"{len(packet.samples)} samples."
                )

            packet_sample = (
                packet.samples[packet_index]
            )

            channel = get_channel(
                packet_index
            )

            if channel is None:
                raise RuntimeError(
                    f"I24 packet index "
                    f"{packet_index} has no EMG "
                    f"channel mapping."
                )

            samples.append(
                ChannelSample(
                    channel=channel,
                    value=packet_sample.value,
                    packet_index=packet_index,
                    status=frozenset(
                        packet_sample.status
                    ),
                    signal_type=(
                        "EMG_1"
                        if channel.value == "EMG1"
                        else "EMG_2"
                    ),
                )
            )

        # --------------------------------------------------------------
        # Build our I24 EMG frame
        # --------------------------------------------------------------

        frame = EMGFrame(
            samples=tuple(samples),
            time_offset=packet.time_offset,
        )

        # Store frame in acquisition buffer.
        self._buffer.append(frame)

        return frame

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    @property
    def sampling_info(self) -> SamplingInfo:
        """
        Return I24 sampling information.
        """

        if self._sampling is None:
            raise RuntimeError(
                "I24 has not been initialized."
            )

        return self._sampling

    @property
    def buffer(self) -> EMGBuffer:
        """
        Access the I24 acquisition buffer.
        """

        return self._buffer

    @property
    def is_running(self) -> bool:
        """
        Return whether I24 live acquisition is active.
        """

        return self._running