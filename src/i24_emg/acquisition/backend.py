from __future__ import annotations

import asyncio
from typing import Optional

from science_mode_4.dyscom.dyscom_init import (
    DyscomInitParams,
    DyscomInitResult,
)
from science_mode_4.dyscom.dyscom_layer import LayerDyscom
from science_mode_4.dyscom.dyscom_send_live_data import (
    PacketDyscomSendLiveData,
)
from science_mode_4.dyscom.dyscom_types import (
    DyscomFrequencyOut,
    DyscomPowerModulePowerType,
    DyscomPowerModuleType,
)
from science_mode_4.protocol.commands import Commands


class DyscomBackend:
    """
    Adapter between the ScienceMode4 Dyscom layer and the
    I24 EMG library.

    ScienceMode4 handles the actual hardware communication
    and protocol.

    This class intentionally remains signal-agnostic. It does
    not interpret EMG_1, EMG_2, BI, breathing, or ADS129x
    physical channels.

    Responsibilities:
        - I24 module power control
        - Dyscom initialization
        - acquisition start/stop
        - retrieval of raw live-data packets
        - exposing initialization information
        - exposing the device-reported sampling frequency
        - exposing the ADS129x register map

    Signal interpretation belongs to I24Acquisition.
    """

    _FREQUENCY_MAP: dict[DyscomFrequencyOut, int] = {
        DyscomFrequencyOut.SAMPLES_PER_SECOND_32K: 32_000,
        DyscomFrequencyOut.SAMPLES_PER_SECOND_16K: 16_000,
        DyscomFrequencyOut.SAMPLES_PER_SECOND_8K: 8_000,
        DyscomFrequencyOut.SAMPLES_PER_SECOND_4K: 4_000,
        DyscomFrequencyOut.SAMPLES_PER_SECOND_2K: 2_000,
        DyscomFrequencyOut.SAMPLES_PER_SECOND_1K: 1_000,
        DyscomFrequencyOut.SAMPLES_PER_SECOND_500: 500,
        DyscomFrequencyOut.SAMPLES_PER_SECOND_250: 250,
    }

    def __init__(self, dyscom: LayerDyscom) -> None:
        self._dyscom = dyscom

        self._init_result: Optional[DyscomInitResult] = None

        self._running = False

    # =========================================================
    # Power
    # =========================================================

    async def power_on(
        self,
        module: DyscomPowerModuleType,
    ) -> None:
        """
        Switch on an I24 measurement module.
        """

        await self._dyscom.power_module(
            module,
            DyscomPowerModulePowerType.SWITCH_ON,
        )

    async def power_off(
        self,
        module: DyscomPowerModuleType,
    ) -> None:
        """
        Switch off an I24 measurement module.
        """

        await self._dyscom.power_module(
            module,
            DyscomPowerModulePowerType.SWITCH_OFF,
        )

    # =========================================================
    # Initialization
    # =========================================================

    async def initialize(
        self,
        params: DyscomInitParams,
    ) -> DyscomInitResult:
        """
        Initialize the I24 using the supplied Dyscom parameters.

        DyscomInitParams contains the configuration sent to the
        I24, including requested signal types and the associated
        Dyscom/ADS129x configuration.

        The returned DyscomInitResult contains information
        reported by the device, including:

            - ADS129x register map
            - measurement file ID
            - initialization state
            - output sampling frequency
        """

        if self._running:
            raise RuntimeError(
                "Cannot initialize I24 while acquisition is running."
            )

        self._init_result = await self._dyscom.init(params)

        return self._init_result

    # =========================================================
    # Acquisition control
    # =========================================================

    async def start(self) -> None:
        """
        Start I24 live acquisition.
        """

        if self._init_result is None:
            raise RuntimeError(
                "Dyscom must be initialized before acquisition can start."
            )

        if self._running:
            return

        await self._dyscom.start()

        self._running = True

    async def stop(self) -> None:
        """
        Stop I24 live acquisition.
        """

        if not self._running:
            return

        await self._dyscom.stop()

        self._running = False

    # =========================================================
    # Live data
    # =========================================================

    async def get_live_packet(
        self,
        timeout: float = 1.0,
    ) -> PacketDyscomSendLiveData:
        """
        Wait for and return the next raw I24
        DL_SEND_LIVE_DATA packet.

        This method does NOT interpret the packet.

        In particular, it does not decide which packet sample
        represents EMG_1, EMG_2, BI, or any other signal.

        That interpretation belongs to I24Acquisition.
        """

        if not self._running:
            raise RuntimeError(
                "Live acquisition is not running."
            )

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        loop = asyncio.get_running_loop()

        deadline = loop.time() + timeout

        while True:
            packet = (
                self._dyscom
                .packet_buffer
                .get_packet_from_buffer()
            )

            if packet is not None:

                if (
                    packet.command == Commands.DL_SEND_LIVE_DATA
                    and isinstance(
                        packet,
                        PacketDyscomSendLiveData,
                    )
                ):
                    return packet

            remaining = deadline - loop.time()

            if remaining <= 0:
                break

            await asyncio.sleep(
                min(0.001, remaining)
            )

        raise TimeoutError(
            "No I24 live-data packet received within "
            f"{timeout:.3f} seconds."
        )

    # =========================================================
    # Initialization information
    # =========================================================

    @property
    def init_result(
        self,
    ) -> Optional[DyscomInitResult]:
        """
        Return the result of the most recent I24 initialization.
        """

        return self._init_result

    @property
    def frequency_out(
        self,
    ) -> Optional[DyscomFrequencyOut]:
        """
        Return the frequency enum reported by the I24.
        """

        if self._init_result is None:
            return None

        return self._init_result.frequency_out

    @property
    def sampling_rate_hz(
        self,
    ) -> Optional[int]:
        """
        Return the device-reported sampling rate in Hz.

        This value comes from the Dyscom initialization ACK.
        """

        frequency = self.frequency_out

        if frequency is None:
            return None

        try:
            return self._FREQUENCY_MAP[frequency]

        except KeyError as exc:
            raise ValueError(
                f"Unsupported I24 frequency: {frequency!r}"
            ) from exc

    @property
    def ads129x(self):
        """
        Return the ADS129x register map returned by
        I24 initialization.
        """

        if self._init_result is None:
            return None

        return self._init_result.register_map_ads129x

    @property
    def measurement_file_id(
        self,
    ) -> Optional[str]:
        """
        Return the measurement file ID reported by the I24.
        """

        if self._init_result is None:
            return None

        return self._init_result.measurement_file_id

    @property
    def is_running(self) -> bool:
        """
        Return True when live acquisition is active.
        """

        return self._running