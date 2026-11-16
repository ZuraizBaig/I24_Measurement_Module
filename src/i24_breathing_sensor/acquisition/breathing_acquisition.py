
"""
I24 Breathing Sensor Acquisition Library
========================================

Reusable acquisition function for the BRAEBON breathing sensor connected
through the HASOMED I24 and ScienceMode4 / Dyscom.

Configured signal order:
    Index 0: BI
    Index 1: EMG_1
    Index 2: EMG_2
    Index 3: BREATHING (logical Channel 4)

Only the fourth configured sample is written to CSV.

This module deliberately contains no command-line interface and does not
create filenames. The calling application or example supplies output_file.

No filtering, event detection, or feature extraction is performed here.
"""

import asyncio
from pathlib import Path
from time import perf_counter

from science_mode_4 import (
    Commands,
    DeviceI24,
    SerialPortConnection,
)

from science_mode_4.dyscom.ads129x.ads129x_config_register_1 import (
    Ads129xOutputDataRate,
    Ads129xPowerMode,
)

from science_mode_4.dyscom.dyscom_get_operation_mode import (
    PacketDyscomGetAckOperationMode,
)

from science_mode_4.dyscom.dyscom_send_live_data import (
    PacketDyscomSendLiveData,
)

from science_mode_4.dyscom.dyscom_types import (
    DyscomGetType,
    DyscomInitParams,
    DyscomPowerModulePowerType,
    DyscomPowerModuleType,
    DyscomSignalType,
)

from science_mode_4.protocol.types import ResultAndError
from science_mode_4.utils.logger import logger

from .csv_writer import BreathingCsvWriter


# ---------------------------------------------------------------------
# Acquisition configuration
# ---------------------------------------------------------------------

# Keep this order synchronized with the working I24/Dyscom configuration.
SIGNAL_CONFIGURATION = [
    DyscomSignalType.BI,
    DyscomSignalType.EMG_1,
    DyscomSignalType.EMG_2,
    DyscomSignalType.BREATHING,
]

# Python list index 3 corresponds to the fourth configured channel.
BREATHING_CHANNEL_INDEX = 3

OPERATION_MODE_POLL_INTERVAL_S = 1.0


async def acquire_breathing_signal(
    com_port: str,
    duration_s: float,
    output_file: str | Path,
) -> int:
    """Acquire raw breathing samples from I24 Channel 4 and save them to CSV.

    Parameters
    ----------
    com_port:
        Serial port connected to the I24, e.g. "COM5".

    duration_s:
        Requested acquisition duration in seconds. Must be positive.

    output_file:
        Explicit CSV destination supplied by the caller.

    Returns
    -------
    int
        Number of Channel 4 samples written to the CSV.

    Notes
    -----
    The function does not generate an output filename. Filename generation
    belongs in an example or in the application using this library.
    """

    if duration_s <= 0:
        raise ValueError("duration_s must be greater than zero.")

    if output_file is None:
        raise ValueError("output_file must be provided by the caller.")

    output_path = Path(output_file)

    connection = SerialPortConnection(com_port)
    device = None
    dyscom = None
    csv_writer = None

    connection_open = False
    measurement_powered = False
    measurement_started = False

    sample_count = 0
    start_time = None
    acquisition_error = None

    # Preserve and restore the ScienceMode4 logger state.
    sm4_logger = logger()
    previous_logger_state = sm4_logger.disabled
    sm4_logger.disabled = True

    try:
        # -------------------------------------------------------------
        # 1. Connect to I24
        # -------------------------------------------------------------

        print(f"\nConnecting to I24 through {com_port}...")

        connection.open()
        connection_open = True

        device = DeviceI24(connection)

        await device.initialize()

        print("I24 initialized successfully.")

        # -------------------------------------------------------------
        # 2. Initialize Dyscom and power the measurement module
        # -------------------------------------------------------------

        dyscom = device.get_layer_dyscom()

        print("Powering on measurement module...")

        await dyscom.power_module(
            DyscomPowerModuleType.MEASUREMENT,
            DyscomPowerModulePowerType.SWITCH_ON,
        )

        measurement_powered = True

        # -------------------------------------------------------------
        # 3. Configure BI, EMG1, EMG2 and BREATHING
        # -------------------------------------------------------------

        init_params = DyscomInitParams()

        init_params.signal_type = SIGNAL_CONFIGURATION.copy()

        init_params.register_map_ads129x.config_register_1.output_data_rate = (
            Ads129xOutputDataRate.HR_MODE_4_KSPS__LP_MODE_2_KSPS
        )

        init_params.register_map_ads129x.config_register_1.power_mode = (
            Ads129xPowerMode.HIGH_RESOLUTION
        )

        print("Initializing BI, EMG1, EMG2 and BREATHING...")

        await dyscom.init(init_params)

        # -------------------------------------------------------------
        # 4. Prepare CSV writer
        # -------------------------------------------------------------

        csv_writer = BreathingCsvWriter(
            filename=str(output_path),
            headers=[
                "sample_index",
                "packet_number",
                "device_time_offset_raw",
                "host_elapsed_s",
                "breathing_ch4_raw",
                "breathing_ch4_status",
            ],
        )

        csv_writer.start()

        # -------------------------------------------------------------
        # 5. Start measurement
        # -------------------------------------------------------------

        print("Starting I24 measurement...")

        await dyscom.start()

        measurement_started = True

        print("\n---------------------------------------")
        print("I24 BREATHING ACQUISITION STARTED")
        print("---------------------------------------")
        print(f"COM port: {com_port}")
        print(f"Duration: {duration_s:.2f} seconds")
        print("Acquisition channel: Channel 4 (BREATHING)")
        print(f"Output: {output_path}")
        print("---------------------------------------\n")

        start_time = perf_counter()

        next_operation_poll = (
            start_time + OPERATION_MODE_POLL_INTERVAL_S
        )

        # -------------------------------------------------------------
        # 6. Receive live packets and extract Channel 4
        # -------------------------------------------------------------

        while (perf_counter() - start_time) < duration_s:

            now = perf_counter()

            if now >= next_operation_poll:

                dyscom.send_get_operation_mode()

                next_operation_poll = (
                    now + OPERATION_MODE_POLL_INTERVAL_S
                )

            live_data_counter = 0

            while True:

                # First call waits for a packet.
                # Subsequent calls drain the packet buffer.
                ack = dyscom.packet_buffer.get_packet_from_buffer(
                    live_data_counter == 0
                )

                if ack is None:
                    break

                # -----------------------------------------------------
                # Device operation-mode acknowledgement
                # -----------------------------------------------------

                if (
                    ack.command == Commands.DL_GET_ACK
                    and ack.kind == DyscomGetType.OPERATION_MODE
                ):

                    om_ack: PacketDyscomGetAckOperationMode = ack

                    if om_ack.result_error != ResultAndError.NO_ERROR:

                        raise RuntimeError(
                            "I24 operation-mode request failed: "
                            f"{om_ack.result_error}"
                        )

                # -----------------------------------------------------
                # Live measurement data
                # -----------------------------------------------------

                elif ack.command == Commands.DL_SEND_LIVE_DATA:

                    live_data_counter += 1

                    sld: PacketDyscomSendLiveData = ack

                    # Validate packet structure.
                    if len(sld.samples) <= BREATHING_CHANNEL_INDEX:

                        raise RuntimeError(
                            "Live-data packet does not contain the "
                            "configured Channel 4 sample. "
                            f"Received {len(sld.samples)} samples."
                        )

                    # Extract only Channel 4.
                    breathing_sample = sld.samples[
                        BREATHING_CHANNEL_INDEX
                    ]

                    breathing_value = breathing_sample.value

                    # Preserve status flags.
                    status_flags = getattr(
                        breathing_sample,
                        "status",
                        []
                    )

                    breathing_status = ";".join(
                        sorted(
                            getattr(flag, "name", str(flag))
                            for flag in status_flags
                        )
                    )

                    # Host-side packet reception timestamp.
                    host_elapsed_s = perf_counter() - start_time

                    # Write one respiratory sample.
                    csv_writer.write(
                        [
                            sample_count,
                            ack.number,
                            sld.time_offset,
                            f"{host_elapsed_s:.9f}",
                            repr(breathing_value),
                            breathing_status,
                        ]
                    )

                    sample_count += 1

        # -------------------------------------------------------------
        # 7. Acquisition summary
        # -------------------------------------------------------------

        elapsed_s = (
            perf_counter() - start_time
            if start_time is not None
            else 0.0
        )

        measured_packet_rate = (
            sample_count / elapsed_s
            if elapsed_s > 0
            else 0.0
        )

        print("\n---------------------------------------")
        print("ACQUISITION FINISHED")
        print("---------------------------------------")
        print(f"Elapsed time: {elapsed_s:.3f} s")
        print(f"Channel 4 samples written: {sample_count}")
        print(
            f"Received packet rate: "
            f"{measured_packet_rate:.2f} packets/s"
        )
        print(f"CSV destination: {output_path}")
        print("---------------------------------------")

    except BaseException as exc:

        acquisition_error = exc
        raise

    finally:

        # -------------------------------------------------------------
        # 8. Cleanup
        # -------------------------------------------------------------

        # Stop measurement before powering down.
        if measurement_started and dyscom is not None:

            try:
                await dyscom.stop()
                print("Measurement stopped.")

            except Exception as exc:
                print(
                    f"Warning while stopping measurement: {exc}"
                )

        # Power off measurement module.
        if measurement_powered and dyscom is not None:

            try:
                await dyscom.power_module(
                    DyscomPowerModuleType.MEASUREMENT,
                    DyscomPowerModulePowerType.SWITCH_OFF,
                )

                print("Measurement module powered off.")

            except Exception as exc:
                print(
                    f"Warning while powering off measurement module: {exc}"
                )

        # Flush pending CSV samples.
        if csv_writer is not None:

            try:
                csv_writer.stop()
                print(f"CSV writer closed: {output_path}")

            except Exception as exc:

                if acquisition_error is None:
                    raise

                print(
                    f"Warning while closing CSV writer: {exc}"
                )

        # Close I24 connection.
        if connection_open:

            try:
                connection.close()
                print("I24 connection closed.")

            except Exception as exc:
                print(
                    f"Warning while closing I24 connection: {exc}"
                )

        # Restore original logger state.
        sm4_logger.disabled = previous_logger_state

    return sample_count
