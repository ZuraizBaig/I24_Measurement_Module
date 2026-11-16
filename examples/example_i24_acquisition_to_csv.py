"""
Example: Acquire EMG signals from the HASOMED I24
and save them to a CSV file.

The script records both EMG1 and EMG2 for a selected
duration and saves the samples with timestamps.
"""

import asyncio
from pathlib import Path

from science_mode_4 import DeviceI24, SerialPortConnection
from science_mode_4.dyscom.dyscom_types import DyscomPowerModuleType

from i24_emg.acquisition.backend import DyscomBackend
from i24_emg.acquisition.config import I24AcquisitionConfig
from i24_emg.acquisition.i24_acquisition import I24Acquisition
from i24_emg.emg_tools.emg_tools import I24EMGTools


# --------------------------------------------------
# Acquisition parameters
# --------------------------------------------------

COM_PORT = "COM5"       # Change this to your I24 COM port
DURATION_SECONDS = 10    # Recording duration in seconds
BUFFER_SIZE = 10_000    # Maximum frames retained by the library

PROJECT_ROOT = Path(__file__).resolve().parents[1]
'''
# Save the CSV in the examples/output directory.
OUTPUT_FILE = (
    Path(__file__).parent
    / "outputs"
    / "i24_emg_recording.csv"
)
'''
OUTPUT_FILE = PROJECT_ROOT / "output" / "acquisitions/i24_emg_recording.csv"


async def main():
    connection = None
    backend = None
    acquisition = None

    try:
        # 1. Open the serial connection to the I24.
        connection = SerialPortConnection(COM_PORT)
        connection.open()
        print(f"Connected to {COM_PORT}")

        # 2. Initialize the I24 device.
        device = DeviceI24(connection)
        await device.initialize()

        # 3. Access the device's Dyscom communication layer.
        dyscom = device.get_layer_dyscom()

        # 4. Create the backend used by our acquisition library.
        backend = DyscomBackend(dyscom)

        # 5. Power on the measurement module.
        await backend.power_on(
            DyscomPowerModuleType.MEASUREMENT
        )

        # 6. Configure the I24 acquisition.
        # The library uses its fixed I24 signal configuration.
        config = I24AcquisitionConfig(
            buffer_size=BUFFER_SIZE,
        )

        # 7. Create and initialize the acquisition interface.
        acquisition = I24Acquisition(
            backend=backend,
            config=config,
        )
        await acquisition.initialize()

        # Read the sampling rate reported by the device.
        sampling_rate = (
            acquisition.sampling_info.sampling_rate_hz
        )
        print(f"Sampling rate: {sampling_rate} Hz")

        # 8. Start live acquisition.
        await acquisition.start()
        print("Acquisition started.")

        # 9. Create the EMG tools interface.
        tools = I24EMGTools(acquisition)

        # 10. Record both channels and save them to CSV.
        data = await tools.record_and_save_csv(
            duration=DURATION_SECONDS,
            filename=OUTPUT_FILE,
            channel="both",
        )

        print("\nRecording completed.")
        print(f"EMG1 samples: {len(data['EMG1'])}")
        print(f"EMG2 samples: {len(data['EMG2'])}")
        print(f"CSV saved to: {OUTPUT_FILE.resolve()}")

    finally:
        # 11. Stop acquisition and release hardware resources.
        if acquisition is not None:
            try:
                await acquisition.stop()
            except Exception as exc:
                print(f"Warning while stopping acquisition: {exc}")

        if backend is not None:
            try:
                await backend.power_off(
                    DyscomPowerModuleType.MEASUREMENT
                )
            except Exception as exc:
                print(f"Warning while powering off: {exc}")

        if connection is not None:
            try:
                connection.close()
            except Exception as exc:
                print(f"Warning while closing connection: {exc}")

        print("I24 cleanup completed.")


if __name__ == "__main__":
    asyncio.run(main())