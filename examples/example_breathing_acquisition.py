
import asyncio
from datetime import datetime
from pathlib import Path

from utils.serial_communication import ExampleUtils
from i24_breathing_sensor.acquisition import acquire_breathing_signal


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DURATION_S = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = DATA_DIR / f"breathing_ch4_{TIMESTAMP}.csv"


# --------------------------------------------------
# Acquisition
# --------------------------------------------------

async def main():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Use the existing ScienceMode4 utility for COM-port detection
    com_port = ExampleUtils.get_comport_from_commandline_argument()

    print("\nI24 Breathing Sensor Acquisition")
    print("--------------------------------")
    print(f"COM port:       {com_port}")
    print(f"Duration:       {DURATION_S} seconds")
    print(f"Output file:    {OUTPUT_FILE}")
    print("--------------------------------\n")

    sample_count = await acquire_breathing_signal(
        com_port=com_port,
        duration_s=DURATION_S,
        output_file=OUTPUT_FILE,
    )

    print("\nAcquisition completed.")
    print(f"Samples acquired: {sample_count}")
    print(f"Data saved to:    {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
