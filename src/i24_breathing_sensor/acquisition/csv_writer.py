
"""
Asynchronous CSV writer for I24 breathing acquisition.

CSV writing is performed in a background thread to avoid
blocking the measurement acquisition loop.
"""

import csv
import threading

from pathlib import Path
from queue import Queue


class BreathingCsvWriter:
    """Background CSV writer for respiratory samples."""

    _STOP = object()

    def __init__(self, filename: str, headers: list[str]) -> None:
        self.filename = Path(filename)
        self.headers = headers

        self._queue = Queue()
        self._thread = None
        self._ready = threading.Event()

        self._error = None
        self._started = False

    def start(self) -> None:
        """Create the output file and start the writer thread."""

        if self._started:
            raise RuntimeError("CSV writer is already running.")

        self.filename.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._ready.clear()
        self._error = None

        self._thread = threading.Thread(
            target=self._background_writer,
            name="BreathingCSVWriter",
            daemon=True
        )

        self._thread.start()
        self._ready.wait()

        if self._error is not None:
            raise RuntimeError(
                "Unable to initialize CSV writer."
            ) from self._error

        self._started = True

    def write(self, row: list) -> None:
        """Queue one respiratory sample for writing."""

        if not self._started:
            raise RuntimeError("CSV writer is not running.")

        if self._error is not None:
            raise RuntimeError("CSV writer failed.") from self._error

        self._queue.put(row)

    def stop(self) -> None:
        """Flush pending samples and terminate the writer."""

        if self._thread is None:
            return

        if self._thread.is_alive():
            self._queue.put(self._STOP)

        self._thread.join()
        self._started = False

        if self._error is not None:
            raise RuntimeError("CSV writing failed.") from self._error

    def _background_writer(self) -> None:
        """Process queued rows and write them to CSV."""

        try:
            with self.filename.open(
                "w",
                newline="",
                encoding="utf-8",
                buffering=1024 * 1024
            ) as csv_file:

                writer = csv.writer(csv_file)
                writer.writerow(self.headers)

                self._ready.set()

                while True:
                    row = self._queue.get()

                    if row is self._STOP:
                        break

                    writer.writerow(row)

        except Exception as exc:
            self._error = exc
            self._ready.set()
