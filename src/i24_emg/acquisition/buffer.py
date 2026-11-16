from __future__ import annotations

from collections import deque
from itertools import islice

from .models import EMGFrame


class EMGBuffer:
    """
    Fixed-size FIFO buffer for I24 EMG acquisition frames.

    When the buffer reaches its maximum size, the oldest frame
    is automatically removed when a new frame is appended.
    """

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError(
                "max_size must be greater than zero."
            )

        self._buffer: deque[EMGFrame] = deque(
            maxlen=max_size
        )

    # ---------------------------------------------------------
    # Modification
    # ---------------------------------------------------------

    def append(self, frame: EMGFrame) -> None:
        """
        Add one EMG acquisition frame.

        If the buffer is full, the oldest frame is automatically
        discarded by the underlying deque.
        """

        self._buffer.append(frame)

    def clear(self) -> None:
        """Remove all buffered EMG frames."""

        self._buffer.clear()

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    @property
    def size(self) -> int:
        """Return the number of frames currently stored."""

        return len(self._buffer)

    @property
    def max_size(self) -> int:
        """Return the maximum number of frames the buffer can hold."""

        return self._buffer.maxlen or 0

    @property
    def is_empty(self) -> bool:
        """Return True when the buffer contains no frames."""

        return not self._buffer

    @property
    def is_full(self) -> bool:
        """Return True when the buffer has reached maximum capacity."""

        return len(self._buffer) == self.max_size

    # ---------------------------------------------------------
    # Access
    # ---------------------------------------------------------

    def get_all(self) -> tuple[EMGFrame, ...]:
        """
        Return all buffered frames in chronological order.

        The oldest frame is first and the newest frame is last.
        """

        return tuple(self._buffer)

    def get_latest(
        self,
        count: int,
    ) -> tuple[EMGFrame, ...]:
        """
        Return the latest `count` frames in chronological order.

        If `count` is greater than the number of buffered frames,
        all available frames are returned.

        A non-positive count returns an empty tuple.
        """

        if count <= 0:
            return ()

        start = max(
            len(self._buffer) - count,
            0,
        )

        return tuple(
            islice(
                self._buffer,
                start,
                None,
            )
        )

    @property
    def latest(self) -> EMGFrame | None:
        """
        Return the newest buffered frame.

        Returns None if the buffer is empty.
        """

        if not self._buffer:
            return None

        return self._buffer[-1]

    # ---------------------------------------------------------
    # Python protocol
    # ---------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of buffered frames."""

        return len(self._buffer)

    def __bool__(self) -> bool:
        """Return True when at least one frame is buffered."""

        return bool(self._buffer)