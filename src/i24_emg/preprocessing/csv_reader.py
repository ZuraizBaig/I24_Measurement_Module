
"""CSV loading utilities for I24 EMG recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_emg_csv(
    file_path: str | Path,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
    header: int | None = 0,
) -> pd.DataFrame:
    """Load an EMG recording from a CSV file.

    Args:
        file_path: Path to the CSV file.
        delimiter: Field delimiter used in the CSV.
        encoding: Text encoding.
        header: Row containing column names, or None if there is no
            header row.

    Returns:
        A pandas DataFrame containing the loaded recording.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or contains no data columns.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if not delimiter:
        raise ValueError("delimiter must not be empty.")

    try:
        data = pd.read_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            header=header,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {path}") from exc

    if data.empty:
        raise ValueError(f"CSV file contains no data rows: {path}")

    if data.shape[1] == 0:
        raise ValueError(f"CSV file contains no columns: {path}")

    return data


def extract_channel(
    data: pd.DataFrame,
    channel: str | int,
    *,
    allow_non_numeric: bool = False,
) -> np.ndarray:
    """Extract one channel from a DataFrame as a NumPy array.

    Args:
        data: DataFrame containing the recording.
        channel: Column name or zero-based column index.
        allow_non_numeric: If False, reject non-numeric channel data.

    Returns:
        A one-dimensional NumPy array.

    Note:
        This function does not infer I24 packet indices or map physical
        hardware channels to EMG1/EMG2. Supply the correct column
        explicitly.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    if isinstance(channel, bool):
        raise ValueError("channel must be a column name or integer index.")

    if isinstance(channel, int):
        if channel < 0 or channel >= data.shape[1]:
            raise IndexError(
                f"Column index {channel} is outside the available range "
                f"0 to {data.shape[1] - 1}."
            )
        series = data.iloc[:, channel]

    elif isinstance(channel, str):
        if channel not in data.columns:
            raise KeyError(f"Column not found: {channel}")
        series = data[channel]

    else:
        raise TypeError("channel must be a string or integer index.")

    if allow_non_numeric:
        try:
            values = pd.to_numeric(series, errors="raise").to_numpy(
                dtype=float
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Channel {channel!r} contains non-numeric values."
            ) from exc
    else:
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(
                f"Channel {channel!r} must contain numeric data."
            )
        values = series.to_numpy(dtype=float)

    if values.ndim != 1:
        raise ValueError("Extracted channel must be one-dimensional.")

    if values.size == 0:
        raise ValueError("Extracted channel must not be empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Channel {channel!r} contains NaN or infinite values."
        )

    return values.copy()


def extract_channels(
    data: pd.DataFrame,
    channels: list[str | int],
) -> dict[str, np.ndarray]:
    """Extract multiple channels into a dictionary.

    The dictionary keys are the supplied channel names or stringified
    column indices.
    """
    if not channels:
        raise ValueError("channels must contain at least one channel.")

    extracted: dict[str, np.ndarray] = {}

    for channel in channels:
        key = str(channel)

        if key in extracted:
            raise ValueError(f"Duplicate channel requested: {key}")

        extracted[key] = extract_channel(data, channel)

    return extracted
