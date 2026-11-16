
"""
Visualization of respiratory waveforms and detected events.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_breathing_curve(
    signal,
    sampling_rate,
    title="I24 Breathing Signal",
    save_path=None,
    show=True,
):
    """Plot the raw breathing waveform."""

    x = np.asarray(signal, dtype=float)

    if x.ndim != 1 or len(x) == 0:
        raise ValueError("Signal must be a non-empty 1D array.")

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    time = np.arange(len(x)) / sampling_rate

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        time,
        x,
        linewidth=0.8,
        label="Raw breathing signal"
    )

    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Raw signal amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


def plot_detected_events(
    signal,
    sampling_rate,
    result,
    title="Breathing Event Detection",
    save_path=None,
    show=True,
):
    """
    Plot the raw waveform with detected positive peaks,
    negative troughs and validated/rejected cycles.
    """

    x = np.asarray(signal, dtype=float)

    if x.ndim != 1 or len(x) == 0:
        raise ValueError("Signal must be a non-empty 1D array.")

    time = np.arange(len(x)) / sampling_rate

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        time,
        x,
        color="steelblue",
        linewidth=0.8,
        label="Raw breathing signal",
        zorder=2,
    )

    # ---------------------------------------------------------
    # Cycle validation regions
    # ---------------------------------------------------------

    for cycle in result.cycles:

        start_time = cycle.start_trough.time_s
        end_time = cycle.end_trough.time_s

        if cycle.is_valid:

            ax.axvspan(
                start_time,
                end_time,
                color="green",
                alpha=0.08,
                zorder=0,
            )

        else:

            ax.axvspan(
                start_time,
                end_time,
                color="red",
                alpha=0.10,
                zorder=0,
            )

    # ---------------------------------------------------------
    # Positive peaks
    # ---------------------------------------------------------

    if result.positive_peaks:

        peak_times = [
            event.time_s
            for event in result.positive_peaks
        ]

        peak_values = [
            event.amplitude
            for event in result.positive_peaks
        ]

        ax.scatter(
            peak_times,
            peak_values,
            color="red",
            marker="^",
            s=65,
            label="Positive peaks",
            zorder=4,
        )

    # ---------------------------------------------------------
    # Negative troughs
    # ---------------------------------------------------------

    if result.negative_troughs:

        trough_times = [
            event.time_s
            for event in result.negative_troughs
        ]

        trough_values = [
            event.amplitude
            for event in result.negative_troughs
        ]

        ax.scatter(
            trough_times,
            trough_values,
            color="darkorange",
            marker="v",
            s=65,
            label="Negative troughs",
            zorder=4,
        )

    # ---------------------------------------------------------
    # Cycle labels
    # ---------------------------------------------------------

    for cycle in result.cycles:

        midpoint = (
            cycle.start_trough.time_s
            + cycle.end_trough.time_s
        ) / 2

        ax.text(
            midpoint,
            max(cycle.start_trough.amplitude,
                cycle.end_trough.amplitude),
            str(cycle.cycle_index),
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Raw signal amplitude")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax
