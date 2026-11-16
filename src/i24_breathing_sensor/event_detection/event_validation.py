
"""
Respiratory Event Validation
=============================

Validates candidate respiratory cycles based on:
    1. Presence of a positive peak.
    2. Correct peak location.
    3. Cycle duration.
    4. Positive peak prominence (optional).
    5. Peak-to-peak amplitude (optional).

Rejected cycles are retained with rejection reasons.
"""

from .models import RespiratoryCycle


def validate_respiratory_cycles(
    cycles,
    min_cycle_duration_s=1.0,
    max_cycle_duration_s=10.0,
    min_peak_to_peak_amplitude=None,
    min_positive_peak_prominence=None,
):
    """
    Validate candidate respiratory cycles.

    Parameters
    ----------
    cycles : list[RespiratoryCycle]
        Candidate respiratory cycles.

    min_cycle_duration_s : float
        Minimum acceptable cycle duration.

    max_cycle_duration_s : float
        Maximum acceptable cycle duration.

    min_peak_to_peak_amplitude : float, optional
        Minimum acceptable peak-to-peak amplitude.

    min_positive_peak_prominence : float, optional
        Minimum positive peak prominence.

    Returns
    -------
    list[RespiratoryCycle]
        Validated cycles with acceptance status and rejection reasons.
    """

    if min_cycle_duration_s <= 0:
        raise ValueError(
            "Minimum cycle duration must be positive."
        )

    if max_cycle_duration_s <= min_cycle_duration_s:
        raise ValueError(
            "Maximum cycle duration must exceed minimum duration."
        )

    validated_cycles = []

    for cycle in cycles:

        rejection_reasons = []

        # 1. Check whether positive peak exists.
        if cycle.positive_peak is None:

            rejection_reasons.append(
                "Missing positive peak"
            )

        else:

            # 2. Check whether peak lies within cycle boundaries.
            peak_index = cycle.positive_peak.sample_index

            if not (
                cycle.start_trough.sample_index
                < peak_index
                < cycle.end_trough.sample_index
            ):

                rejection_reasons.append(
                    "Positive peak outside cycle boundaries"
                )

            # 3. Check positive peak prominence.
            if (
                min_positive_peak_prominence is not None
                and cycle.positive_peak.prominence
                < min_positive_peak_prominence
            ):

                rejection_reasons.append(
                    "Positive peak prominence too low"
                )

        # 4. Validate cycle duration.
        if cycle.duration_s < min_cycle_duration_s:

            rejection_reasons.append(
                "Cycle duration too short"
            )

        elif cycle.duration_s > max_cycle_duration_s:

            rejection_reasons.append(
                "Cycle duration too long"
            )

        # 5. Validate peak-to-peak amplitude.
        if (
            min_peak_to_peak_amplitude is not None
            and cycle.peak_to_peak_amplitude
            < min_peak_to_peak_amplitude
        ):

            rejection_reasons.append(
                "Peak-to-peak amplitude too low"
            )

        # Store validation results.
        cycle.rejection_reasons = rejection_reasons
        cycle.is_valid = len(rejection_reasons) == 0

        validated_cycles.append(cycle)

    return validated_cycles
