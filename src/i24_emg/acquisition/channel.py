from __future__ import annotations

from .models import I24Channel


# =========================================================
# Fixed I24 EMG live-data mapping
# =========================================================
#
# I24 signal configuration:
#
#   Channel 1 -> BI
#   Channel 2 -> EMG1
#   Channel 3 -> EMG2
#   Channel 4 -> BREATHING
#
# For the I24 live-data packet used by this acquisition
# module, the EMG signals are exposed at:
#
#   packet index 2 -> EMG1
#   packet index 3 -> EMG2
#
# This mapping is I24-specific.
#
# There is deliberately NO EMG3 mapping.
#


PACKET_TO_CHANNEL: dict[int, I24Channel] = {
    2: I24Channel.EMG1,
    3: I24Channel.EMG2,
}


# Fixed reverse mapping:
#
#   EMG1 -> packet index 2
#   EMG2 -> packet index 3
#

CHANNEL_TO_PACKET: dict[I24Channel, int] = {
    channel: packet_index
    for packet_index, channel in PACKET_TO_CHANNEL.items()
}


def get_channel(
    packet_index: int,
) -> I24Channel | None:
    """
    Convert an I24 live-data packet index into an EMG
    channel.

    Returns None when the packet entry is not one of the
    EMG channels exposed by this module.
    """

    return PACKET_TO_CHANNEL.get(
        packet_index
    )


def get_packet_index(
    channel: I24Channel,
) -> int:
    """
    Convert an I24 EMG channel into its fixed live-data
    packet index.
    """

    try:
        return CHANNEL_TO_PACKET[channel]

    except KeyError as exc:
        raise ValueError(
            f"No live-data packet mapping exists for "
            f"{channel.value}."
        ) from exc


def is_emg_packet_index(
    packet_index: int,
) -> bool:
    """
    Return True when the packet index corresponds to an
    EMG channel exposed by this module.
    """

    return packet_index in PACKET_TO_CHANNEL