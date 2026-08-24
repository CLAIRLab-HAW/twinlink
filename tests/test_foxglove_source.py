"""FoxgloveSource wire parsing -- the pure/testable pieces (no WebSocket).

``parse_message_data`` carries the bridge's receive timestamp so consumers can split sensor->bridge from bridge->client
lag (background: docs/hrl-real-perception-plan.md, Phase 0).
"""

import struct

from twinlink.sources.foxglove import parse_message_data, select_channels


def _frame(sub_id: int, recv_ns: int, payload: bytes) -> bytes:
    return b"\x01" + struct.pack("<I", sub_id) + struct.pack("<Q", recv_ns) + payload


def test_parse_message_data_returns_recv_stamp_seconds():
    parsed = parse_message_data(_frame(7, 1_500_000_000, b"cdr"))
    assert parsed is not None
    sub_id, recv_stamp, payload = parsed
    assert sub_id == 7
    assert recv_stamp == 1.5
    assert payload == b"cdr"


def test_parse_message_data_rejects_non_message_frames():
    assert parse_message_data(b"") is None
    assert parse_message_data(b"\x02" + b"\x00" * 12) is None  # wrong opcode
    assert parse_message_data(b"\x01\x00\x00") is None  # too short


def test_select_channels_filters_topic_and_encoding():
    channels = [
        {"id": 1, "topic": "/a", "encoding": "cdr"},
        {"id": 2, "topic": "/b", "encoding": "json"},
        {"id": 3, "topic": "/c"},  # encoding defaults to cdr
    ]
    picked = select_channels(channels, {"/a", "/b", "/c"})
    assert [ch["id"] for ch in picked] == [1, 3]
