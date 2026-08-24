"""rmw_zenoh wire-format helpers of the native uplink (ZenohPublisher path).

These formats are rmw_zenoh *internals* (verified against the jazzy branch of
ros2/rmw_zenoh — liveliness_utils.cpp, attachment_helpers.cpp, docs/design.md);
the tests pin them so a silent change in our helpers is caught locally.  The
golden liveliness tokens below are verbatim from the upstream design doc.

Pure-python throughout: zenoh / rosbags are only needed for the optional
cross-checks at the end (skipped when the extra is not installed).
"""

import struct

import pytest

from twinlink.sources.zenoh_source import (
    RMW_GID_STORAGE_SIZE,
    liveliness_subscriber_query,
    mangle_ros_type,
    parse_liveliness_token,
    rmw_attachment_bytes,
    topic_keyexpr,
)

#: Verbatim from rmw_zenoh docs/design.md (jazzy).
_HASH = "RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18"
SUB_TOKEN = (
    "@ros2_lv/0/aac3178e146ba6f1fc6e6a4085e77f21/0/10/MS/%/%/listener/"
    f"%chatter/std_msgs::msg::dds_::String_/{_HASH}/::,10:,:,:,,"
)
PUB_TOKEN = (
    "@ros2_lv/0/8b20917502ee955ac4476e0266340d5c/0/10/MP/%/%/talker/"
    f"%chatter/std_msgs::msg::dds_::String_/{_HASH}/::,7:,:,:,,"
)


# --------------------------------------------------------------------------- #
# type mangling + keyexprs
# --------------------------------------------------------------------------- #
def test_mangle_ros_type():
    assert (
        mangle_ros_type("sensor_msgs/msg/JointState")
        == "sensor_msgs::msg::dds_::JointState_"
    )
    assert mangle_ros_type("std_msgs/msg/Bool") == "std_msgs::msg::dds_::Bool_"
    # short pkg/Type form normalizes like the foxglove schema parser
    assert mangle_ros_type("std_msgs/String") == "std_msgs::msg::dds_::String_"
    with pytest.raises(ValueError):
        mangle_ros_type("not-a-type")


def test_topic_keyexpr_matches_design_doc_example():
    assert topic_keyexpr(0, "/chatter", "std_msgs::msg::dds_::String_", _HASH) == (
        f"0/chatter/std_msgs::msg::dds_::String_/{_HASH}"
    )


def test_topic_keyexpr_strips_slashes_only_at_the_ends():
    assert topic_keyexpr(0, "/twin/plan_goal", "T", "H") == "0/twin/plan_goal/T/H"


def test_liveliness_subscriber_query():
    assert liveliness_subscriber_query(0, "/twin/plan_goal") == (
        "@ros2_lv/0/*/*/*/MS/*/*/*/%twin%plan_goal/*/*/*"
    )


# --------------------------------------------------------------------------- #
# liveliness token parsing (golden tokens from upstream)
# --------------------------------------------------------------------------- #
def test_parse_subscriber_token():
    info = parse_liveliness_token(SUB_TOKEN)
    assert info == {
        "entity": "MS",
        "node": "listener",
        "topic": "/chatter",
        "type_name": "std_msgs::msg::dds_::String_",
        "type_hash": _HASH,
        "qos": "::,10:,:,:,,",
    }


def test_parse_publisher_token():
    info = parse_liveliness_token(PUB_TOKEN)
    assert info["entity"] == "MP"
    assert info["topic"] == "/chatter"
    assert info["type_hash"] == _HASH


def test_parse_rejects_node_tokens_and_foreign_keyexprs():
    node_token = "@ros2_lv/0/aac3178e146ba6f1fc6e6a4085e77f21/0/0/NN/%/%/listener"
    assert parse_liveliness_token(node_token) is None
    assert parse_liveliness_token("0/chatter/std_msgs::msg::dds_::String_/H") is None
    assert parse_liveliness_token("") is None


# --------------------------------------------------------------------------- #
# rmw attachment (a sample without it is DROPPED by rmw_zenoh subscribers)
# --------------------------------------------------------------------------- #
def test_attachment_layout():
    gid = bytes(range(RMW_GID_STORAGE_SIZE))
    raw = rmw_attachment_bytes(7, 1_234_567_890_123, gid)
    # zenoh ext serializer output for (int64, int64, [u8;16]):
    # 8 LE bytes + 8 LE bytes + LEB128 length (16 -> 0x10) + gid = 33 bytes.
    assert len(raw) == 33
    seq, ts = struct.unpack_from("<qq", raw)
    assert (seq, ts) == (7, 1_234_567_890_123)
    assert raw[16] == RMW_GID_STORAGE_SIZE
    assert raw[17:] == gid


def test_attachment_rejects_bad_gid():
    with pytest.raises(ValueError):
        rmw_attachment_bytes(1, 0, b"short")


def test_attachment_matches_zenoh_ext_serializer():
    """Cross-check the hand-rolled bytes against eclipse-zenoh's own codec."""
    ze = pytest.importorskip("zenoh.ext")
    gid = bytes(range(16))
    ours = rmw_attachment_bytes(7, 1_234_567_890_123, gid)
    theirs = ze.z_serialize((ze.Int64(7), ze.Int64(1_234_567_890_123), gid))
    assert ours == theirs.to_bytes()


# --------------------------------------------------------------------------- #
# CDR payloads (identical bytes on the zenoh and foxglove transports)
# --------------------------------------------------------------------------- #
def test_cdr_roundtrip_of_the_twin_types():
    typesys = pytest.importorskip("rosbags.typesys")
    import numpy as np

    ts = typesys.get_typestore(typesys.Stores.LATEST)
    JS = ts.types["sensor_msgs/msg/JointState"]
    Hdr = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    msg = JS(
        header=Hdr(stamp=Time(sec=0, nanosec=0), frame_id=""),
        name=["arm_0_shoulder_pan_joint"],
        position=np.array([0.5], dtype=np.float64),
        velocity=np.array([]),
        effort=np.array([]),
    )
    raw = ts.serialize_cdr(msg, "sensor_msgs/msg/JointState")
    back = ts.deserialize_cdr(raw, "sensor_msgs/msg/JointState")
    assert list(back.name) == ["arm_0_shoulder_pan_joint"]
    assert back.position[0] == pytest.approx(0.5)

    for typename, kwargs in (
        ("std_msgs/msg/Bool", {"data": True}),
        ("std_msgs/msg/String", {"data": '{"action": "prepare"}'}),
    ):
        m = ts.types[typename](**kwargs)
        assert (
            ts.deserialize_cdr(ts.serialize_cdr(m, typename), typename).data
            == kwargs["data"]
        )


# --------------------------------------------------------------------------- #
# loopback integration: discovery -> keyexpr -> publish -> attachment
# --------------------------------------------------------------------------- #
def test_uplink_loopback_discovers_and_publishes():
    """The full uplink path over a real zenoh session (in-process loopback).

    An rmw_zenoh-style subscriber is faked with a liveliness token + a plain
    subscriber on the data keyexpr; the uplink must discover type+hash from
    the token, publish CDR onto the exact keyexpr and attach a valid rmw
    attachment.  Verifies our plumbing — the rmw side itself is covered by
    the smoke test at the real robot.
    """
    zenoh = pytest.importorskip("zenoh")
    pytest.importorskip("rosbags")
    import time

    from twinlink.sources.zenoh_source import ZenohUplink

    cfg = zenoh.Config()
    cfg.insert_json5("scouting/multicast/enabled", "false")
    session = zenoh.open(cfg)
    try:
        type_name = mangle_ros_type("std_msgs/msg/String")
        type_hash = "RIHS01_" + "ab" * 32
        token = session.liveliness().declare_token(
            f"@ros2_lv/0/{'0' * 32}/0/10/MS/%/%/plan_server/"
            f"%twin%arm_cmd/{type_name}/{type_hash}/::,10:,:,:,,"
        )
        received = []
        sub = session.declare_subscriber(
            f"0/twin/arm_cmd/{type_name}/{type_hash}", lambda s: received.append(s)
        )

        uplink = ZenohUplink(domain_id=0)
        uplink._session = session  # share the session: no scouting in tests
        pub = uplink.publisher("/twin/arm_cmd", "std_msgs/msg/String").start()
        String = pub.typestore.types["std_msgs/msg/String"]
        pub.publish(String(data='{"action": "prepare", "request_id": "arm-1"}'))

        deadline = time.time() + 5.0
        while not received and time.time() < deadline:
            time.sleep(0.05)
        assert received, "no sample arrived on the discovered keyexpr"
        sample = received[0]
        raw = sample.attachment.to_bytes()
        seq, _ts = struct.unpack_from("<qq", raw)
        assert (len(raw), seq, raw[16]) == (33, 1, RMW_GID_STORAGE_SIZE)
        msg = pub.typestore.deserialize_cdr(
            sample.payload.to_bytes(), "std_msgs/msg/String"
        )
        assert msg.data == '{"action": "prepare", "request_id": "arm-1"}'
        pub.close()
        sub.undeclare()
        token.undeclare()
    finally:
        session.close()
