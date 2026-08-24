"""Live source + uplink over a native Zenoh client (no ROS install needed).

For robots running ``rmw_zenoh`` (like this Husky) a plain Zenoh client can join
the robot's Zenoh graph directly and subscribe to ROS 2 topics. Unlike a
visualization bridge, Zenoh is a robotics-grade transport (pub/sub + query, low
latency, bidirectional-capable). Payloads are the same CDR ROS 2 puts on the
wire, decoded via the same path as :class:`~twinlink.sources.mcap.McapSource` / :class:`~twinlink.sources.foxglove.FoxgloveSource`.

    rmw_zenoh graph ──zenoh (CDR)──▶ ZenohSource ──▶ RobotState ──▶ MujocoSink
    ZenohPublisher ──zenoh (CDR + rmw attachment)──▶ rmw_zenoh subscriber

rmw_zenoh maps a topic to a key expression (verified against the jazzy branch
of ros2/rmw_zenoh, ``liveliness_utils.cpp`` / ``docs/design.md``)::

    <domain_id>/<topic without leading '/'>/<mangled_type>/<type_hash>
    e.g. 0/chatter/std_msgs::msg::dds_::String_/RIHS01_df668c74…

The *source* subscribes with a ``**`` wildcard for the type/hash tail, so it
doesn't depend on the exact type mangling, e.g.
``0/a200_0553/platform/joint_states/**``.  The *publisher* cannot wildcard — it
must put on the exact key expression the rmw_zenoh subscriber declared, so
:class:`ZenohUplink` discovers type + hash from the subscriber's liveliness
token in the ``@ros2_lv`` admin space (with pinned hashes as offline fallback).
Every put carries the rmw_zenoh per-message attachment (sequence number,
source timestamp, gid) — **a sample without it is dropped** by the subscriber
(``rmw_subscription_data.cpp``).

Caveats (verify at the robot):

* the client's **zenoh protocol version must match** the robot's rmw_zenoh zenoh
  version, or the session won't connect;
* the keyexpr/attachment/liveliness layouts are rmw_zenoh *internals* (stable on
  jazzy, no upstream guarantee) — on an rmw_zenoh upgrade re-run the uplink
  smoke test before trusting it (the foxglove transport stays the fallback);
* CDR on the wire carries no schema, so every type must be known at decode time.
  Standard ``sensor_msgs`` etc. ship with the rosbags typestore; the ``moveit_msgs``
  needed for the plan preview (``DisplayTrajectory``) are registered here from
  their ``.msg`` definitions -- see :func:`_register_moveit_types`;
* latched/transient-local topics (``robot_description``, ``tf_static``) need a
  querying subscriber -- live high-rate topics (joint_states, points) are fine.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from typing import Dict, List, Optional

from .base import StateSource

log = logging.getLogger("twinlink.zenoh")

#: rmw gid length (RMW_GID_STORAGE_SIZE) — fixed 16 bytes on ROS 2 jazzy.
RMW_GID_STORAGE_SIZE = 16
#: rmw_zenoh liveliness admin space + the entity-kind tag of a subscriber.
LIVELINESS_ADMIN_SPACE = "@ros2_lv"
LIVELINESS_SUBSCRIBER = "MS"
_LIVELINESS_ENTITIES = ("NN", "MP", "MS", "SS", "SC")


# --------------------------------------------------------------------------- #
# rmw_zenoh wire-format helpers (pure, testable without zenoh installed)
# --------------------------------------------------------------------------- #
def mangle_ros_type(msgtype: str) -> str:
    """ROS type name -> the DDS-mangled form rmw_zenoh puts in key expressions.

    ``sensor_msgs/msg/JointState`` -> ``sensor_msgs::msg::dds_::JointState_``
    (also accepts the short ``pkg/Type`` form).
    """
    parts = msgtype.split("/")
    if len(parts) == 2:  # pkg/Type -> pkg/msg/Type
        parts = [parts[0], "msg", parts[1]]
    if len(parts) != 3:
        raise ValueError(f"not a ROS type name: {msgtype!r}")
    pkg, sub, name = parts
    return f"{pkg}::{sub}::dds_::{name}_"


def topic_keyexpr(domain_id: int, topic: str, type_name: str, type_hash: str) -> str:
    """The exact data keyexpr rmw_zenoh pubs/subs use for one topic."""
    return f"{domain_id}/{topic.strip('/')}/{type_name}/{type_hash}"


def liveliness_subscriber_query(domain_id: int, topic: str) -> str:
    """Liveliness selector matching every rmw_zenoh *subscriber* of ``topic``.

    Token layout (13 segments): ``@ros2_lv/<domain>/<zid>/<nid>/<id>/<entity>/
    <enclave>/<namespace>/<node>/<topic>/<type>/<hash>/<qos>`` with ``/`` in
    names mangled to ``%``.
    """
    mangled = topic if topic.startswith("/") else f"/{topic}"
    mangled = mangled.replace("/", "%")
    return (
        f"{LIVELINESS_ADMIN_SPACE}/{domain_id}/*/*/*/{LIVELINESS_SUBSCRIBER}"
        f"/*/*/*/{mangled}/*/*/*"
    )


def parse_liveliness_token(keyexpr: str) -> Optional[dict]:
    """Parse an rmw_zenoh liveliness token into its topic entity fields.

    Returns ``None`` for non-topic tokens (nodes) and foreign keyexprs."""
    parts = keyexpr.split("/")
    if len(parts) < 13 or parts[0] != LIVELINESS_ADMIN_SPACE:
        return None
    entity = parts[5]
    if entity not in _LIVELINESS_ENTITIES or entity == "NN":
        return None
    return {
        "entity": entity,
        "node": parts[8],
        "topic": parts[9].replace("%", "/"),
        "type_name": parts[10],
        "type_hash": parts[11],
        "qos": parts[12],
    }


def rmw_attachment_bytes(
    sequence_number: int, source_timestamp_ns: int, gid: bytes
) -> bytes:
    """Serialize the rmw_zenoh per-message attachment (33 bytes).

    rmw_zenoh subscribers DROP any sample without this attachment.  Layout =
    zenoh ``ext`` serializer output for ``(int64, int64, [u8;16])``: two
    little-endian int64s, then the gid as a length-prefixed sequence (LEB128
    varint — 16 is the single byte ``0x10``).  Verified against rmw_zenoh_cpp
    ``attachment_helpers.cpp`` (jazzy) and eclipse-zenoh 1.9 ``z_serialize``.
    """
    if len(gid) != RMW_GID_STORAGE_SIZE:
        raise ValueError(f"gid must be {RMW_GID_STORAGE_SIZE} bytes, got {len(gid)}")
    return (
        struct.pack("<qq", sequence_number, source_timestamp_ns)
        + bytes([RMW_GID_STORAGE_SIZE])
        + gid
    )


def _session_config(zenoh, mode: str, connect):
    """zenoh.Config for a client/peer session, optionally pinned to endpoints."""
    cfg = zenoh.Config()
    cfg.insert_json5("mode", json.dumps(mode))
    if connect:
        endpoints = connect if isinstance(connect, list) else [connect]
        cfg.insert_json5("connect/endpoints", json.dumps(endpoints))
    return cfg


# moveit_msgs is not shipped by the rosbags typestore, but the plan-preview topic
# (``moveit_msgs/msg/DisplayTrajectory``) needs it to decode CDR off the wire. Its
# transitive deps that *are* standard (trajectory_msgs, sensor_msgs, shape_msgs,
# geometry_msgs, std_msgs) already live in the typestore; only these definitions
# are missing. Field order/types must match the wire layout exactly -- these are
# the stable upstream ``.msg`` definitions (unchanged across ROS 2 distros).
_MOVEIT_MSG_DEFS = {
    "object_recognition_msgs/msg/ObjectType": ("string key\n" "string db\n"),
    "moveit_msgs/msg/CollisionObject": (
        "std_msgs/Header header\n"
        "geometry_msgs/Pose pose\n"
        "string id\n"
        "object_recognition_msgs/ObjectType type\n"
        "shape_msgs/SolidPrimitive[] primitives\n"
        "geometry_msgs/Pose[] primitive_poses\n"
        "shape_msgs/Mesh[] meshes\n"
        "geometry_msgs/Pose[] mesh_poses\n"
        "shape_msgs/Plane[] planes\n"
        "geometry_msgs/Pose[] plane_poses\n"
        "string[] subframe_names\n"
        "geometry_msgs/Pose[] subframe_poses\n"
        "byte ADD=0\n"
        "byte REMOVE=1\n"
        "byte APPEND=2\n"
        "byte MOVE=3\n"
        "byte operation\n"
    ),
    "moveit_msgs/msg/AttachedCollisionObject": (
        "string link_name\n"
        "moveit_msgs/CollisionObject object\n"
        "string[] touch_links\n"
        "trajectory_msgs/JointTrajectory detach_posture\n"
        "float64 weight\n"
    ),
    "moveit_msgs/msg/RobotState": (
        "sensor_msgs/JointState joint_state\n"
        "sensor_msgs/MultiDOFJointState multi_dof_joint_state\n"
        "moveit_msgs/AttachedCollisionObject[] attached_collision_objects\n"
        "bool is_diff\n"
    ),
    "moveit_msgs/msg/RobotTrajectory": (
        "trajectory_msgs/JointTrajectory joint_trajectory\n"
        "trajectory_msgs/MultiDOFJointTrajectory multi_dof_joint_trajectory\n"
    ),
    "moveit_msgs/msg/DisplayTrajectory": (
        "string model_id\n"
        "moveit_msgs/RobotTrajectory[] trajectory\n"
        "moveit_msgs/RobotState trajectory_start\n"
    ),
}


def _register_moveit_types(typestore) -> None:
    """Register moveit_msgs (plan preview) into ``typestore`` if not already there."""
    if "moveit_msgs/msg/DisplayTrajectory" in typestore.types:
        return
    try:
        from rosbags.typesys import get_types_from_msg
    except ImportError:
        log.debug("rosbags too old for get_types_from_msg; moveit preview unavailable")
        return
    try:
        add = {}
        for name, text in _MOVEIT_MSG_DEFS.items():
            add.update(get_types_from_msg(text, name))
        typestore.register(add)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("could not register moveit_msgs types (preview disabled): %s", exc)


class ZenohSource(StateSource):
    def __init__(
        self,
        connect: Optional[str] = None,
        *,
        mode: Optional[str] = None,
        domain_id: int = 0,
        topics: Optional[List[str]] = None,
        key_template: str = "{domain}/{topic}/**",
        store: str = "LATEST",
        reconnect: bool = True,
    ) -> None:
        super().__init__()
        self.connect = connect
        # default to client mode when a router endpoint is given, else peer (LAN scouting)
        self.mode = mode or ("client" if connect else "peer")
        self.domain_id = domain_id
        self.topics = topics
        self.key_template = key_template
        self.store = store
        self.reconnect = reconnect

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._session = None
        self._subs: list = []
        self._typestore = None

    # ------------------------------------------------------------------ #
    def start(self) -> "ZenohSource":
        assert self.state is not None, "bind() before start()"
        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="zenoh-source", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()  # unblocks _run; its finally tears the session down
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._running = False

    def _wanted_topics(self) -> list:
        if self.topics is not None:
            return list(self.topics)
        return self.mapping.topics() if self.mapping is not None else []

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            import zenoh
        except ImportError:
            log.error("ZenohSource needs eclipse-zenoh: pip install 'twinlink[zenoh]'")
            self._running = False
            return
        try:
            from rosbags.typesys import Stores, get_typestore
        except ImportError:
            log.error("ZenohSource needs rosbags: pip install 'twinlink[zenoh]'")
            self._running = False
            return
        self._typestore = get_typestore(getattr(Stores, self.store, Stores.LATEST))
        _register_moveit_types(self._typestore)

        backoff = 0.5
        while not self._stop.is_set():
            try:
                self._session = zenoh.open(
                    _session_config(zenoh, self.mode, self.connect)
                )
                log.info(
                    "zenoh session open (mode=%s, connect=%s)",
                    self.mode,
                    self.connect or "scout",
                )
                if self._declare(zenoh):
                    self._stop.wait()  # connected; zenoh delivers on its own threads
                    break
            except Exception as exc:
                log.warning(
                    "zenoh connect failed (mode=%s connect=%s): %s",
                    self.mode,
                    self.connect,
                    exc,
                )
            finally:
                self._teardown()
            if not self.reconnect or self._stop.is_set():
                break
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 5.0)
        self._running = False
        log.info("zenoh source stopped")

    def _declare(self, zenoh) -> int:
        n = 0
        for topic in self._wanted_topics():
            msgtype = self.mapping.topic_type(topic)
            if not msgtype:
                log.warning(
                    "no message type known for %s; set topic_types in config", topic
                )
                continue
            if msgtype not in self._typestore.types:
                log.warning(
                    "type %s has no schema over zenoh — skipping %s", msgtype, topic
                )
                continue
            key = self.key_template.format(
                domain=self.domain_id, topic=topic.lstrip("/")
            )
            self._subs.append(
                self._session.declare_subscriber(key, self._make_cb(topic, msgtype))
            )
            log.info("subscribed %s [%s] via %s", topic, msgtype, key)
            n += 1
        if n == 0:
            log.warning("no subscriptions established")
        return n

    def _teardown(self) -> None:
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                pass
        self._subs = []
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def _make_cb(self, topic: str, msgtype: str):
        def _cb(sample) -> None:
            try:
                payload = sample.payload.to_bytes()
                msg = self._typestore.deserialize_cdr(payload, msgtype)
                self.mapping.apply(topic, msgtype, msg, self.state)
            except Exception as exc:
                log.debug("decode failed on %s (%s): %s", topic, msgtype, exc)

        return _cb


# --------------------------------------------------------------------------- #
# Uplink: publish onto rmw_zenoh topics without ROS
# --------------------------------------------------------------------------- #
class ZenohUplink:
    """One shared zenoh session for all uplink publishers of a client.

    The write-side sibling of :class:`ZenohSource` and the zenoh counterpart of
    a ``foxglove_bridge`` with clientPublish: publishers made from one uplink
    share a single session (one TCP connection / tokio runtime instead of one
    per topic).  The uplink also does the keyexpr *discovery*: publishing needs
    the exact ``<domain>/<topic>/<type>/<hash>`` the subscriber declared, so it
    queries the subscriber's ``@ros2_lv`` liveliness token — always
    version-correct, zero configuration.  ``type_hashes`` (ROS type ->
    ``RIHS01_…``) is the offline fallback when nothing is discoverable, e.g.
    pinned in a robot_contract profile from ``ros2 topic info --verbose``.

        uplink = ZenohUplink("tcp/10.42.42.159:7447")
        pub = uplink.publisher("/twin/arm_cmd", "std_msgs/msg/String").start()
        pub.publish(msg)          # rosbags message object, CDR on the wire
    """

    def __init__(
        self,
        connect: Optional[str] = None,
        *,
        mode: Optional[str] = None,
        domain_id: int = 0,
        type_hashes: Optional[Dict[str, str]] = None,
        discovery_timeout: float = 1.5,
        store: str = "LATEST",
    ) -> None:
        self.connect = connect
        self.mode = mode or ("client" if connect else "peer")
        self.domain_id = int(domain_id)
        self.type_hashes = dict(type_hashes or {})
        self.discovery_timeout = float(discovery_timeout)
        self.store = store
        self._session = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def open(self):
        """The shared session (opened lazily; raises when unreachable)."""
        with self._lock:
            if self._session is None:
                import zenoh

                self._session = zenoh.open(
                    _session_config(zenoh, self.mode, self.connect)
                )
                log.info(
                    "zenoh uplink session open (mode=%s, connect=%s)",
                    self.mode,
                    self.connect or "scout",
                )
            return self._session

    def publisher(self, topic: str, msgtype: str) -> "ZenohPublisher":
        """An (unstarted) publisher for ``topic`` on this shared session."""
        return ZenohPublisher(self, topic, msgtype)

    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = None

    # ------------------------------------------------------------------ #
    def resolve_keyexpr(self, topic: str, msgtype: str) -> str:
        """The exact data keyexpr to publish ``topic`` on.

        Primary: the type name + hash from a live subscriber's liveliness
        token (the subscriber defines what it accepts).  Fallback: a pinned
        hash from ``type_hashes``.  Raises ``ConnectionError`` when neither is
        available — callers with retry loops (e.g. TwinMotionClient.connect)
        treat that like a bridge that is not up yet.
        """
        session = self.open()
        expected = mangle_ros_type(msgtype)
        tokens: List[dict] = []
        try:
            replies = session.liveliness().get(
                liveliness_subscriber_query(self.domain_id, topic),
                timeout=self.discovery_timeout,
            )
            for reply in replies:
                sample = getattr(reply, "ok", None)
                if sample is None:
                    continue
                info = parse_liveliness_token(str(sample.key_expr))
                if info is not None:
                    tokens.append(info)
        except Exception as exc:
            log.debug("liveliness discovery failed on %s: %s", topic, exc)
        for info in tokens:
            if info["type_name"] == expected:
                return topic_keyexpr(
                    self.domain_id, topic, info["type_name"], info["type_hash"]
                )
        if tokens:
            # A subscriber exists but under another type: trust the graph —
            # it is the side that deserializes.
            info = tokens[0]
            log.warning(
                "subscriber on %s has type %s (expected %s) — "
                "publishing with the graph's type",
                topic,
                info["type_name"],
                expected,
            )
            return topic_keyexpr(
                self.domain_id, topic, info["type_name"], info["type_hash"]
            )
        pinned = self.type_hashes.get(msgtype)
        if pinned:
            log.info("no live subscriber on %s — using the pinned type hash", topic)
            return topic_keyexpr(self.domain_id, topic, expected, pinned)
        raise ConnectionError(
            f"no rmw_zenoh subscriber for {topic} [{msgtype}] discoverable via "
            f"liveliness (plan server running? router reachable?) and no pinned "
            f"type hash configured"
        )


class ZenohPublisher:
    """Publish ROS messages onto an rmw_zenoh topic — no ROS, no bridge.

    Interface-compatible with
    :class:`twinlink.sources.foxglove.FoxglovePublisher` (``typestore`` /
    ``start`` / ``publish`` / ``publish_raw`` / ``close``), so twin clients can
    switch transports without touching call sites.  Build messages with
    :attr:`typestore` (``pub.typestore.types[name](...)``); the CDR bytes on
    the wire are identical to the foxglove path.  Every put carries the
    rmw_zenoh attachment (see :func:`rmw_attachment_bytes`) — without it the
    subscriber drops the sample.
    """

    def __init__(
        self,
        uplink: ZenohUplink,
        topic: str,
        msgtype: str,
        *,
        store: Optional[str] = None,
    ) -> None:
        self.uplink = uplink
        self.topic = topic
        self.msgtype = msgtype  # e.g. "sensor_msgs/msg/JointState"
        self.store = store or uplink.store
        self._typestore = None
        self._pub = None
        self._gid = os.urandom(RMW_GID_STORAGE_SIZE)
        self._seq = 1

    @property
    def typestore(self):
        return self._typestore

    def start(self) -> "ZenohPublisher":
        import zenoh
        from rosbags.typesys import Stores, get_typestore

        self._typestore = get_typestore(getattr(Stores, self.store, Stores.LATEST))
        keyexpr = self.uplink.resolve_keyexpr(self.topic, self.msgtype)
        # Reliable QoS the way rmw_zenoh maps it (congestion control BLOCK) —
        # the /twin/* uplinks are low-rate commands that must not be shed.
        self._pub = self.uplink.open().declare_publisher(
            keyexpr,
            congestion_control=zenoh.CongestionControl.BLOCK,
            reliability=zenoh.Reliability.RELIABLE,
        )
        log.info("zenoh uplink ready: %s [%s] -> %s", self.topic, self.msgtype, keyexpr)
        return self

    def publish(self, msg) -> None:
        """Serialize a message (built from :attr:`typestore`) and put it."""
        self.publish_raw(self._typestore.serialize_cdr(msg, self.msgtype))

    def publish_raw(self, payload: bytes) -> None:
        attachment = rmw_attachment_bytes(self._seq, time.time_ns(), self._gid)
        self._seq += 1
        self._pub.put(bytes(payload), attachment=attachment)

    def close(self) -> None:
        """Undeclare this publisher (the shared uplink session stays open)."""
        if self._pub is not None:
            try:
                self._pub.undeclare()
            except Exception:
                pass
            self._pub = None
