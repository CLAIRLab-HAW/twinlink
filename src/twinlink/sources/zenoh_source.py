"""Live source over a native Zenoh client (no ROS install needed).

For robots running ``rmw_zenoh`` (like this Husky) a plain Zenoh client can join
the robot's Zenoh graph directly and subscribe to ROS 2 topics. Unlike a
visualization bridge, Zenoh is a robotics-grade transport (pub/sub + query, low
latency, bidirectional-capable). Payloads are the same CDR ROS 2 puts on the
wire, decoded via the same path as :class:`McapSource` / :class:`FoxgloveSource`.

    rmw_zenoh graph ──zenoh (CDR)──▶ ZenohSource ──▶ RobotState ──▶ MujocoSink

rmw_zenoh maps a topic to a key expression roughly like::

    <domain_id>/<topic without leading '/'>/<mangled_type>/<type_hash>

We subscribe with a ``**`` wildcard for the type/hash tail, so we don't depend on
the exact type mangling, e.g. ``0/a200_0553/platform/joint_states/**``.

Caveats (verify at the robot):
* the client's **zenoh protocol version must match** the robot's rmw_zenoh zenoh
  version, or the session won't connect;
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
import threading
from typing import List, Optional

from .base import StateSource

log = logging.getLogger("twinlink.zenoh")


# moveit_msgs is not shipped by the rosbags typestore, but the plan-preview topic
# (``moveit_msgs/msg/DisplayTrajectory``) needs it to decode CDR off the wire. Its
# transitive deps that *are* standard (trajectory_msgs, sensor_msgs, shape_msgs,
# geometry_msgs, std_msgs) already live in the typestore; only these definitions
# are missing. Field order/types must match the wire layout exactly -- these are
# the stable upstream ``.msg`` definitions (unchanged across ROS 2 distros).
_MOVEIT_MSG_DEFS = {
    "object_recognition_msgs/msg/ObjectType": (
        "string key\n"
        "string db\n"
    ),
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
        self._thread = threading.Thread(target=self._run, name="zenoh-source", daemon=True)
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
                cfg = zenoh.Config()
                cfg.insert_json5("mode", json.dumps(self.mode))
                if self.connect:
                    endpoints = self.connect if isinstance(self.connect, list) else [self.connect]
                    cfg.insert_json5("connect/endpoints", json.dumps(endpoints))
                self._session = zenoh.open(cfg)
                log.info("zenoh session open (mode=%s, connect=%s)", self.mode, self.connect or "scout")
                if self._declare(zenoh):
                    self._stop.wait()  # connected; zenoh delivers on its own threads
                    break
            except Exception as exc:
                log.warning("zenoh connect failed (mode=%s connect=%s): %s", self.mode, self.connect, exc)
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
                log.warning("no message type known for %s; set topic_types in config", topic)
                continue
            if msgtype not in self._typestore.types:
                log.warning("type %s has no schema over zenoh — skipping %s", msgtype, topic)
                continue
            key = self.key_template.format(domain=self.domain_id, topic=topic.lstrip("/"))
            self._subs.append(self._session.declare_subscriber(key, self._make_cb(topic, msgtype)))
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
