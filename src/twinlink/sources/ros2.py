"""Live ROS 2 source.

Subscribes to exactly the topics named in the mapping and feeds every message through ``mapping.apply(...)`` -- the same
path the MCAP source uses, so the twin behaves identically online and offline.

``rclpy`` is imported lazily so the rest of TwinLink (and the MuJoCo/MCAP example) works on a machine without a ROS
installation.

Pluggable middleware ("BabyROS" & co.)
--------------------------------------
The rclpy specifics live in three small overridable methods -- ``_init_node``,
``_subscribe`` and ``_spin``.  To drive TwinLink from a different middleware,
subclass and override those three; the decoding/mapping layer is reused as-is::

    class BabyRosSource(Ros2Source):
        def _init_node(self): ...
        def _subscribe(self, topic, type_str, on_msg): ...
        def _spin(self): ...
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Callable, List, Optional

from .base import StateSource

log = logging.getLogger("twinlink.ros2")

# Topics for which we prefer best-effort QoS (high-rate sensor streams).
_BEST_EFFORT_ROLES = {"image", "camera_info", "tf", "tf_static"}


def resolve_msg_class(type_str: str):
    """'sensor_msgs/msg/JointState' ─▶ the rclpy message class."""
    parts = type_str.replace(".msg.", "/msg/").split("/")
    pkg, name = parts[0], parts[-1]
    module = importlib.import_module(f"{pkg}.msg")
    return getattr(module, name)


class Ros2Source(StateSource):
    def __init__(self, node_name: str = "twinlink_listener") -> None:
        super().__init__()
        self.node_name = node_name
        self._node = None
        self._executor = None
        self._thread: Optional[threading.Thread] = None
        self._owns_rclpy = False

    # ------------------------------------------------------------------ #
    def start(self) -> "Ros2Source":
        assert self.state is not None and self.mapping is not None, "bind() before start()"
        self._init_node()
        for topic in self.mapping.topics():
            type_str = self.mapping.topic_type(topic)
            if not type_str:
                log.warning("No message type known for %s; skipping. Set topic_types in config.", topic)
                continue
            role = self.mapping.role_of(topic)
            self._subscribe(topic, type_str, self._make_callback(topic, type_str), role)
        self._thread = threading.Thread(target=self._spin, name="ros2-source", daemon=True)
        self._thread.start()
        self._running = True
        return self

    def stop(self) -> None:
        self._running = False
        try:
            if self._executor is not None:
                self._executor.shutdown()
            if self._node is not None:
                self._node.destroy_node()
        finally:
            if self._owns_rclpy:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _make_callback(self, topic: str, type_str: str) -> Callable:
        def _cb(msg) -> None:
            try:
                self.mapping.apply(topic, type_str, msg, self.state)
            except Exception as exc:  # never let a bad frame kill the executor
                log.debug("apply failed on %s: %s", topic, exc)

        return _cb

    # ------------------------------------------------------------------ #
    # middleware hooks -- override these for non-rclpy backends
    # ------------------------------------------------------------------ #
    def _init_node(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Live mode needs ROS 2 (rclpy). On a machine without ROS, use "
                "McapSource (mock mode) instead, or subclass Ros2Source for your "
                "middleware (e.g. BabyROS)."
            ) from exc
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._node = Node(self.node_name)
        self._executor = rclpy.executors.MultiThreadedExecutor()
        self._executor.add_node(self._node)

    def _subscribe(self, topic: str, type_str: str, on_msg: Callable, role: Optional[str]) -> None:
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

        msg_class = resolve_msg_class(type_str)
        reliability = ReliabilityPolicy.BEST_EFFORT if role in _BEST_EFFORT_ROLES else ReliabilityPolicy.RELIABLE
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=reliability)
        self._node.create_subscription(msg_class, topic, on_msg, qos)
        log.info("subscribed %s [%s] reliability=%s", topic, type_str, reliability.name)

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as exc:  # pragma: no cover
            log.debug("executor stopped: %s", exc)
