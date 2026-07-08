"""Robot-agnostic mapping from ROS messages to :class:`RobotState`.

A :class:`RobotMapping` is *configuration*, not code: it names which topics on
*your* robot carry joint states, TF, odometry and camera images.  The decoders
below translate the standard ``sensor_msgs`` / ``nav_msgs`` / ``tf2_msgs`` types
into the state model.  Because both the MCAP reader (``rosbags``) and live
``rclpy`` expose messages through the same attribute interface
(``msg.header.stamp.sec`` etc.), the very same mapping drives mock and live
modes unchanged.

Adapting TwinLink to a different robot is therefore usually just a new YAML
file -- see ``configs/`` for an example.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .state import CameraFrame, ObstacleCloud, PlannedTrajectory, RobotState, Transform

# Default ROS message type per role -- used by the live source to pick the
# message class to subscribe with.  Override per-topic in YAML if needed.
ROLE_DEFAULT_TYPE = {
    "joint_states": "sensor_msgs/msg/JointState",
    "tf": "tf2_msgs/msg/TFMessage",
    "tf_static": "tf2_msgs/msg/TFMessage",
    "odom": "nav_msgs/msg/Odometry",
    "image": "sensor_msgs/msg/Image",
    "camera_info": "sensor_msgs/msg/CameraInfo",
    "points": "sensor_msgs/msg/PointCloud2",
    "planned_path": "moveit_msgs/msg/DisplayTrajectory",
}


def stamp_to_sec(stamp) -> float:
    """builtin_interfaces/Time -> float seconds (tolerant of missing fields)."""
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except AttributeError:
        return 0.0


@dataclass
class CameraMap:
    name: str
    image_topic: Optional[str] = None
    info_topic: Optional[str] = None
    is_depth: bool = False


@dataclass
class RobotMapping:
    """Declarative description of how a robot's topics map into RobotState."""

    joint_states_topics: List[str] = field(default_factory=list)
    tf_topics: List[str] = field(default_factory=list)
    tf_static_topics: List[str] = field(default_factory=list)
    odom_topic: Optional[str] = None
    base_link: str = "base_link"
    odom_frame: str = "odom"
    cameras: List[CameraMap] = field(default_factory=list)
    # Obstacle point clouds (name -> topic) and MoveIt's planned path.
    points_topics: Dict[str, str] = field(default_factory=dict)
    planned_path_topic: Optional[str] = None
    points_max: int = 60000  # subsample huge clouds before storing

    # Joint-name handling: ROS joint name -> simulator/model joint name.
    joint_remap: Dict[str, str] = field(default_factory=dict)
    # If set, only these (ROS) joint names are ingested.
    joint_include: Optional[List[str]] = None

    # Odometry frequently carries an absolute (e.g. UTM/GPS) origin that is
    # useless for a local twin.  When true, the base pose is tracked relative
    # to the first odom sample seen.
    base_pose_relative_to_start: bool = False

    # Optional explicit per-topic ROS type (needed only for live mode when the
    # role default is not appropriate).
    topic_types: Dict[str, str] = field(default_factory=dict)

    _origin: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # introspection used by the sources
    # ------------------------------------------------------------------ #
    def topics(self) -> List[str]:
        ts: List[str] = []
        ts += self.joint_states_topics
        ts += self.tf_topics
        ts += self.tf_static_topics
        if self.odom_topic:
            ts.append(self.odom_topic)
        for cam in self.cameras:
            if cam.image_topic:
                ts.append(cam.image_topic)
            if cam.info_topic:
                ts.append(cam.info_topic)
        ts += list(self.points_topics.values())
        if self.planned_path_topic:
            ts.append(self.planned_path_topic)
        return sorted(set(ts))

    def role_of(self, topic: str) -> Optional[str]:
        if topic in self.joint_states_topics:
            return "joint_states"
        if topic in self.tf_topics:
            return "tf"
        if topic in self.tf_static_topics:
            return "tf_static"
        if self.odom_topic and topic == self.odom_topic:
            return "odom"
        for cam in self.cameras:
            if topic == cam.image_topic:
                return "image"
            if topic == cam.info_topic:
                return "camera_info"
        if topic in self.points_topics.values():
            return "points"
        if self.planned_path_topic and topic == self.planned_path_topic:
            return "planned_path"
        return None

    def topic_type(self, topic: str) -> Optional[str]:
        if topic in self.topic_types:
            return self.topic_types[topic]
        role = self.role_of(topic)
        return ROLE_DEFAULT_TYPE.get(role) if role else None

    # ------------------------------------------------------------------ #
    # the single entry point used by every source
    # ------------------------------------------------------------------ #
    def apply(self, topic: str, msgtype: str, msg, state: RobotState) -> None:
        role = self.role_of(topic)
        if role == "joint_states":
            self._decode_joint_states(msg, state)
        elif role in ("tf", "tf_static"):
            self._decode_tf(msg, state)
        elif role == "odom":
            self._decode_odom(msg, state)
        elif role == "image":
            self._decode_image(msg, state, self._camera_for(topic, "image"))
        elif role == "camera_info":
            self._decode_camera_info(msg, state, self._camera_for(topic, "camera_info"))
        elif role == "points":
            self._decode_points(msg, state, self._name_for_points(topic))
        elif role == "planned_path":
            self._decode_planned_path(msg, state)

    def _name_for_points(self, topic: str) -> str:
        for name, t in self.points_topics.items():
            if t == topic:
                return name
        return topic

    def _camera_for(self, topic: str, kind: str) -> CameraMap:
        for cam in self.cameras:
            if kind == "image" and topic == cam.image_topic:
                return cam
            if kind == "camera_info" and topic == cam.info_topic:
                return cam
        raise KeyError(topic)

    # ------------------------------------------------------------------ #
    # decoders
    # ------------------------------------------------------------------ #
    def _map_name(self, name: str) -> str:
        return self.joint_remap.get(name, name)

    def _decode_joint_states(self, msg, state: RobotState) -> None:
        names = list(msg.name)
        pos = list(msg.position)
        vel = list(msg.velocity) if len(getattr(msg, "velocity", [])) else None
        eff = list(msg.effort) if len(getattr(msg, "effort", [])) else None
        stamp = stamp_to_sec(msg.header.stamp)
        for i, ros_name in enumerate(names):
            if self.joint_include is not None and ros_name not in self.joint_include:
                continue
            if i >= len(pos):
                continue
            state.update_joint(
                self._map_name(ros_name),
                float(pos[i]),
                float(vel[i]) if vel is not None and i < len(vel) else None,
                float(eff[i]) if eff is not None and i < len(eff) else None,
                stamp,
            )

    def _decode_tf(self, msg, state: RobotState) -> None:
        for tr in msg.transforms:
            t = tr.transform
            state.set_transform(
                Transform(
                    np.array([t.translation.x, t.translation.y, t.translation.z], float),
                    np.array([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w], float),
                    stamp_to_sec(tr.header.stamp),
                    tr.header.frame_id,
                    tr.child_frame_id,
                )
            )

    def _decode_odom(self, msg, state: RobotState) -> None:
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        translation = np.array([p.x, p.y, p.z], float)
        if self.base_pose_relative_to_start:
            if self._origin is None:
                self._origin = translation.copy()
            translation = translation - self._origin
        state.set_base_pose(
            Transform(
                translation,
                np.array([o.x, o.y, o.z, o.w], float),
                stamp_to_sec(msg.header.stamp),
                self.odom_frame,
                getattr(msg, "child_frame_id", "") or self.base_link,
            )
        )

    def _decode_image(self, msg, state: RobotState, cam: CameraMap) -> None:
        image = image_to_numpy(msg)
        state.set_camera(
            cam.name,
            CameraFrame(
                image=image,
                encoding=str(msg.encoding),
                stamp=stamp_to_sec(msg.header.stamp),
                frame_id=msg.header.frame_id,
                width=int(msg.width),
                height=int(msg.height),
            ),
        )

    def _decode_camera_info(self, msg, state: RobotState, cam: CameraMap) -> None:
        try:
            K = np.array(list(msg.k), float).reshape(3, 3)
        except Exception:
            return
        state.set_camera_intrinsics(cam.name, K)

    def _decode_points(self, msg, state: RobotState, name: str) -> None:
        pts = pointcloud2_to_xyz(msg, max_points=self.points_max)
        state.set_obstacles(
            name,
            ObstacleCloud(points=pts, frame_id=msg.header.frame_id, stamp=stamp_to_sec(msg.header.stamp)),
        )

    def _decode_planned_path(self, msg, state: RobotState) -> None:
        # moveit_msgs/DisplayTrajectory -> the last RobotTrajectory's joint path.
        trajs = getattr(msg, "trajectory", None)
        if not trajs:
            return
        jt = trajs[-1].joint_trajectory
        names = list(jt.joint_names)
        if not names or not len(jt.points):
            return
        positions = np.array([list(p.positions) for p in jt.points], float)
        times = np.array(
            [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in jt.points], float
        )
        state.set_planned_trajectory(
            PlannedTrajectory(joint_names=names, positions=positions, times=times, stamp=stamp_to_sec(msg.trajectory_start.joint_state.header.stamp) if hasattr(msg, "trajectory_start") else 0.0)
        )

    # ------------------------------------------------------------------ #
    # construction from config
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict) -> "RobotMapping":
        cams = [
            CameraMap(
                name=c["name"],
                image_topic=c.get("image_topic"),
                info_topic=c.get("info_topic"),
                is_depth=bool(c.get("is_depth", False)),
            )
            for c in d.get("cameras", [])
        ]
        return cls(
            joint_states_topics=list(d.get("joint_states_topics", [])),
            tf_topics=list(d.get("tf_topics", [])),
            tf_static_topics=list(d.get("tf_static_topics", [])),
            odom_topic=d.get("odom_topic"),
            base_link=d.get("base_link", "base_link"),
            odom_frame=d.get("odom_frame", "odom"),
            cameras=cams,
            joint_remap=dict(d.get("joint_remap", {})),
            joint_include=d.get("joint_include"),
            base_pose_relative_to_start=bool(d.get("base_pose_relative_to_start", False)),
            topic_types=dict(d.get("topic_types", {})),
            points_topics=dict(d.get("points_topics", {})),
            planned_path_topic=d.get("planned_path_topic"),
            points_max=int(d.get("points_max", 60000)),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "RobotMapping":
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})


# ---------------------------------------------------------------------- #
# image helpers
# ---------------------------------------------------------------------- #
_ENCODING_INFO = {
    # encoding -> (numpy dtype, channels)
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1),
    "8uc1": (np.uint8, 1),
    "8uc3": (np.uint8, 3),
    "mono16": (np.uint16, 1),
    "16uc1": (np.uint16, 1),
    "32fc1": (np.float32, 1),
}


def image_to_numpy(msg) -> np.ndarray:
    """Decode a ``sensor_msgs/Image`` into an HxW[xC] numpy array.

    Honours the row stride (``step``).  Unknown encodings fall back to a raw
    ``uint8`` reshape so the pipeline never hard-fails on an exotic format.
    """
    enc = str(msg.encoding).lower()
    dtype, channels = _ENCODING_INFO.get(enc, (np.uint8, 1))

    raw = msg.data
    if isinstance(raw, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(bytes(raw), dtype=dtype)
    else:
        buf = np.asarray(raw).view(dtype).reshape(-1)

    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    itemsize = np.dtype(dtype).itemsize
    if step and step != width * channels * itemsize:
        # Stride-aware: trim padding bytes at the end of each row.
        per_row = step // itemsize
        buf = buf[: per_row * height].reshape(height, per_row)
        buf = buf[:, : width * channels]
    else:
        buf = buf[: height * width * channels]
    img = buf.reshape(height, width, channels) if channels > 1 else buf.reshape(height, width)
    return np.ascontiguousarray(img)


_PC2_NP = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def pointcloud2_to_xyz(msg, max_points: int = 60000) -> np.ndarray:
    """Extract finite (N, 3) xyz from a ``sensor_msgs/PointCloud2`` (any layout)."""
    fields = {f.name: (int(f.offset), int(f.datatype)) for f in msg.fields}
    if not all(k in fields for k in ("x", "y", "z")):
        return np.zeros((0, 3), float)
    raw = msg.data
    buf = np.frombuffer(bytes(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else np.asarray(raw).tobytes(), dtype=np.uint8)
    n = int(msg.width) * int(msg.height)
    step = int(msg.point_step)
    rows = buf[: n * step].reshape(n, step)

    def column(name):
        off, dt = fields[name]
        npdt = _PC2_NP.get(dt, np.float32)
        size = np.dtype(npdt).itemsize
        return rows[:, off : off + size].copy().view(npdt).reshape(-1).astype(np.float32)

    xyz = np.stack([column("x"), column("y"), column("z")], axis=1)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(0).choice(len(xyz), max_points, replace=False)]
    return xyz
