"""Robot-agnostic mapping from ROS messages to :class:`~twinlink.state.RobotState`.

A :class:`RobotMapping` is *configuration*, not code: it names which topics on
*your* robot carry joint states, TF, odometry and camera images.  The decoders
below translate the standard ``sensor_msgs`` / ``nav_msgs`` / ``tf2_msgs`` types
into the state model.  Because both the MCAP reader (``rosbags``) and live
``rclpy`` expose messages through the same attribute interface
(``msg.header.stamp.sec`` etc.), the very same mapping drives mock and live
modes unchanged.

Adapting TwinLink to a different robot is therefore usually just a new YAML file -- see ``configs/`` for an example.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import struct

import numpy as np

from .state import CameraFrame, ObstacleCloud, PlannedTrajectory, RobotState, Transform

# Default ROS message type per role -- used by the live source to pick the message class to subscribe with.  Override
# per-topic in YAML if needed.
ROLE_DEFAULT_TYPE = {
    "joint_states": "sensor_msgs/msg/JointState",
    "tf": "tf2_msgs/msg/TFMessage",
    "tf_static": "tf2_msgs/msg/TFMessage",
    "odom": "nav_msgs/msg/Odometry",
    "image": "sensor_msgs/msg/Image",
    "compressed_image": "sensor_msgs/msg/CompressedImage",
    "camera_info": "sensor_msgs/msg/CameraInfo",
    "points": "sensor_msgs/msg/PointCloud2",
    "planned_path": "moveit_msgs/msg/DisplayTrajectory",
    "string": "std_msgs/msg/String",
}


def stamp_to_sec(stamp) -> float:
    """builtin_interfaces/Time ─▶ float seconds (tolerant of missing fields)."""
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except AttributeError:
        return 0.0


@dataclass
class CameraMap:
    name: str
    image_topic: str | None = None
    info_topic: str | None = None
    is_depth: bool = False
    #: Store CompressedImage payloads undecoded (CameraFrame.raw); the consumer
    #: decodes via CameraFrame.ensure_decoded().  Keeps a live source's ingest
    #: thread ahead of the wire rate -- see CameraFrame.  Raw (uncompressed)
    #: Image messages are unaffected.  Off by default: eager consumers
    #: (MujocoSink, TwinlinkCamera, ...) read ``frame.image`` directly.
    lazy_decode: bool = False


@dataclass
class RobotMapping:
    """Declarative description of how a robot's topics map into RobotState."""

    joint_states_topics: list[str] = field(default_factory=list)
    tf_topics: list[str] = field(default_factory=list)
    tf_static_topics: list[str] = field(default_factory=list)
    odom_topic: str | None = None
    base_link: str = "base_link"
    odom_frame: str = "odom"
    cameras: list[CameraMap] = field(default_factory=list)
    # Obstacle point clouds (name ─▶ topic) and MoveIt's planned path.
    points_topics: dict[str, str] = field(default_factory=dict)
    planned_path_topic: str | None = None
    points_max: int = 60000  # subsample huge clouds before storing
    # Plain std_msgs/String topics (name ─▶ topic): the latest payload lands in ``state.extra(name)`` — e.g. the
    # /twin/arm_state JSON downlink.
    string_topics: dict[str, str] = field(default_factory=dict)

    # Joint-name handling: ROS joint name ─▶ simulator/model joint name.
    joint_remap: dict[str, str] = field(default_factory=dict)
    # If set, only these (ROS) joint names are ingested.
    joint_include: list[str] | None = None

    # Odometry frequently carries an absolute (e.g. UTM/GPS) origin that is useless for a local twin.  When true, the
    # base pose is tracked relative to the first odom sample seen.
    base_pose_relative_to_start: bool = False

    # Optional explicit per-topic ROS type (needed only for live mode when the role default is not appropriate).
    topic_types: dict[str, str] = field(default_factory=dict)

    _origin: np.ndarray | None = field(default=None, init=False, repr=False)
    # Parsed 3x3 K per camera name (camera_info streams at frame rate but the intrinsics are constant; see
    # _decode_camera_info).
    _info_cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # introspection used by the sources
    # ------------------------------------------------------------------ #
    def topics(self) -> list[str]:
        ts: list[str] = []
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
        ts += list(self.string_topics.values())
        return sorted(set(ts))

    def role_of(self, topic: str) -> str | None:
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
        if topic in self.string_topics.values():
            return "string"
        return None

    def topic_type(self, topic: str) -> str | None:
        if topic in self.topic_types:
            return self.topic_types[topic]
        role = self.role_of(topic)
        return ROLE_DEFAULT_TYPE.get(role) if role else None

    # ------------------------------------------------------------------ #
    # the single entry point used by every source
    # ------------------------------------------------------------------ #
    def apply(self, topic: str, msgtype: str, msg, state: RobotState, recv_stamp: float = 0.0) -> None:
        """Decode one message into ``state``.

        ``recv_stamp`` (epoch seconds, optional): when the transport knows the moment a relaying bridge received the
        message (foxglove ws MessageData carries one), it is forwarded onto image frames for latency splitting.
        """
        role = self.role_of(topic)
        if role == "joint_states":
            self._decode_joint_states(msg, state)
        elif role in ("tf", "tf_static"):
            self._decode_tf(msg, state)
        elif role == "odom":
            self._decode_odom(msg, state)
        elif role == "image":
            self._decode_image(msg, msgtype, state, self._camera_for(topic, "image"), recv_stamp=recv_stamp)
        elif role == "camera_info":
            self._decode_camera_info(msg, state, self._camera_for(topic, "camera_info"))
        elif role == "points":
            self._decode_points(msg, state, self._name_for_points(topic))
        elif role == "planned_path":
            self._decode_planned_path(msg, state)
        elif role == "string":
            state.set_extra(self._name_for_string(topic), str(msg.data))

    def _name_for_points(self, topic: str) -> str:
        for name, t in self.points_topics.items():
            if t == topic:
                return name
        return topic

    def _name_for_string(self, topic: str) -> str:
        for name, t in self.string_topics.items():
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

    def _decode_image(self, msg, msgtype: str, state: RobotState, cam: CameraMap, recv_stamp: float = 0.0) -> None:
        import time as _time

        arrival = _time.monotonic()
        raw_payload: bytes | None = None
        raw_format = ""
        if _is_compressed_image(msgtype, msg):
            if cam.lazy_decode:
                # Store the compressed payload as-is; the consumer decodes (CameraFrame.ensure_decoded).  Width/height
                # are unknown until then.
                image, encoding = None, ""
                height = width = 0
                raw_payload = _msg_bytes(msg)
                raw_format = str(getattr(msg, "format", "") or "")
            else:
                image, encoding = compressed_image_to_numpy(msg, is_depth=cam.is_depth)
                height, width = int(image.shape[0]), int(image.shape[1])
        else:
            image = image_to_numpy(msg)
            encoding = str(msg.encoding)
            height, width = int(msg.height), int(msg.width)
        state.set_camera(
            cam.name,
            CameraFrame(
                image=image,
                encoding=encoding,
                stamp=stamp_to_sec(msg.header.stamp),
                frame_id=msg.header.frame_id,
                width=width,
                height=height,
                raw=raw_payload,
                raw_format=raw_format,
                is_depth=cam.is_depth,
                arrival_monotonic=arrival,
                bridge_recv_stamp=recv_stamp,
            ),
        )

    def _decode_camera_info(self, msg, state: RobotState, cam: CameraMap) -> None:
        # Intrinsics are constant per session but stream at frame rate: parse the 3x3 K once, then only re-apply while
        # the state still lacks it (fresh session / after clear_camera the frame entry is gone and the re-attach relies
        # on this streaming topic).
        K = self._info_cache.get(cam.name)
        if K is None:
            try:
                K = np.array(list(msg.k), float).reshape(3, 3)
            except Exception:
                return
            self._info_cache[cam.name] = K
        frame = state.camera(cam.name)
        if frame is None or frame.intrinsics is None:
            state.set_camera_intrinsics(cam.name, K)

    def _decode_points(self, msg, state: RobotState, name: str) -> None:
        pts = pointcloud2_to_xyz(msg, max_points=self.points_max)
        state.set_obstacles(
            name, ObstacleCloud(points=pts, frame_id=msg.header.frame_id, stamp=stamp_to_sec(msg.header.stamp))
        )

    def _decode_planned_path(self, msg, state: RobotState) -> None:
        # moveit_msgs/DisplayTrajectory ─▶ the last RobotTrajectory's joint path.
        trajs = getattr(msg, "trajectory", None)
        if not trajs:
            return
        jt = trajs[-1].joint_trajectory
        names = list(jt.joint_names)
        if not names or not len(jt.points):
            return
        positions = np.array([list(p.positions) for p in jt.points], float)
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in jt.points], float)
        state.set_planned_trajectory(
            PlannedTrajectory(
                joint_names=names,
                positions=positions,
                times=times,
                stamp=(
                    stamp_to_sec(msg.trajectory_start.joint_state.header.stamp)
                    if hasattr(msg, "trajectory_start")
                    else 0.0
                ),
            )
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
            string_topics=dict(d.get("string_topics", {})),
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
    # encoding ─▶ (numpy dtype, channels)
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

    Honours the row stride (``step``).  Unknown encodings fall back to a raw ``uint8`` reshape so the pipeline never
    hard-fails on an exotic format.
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


def pointcloud2_to_xyz(msg, max_points: int | None = 60000, max_range: float | None = None) -> np.ndarray:
    """Extract finite (N, 3) xyz from a ``sensor_msgs/PointCloud2`` (any layout).

    ``max_range`` (metres, sensor frame) drops points farther than that from the origin — depth cameras such as the D435
    produce "flying pixel" artefacts at invalid-depth boundaries, and cropping by range removes them at the source.
    ``max_points=None`` disables subsampling (batch use).
    """
    fields = {f.name: (int(f.offset), int(f.datatype)) for f in msg.fields}
    if not all(k in fields for k in ("x", "y", "z")):
        return np.zeros((0, 3), float)
    raw = msg.data
    buf = np.frombuffer(
        (bytes(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else np.asarray(raw).tobytes()), dtype=np.uint8
    )
    n = int(msg.width) * int(msg.height)
    step = int(msg.point_step)
    rows = buf[: n * step].reshape(n, step)

    def column(name):
        off, dt = fields[name]
        npdt = _PC2_NP.get(dt, np.float32)
        size = np.dtype(npdt).itemsize
        return rows[:, off : off + size].copy().view(npdt).reshape(-1).astype(np.float32)

    xyz = np.stack([column("x"), column("y"), column("z")], axis=1)
    valid = np.isfinite(xyz).all(axis=1)
    if max_range is not None:
        valid &= np.linalg.norm(xyz, axis=1) <= max_range
    xyz = xyz[valid]
    if max_points is not None and len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(0).choice(len(xyz), max_points, replace=False)]
    return xyz


# ---------------------------------------------------------------------- #
# CompressedImage helpers (sensor_msgs/CompressedImage)
# ---------------------------------------------------------------------- #
# ``compressed_image_transport`` (color) puts the raw JPEG/PNG bytes straight in
# ``msg.data`` -- no header.  ``compressed_depth_image_transport`` (depth) prepends
# a 12-byte ``ConfigHeader`` (4-byte format enum + 2 float ``depthParam``) before
# the PNG or RVL payload; see upstream ``codec.cpp``.  We branch on ``is_depth``
# (the ``CameraMap`` flag) and, for depth, on ``msg.format`` ("... compressedDepth
# rvl" ─▶ RVL, otherwise PNG) -- exactly the way the C++ codec does.


# ConfigHeader: 4-byte enum + 2x float32 (depthQuantA, depthQuantB).
_DEPTH_CONFIG_HEADER = 12


def _is_compressed_image(msgtype: str, msg) -> bool:
    """True for ``sensor_msgs/CompressedImage`` (by type, or by shape as fallback)."""
    if msgtype and msgtype.endswith("CompressedImage"):
        return True
    return not hasattr(msg, "width")


def _msg_bytes(msg) -> bytes:
    raw = msg.data
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    return bytes(bytearray(raw))


def compressed_image_to_numpy(msg, is_depth: bool = False) -> tuple[np.ndarray, str]:
    """Decode a ``sensor_msgs/CompressedImage`` into ``(HxW[xC] array, encoding)``.

    * Color (``is_depth=False``): JPEG/PNG via ``cv2.imdecode`` ─▶ BGR8, encoding
      ``"bgr8"`` (the HSV detector wants BGR).
    * Depth (``is_depth=True``): strips the 12-byte ConfigHeader, then PNG
      (``cv2.imdecode(IMREAD_UNCHANGED)`` ─▶ uint16 mm) or RVL
      (:func:`rvl_decompress` ─▶ uint16 mm).  A 32FC1 source is dequantized to
      float metres via the header's ``depthParam`` (0 ─▶ NaN).  Returns encoding
      ``"16uc1"`` (uint16 mm) or ``"32fc1"`` (float metres).
    """
    data = _msg_bytes(msg)
    fmt = str(getattr(msg, "format", "") or "")
    return decode_compressed_bytes(data, fmt, is_depth=is_depth)


def decode_compressed_bytes(
    data: bytes, fmt: str, *, is_depth: bool = False, allow_rvl: bool = True
) -> tuple[np.ndarray, str]:
    """Bytes-level CompressedImage decode -- see :func:`compressed_image_to_numpy`.

    This is the entry point of the *lazy* path (``CameraFrame.ensure_decoded``), which passes ``allow_rvl=False``: the
    pure-Python RVL decoder takes seconds per 640x480 frame, unusable live -- reject loudly (publish PNG instead:
    ``format`` parameter of the compressed_depth_image_transport publisher) rather than silently stalling the pipeline.
    Offline/batch use (MCAP) keeps RVL support via the default.
    """
    fmt = (fmt or "").lower()

    # Reject the live-incompatible RVL path *before* importing cv2: the lazy path runs without opencv installed (CI /
    # minimal installs), and this raise is the whole point -- it must not be shadowed by a ModuleNotFoundError.
    if is_depth and "rvl" in fmt and not allow_rvl:
        raise ValueError(
            f"RVL depth stream rejected (format={fmt!r}): the pure-Python RVL "
            "decoder is too slow for a live pipeline -- set the robot's "
            "compressed_depth_image_transport 'format' parameter to 'png'"
        )

    import cv2

    if not is_depth:
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"compressed color decode failed (format={fmt!r})")
        return img, "bgr8"

    return _decode_depth_compressed(data, fmt)


def _decode_depth_compressed(data: bytes, fmt: str) -> tuple[np.ndarray, str]:
    import cv2

    if len(data) < _DEPTH_CONFIG_HEADER:
        raise ValueError(
            f"depth CompressedImage shorter than ConfigHeader (12 B): {len(data)} B, "
            f"format={fmt!r} -- empty data means the plain 'compressed' (JPEG) "
            "transport, which cannot encode 16-bit depth; subscribe to the "
            "'.../compressedDepth' topic instead"
        )
    header, payload = data[:_DEPTH_CONFIG_HEADER], data[_DEPTH_CONFIG_HEADER:]
    enc_prefix = fmt.split(";", 1)[0].strip()
    is_float = "32f" in enc_prefix
    depth_a, depth_b = struct.unpack_from("<ff", header, 4)

    if "rvl" in fmt:
        cols, rows = struct.unpack_from("<II", payload, 0)
        if rows == 0 or cols == 0:
            raise ValueError(f"malformed RVL header: {cols}x{rows}")
        raw = rvl_decompress(payload[8:], rows * cols).reshape(rows, cols)
    else:  # png / depth_png / default
        buf = np.frombuffer(payload, dtype=np.uint8)
        raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"depth PNG decode failed (format={fmt!r})")

    if is_float:
        inv = raw.astype(np.float32)
        depth = np.where(inv > 0, depth_a / (inv - depth_b), np.float32(np.nan))
        return depth, "32fc1"
    return raw.astype(np.uint16, copy=False), "16uc1"


def rvl_decompress(data: bytes, num_pixels: int) -> np.ndarray:
    """Decompress an RVL bitstream ─▶ flat ``uint16`` array (``num_pixels`` long).

    Faithful Python port of ``compressed_depth_image_transport::RvlCodec::
    DecompressRVL`` (Wilson, "Fast Lossless Depth Image Compression", SIGCHI'17):
    VLE-packed 3-bit payloads in 4-bit nibbles of little-endian 32-bit words,
    run-length over zero/nonzero runs and zigzag-delta over the nonzero stream.
    """
    words = np.frombuffer(data, dtype="<u4")
    state = {"word": 0, "nib": 0, "wp": 0}

    def _vle() -> int:
        value, bits = 0, 29
        while True:
            if state["nib"] == 0:
                state["word"] = int(words[state["wp"]])
                state["wp"] += 1
                state["nib"] = 8
            nibble = (state["word"] >> 28) & 0xF
            value |= (nibble & 0x7) << (29 - bits)
            state["word"] = (state["word"] << 4) & 0xFFFFFFFF
            state["nib"] -= 1
            bits -= 3
            if not (nibble & 0x8):
                return value

    out = np.zeros(num_pixels, dtype=np.uint16)
    idx, remaining, previous = 0, num_pixels, 0
    while remaining:
        zeros = _vle()
        remaining -= zeros
        if remaining < 0 or idx + zeros > num_pixels:
            raise ValueError("malformed RVL stream (zero run overruns image)")
        for _ in range(zeros):
            out[idx] = 0
            idx += 1
        nonzeros = _vle()
        remaining -= nonzeros
        if remaining < 0 or idx + nonzeros > num_pixels:
            raise ValueError("malformed RVL stream (nonzero run overruns image)")
        for _ in range(nonzeros):
            positive = _vle()
            delta = (positive >> 1) ^ -(positive & 1)
            current = (previous + delta) & 0xFFFF
            out[idx] = current
            idx += 1
            previous = current
    return out
