"""In-memory digital-twin state.

:class:`RobotState` is the heart of TwinLink: a thread-safe, robot-agnostic
snapshot of the *live* robot.  Sources (ROS 2, an MCAP recording, a bare URDF)
write into it; sinks (MuJoCo, Isaac Sim, ...) read from it.  Nothing in this
module knows about a particular robot model -- joints, frames and cameras are
addressed by name.

The contract is intentionally tiny so that it is cheap to update at sensor rates and cheap to sample from a render loop:

    source thread(s)  --writes-->  RobotState  <--reads--  sink (main thread)

All access is guarded by a single re-entrant lock.  Writers bump a monotonic ``revision`` counter so a consumer can
cheaply detect "did anything change".
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _now() -> float:
    return time.time()


@dataclass
class JointState:
    """State of a single named joint (SI units: rad / rad·s⁻¹ / N·m)."""

    position: float = 0.0
    velocity: Optional[float] = None
    effort: Optional[float] = None
    stamp: float = 0.0  # source clock, seconds


@dataclass
class Transform:
    """A rigid transform ``frame_id -> child_frame_id``.

    Rotation is stored as an ``xyzw`` quaternion to match ROS / geometry_msgs.
    """

    translation: np.ndarray  # (3,) float64
    rotation: np.ndarray  # (4,) float64, xyzw
    stamp: float = 0.0
    frame_id: str = ""
    child_frame_id: str = ""

    @staticmethod
    def identity() -> "Transform":
        return Transform(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))


@dataclass
class CameraFrame:
    """A single image plus optional intrinsics, addressed by sensor name.

    Two storage modes:

    * **eager** (default): ``image`` holds the decoded array, ``raw`` is None.
    * **lazy** (``CameraMap.lazy_decode``): ``image`` is None and ``raw`` holds
      the still-compressed ``CompressedImage`` payload; the *consumer* calls
      :meth:`ensure_decoded` on the frame it actually reads.  This keeps the
      ingest thread of a live source (e.g. ``FoxgloveSource``) free of
      JPEG/PNG decoding -- with a 30 Hz colour + 15 Hz depth stream only the
      handful of frames perception actually consumes get decoded instead of
      every frame on the wire, so the receive loop cannot fall behind the
      arrival rate (the root cause of ever-growing camera latency).
    """

    image: Optional[np.ndarray]  # HxWx3 uint8 (color) or HxW (depth); None = lazy
    encoding: str = "rgb8"
    stamp: float = 0.0
    frame_id: str = ""
    intrinsics: Optional[np.ndarray] = None  # 3x3 K
    width: int = 0
    height: int = 0
    #: Still-compressed payload (lazy mode); cleared after ensure_decoded().
    raw: Optional[bytes] = None
    #: The CompressedImage ``format`` string (lazy mode), e.g. "rgb8; jpeg".
    raw_format: str = ""
    is_depth: bool = False
    #: time.monotonic() on this machine when the frame arrived at the source.
    #: Freshness gate for consumers: frames arrived after a motion finished
    #: were captured at (or after) the settled pose.  0.0 = unknown/legacy.
    arrival_monotonic: float = 0.0
    #: Receive timestamp of the relaying bridge (epoch seconds), if the
    #: transport carries one (foxglove ws MessageData).  Splits sensor->bridge
    #: from bridge->client lag when clocks differ.  0.0 = unknown.
    bridge_recv_stamp: float = 0.0

    def ensure_decoded(self) -> bool:
        """Decode a lazy frame in place; True if an image is available.

        Idempotent: eager frames and already-decoded frames return True immediately.  Decoding errors are logged once
        per process and yield False (callers treat it as "no frame yet").  RVL depth payloads are rejected here -- the
        pure-Python RVL decoder is far too slow for a live pipeline (offline/MCAP use goes through the eager path, which
        still supports it).
        """
        if self.image is not None:
            return True
        if self.raw is None:
            return False
        from .mapping import decode_compressed_bytes  # local: avoid import cycle

        try:
            image, encoding = decode_compressed_bytes(
                self.raw, self.raw_format, is_depth=self.is_depth, allow_rvl=False
            )
        except Exception as exc:
            logging.getLogger("twinlink.state").warning(
                "lazy image decode failed (format=%r, depth=%s): %s", self.raw_format, self.is_depth, exc
            )
            self.raw = None  # do not retry a poisoned payload
            return False
        self.image = image
        self.encoding = encoding
        self.height, self.width = int(image.shape[0]), int(image.shape[1])
        self.raw = None  # free the compressed payload
        return True


@dataclass
class ObstacleCloud:
    """A set of obstacle points (e.g. a sensor point cloud or octomap voxels)."""

    points: np.ndarray  # (N, 3) float, in `frame_id`
    frame_id: str = ""
    stamp: float = 0.0


@dataclass
class PlannedTrajectory:
    """A planned joint-space path (e.g. from MoveIt's display_planned_path)."""

    joint_names: List[str]
    positions: np.ndarray  # (K, J) waypoint positions
    times: np.ndarray  # (K,) time_from_start (seconds)
    stamp: float = 0.0

    def duration(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0


def _transform_matrix(tf: "Transform") -> np.ndarray:
    """``Transform`` (xyzw quaternion) as a 4x4 homogeneous matrix."""
    x, y, z, w = (float(v) for v in tf.rotation)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError(
            f"degenerate quaternion {tuple(tf.rotation)!r} on edge "
            f"{tf.frame_id!r} -> {tf.child_frame_id!r}: refusing to read it "
            "as 'no rotation'."
        )
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    mat = np.eye(4)
    mat[:3, :3] = np.array(
        [[1.0 - (yy + zz), xy - wz, xz + wy], [xy + wz, 1.0 - (xx + zz), yz - wx], [xz - wy, yz + wx, 1.0 - (xx + yy)]]
    )
    mat[:3, 3] = np.asarray(tf.translation, dtype=float)
    return mat


class RobotState:
    """Thread-safe, model-agnostic snapshot of a robot."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._joints: Dict[str, JointState] = {}
        self._transforms: Dict[Tuple[str, str], Transform] = {}
        self._base_pose: Optional[Transform] = None
        self._cameras: Dict[str, CameraFrame] = {}
        self._obstacles: Dict[str, ObstacleCloud] = {}
        self._planned_trajectory: Optional[PlannedTrajectory] = None
        self._extras: Dict[str, Any] = {}
        self._revision = 0
        self._last_update = 0.0

    # ------------------------------------------------------------------ #
    # writers (called by sources)
    # ------------------------------------------------------------------ #
    def update_joint(
        self,
        name: str,
        position: float,
        velocity: Optional[float] = None,
        effort: Optional[float] = None,
        stamp: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._joints[name] = JointState(
                float(position),
                None if velocity is None else float(velocity),
                None if effort is None else float(effort),
                _now() if stamp is None else float(stamp),
            )
            self._touch()

    def update_joints(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        velocities: Optional[Sequence[float]] = None,
        efforts: Optional[Sequence[float]] = None,
        stamp: Optional[float] = None,
    ) -> None:
        st = _now() if stamp is None else float(stamp)
        with self._lock:
            for i, n in enumerate(names):
                if i >= len(positions):
                    break
                self._joints[n] = JointState(
                    float(positions[i]),
                    (float(velocities[i]) if velocities is not None and i < len(velocities) else None),
                    (float(efforts[i]) if efforts is not None and i < len(efforts) else None),
                    st,
                )
            self._touch()

    def set_transform(self, transform: Transform) -> None:
        with self._lock:
            self._transforms[(transform.frame_id, transform.child_frame_id)] = transform
            self._touch()

    def set_base_pose(self, transform: Transform) -> None:
        with self._lock:
            self._base_pose = transform
            self._touch()

    def set_camera(self, name: str, frame: CameraFrame) -> None:
        with self._lock:
            # Preserve intrinsics that may have arrived on a separate topic.
            if frame.intrinsics is None:
                prev = self._cameras.get(name)
                if prev is not None and prev.intrinsics is not None:
                    frame.intrinsics = prev.intrinsics
            self._cameras[name] = frame
            self._touch()

    def clear_camera(self, name: str) -> None:
        """Drop a camera's cached frame.

        Consumers call this when their source session (re)starts, so a poll loop never mistakes the previous session's
        last frame for live data; the entry repopulates when the new session's first image arrives (intrinsics re-attach
        from the streaming CameraInfo topic).
        """
        with self._lock:
            self._cameras.pop(name, None)
            self._touch()

    def set_camera_intrinsics(self, name: str, intrinsics: np.ndarray) -> None:
        with self._lock:
            prev = self._cameras.get(name)
            if prev is not None:
                prev.intrinsics = intrinsics
            else:
                self._extras[f"camera_info/{name}"] = intrinsics
            self._touch()

    def set_obstacles(self, name: str, cloud: ObstacleCloud) -> None:
        with self._lock:
            self._obstacles[name] = cloud
            self._touch()

    def set_planned_trajectory(self, traj: PlannedTrajectory) -> None:
        with self._lock:
            self._planned_trajectory = traj
            self._touch()

    def set_extra(self, key: str, value: Any) -> None:
        with self._lock:
            self._extras[key] = value
            self._touch()

    def _touch(self) -> None:
        self._revision += 1
        self._last_update = _now()

    # ------------------------------------------------------------------ #
    # readers (called by sinks)
    # ------------------------------------------------------------------ #
    def joint_names(self) -> List[str]:
        with self._lock:
            return list(self._joints.keys())

    def joint(self, name: str) -> Optional[JointState]:
        with self._lock:
            return self._joints.get(name)

    def joint_position(self, name: str, default: float = float("nan")) -> float:
        with self._lock:
            j = self._joints.get(name)
            return j.position if j is not None else default

    def joint_positions(self, names: Iterable[str]) -> np.ndarray:
        with self._lock:
            return np.array([self._joints[n].position if n in self._joints else np.nan for n in names], dtype=float)

    def joints(self) -> Dict[str, JointState]:
        with self._lock:
            return dict(self._joints)

    def base_pose(self) -> Optional[Transform]:
        with self._lock:
            return self._base_pose

    def transform(self, parent: str, child: str) -> Optional[Transform]:
        with self._lock:
            return self._transforms.get((parent, child))

    def transforms(self) -> Dict[Tuple[str, str], Transform]:
        with self._lock:
            return dict(self._transforms)

    def chain(self, source: str, target: str) -> Optional[np.ndarray]:
        """4x4 matrix mapping points from ``source`` into ``target``.

        ``transform()`` is a plain dict hit and therefore only answers for edges that were published as such.  A
        camera's optical frame sits several hops from the world frame, so back-projection needs the composed chain.
        Breadth-first over the undirected edge set, each edge used forwards or inverted as required.

        Returns ``None`` when the frames are not connected -- deliberately, rather than an identity matrix: a silent
        identity would place every obstacle at the robot's origin while still producing a plausible looking point cloud.

        ``source == target`` is identity only for a frame the graph actually KNOWS -- the shortcut must NOT run before
        the look-up.  Taken early, ``chain("nope", "nope")`` hands back ``eye(4)`` for a frame that appears in no edge
        at all, and a recording whose ``frame_id`` happens to equal the caller's world frame gets exactly the silent
        identity this docstring promises to refuse.
        """
        with self._lock:
            edges = dict(self._transforms)

        if source == target:
            known = any(source in edge for edge in edges)
            return np.eye(4) if known else None

        # Undirected adjacency: tf edges are stored as (parent, child) but a chain may traverse either way.
        adjacency: Dict[str, List[str]] = {}
        for parent, child in edges:
            adjacency.setdefault(child, []).append(parent)
            adjacency.setdefault(parent, []).append(child)

        previous: Dict[str, str] = {}
        queue: List[str] = [source]
        seen = {source}
        while queue:
            node = queue.pop(0)
            if node == target:
                break
            for neighbour in adjacency.get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    previous[neighbour] = node
                    queue.append(neighbour)
        if target not in seen:
            return None

        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        path.reverse()  # source ... target

        # A stored transform (parent, child) maps points from the CHILD frame into the PARENT frame -- the ROS
        # convention.  Walking a -> b: if b is a's parent the stored matrix already points our way; if a is b's parent
        # we need its inverse.  Deciding this from the edge dictionary rather than from a direction flag is what keeps
        # the two cases from being swapped.
        mat = np.eye(4)
        for a, b in zip(path, path[1:]):
            if (b, a) in edges:
                step = _transform_matrix(edges[(b, a)])
            else:
                step = np.linalg.inv(_transform_matrix(edges[(a, b)]))
            mat = step @ mat
        return mat

    def camera(self, name: str) -> Optional[CameraFrame]:
        with self._lock:
            return self._cameras.get(name)

    def cameras(self) -> Dict[str, CameraFrame]:
        with self._lock:
            return dict(self._cameras)

    def obstacles(self, name: Optional[str] = None):
        with self._lock:
            if name is not None:
                return self._obstacles.get(name)
            return dict(self._obstacles)

    def planned_trajectory(self) -> Optional[PlannedTrajectory]:
        with self._lock:
            return self._planned_trajectory

    def extra(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._extras.get(key, default)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def last_update(self) -> float:
        with self._lock:
            return self._last_update

    def summary(self) -> str:
        with self._lock:
            return (
                f"RobotState(joints={len(self._joints)}, tf={len(self._transforms)}, "
                f"cameras={len(self._cameras)}, "
                f"base_pose={'set' if self._base_pose is not None else 'none'}, "
                f"rev={self._revision})"
            )
