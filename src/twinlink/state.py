"""In-memory digital-twin state.

:class:`RobotState` is the heart of TwinLink: a thread-safe, robot-agnostic
snapshot of the *live* robot.  Sources (ROS 2, an MCAP recording, a bare URDF)
write into it; sinks (MuJoCo, Isaac Sim, ...) read from it.  Nothing in this
module knows about a particular robot model -- joints, frames and cameras are
addressed by name.

The contract is intentionally tiny so that it is cheap to update at sensor
rates and cheap to sample from a render loop:

    source thread(s)  --writes-->  RobotState  <--reads--  sink (main thread)

All access is guarded by a single re-entrant lock.  Writers bump a monotonic
``revision`` counter so a consumer can cheaply detect "did anything change".
"""
from __future__ import annotations

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
    """A single image plus optional intrinsics, addressed by sensor name."""

    image: np.ndarray  # HxWx3 uint8 (color) or HxW (depth)
    encoding: str = "rgb8"
    stamp: float = 0.0
    frame_id: str = ""
    intrinsics: Optional[np.ndarray] = None  # 3x3 K
    width: int = 0
    height: int = 0


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
                    float(velocities[i]) if velocities is not None and i < len(velocities) else None,
                    float(efforts[i]) if efforts is not None and i < len(efforts) else None,
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
            return np.array(
                [self._joints[n].position if n in self._joints else np.nan for n in names],
                dtype=float,
            )

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
