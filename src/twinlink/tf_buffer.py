"""Offline TF tree with BFS path lookup and time interpolation (no ROS).

Feeds on recorded ``tf2_msgs/TFMessage`` data (rosbags/MCAP replay, foxglove captures) and answers "map points from
frame A into frame B at time t" — including multi-hop chains and linear/slerp interpolation between samples.

``apps/octomap_explorer`` imports this from here rather than keeping a copy.

**``transforms3d`` checked and rejected** (measured 2026-08-29 against 0.4.2): 2000 matrix-to-quaternion conversions
took 68.4 ms there against 20.6 ms in :func:`_matrix_to_quat`, a factor of 3.3, because it solves the K matrix via
``np.linalg.eigh`` instead of branching.  (``scipy`` is rejected as well; the reason stands in
``twinlink.quaternion``.)

Note the complement in :class:`twinlink.RobotState`: the state keeps only the
*latest* transform per edge (live-twin use); this buffer keeps *time series*
for offline batch processing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


import numpy as np


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion ─▶ 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [[1.0 - (yy + zz), xy - wz, xz + wy], [xy + wz, 1.0 - (xx + zz), yz - wx], [xz - wy, yz + wx, 1.0 - (xx + yy)]]
    )


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix ─▶ xyzw quaternion (Shepperd's method).

    This function exists a second time, line for line the same: ``robot_contract.twin_protocol.mat_to_quat_xyzw``.
    That is the price of the layering decision that ``twinlink`` does not depend on ``robot_contract`` (see
    ``task_sim.py``).  To keep the two from drifting, ``tests/test_quat_parity_with_robot_contract.py`` compares them at
    the delicate branches -- whoever changes something here changes it there too.
    """
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w], float)


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between xyzw quaternions (numpy only)."""
    q0 = np.asarray(q0, float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take the short arc
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly parallel ─▶ lerp, renormalise
        out = q0 + alpha * (q1 - q0)
        return out / np.linalg.norm(out)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    return (np.sin((1.0 - alpha) * theta0) / sin0) * q0 + (np.sin(alpha * theta0) / sin0) * q1


@dataclass
class Transform:
    translation: np.ndarray  # (3,)
    rotation: np.ndarray  # quaternion (x, y, z, w)

    def as_matrix(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = _quat_to_matrix(self.rotation)
        mat[:3, 3] = self.translation
        return mat

    @staticmethod
    def identity() -> "Transform":
        return Transform(np.zeros(3), np.array([0, 0, 0, 1.0]))

    @staticmethod
    def from_ros(ros_tf) -> "Transform":
        t = ros_tf.transform.translation
        r = ros_tf.transform.rotation
        return Transform(np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w]))

    def inverse(self) -> "Transform":
        rot = _quat_to_matrix(self.rotation).T
        return Transform(-(rot @ self.translation), _matrix_to_quat(rot))

    def compose(self, other: "Transform") -> "Transform":
        mat = self.as_matrix() @ other.as_matrix()
        return Transform(mat[:3, 3], _matrix_to_quat(mat[:3, :3]))


class TFBuffer:
    """TF tree from recorded messages: static + time-stamped dynamic edges."""

    def __init__(self) -> None:
        # static: (parent, child) ─▶ Transform
        self._static: dict[tuple[str, str], Transform] = {}
        # dynamic: (parent, child) ─▶ sorted list of (timestamp_ns, Transform)
        self._dynamic: dict[tuple[str, str], list] = defaultdict(list)

    def add_static(self, ros_tf_msg) -> None:
        for tf in ros_tf_msg.transforms:
            key = (tf.header.frame_id, tf.child_frame_id)
            self._static[key] = Transform.from_ros(tf)

    def add_dynamic(self, timestamp_ns: int, ros_tf_msg) -> None:
        for tf in ros_tf_msg.transforms:
            key = (tf.header.frame_id, tf.child_frame_id)
            self._dynamic[key].append((timestamp_ns, Transform.from_ros(tf)))

    def finalize(self) -> None:
        """Sort the dynamic series once after ingestion (before lookups)."""
        for key in self._dynamic:
            self._dynamic[key].sort(key=lambda x: x[0])

    # ------------------------------------------------------------------ #
    def _lookup_single(self, parent: str, child: str, timestamp_ns: int) -> Transform | None:
        key = (parent, child)
        if key in self._static:
            return self._static[key]
        if key in self._dynamic:
            entries = self._dynamic[key]
            if not entries:
                return None
            times = [e[0] for e in entries]
            idx = int(np.searchsorted(times, timestamp_ns))
            if idx == 0:
                return entries[0][1]
            if idx >= len(entries):
                return entries[-1][1]
            t0, tf0 = entries[idx - 1]
            t1, tf1 = entries[idx]
            alpha = (timestamp_ns - t0) / (t1 - t0)
            trans = tf0.translation + alpha * (tf1.translation - tf0.translation)
            rot = _slerp(tf0.rotation, tf1.rotation, alpha)
            return Transform(trans, rot)
        return None

    def _bfs_path(self, source: str, target: str) -> list[tuple[str, str, bool]] | None:
        """BFS over the TF graph. Returns list of (parent, child, forward)."""
        all_edges = set(self._static.keys()) | set(self._dynamic.keys())
        graph: dict[str, list[str]] = defaultdict(list)
        for p, c in all_edges:
            graph[p].append(c)
            graph[c].append(p)

        visited = {source}
        queue: list[tuple[str, list]] = [(source, [])]
        while queue:
            node, path = queue.pop(0)
            if node == target:
                return path
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    edge_fwd = (node, neighbor) in all_edges
                    queue.append((neighbor, path + [(node, neighbor, edge_fwd)]))
        return None

    def lookup(self, source: str, target: str, timestamp_ns: int) -> np.ndarray | None:
        """4x4 matrix mapping points from ``source`` into ``target`` frame."""
        if source == target:
            return np.eye(4)

        path = self._bfs_path(source, target)
        if path is None:
            return None

        # Accumulate M so that p_target = M @ p_source.  Walking each edge
        # node─▶next (toward target) we need T_next_node (maps node-frame into
        # next-frame) and left-multiply: M = T_next_node @ M.
        #   - forward edge (node,next) stores T_node_next (next─▶node) ─▶ invert
        #   - reverse edge (next,node) stores T_next_node (node─▶next) ─▶ as-is
        mat = np.eye(4)
        for node, nxt, forward in path:
            if forward:
                tf = self._lookup_single(node, nxt, timestamp_ns)
                if tf is None:
                    return None
                mat = tf.inverse().as_matrix() @ mat
            else:
                tf = self._lookup_single(nxt, node, timestamp_ns)
                if tf is None:
                    return None
                mat = tf.as_matrix() @ mat
        return mat
