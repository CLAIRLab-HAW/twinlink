"""Kinematics and the client-side collision gate on Pinocchio.

The MuJoCo sibling is :class:`twinlink.kinematics.ArmIK` together with ``TwinTaskSim.arm_config_collides``.  This one
exists because a SAPIEN articulation is not an ``MjModel`` -- but the reason to pick Pinocchio over a second MuJoCo
model is what it reads: ``pin.removeCollisionPairs`` takes **the SRDF ``move_group`` itself loads**, so the disabled
pair set agrees with the planner by construction instead of by measurement.  A second hand-kept scene, its drift and
the per-run drift metric that would be needed to watch it all disappear with that.

Measured 2026-08-28 (M4, pin 4.1.0) against the URDF bundle and ``/clearpath/robot.srdf``: model 1.9 ms, collision
model 38.3 ms over 27 geometries, 333 pairs reduced to 190 by the SRDF, full ``computeCollisions`` 0.073 ms, FK plus
frame placements 0.59 us.  Twenty belief boxes cost 0.9 ms to add and lift the full check to 0.267 ms.
"""

from __future__ import annotations

from typing import NamedTuple
from collections.abc import Sequence

import numpy as np


class BeliefBox(NamedTuple):
    """One obstacle as the app BELIEVES it, in the model's root frame."""

    name: str
    half_extents_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


#: Damped-least-squares parameters.  Chosen to match ``twinlink.kinematics.ArmIK`` so both solvers accept and reject
#: the same goals; a solver that converged differently would move the client-side IK gate without anybody touching
#: the gate.
_DAMPING = 1e-6
_MAX_ITERS = 200
_TOLERANCE_M = 1e-4

#: Body prefixes of the movable manipulator.  Only these get paired with belief boxes: pairing the platform as well
#: would let a box standing under the chassis veto every configuration.
_MANIPULATOR_PREFIXES = ("arm_0", "rg6", "camera_0")


class PinocchioKinematics:
    """Forward kinematics, damped-least-squares IK and a collision gate over a Pinocchio model.

    :param urdf_path: the flat URDF -- the container-generated artefact, the same file the world loads.
    :param srdf_path: the SRDF ``move_group`` loads (``/clearpath/robot.srdf``).
    :param mesh_dir: root the URDF's relative mesh paths resolve against.
    :param joints: the actuated arm joints, in the order the wire uses.
    :param tcp_frame: the frame IK solves for.
    """

    def __init__(
        self,
        urdf_path: str,
        srdf_path: str,
        *,
        mesh_dir: str,
        joints: Sequence[str],
        tcp_frame: str,
    ) -> None:
        import coal
        import pinocchio as pin

        self._pin = pin
        self._coal = coal
        self._urdf_path = str(urdf_path)
        self._srdf_path = str(srdf_path)
        self._mesh_dir = str(mesh_dir)
        self.joints: tuple[str, ...] = tuple(joints)
        self.tcp_frame = tcp_frame

        self.model = pin.buildModelFromUrdf(self._urdf_path)
        # ``package_dirs`` by keyword: the positional four-argument form makes pinocchio warn that a package dir was
        # passed "via argument geometry_model", which is a signature ambiguity and not a real problem -- but a
        # warning nobody can act on trains people to ignore warnings.
        self.geom = pin.buildGeomFromUrdf(
            self.model,
            self._urdf_path,
            pin.GeometryType.COLLISION,
            package_dirs=self._mesh_dir,
        )
        #: Pair count BEFORE the SRDF -- kept so a test can show the SRDF did something.
        self.geom.addAllCollisionPairs()
        self.n_pairs_before_srdf = len(self.geom.collisionPairs)
        pin.removeCollisionPairs(self.model, self.geom, self._srdf_path)
        #: Pairs after the SRDF; this is the set ``move_group`` checks too.
        self.n_pairs = len(self.geom.collisionPairs)
        self._robot_pairs = self.n_pairs

        self.data = self.model.createData()
        self._gdata = self.geom.createData()

        # Joint -> configuration index.  ``nq`` exceeds ``nv`` because continuous joints carry (cos, sin); the arm
        # joints are revolute with one entry each, which is why a plain index map suffices.
        self._qidx: dict[str, int] = {}
        self._vidx: dict[str, int] = {}
        for name in self.joints:
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                raise KeyError(f"arm joint {name!r} not in model")
            self._qidx[name] = int(self.model.idx_qs[jid])
            self._vidx[name] = int(self.model.idx_vs[jid])

        # Names, not indices: ``removeGeometryObject`` shifts every later index, so bookkeeping by index would be
        # correct only until the first belief update.
        self._arm_geom_names: tuple[str, ...] = tuple(
            g.name for g in self.geom.geometryObjects if g.name.startswith(_MANIPULATOR_PREFIXES)
        )
        self._belief_names: tuple[str, ...] = ()

    # ------------------------------------------------------------------ #
    # configuration
    # ------------------------------------------------------------------ #
    def _q(self, joints: dict[str, float]) -> np.ndarray:
        q = self._pin.neutral(self.model)
        for name, value in joints.items():
            idx = self._qidx.get(name)
            if idx is not None:
                q[idx] = float(value)
        return q

    # ------------------------------------------------------------------ #
    # forward kinematics
    # ------------------------------------------------------------------ #
    def frame_pose(self, name: str, joints: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """``(position (3,), rotation (3,3))`` of ``name`` in the model's root frame."""
        fid = self.model.getFrameId(name)
        if fid >= self.model.nframes:
            raise KeyError(f"frame {name!r} not in model")
        q = self._q(joints)
        self._pin.forwardKinematics(self.model, self.data, q)
        self._pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[fid]
        return np.asarray(placement.translation, dtype=float), np.asarray(placement.rotation, dtype=float)

    # ------------------------------------------------------------------ #
    # inverse kinematics
    # ------------------------------------------------------------------ #
    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        seed: dict[str, float],
    ) -> dict[str, float] | None:
        """Damped least squares on the TCP frame; ``None`` when it does not converge."""
        pin = self._pin
        fid = self.model.getFrameId(self.tcp_frame)
        target = pin.SE3(np.asarray(target_rot, dtype=float), np.asarray(target_pos, dtype=float))
        q = self._q(seed)
        cols = [self._vidx[name] for name in self.joints]
        for _ in range(_MAX_ITERS):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            err = pin.log(self.data.oMf[fid].actInv(target)).vector
            if float(np.linalg.norm(err)) < _TOLERANCE_M:
                return {name: float(q[self._qidx[name]]) for name in self.joints}
            jac = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.LOCAL)[:, cols]
            step = jac.T @ np.linalg.solve(jac @ jac.T + _DAMPING * np.eye(6), err)
            dq = np.zeros(self.model.nv)
            for k, col in enumerate(cols):
                dq[col] = step[k]
            q = pin.integrate(self.model, q, dq)
        return None

    # ------------------------------------------------------------------ #
    # the believed world
    # ------------------------------------------------------------------ #
    def set_belief_boxes(self, boxes: Sequence[BeliefBox]) -> None:
        """Replace the obstacle set with what the app currently believes.

        Replace, never append: the pool mirrors what ``scene_sync`` publishes to ``move_group``, and an obstacle that
        outlived its removal would make the gate refuse configurations the planner accepts.

        The pair list is rebuilt from the SRDF rather than patched, because ``removeGeometryObject`` renumbers every
        later object -- patching would be correct only until the first update.  Re-reading a 203-line SRDF costs
        milliseconds and happens per scene update, not per control tick.
        """
        pin, coal = self._pin, self._coal
        for name in reversed(self._belief_names):
            if self.geom.existGeometryName(name):
                self.geom.removeGeometryObject(name)
        self._belief_names = ()

        self.geom.removeAllCollisionPairs()
        self.geom.addAllCollisionPairs()
        pin.removeCollisionPairs(self.model, self.geom, self._srdf_path)
        self._robot_pairs = len(self.geom.collisionPairs)

        arm_ids = [self.geom.getGeometryId(name) for name in self._arm_geom_names]
        added: list[str] = []
        for box in boxes:
            hx, hy, hz = box.half_extents_m
            shape = coal.Box(2.0 * hx, 2.0 * hy, 2.0 * hz)
            placement = pin.SE3(_rotation_from_xyzw(box.quaternion_xyzw), np.asarray(box.position_m, dtype=float))
            gid = self.geom.addGeometryObject(pin.GeometryObject(box.name, 0, placement, shape))
            for arm_id in arm_ids:
                self.geom.addCollisionPair(pin.CollisionPair(arm_id, gid))
            added.append(box.name)
        self._belief_names = tuple(added)
        self.n_pairs = len(self.geom.collisionPairs)
        self._gdata = self.geom.createData()

    # ------------------------------------------------------------------ #
    # the gate
    # ------------------------------------------------------------------ #
    def enabled_link_pairs(self) -> frozenset:
        """Link pairs the SRDF LEAVES enabled -- the set ``move_group`` checks, as link names.

        Handed out so a second collision engine can be asked the same question.  Without it the comparison degrades
        into nonsense: a robot always has structural pairs in permanent contact (chassis, top plate, wheels -- 68 of
        them measured on 2026-08-28), and any check that counts those calls every configuration a self-collision.

        Pairs, not geometries: Pinocchio carries one geometry object per collision shape and SAPIEN one per link, so
        the shared vocabulary is the link name.
        """
        pairs = set()
        for pair in list(self.geom.collisionPairs)[: self._robot_pairs]:
            first = self.geom.geometryObjects[pair.first]
            second = self.geom.geometryObjects[pair.second]
            a = self.model.frames[first.parentFrame].name
            b = self.model.frames[second.parentFrame].name
            if a != b:
                pairs.add(frozenset((a, b)))
        return frozenset(pairs)

    def config_collides(self, joints: dict[str, float], *, obstacles_only: bool = False) -> bool:
        """True when the configuration is invalid.

        ``obstacles_only`` restricts the verdict to the arm-versus-belief pairs -- the pre-send probe for a pose
        goal, where the IK branch ``move_group`` will pick is not yet known and only the obstacle question is
        answerable.
        """
        q = self._q(joints)
        self._pin.computeCollisions(self.model, self.data, self.geom, self._gdata, q, False)
        for i, result in enumerate(self._gdata.collisionResults):
            if not result.isCollision():
                continue
            if obstacles_only and i < self._robot_pairs:
                continue
            return True
        return False


def _rotation_from_xyzw(quat: Sequence[float]) -> np.ndarray:
    """Rotation matrix of an xyzw quaternion -- the order the ``/twin/*`` wire uses."""
    x, y, z, w = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
