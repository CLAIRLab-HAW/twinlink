"""Kinematics utilities over a twin MuJoCo model (robot-agnostic).

Turns Cartesian TCP targets into joint-space goals for the MoveIt interface, for any task app.  The class is a
*kinematics utility only*: it never produces trajectories -- the collision-free path between configurations remains
MoveIt's job.

Robot knowledge (joint names, TCP body) comes in through the constructor -- twinlink stays free of any robot profile;
callers (e.g. ``husky_sdk.motion``) bind their profile values.  Runs on a private ``MjData`` scratch copy so IK
iterations never disturb the live simulation state.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np

log = logging.getLogger("twinlink.kinematics")

#: Duck-typed config fields read off ``cfg`` (all optional):
#: ``ik_max_iters`` (200), ``ik_tolerance_pos`` (0.004 m),
#: ``ik_tolerance_rot`` (0.02 rad).
_IK_DEFAULTS = {"ik_max_iters": 200, "ik_tolerance_pos": 0.004, "ik_tolerance_rot": 0.02}


def top_down_grasp_matrix(yaw: float = 0.0) -> np.ndarray:
    """TCP rotation for a vertical top-down grasp with the given yaw.

    Assumes a TCP frame with +z along the approach axis (e.g. the RG6's ``rg6_hand_tcp``): a top-down grasp points TCP-z
    at the floor (-world z); ``yaw`` rotates the finger axis around the world vertical.
    """
    cz, sz = np.cos(yaw), np.sin(yaw)
    # Columns: x/y/z axes of the TCP frame in world coordinates.
    return np.array([[cz, sz, 0.0], [sz, -cz, 0.0], [0.0, 0.0, -1.0]])


class Kinematics(Protocol):
    """What a motion planner needs from a robot model -- FK, IK and the validity gate.

    :class:`ArmIK` satisfies it over MuJoCo, :class:`twinlink.pin_kinematics.PinocchioKinematics` over Pinocchio.
    The protocol exists so ``ArmMotionPlanner`` stops naming ``MjModel`` in its constructor: as long as it did, one
    simulator leaked through an interface that was never declared.
    """

    def frame_pose(self, name: str, joints: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """``(position (3,), rotation (3,3))`` of a frame at the given configuration."""

    def solve_ik(
        self, target_pos: np.ndarray, target_rot: np.ndarray, seed: dict[str, float]
    ) -> dict[str, float] | None:
        """Joint values reaching the pose, or ``None`` when the solver does not converge."""

    def config_collides(self, joints: dict[str, float], *, obstacles_only: bool = False) -> bool:
        """True when the configuration is invalid."""


class ArmIK:
    """Damped-least-squares IK over selected arm joints of a twin model."""

    def __init__(self, model, cfg=None, *, joints: Sequence[str], tcp_body: str) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.cfg = cfg
        self.joints: tuple[str, ...] = tuple(joints)
        self._scratch = mujoco.MjData(model)

        self._tcp_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tcp_body)
        if self._tcp_body < 0:
            raise KeyError(f"TCP body {tcp_body!r} not in model")
        self._qpos_adr: dict[str, int] = {}
        self._dof_adr: dict[str, int] = {}
        self._range: dict[str, tuple[float, float]] = {}
        for name in self.joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"arm joint {name!r} not in model")
            self._qpos_adr[name] = int(model.jnt_qposadr[jid])
            self._dof_adr[name] = int(model.jnt_dofadr[jid])
            lo, hi = model.jnt_range[jid]
            self._range[name] = (float(lo), float(hi)) if hi > lo else (-2 * np.pi, 2 * np.pi)

    def _cfg(self, field: str) -> float:
        return getattr(self.cfg, field, _IK_DEFAULTS[field]) if self.cfg is not None else _IK_DEFAULTS[field]

    # ------------------------------------------------------------------ #
    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_mat: np.ndarray | None = None,
        seed: dict[str, float] | None = None,
        *,
        damping: float = 0.08,
        step_scale: float = 0.9,
    ) -> dict[str, float] | None:
        """Return arm joint positions reaching the target, or ``None``.

        ``target_mat`` constrains full orientation (3x3); ``None`` solves position-only.  ``seed`` initialises the
        search (defaults to the scratch state's current arm pose).
        """
        mujoco = self._mujoco
        data = self._scratch
        # Fresh scratch state; only the arm dofs matter for TCP placement.
        mujoco.mj_resetData(self.model, data)
        if seed:
            for name, val in seed.items():
                if name in self._qpos_adr:
                    data.qpos[self._qpos_adr[name]] = float(val)

        dofs = np.array([self._dof_adr[j] for j in self.joints], dtype=int)
        target_pos = np.asarray(target_pos, dtype=float)
        tol_pos = float(self._cfg("ik_tolerance_pos"))
        tol_rot = float(self._cfg("ik_tolerance_rot"))

        for it in range(int(self._cfg("ik_max_iters"))):
            mujoco.mj_forward(self.model, data)
            cur_pos = data.xpos[self._tcp_body]
            cur_mat = data.xmat[self._tcp_body].reshape(3, 3)

            err_pos = target_pos - cur_pos
            if target_mat is not None:
                err_rot = 0.5 * (
                    np.cross(cur_mat[:, 0], target_mat[:, 0])
                    + np.cross(cur_mat[:, 1], target_mat[:, 1])
                    + np.cross(cur_mat[:, 2], target_mat[:, 2])
                )
                err = np.concatenate([err_pos, err_rot])
            else:
                err_rot = np.zeros(3)
                err = err_pos

            if np.linalg.norm(err_pos) < tol_pos and np.linalg.norm(err_rot) < tol_rot:
                sol = {j: float(data.qpos[self._qpos_adr[j]]) for j in self.joints}
                log.debug("IK converged in %d iters (pos err %.4f)", it, np.linalg.norm(err_pos))
                return sol

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, data, jacp, jacr, self._tcp_body)
            if target_mat is not None:
                jac = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            else:
                jac = jacp[:, dofs]

            # Damped least squares: dq = J^T (J J^T + λ²I)^-1 e
            jjt = jac @ jac.T + (damping**2) * np.eye(jac.shape[0])
            dq = jac.T @ np.linalg.solve(jjt, err)
            dq = np.clip(dq * step_scale, -0.3, 0.3)

            for j, joint in enumerate(self.joints):
                adr = self._qpos_adr[joint]
                lo, hi = self._range[joint]
                data.qpos[adr] = float(np.clip(data.qpos[adr] + dq[j], lo, hi))

        log.debug(
            "IK did not converge (pos err %.4f, rot err %.4f) for target %s",
            np.linalg.norm(err_pos),
            np.linalg.norm(err_rot),
            np.round(target_pos, 3),
        )
        return None
