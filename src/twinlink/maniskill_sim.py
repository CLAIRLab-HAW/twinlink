"""The ManiSkill/SAPIEN evaluation world.

Sibling of :class:`twinlink.task_sim.TwinTaskSim`: the same injected robot facts and the same surface towards
``husky_sdk.motion`` -- but SAPIEN owns physics and rendering, and kinematics come from a handed-in provider instead
of an ``MjModel``.  The ManiSkill environment is handed in as well: twinlink knows no tasks and imports no app.

**Who owns the tick.** ``ArmMotionPlanner._mirror_stream`` writes every downlink sample back with
``set_arm_command`` and then calls ``step_physics(1)``.  On the ROS route those samples ORIGINATE here -- they
travelled bridge, ros2_control, downlink, app -- so writing them back would be a loop and stepping would give one
world two steppers.  ``owns_tick`` selects: in process both do the obvious thing, on the ROS route the writes are
no-ops and ``step_physics`` returns the events accumulated since the previous call without advancing.

Measured 2026-08-28 (M4, macOS arm64, MoltenVK): the a200 bundle loads into a ManiSkill env with 52 links and 16
active joints; ``qpos`` is (1, 16) while the arm controller's action is (6,) -- the action is NOT qpos-shaped, and
building it as if it were is why this module walks the controller's sub-controllers instead.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .events import SimEvents

#: Rotation from the OpenCV camera convention SAPIEN reports extrinsics in to the OpenGL one ``perception``
#: back-projects with.  Only the fallback: an observation carries ``cam2world_gl`` ready-made, and composing this by
#: hand is where a sign error produces plausible, wrong world poses.
_R_OPENGL_FROM_OPENCV = np.diag([1.0, -1.0, -1.0])


class ManiSkillTaskSim:
    """The evaluation world behind the ``husky_sdk.motion`` surface.

    :param env: a constructed ManiSkill environment whose agent is the robot.  Injected, never built here.
    :param arm_joints: the actuated arm joints in wire order.
    :param kinematics: a :class:`twinlink.kinematics.Kinematics` provider (Pinocchio on this route).
    :param control_dt: seconds per control tick, the value the motion planner paces with.
    :param owns_tick: ``True`` in process, ``False`` when a bridge executes trajectories against this world.
    :param gripper_follower_factors: driver joint -> follower ratios, handed in from the profile.
    :param gripper_linkage: the width <-> driver-angle mapping, handed in from the profile.
    """

    def __init__(
        self,
        env,
        *,
        arm_joints: Sequence[str],
        kinematics,
        control_dt: float,
        owns_tick: bool,
        gripper_follower_factors: Dict[str, float] | None = None,
        gripper_linkage=None,
    ) -> None:
        self.env = env.unwrapped
        self.arm_joints: Tuple[str, ...] = tuple(arm_joints)
        self.kinematics = kinematics
        self.control_dt = float(control_dt)
        self.owns_tick = bool(owns_tick)
        self._followers = dict(gripper_follower_factors or {})
        self._linkage = gripper_linkage
        self._pending = SimEvents()
        self._robot = self.env.agent.robot
        self._qidx = {j.name: i for i, j in enumerate(self._robot.get_active_joints())}
        self._command: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # reads -- identical in both modes, in process, no round trip
    # ------------------------------------------------------------------ #
    def _qpos(self) -> np.ndarray:
        return self._robot.get_qpos()[0].cpu().numpy()

    def arm_positions(self) -> Dict[str, float]:
        qpos = self._qpos()
        return {name: float(qpos[self._qidx[name]]) for name in self.arm_joints if name in self._qidx}

    def joint_positions(self) -> Dict[str, float]:
        """Every active joint, not only the arm -- what the plant publishes on the joint-state buses."""
        qpos = self._qpos()
        return {name: float(qpos[i]) for name, i in self._qidx.items()}

    def gripper_width_m(self) -> float:
        """The MEASURED jaw separation.

        Never the commanded value: ``plan_server`` forms its grasp verdict from the width the bridge reports, and
        echoing the command back would make that verdict tautological.
        """
        if self._linkage is None:
            return float("nan")
        driver = self._linkage.driver_joint
        idx = self._qidx.get(driver)
        if idx is None:
            return float("nan")
        return float(self._linkage.width_from_angle(float(self._qpos()[idx])))

    def tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.kinematics.frame_pose(self.kinematics.tcp_frame, self.arm_positions())

    def fk_body_pose(self, name: str, joints: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        return self.kinematics.frame_pose(name, joints)

    def arm_config_collides(self, joints: Dict[str, float], *, obstacles_only: bool = False) -> bool:
        return self.kinematics.config_collides(joints, obstacles_only=obstacles_only)

    # ------------------------------------------------------------------ #
    # writes and the tick -- mode dependent
    # ------------------------------------------------------------------ #
    def set_arm_command(self, joints: Dict[str, float]) -> None:
        if not self.owns_tick:
            return
        self._command.update({k: float(v) for k, v in joints.items()})

    def command_gripper(self, close: bool, *, grasp: bool = True) -> None:
        if not self.owns_tick or self._linkage is None:
            return
        driver = float(self._linkage.closed_rad if close else self._linkage.open_rad)
        self._command[self._linkage.driver_joint] = driver
        for follower, factor in self._followers.items():
            self._command[follower] = driver * float(factor)

    def step_physics(self, n: int = 1) -> SimEvents:
        if not self.owns_tick:
            events, self._pending = self._pending, SimEvents()
            return events
        events = SimEvents()
        for _ in range(max(1, int(n))):
            self.env.step(self._action_from_command())
            events.merge(self._collect_events())
        return events

    def apply_external_command(self, joints: Dict[str, float]) -> SimEvents:
        """Advance the world from a command that came from OUTSIDE (the bridge).

        The counterpart of :meth:`step_physics` for ``owns_tick=False``: the bridge calls this per control tick, and
        the events land in the buffer the app drains through ``step_physics``.
        """
        self._command.update({k: float(v) for k, v in joints.items()})
        self.env.step(self._action_from_command())
        events = self._collect_events()
        self._pending.merge(events)
        return events

    def _action_from_command(self) -> np.ndarray:
        """The command dict as the env's action vector.

        NOT a copy of ``qpos``: measured 2026-08-28, ``qpos`` is (16,) while the arm controller's action is (6,).
        The vector is the concatenation of the sub-controllers' segments, each in ITS joint order -- so it is built
        by walking the controller rather than by slicing the state.
        """
        qpos = self._qpos()
        segments: list[float] = []
        for sub in self.env.agent.controller.controllers.values():
            for joint in sub.joints:
                name = joint.name
                fallback = float(qpos[self._qidx[name]]) if name in self._qidx else 0.0
                segments.append(float(self._command.get(name, fallback)))
        return np.asarray(segments, dtype=np.float32)

    def _collect_events(self) -> SimEvents:
        events = SimEvents()
        forces = self.contact_forces(self.monitored_links())
        if forces and max(forces.values()) > 0.0:
            events.robot_obstacle_collision = True
        return events

    def monitored_links(self) -> Tuple[str, ...]:
        """The manipulator links whose contact the collision monitor watches."""
        return tuple(link.name for link in self._robot.get_links() if link.name.startswith(("arm_0", "rg6")))

    def reset(self, *, seed: int) -> None:
        """Seeded episode reset.  Clears the pending events so an episode never inherits the previous one's."""
        self.env.reset(seed=int(seed))
        self._command = {}
        self._pending = SimEvents()

    # ------------------------------------------------------------------ #
    # the world
    # ------------------------------------------------------------------ #
    def _sensor(self, camera: str, block: str):
        obs = self.env.get_obs()
        try:
            return obs[block][camera]
        except (KeyError, TypeError):
            return None

    def render_rgb(self, camera: str) -> np.ndarray | None:
        """RGB uint8 ``(H, W, 3)`` of a named sensor, or ``None`` when the env carries no image observation."""
        data = self._sensor(camera, "sensor_data")
        if data is None or "rgb" not in data:
            return None
        return data["rgb"][0].cpu().numpy().astype(np.uint8)

    def render_depth(self, camera: str) -> np.ndarray | None:
        """Metric depth ``(H, W)`` float32 in metres, or ``None``."""
        data = self._sensor(camera, "sensor_data")
        if data is None or "depth" not in data:
            return None
        depth = data["depth"][0].cpu().numpy().astype(np.float32)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        # ManiSkill reports depth as int16 MILLIMETRES in the rgbd observation mode; a metre-scale scene never
        # produces values above 100, so the threshold separates the two without a mode flag to keep in sync.
        return depth / 1000.0 if float(depth.max(initial=0.0)) > 100.0 else depth

    def camera_matrix(self, camera: str) -> np.ndarray | None:
        param = self._sensor(camera, "sensor_param")
        if param is None or "intrinsic_cv" not in param:
            return None
        return param["intrinsic_cv"][0].cpu().numpy().astype(np.float64)

    def camera_pose(self, camera: str) -> Tuple[np.ndarray, np.ndarray] | None:
        """``(position (3,), rotation cam->world (3,3))`` in the OpenGL convention.

        Taken from ``cam2world_gl``, which the observation already carries -- not composed from ``extrinsic_cv`` by
        hand, because that is where a sign error hides.
        """
        param = self._sensor(camera, "sensor_param")
        if param is None or "cam2world_gl" not in param:
            return None
        m = param["cam2world_gl"][0].cpu().numpy().astype(float)
        return m[:3, 3].copy(), m[:3, :3].copy()

    def object_poses(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """True poses of the task objects -- GROUND TRUTH, never fed into the planning scene."""
        poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, actor in getattr(self.env, "task_objects", {}).items():
            pose = actor.pose
            poses[name] = (
                pose.p[0].cpu().numpy().astype(float),
                pose.q[0].cpu().numpy().astype(float),
            )
        return poses

    def contact_forces(self, link_names: Sequence[str]) -> Dict[str, float]:
        """Per-link net contact force magnitude in newton.

        ``Articulation.get_net_contact_forces(link_names)`` returns ``(num_envs, len(link_names), 3)``; the norm over
        the last axis is the magnitude, and environment 0 is the digital twin (measured 2026-08-28).
        """
        names = list(link_names)
        if not names:
            return {}
        forces = self._robot.get_net_contact_forces(names)
        magnitudes = np.linalg.norm(forces[0].cpu().numpy(), axis=-1)
        return {name: float(m) for name, m in zip(names, magnitudes)}

    def self_collides(self, joints: Dict[str, float]) -> bool:
        """Does the PHYSICS see robot-versus-robot contact at this configuration?

        Only used by the regression test that compares the two URDF ingestions: the oracle answers the same question
        from Pinocchio, and a disagreement means the libraries read the model differently.  Not part of the control
        path -- the gate is the oracle's, because that is the one whose disabled pairs match ``move_group``.
        """
        saved = self._robot.get_qpos().clone()
        try:
            probe = saved.clone()
            for name, value in joints.items():
                idx = self._qidx.get(name)
                if idx is not None:
                    probe[0, idx] = float(value)
            self._robot.set_qpos(probe)
            return any(f > 1e-6 for f in self.contact_forces(self.monitored_links()).values())
        finally:
            self._robot.set_qpos(saved)
