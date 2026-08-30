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

import logging
from collections.abc import Sequence

import clearlog
import numpy as np

from .events import SimEvents
from .kinematics import WorldFramed

log = logging.getLogger("twinlink.maniskill")

#: Rotation from the OpenCV camera convention SAPIEN reports extrinsics in to the OpenGL one ``perception``
#: back-projects with.  Only the fallback: an observation carries ``cam2world_gl`` ready-made, and composing this by
#: hand is where a sign error produces plausible, wrong world poses.
_R_OPENGL_FROM_OPENCV = np.diag([1.0, -1.0, -1.0])

#: Steps a reset spends settling onto the home pose.  SET, not measured (2026-08-28): enough that the arm arrives
#: under the gains in ``agent``, few enough that a reset stays imperceptible next to a scene build.
_SETTLE_STEPS = 60


class ManiSkillTaskSim:
    """The evaluation world behind the ``husky_sdk.motion`` surface.

    :param env: a constructed ManiSkill environment whose agent is the robot.  Injected, never built here.
    :param arm_joints: the actuated arm joints in wire order.
    :param kinematics: a :class:`twinlink.kinematics.Kinematics` provider (Pinocchio on this route).
    :param control_dt: seconds per control tick, the value the motion planner paces with.
    :param owns_tick: ``True`` in process, ``False`` when a bridge executes trajectories against this world.
    :param gripper_follower_factors: driver joint -> follower ratios, handed in from the profile.
    :param gripper_linkage: the width <-> driver-angle mapping, handed in from the profile.
    :param home_pose: joint values a reset settles on.  Not optional in practice: SAPIEN starts the articulation at
        the URDF zero configuration, and on this robot that pose puts the open hand into the upper arm -- the reflex
        guard reported a predicted self-collision from the first second, correctly (confirmed visually in Foxglove
        on 2026-08-28: the arm lies horizontal with the gripper turned into it).  The real robot never sits there,
        and ``TwinTaskSim`` takes a ``home_pose`` for the same reason.
    """

    def __init__(
        self,
        env,
        *,
        arm_joints: Sequence[str],
        kinematics,
        control_dt: float,
        owns_tick: bool,
        gripper_follower_factors: dict[str, float] | None = None,
        gripper_linkage=None,
        gripper_driver_joint: str | None = None,
        home_pose: dict[str, float] | None = None,
    ) -> None:
        # Per-frame gate for the sensor lookup: a camera name missing from the observation is a property of the
        # environment, so it is reported the first time and not once per frame afterwards.
        self._sensor_once = clearlog.once(log)
        self.env = env.unwrapped
        self.arm_joints: tuple[str, ...] = tuple(arm_joints)
        # World-framed, because everything else on this surface is: object poses, the planner's targets and the
        # belief boxes are all in the world, while Pinocchio answers in the URDF root -- and this world stands the
        # root off the floor.  See twinlink.kinematics.WorldFramed for the measurement.
        self.kinematics = WorldFramed(kinematics, lambda: self._robot.pose.to_transformation_matrix()[0].cpu())
        self.control_dt = float(control_dt)
        self.owns_tick = bool(owns_tick)
        self._followers = dict(gripper_follower_factors or {})
        self._linkage = gripper_linkage
        # The driver joint belongs to the GRIPPER in the profile, not to the linkage -- and twinlink may not read a
        # profile, so it is injected like everything else the robot knows about itself.
        self._driver_joint = gripper_driver_joint
        self._home_pose = dict(home_pose or {})
        self._pending = SimEvents()
        self._robot = self.env.agent.robot
        self._qidx = {j.name: i for i, j in enumerate(self._robot.get_active_joints())}
        self._command: dict[str, float] = {}
        #: Control steps taken since this world was built.  The world's own clock, in ticks -- see
        #: :meth:`sim_time_s`.
        self._steps = 0

    # ------------------------------------------------------------------ #
    # reads -- identical in both modes, in process, no round trip
    # ------------------------------------------------------------------ #
    def sim_time_s(self) -> float:
        """Simulated seconds since this world was built -- the world's OWN clock, not the wall's.

        Every step advances the physics by exactly one control timestep whether it took a millisecond or a second of
        wall time, so this is the only honest answer to "how long has the robot been moving".  It is what the bridge
        publishes on ``/clock``, which is why two properties matter more than precision:


        * **It never goes backwards, not even across a reset.**  A ROS clock that jumps back invalidates every TF
          buffer and message queue in the graph, and ``rclpy`` does not recover -- so :meth:`reset` starts a new
          episode without starting a new clock.
        * **It counts steps, not wall time.**  A world that renders slowly simply produces a slow clock, and
          everything paced by that clock slows with it.  That is the whole reason for publishing it.

        The step comes from the ENVIRONMENT, not from ``control_dt``.  The two are different quantities that read
        alike: ``control_dt`` is what the motion planner paces its samples with (0.02 s in ``maniskill-eval``),
        while ``env.control_timestep`` is how much simulated time one ``env.step`` actually buys (0.01 s there, from
        ``control_freq = 100``).  Measured 2026-08-29: taking ``control_dt`` published a clock running at exactly
        twice the world's rate -- the same class of error a published clock exists to remove.  ``control_dt`` stays
        the fallback for a world whose environment states no timestep.
        """
        return self._steps * self.step_dt

    @property
    def step_dt(self) -> float:
        """Simulated seconds one ``env.step`` advances the world.

        :return: ``env.control_timestep`` where the environment states one, else the injected ``control_dt``.
        """
        stated = getattr(self.env, "control_timestep", None)
        return float(stated) if stated else self.control_dt

    def _step_once(self, action: np.ndarray) -> None:
        """The ONE place the environment is stepped, so the clock cannot miss a step.

        Every caller goes through here: the in-process tick, the bridge's external command, and the settle at
        reset.  A second ``env.step`` elsewhere would advance physics without advancing :meth:`sim_time_s`, and the
        graph would see a clock that lags the world it describes.
        """
        self.env.step(action)
        self._steps += 1

    def _qpos(self) -> np.ndarray:
        return self._robot.get_qpos()[0].cpu().numpy()

    def arm_positions(self) -> dict[str, float]:
        qpos = self._qpos()
        return {name: float(qpos[self._qidx[name]]) for name in self.arm_joints if name in self._qidx}

    def joint_positions(self) -> dict[str, float]:
        """Every active joint, not only the arm -- what the plant publishes on the joint-state buses."""
        qpos = self._qpos()
        return {name: float(qpos[i]) for name, i in self._qidx.items()}

    def gripper_width_m(self) -> float:
        """The MEASURED jaw separation, read off the driver joint.

        Never the commanded value: ``plan_server`` forms its grasp verdict from the width the bridge reports, and
        echoing the command back would make that verdict tautological.
        """
        if self._linkage is None or self._driver_joint is None:
            return float("nan")
        idx = self._qidx.get(self._driver_joint) if self._driver_joint else None
        if idx is None:
            return float("nan")
        return float(self._linkage.width_from_angle(float(self._qpos()[idx])))

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the tool centre point is, IN THE WORLD -- the frame the object poses are in."""
        return self.kinematics.frame_pose(self.kinematics.tcp_frame, self.arm_positions())

    def fk_body_pose(self, name: str, joints: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Where a named body would be at that configuration, in the world frame."""
        return self.kinematics.frame_pose(name, joints)

    def arm_config_collides(self, joints: dict[str, float], *, obstacles_only: bool = False) -> bool:
        return self.kinematics.config_collides(joints, obstacles_only=obstacles_only)

    # ------------------------------------------------------------------ #
    # writes and the tick -- mode dependent
    # ------------------------------------------------------------------ #
    def set_arm_command(self, joints: dict[str, float]) -> None:
        if not self.owns_tick:
            return
        self._command.update({k: float(v) for k, v in joints.items()})

    def command_gripper(self, close: bool, *, grasp: bool = True) -> None:
        if not self.owns_tick or self._linkage is None:
            return
        driver = float(self._linkage.closed_rad if close else self._linkage.open_rad)
        if self._driver_joint:
            self._command[self._driver_joint] = driver
        for follower, factor in self._followers.items():
            self._command[follower] = driver * float(factor)

    def step_physics(self, n: int = 1) -> SimEvents:
        if not self.owns_tick:
            events, self._pending = self._pending, SimEvents()
            return events
        events = SimEvents()
        for _ in range(max(1, int(n))):
            self._step_once(self._action_from_command())
            events.merge(self._collect_events())
        return events

    def apply_external_gripper(self, close: bool) -> None:
        """Command the gripper from OUTSIDE (the bridge), regardless of who owns the tick.

        The counterpart of :meth:`apply_external_command` for the hand.  :meth:`command_gripper` is a no-op on the
        ROS route ON PURPOSE -- there the motion planner runs in its ``real=True`` shape and the backend commands
        over ``/twin/gripper_cmd``, so a second write from the planner would meet a bridge that refuses commands
        during a motion.  But the bridge itself must be able to command, and measured 2026-08-28 it could not: the
        action was accepted, the jaws never moved, and the reported width stayed where it was.
        """
        if self._linkage is None:
            return
        driver = float(self._linkage.closed_rad if close else self._linkage.open_rad)
        if self._driver_joint:
            self._command[self._driver_joint] = driver
        for follower, factor in self._followers.items():
            self._command[follower] = driver * float(factor)

    def apply_external_command(self, joints: dict[str, float]) -> SimEvents:
        """Advance the world from a command that came from OUTSIDE (the bridge).

        The counterpart of :meth:`step_physics` for ``owns_tick=False``: the bridge calls this per control tick, and
        the events land in the buffer the app drains through ``step_physics``.
        """
        self._command.update({k: float(v) for k, v in joints.items()})
        self._step_once(self._action_from_command())
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
            # ONE entry per action dimension, not per joint.  A mimic controller drives several joints from a single
            # value -- measured 2026-08-28 on the RG6, whose six joints occupy exactly one dimension, so building
            # per joint produced a 12-vector where the environment expected 7 and every step raised.
            width = int(np.prod(sub.action_space.shape))
            for joint in list(sub.joints)[:width]:
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

    def task_success(self) -> bool:
        """The world's own verdict on the task -- ``False`` where the environment states none.

        Read from the environment rather than recomputed from the object poses this class also hands out: a second
        implementation of a success predicate is a second truth, and the study rests on this world being the only
        one.  Environments without an ``evaluate`` (the empty world the bridge tests run against) simply have no
        verdict to give, which is not the same as a failed task -- the runner books that as no episode at all.
        """
        evaluate = getattr(self.env, "evaluate", None)
        if evaluate is None:
            return False
        info = evaluate()
        value = info.get("success") if isinstance(info, dict) else None
        return False if value is None else bool(np.asarray(value).reshape(-1)[0])

    def monitored_links(self) -> tuple[str, ...]:
        """The manipulator links whose contact the collision monitor watches."""
        return tuple(link.name for link in self._robot.get_links() if link.name.startswith(("arm_0", "rg6")))

    def reset(self, *, seed: int) -> None:
        """Seeded episode reset.  Clears the pending events so an episode never inherits the previous one's.

        The step count is deliberately NOT cleared: :meth:`sim_time_s` is published as ``/clock``, and a ROS clock
        that jumps backwards invalidates every TF buffer and message queue in the graph.  A new episode gets a new
        scene, not a new clock.
        """
        self.env.reset(seed=int(seed))

        self._command = {}
        self._pending = SimEvents()
        self.settle_to_home()

    def settle_to_home(self) -> None:
        """Place the robot at its home pose, hand open.

        Written into ``qpos`` rather than driven there: at reset there is no controller yet and no time to travel,
        and the alternative -- starting at the URDF zero configuration -- is a self-colliding pose on this robot.
        """
        if not self._home_pose:
            return
        qpos = self._robot.get_qpos().clone()
        for name, value in self._home_pose.items():
            idx = self._qidx.get(name)
            if idx is not None:
                qpos[0, idx] = float(value)
        if self._linkage is not None and self._driver_joint:
            open_rad = float(self._linkage.open_rad)
            for name in (self._driver_joint, *self._followers):
                idx = self._qidx.get(name)
                if idx is not None:
                    qpos[0, idx] = open_rad
        self._robot.set_qpos(qpos)
        self._command = {name: float(qpos[0, idx]) for name, idx in self._qidx.items()}
        # Writing qpos is not enough: the PD controller keeps the target it captured at reset and pulls straight
        # back on the next step -- measured 2026-08-28, the world reported the zero pose again a second later.  Its
        # own reset re-reads the current configuration as the target, and a few steps let the state settle onto it.
        controller = getattr(self.env.agent, "controller", None)
        if controller is not None and hasattr(controller, "reset"):
            controller.reset()
        action = self._action_from_command()
        for _ in range(_SETTLE_STEPS):
            self._step_once(action)

    # ------------------------------------------------------------------ #
    # the world
    # ------------------------------------------------------------------ #
    def _sensor(self, camera: str, block: str):
        obs = self.env.get_obs()
        try:
            return obs[block][camera]
        except (KeyError, TypeError):
            # Runs per frame, so it is gated: a camera name that is not in the observation is wrong for the whole
            # run, not for this one frame.  None reaches the caller as "no image", which reads like a dropped
            # frame rather than a name that never existed.
            self._sensor_once.debug("no %s/%s in the observation -- available: %s", block, camera, sorted(obs))
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

    def camera_pose(self, camera: str) -> tuple[np.ndarray, np.ndarray] | None:
        """``(position (3,), rotation cam->world (3,3))`` in the OpenGL convention.

        Taken from ``cam2world_gl``, which the observation already carries -- not composed from ``extrinsic_cv`` by
        hand, because that is where a sign error hides.
        """
        param = self._sensor(camera, "sensor_param")
        if param is None or "cam2world_gl" not in param:
            return None
        m = param["cam2world_gl"][0].cpu().numpy().astype(float)
        return m[:3, 3].copy(), m[:3, :3].copy()

    def object_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """True poses of the task objects -- GROUND TRUTH, never fed into the planning scene."""
        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, actor in getattr(self.env, "task_objects", {}).items():
            pose = actor.pose
            poses[name] = (
                pose.p[0].cpu().numpy().astype(float),
                pose.q[0].cpu().numpy().astype(float),
            )
        return poses

    def object_half_extents(self, label: str) -> np.ndarray:
        """Half extents of a task object, AS THE SCENE REGISTERED THEM.

        The sibling of :meth:`twinlink.task_sim.TwinTaskSim.object_half_extents` and asked by the same readers.
        Taken from the environment rather than measured off the collision shapes: whoever built the scene already
        wrote the box down -- from a declaration, from an authored part list -- and re-deriving it here would be a
        second description of one body, free to drift from the first.

        :param label: the object's name.
        :returns: the half extents (m).
        :raises KeyError: the scene registers no extents under this label.
        """
        registered = getattr(self.env, "body_half_extents", None)
        if registered is None or label not in registered:
            raise KeyError(
                f"the scene registers no half extents for {label!r} -- an environment whose bodies are read "
                "geometrically has to expose `body_half_extents`"
            )
        return np.asarray(registered[label], dtype=float)

    def object_yaw(self, label: str) -> float:
        """In-plane rotation of the object (rad), from the pose the world reports.

        The MuJoCo world reads this straight off its rotation matrix; here the pose is a quaternion, so it goes
        through the shared conversion rather than through an ``arctan2`` written out again with its own sign
        convention -- exactly what ``TwinTaskSim.object_yaw`` points at.

        :param label: the object's name.
        :returns: the yaw in radians.
        :raises KeyError: no such object.
        """
        from .quaternion import quat_to_yaw_wxyz

        return float(quat_to_yaw_wxyz(self.object_poses()[label][1]))

    def graspable_labels(self) -> frozenset[str]:
        """The names a grasp may take -- the scene's task objects."""
        return frozenset(getattr(self.env, "task_objects", {}))

    def display_object(self, label: str, position, yaw: float = 0.0) -> None:
        """Set an object down at a pose, at rest.

        The counterpart of ``TwinTaskSim.display_object`` and used for the same thing: putting a body where the
        scatter cannot author it -- the stacking support at the tower spot, the pen in its holder.  The velocity
        is cleared with the pose, otherwise the body carries its old motion into the new place and drifts out of
        it while the caller believes it was set down.

        :param label: the object's name.
        :param position: where its origin goes.
        :param yaw: rotation about the vertical axis (rad).
        :raises KeyError: no such object.
        """
        import sapien
        import torch

        actor = getattr(self.env, "task_objects", {})[label]
        half = float(yaw) / 2.0
        actor.set_pose(
            sapien.Pose(p=[float(v) for v in position], q=[float(np.cos(half)), 0.0, 0.0, float(np.sin(half))])
        )
        zero = torch.zeros_like(actor.linear_velocity)
        actor.set_linear_velocity(zero)
        actor.set_angular_velocity(torch.zeros_like(actor.angular_velocity))

    def set_object_quat(self, label: str, quat_wxyz) -> None:
        """Turn an object where it stands, at rest -- the sibling of ``TwinTaskSim.set_object_quat``.

        :param label: the object's name.
        :param quat_wxyz: the new orientation; normalised here.
        :raises KeyError: no such object.
        """
        import sapien
        import torch

        actor = getattr(self.env, "task_objects", {})[label]
        quat = np.asarray(quat_wxyz, dtype=float)
        quat = quat / np.linalg.norm(quat)
        position = actor.pose.p[0].cpu().numpy().astype(float)
        actor.set_pose(sapien.Pose(p=[float(v) for v in position], q=[float(v) for v in quat]))
        actor.set_linear_velocity(torch.zeros_like(actor.linear_velocity))
        actor.set_angular_velocity(torch.zeros_like(actor.angular_velocity))

    def pairwise_contact_force(self, label: str, link_names: Sequence[str]) -> dict[str, float]:
        """Contact force between one task object and each named robot link [N].

        ``contact_forces`` answers the other question -- what is the worst thing this link touches -- and a grasp
        needs to know WHICH body the jaws are on.  Both go through ``get_pairwise_contact_forces`` for the reason
        that method spells out: the RG6's four-bar touches itself by construction, so a net force on a finger is
        never evidence of a grasp.

        :param label: the object's name.
        :param link_names: the robot links to ask about.
        :returns: force magnitude per link, zero for a link the robot does not have.
        :raises KeyError: no such object.
        """
        actor = getattr(self.env, "task_objects", {})[label]
        links = {link.name: link for link in self._robot.get_links()}
        out: dict[str, float] = {}
        for name in link_names:
            link = links.get(name)
            if link is None:
                out[name] = 0.0
                continue
            force = self.env.scene.get_pairwise_contact_forces(actor, link)
            out[name] = float(np.linalg.norm(force[0].cpu().numpy()))
        return out

    def close(self) -> None:
        """Release the environment -- SAPIEN holds a renderer and a device context per world."""
        self.env.close()

    def settle(self, max_ticks: int = 100, vel_eps: float = 0.01) -> SimEvents:
        """Step until the task objects are at rest, or ``max_ticks`` elapsed.

        The sibling of ``TwinTaskSim.settle`` and asked for by the same callers: a placement is judged after the
        object has come to rest, not while it is still falling into place. The MuJoCo one reads free-joint
        velocities out of ``qvel``; here SAPIEN answers per actor, which is the same question in the engine's own
        terms.

        The GRASPED object is excluded there and cannot be here -- this world does not know which one is held, and
        that read is exactly the one that does not travel (see ``hrl.env.task_objects``). A carried object moves
        with the arm, so a settle during a carry runs to the tick limit rather than returning early. Callers ask
        for a settle when they have put something down, which is when it matters.

        :param max_ticks: how long to wait at most.
        :param vel_eps: linear speed below which an object counts as standing [m/s].
        :returns: what happened while waiting.
        """
        events = SimEvents()
        for _ in range(max(0, int(max_ticks))):
            events.merge(self.step_physics(1))
            speeds = [
                float(np.linalg.norm(np.asarray(actor.linear_velocity[0].cpu(), dtype=float)))
                for actor in getattr(self.env, "task_objects", {}).values()
            ]
            if not speeds or max(speeds) < vel_eps:
                break
        return events

    def contact_forces(self, link_names: Sequence[str]) -> dict[str, float]:
        """Per-link contact force magnitude in newton, against things that are NOT the robot.

        **Pairwise, not net, and that distinction is the whole method.**
        ``Articulation.get_net_contact_forces`` sums every contact on a link, self-contacts included -- and the RG6
        is a four-bar linkage whose struts touch each other by construction.  Measured 2026-08-28 against the
        running stack: it reported 34 kN on ``rg6_gripper_finger_1_truss_arm`` and 8 kN on the bracket with nothing
        in the scene at all, and the collision monitor duly froze the plant on the gripper's own mechanism.

        A collision is contact with something that is not the robot, so the forces come from
        ``scene.get_pairwise_contact_forces(actor, link)`` over the scene's non-robot actors.  In a world without
        task objects that is legitimately empty -- there is nothing to collide with.
        """
        names = list(link_names)
        others = self._non_robot_actors()
        if not names or not others:
            return {name: 0.0 for name in names}
        links = {link.name: link for link in self._robot.get_links()}
        out: dict[str, float] = {}
        for name in names:
            link = links.get(name)
            if link is None:
                continue
            worst = 0.0
            for actor in others:
                force = self.env.scene.get_pairwise_contact_forces(actor, link)
                worst = max(worst, float(np.linalg.norm(force[0].cpu().numpy())))
            out[name] = worst
        return out

    def _non_robot_actors(self) -> list:
        """The scene's actors that are not part of the robot -- what a collision can be WITH.

        ``collidable_actors`` if the environment offers it, ``task_objects`` otherwise.  The distinction is not
        pedantry: a table is not a task object, and a monitor that watched only the task objects would stay silent
        while the arm drove through the table it is picking from.
        """
        listed = getattr(self.env, "collidable_actors", None)
        if callable(listed):
            return list(listed())
        return list(getattr(self.env, "task_objects", {}).values())

    def self_collides(self, joints: dict[str, float]) -> bool:
        """Does the PHYSICS see robot-versus-robot contact the SRDF has NOT disabled?

        Only used by the regression test that compares the two URDF ingestions: the oracle answers the same question
        from Pinocchio, and a disagreement means the libraries read the model differently.  Not part of the control
        path -- the gate is the oracle's, because that is the one whose disabled pairs match ``move_group``.

        Two things this must NOT do, both measured on 2026-08-28.  It must not reuse :meth:`contact_forces`, which
        answers about the SCENE (table, cubes) and can never report the robot against itself.  And it must not count
        every self-contact: 68 pairs touch permanently by construction (chassis, top plate, wheels), so an unfiltered
        answer is "yes" at every configuration.  The SRDF-enabled pairs come from the oracle, which is what makes the
        two sides answer one question instead of two.

        The scene is STEPPED here, because PhysX generates no contacts for a configuration merely written into
        ``qpos``.  The configuration is restored afterwards; the step still advances everything else in the world by
        one tick, which is why this belongs to an in-process regression world and not beside a running measurement.
        """
        enabled = getattr(self.kinematics, "enabled_link_pairs", None)
        if enabled is None:
            raise RuntimeError("no kinematics to borrow the SRDF pair set from -- the answer would be meaningless")
        allowed = enabled()
        saved = self._robot.get_qpos().clone()
        try:
            probe = saved.clone()
            for name, value in joints.items():
                idx = self._qidx.get(name)
                if idx is not None:
                    probe[0, idx] = float(value)
            self._robot.set_qpos(probe)
            self.env.scene.px.step()
            links = {link.name for link in self._robot.get_links()}
            for contact in self.env.scene.get_contacts():
                first, second = contact.bodies[0].entity.name, contact.bodies[1].entity.name
                if first not in links or second not in links:
                    continue
                if frozenset((first, second)) not in allowed:
                    continue
                # Penetration, not impulse: a configuration just written into qpos has had no time to build one.
                if any(point.separation < 0.0 for point in contact.points):
                    return True
            return False
        finally:
            self._robot.set_qpos(saved)
