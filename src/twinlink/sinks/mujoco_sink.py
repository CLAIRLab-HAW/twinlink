"""MuJoCo digital-twin sink.

Drives a MuJoCo model from a :class:`RobotState`: every joint in the state that
also exists in the model is written to ``qpos`` by name, an optional free joint
is driven from the base pose, ``mj_forward`` updates the kinematics, and the
scene is rendered.  The sink is *robot-agnostic* -- it only needs joint-name
correspondence between the state and the model.

With ``physics=True`` the sink instead *simulates* (``mj_step``): the state's
joints are held at their commanded values while the free base settles under
gravity, so a robot spawned above the ground falls and comes to rest on it.

Rendering backends:

* ``offscreen`` -- ``mujoco.Renderer`` to a numpy frame shown with OpenCV.
  Works with a plain ``python3`` (no ``mjpython`` needed); the default.
* ``viewer``    -- interactive ``mujoco.viewer`` window.  On macOS this needs
  to be launched with ``mjpython``.
* ``none``      -- maintain kinematics without rendering (headless).

Sensor data (e.g. a RealSense colour image carried in the state) is shown in a
side window so the recorded camera and the simulated twin are visible together.

With ``show_obstacles=True`` the free camera **auto-frames** itself on the union
of the robot and the first obstacle cloud that arrives (see ``auto_frame``): an
observed scene sits wherever the sensor happened to look, which is regularly
outside a robot-centred view, so without this the voxels are off-screen and the
overlay looks broken.  Passing ``cam_distance`` / ``cam_lookat`` opts out.
"""
from __future__ import annotations

import logging
import math
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from .base import StateSink

log = logging.getLogger("twinlink.mujoco")


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], float)


# ROS optical frame (z-forward, x-right, y-down) expressed in the camera *link*
# frame (x-forward, y-left, z-up) — REP 103, rpy = (-pi/2, 0, -pi/2). Applied when
# a point cloud's *_optical_frame isn't a model body (it's driver-only tf) and we
# fall back to the camera link, so the cloud lands in front of the camera, not above.
_R_LINK_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# Free-camera fallbacks, used for whichever cam_* argument the caller left out.
# Framed on the robot: right for a plain kinematic twin, too tight once an
# obstacle cloud several metres away joins the scene -- hence auto_frame.
_CAM_DEFAULT_DISTANCE = 2.5
_CAM_DEFAULT_AZIMUTH = 135.0
_CAM_DEFAULT_ELEVATION = -20.0
# Padding on the auto-framed bounding sphere, so the outermost voxels don't sit
# exactly on the image border.
_AUTO_FRAME_MARGIN = 1.15


class MujocoSink(StateSink):
    def __init__(
        self,
        model,
        *,
        render: str = "offscreen",
        display: bool = True,
        width: int = 640,
        height: int = 480,
        camera: Optional[str] = None,
        cam_distance: Optional[float] = None,
        cam_azimuth: Optional[float] = None,
        cam_elevation: Optional[float] = None,
        cam_lookat=None,
        auto_frame: Optional[bool] = None,
        base_free_joint: Optional[str] = None,
        joint_remap: Optional[Dict[str, str]] = None,
        show_sensor_camera: Optional[str] = None,
        snapshot_path: Optional[str] = None,
        snapshot_every: float = 0.0,
        keep_visual: bool = False,
        colored: bool = True,
        physics: bool = False,
        spawn_height: Optional[float] = None,
        hold_joints: bool = True,
        wheel_frictionloss: float = 5.0,
        wheel_damping: float = 0.5,
        show_obstacles: bool = False,
        obstacle_voxel: float = 0.04,
        obstacle_max: int = 4000,
        preview: bool = False,
        preview_hold: float = 1.0,
    ) -> None:
        super().__init__()
        self._model_arg = model
        self.show_obstacles = show_obstacles
        self.obstacle_voxel = obstacle_voxel
        self.obstacle_max = obstacle_max
        self.preview = preview
        self.preview_hold = preview_hold
        self.keep_visual = keep_visual
        self.colored = colored
        self.physics = physics
        self.spawn_height = spawn_height
        self.hold_joints = hold_joints
        self.wheel_frictionloss = wheel_frictionloss
        self.wheel_damping = wheel_damping
        self.render_mode = render
        self.display = display
        self.width = width
        self.height = height
        self.camera = camera
        self.cam_distance = _CAM_DEFAULT_DISTANCE if cam_distance is None else float(cam_distance)
        self.cam_azimuth = _CAM_DEFAULT_AZIMUTH if cam_azimuth is None else float(cam_azimuth)
        self.cam_elevation = _CAM_DEFAULT_ELEVATION if cam_elevation is None else float(cam_elevation)
        self.cam_lookat = cam_lookat
        # Auto-framing only ever touches lookat + distance, so an explicit value
        # for either one is taken as "the caller has framed this deliberately".
        # azimuth/elevation stay free: framing the whole bounding sphere works
        # from any viewing direction.
        self.auto_frame = (
            bool(show_obstacles and cam_distance is None and cam_lookat is None)
            if auto_frame is None else bool(auto_frame)
        )
        self.base_free_joint = base_free_joint
        self.joint_remap = dict(joint_remap or {})
        self.show_sensor_camera = show_sensor_camera
        self.snapshot_path = snapshot_path
        self.snapshot_every = snapshot_every

        self.model = None
        self.data = None
        self._joint_qpos: Dict[str, int] = {}  # state-name -> qpos address
        self._joint_dof: Dict[str, int] = {}  # state-name -> dof (qvel) address
        self._base_qpos: Optional[int] = None
        self._phys_last: Optional[float] = None
        self._renderer = None
        self._viewer = None
        self._mjcam = None
        self._last_snapshot = -1e9
        # Plain ASCII: OpenCV's macOS Cocoa backend mojibakes non-ASCII window
        # titles (e.g. an em-dash renders as "â€""), so avoid Unicode here.
        self._win_main = "TwinLink - MuJoCo twin"
        self._win_cam = "TwinLink - sensor camera"
        self._frame_body: Dict[str, int] = {}  # cloud frame_id -> mujoco body id (cache)
        self._preview_pos: Optional[Dict[str, float]] = None
        self._last_traj = None
        self._traj_start = 0.0
        self._win_ready = False
        self._mouse_last: Optional[tuple] = None
        self._autoframed = False  # auto_frame is one-shot: fires on the first cloud

        # Goal preview: ghost arm at the target pose before planning.
        self._goal_preview_pos: Optional[Dict[str, float]] = None
        self._goal_preview_until: float = 0.0  # wall-clock deadline
        self._goal_preview_name: str = ""
        # Trajectory preview state: "idle" (live), "planning" (ghost visible)
        self._preview_state: str = "idle"
        # Slow down trajectory playback so it's visible (real traj is ~0.1s).
        self._preview_speed: float = 0.5  # 0.5 = 2x slower than real-time
        # When True, incoming display_planned_path messages are ignored (no ghost).
        # Set by cancel_preview() (after user confirms execution) and cleared by
        # show_goal_preview() / unlock_ghost() (next goal starts a new cycle).
        self._ghost_locked: bool = False
        # When True, the planned-trajectory ghost replays in a loop instead of
        # disappearing after one playback.  Set by the demo in safe-execute mode
        # so the user can review the motion repeatedly until they confirm.
        self._loop_preview: bool = False

    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        import mujoco

        self.model = self._load_model(mujoco)
        self.data = mujoco.MjData(self.model)
        self._index_joints(mujoco)
        self._hide_collision_geoms(mujoco)
        if self.physics:
            self._brake_free_hinges(mujoco)
        if self.spawn_height is not None and self._base_qpos is not None:
            self.data.qpos[self._base_qpos + 2] = self.spawn_height
        self._setup_render(mujoco)
        mujoco.mj_forward(self.model, self.data)
        log.info(
            "MuJoCo model ready: %d joints mapped, base_free_joint=%s, physics=%s, render=%s",
            len(self._joint_qpos),
            self.base_free_joint if self._base_qpos is not None else "none",
            self.physics,
            self.render_mode,
        )

    def _hide_collision_geoms(self, mujoco) -> None:
        # When a model carries both visual (non-colliding) and collision geoms
        # (e.g. for physics), hide the collision ones so the render stays clean.
        nonplane = [i for i in range(self.model.ngeom)
                    if int(self.model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_PLANE)]
        has_visual = any(self.model.geom_contype[i] == 0 for i in nonplane)
        collision = [i for i in nonplane if self.model.geom_contype[i] != 0]
        if has_visual and collision:
            for i in collision:
                self.model.geom_rgba[i, 3] = 0.0

    def _brake_free_hinges(self, mujoco) -> None:
        # Unlimited hinge joints are typically wheels/casters. Frictionless,
        # they let the robot roll away after landing; give them joint friction
        # + damping so a "parked" robot stays put (motor/gearbox holding torque).
        n = 0
        for jid in range(self.model.njnt):
            if int(self.model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE) and not self.model.jnt_limited[jid]:
                dof = int(self.model.jnt_dofadr[jid])
                self.model.dof_frictionloss[dof] = self.wheel_frictionloss
                self.model.dof_damping[dof] = self.wheel_damping
                n += 1
        if n:
            log.info("braked %d free hinge joint(s) (frictionloss=%.1f, damping=%.1f)",
                     n, self.wheel_frictionloss, self.wheel_damping)

    def _load_model(self, mujoco):
        m = self._model_arg
        if isinstance(m, mujoco.MjModel):
            return m
        path = str(m)
        if path.endswith(".urdf"):
            from ..urdf_mujoco import load_mujoco_from_urdf

            return load_mujoco_from_urdf(
                path,
                floating_base=self.base_free_joint is not None,
                keep_visual=self.keep_visual,
                colored=self.colored,
            )
        return mujoco.MjModel.from_xml_path(path)

    def _index_joints(self, mujoco) -> None:
        # Map every 1-DoF joint in the model by name so state joints can be
        # written by name.  joint_remap lets the state use different names.
        single_dof = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}
        model_joint_adr: Dict[str, int] = {}
        model_joint_dof: Dict[str, int] = {}
        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name is None:
                continue
            if int(self.model.jnt_type[jid]) in single_dof:
                model_joint_adr[name] = int(self.model.jnt_qposadr[jid])
                model_joint_dof[name] = int(self.model.jnt_dofadr[jid])
            elif int(self.model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                if self.base_free_joint is None or name == self.base_free_joint:
                    self._base_qpos = int(self.model.jnt_qposadr[jid])
                    self.base_free_joint = name

        # The state addresses joints by (possibly remapped) name.
        for model_name, adr in model_joint_adr.items():
            self._joint_qpos[model_name] = adr
            self._joint_dof[model_name] = model_joint_dof[model_name]
        for state_name, model_name in self.joint_remap.items():
            if model_name in model_joint_adr:
                self._joint_qpos[state_name] = model_joint_adr[model_name]
                self._joint_dof[state_name] = model_joint_dof[model_name]

    def _setup_render(self, mujoco) -> None:
        if self.render_mode == "viewer":
            import mujoco.viewer

            try:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data, show_left_ui=True, show_right_ui=True
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"MuJoCo interactive viewer could not start ({exc}). On macOS the "
                    "passive viewer must run under mjpython, e.g.:\n"
                    "  mjpython <script.py> --render viewer ...\n"
                    "(mjpython ships with the mujoco pip package.)"
                ) from exc
            return
        if self.render_mode == "none" and not self.snapshot_path:
            return

        # offscreen (also used headless when a snapshot is requested). Reserve
        # extra scene geoms for obstacle voxels.
        extra = self.obstacle_max + 1000 if self.show_obstacles else 0
        self._renderer = mujoco.Renderer(self.model, self.height, self.width, max_geom=10000 + extra)
        self._mjcam = mujoco.MjvCamera()
        self._mjcam.distance = self.cam_distance
        self._mjcam.azimuth = self.cam_azimuth
        self._mjcam.elevation = self.cam_elevation
        lookat = self.cam_lookat
        if lookat is None:
            lookat = self._guess_lookat(mujoco)
        self._mjcam.lookat[:] = lookat

    def _guess_lookat(self, mujoco) -> np.ndarray:
        # Centre on the arm base if present, else the model centroid.
        for cand in ("arm_0_base_link", "arm_0_base_link_inertia", "base_link"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cand)
            if bid >= 0:
                return self.data.xpos[bid].copy() + np.array([0, 0, 0.2])
        return np.array([0.0, 0.0, 0.5])

    # ------------------------------------------------------------------ #
    def update(self) -> bool:
        import mujoco

        assert self.state is not None
        # Check for new planned trajectory before rendering
        if self.preview:
            self._check_new_trajectory()
        if self.physics:
            self._step_physics(mujoco)
        else:
            self._apply_kinematics(mujoco)
        return self._render(mujoco)

    def cancel_preview(self) -> None:
        """Cancel all ghost previews and lock against new trajectory ghosts.

        Called by the demo after the user confirms execution — the live motion
        from /joint_states should NOT be ghosted, and the execute phase's
        display_planned_path must NOT re-trigger a ghost.
        Unlocked by the next show_goal_preview() / unlock_ghost() (new cycle).
        """
        self._goal_preview_pos = None
        self._goal_preview_until = 0.0
        self._preview_state = "idle"
        self._last_traj = None
        self._ghost_locked = True

    def unlock_ghost(self) -> None:
        """Unlock the ghost for the next planned trajectory *without* a static goal preview.

        Unlike ``show_goal_preview()``, this does NOT render a static ghost at
        the target pose — only the upcoming planned-trajectory ghost will be
        visible.  Use this in safe-execute mode where the user should review the
        animated trajectory, not a static goal pose that would flash before the
        plan arrives.

        Snapshot the current in-state trajectory into ``_last_traj`` so that
        ``_check_new_trajectory()`` does NOT re-pick it up as "new" the moment
        the lock is released.  Without this, the previous execution's
        display_planned_path (still held in the state) would flash briefly
        before the new plan arrives.  Only a genuinely new trajectory object
        (from the new plan) will trigger the ghost.
        """
        self._goal_preview_pos = None
        self._goal_preview_until = 0.0
        self._preview_state = "idle"
        self._ghost_locked = False
        # Snapshot whatever trajectory is currently in the state so it's not
        # mistaken for "new" on the next _check_new_trajectory() call.
        self._last_traj = self.state.planned_trajectory() if self.state else None

    def set_loop_preview(self, loop: bool) -> None:
        """Enable or disable looping of the planned-trajectory ghost preview.

        When enabled, the ghost trajectory replays from the start after
        finishing instead of disappearing.  Used in safe-execute mode so the
        user can review the planned motion repeatedly until they confirm or skip.
        """
        self._loop_preview = loop

    def _check_new_trajectory(self) -> None:
        """Detect a newly arrived planned trajectory and start ghost playback.

        Ignored when _ghost_locked (after user confirmed execution — the
        execute phase re-publishes display_planned_path, which should NOT
        trigger a ghost).
        """
        if self._ghost_locked:
            return
        traj = self.state.planned_trajectory() if self.state else None
        if traj is None:
            return
        if traj is not self._last_traj:
            self._last_traj = traj
            self._traj_start = time.monotonic()
            self._preview_state = "planning"
            scaled = traj.duration() / self._preview_speed
            log.info("ghost trajectory: %d waypoints, %.1fs real → %.1fs playback (%.1fx slow)",
                     len(traj.times), traj.duration(), scaled, 1.0 / self._preview_speed)

    def _apply_kinematics(self, mujoco) -> None:
        # NOTE: in ghost-preview mode we do NOT move the real arm — the ghost
        # is rendered as a separate semi-transparent overlay in _render.
        # Only apply live joint positions.
        joints = self.state.joints()
        for name, adr in self._joint_qpos.items():
            j = joints.get(name)
            if j is not None and not math.isnan(j.position):
                self.data.qpos[adr] = j.position

        if self._base_qpos is not None:
            bp = self.state.base_pose()
            if bp is not None:
                self.data.qpos[self._base_qpos : self._base_qpos + 3] = bp.translation
                self.data.qpos[self._base_qpos + 3 : self._base_qpos + 7] = _xyzw_to_wxyz(bp.rotation)
        mujoco.mj_forward(self.model, self.data)

    def _step_physics(self, mujoco) -> None:
        # Real-time: step enough substeps to cover the wall-clock since last call.
        now = time.monotonic()
        if self._phys_last is None:
            n = 1
        else:
            n = int(np.clip(round((now - self._phys_last) / self.model.opt.timestep), 1, 40))
        self._phys_last = now

        joints = self.state.joints() if self.hold_joints else {}
        for _ in range(n):
            # Hold commanded joints at their target; the free base stays dynamic.
            for name, adr in self._joint_qpos.items():
                j = joints.get(name)
                if j is not None and not math.isnan(j.position):
                    self.data.qpos[adr] = j.position
                    self.data.qvel[self._joint_dof[name]] = 0.0
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ #
    # plan preview (B) and obstacle rendering (C)
    # ------------------------------------------------------------------ #

    # --- goal preview: ghost arm at target pose before planning ----------
    def show_goal_preview(self, goal_positions: Dict[str, float], name: str = "",
                          hold: float = 1.5) -> None:
        """Set a goal pose to be rendered as a ghost overlay for ``hold`` seconds.

        Called by the demo when it sends a planning goal to MoveIt — the user
        sees the target pose *before* the planned trajectory comes back.
        ``goal_positions`` maps joint names (as in the state) to radian values.
        """
        self._goal_preview_pos = goal_positions
        self._goal_preview_until = time.monotonic() + hold
        self._goal_preview_name = name
        self._preview_state = "goal"
        # Cancel any ongoing trajectory preview — the new goal takes over.
        self._last_traj = None
        # Unlock ghost: a new goal starts a fresh preview cycle.
        self._ghost_locked = False
        # Do NOT call self.update() here — it would invoke cv2.imshow/waitKey
        # from the goal-loop thread while the main render thread does the same.
        # OpenCV is not thread-safe -> deadlock. The main loop (60 Hz) picks up
        # the ghost within ~16 ms.

    def _goal_preview_positions(self) -> Optional[Dict[str, float]]:
        """Return active goal-preview joint targets, or None if expired."""
        if self._goal_preview_pos is None:
            return None
        if time.monotonic() > self._goal_preview_until:
            self._goal_preview_pos = None
            return None
        return self._goal_preview_pos

    def _draw_goal_ghost(self, mujoco, scene) -> None:
        """Draw colored spheres at each arm link's position at the goal-preview pose.

        Color coding:
          - green  = goal preview (target pose before planning)
          - yellow = planned trajectory (ghost playback of display_planned_path)

        We temporarily set qpos to the goal, ``mj_forward``, read the body
        world positions of the arm chain, restore qpos, then add translucent
        spheres at those locations — a visual "where the arm is heading"
        indicator.
        """
        goal = self._goal_preview_positions()
        if goal is None or self.model is None:
            return

        # Green for goal preview, yellow for planned-trajectory playback
        if self._preview_state == "planning":
            sphere_rgba = np.array([0.95, 0.85, 0.20, 0.55], np.float32)  # yellow
        else:
            sphere_rgba = np.array([0.20, 0.85, 0.30, 0.55], np.float32)  # green

        # Save and temporarily set goal joints
        saved = {}
        for name, adr in self._joint_qpos.items():
            if name in goal:
                saved[adr] = self.data.qpos[adr]
                self.data.qpos[adr] = goal[name]
        mujoco.mj_forward(self.model, self.data)

        # Draw a sphere at each arm body's world position
        eye = np.eye(3).flatten()
        sphere_size = np.array([0.04, 0.04, 0.04])
        for bid in range(self.model.nbody):
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not bname or not bname.startswith("arm_0"):
                continue
            pos = self.data.xpos[bid].copy()
            if scene.ngeom < scene.maxgeom:
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                    sphere_size, np.ascontiguousarray(pos, float),
                    eye, sphere_rgba,
                )
                scene.ngeom += 1

        # Restore qpos
        for adr, val in saved.items():
            self.data.qpos[adr] = val
        mujoco.mj_forward(self.model, self.data)

    def _draw_preview_ghost(self, mujoco, scene, preview_pos: Dict[str, float]) -> None:
        """Draw yellow spheres at each arm link during trajectory-preview playback.

        Unlike ``_draw_goal_ghost`` (which uses the goal-preview positions),
        this uses the interpolated positions from the planned trajectory that
        ``_preview_positions`` already wrote into qpos via ``_apply_kinematics``.
        Since qpos is already set, we just read body positions directly.
        """
        if self.model is None:
            return
        sphere_rgba = np.array([0.95, 0.85, 0.20, 0.55], np.float32)  # yellow
        eye = np.eye(3).flatten()
        sphere_size = np.array([0.04, 0.04, 0.04])
        for bid in range(self.model.nbody):
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not bname or not bname.startswith("arm_0"):
                continue
            pos = self.data.xpos[bid].copy()
            if scene.ngeom < scene.maxgeom:
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                    sphere_size, np.ascontiguousarray(pos, float),
                    eye, sphere_rgba,
                )
                scene.ngeom += 1

    def _preview_positions(self) -> Optional[Dict[str, float]]:
        """Interpolated joint targets while playing the latest planned path.

        Playback is slowed by ``_preview_speed`` so a 0.1s trajectory takes
        ~0.5s to play out — long enough for the user to see the motion.
        _preview_state is set by _check_new_trajectory (called in update()).
        """
        if self._preview_state != "planning":
            return None
        traj = self._last_traj
        if traj is None:
            return None
        scaled_duration = traj.duration() / self._preview_speed
        elapsed = time.monotonic() - self._traj_start
        if elapsed > scaled_duration + self.preview_hold:
            if self._loop_preview:
                # Loop: restart from the beginning so the ghost trajectory
                # keeps playing until the user confirms or skips.
                self._traj_start = time.monotonic()
                elapsed = 0.0
            else:
                self._preview_state = "idle"
                return None  # finished -> hand back to live joints
        # Map wall-clock elapsed -> trajectory time (slowed)
        t = min(elapsed * self._preview_speed, traj.duration())
        out = {}
        for j, name in enumerate(traj.joint_names):
            model_name = self.joint_remap.get(name, name)
            if model_name in self._joint_qpos:
                out[model_name] = float(np.interp(t, traj.times, traj.positions[:, j]))
        return out

    def _body_for_frame(self, mujoco, frame_id: str):
        """Return (body_id, optical_fix) for a cloud frame; -1 if none found.

        optical_fix means the exact frame isn't a model body (it's driver-only
        tf) and we fell back to the camera link, so the ROS optical rotation
        must be applied to the points."""
        if frame_id in self._frame_body:
            return self._frame_body[frame_id]
        name = frame_id.split("/")[-1]  # strip namespace, e.g. a200_0553/cam -> cam
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        optical_fix = False
        if bid < 0:  # optical frames are driver-only tf -> use the camera link + correct
            for cand in ("camera_0_link", "camera_0_bottom_screw_frame"):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cand)
                if bid >= 0:
                    break
            optical_fix = name.endswith("optical_frame")
        if bid < 0:
            log.warning("no MuJoCo body for obstacle frame %r; obstacles not shown", frame_id)
        self._frame_body[frame_id] = (bid, optical_fix)
        return bid, optical_fix

    def _voxelize(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        keys = np.floor(pts / self.obstacle_voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        vox = pts[idx]
        if len(vox) > self.obstacle_max:
            vox = vox[:: max(1, len(vox) // self.obstacle_max)]
        return vox

    def _world_clouds(self, mujoco):
        """Yield each obstacle cloud's points in world coordinates.

        Placement goes through the twin's own kinematics: the cloud's frame_id
        is resolved to a model body, whose current pose transforms the points.
        """
        for cloud in self.state.obstacles().values():
            if cloud.points is None or len(cloud.points) == 0:
                continue
            bid, optical_fix = self._body_for_frame(mujoco, cloud.frame_id)
            if bid < 0:
                continue
            R = self.data.xmat[bid].reshape(3, 3)  # world <- body
            if optical_fix:  # body <- optical, so world <- optical
                R = R @ _R_LINK_FROM_OPTICAL
            yield cloud.points @ R.T + self.data.xpos[bid]

    def _draw_obstacles(self, mujoco, scene) -> None:
        size = np.full(3, self.obstacle_voxel * 0.5)
        eye = np.eye(3).flatten()
        rgba = np.array([0.90, 0.35, 0.20, 1.0], np.float32)
        for world in self._world_clouds(mujoco):
            for p in self._voxelize(world):
                if scene.ngeom >= scene.maxgeom:
                    break
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                    size, np.ascontiguousarray(p, float), eye, rgba,
                )
                scene.ngeom += 1

    # --- auto-framing: fit the free camera around robot + obstacle cloud ----
    def _robot_aabb(self, mujoco):
        """World AABB over the model's geoms, or None. Ground plane excluded."""
        lo = hi = None
        for i in range(self.model.ngeom):
            if int(self.model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue  # infinite -- would swallow the bounds
            r = float(self.model.geom_rbound[i])  # geom bounding-sphere radius
            c = self.data.geom_xpos[i]
            glo, ghi = c - r, c + r
            lo = glo if lo is None else np.minimum(lo, glo)
            hi = ghi if hi is None else np.maximum(hi, ghi)
        return lo, hi

    def _maybe_auto_frame(self, mujoco) -> None:
        """Frame the free camera on robot + obstacles, once, on the first cloud.

        An obstacle cloud sits wherever the sensor looked -- in a recording that
        can be metres off to the side of the robot, entirely outside the
        robot-centred startup view, which makes a working overlay look like a
        broken one.  So on the first non-empty cloud, aim the camera at the
        centre of the combined AABB and pull it back far enough for the
        enclosing sphere to fit the vertical FOV.  Because the *sphere* is
        fitted, azimuth/elevation keep whatever the caller chose.

        One-shot by design: re-framing on every cloud would fight the user's
        mouse.  Not applied to an explicit cam_distance / cam_lookat.
        """
        if self._autoframed or self.state is None:
            return
        lo = hi = None
        for world in self._world_clouds(mujoco):
            clo, chi = world.min(axis=0), world.max(axis=0)
            lo = clo if lo is None else np.minimum(lo, clo)
            hi = chi if hi is None else np.maximum(hi, chi)
        if lo is None:
            return  # no cloud yet -- keep the startup framing and retry next frame
        rlo, rhi = self._robot_aabb(mujoco)
        if rlo is not None:
            lo, hi = np.minimum(lo, rlo), np.maximum(hi, rhi)

        lookat = 0.5 * (lo + hi)
        radius = 0.5 * float(np.linalg.norm(hi - lo))
        fovy = math.radians(float(self.model.vis.global_.fovy) or 45.0)
        distance = max(radius / max(math.tan(0.5 * fovy), 1e-3) * _AUTO_FRAME_MARGIN, 0.1)
        if not (np.isfinite(lookat).all() and math.isfinite(distance)):
            # The standard decoder drops non-finite points; a hand-filled cloud
            # might not. Framing on a NaN would blank the render -- worse than
            # not framing at all, so keep the startup view and say so once.
            log.warning("auto-frame skipped: obstacle bounds are not finite")
            self._autoframed = True
            return
        self.cam_lookat, self.cam_distance = lookat, distance
        for cam in (self._mjcam, getattr(self._viewer, "cam", None)):
            if cam is not None:
                cam.lookat[:] = lookat
                cam.distance = distance
        self._autoframed = True
        log.info("auto-framed camera on robot+obstacles: lookat=(%.2f, %.2f, %.2f), "
                 "distance=%.2f (pass cam_lookat/cam_distance to keep your own framing)",
                 lookat[0], lookat[1], lookat[2], distance)

    def _render(self, mujoco) -> bool:
        if self.auto_frame:
            self._maybe_auto_frame(mujoco)  # before update_scene: same-frame effect
        if self.render_mode == "viewer":
            if self._viewer is None or not self._viewer.is_running():
                return False
            if self.show_obstacles:  # draw obstacle voxels into the viewer overlay
                self._viewer.user_scn.ngeom = 0
                self._draw_obstacles(mujoco, self._viewer.user_scn)
            self._viewer.sync()
            return True

        frame = None
        if self._renderer is not None:
            if self._mjcam is not None:
                self._renderer.update_scene(self.data, camera=self._mjcam)
            else:
                self._renderer.update_scene(self.data)
            if self.show_obstacles:
                self._draw_obstacles(mujoco, self._renderer.scene)
            frame = self._renderer.render()  # HxWx3 RGB uint8

            # --- ghost overlay: render the arm at the preview/goal position ---
            ghost_positions = self._ghost_positions()
            if ghost_positions is not None and frame is not None:
                log.debug("ghost_positions: %d joints, state=%s",
                          len(ghost_positions), self._preview_state)
                ghost_frame = self._render_ghost(mujoco, ghost_positions)
                if ghost_frame is not None:
                    frame = self._alpha_blend(frame, ghost_frame, alpha=0.45)
                else:
                    log.debug("ghost_frame was None — renderer not ready?")
            else:
                if self._preview_state == "planning":
                    log.debug("ghost_positions=None but state=planning — "
                             "preview_positions returned None (traj expired?)")

        self._maybe_snapshot(frame)

        if self.render_mode == "none" or not self.display or frame is None:
            return True
        return self._show(frame)

    def _ghost_positions(self) -> Optional[Dict[str, float]]:
        """Return the joint positions for the ghost overlay, or None.

        Priority: planned-trajectory preview (animated) > goal preview (static).
        When a planned trajectory arrives, it takes over the ghost so the user
        sees the full animated path — not just the static end pose.
        """
        if self.preview and self._preview_state == "planning":
            prev = self._preview_positions()
            if prev is not None:
                return prev
        gp = self._goal_preview_positions()
        if gp is not None:
            return gp
        return None

    def _render_ghost(self, mujoco, ghost_pos: Dict[str, float]):
        """Render a frame with the arm at ghost_pos; return RGB or None."""
        if self._renderer is None:
            return None
        # Save qpos, set ghost joints, forward, render, restore
        saved = {}
        for name, adr in self._joint_qpos.items():
            if name in ghost_pos:
                saved[adr] = self.data.qpos[adr]
                self.data.qpos[adr] = ghost_pos[name]
        mujoco.mj_forward(self.model, self.data)

        if self._mjcam is not None:
            self._renderer.update_scene(self.data, camera=self._mjcam)
        else:
            self._renderer.update_scene(self.data)
        ghost_frame = self._renderer.render()

        for adr, val in saved.items():
            self.data.qpos[adr] = val
        mujoco.mj_forward(self.model, self.data)
        return ghost_frame

    @staticmethod
    def _alpha_blend(base: np.ndarray, overlay: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """Blend overlay over base with given alpha (overlay is the ghost)."""
        return ((1.0 - alpha) * base.astype(np.float32) +
                alpha * overlay.astype(np.float32)).astype(np.uint8)

    # ------------------------------------------------------------------ #
    def _show(self, frame_rgb: np.ndarray) -> bool:
        try:
            import cv2
        except ImportError:
            # No GUI backend -- degrade to snapshots only, once.
            if self.display:
                log.warning("OpenCV not available; disabling live display (use snapshot_path).")
                self.display = False
            return True
        try:
            if not self._win_ready:  # create window + wire mouse orbit/zoom once
                cv2.namedWindow(self._win_main, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(self._win_main, self._on_mouse)
                self._win_ready = True
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._overlay_hud(cv2, bgr)
            cv2.imshow(self._win_main, bgr)
            self._maybe_show_sensor(cv2)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC quits
                return False
            self._handle_key(key)
            return True
        except cv2.error as exc:  # headless OpenCV build
            log.warning("OpenCV display failed (%s); disabling live display.", exc)
            self.display = False
            return True

    # ------------------------------------------------------------------ #
    # offscreen-window navigation (mouse orbit/pan/zoom + keys) — no mjpython
    # ------------------------------------------------------------------ #
    def _on_mouse(self, event, x, y, flags, param) -> None:
        cam = self._mjcam
        if cam is None:
            return
        import cv2

        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._mouse_last = (x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._mouse_last = None
        elif event == cv2.EVENT_MOUSEMOVE and self._mouse_last is not None:
            dx, dy = x - self._mouse_last[0], y - self._mouse_last[1]
            self._mouse_last = (x, y)
            pan = (flags & cv2.EVENT_FLAG_RBUTTON) or (flags & cv2.EVENT_FLAG_SHIFTKEY)
            if pan:
                self._pan(cam, dx, dy)
            else:  # orbit
                cam.azimuth = (cam.azimuth - dx * 0.3) % 360
                cam.elevation = float(np.clip(cam.elevation - dy * 0.3, -89.0, 89.0))
        elif event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            # On the macOS Cocoa backend the wheel delta is delivered in the
            # x/y callback args (y = vertical, x = horizontal), NOT in `flags`,
            # which there only carries modifier keys. getMouseWheelDelta(flags)
            # would read those modifiers as the delta and return 0, so every
            # scroll -- regardless of direction -- hit the `else` branch and
            # only ever zoomed out. Read the axis delta on macOS instead.
            if sys.platform == "darwin":
                delta = y if event == cv2.EVENT_MOUSEWHEEL else x
            elif hasattr(cv2, "getMouseWheelDelta"):
                delta = cv2.getMouseWheelDelta(flags)
            else:
                delta = flags
            if delta:
                steps = max(1, abs(int(delta)))
                cam.distance *= 0.9 ** steps if delta > 0 else 1.1 ** steps

    def _pan(self, cam, dx, dy) -> None:
        az = np.radians(cam.azimuth)
        right = np.array([np.sin(az), -np.cos(az), 0.0])
        scale = 0.0015 * cam.distance
        cam.lookat[:] = np.asarray(cam.lookat) + right * (-dx * scale) + np.array([0, 0, 1]) * (dy * scale)

    def _handle_key(self, key: int) -> None:
        cam = self._mjcam
        if cam is None or key in (0, 255):
            return
        c = chr(key) if 32 <= key < 128 else ""
        if c in ("+", "="):
            cam.distance *= 0.9
        elif c in ("-", "_"):
            cam.distance *= 1.1
        elif c == "a":
            cam.azimuth = (cam.azimuth - 5) % 360
        elif c == "d":
            cam.azimuth = (cam.azimuth + 5) % 360
        elif c == "w":
            cam.elevation = float(np.clip(cam.elevation + 5, -89.0, 89.0))
        elif c == "s":
            cam.elevation = float(np.clip(cam.elevation - 5, -89.0, 89.0))

    def _overlay_hud(self, cv2, bgr) -> None:
        n = len(self._joint_qpos)
        txt = f"TwinLink  joints:{n}"
        if self.state is not None:
            bp = self.state.base_pose()
            if bp is not None:
                txt += f"  base:({bp.translation[0]:+.2f},{bp.translation[1]:+.2f})"
        cv2.putText(bgr, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 60), 1, cv2.LINE_AA)

        # State indicator (color-coded) — only shown during preview, not live
        gp = self._goal_preview_positions()
        if self._preview_state == "planning" and gp is not None:
            label = "◆ PLANNED TRAJECTORY (yellow ghost) — review, then confirm"
            color = (0, 220, 220)  # yellow (BGR)
            cv2.putText(bgr, label, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        elif gp is not None:
            label = f"▶ GOAL: {self._goal_preview_name}" if self._goal_preview_name else "▶ GOAL PREVIEW"
            color = (60, 220, 60)  # green (BGR)
            cv2.putText(bgr, label, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        hint = "drag: orbit  scroll/+-: zoom  shift-drag: pan  ESC: quit"
        cv2.putText(bgr, hint, (10, bgr.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    def _maybe_show_sensor(self, cv2=None) -> None:
        if not self.show_sensor_camera or not self.display:
            return
        if cv2 is None:
            try:
                import cv2  # noqa
            except ImportError:
                return
        cam = self.state.camera(self.show_sensor_camera)
        if cam is None:
            return
        img = cam.image
        if img.ndim == 2:  # depth -> colormap
            norm = np.zeros_like(img, dtype=np.uint8)
            valid = np.isfinite(img) & (img > 0)
            if valid.any():
                vmax = float(np.percentile(img[valid], 95)) or 1.0
                norm = np.clip(img / vmax * 255.0, 0, 255).astype(np.uint8)
            disp = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        elif "rgb" in cam.encoding.lower():
            disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            disp = img
        cv2.imshow(self._win_cam, disp)

    def _maybe_snapshot(self, frame_rgb) -> None:
        if not self.snapshot_path or frame_rgb is None:
            return
        clock = self.state.last_update if self.state else 0.0
        if self.snapshot_every > 0 and (clock - self._last_snapshot) < self.snapshot_every:
            return
        self._last_snapshot = clock
        try:
            import cv2

            cv2.imwrite(self.snapshot_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            log.debug("snapshot write failed: %s", exc)

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        if self.display:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
