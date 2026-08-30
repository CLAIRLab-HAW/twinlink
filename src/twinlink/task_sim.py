"""Task simulation on the compiled robot model (robot-agnostic).

The execution model is that of
:class:`twinlink.sinks.mujoco_sink.MujocoSink` with ``physics=True``:
*commanded joints are written into qpos and held every substep (qvel=0),
while everything else obeys physics*.

Grasping follows the same kinematic philosophy: if the gripper closes within reach of a registered object, that object
is carried kinematically (its free joint follows the TCP pose every substep); opening hands it back to physics.

This module knows neither task nor robot: *which* objects are graspable is said by the subclass; *what* the robot is
called is said by :class:`RobotSimSpec`.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .events import SimEvents
from .mjcf_scene import OBSTACLE_PARK, OBSTACLE_POOL_SIZE, camera_extrinsics, camera_intrinsics, obstacle_body_name
from .quaternion import mat_to_quat_wxyz, quat_about_z_wxyz, quat_conj_wxyz, quat_mul_wxyz

log = logging.getLogger("twinlink.task_sim")


class GripperLinkage(Protocol):
    """The mapping grasp width ◀─▶ driver joint -- HANDED IN, never here.

    This module only describes what it needs from the mapping; the formula itself belongs to the robot, not to the sim.
    ``husky_sdk.sim`` passes ``profile.gripper.linkage`` through for that.

    Why the mapping and not two anchors: if the sim were given only ``gripper_open``/``gripper_closed`` and interpolated
    linearly, that would be one more copy of the same computation next to ``plan_bridge.plan_server``.  Estimated anchor
    pairs are off (the open hand would stand for 93.7 mm instead of 159 mm), and because every copy rounds for itself,
    correcting a single one would drive the others apart all the more.  ``twinlink`` deliberately does not depend on
    ``robot_contract``, so the mapping is passed in.
    """

    @property
    def open_rad(self) -> float:
        """Driver joint of the widest open hand."""

    @property
    def closed_rad(self) -> float:
        """Driver joint of the fully closed hand."""

    @property
    def max_width_m(self) -> float:
        """Largest clear width the linkage can produce."""

    def width_from_angle(self, q: float) -> float:
        """Clear width [m] at the driver joint ``q`` [rad]."""

    def angle_from_width(self, width_m: float) -> float:
        """Driver joint [rad] for the clear width ``width_m`` [m]."""


@dataclass(frozen=True)
class RobotSimSpec:
    """The robot facts the task sim needs.

    twinlink is a self-contained package and deliberately does NOT depend on ``robot_contract`` -- these values are
    handed in.  For robots with a robot-contract profile ``husky_sdk.sim.robot_sim_spec()`` does that.
    """

    #: Body prefixes of the movable manipulator; everything else is platform.
    manipulator_prefixes: tuple[str, ...]
    #: Prefixes of the hand assembly (gripper + wrist-mounted sensors).
    hand_prefixes: tuple[str, ...]
    #: Prefixes of the gripper shells -- ONLY the jaws/the gripper housing,
    #: without sensors riding along.  Only these geoms are made permeable for
    #: graspable objects (kinematic grasping, see
    #: ``TwinTaskSim._setup_collision_masks``); a wrist-mounted camera belongs
    #: to the hand but must keep bumping into objects, otherwise it loses its
    #: collision events.
    gripper_prefixes: tuple[str, ...]
    #: Bodies far from the arm -- contact hand◀─▶here = folded configuration.
    far_arm_bodies: tuple[str, ...]
    #: Jaw travel of the gripper model (m) with the hand open.
    gripper_stroke_m: float
    #: The GRASP SURFACES, by name -- not "everything with a gripper prefix".
    #: The half pad width is measured over these only: on 2026-08-19 the median
    #: over the whole hand landed on a lever (39.2 mm instead of 6.8 mm), and
    #: the capture tolerance was three times too large as a result.  Empty =
    #: not configured (then the measurement falls back on the whole hand).
    #: Body whose pose counts as the TCP.
    tcp_body: str
    #: Joints of the planning group, in SRDF order.
    arm_joints: tuple[str, ...]
    pad_bodies: tuple[str, ...] = ()


#: TCP proximity (m) within which a closing gripper captures an object.  The
#: grasp TCP sits ~45 mm above the object centre (fingertips extend ~60 mm past
#: the TCP and must clear the table), so the radius allows for that offset
#: plus IK error while staying far below the 110 mm object spawn separation.
GRASP_RADIUS = 0.07

#: Maximum yaw misalignment (deg, modulo 90) between the object's faces and the
#: gripper pads for a capture.  A parallel-jaw gripper needs the object faces
#: roughly parallel to the pads: beyond this the pads meet edges/corners and
#: the object twists out instead of being grasped.  Within the tolerance the
#: flat rubber pads square the object up while closing (see ``_try_grasp``).
GRASP_MAX_MISALIGN_DEG = 20.0

#: How much TILT the pads can still straighten out while closing -- beyond
#: that there is no grasp.
#:
#: Deliberately applies ONLY to the tilt, not to the yaw angle: the yaw path is
#: pinned by golden traces (tightened, the block stack breaks immediately) and
#: measured in the study at at most 2.1 degrees.
#:
#: Why the limit exists: belief ─▶ action ─▶ change of the world ─▶ read back
#: as "truth" ─▶ belief.  Causally legitimate (a real gripper really does
#: straighten a pen when it grabs it), but without a limit the twin models it
#: as a FREE snap: everything up to :data:`GRASP_MAX_MISALIGN_DEG` is corrected
#: for nothing, with no failure case.  That turns abstraction errors into
#: nothing -- and a coarse rung looks sufficient because the twin itself
#: repaired the consequence of its coarseness.  Compliance is real, but
#: limited: flat pads straighten a slightly crooked body, a strongly crooked
#: one slips out.
#:
#: THE VALUE IS A MODELLING DECISION, not a measurement result, and explicitly
#: a knob.  The measured distribution over 68 container runs ranges from 0.0 to
#: 2.1 degrees -- the limit binds NOWHERE today.  It is a guard against a
#: failure mode, not a correction of a result.
PAD_SQUARE_LIMIT_DEG = 15.0

#: Robot-vs-table/ground contacts shallower than this are not collision
#: events: the finger collision envelope is the union over the whole finger
#: sweep and overstates fingertip depth by up to ~15 mm during a tabletop
#: grasp.  Real crashes (mis-planned motions) penetrate far deeper.
COLLISION_PENETRATION_TOL = 0.02


def _wrap_half(angle: float) -> float:
    """Wrap an angle (rad) into [-pi/2, pi/2) -- axis alignment modulo 180 deg."""
    return float((angle + np.pi / 2.0) % np.pi - np.pi / 2.0)


def _wrap_quarter(angle: float) -> float:
    """Wrap an angle (rad) into [-pi/4, pi/4) -- yaw modulo a square's symmetry."""
    quarter = np.pi / 2.0
    return float((angle + quarter / 2.0) % quarter - quarter / 2.0)


class TwinTaskSim:
    """MuJoCo scene + twin execution semantics for a manipulation task.

    The generic half of a task sim: joint
    indexing, contact classification, the perceived-obstacle pool, kinematic
    grasping, stepping, the goal-state collision gate and the render/camera
    plumbing.  Task knowledge enters through exactly two hooks
    (:meth:`register_graspables`, :meth:`support_geom_names`) plus the
    constructor arguments; robot knowledge enters through
    :class:`RobotSimSpec`.
    """

    def __init__(
        self,
        model,
        spec: RobotSimSpec,
        *,
        scene_prefix: str,
        n_substeps: int = 10,
        n_obstacle_slots: int = OBSTACLE_POOL_SIZE,
        default_span: float = 0.045,
        release_clearance: float = 0.005,
        gripper_follower_factors: dict[str, float],
        gripper_linkage: GripperLinkage,
        home_pose: dict[str, float],
        render_size: tuple[int, int] = (640, 480),
        actuated_gripper: bool = False,
        gripper_ramp_ticks: int = 0,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.spec = spec
        self.model = model
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(n_substeps)
        self.control_dt = float(self.model.opt.timestep) * self.n_substeps
        #: Body-name prefix of the app's own scene furniture (never hidden
        #: from the render passes, see :meth:`_hide_robot_collision_geoms`).
        #: Mandatory constructor argument on purpose: with a default of ``""``
        #: every ``bname.startswith(prefix)`` is true and the render-group
        #: split turns into a silent no-op.  An empty string is still allowed
        #: -- it means "this sim authors no furniture of its own" and is
        #: handled explicitly, never by the startswith accident.
        self._scene_prefix = scene_prefix
        #: The two obstacle body families of THIS scene, derived from the same
        #: prefix the constructor was given -- not from the module constants
        #: (those are nailed to ``hrl_`` and exist only for legacy consumers).
        #: The prefix must be wired through COMPLETELY: if only the render
        #: split followed it, and not classification and pool indexing as well,
        #: a second app would be silently blind to the entire obstacle class.
        #: ``scene_prefix=""`` is harmless here -- the derived values stay
        #: non-empty, so a ``startswith`` cannot match everything.
        self._obstacle_body_prefix = f"{scene_prefix}obstacle_"
        self._distractor_body_prefix = f"{scene_prefix}distractor_"
        #: Closing span (m) assumed when nothing is captured (task object size).
        self._default_span = float(default_span)
        #: Clearance per side (m) with which the hand RELEASES an object.
        #: Tearing it wide open to release is a choice, not a necessity, and
        #: it costs: measured on the container (2026-08-16) the fully open
        #: hand stood in the coarsened gate after placing down, and move_group
        #: refused the retreat -- ``2 contact(s) detected : gate_0 -
        #: rg6_gripper_finger_2_flex_finger, gate_1 -
        #: rg6_gripper_finger_1_flex_finger``.  A
        #: real RG6 opens only as far as the object demands.
        self._release_clearance = float(release_clearance)
        self._gripper_follower_factors: dict[str, float] = dict(gripper_follower_factors)
        #: Width ◀─▶ driver joint.  See :class:`GripperLinkage`: the formula
        #: belongs to the robot and is handed in so that it does not stand in
        #: the stack for the fourth time.
        self._linkage = gripper_linkage
        self._gripper_open = float(gripper_linkage.open_rad)
        self._gripper_closed = float(gripper_linkage.closed_rad)
        self._home_pose: dict[str, float] = dict(home_pose)
        self._render_size: tuple[int, int] = (int(render_size[0]), int(render_size[1]))
        #: Opt-in: drive the follower joints through the model's actuators
        #: instead of pinning their qpos.  Off is the twin's normal regime --
        #: everything commanded is held, and the gripper shells are permeable
        #: to the objects (grasping is a kinematic capture).  On, the fingers
        #: close against the object and a grip force exists; see
        #: :meth:`_drive_gripper` and :meth:`_setup_collision_masks`.
        self._actuated_gripper = bool(actuated_gripper)
        #: follower joint ─▶ actuator id, filled only in the actuated regime.
        self._gripper_actuators: dict[str, int] = {}
        #: Over how many ticks the closing runs VISIBLY in the non-actuated
        #: regime.  ``0`` = off: the joints stand on the target in the first
        #: substep, in the picture the hand jumps open and shut binarily.
        #:
        #: Only the WAY there changes, never the target -- and the default
        #: stays off because a longer closing motion shifts timing behaviour
        #: and with it measured values.  Switch on explicitly for recordings.
        self._gripper_ramp_ticks = max(0, int(gripper_ramp_ticks))
        #: Angle at which the running ramp started, and how many ticks it
        #: still has to go.
        self._gripper_ramp_from: float = 0.0
        self._gripper_ramp_left: int = 0
        self._gripper_ramp_goal: float | None = None
        #: The angle that was REALLY written in this tick -- what the renderer
        #: shows.  Without a ramp always the command.
        self._gripper_applied: float = float(self._gripper_open)

        self._joint_qpos: dict[str, int] = {}
        self._joint_dof: dict[str, int] = {}
        self._index_joints()
        if self._actuated_gripper:
            self._index_gripper_actuators()
        # (contype, conaffinity) per object geom, to suspend/restore contacts while the object is carried (see
        # _suspend_object_contacts).
        self._object_contact_masks: dict[str, dict[int, tuple[int, int]]] = {}
        # Scratch MjData for goal-state collision checks (lazily created).
        self._collision_scratch = None
        #: Scratch for the jaw height at the closed hand (see
        #: :meth:`_pad_heights`) -- kept apart from the collision scratch so
        #: that the two do not overwrite each other.
        self._pad_scratch = None
        self._pad_width = None
        self._tcp_body_id = self._body_id(spec.tcp_body)
        self._graspable: dict[str, dict] = {}
        self.register_graspables()
        self._classify_geoms()
        self._setup_collision_masks()
        self._hide_robot_collision_geoms()
        self._index_obstacle_pool(n_obstacle_slots)

        # Commanded state (held every substep, twin-style).
        self._arm_command: dict[str, float] = dict(self._home_pose)
        self._gripper_command: float = self._gripper_open
        #: Whether the hand is commanded CLOSED -- see :meth:`gripper_closed`.
        self._gripper_closing: bool = False
        # label ─▶ (pos offset in TCP frame, quat offset) while carried.
        self._grasped: str | None = None
        self._grasp_offset: tuple[np.ndarray, np.ndarray] | None = None
        # Closing span (m) of the captured face pair -- drives the finger command width; None falls back to the default
        # span.
        self._grasp_span: float | None = None
        #: How far the capture had to rotate the object while gripping (rad).
        #: See :meth:`grasp_misalign_deg`.
        self._grasp_misalign: float | None = None
        #: Gap at the moment of capture (m).  See :meth:`grasp_gap`.
        self._grasp_gap0: float | None = None
        #: Worst distance of the carried body to the real world during the
        #: whole journey.  See :meth:`carried_world_gap_min`.
        self._carry_gap_min: float | None = None
        self._carry_tick: int = 0
        #: Has the carried body already left the support surface?  Only from
        #: then on is a distance a statement about the JOURNEY.
        self._carry_airborne: bool = False
        #: WHAT came closest to the carried body.  A number without a name is
        #: not actionable: "0.0 mm" does not say whether that was the table
        #: while setting down or a gate post halfway along.
        self._carry_gap_who: str = ""
        self._carry_gap_min_who: str = ""
        # Events raised outside step_physics (grasp/release on command) are accumulated here and drained by the next
        # step_physics call.
        self._event_acc = SimEvents()

        self._renderer = None
        self._render_wh: tuple[int, int] | None = None

        mujoco.mj_forward(self.model, self.data)
        log.info(
            "sim ready: %d joints indexed, control_dt=%.3fs (%d substeps)",
            len(self._joint_qpos),
            self.control_dt,
            self.n_substeps,
        )

    # ------------------------------------------------------------------ #
    # subclass hooks (the only way task knowledge enters)
    # ------------------------------------------------------------------ #
    def register_graspables(self) -> None:
        """Subclasses register their graspable objects here.

        Default: none -- a sim without graspable objects is permitted (pure motion/collision study).
        """

    def support_geom_names(self) -> frozenset:
        """Geom names of the support surface (table or similar) for the contact classes."""
        return frozenset()

    # ------------------------------------------------------------------ #
    # model indexing (pattern from MujocoSink._index_joints)
    # ------------------------------------------------------------------ #
    def _index_joints(self) -> None:
        mujoco = self._mujoco
        single_dof = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name is None or int(self.model.jnt_type[jid]) not in single_dof:
                continue
            self._joint_qpos[name] = int(self.model.jnt_qposadr[jid])
            self._joint_dof[name] = int(self.model.jnt_dofadr[jid])

    def _index_gripper_actuators(self) -> None:
        """Map every follower joint onto the actuator that drives it.

        Found through ``actuator_trnid``, not through a name convention: the app authors the actuators and may call them
        whatever it likes.  A follower without an actuator is a hard error -- the point of the actuated regime is that a
        closing command produces a force, and falling back to the kinematic hold would produce the same
        green-but-forceless state the flag exists to end.
        """
        mujoco = self._mujoco
        for aid in range(self.model.nu):
            if int(self.model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            jid = int(self.model.actuator_trnid[aid, 0])
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname in self._gripper_follower_factors:
                self._gripper_actuators[jname] = aid
        missing = sorted(set(self._gripper_follower_factors) - set(self._gripper_actuators))
        if missing:
            raise RuntimeError(
                "actuated_gripper=True, but the model has no actuator for "
                f"{missing} -- the closing command could not reach them"
            )

    def _free_joint_qpos(self, joint: str) -> int:
        mujoco = self._mujoco
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0:
            raise KeyError(f"free joint {joint!r} not in model")
        return int(self.model.jnt_qposadr[jid])

    def _body_id(self, body: str) -> int:
        mujoco = self._mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise KeyError(f"body {body!r} not in model")
        return bid

    def _classify_geoms(self) -> None:
        """Cache geom-id sets for contact classification."""
        mujoco = self._mujoco
        self._robot_geoms: set = set()
        self._platform_geoms: set = set()
        self._table_geoms: set = set()
        self._ground_geoms: set = set()
        self._obstacle_geoms: set = set()
        self._object_geoms: dict[int, str] = {}
        support = self.support_geom_names()
        self._object_bodies = {entry["body"]: label for label, entry in self._graspable.items()}
        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if self.model.geom_contype[gid] == 0 and self.model.geom_conaffinity[gid] == 0:
                continue  # visual-only geometry (meshes, markers)
            if gname in support:
                self._table_geoms.add(gid)
            elif gname == "twinlink_ground" or int(self.model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                self._ground_geoms.add(gid)
            elif bname.startswith((self._obstacle_body_prefix, self._distractor_body_prefix)):
                # Perceived-obstacle pool slots + authored distractors: the things the arm must plan around (goal gate +
                # contact events). Checked BEFORE the graspable bodies: a distractor the task promotes to a graspable
                # stays an obstacle for the contact classes (unchanged from the pre-split behaviour).
                self._obstacle_geoms.add(gid)
            elif bid in self._object_bodies:
                self._object_geoms[gid] = self._object_bodies[bid]
            elif bname.startswith(self.spec.manipulator_prefixes):
                # Only the manipulator counts for collision penalties; the chassis standing on the ground is normal.
                # (Classify purely by body name: for the welded robot, body_rootid points at base_link, not the world --
                # a root==0 filter silently drops every manipulator geom and no collision would ever fire.)
                self._robot_geoms.add(gid)
            elif bname:
                # Chassis / plates / bumpers / sensor arch: the static part of the robot the manipulator must not touch.
                self._platform_geoms.add(gid)
        if not self._robot_geoms:
            raise RuntimeError("no manipulator geoms classified -- collision events would be blind")
        # Subsets for the goal-state self-collision gate: the hand assembly (gripper + wrist camera) versus arm links
        # far from the wrist.  A contact between those means a fully folded, invalid configuration (the class of state
        # MoveIt rejects as robot self-collision).
        self._hand_geoms: set = set()
        self._armfar_geoms: set = set()
        #: ONLY the grasp surfaces.  Kept apart from _hand_geoms because "the
        #: hand" also contains housing and levers: on 2026-08-19 the median of
        #: the hand widths landed on the moment_arm (39.2 mm) instead of on a
        #: pad (6.8 mm), and the capture tolerance was three times too large as
        #: a result.  With the old gripper model the median happened to hit a
        #: finger.
        self._pad_geoms: set = set()
        armfar_bodies = set(self.spec.far_arm_bodies)
        pad_bodies = set(getattr(self.spec, "pad_bodies", ()) or ())
        for gid in self._robot_geoms:
            bid = int(self.model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.startswith(self.spec.hand_prefixes):
                self._hand_geoms.add(gid)
                if bname in pad_bodies:
                    self._pad_geoms.add(gid)
            elif bname in armfar_bodies:
                self._armfar_geoms.add(gid)
        if pad_bodies and not self._pad_geoms:
            # Fail loudly instead of silently falling back on the whole hand -- it is exactly that silent fallback
            # which tripled the tolerance.
            raise RuntimeError(f"pad_bodies {sorted(pad_bodies)} not found in model -- grip surfaces not classifiable")
        # Registered graspables that are ALSO classified as obstacles (pool slots or authored clutter the task promoted
        # to a target) stay obstacles for the validity and settling questions: the arm must plan around them, so parking
        # them away would make the goal gate blind to exactly the objects it exists for.  Only the sim's own payload is
        # parked/settled -- the pre-split behaviour, expressed without task knowledge.
        self._non_obstacle_graspables: tuple[str, ...] = tuple(
            label for label, entry in self._graspable.items() if not (self._obstacle_geoms & set(entry["geoms"]))
        )

    def _setup_collision_masks(self) -> None:
        """Let the gripper envelope pass through graspables (grasping is kinematic).

        Grasping is proximity capture + kinematic carry (see module docstring), so finger--object contact forces are
        artifacts: the open fingers would shove an object away while descending onto it.  Contact bitmasks exclude
        exactly the gripper◀─▶object pairs; the gripper still collides with table/ground and the arm still collides with
        the objects (knocking them over stays possible).

        Masks: world/robot keep (1, 1); objects get (2, 3); gripper geoms get (4, 1).  gripper&object: 4&3 = 2&1 = 0 ─▶
        no contact.

        Every REGISTERED graspable gets this, task objects and dynamic clutter alike.  Only ``spec.gripper_prefixes`` --
        the jaws/housing -- becomes permeable, NOT the whole ``hand_prefixes`` assembly: a wrist-mounted camera rides
        along with the hand but is not a jaw, and making it permeable would silently drop its obstacle-contact events.

        ``actuated_gripper=True`` inverts the premise and skips the permeability: there the jaws are driven by servos
        and must MEET the object -- a permeable jaw yields exactly zero finger contacts and zero grip force.  The
        objects keep their (2, 3) mask either way.
        """
        mujoco = self._mujoco
        graspable_geoms = {g for entry in self._graspable.values() for g in entry["geoms"]}
        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if gid in graspable_geoms:
                self.model.geom_contype[gid] = 2
                self.model.geom_conaffinity[gid] = 3
            elif (
                not self._actuated_gripper
                and bname.startswith(self.spec.gripper_prefixes)
                and self.model.geom_contype[gid] != 0
            ):
                self.model.geom_contype[gid] = 4
                self.model.geom_conaffinity[gid] = 1

    #: Geom group used for the robot's collision shells; excluded from every
    #: render pass (RGB *and* depth -- alpha tricks only work for RGB).
    _HIDDEN_GEOM_GROUP = 4

    def _robot_geom_ids(self) -> list[int]:
        """Geom ids of the robot's own kinematic tree.

        Positively identified through ``body_rootid``: every body the URDF contributed shares the robot's root body (the
        welded base link), while each piece of scene furniture is its own world child.  Measured on the a200-0553 scene:
        45 of 58 bodies under ``base_link``.

        Deliberately not the inverse rule -- "everything without the app's scene prefix is robot" -- which makes every
        body an app places under a name of its own invisible to RGB *and* depth, silently: the cameras then see the
        surface behind it and a plausible-looking point cloud comes back without the object in it.
        """
        root = int(self.model.body_rootid[self._tcp_body_id])
        return [
            gid
            for gid in range(self.model.ngeom)
            if int(self.model.body_rootid[int(self.model.geom_bodyid[gid])]) == root
        ]

    def _hide_robot_collision_geoms(self) -> None:
        """Exclude the robot's collision geometry from rendering.

        The robot carries both visual meshes and collision geoms; the
        collision shells (e.g. a gripper base box, much larger than the
        visible gripper) would otherwise occlude the cameras.  The sink's
        alpha=0 trick only affects the RGB pass, while MuJoCo's depth pass
        rasterises transparent geoms too -- so the shells move into a geom
        *group* that all render calls disable via ``MjvOption``.  Restricted
        to the robot subtree; everything else has no separate visual geometry
        and must stay visible.  Groups are visualisation-only.
        """
        robot = self._robot_geom_ids()
        robot_visuals = sum(
            1 for gid in robot if self.model.geom_contype[gid] == 0 and self.model.geom_conaffinity[gid] == 0
        )
        if robot_visuals == 0:
            return  # nothing to fall back on -- keep collision geoms visible
        hidden = 0
        for gid in robot:
            if self.model.geom_contype[gid] != 0 or self.model.geom_conaffinity[gid] != 0:
                self.model.geom_group[gid] = self._HIDDEN_GEOM_GROUP
                hidden += 1
        log.info("moved %d robot collision geom(s) out of the render groups", hidden)

    # ------------------------------------------------------------------ #
    # perceived-obstacle pool (mirror of the tracked real-world obstacles)
    # ------------------------------------------------------------------ #
    #: RGBA of an active obstacle slot (semi-transparent so the twin shows
    #: the perceived box without hiding what is behind it).
    _OBSTACLE_RGBA = (0.85, 0.45, 0.10, 0.55)

    def _index_obstacle_pool(self, n_slots: int) -> None:
        """Cache (body, geom) ids of the pool and park every slot.

        The slot names follow the constructor prefix (``scene_prefix``), exactly like the classification -- searched
        for with the module default, an app with its own prefix would not find its own pool and would run silently
        without obstacles.
        """
        mujoco = self._mujoco
        self._obstacle_slots: list[tuple[int, int]] = []
        for i in range(int(n_slots)):
            name = obstacle_body_name(i, prefix=self._scene_prefix)
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            if bid < 0 or gid < 0:
                break
            self._obstacle_slots.append((bid, gid))
            self._park_obstacle_slot(i)
        self._active_obstacles = 0

    def _park_obstacle_slot(self, index: int) -> None:
        """Deactivate one slot: far from every camera frustum, contacts off."""
        bid, gid = self._obstacle_slots[index]
        px, py, pz = OBSTACLE_PARK
        self.model.body_pos[bid] = (px + 0.5 * index, py, pz)
        self.model.body_quat[bid] = (1.0, 0.0, 0.0, 0.0)
        self._write_obstacle_geom(gid, np.full(3, 0.02))
        self.model.geom_contype[gid] = 0
        self.model.geom_conaffinity[gid] = 0
        self.model.geom_rgba[gid] = (*self._OBSTACLE_RGBA[:3], 0.0)

    def _write_obstacle_geom(self, gid: int, half: np.ndarray) -> None:
        """Set a box geom's half extents plus the derived collision bounds."""
        self.model.geom_size[gid] = half
        self.model.geom_rbound[gid] = float(np.linalg.norm(half))
        if hasattr(self.model, "geom_aabb"):  # BVH midphase bounds
            self.model.geom_aabb[gid, :3] = 0.0
            self.model.geom_aabb[gid, 3:] = half

    def n_obstacle_slots(self) -> int:
        return len(self._obstacle_slots)

    @contextlib.contextmanager
    def obstacles_hidden(self):
        """Exclude the pool from renders while a depth frame is captured.

        The mirrored boxes must never appear in the depth image the obstacle pipeline itself consumes (sim cameras): a
        stale box would otherwise occlude the very view that should prove its space empty -- the mirror would keep
        itself alive.  Visual-only (geom groups), physics untouched.
        """
        gids = [gid for _bid, gid in self._obstacle_slots]
        saved = [int(self.model.geom_group[g]) for g in gids]
        for g in gids:
            self.model.geom_group[g] = self._HIDDEN_GEOM_GROUP
        try:
            yield
        finally:
            for g, s in zip(gids, saved):
                self.model.geom_group[g] = s

    def set_obstacles(self, boxes: list) -> int:
        """Mirror perceived obstacles into the collision pool.

        ``boxes`` are duck-typed (``.center`` (3,) world, ``.size`` (3,) full extents).  Slots beyond ``len(boxes)`` are
        parked; when more boxes than slots arrive the largest ones win (safety-relevant volume first).  The boxes enter
        physics immediately: MoveIt-side planning uses the planning-scene copy, this pool covers the twin, the contact
        events and the client-side IK goal gate (:meth:`arm_config_collides`).

        A slot is a PERCEPTION of the world, never a body in it: it must not exert force on the objects it depicts (same
        reasoning as the permeable gripper shells).  In sim a perceived box lands exactly on the body it was perceived
        from -- with ordinary contacts the solver would eject the original out from under its own ghost.

        Returns the number of active slots.
        """
        boxes = list(boxes)
        if len(boxes) > len(self._obstacle_slots):
            boxes.sort(key=lambda b: -float(np.prod(np.asarray(b.size, dtype=float))))
            log.warning(
                "%d obstacles exceed the %d-slot pool -- keeping the largest", len(boxes), len(self._obstacle_slots)
            )
            boxes = boxes[: len(self._obstacle_slots)]
        for i, (bid, gid) in enumerate(self._obstacle_slots):
            if i < len(boxes):
                box = boxes[i]
                half = np.maximum(np.asarray(box.size, dtype=float) / 2.0, 1e-3)
                yaw = float(getattr(box, "yaw", 0.0))
                self.model.body_pos[bid] = np.asarray(box.center, dtype=float)
                self.model.body_quat[bid] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
                self._write_obstacle_geom(gid, half)
                # Masks (see _setup_collision_masks for the scheme): world and robot keep (1, 1), graspables get (2, 3),
                # gripper shells (4, 1).  A slot gets (4, 5) -- permeable to graspables (4&3 = 2&5 = 0), still felt by
                # the arm (1&5 = 1) and by the gripper shells (4&5 = 4), so the goal gate and the
                # robot_obstacle_collision events keep seeing exactly the objects they exist for.
                self.model.geom_contype[gid] = 4
                self.model.geom_conaffinity[gid] = 5
                self.model.geom_rgba[gid] = self._OBSTACLE_RGBA
            else:
                self._park_obstacle_slot(i)
        self._active_obstacles = len(boxes)
        self._mujoco.mj_forward(self.model, self.data)
        return self._active_obstacles

    # ------------------------------------------------------------------ #
    # commands (written by the motion layer / gripper interface)
    # ------------------------------------------------------------------ #
    def set_arm_command(self, positions: dict[str, float]) -> None:
        """Set the held target for (a subset of) the arm joints."""
        for name, value in positions.items():
            if name in self._joint_qpos:
                self._arm_command[name] = float(value)

    def arm_command(self) -> dict[str, float]:
        return dict(self._arm_command)

    def arm_positions(self) -> dict[str, float]:
        """Current arm joint positions (== command, joints are held)."""
        return {j: float(self.data.qpos[self._joint_qpos[j]]) for j in self.spec.arm_joints}

    def command_gripper(self, close: bool, *, grasp: bool = True) -> None:
        """Binary open/close -- the gripper service semantics.

        Like a real gripper, closing stops at the object: when one is captured, the finger command corresponds to the
        object width (linear stroke model) instead of the fully-closed angle, so the rendered and collision-checked
        posture matches an actual grip.

        ``grasp=False`` (real-hardware mode) drives only the finger posture for the dashboard twin -- no proximity
        capture, no kinematic carry -- so the sim never raises grasp/drop events that would be mistaken for the real
        gripper's outcome.
        """
        self._gripper_closing = bool(close)
        log.debug("gripper commanded %s (capture %s)", "closed" if close else "open", "on" if grasp else "off")
        if close:
            if grasp:
                self._try_grasp()
            if grasp and self._grasped is not None:
                span = self._grasp_span if self._grasp_span is not None else self._default_span
                width = min(span, self.spec.gripper_stroke_m)
                # Width ─▶ joint is done by the linkage kinematics, not by this method.  A straight line between two
                # anchors (``closed * (1 - width/stroke)``) gave 0.43 rad for 50 mm where the geometry demands 0.32 rad
                # -- the jaws of the twin therefore stood somewhere other than those of the model move_group plans
                # against.
                self._gripper_command = self._linkage.angle_from_width(width)
            else:
                self._gripper_command = self._gripper_closed
        else:
            # Open only as far as the HELD object demands -- the width BEFORE the release, because ``_release``
            # forgets the span.  If the hand holds nothing there is no measure against which "as far as necessary"
            # could be measured: then it opens fully (that is also the grasp preparation case, where the hand has to go
            # AROUND the object).
            span = self._grasp_span if self._grasped is not None else None
            if span is None:
                self._gripper_command = self._gripper_open
            else:
                self._gripper_command = max(
                    self._linkage.angle_from_width(span + 2.0 * self._release_clearance), self._gripper_open
                )
            if grasp:
                self._release()

    def gripper_closed(self) -> bool:
        """Is the hand commanded CLOSED?

        The last command given, not a geometric threshold.  A comparison against half the closed position would be
        skewed -- a hand holding a 10 cm object stands far below it and would count as open -- and falls apart entirely
        as soon as the hand only opens to object width plus clearance for the release: with the real linkage every span
        below 40 mm then lies ABOVE the threshold again, so a hand that had just released reported itself as closed.
        """
        return self._gripper_closing

    @property
    def gripper_command_rad(self) -> float:
        """The driver joint value the sim currently holds [rad].

        This is the quantity that is actually written into the model -- and
        therefore the one on which it can be checked whether the twin has the
        hand standing where the linkage kinematics demands.  If
        :meth:`command_gripper` computed it from a straight line of its own, it
        landed on 0.43 instead of 0.32 rad at a 50 mm grasp width; invisible
        from the outside, because :meth:`gripper_width_m` walked the same line
        backwards and covered the error up.
        """
        return float(self._gripper_command)

    def gripper_width_m(self) -> float:
        """Commanded finger opening in METRES (0 = shut, stroke = wide).

        The inverse of the linkage :meth:`command_gripper` drives the joints with, and the one number a caller needs to
        make a REAL gripper hold the same posture as the twin: closing on an object stops at that object's width, so
        this follows the grasped object's span rather than a fixed open/shut pair.

        Why it is public.  Without it the twin's aperture lives only in ``_gripper_command``, and anyone mirroring the
        twin to real hardware can only send binary open/shut.  Measured in the husky-offboard container 2026-08-16:
        through a whole cell run the RG6 joints stood at 0.0 -- wide open -- while the twin had closed on a 10 cm block.
        move_group was therefore collision-checking a splayed hand that did not exist, on every rung and for every
        object, and the aperture could not follow the object at all.
        """
        return max(0.0, self._linkage.width_from_angle(self._gripper_command))

    # ------------------------------------------------------------------ #
    # grasping (kinematic carry)
    # ------------------------------------------------------------------ #
    def register_graspable(self, label: str, joint: str, body_id: int, half_extents) -> None:
        """Register one grabbable free body under ``label``.

        Called from :meth:`register_graspables`.  Each entry carries the free-joint addresses, the body id, the half
        extents (grasp-span checks) and the body's collision geoms (contact suspension while carried).  Unknown
        joints/bodies are skipped silently -- a scene may legitimately omit an optional object.
        """
        mujoco = self._mujoco
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0 or body_id < 0:
            return
        geoms = [
            g
            for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == body_id
            and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])
        ]
        self._graspable[label] = {
            "qpos": int(self.model.jnt_qposadr[jid]),
            "dof": int(self.model.jnt_dofadr[jid]),
            "body": body_id,
            "half": np.asarray(half_extents, dtype=float),
            "geoms": geoms,
        }

    def _horizontal_axes(self, entry) -> list:
        """Yaw angles of the horizontal body axes -- at most two.

        Forming grasp axes from ``obj_yaw`` and ``obj_yaw + 90 degrees`` presupposes that the z axis of the body stands
        VERTICAL: only then are x and y the horizontal ones.  For the LYING marker its length lies horizontal, and both
        candidates point past the graspable direction -- the capture fell back on the bounding box and grasped from
        12-14 mm away.

        What is taken are the body axes that REALLY lie horizontal (tilt below 30 degrees), and for each of them the
        perpendicular in the plane.  For a body standing upright those are exactly x and y.
        """
        R = self.data.xmat[entry["body"]].reshape(3, 3)
        angle = []
        for k in range(3):
            axis = R[:, k]
            if abs(float(axis[2])) > np.cos(np.radians(60.0)):
                continue  # stands too steeply
            angle.append(float(np.arctan2(axis[1], axis[0])))
        if not angle:
            return [float(np.arctan2(R[1, 0], R[0, 0]))]
        # Two axes suffice: the third is the perpendicular of the first.
        chosen = [angle[0]]
        for w in angle[1:]:
            if all(abs(_wrap_half(w - g - np.pi / 2.0)) > 1e-3 and abs(_wrap_half(w - g)) > 1e-3 for g in chosen):
                chosen.append(w)
        if len(chosen) == 1:
            chosen.append(chosen[0] + np.pi / 2.0)
        return chosen[:2]

    def _pad_heights(self) -> list[float]:
        """World z of the hand geoms at the CLOSED jaw position.

        Computed on a scratch copy so that the running scene stays untouched
        -- the same procedure as :meth:`arm_config_collides`.

        ``_gripper_closed`` is taken and not the width the object will demand later: that is only known once the capture
        has been decided, and the capture hangs on this height.  The closed position is a prediction without a circle.
        """
        mujoco = self._mujoco
        if not self._hand_geoms:
            return []
        if self._pad_scratch is None:
            self._pad_scratch = mujoco.MjData(self.model)
        scratch = self._pad_scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0.0
        for joint, factor in self._gripper_follower_factors.items():
            adr = self._joint_qpos.get(joint)
            if adr is not None:
                scratch.qpos[adr] = self._gripper_closed * float(factor)
        mujoco.mj_forward(self.model, scratch)
        return sorted(float(scratch.geom_xpos[g][2]) for g in self._hand_geoms)

    def _grip_reference(self, entry) -> np.ndarray:
        """Where the jaws grip the body -- from the WORLD, not the bounding box.

        Objects are grasped at their TOP SIDE (the grasp skills lower onto ``upper edge - span/2``).  The upper edge has
        to come from the actual attitude: ``entry["half"]`` is the AABB IN THE BODY FRAME and, for a lying pen, carries
        its LENGTH as its height.

        Measured: ``marker/pick`` reported success with a distance of +12 to +14 mm between jaws and pen.  The reference
        point lay about 57 mm above the lying body, :meth:`_span_between_pads` found no geom there and fell back on the
        bounding box (26 mm instead of an 18 mm shaft) -- the jaws closed to 26 mm and touched nothing.

        For a body standing upright the two coincide.
        """
        pos = self.data.xpos[entry["body"]].copy()
        ref = pos.copy()
        # Where the jaws REALLY are.  A constructed point ("upper edge minus
        # half the span") was the source of three errors in a row: it assumes
        # the task drives exactly there.  It does not -- for a flat object its
        # lower bound (table + span/2) lifts the jaws, and the capture then
        # measured 71.6 mm against its 70 mm radius and refused by 1.6 mm.  The
        # JAWS are the lowest parts of the hand; an average over the whole
        # assembly lies about 9 cm higher on the RG6, that is, in the wrist.
        #
        # The measurement is taken at the CLOSED jaw position, not at the
        # current one: the RG6 finger swivels, and ``_try_grasp`` runs while
        # the hand still stands open.  Open, its geom sits about 41 mm ABOVE
        # the TCP, closed 38 mm below it -- the reference thereby wandered by
        # 57 mm, and the jaw band lay where the pads ARE RIGHT NOW instead of
        # where they will grip (measured on the lid: band 43 mm off, capture
        # fell back on the bounding box and reported "no closable face pair"
        # at a knob of 36 mm).
        heights = self._pad_heights()
        if heights:
            bottom = heights[: max(1, len(heights) // 3)]
            ref[2] = float(np.mean(bottom))
            return ref
        top = -np.inf
        for gid in entry["geoms"]:
            center = self.model.geom_aabb[gid][:3]
            half = self.model.geom_aabb[gid][3:]
            R = self.data.geom_xmat[gid].reshape(3, 3)
            basis = self.data.geom_xpos[gid] + R @ center
            top = max(top, float(basis[2] + (np.abs(R) @ half)[2]))
        if not np.isfinite(top):
            top = float(pos[2]) + float(entry["half"][2])
        ref[2] = top - self._default_span / 2.0
        return ref

    def _pad_half_width(self) -> float:
        """Half width of a pad ACROSS the closing direction (m).

        Measured on the compiled model instead of maintained by hand: 6.8 mm for the RG6 (``flex_finger``).  It is the
        tolerance with which the TCP axis still hits a geom.

        Measured over ``_pad_geoms``, NOT over the whole hand: housing (41.8 mm), bracket (76.3 mm) and lever (39.2 mm)
        lie there too, and the median lands on a lever -- 12.9 mm would be an artefact of the geom mixture, not a pad
        width.
        """
        if self._pad_width is not None:
            return self._pad_width
        _pos, mat = self.tcp_pose()
        across = mat[:, 0]
        widths = []
        for gid in self._pad_geoms or self._hand_geoms:
            half = self.model.geom_aabb[gid][3:]
            R = self.data.geom_xmat[gid].reshape(3, 3)
            widths.append(float(np.abs(R.T @ across) @ half))
        self._pad_width = float(np.median(widths)) if widths else 0.0
        return self._pad_width

    def _span_between_pads(self, entry, ref, obj_yaw, tcp_xy=None):
        """Width of the body at jaw height, along its two horizontal axes --
        or ``None`` if nothing stands there.

        The band of height ``default_span`` lies around the GRASP POINT (``ref``), not around the TCP: at the grasp that
        one stands about 45 mm above it, and a band around it would lie completely above the object.  Only what reaches
        into this band is counted -- exactly that has to fit between the pads.  A body that grows wide further down (the
        disc of a lid) is none of the capture condition's business; whether it obstructs LATER is decided by the
        collision.

        ``None`` means "no geom of this body stands at jaw height"; then the bounding box remains, instead of allowing a
        grasp out of nothing.
        """
        band = float(self._default_span) / 2.0
        lo_z, hi_z = float(ref[2]) - band, float(ref[2]) + band
        c, s = float(np.cos(obj_yaw)), float(np.sin(obj_yaw))
        axes = (np.array([c, s, 0.0]), np.array([-s, c, 0.0]))
        limits = [[np.inf, -np.inf], [np.inf, -np.inf]]
        found = False
        for gid in range(self.model.ngeom):
            if int(self.model.geom_bodyid[gid]) != int(entry["body"]):
                continue
            center_local = self.model.geom_aabb[gid][:3]
            half = self.model.geom_aabb[gid][3:]
            R = self.data.geom_xmat[gid].reshape(3, 3)
            basis = self.data.geom_xpos[gid] + R @ center_local
            reach = np.abs(R) @ half
            if basis[2] + reach[2] < lo_z or basis[2] - reach[2] > hi_z:
                continue  # does not lie at jaw height
            if tcp_xy is not None:
                # ...and it has to lie HORIZONTALLY between the jaws.  Without this check every geom of the body that
                # happens to stand at jaw height counted -- including one 50 mm to the side that the pads do not reach
                # at all.  That is exactly how the lid with the off-centre knob still succeeded on the coarsest rung
                # (measured 2026-08-17): the jaws stood above the bare disc and the capture measured the knob.
                tol = self._pad_half_width()
                beside = False
                for axis in axes:
                    d = abs(float((basis - np.array([tcp_xy[0], tcp_xy[1], basis[2]])) @ axis))
                    if d > float(np.abs(R.T @ axis) @ half) + tol:
                        beside = True
                        break
                if beside:
                    continue
            found = True
            for i, axis in enumerate(axes):
                center = float(basis @ axis)
                far = float(np.abs(R.T @ axis) @ half)
                limits[i][0] = min(limits[i][0], center - far)
                limits[i][1] = max(limits[i][1], center + far)
        if not found:
            return None
        return (limits[0][1] - limits[0][0], limits[1][1] - limits[1][0])

    def _try_grasp(self) -> None:
        """Proximity capture over every graspable free body.

        Parallel-jaw constraints: the pads must close across a pair of opposing faces -- alignment of the pad axis with
        one of the object's horizontal axes (modulo 180 deg per axis) AND that face pair's span within the stroke.  For
        a square object this reduces to the classic modulo-90 check; an elongated box is only captured across its short
        side.
        """
        if self._grasped is not None:
            return
        tcp_pos, tcp_mat = self.tcp_pose()
        gripper_yaw = self.gripper_pad_yaw()
        tol = np.radians(GRASP_MAX_MISALIGN_DEG)
        best = None  # (label, dist, signed misalign, span)
        for label, entry in self._graspable.items():
            # Grip-point reference: objects are gripped by their TOP slice (grasp skills descend to top - span/2), so
            # tall boxes must be measured there, not at the body centre.  For a default-sized object the two coincide
            # (half height == span/2) -- classic behaviour kept.
            ref = self._grip_reference(entry)
            dist = float(np.linalg.norm(ref - tcp_pos))
            if dist >= (best[1] if best is not None else GRASP_RADIUS):
                continue
            half = entry["half"]
            # The span comes from the geometry BETWEEN the jaws, not from the bounding box of the whole body.  For a
            # cubic body that is the same thing; for everything else it was the coarsest conceivable abstraction.
            # Measured: a lid with a 180 mm disc and a 30 mm knob reported "no closable face pair" on ALL
            # four object rungs, because 180 mm stood against a 156 mm jaw travel -- independent of how finely the
            # object was modelled.  So the object axis could not bind over the grasp at all.
            axes = self._horizontal_axes(entry)
            local = self._span_between_pads(entry, ref, axes[0], tcp_xy=tcp_pos[:2])
            spans = local if local is not None else (2.0 * float(half[0]), 2.0 * float(half[1]))
            candidates = []
            for axis_yaw, span in ((axes[0], spans[0]), (axes[1], spans[1])):
                if span >= self.spec.gripper_stroke_m:
                    continue  # this face pair does not fit between the pads
                mis = _wrap_half(axis_yaw - gripper_yaw)
                if abs(mis) <= tol:
                    candidates.append((abs(mis), mis, span))
            if not candidates:
                log.debug("%s in reach but no closable face pair", label)
                continue
            _abs_mis, mis, span = min(candidates)
            best = (label, dist, mis, span)
        if best is None:
            # The commonest report is "closing the gripper did nothing", and this is the branch behind it: nothing
            # was within GRASP_RADIUS at all.  The two branches above already explain themselves; this one did not.
            log.debug(
                "no graspable within %.0f mm of the TCP (%d candidate(s) known)",
                GRASP_RADIUS * 1e3,
                len(self._graspable),
            )
            return
        label, best_dist, best_misalign, span = best
        adr = self._graspable[label]["qpos"]
        # Pad squaring: while closing, the flat pads rotate a slightly misaligned object until its faces sit flush --
        # snap the yaw onto the pad orientation before recording the carry offset.
        if abs(best_misalign) > 1e-6:
            snap = quat_about_z_wxyz(-best_misalign)
            self.data.qpos[adr + 3 : adr + 7] = quat_mul_wxyz(snap, self.data.qpos[adr + 3 : adr + 7].copy())
            mujoco = self._mujoco
            mujoco.mj_forward(self.model, self.data)
        if not self._square_tilt(adr, self._graspable[label]):
            log.debug("%s tilted too far -- the pads cannot align it", label)
            return
        obj_pos = self.data.qpos[adr : adr + 3].copy()
        obj_quat = self.data.qpos[adr + 3 : adr + 7].copy()  # wxyz
        # Offset of the object in the TCP frame, reapplied while carrying.
        rel_pos = tcp_mat.T @ (obj_pos - tcp_pos)
        tcp_quat = mat_to_quat_wxyz(tcp_mat)
        rel_quat = quat_mul_wxyz(quat_conj_wxyz(tcp_quat), obj_quat)
        self._grasped = label
        self._grasp_offset = (rel_pos, rel_quat)
        self._grasp_span = span
        self._grasp_misalign = float(best_misalign)
        self._carry_gap_min = None  # a new journey has its own worst moment
        self._carry_airborne = False
        # Record the gap NOW -- at the moment of gripping.  A later query measures the CARRY state: there the object is
        # kinematically welded on and the hand assembly inevitably overlaps it (measured 2026-08-17 on the marker: -13.3
        # mm, which looked like a penetration at the grasp and was none).
        self._suspend_object_contacts(label)
        self._event_acc.grasp_acquired = label
        log.debug(
            "grasped %s (dist %.3f m, squared %.0f deg, span %.0f mm)",
            label,
            best_dist,
            np.degrees(best_misalign),
            span * 1e3,
        )

    def _symmetry_axis(self, entry) -> np.ndarray | None:
        """Axis in the BODY FRAME about which the body is rotationally
        symmetric -- or ``None`` if it is about none.

        A body of revolution has no ATTITUDE about its axis: a lying pen rolled by 39 degrees about itself cannot be
        told apart from an unrolled one.  Reading the discrete body axes is right for a box and here invents a
        misalignment that does not exist.

        Strict: EVERY geom has to play along.  The handle of a cup is a capsule across the cup axis -- that makes the
        cup not symmetric, and the function says so (``None``).
        """
        mujoco = self._mujoco
        axis = None
        for gid in entry.get("geoms", ()):  # type: ignore[union-attr]
            typ = int(self.model.geom_type[gid])
            if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                continue  # symmetric about every axis
            if typ not in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                return None
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, self.model.geom_quat[gid])
            own = R.reshape(3, 3)[:, 2]  # the local z is the axis
            if axis is None:
                axis = own
            elif abs(float(axis @ own)) < 0.999:
                return None  # two axes, no symmetry
        return axis

    def _square_tilt(self, adr: int, entry=None) -> bool:
        """Pad squaring for the TILT -- the counterpart to the yaw angle.

        Flat jaws closing around a body straighten it out: the surfaces they touch lay themselves flat against the pads.
        For the YAW ANGLE that is done by the snap above; if the tilt stayed as it is, the object would be carried
        CROOKED.

        Measured: a pen leans 6 degrees in its holder, is carried 6 degrees crooked and afterwards fits into no cup any
        more -- move_group reports ``RRTConnect: Unable to sample any valid states for goal tree``.  The geometry was
        right, only the mechanism incomplete.

        The snap goes to the NEAREST axis-parallel attitude, not stubbornly to the vertical: otherwise a lying body
        would be stood upright at the grasp -- a motion that does not exist.  The rotation is thereby always the
        smallest possible one, and zero for an axis-parallel cubic body.
        """
        q = self.data.qpos[adr + 3 : adr + 7].copy()
        R = np.zeros(9)
        self._mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        # Remove ONLY the tilt, keep the yaw angle.  Snapping to the nearest
        # WORLD-axis-parallel attitude broke the pad alignment the yaw snap had
        # just established -- for the blocks of the stack a rotation of over
        # 15 degrees, which is exactly the free large correction this latch is
        # built against.
        #
        # What is sought is the smallest rotation that brings the body-own axis
        # closest to the vertical ONTO the vertical.  A lying body stays lying:
        # there the nearest axis is a different one, and the rotation is small
        # again.
        sym = self._symmetry_axis(entry) if entry is not None else None
        if sym is not None:
            # Only the attitude of THIS axis counts.  Ideally it is vertical (standing) or horizontal (lying) -- which
            # of the two is decided by the smaller rotation.  The roll about it is not an attitude and is not touched.
            axis = R @ sym
            upright_cos = float(np.clip(abs(axis[2]), 0.0, 1.0))
            tilt_from_z, tilt_from_plane = float(np.arccos(upright_cos)), float(np.arcsin(upright_cos))
            if min(tilt_from_z, tilt_from_plane) > np.radians(PAD_SQUARE_LIMIT_DEG):
                return False
            if tilt_from_z <= tilt_from_plane:
                goal = np.array([0.0, 0.0, 1.0 if axis[2] >= 0 else -1.0])
            else:
                flat = np.array([axis[0], axis[1], 0.0])
                norm = float(np.linalg.norm(flat))
                if norm < 1e-9:
                    return True  # degenerate: nothing to align
                goal = flat / norm
            angle = min(tilt_from_z, tilt_from_plane)
            return self._tilt_onto(adr, q, axis, goal, angle)

        k = int(np.argmax(np.abs(R[2, :])))
        axis = R[:, k] * (1.0 if R[2, k] >= 0 else -1.0)
        goal = np.array([0.0, 0.0, 1.0])
        angle = float(np.arccos(np.clip(float(axis @ goal), -1.0, 1.0)))
        if np.degrees(angle) > PAD_SQUARE_LIMIT_DEG:
            return False
        return self._tilt_onto(adr, q, axis, goal, angle)

    def _tilt_onto(self, adr: int, q, axis, goal, angle: float) -> bool:
        """Tip ``axis`` onto ``goal`` -- the smallest possible rotation."""
        if angle < 1e-9:
            return True
        rotate = np.cross(axis, goal)
        norm = float(np.linalg.norm(rotate))
        if norm < 1e-9:
            return True  # parallel or antiparallel
        rotate = rotate / norm
        correction = np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * rotate)])
        self.data.qpos[adr + 3 : adr + 7] = quat_mul_wxyz(correction, q)
        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _fingers_settled(self, tol: float = 0.02) -> bool:
        """Have the fingers reached their commanded width?

        Only THEN may the grasp gap be measured.  At the moment of capture the jaws still stand open (measured +80 mm),
        after the lift the carried body overlaps the hand assembly (-13 mm) -- both numbers looked like statements about
        the grasp quality and were none.
        """
        for joint, factor in self._gripper_follower_factors.items():
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint)
            if jid < 0:
                continue
            actual = float(self.data.qpos[self.model.jnt_qposadr[jid]])
            target = float(self._gripper_command) * float(factor)
            if abs(actual - target) > tol:
                return False
        return True

    def grasp_misalign_deg(self):
        """How crooked the object stood while being gripped (degrees), or ``None``.

        The pads straighten a slightly crooked object out while closing.  Merely logging that rotation and discarding it
        would, for a sufficiency measurement, mean losing THE error a coarse model produces: the robot aims according to
        its box, the real body stands differently, and the generosity of the capture irons it out.  move_group cannot
        report that -- jaw contact is explicitly allowed during the grasp, otherwise every grasp would be a start state
        in collision.
        """
        if self._grasped is None or self._grasp_misalign is None:
            return None
        return float(np.degrees(self._grasp_misalign))

    def grasp_gap(self):
        """Smallest distance gripper◀─▶grasped body (m), or ``None``.

        The CHECK to go with the model: the capture decides on distance, orientation and span and then welds -- whether
        the jaws really touch the body would otherwise stand nowhere.  The measurement is taken as soon as the fingers
        have reached their width (:meth:`_fingers_settled`): at the moment of capture they still stand open (+80 mm),
        after the lift the carried body overlaps the hand assembly (-13 mm).

        Reading: ``< 0`` penetration, ``~ 0`` contact, clearly ``> 0`` the jaws
        grab into thin air.  ``None`` means "no grasp, no statement" -- not
        "fine".
        """
        return self._grasp_gap0

    def carried_world_gap(self):
        """Smallest distance of the CARRIED body to the real world (m).

        The reality check for the TRANSPORT: the carried body has no contacts
        (for good reason, see :meth:`_suspend_object_contacts`), move_group
        checks the BELIEVED body, and the REAL one travels unhindered through
        the real world.  The direction is the dangerous one -- that is not
        reported as "reality forbids what belief allows", it is counted as a
        SUCCESS.

        The count runs against everything that belongs neither to the robot nor
        to the carried body: table, furniture, obstacles.  The floor stays out
        -- an object that has been set down touches it as intended.

        ``< 0`` means: the real body is stuck in real stuff.  ``None`` means "nothing carried, no statement".
        ``mj_geomDistance`` is a pure distance query and does not need the switched-off contacts.
        """
        if self._grasped is None:
            return None
        entry = self._graspable.get(self._grasped)
        if entry is None:
            return None
        own_geoms = set(entry["geoms"])
        smallest = float("inf")
        next = ""
        for gid in range(self.model.ngeom):
            if gid in own_geoms or gid in self._hand_geoms:
                continue
            if int(self.model.geom_bodyid[gid]) in self._robot_bodies():
                continue
            name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if "ground" in name or int(self.model.geom_type[gid]) == int(self._mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            # The SUPPORT SURFACE stays out.  When picking up, the object still lies on it, and when placing down it
            # settles onto it again -- both as intended.  Without this line the minimum of every cell stood at 0 and the
            # number said nothing (measured 2026-08-17).
            if name in self.support_geom_names():
                continue
            for own in own_geoms:
                d = float(self._mujoco.mj_geomDistance(self.model, self.data, int(gid), int(own), 1.0, None))
                if d < smallest:
                    smallest, next = d, name
        self._carry_gap_who = next
        return None if not np.isfinite(smallest) else smallest

    def carried_world_gap_min(self):
        """Worst distance of the carried body to the real world (m).

        :meth:`carried_world_gap` measures NOW; this number records the worst
        moment of the whole journey.  A distance at the goal says nothing about
        the way there: the body can have driven straight through an obstacle
        and be free again at the goal -- and nobody else notices that, because
        its contacts are switched off.
        """
        return self._carry_gap_min

    def carried_world_gap_who(self) -> str:
        """Name of the geom that came closest to the carried body."""
        return self._carry_gap_min_who

    def _robot_bodies(self) -> set:
        """Body ids of the robot -- it is not what is checked against here."""
        if getattr(self, "_robot_body_cache", None) is None:
            ids = set()
            for bid in range(self.model.nbody):
                bname = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                if bname.startswith(self.spec.manipulator_prefixes):
                    ids.add(bid)
            self._robot_body_cache = ids
        return self._robot_body_cache

    def _measure_gap(self, entry) -> float | None:
        """Smallest distance hand◀─▶body RIGHT NOW (m), or ``None``."""
        if entry is None or not self._hand_geoms:
            return None
        smallest = float("inf")
        for hand in self._hand_geoms:
            for gid in entry["geoms"]:
                d = float(self._mujoco.mj_geomDistance(self.model, self.data, int(hand), int(gid), 1.0, None))
                smallest = min(smallest, d)
        return None if not np.isfinite(smallest) else smallest

    def _release(self) -> None:
        if self._grasped is None:
            return
        label = self._grasped
        self._grasped = None
        self._grasp_offset = None
        self._grasp_span = None
        self._grasp_misalign = None
        self._grasp_gap0 = None
        # ``_carry_gap_min`` is NOT cleared here: it describes the journey that has just ended, and it is picked up
        # after the placing down.  It is cleared at the next grasp.
        self._carry_tick = 0
        # Hand the object back to physics at rest: restore its contacts and clear any residual solver velocity
        # accumulated while pinned.
        self._restore_object_contacts(label)
        dof = self._free_joint_dof(label)
        self.data.qvel[dof : dof + 6] = 0.0
        self._event_acc.grasp_lost = label
        log.debug("released %s", label)

    def _suspend_object_contacts(self, label: str) -> None:
        """Turn off all contacts of a carried object.

        While carried the object is kinematically pinned to the TCP -- it is effectively part of the end effector.
        Leaving its contacts on lets the contact solver fight the pin whenever an IK solution sweeps a robot link (e.g.
        the upper arm) through the carry zone; the growing penetration then discharges as a catapult impulse on release.
        """
        saved: dict[int, tuple[int, int]] = {}
        entry = self._graspable.get(label)
        for gid in entry["geoms"] if entry else []:
            saved[gid] = (int(self.model.geom_contype[gid]), int(self.model.geom_conaffinity[gid]))
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0
        self._object_contact_masks[label] = saved

    def _restore_object_contacts(self, label: str) -> None:
        saved = self._object_contact_masks.pop(label, {})
        for gid, (contype, conaffinity) in saved.items():
            self.model.geom_contype[gid] = contype
            self.model.geom_conaffinity[gid] = conaffinity

    def grasped_label(self) -> str | None:
        return self._grasped

    # ------------------------------------------------------------------ #
    # belief display (real mode: the twin SHOWS beliefs, it is not truth)
    # ------------------------------------------------------------------ #
    #: Where un-localized objects wait in real mode: behind the robot, outside
    #: every scene-camera frustum and clear of the collision-check parking
    #: (5+, 5+) and the obstacle-pool parking (3+, -3).
    _OBJECT_PARK = (-2.0, 2.0, 0.03)
    #: Grip-point offset (m) along the TCP approach axis used by the display
    #: carry -- the real pick descends to object centre + descend_offset, so the
    #: carried object's centre rides this far "below" the TCP.
    _CARRY_OFFSET = 0.065

    def display_object(self, label: str, position, yaw: float = 0.0) -> None:
        """Teleport an object body to a believed pose (display only, at rest).

        Real mode: the sim objects are not physics ground truth (the real ones are); they exist so the dashboard twin
        shows the scene.  This writes the belief into the free joint -- afterwards the object simply obeys physics again
        (settles onto the floor/stack).
        """
        adr = self._graspable[label]["qpos"]
        self.data.qpos[adr : adr + 3] = np.asarray(position, dtype=float)
        self.data.qpos[adr + 3 : adr + 7] = quat_about_z_wxyz(float(yaw))
        dof = self._free_joint_dof(label)
        self.data.qvel[dof : dof + 6] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def park_object(self, label: str, index: int = 0) -> None:
        """Move an un-localized object out of every camera view (display only)."""
        px, py, pz = self._OBJECT_PARK
        self.display_object(label, (px - 0.3 * index, py, pz))

    def display_carry(self, label: str) -> None:
        """Pin an object under the TCP for the dashboard twin -- event-free.

        The real-mode counterpart of the kinematic carry: the real gripper holds the real object, the sim only *shows*
        it.  Reuses the carry machinery (:meth:`_carry_grasped` follows the TCP every substep, contacts are suspended so
        the pin cannot fight the gripper shells) but never touches the event accumulator -- a display pin must not look
        like a grasp/drop to the skill layer.
        """
        if self._grasped == label:
            return
        if self._grasped is not None:
            self.display_release(self._grasped)
        tcp_pos, tcp_mat = self.tcp_pose()
        rel_pos = np.array([0.0, 0.0, self._CARRY_OFFSET])
        adr = self._graspable[label]["qpos"]
        self.data.qpos[adr : adr + 3] = tcp_pos + tcp_mat @ rel_pos
        self.data.qpos[adr + 3 : adr + 7] = mat_to_quat_wxyz(tcp_mat)
        dof = self._free_joint_dof(label)
        self.data.qvel[dof : dof + 6] = 0.0
        self._grasped = label
        self._grasp_offset = (rel_pos, np.array([1.0, 0.0, 0.0, 0.0]))
        self._suspend_object_contacts(label)
        self._mujoco.mj_forward(self.model, self.data)

    def display_release(self, label: str, position=None, yaw: float = 0.0) -> None:
        """End a display carry (event-free); optionally re-place the object."""
        if self._grasped == label:
            self._grasped = None
            self._grasp_offset = None
            self._restore_object_contacts(label)
            dof = self._free_joint_dof(label)
            self.data.qvel[dof : dof + 6] = 0.0
        if position is not None:
            self.display_object(label, position, yaw)

    def _carry_grasped(self) -> None:
        if self._grasped is None or self._grasp_offset is None:
            return
        tcp_pos, tcp_mat = self.tcp_pose()
        rel_pos, rel_quat = self._grasp_offset
        adr = self._graspable[self._grasped]["qpos"]
        self.data.qpos[adr : adr + 3] = tcp_pos + tcp_mat @ rel_pos
        tcp_quat = mat_to_quat_wxyz(tcp_mat)
        self.data.qpos[adr + 3 : adr + 7] = quat_mul_wxyz(tcp_quat, rel_quat)
        dof = self._free_joint_dof(self._grasped)
        self.data.qvel[dof : dof + 6] = 0.0

    def _free_joint_dof(self, label: str) -> int:
        return self._graspable[label]["dof"]

    # ------------------------------------------------------------------ #
    # stepping
    # ------------------------------------------------------------------ #
    def step_physics(self, n_ticks: int = 1) -> SimEvents:
        """Advance ``n_ticks`` control periods, holding all commanded state.

        Returns the events (collisions, grasp changes) accumulated over the stepped interval.
        """
        mujoco = self._mujoco
        events = SimEvents()
        events.merge(self._event_acc)
        self._event_acc = SimEvents()
        # Measure the grasp gap as soon as the FINGERS HAVE REACHED THEIR WIDTH -- not at the moment of capture (they
        # still stand open there, measured +80 mm) and not after the lift (there the carried 140 mm pen overlaps the
        # hand assembly, measured -13.3 mm).  Both moments looked like statements about the grasp quality and were
        # none.
        if self._grasped is not None and self._grasp_gap0 is None and self._gripper_closing and self._fingers_settled():
            self._grasp_gap0 = self._measure_gap(self._graspable.get(self._grasped))
        # Record the WORST moment of the journey: a distance at the goal says nothing about the way there, and it is
        # exactly there that a carried body travels unhindered through real stuff (its contacts are off).  The
        # measurement is taken per CALL: the motion layer calls ``step_physics(1)`` per waypoint, that is the natural
        # granularity.  (A modulo on ticks would be wrong here -- the block runs once per call, not per tick.)
        if self._grasped is not None:
            self._carry_tick += 1
            now = self.carried_world_gap()
            if now is not None:
                # Only count once the body has left the support surface.  Right after the grasp it still lies on the
                # table, and a distance of zero is NO statement about the journey there -- it would nail the minimum of
                # every cell to 0.
                if not self._carry_airborne:
                    if now > 0.005:
                        self._carry_airborne = True
                elif self._carry_gap_min is None or now < self._carry_gap_min:
                    self._carry_gap_min = now
                    self._carry_gap_min_who = self._carry_gap_who
        for _ in range(int(n_ticks)):
            # ONCE per tick, not per joint: the ramp is a state that keeps running, and the followers have to see the
            # same angle -- otherwise the two jaws stand at different widths.
            angle = self._gripper_tick_angle()
            gripper_targets = {joint: angle * factor for joint, factor in self._gripper_follower_factors.items()}
            # Where the fingers stand as this tick begins -- the actuated regime ramps its setpoint from here (see
            # _drive_gripper).
            start = (
                {j: float(self.data.qpos[self._joint_qpos[j]]) for j in gripper_targets if j in self._joint_qpos}
                if self._actuated_gripper
                else {}
            )
            for substep in range(self.n_substeps):
                for name, value in self._arm_command.items():
                    adr = self._joint_qpos.get(name)
                    if adr is not None:
                        self.data.qpos[adr] = value
                        self.data.qvel[self._joint_dof[name]] = 0.0
                if self._actuated_gripper:
                    self._drive_gripper(start, gripper_targets, (substep + 1) / self.n_substeps)
                else:
                    for name, value in gripper_targets.items():
                        adr = self._joint_qpos.get(name)
                        if adr is not None:
                            self.data.qpos[adr] = value
                            self.data.qvel[self._joint_dof[name]] = 0.0
                self._carry_grasped()
                mujoco.mj_step(self.model, self.data)
                self._scan_contacts(events)
        return events

    def gripper_angle_applied(self) -> float:
        """The angle that was REALLY written last.

        Not the same as :meth:`gripper_command_rad`: that is the COMMAND.  During a visible ramp the two run apart, and
        what one sees in the picture is this one.
        """
        return float(self._gripper_applied)

    def _gripper_tick_angle(self) -> float:
        """Return this tick's gripper angle and advance the ramp.

        Without a ramp (the default) that is simply the command -- byte-identical to the existing path.  With a ramp the
        angle travels over ``gripper_ramp_ticks`` ticks from the position at the command change to the target; the
        target itself never changes.
        """
        goal = float(self._gripper_command)
        if self._actuated_gripper or not self._gripper_ramp_ticks:
            self._gripper_applied = goal
            return goal
        if self._gripper_ramp_goal is None or goal != self._gripper_ramp_goal:
            # A new command: set off from where the fingers ARE.
            self._gripper_ramp_from = float(self._gripper_applied)
            self._gripper_ramp_goal = goal
            self._gripper_ramp_left = self._gripper_ramp_ticks
        if self._gripper_ramp_left <= 0:
            self._gripper_applied = goal
            return goal
        self._gripper_ramp_left -= 1
        rest = self._gripper_ramp_left / float(self._gripper_ramp_ticks)
        self._gripper_applied = goal + rest * (self._gripper_ramp_from - goal)
        return self._gripper_applied

    def _drive_gripper(self, start: dict[str, float], targets: dict[str, float], alpha: float) -> None:
        """Write ``data.ctrl`` for the follower joints; do NOT pin their qpos.

        Two halves, and the flag is worthless without either.  Writing ctrl is the obvious one -- ``ctrl`` left at zero
        means the position servos hold the OPEN angle, so a closing command is answered with a saturated counter-torque
        (measured: 20 N.m on every follower).  Leaving qpos alone is the other: a joint overwritten every substep cannot
        stop at the object, so no contact force can build.

        The setpoint is ramped across the substeps of a tick instead of jumping: the command changes binary
        open◀─▶closed, and a step of that size drives the fingers through the object before the contact solver ever sees
        them.
        """
        for joint, target in targets.items():
            aid = self._gripper_actuators.get(joint)
            if aid is None:
                continue
            begin = start.get(joint, target)
            self.data.ctrl[aid] = begin + alpha * (target - begin)

    def _scan_contacts(self, events: SimEvents) -> None:
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if float(con.dist) > -COLLISION_PENETRATION_TOL:
                continue  # graze within the envelope-model tolerance
            g1, g2 = int(con.geom1), int(con.geom2)
            pair = {g1, g2}
            if pair & self._robot_geoms:
                if pair & self._table_geoms:
                    events.robot_table_collision = True
                if pair & self._ground_geoms:
                    events.robot_ground_collision = True
                if pair & self._obstacle_geoms:
                    events.robot_obstacle_collision = True

    # ------------------------------------------------------------------ #
    # goal-state validity (mirror of MoveIt's start/goal collision check)
    # ------------------------------------------------------------------ #
    def arm_config_collides(
        self, joints: dict[str, float], *, penetration: float = 0.003, obstacles_only: bool = False
    ) -> bool:
        """True if the arm configuration is in *robot self-collision*.

        Mirrors move_group's start/goal state validation on a scratch ``MjData``.  Three pair classes invalidate a
        configuration:

        * manipulator vs. platform (from the MoveIt log: "contact between
          'base_link' and 'rg6_onrobot_rg6_base_link'"),
        * hand assembly (gripper + wrist camera) vs. shoulder/upper-arm/
          forearm -- a folded elbow that sweeps the arm through the gripper's
          carry zone, and
        * manipulator vs. perceived obstacles / distractors -- the pool mirrors
          what scene_sync publishes to move_group, so the client-side IK gate
          rejects goal states move_group would reject.

        Table/ground/graspable contacts are deliberately NOT part of this gate: the finger envelope honestly dips toward
        the table during grasps, and execution crashes remain covered by the contact-event scan.

        ``obstacles_only=True`` restricts the gate to the obstacle pairs -- the pose-goal pre-send probe uses it, since
        a hand-vs-obstacle hit at a given TCP pose is independent of the IK branch move_group picks.
        """
        mujoco = self._mujoco
        if self._collision_scratch is None:
            self._collision_scratch = mujoco.MjData(self.model)
        scratch = self._collision_scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0.0
        for name, value in joints.items():
            adr = self._joint_qpos.get(name)
            if adr is not None:
                scratch.qpos[adr] = float(value)
        # Park the sim's own payload out of the way: its contacts are not part of the validity question (and a carried
        # object travels with the TCP anyway).  Graspables that are ALSO obstacles stay put -- the gate exists to reject
        # configurations that reach into them.
        for index, label in enumerate(self._non_obstacle_graspables):
            adr = self._graspable[label]["qpos"]
            scratch.qpos[adr : adr + 3] = (5.0 + 0.5 * index, 5.0, 0.1)
        mujoco.mj_forward(self.model, scratch)
        for i in range(scratch.ncon):
            con = scratch.contact[i]
            if float(con.dist) > -penetration:
                continue
            pair = {int(con.geom1), int(con.geom2)}
            if pair & self._robot_geoms and pair & self._obstacle_geoms:
                return True
            if obstacles_only:
                continue
            if pair & self._robot_geoms and pair & self._platform_geoms:
                return True
            if pair & self._hand_geoms and pair & self._armfar_geoms:
                return True
        return False

    def settle(self, max_ticks: int = 100, vel_eps: float = 0.01) -> SimEvents:
        """Step until the sim's *free* payload is at rest (or ``max_ticks`` elapsed).

        Obstacle-classified graspables are scene furniture, not payload: they are excluded, so a jittering piece of
        clutter cannot stretch every settle (and with it every reset) to the tick limit.
        """
        events = SimEvents()
        for _ in range(max_ticks):
            events.merge(self.step_physics(1))
            speeds = []
            for label in self._non_obstacle_graspables:
                if label == self._grasped:
                    continue
                dof = self._free_joint_dof(label)
                speeds.append(float(np.linalg.norm(self.data.qvel[dof : dof + 3])))
            if not speeds or max(speeds) < vel_eps:
                break
        return events

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    def reset_robot(self) -> None:
        """Reset physics, home the arm and open the gripper.

        The generic head of a task reset: the subclass calls this and then distributes its own objects (which is the
        task-specific part).
        """
        mujoco = self._mujoco
        mujoco.mj_resetData(self.model, self.data)
        for label in list(self._object_contact_masks):
            self._restore_object_contacts(label)  # reset may interrupt a carry
        self._grasped = None
        self._grasp_offset = None
        self._grasp_span = None
        self._arm_command = dict(self._home_pose)
        self._gripper_command = self._gripper_open
        self._gripper_closing = False
        self._event_acc = SimEvents()

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def fk_body_pose(self, body: str, arm_joints: dict[str, float]) -> tuple[np.ndarray, np.ndarray] | None:
        """Forward-kinematics world pose of ``body`` for given arm joint angles.

        Uses a scratch ``MjData``, so the pose comes back in MuJoCo world
        coordinates -- and that origin is *ground-referenced*, not the URDF
        root: the converter raises the welded base until the wheels rest on
        z=0, which puts the world origin at ``base_footprint``, 0.132 m below
        the Husky's ``base_link``.  A caller that treats this as a
        ``base_link`` pose is off by exactly that much.  Used by the
        real-hardware camera to resolve the wrist RealSense pose from the
        *live* arm joint_states.
        """
        mujoco = self._mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            return None
        if self._collision_scratch is None:
            self._collision_scratch = mujoco.MjData(self.model)
        scratch = self._collision_scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0.0
        for name, value in arm_joints.items():
            adr = self._joint_qpos.get(name)
            if adr is not None:
                scratch.qpos[adr] = float(value)
        mujoco.mj_forward(self.model, scratch)
        return scratch.xpos[bid].copy(), scratch.xmat[bid].reshape(3, 3).copy()

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP world pose: ``(position (3,), rotation matrix (3,3))``."""
        pos = self.data.xpos[self._tcp_body_id].copy()
        mat = self.data.xmat[self._tcp_body_id].reshape(3, 3).copy()
        return pos, mat

    def object_position(self, label: str) -> np.ndarray:
        return self.data.xpos[self._graspable[label]["body"]].copy()

    def object_quat_wxyz(self, label: str) -> np.ndarray:
        adr = self._graspable[label]["qpos"]
        return self.data.qpos[adr + 3 : adr + 7].copy()

    def object_yaw(self, label: str) -> float:
        """In-plane rotation of the object's faces (rad, meaningful modulo 90 deg)."""
        mat = self.data.xmat[self._graspable[label]["body"]].reshape(3, 3)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))

    def gripper_pad_yaw(self) -> float:
        """World yaw of the finger-opening axis (the pads face +-TCP-y)."""
        _pos, mat = self.tcp_pose()
        pad_axis = mat[:, 1]
        # The pads close along the axis; the *face normal* orientation modulo 90 deg is what matters, so the
        # perpendicular works equally.
        return float(np.arctan2(pad_axis[1], pad_axis[0]))

    # ------------------------------------------------------------------ #
    # rendering / cameras
    # ------------------------------------------------------------------ #
    def _ensure_renderer(self, width: int, height: int):
        mujoco = self._mujoco
        if self._renderer is None or self._render_wh != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width, max_geom=10000)
            self._render_wh = (width, height)
            self._scene_option = mujoco.MjvOption()
            self._scene_option.geomgroup[self._HIDDEN_GEOM_GROUP] = 0
        return self._renderer

    def render_rgb(self, camera: str, width: int | None = None, height: int | None = None) -> np.ndarray:
        w = width or self._render_size[0]
        h = height or self._render_size[1]
        renderer = self._ensure_renderer(w, h)
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera=camera, scene_option=self._scene_option)
        return renderer.render()

    def render_depth(self, camera: str, width: int | None = None, height: int | None = None) -> np.ndarray:
        w = width or self._render_size[0]
        h = height or self._render_size[1]
        renderer = self._ensure_renderer(w, h)
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=camera, scene_option=self._scene_option)
        depth = renderer.render()
        renderer.disable_depth_rendering()
        return depth

    def camera_matrix(self, camera: str, width: int | None = None, height: int | None = None) -> np.ndarray:
        w = width or self._render_size[0]
        h = height or self._render_size[1]
        return camera_intrinsics(self.model, camera, w, h)

    def camera_pose(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        return camera_extrinsics(self.data, self.model, camera)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ #
    # Quaternion algebra: ``twinlink.quaternion`` -- MuJoCo's own ``mju_*``
    # operations.  Until 2026-08-23 it stood here by hand, line for line the
    # same as in two further modules.
    # ------------------------------------------------------------------ #
