"""Task-Simulation auf dem kompilierten Robotermodell (roboter-agnostisch).

Aus ``hrl.env.sim`` extrahiert (2026-07-31).  Das Ausführungsmodell ist das
von :class:`twinlink.sinks.mujoco_sink.MujocoSink` mit ``physics=True``:
*kommandierte Gelenke werden in qpos geschrieben und jeden Substep gehalten
(qvel=0), während alles andere der Physik gehorcht*.

Greifen folgt derselben kinematischen Philosophie: schließt der Greifer in
Reichweite eines registrierten Objekts, wird dieses kinematisch getragen (sein
Free-Joint folgt jeden Substep der TCP-Pose); Öffnen gibt es an die Physik
zurück.

Dieses Modul kennt weder Task noch Roboter: *welche* Objekte greifbar sind,
sagt die Subklasse; *wie* der Roboter heißt, sagt :class:`RobotSimSpec`.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from .events import SimEvents
from .mjcf_scene import (
    OBSTACLE_PARK,
    OBSTACLE_POOL_SIZE,
    camera_extrinsics,
    camera_intrinsics,
    obstacle_body_name,
)

log = logging.getLogger("twinlink.task_sim")


class GripperLinkage(Protocol):
    """Die Abbildung Greifweite <-> Treibergelenk -- HEREINGEREICHT, nie hier.

    Dieses Modul beschreibt nur, was es von der Abbildung braucht; die Formel
    selbst gehört dem Roboter, nicht der Sim.  ``husky_sdk.sim`` reicht dafür
    ``profile.gripper.linkage`` durch, wo sie aus der Getriebegeometrie folgt.

    Warum als Protokoll und nicht als eigene Klasse.  Bis 2026-08-16 bekam die
    Sim statt der Abbildung zwei Anker (``gripper_open`` 0.0, ``gripper_closed``
    0.6) und interpolierte selbst linear dazwischen -- die DRITTE Kopie
    derselben Rechnung, neben ``plan_bridge.plan_server`` und dem
    C++-Treiber ``rg6_control``.  Alle drei Anker-Paare waren falsch (die
    offene Hand stand für 93,7 mm statt 159 mm), und weil jede Kopie für sich
    gerundet rechnete, hätte das Korrigieren einer einzelnen die anderen erst
    recht auseinandergetrieben.  Eine vierte Kopie hier wäre derselbe Fehler
    noch einmal; ``twinlink`` hängt bewusst nicht an ``robot_contract``, also
    wird die Abbildung übergeben.
    """

    @property
    def open_rad(self) -> float:
        """Treibergelenk der weitest offenen Hand."""

    @property
    def closed_rad(self) -> float:
        """Treibergelenk der ganz geschlossenen Hand."""

    @property
    def max_width_m(self) -> float:
        """Grösste lichte Weite, die das Getriebe hergibt."""

    def width_from_angle(self, q: float) -> float:
        """Lichte Weite [m] beim Treibergelenk ``q`` [rad]."""

    def angle_from_width(self, width_m: float) -> float:
        """Treibergelenk [rad] für die lichte Weite ``width_m`` [m]."""


@dataclass(frozen=True)
class RobotSimSpec:
    """Die Roboter-Fakten, die die Task-Sim braucht.

    twinlink ist ein eigenständiges Paket und hängt bewusst NICHT an
    ``robot_contract`` -- diese Werte werden hereingereicht.  Für Roboter mit
    einem robot-contract-Profil erledigt das ``husky_sdk.sim.robot_sim_spec()``.
    """

    #: Körper-Präfixe des beweglichen Manipulators; alles andere ist Plattform.
    manipulator_prefixes: Tuple[str, ...]
    #: Präfixe der Hand-Baugruppe (Greifer + handgelenkmontierte Sensorik).
    hand_prefixes: Tuple[str, ...]
    #: Präfixe der Greiferschalen -- NUR die Backen/das Greifergehäuse, ohne
    #: mitfahrende Sensorik.  Nur diese Geoms werden für greifbare Objekte
    #: durchlässig gemacht (kinematisches Greifen, siehe
    #: ``TwinTaskSim._setup_collision_masks``); eine handgelenkmontierte Kamera
    #: gehört zur Hand, muss aber weiter an Objekten anschlagen, sonst verliert
    #: sie ihre Kollisionsereignisse.
    gripper_prefixes: Tuple[str, ...]
    #: Armferne Körper -- Kontakt Hand<->hier = gefaltete Konfiguration.
    far_arm_bodies: Tuple[str, ...]
    #: Backengang des Greifermodells (m) bei offener Hand.
    gripper_stroke_m: float
    #: Körper, dessen Pose als TCP gilt.
    tcp_body: str
    #: Gelenke der Planungsgruppe, in SRDF-Reihenfolge.
    arm_joints: Tuple[str, ...]


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

#: Wieviel NEIGUNG die Pads beim Schliessen noch ausrichten koennen --
#: darueber gibt es keinen Griff.
#:
#: Gilt bewusst NUR fuer die Neigung, nicht fuer den Gierwinkel.  Der
#: Gierpfad ist alt, durch Golden-Spuren gepinnt (der Wuerfelturm bricht
#: sofort, wenn man ihn enger zieht -- am 2026-08-17 mit 15 Grad probiert:
#: sieben Bestandstests rot) und in der Studie mit hoechstens 2,1 Grad
#: gemessen.  Die NEIGUNG dagegen wurde erst an diesem Tag eingefuehrt
#: und hatte GAR KEINE Grenze: sie schnappte auf die naechste
#: achsparallele Lage, wie schief der Koerper auch stand.
#:
#: Warum es diese Grenze gibt (Owner-Frage 2026-08-17): "das Anpassen der
#: Orientierung in moveit fuehrt zu einer Anpassung in mujoco, dieser Loop
#: ist fuer die Studie gefaehrlich".  Er ist es.  Glaube -> Handlung ->
#: Weltaenderung -> wird als "Wahrheit" zurueckgelesen -> Glaube.  Kausal
#: legitim (ein echter Greifer richtet einen Stift beim Zupacken wirklich
#: auf), aber der Zwilling modellierte es als KOSTENLOSEN Schnapp: alles
#: bis :data:`GRASP_MAX_MISALIGN_DEG` wurde umsonst korrigiert, ohne
#: Fehlerfall.  Damit verwandelt er Abstraktionsfehler in nichts -- und
#: eine grobe Sprosse sieht ausreichend aus, weil der Zwilling die Folge
#: ihrer Grobheit selbst repariert hat.
#:
#: Nachgiebigkeit ist echt, aber begrenzt: flache Pads richten einen
#: leicht schiefen Koerper aus, ein stark schiefer rutscht ab.
#:
#: DER WERT IST EINE MODELLENTSCHEIDUNG, kein Messergebnis, und
#: ausdruecklich ein Regler.  Die gemessene Verteilung ueber 68
#: Container-Laeufe reicht von 0,0 bis 2,1 Grad -- die Grenze bindet
#: heute NIRGENDS.  Sie ist eine Wache gegen einen Fehlermodus, keine
#: Korrektur eines Ergebnisses.
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

    The generic half of what used to be ``hrl.env.sim.StackCubesSim``: joint
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
        gripper_follower_factors: Dict[str, float],
        gripper_linkage: GripperLinkage,
        home_pose: Dict[str, float],
        render_size: Tuple[int, int] = (640, 480),
        actuated_gripper: bool = False,
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
        #: Die beiden Hindernis-Körperfamilien DIESER Szene, aus demselben
        #: Präfix abgeleitet, den der Konstruktor bekommen hat -- nicht aus den
        #: Modulkonstanten ``OBSTACLE_BODY_PREFIX`` / ``DISTRACTOR_BODY_PREFIX``
        #: (die sind auf ``hrl_`` festgenagelt und existieren nur noch für
        #: Altkonsumenten, die sie direkt importieren).  Bis 2026-08-01 war der
        #: Präfix nur halb verdrahtet: Render-Trennung folgte ihm, Klassifikation
        #: und Pool-Indizierung nicht -- eine zweite App wäre still blind für
        #: die gesamte Hindernisklasse gewesen (siehe
        #: ``tests/test_task_sim.py::test_scene_prefix_drives_classification``).
        #: Anders als bei der Render-Trennung ist ``scene_prefix=""`` hier
        #: harmlos: die abgeleiteten Werte (``"obstacle_"`` / ``"distractor_"``)
        #: bleiben nichtleer, ein ``startswith`` kann also nicht auf alles passen.
        self._obstacle_body_prefix = f"{scene_prefix}obstacle_"
        self._distractor_body_prefix = f"{scene_prefix}distractor_"
        #: Closing span (m) assumed when nothing is captured (task object size).
        self._default_span = float(default_span)
        #: Spiel je Seite (m), mit dem die Hand ein Objekt FREIGIBT.  Zum
        #: Loslassen ganz aufzureissen ist eine Wahl, keine Notwendigkeit,
        #: und sie kostet: am Container gemessen (2026-08-16) stand die
        #: volle Hand nach dem Ablegen im vergroeberten Tor, und move_group
        #: verweigerte den Rueckzug -- ``2 contact(s) detected : gate_0 -
        #: rg6_right_inner_finger, gate_1 - rg6_left_inner_finger``.  Ein
        #: echter RG6 oeffnet nur so weit, wie das Objekt es verlangt.
        self._release_clearance = float(release_clearance)
        self._gripper_follower_factors: Dict[str, float] = dict(gripper_follower_factors)
        #: Weite <-> Treibergelenk.  Siehe :class:`GripperLinkage`: die Formel
        #: gehört dem Roboter und wird hereingereicht, damit sie nicht zum
        #: vierten Mal im Stack steht.
        self._linkage = gripper_linkage
        self._gripper_open = float(gripper_linkage.open_rad)
        self._gripper_closed = float(gripper_linkage.closed_rad)
        self._home_pose: Dict[str, float] = dict(home_pose)
        self._render_size: Tuple[int, int] = (int(render_size[0]), int(render_size[1]))
        #: Opt-in: drive the follower joints through the model's actuators
        #: instead of pinning their qpos.  Off is the twin's normal regime --
        #: everything commanded is held, and the gripper shells are permeable
        #: to the objects (grasping is a kinematic capture).  On, the fingers
        #: close against the object and a grip force exists; see
        #: :meth:`_drive_gripper` and :meth:`_setup_collision_masks`.
        self._actuated_gripper = bool(actuated_gripper)
        #: follower joint -> actuator id, filled only in the actuated regime.
        self._gripper_actuators: Dict[str, int] = {}

        self._joint_qpos: Dict[str, int] = {}
        self._joint_dof: Dict[str, int] = {}
        self._index_joints()
        if self._actuated_gripper:
            self._index_gripper_actuators()
        # (contype, conaffinity) per object geom, to suspend/restore contacts
        # while the object is carried (see _suspend_object_contacts).
        self._object_contact_masks: Dict[str, Dict[int, Tuple[int, int]]] = {}
        # Scratch MjData for goal-state collision checks (lazily created).
        self._collision_scratch = None
        self._tcp_body_id = self._body_id(spec.tcp_body)
        self._graspable: Dict[str, Dict] = {}
        self.register_graspables()
        self._classify_geoms()
        self._setup_collision_masks()
        self._hide_robot_collision_geoms()
        self._index_obstacle_pool(n_obstacle_slots)

        # Commanded state (held every substep, twin-style).
        self._arm_command: Dict[str, float] = dict(self._home_pose)
        self._gripper_command: float = self._gripper_open
        #: Ob die Hand ZU kommandiert ist -- siehe :meth:`gripper_closed`.
        self._gripper_closing: bool = False
        # label -> (pos offset in TCP frame, quat offset) while carried.
        self._grasped: Optional[str] = None
        self._grasp_offset: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # Closing span (m) of the captured face pair -- drives the finger
        # command width; None falls back to the default span.
        self._grasp_span: Optional[float] = None
        #: Wie weit der Fang das Objekt beim Zupacken drehen musste (rad).
        #: Siehe :meth:`grasp_misalign_deg`.
        self._grasp_misalign: Optional[float] = None
        #: Spalt im Moment des Fangs (m).  Siehe :meth:`grasp_gap`.
        self._grasp_gap0: Optional[float] = None
        # Events raised outside step_physics (grasp/release on command) are
        # accumulated here and drained by the next step_physics call.
        self._event_acc = SimEvents()

        self._renderer = None
        self._render_wh: Optional[Tuple[int, int]] = None

        mujoco.mj_forward(self.model, self.data)
        log.info(
            "sim ready: %d joints indexed, control_dt=%.3fs (%d substeps)",
            len(self._joint_qpos), self.control_dt, self.n_substeps,
        )

    # ------------------------------------------------------------------ #
    # subclass hooks (the only way task knowledge enters)
    # ------------------------------------------------------------------ #
    def register_graspables(self) -> None:
        """Subklassen registrieren hier ihre greifbaren Objekte.

        Default: keine -- eine Sim ohne Greifobjekte ist zulässig (reine
        Bewegungs-/Kollisionsuntersuchung).
        """

    def support_geom_names(self) -> frozenset:
        """Geom-Namen der Auflagefläche (Tisch o. ä.) für die Kontaktklassen."""
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

        Found through ``actuator_trnid``, not through a name convention: the
        app authors the actuators and may call them whatever it likes.  A
        follower without an actuator is a hard error -- the whole point of the
        actuated regime is that a closing command produces a force, and
        falling back to the kinematic hold for the missing ones would produce
        the same green-but-forceless state the flag exists to end.
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
        self._object_geoms: Dict[int, str] = {}
        support = self.support_geom_names()
        self._object_bodies = {
            entry["body"]: label for label, entry in self._graspable.items()
        }
        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if self.model.geom_contype[gid] == 0 and self.model.geom_conaffinity[gid] == 0:
                continue  # visual-only geometry (meshes, markers)
            if gname in support:
                self._table_geoms.add(gid)
            elif gname == "twinlink_ground" or int(self.model.geom_type[gid]) == int(
                mujoco.mjtGeom.mjGEOM_PLANE
            ):
                self._ground_geoms.add(gid)
            elif bname.startswith(
                (self._obstacle_body_prefix, self._distractor_body_prefix)
            ):
                # Perceived-obstacle pool slots + authored distractors: the
                # things the arm must plan around (goal gate + contact events).
                # Checked BEFORE the graspable bodies: a distractor the task
                # promotes to a graspable stays an obstacle for the contact
                # classes (unchanged from the pre-split behaviour).
                self._obstacle_geoms.add(gid)
            elif bid in self._object_bodies:
                self._object_geoms[gid] = self._object_bodies[bid]
            elif bname.startswith(self.spec.manipulator_prefixes):
                # Only the manipulator counts for collision penalties; the
                # chassis standing on the ground is normal.  (Classify purely
                # by body name: for the welded robot, body_rootid points at
                # base_link, not the world -- a root==0 filter silently drops
                # every manipulator geom and no collision would ever fire.)
                self._robot_geoms.add(gid)
            elif bname:
                # Chassis / plates / bumpers / sensor arch: the static part of
                # the robot the manipulator must not touch.
                self._platform_geoms.add(gid)
        if not self._robot_geoms:
            raise RuntimeError(
                "no manipulator geoms classified -- collision events would be blind"
            )
        # Subsets for the goal-state self-collision gate: the hand assembly
        # (gripper + wrist camera) versus arm links far from the wrist.  A
        # contact between those means a fully folded, invalid configuration
        # (the class of state MoveIt rejects as robot self-collision).
        self._hand_geoms: set = set()
        self._armfar_geoms: set = set()
        armfar_bodies = set(self.spec.far_arm_bodies)
        for gid in self._robot_geoms:
            bid = int(self.model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.startswith(self.spec.hand_prefixes):
                self._hand_geoms.add(gid)
            elif bname in armfar_bodies:
                self._armfar_geoms.add(gid)
        # Registered graspables that are ALSO classified as obstacles (pool
        # slots or authored clutter the task promoted to a target) stay
        # obstacles for the validity and settling questions: the arm must plan
        # around them, so parking them away would make the goal gate blind to
        # exactly the objects it exists for.  Only the sim's own payload is
        # parked/settled -- the pre-split behaviour, expressed without task
        # knowledge.
        self._non_obstacle_graspables: Tuple[str, ...] = tuple(
            label for label, entry in self._graspable.items()
            if not (self._obstacle_geoms & set(entry["geoms"]))
        )

    def _setup_collision_masks(self) -> None:
        """Let the gripper envelope pass through graspables (grasping is kinematic).

        Grasping is modelled as proximity capture + kinematic carry (see module
        docstring), so finger--object contact forces are artifacts: the open
        fingers would shove an object away while descending onto it.  Contact
        bitmasks exclude exactly the gripper<->object pairs; the gripper still
        collides with table/ground (bad approach poses stay detectable) and the
        arm still collides with the objects (knocking them over stays possible).

        Masks: world/robot keep (contype=1, conaffinity=1); objects get (2, 3);
        gripper geoms get (4, 1).  gripper&object: 4&3 = 2&1 = 0 -> no contact.

        Every REGISTERED graspable gets this treatment, task objects and
        dynamic clutter alike: grasping is kinematic for both -- the open
        fingers would shove the box away while descending, and the goal gate
        would veto the very grasp.  The ARM still collides with them (knocking
        them over stays detectable).

        Only ``spec.gripper_prefixes`` -- the jaws/housing -- becomes
        permeable, NOT the whole ``hand_prefixes`` assembly: a wrist-mounted
        camera rides along with the hand but is not a jaw, and making it
        permeable would silently drop its obstacle-contact events.

        ``actuated_gripper=True`` inverts the premise and therefore skips the
        permeability: there the jaws are driven by servos and must MEET the
        object -- a permeable jaw yields exactly zero finger contacts and zero
        grip force, which is what the flag was measured doing before this.
        The objects keep their (2, 3) mask either way, so the arm still
        collides with them and the perceived-obstacle slots (4, 5) stay
        permeable to them.
        """
        mujoco = self._mujoco
        graspable_geoms = {
            g for entry in self._graspable.values() for g in entry["geoms"]
        }
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

    def _robot_geom_ids(self) -> List[int]:
        """Geom ids of the robot's own kinematic tree.

        Positively identified through ``body_rootid``: every body the URDF
        contributed shares the robot's root body (the welded base link), while
        each piece of scene furniture is its own world child and therefore its
        own root.  Measured on the a200-0553 scene 2026-08-16: 45 of 58 bodies
        under ``base_link``, every furniture body a root of its own.

        The rule used to be the inverse one -- "everything that does NOT carry
        the app's scene prefix is robot" -- and that made every body an app
        placed under a name of its own invisible to RGB *and* depth, silently:
        the cameras then see the surface behind it and a plausible-looking
        point cloud comes back without the object in it.
        """
        root = int(self.model.body_rootid[self._tcp_body_id])
        return [
            gid for gid in range(self.model.ngeom)
            if int(self.model.body_rootid[int(self.model.geom_bodyid[gid])]) == root
        ]

    def _hide_robot_collision_geoms(self) -> None:
        """Exclude the robot's collision geometry from rendering.

        The role of ``MujocoSink._hide_collision_geoms``: the robot carries
        both visual meshes and collision geoms (``with_collision=True``); the
        collision shells (e.g. a gripper's base box, much larger than the
        visible gripper) would otherwise occlude the cameras.  The sink's alpha=0 trick
        only affects the RGB pass, while MuJoCo's depth pass rasterises
        transparent geoms too -- so we move the shells into a geom *group* that
        all render calls disable via ``MjvOption``.  Restricted to the robot
        subtree (:meth:`_robot_geom_ids`) -- everything else in the scene has
        no separate visual geometry and must stay visible.  Groups are
        visualisation-only; physics is untouched.
        """
        robot = self._robot_geom_ids()
        robot_visuals = sum(
            1 for gid in robot
            if self.model.geom_contype[gid] == 0 and self.model.geom_conaffinity[gid] == 0
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

        Die Slot-Namen folgen dem Konstruktor-Präfix (``scene_prefix``), genau
        wie die Klassifikation -- mit dem Modul-Default gesucht, fände eine App
        mit eigenem Präfix ihren eigenen Pool nicht und liefe still ohne
        Hindernisse (bis 2026-08-01 der Fall).
        """
        mujoco = self._mujoco
        self._obstacle_slots: List[Tuple[int, int]] = []
        for i in range(int(n_slots)):
            name = obstacle_body_name(i, prefix=self._scene_prefix)
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom"
            )
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

        The mirrored boxes must never appear in the depth image the obstacle
        pipeline itself consumes (sim cameras): a stale box would otherwise
        occlude the very view that should prove its space empty -- the mirror
        would keep itself alive.  Visual-only (geom groups), physics untouched.
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

    def set_obstacles(self, boxes: List) -> int:
        """Mirror perceived obstacles into the collision pool.

        ``boxes`` are duck-typed (``.center`` (3,) world, ``.size`` (3,) full
        extents -- :class:`perception.obstacles.ObstacleBox`).  Slots
        beyond ``len(boxes)`` are parked; when more boxes than slots arrive
        the largest ones win (safety-relevant volume first).  The boxes enter
        physics immediately: MoveIt-side planning uses the planning-scene
        copy, this pool covers the twin (dashboard), the contact events and
        the client-side IK goal gate (:meth:`arm_config_collides`).

        A slot is a PERCEPTION of the world, never a body in it: it must not
        exert force on the objects it depicts (same reasoning as the
        permeable gripper shells in :meth:`_setup_collision_masks`).  In sim
        a perceived box lands exactly on the free-jointed body it was
        perceived from -- with ordinary contacts the solver resolves that
        overlap by ejecting the original out from under its own ghost, on the
        first step after the sync and before the arm has moved.

        Returns the number of active slots.
        """
        boxes = list(boxes)
        if len(boxes) > len(self._obstacle_slots):
            boxes.sort(key=lambda b: -float(np.prod(np.asarray(b.size, dtype=float))))
            log.warning(
                "%d obstacles exceed the %d-slot pool -- keeping the largest",
                len(boxes), len(self._obstacle_slots),
            )
            boxes = boxes[: len(self._obstacle_slots)]
        for i, (bid, gid) in enumerate(self._obstacle_slots):
            if i < len(boxes):
                box = boxes[i]
                half = np.maximum(np.asarray(box.size, dtype=float) / 2.0, 1e-3)
                yaw = float(getattr(box, "yaw", 0.0))
                self.model.body_pos[bid] = np.asarray(box.center, dtype=float)
                self.model.body_quat[bid] = (
                    np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
                )
                self._write_obstacle_geom(gid, half)
                # Masks (see _setup_collision_masks for the scheme): world and
                # robot keep (1, 1), graspables get (2, 3), gripper shells
                # (4, 1).  A slot gets (4, 5) -- permeable to graspables
                # (4&3 = 2&5 = 0), still felt by the arm (1&5 = 1) and by the
                # gripper shells (4&5 = 4), so the goal gate and the
                # robot_obstacle_collision events keep seeing exactly the
                # objects they exist for.
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
    def set_arm_command(self, positions: Dict[str, float]) -> None:
        """Set the held target for (a subset of) the arm joints."""
        for name, value in positions.items():
            if name in self._joint_qpos:
                self._arm_command[name] = float(value)

    def arm_command(self) -> Dict[str, float]:
        return dict(self._arm_command)

    def arm_positions(self) -> Dict[str, float]:
        """Current arm joint positions (== command, joints are held)."""
        return {j: float(self.data.qpos[self._joint_qpos[j]]) for j in self.spec.arm_joints}

    def command_gripper(self, close: bool, *, grasp: bool = True) -> None:
        """Binary open/close -- the gripper service semantics.

        Like a real gripper, closing stops at the object: when one is
        captured, the finger command corresponds to the object width (linear
        stroke model) instead of the fully-closed angle, so the rendered and
        collision-checked finger posture matches an actual grip.

        ``grasp=False`` (real-hardware mode) drives only the finger posture for
        the dashboard twin -- no proximity capture, no kinematic carry -- so the
        sim never raises grasp/drop events that would be mistaken for the real
        gripper's outcome (the real gripper is commanded separately and its
        feedback is what the skill trusts).
        """
        self._gripper_closing = bool(close)
        if close:
            if grasp:
                self._try_grasp()
            if grasp and self._grasped is not None:
                span = (self._grasp_span if self._grasp_span is not None
                        else self._default_span)
                width = min(span, self.spec.gripper_stroke_m)
                # Weite -> Gelenk macht die Getriebekinematik, nicht diese
                # Methode.  Vorher stand hier ``closed * (1 - width/stroke)``,
                # eine Gerade zwischen zwei Ankern: für 50 mm ergab sie 0,43 rad,
                # wo die Geometrie 0,32 rad verlangt -- die Backen des Zwillings
                # standen also woanders als die des Modells, gegen das
                # move_group plant.
                self._gripper_command = self._linkage.angle_from_width(width)
            else:
                self._gripper_command = self._gripper_closed
        else:
            # Nur so weit oeffnen, wie das GEHALTENE Objekt es verlangt --
            # die Weite VOR dem Loslassen, denn ``_release`` vergisst die
            # Spanne.  Haelt die Hand nichts, gibt es kein Mass, an dem
            # "so weit wie noetig" sich messen liesse: dann geht sie ganz
            # auf (das ist auch der Griff-Vorbereitungsfall, wo die Hand um
            # das Objekt HERUM muss).
            span = self._grasp_span if self._grasped is not None else None
            if span is None:
                self._gripper_command = self._gripper_open
            else:
                self._gripper_command = max(
                    self._linkage.angle_from_width(
                        span + 2.0 * self._release_clearance),
                    self._gripper_open,
                )
            if grasp:
                self._release()

    def gripper_closed(self) -> bool:
        """Ist die Hand ZU kommandiert?

        Der zuletzt gegebene Befehl, kein geometrischer Schwellwert.  Bis
        2026-08-16 stand hier ``command >= closed/2``, ein Vergleich mit der
        halben Schliessstellung.  Der war schon immer schief -- eine Hand,
        die ein 10-cm-Objekt haelt, steht weit unter der Schwelle und galt
        als offen -- und wurde falsch, sobald die Hand zum Loslassen nur noch
        auf Objektbreite plus Spiel oeffnet: mit dem echten Getriebe liegt
        jede Spanne unter 40 mm danach WIEDER ueber der Schwelle, eine
        gerade losgelassene Hand haette sich also als geschlossen gemeldet.
        """
        return self._gripper_closing

    @property
    def gripper_command_rad(self) -> float:
        """Der Treibergelenkwert, den die Sim gerade hält [rad].

        Das ist die Grösse, die tatsächlich ins Modell geschrieben wird -- und
        damit die, an der sich prüfen lässt, ob der Zwilling die Hand dort
        stehen hat, wo die Getriebekinematik sie verlangt.  Bis 2026-08-16
        rechnete :meth:`command_gripper` sie aus einer eigenen Geraden aus und
        landete bei 50 mm Griffweite auf 0,43 statt 0,32 rad; sichtbar war das
        von aussen nur, weil :meth:`gripper_width_m` dieselbe Gerade rückwärts
        ging und den Fehler damit zudeckte.
        """
        return float(self._gripper_command)

    def gripper_width_m(self) -> float:
        """Commanded finger opening in METRES (0 = shut, stroke = wide).

        The inverse of the linkage :meth:`command_gripper` drives the joints
        with, and the one number a caller needs to make
        a REAL gripper hold the same posture as the twin: closing on an
        object stops at that object's width, so this follows the grasped
        object's span rather than a fixed open/shut pair.

        Why it is public.  Without it the twin's aperture lives only in
        ``_gripper_command``, and anyone mirroring the twin to real
        hardware can only send binary open/shut.  Measured in the
        husky-offboard container 2026-08-16: through a whole cell run the
        RG6 joints stood at 0.0 -- wide open -- while the twin had closed
        on a 10 cm block.  move_group was therefore collision-checking a
        splayed hand that did not exist, on every rung and for every
        object, and the aperture could not follow the object at all.
        """
        return max(0.0, self._linkage.width_from_angle(self._gripper_command))

    # ------------------------------------------------------------------ #
    # grasping (kinematic carry)
    # ------------------------------------------------------------------ #
    def register_graspable(
        self, label: str, joint: str, body_id: int, half_extents
    ) -> None:
        """Register one grabbable free body under ``label``.

        Called from :meth:`register_graspables`.  Each entry carries the
        free-joint addresses, the body id, the half extents (grasp-span
        checks) and the body's collision geoms (contact suspension while
        carried).  Unknown joints/bodies are skipped silently -- a scene may
        legitimately omit an optional object.
        """
        mujoco = self._mujoco
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0 or body_id < 0:
            return
        geoms = [
            g for g in range(self.model.ngeom)
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
        """Gierwinkel der waagerechten Koerperachsen -- hoechstens zwei.

        ``_try_grasp`` bildete die Greifachsen aus ``obj_yaw`` und
        ``obj_yaw + 90 Grad``.  Das setzt voraus, dass die z-Achse des
        Koerpers SENKRECHT steht: nur dann sind x und y die waagerechten.
        Beim LIEGENDEN Marker liegt seine Laenge waagerecht, und beide
        Kandidaten zeigen an der greifbaren Richtung vorbei -- der Fang
        fiel auf die Huelle zurueck und griff aus 12-14 mm Entfernung
        (2026-08-17, vom Owner in Foxglove gesehen).

        Genommen werden die Koerperachsen, die WIRKLICH waagerecht liegen
        (Neigung unter 30 Grad), und dazu je die Senkrechte in der Ebene.
        Fuer einen aufrecht stehenden Koerper sind das genau x und y --
        die alte Rechnung, unveraendert.
        """
        R = self.data.xmat[entry["body"]].reshape(3, 3)
        winkel = []
        for k in range(3):
            achse = R[:, k]
            if abs(float(achse[2])) > np.cos(np.radians(60.0)):
                continue                        # steht zu steil
            winkel.append(float(np.arctan2(achse[1], achse[0])))
        if not winkel:
            return [float(np.arctan2(R[1, 0], R[0, 0]))]
        # Zwei Achsen genuegen: die dritte ist die Senkrechte der ersten.
        gewaehlt = [winkel[0]]
        for w in winkel[1:]:
            if all(abs(_wrap_half(w - g - np.pi / 2.0)) > 1e-3
                   and abs(_wrap_half(w - g)) > 1e-3 for g in gewaehlt):
                gewaehlt.append(w)
        if len(gewaehlt) == 1:
            gewaehlt.append(gewaehlt[0] + np.pi / 2.0)
        return gewaehlt[:2]

    def _grip_reference(self, entry) -> np.ndarray:
        """Wo die Backen den Koerper fassen -- aus der WELT, nicht der Huelle.

        Objekte werden an ihrer OBERSEITE gegriffen (die Greif-Skills
        senken auf ``Oberkante - Spanne/2``).  Die Oberkante muss aus der
        tatsaechlichen Lage kommen: ``entry["half"]`` ist die AABB IM
        KOERPERFRAME und fuehrt bei einem liegenden Stift dessen LAENGE
        als Hoehe.

        Am 2026-08-17 in der Studie gemessen, nachdem der Owner den Spalt
        gesehen hatte: ``marker/pick`` meldete Erfolg mit +12 bis +14 mm
        Abstand zwischen Backen und Stift.  Der Bezugspunkt lag rund
        57 mm ueber dem liegenden Koerper, :meth:`_span_between_pads`
        fand dort kein Geom und fiel auf die Huelle zurueck (26 mm statt
        18 mm Schaft) -- die Backen schlossen auf 26 mm und beruehrten
        nichts.

        Fuer einen aufrecht stehenden Koerper faellt beides zusammen; der
        Wuerfelturm bewegt sich also nicht.
        """
        pos = self.data.xpos[entry["body"]].copy()
        ref = pos.copy()
        # Wo die Backen WIRKLICH sind.  Ein konstruierter Punkt ("Oberkante
        # minus halbe Spanne") war die Quelle von drei Fehlern in Folge:
        # er unterstellt, dass die Aufgabe genau dorthin faehrt.  Sie tut
        # es nicht -- bei einem flachen Objekt hebt ihre Untergrenze
        # (Tisch + Spanne/2) die Backen an, und der Fang mass dann
        # 71,6 mm gegen seinen 70-mm-Radius und lehnte um 1,6 mm ab
        # (gemessen 2026-08-17 am liegenden Marker).
        # Die BACKEN sind die untersten Teile der Hand -- ein Mittelwert
        # ueber die ganze Baugruppe liegt beim RG6 rund 9 cm hoeher, also
        # im Handgelenk (gemessen 2026-08-17: Geoms von 0,363 bis 0,517).
        hoehen = sorted(float(self.data.geom_xpos[g][2])
                        for g in self._hand_geoms)
        if hoehen:
            unten = hoehen[:max(1, len(hoehen) // 3)]
            ref[2] = float(np.mean(unten))
            return ref
        oben = -np.inf
        for gid in entry["geoms"]:
            mitte = self.model.geom_aabb[gid][:3]
            halb = self.model.geom_aabb[gid][3:]
            R = self.data.geom_xmat[gid].reshape(3, 3)
            basis = self.data.geom_xpos[gid] + R @ mitte
            oben = max(oben, float(basis[2] + (np.abs(R) @ halb)[2]))
        if not np.isfinite(oben):
            oben = float(pos[2]) + float(entry["half"][2])
        ref[2] = oben - self._default_span / 2.0
        return ref

    def _span_between_pads(self, entry, ref, obj_yaw):
        """Breite des Koerpers auf Backenhoehe, entlang seiner beiden
        waagerechten Achsen -- oder ``None``, wenn dort nichts steht.

        Das Band der Hoehe ``default_span`` liegt um den GREIFPUNKT
        (``ref``), nicht um den TCP: der steht beim Griff rund 45 mm
        darueber (``DESCEND_OFFSET``), und ein Band um ihn laege
        vollstaendig ueber dem Objekt -- gemessen am Deckel, der damit
        weiter auf keiner Sprosse greifbar war.
        Gezaehlt wird nur, was in dieses Band ragt: genau das muss zwischen
        die Pads passen.  Ein Koerper, der weiter unten breit wird (die
        Scheibe eines Deckels), geht die Fangbedingung nichts an -- dass er
        SPAETER stoert, entscheidet die Kollision, nicht dieser Test.

        ``None`` heisst "auf Backenhoehe steht kein Geom dieses Koerpers";
        dann bleibt es beim Huellquader, statt einen Griff aus dem Nichts
        zu erlauben.
        """
        band = float(self._default_span) / 2.0
        lo_z, hi_z = float(ref[2]) - band, float(ref[2]) + band
        c, s = float(np.cos(obj_yaw)), float(np.sin(obj_yaw))
        achsen = (np.array([c, s, 0.0]), np.array([-s, c, 0.0]))
        grenzen = [[np.inf, -np.inf], [np.inf, -np.inf]]
        gefunden = False
        for gid in range(self.model.ngeom):
            if int(self.model.geom_bodyid[gid]) != int(entry["body"]):
                continue
            mitte_lokal = self.model.geom_aabb[gid][:3]
            halb = self.model.geom_aabb[gid][3:]
            R = self.data.geom_xmat[gid].reshape(3, 3)
            basis = self.data.geom_xpos[gid] + R @ mitte_lokal
            reichweite = np.abs(R) @ halb
            if basis[2] + reichweite[2] < lo_z or basis[2] - reichweite[2] > hi_z:
                continue                      # liegt nicht auf Backenhoehe
            gefunden = True
            for i, achse in enumerate(achsen):
                mitte = float(basis @ achse)
                weit = float(np.abs(R.T @ achse) @ halb)
                grenzen[i][0] = min(grenzen[i][0], mitte - weit)
                grenzen[i][1] = max(grenzen[i][1], mitte + weit)
        if not gefunden:
            return None
        return (grenzen[0][1] - grenzen[0][0], grenzen[1][1] - grenzen[1][0])

    def _try_grasp(self) -> None:
        """Proximity capture over every graspable free body.

        Parallel-jaw constraints: the pads must close across a pair of
        opposing faces -- alignment of the pad axis with one of the object's
        horizontal axes (modulo 180 deg per axis) AND that face pair's span
        within the stroke.  For a square object this reduces to the classic
        modulo-90 check; an elongated box is only captured across its short
        side.
        """
        if self._grasped is not None:
            return
        tcp_pos, tcp_mat = self.tcp_pose()
        gripper_yaw = self.gripper_pad_yaw()
        tol = np.radians(GRASP_MAX_MISALIGN_DEG)
        best = None  # (label, dist, signed misalign, span)
        for label, entry in self._graspable.items():
            pos = self.data.xpos[entry["body"]]
            # Grip-point reference: objects are gripped by their TOP slice
            # (grasp skills descend to top - span/2), so tall boxes must be
            # measured there, not at the body centre.  For a default-sized
            # object the two coincide (half height == span/2) -- classic
            # behaviour kept.
            ref = self._grip_reference(entry)
            dist = float(np.linalg.norm(ref - tcp_pos))
            if dist >= (best[1] if best is not None else GRASP_RADIUS):
                continue
            mat = self.data.xmat[entry["body"]].reshape(3, 3)
            obj_yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
            half = entry["half"]
            # Die Spanne kommt aus der Geometrie ZWISCHEN den Backen, nicht
            # aus der Huelle des ganzen Koerpers.  Fuer einen Wuerfel ist
            # das dasselbe; fuer alles andere war es die groebste denkbare
            # Abstraktion.  Gemessen an der Suffizienz-Vorstudie
            # (2026-08-17): ein Deckel mit 180-mm-Scheibe und 30-mm-Knauf
            # meldete auf ALLEN vier Objektsprossen "kein schliessbares
            # Flaechenpaar", weil hier 180 mm gegen 156 mm Backengang
            # standen -- unabhaengig davon, wie fein das Objekt modelliert
            # war.  Damit konnte die Objektachse ueber den Griff gar nicht
            # binden.
            achsen = self._horizontal_axes(entry)
            lokal = self._span_between_pads(entry, ref, achsen[0])
            spannen = (lokal if lokal is not None
                       else (2.0 * float(half[0]), 2.0 * float(half[1])))
            candidates = []
            for axis_yaw, span in (
                (achsen[0], spannen[0]),
                (achsen[1], spannen[1]),
            ):
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
            return
        label, best_dist, best_misalign, span = best
        adr = self._graspable[label]["qpos"]
        # Pad squaring: while closing, the flat pads rotate a slightly
        # misaligned object until its faces sit flush -- snap the yaw onto
        # the pad orientation before recording the carry offset.
        if abs(best_misalign) > 1e-6:
            snap = self._quat_about_z(-best_misalign)
            self.data.qpos[adr + 3 : adr + 7] = self._quat_mul(
                snap, self.data.qpos[adr + 3 : adr + 7].copy()
            )
            mujoco = self._mujoco
            mujoco.mj_forward(self.model, self.data)
        if not self._square_tilt(adr):
            log.debug("%s zu stark geneigt -- die Pads koennen das nicht "
                      "ausrichten", label)
            return
        obj_pos = self.data.qpos[adr : adr + 3].copy()
        obj_quat = self.data.qpos[adr + 3 : adr + 7].copy()  # wxyz
        # Offset of the object in the TCP frame, reapplied while carrying.
        rel_pos = tcp_mat.T @ (obj_pos - tcp_pos)
        tcp_quat = self._mat_to_quat(tcp_mat)
        rel_quat = self._quat_mul(self._quat_conj(tcp_quat), obj_quat)
        self._grasped = label
        self._grasp_offset = (rel_pos, rel_quat)
        self._grasp_span = span
        self._grasp_misalign = float(best_misalign)
        # Den Spalt JETZT festhalten -- im Moment des Zupackens.  Eine
        # spaetere Abfrage misst den TRAGEzustand: dort ist das Objekt
        # kinematisch angeschweisst und die Handbaugruppe ueberlappt es
        # zwangslaeufig (gemessen 2026-08-17 am Marker: -13,3 mm, was wie
        # eine Durchdringung beim Griff aussah und keine war).
        self._suspend_object_contacts(label)
        self._event_acc.grasp_acquired = label
        log.debug(
            "grasped %s (dist %.3f m, squared %.0f deg, span %.0f mm)",
            label, best_dist, np.degrees(best_misalign), span * 1e3,
        )

    def _square_tilt(self, adr: int) -> bool:
        """Pad-Squaring fuer die NEIGUNG -- das Gegenstueck zum Gierwinkel.

        Flache Backen, die sich um einen Koerper schliessen, richten ihn
        aus: seine beruehrten Flaechen legen sich flach an die Pads.  Fuer
        den GIERWINKEL tut das der Schnapp darueber seit jeher; die
        Neigung blieb, wie sie war, und das Objekt wurde SCHIEF getragen.

        Am 2026-08-17 in der Suffizienz-Vorstudie aufgelaufen (Owner sah es
        in Foxglove: "der griff sieht aus als sei der greifer nicht genug
        geschlossen"): ein Stift lehnt im Koecher 6 Grad, wurde 6 Grad
        schief getragen und passte danach in keinen Becher mehr -- move_group
        meldete ``RRTConnect: Unable to sample any valid states for goal
        tree``.  Die Geometrie war richtig, nur die Mechanik unvollstaendig.

        Geschnappt wird auf die NAECHSTE achsparallele Lage, nicht stur auf
        senkrecht: ein liegender Koerper bleibt sonst nicht liegen, sondern
        wuerde beim Griff aufgestellt -- eine Bewegung, die es nicht gibt.
        Die Drehung ist damit immer die kleinstmoegliche, und fuer einen
        Wuerfel, der ohnehin achsparallel steht, ist sie null (der
        Wuerfelturm bewegt sich nicht).
        """
        q = self.data.qpos[adr + 3 : adr + 7].copy()
        R = np.zeros(9)
        self._mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        # NUR die Neigung entfernen, den Gierwinkel behalten.  Der erste
        # Entwurf (2026-08-17 frueh) schnappte auf die naechste
        # WELTachsenparallele Lage und machte damit die Pad-Ausrichtung
        # kaputt, die der Gier-Schnapp gerade hergestellt hatte -- fuer
        # die Wuerfel des Turms eine Drehung von ueber 15 Grad, also
        # genau die kostenlose Grosskorrektur, gegen die dieser Riegel
        # gebaut ist.
        #
        # Gesucht ist die kleinste Drehung, die die koerpereigene Achse,
        # die der Senkrechten am naechsten liegt, AUF die Senkrechte
        # bringt.  Ein liegender Koerper bleibt liegen: dort ist die
        # naechste Achse eine andere, und die Drehung ist wieder klein.
        k = int(np.argmax(np.abs(R[2, :])))
        achse = R[:, k] * (1.0 if R[2, k] >= 0 else -1.0)
        ziel = np.array([0.0, 0.0, 1.0])
        winkel = float(np.arccos(np.clip(float(achse @ ziel), -1.0, 1.0)))
        if np.degrees(winkel) > PAD_SQUARE_LIMIT_DEG:
            return False
        if winkel < 1e-9:
            return True
        dreh = np.cross(achse, ziel)
        dreh /= np.linalg.norm(dreh)
        korrektur = np.array([np.cos(winkel / 2.0),
                              *(np.sin(winkel / 2.0) * dreh)])
        self.data.qpos[adr + 3 : adr + 7] = self._quat_mul(korrektur, q)
        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _fingers_settled(self, tol: float = 0.02) -> bool:
        """Haben die Finger ihre kommandierte Weite erreicht?

        Der Griffspalt darf erst DANN gemessen werden.  Im Moment des
        Fangs stehen die Backen noch offen (gemessen +80 mm), nach dem
        Hub ueberlappt der getragene Koerper die Handbaugruppe (-13 mm) --
        beide Zahlen sahen aus wie Aussagen ueber die Griffguete und
        waren keine.
        """
        for joint, factor in self._gripper_follower_factors.items():
            jid = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint)
            if jid < 0:
                continue
            ist = float(self.data.qpos[self.model.jnt_qposadr[jid]])
            soll = float(self._gripper_command) * float(factor)
            if abs(ist - soll) > tol:
                return False
        return True

    def grasp_misalign_deg(self):
        """Wie schief das Objekt beim Zupacken stand (Grad), oder ``None``.

        Die Pads richten ein leicht schiefes Objekt beim Schliessen aus.
        Diese Drehung fand bisher stillschweigend statt: die Zahl wurde
        berechnet, angewandt, auf Debug-Ebene geloggt und verworfen.

        Fuer eine Suffizienzmessung ist sie aber DER Fehler, den ein
        grobes Modell erzeugt: der Roboter zielt nach seinem Quader, der
        echte Koerper steht anders, und die Grosszuegigkeit des Fangs
        buegelt es aus.  move_group kann das nicht melden -- der
        Backenkontakt ist beim Griff ausdruecklich freigegeben, sonst
        waere jeder Griff ein Startzustand in Kollision.
        """
        if self._grasped is None or self._grasp_misalign is None:
            return None
        return float(np.degrees(self._grasp_misalign))

    def grasp_gap(self):
        """Kleinster Abstand Greifer<->gegriffener Koerper (m), oder ``None``.

        Die PRUEFUNG zum Modell: der Fang entscheidet ueber Abstand,
        Ausrichtung und Spanne und schweisst dann -- ob die Backen den
        Koerper wirklich beruehren, stand nirgends.  Gemessen wird, sobald
        die Finger ihre Weite erreicht haben (:meth:`_fingers_settled`):
        im Moment des Fangs stehen sie noch offen (+80 mm), nach dem Hub
        ueberlappt der getragene Koerper die Handbaugruppe (-13 mm).

        Lesart: ``< 0`` Durchdringung, ``~ 0`` Beruehrung, deutlich
        ``> 0`` die Backen greifen ins Leere.  ``None`` heisst "kein
        Griff, keine Aussage" -- nicht "in Ordnung".
        """
        return self._grasp_gap0

    def _measure_gap(self, entry) -> Optional[float]:
        """Kleinster Abstand Hand<->Koerper JETZT (m), oder ``None``."""
        if entry is None or not self._hand_geoms:
            return None
        kleinster = float("inf")
        for hand in self._hand_geoms:
            for gid in entry["geoms"]:
                d = float(self._mujoco.mj_geomDistance(
                    self.model, self.data, int(hand), int(gid), 1.0, None))
                kleinster = min(kleinster, d)
        return None if not np.isfinite(kleinster) else kleinster

    def _release(self) -> None:
        if self._grasped is None:
            return
        label = self._grasped
        self._grasped = None
        self._grasp_offset = None
        self._grasp_span = None
        self._grasp_misalign = None
        self._grasp_gap0 = None
        # Hand the object back to physics at rest: restore its contacts and
        # clear any residual solver velocity accumulated while pinned.
        self._restore_object_contacts(label)
        dof = self._free_joint_dof(label)
        self.data.qvel[dof : dof + 6] = 0.0
        self._event_acc.grasp_lost = label
        log.debug("released %s", label)

    def _suspend_object_contacts(self, label: str) -> None:
        """Turn off all contacts of a carried object.

        While carried the object is kinematically pinned to the TCP -- it is
        effectively part of the end effector.  Leaving its contacts on lets
        the contact solver fight the pin whenever an IK solution sweeps a
        robot link (e.g. the upper arm) through the carry zone; the growing
        penetration then discharges as a catapult impulse on release.
        """
        saved: Dict[int, Tuple[int, int]] = {}
        entry = self._graspable.get(label)
        for gid in (entry["geoms"] if entry else []):
            saved[gid] = (int(self.model.geom_contype[gid]), int(self.model.geom_conaffinity[gid]))
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0
        self._object_contact_masks[label] = saved

    def _restore_object_contacts(self, label: str) -> None:
        saved = self._object_contact_masks.pop(label, {})
        for gid, (contype, conaffinity) in saved.items():
            self.model.geom_contype[gid] = contype
            self.model.geom_conaffinity[gid] = conaffinity

    def grasped_label(self) -> Optional[str]:
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

        Real mode: the sim objects are not physics ground truth (the real ones
        are); they exist so the dashboard twin shows the scene.  This writes
        the belief into the free joint -- afterwards the object simply obeys
        physics again (settles onto the floor/stack).
        """
        adr = self._graspable[label]["qpos"]
        self.data.qpos[adr : adr + 3] = np.asarray(position, dtype=float)
        self.data.qpos[adr + 3 : adr + 7] = self._quat_about_z(float(yaw))
        dof = self._free_joint_dof(label)
        self.data.qvel[dof : dof + 6] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def park_object(self, label: str, index: int = 0) -> None:
        """Move an un-localized object out of every camera view (display only)."""
        px, py, pz = self._OBJECT_PARK
        self.display_object(label, (px - 0.3 * index, py, pz))

    def display_carry(self, label: str) -> None:
        """Pin an object under the TCP for the dashboard twin -- event-free.

        The real-mode counterpart of the kinematic carry: the real gripper
        holds the real object, the sim only *shows* it.  Reuses the carry
        machinery (:meth:`_carry_grasped` follows the TCP every substep,
        contacts are suspended so the pin cannot fight the gripper shells) but
        never touches the event accumulator -- a display pin must not look like
        a grasp/drop to the skill layer.
        """
        if self._grasped == label:
            return
        if self._grasped is not None:
            self.display_release(self._grasped)
        tcp_pos, tcp_mat = self.tcp_pose()
        rel_pos = np.array([0.0, 0.0, self._CARRY_OFFSET])
        adr = self._graspable[label]["qpos"]
        self.data.qpos[adr : adr + 3] = tcp_pos + tcp_mat @ rel_pos
        self.data.qpos[adr + 3 : adr + 7] = self._mat_to_quat(tcp_mat)
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
        tcp_quat = self._mat_to_quat(tcp_mat)
        self.data.qpos[adr + 3 : adr + 7] = self._quat_mul(tcp_quat, rel_quat)
        dof = self._free_joint_dof(self._grasped)
        self.data.qvel[dof : dof + 6] = 0.0

    def _free_joint_dof(self, label: str) -> int:
        return self._graspable[label]["dof"]

    # ------------------------------------------------------------------ #
    # stepping
    # ------------------------------------------------------------------ #
    def step_physics(self, n_ticks: int = 1) -> SimEvents:
        """Advance ``n_ticks`` control periods, holding all commanded state.

        Returns the events (collisions, grasp changes) accumulated over the
        stepped interval.
        """
        mujoco = self._mujoco
        events = SimEvents()
        events.merge(self._event_acc)
        self._event_acc = SimEvents()
        # Den Griffspalt messen, sobald die FINGER IHRE WEITE ERREICHT
        # haben -- nicht im Moment des Fangs (da stehen sie noch offen,
        # gemessen +80 mm) und nicht nach dem Hub (da ueberlappt der
        # getragene 140-mm-Stift die Handbaugruppe, gemessen -13,3 mm).
        # Beide Zeitpunkte sahen aus wie Aussagen ueber die Griffguete und
        # waren keine.
        if (self._grasped is not None and self._grasp_gap0 is None
                and self._gripper_closing and self._fingers_settled()):
            self._grasp_gap0 = self._measure_gap(
                self._graspable.get(self._grasped))
        gripper_targets = {
            joint: self._gripper_command * factor
            for joint, factor in self._gripper_follower_factors.items()
        }
        for _ in range(int(n_ticks)):
            # Where the fingers stand as this tick begins -- the actuated
            # regime ramps its setpoint from here (see _drive_gripper).
            start = (
                {j: float(self.data.qpos[self._joint_qpos[j]])
                 for j in gripper_targets if j in self._joint_qpos}
                if self._actuated_gripper else {}
            )
            for substep in range(self.n_substeps):
                for name, value in self._arm_command.items():
                    adr = self._joint_qpos.get(name)
                    if adr is not None:
                        self.data.qpos[adr] = value
                        self.data.qvel[self._joint_dof[name]] = 0.0
                if self._actuated_gripper:
                    self._drive_gripper(
                        start, gripper_targets, (substep + 1) / self.n_substeps
                    )
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

    def _drive_gripper(
        self, start: Dict[str, float], targets: Dict[str, float], alpha: float
    ) -> None:
        """Write ``data.ctrl`` for the follower joints; do NOT pin their qpos.

        Two halves, and the flag is worthless without either.  Writing ctrl is
        the obvious one -- ``ctrl`` left at zero means the position servos hold
        the OPEN angle, so a closing command was answered with a saturated
        counter-torque (measured: 20 N.m on every follower).  Leaving qpos
        alone is the other: a joint that is overwritten every substep cannot
        stop at the object, so no contact force can build no matter what the
        servo is told.

        The setpoint is ramped across the substeps of a tick instead of
        jumping: the command changes binary open<->closed, and a step of that
        size drives the fingers through the object within one substep before
        the contact solver ever sees them.
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
        self,
        joints: Dict[str, float],
        *,
        penetration: float = 0.003,
        obstacles_only: bool = False,
    ) -> bool:
        """True if the arm configuration is in *robot self-collision*.

        Mirrors move_group's start/goal state validation on a scratch
        ``MjData``.  Three pair classes invalidate a configuration:

        * manipulator vs. platform (the case from the MoveIt log:
          "contact between 'base_link' and 'rg6_onrobot_rg6_base_link'"),
        * hand assembly (gripper + wrist camera) vs. shoulder/upper-arm/
          forearm -- a fully folded elbow that sweeps the arm through the
          gripper's carry zone, and
        * manipulator vs. perceived obstacles / distractors -- the pool
          :meth:`set_obstacles` maintains mirrors what scene_sync publishes
          to move_group, so the client-side IK gate rejects goal states
          move_group would reject as obstacle collisions.

        Table/ground/graspable contacts are deliberately NOT part of this gate:
        the finger envelope honestly dips toward the table during grasps
        (MoveIt's current planning scene contains no table either); execution
        crashes remain covered by the contact-event scan.

        ``obstacles_only=True`` restricts the gate to the obstacle pairs --
        the pose-goal pre-send probe uses it (platform/self validity stays
        move_group's call there, and a hand-vs-obstacle hit at a given TCP
        pose is independent of which IK branch move_group will pick).
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
        # Park the sim's own payload out of the way: its contacts are not part
        # of the validity question (and a carried object travels with the TCP
        # anyway).  Graspables that are ALSO obstacles stay put -- the gate
        # exists to reject configurations that reach into them.
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

        Obstacle-classified graspables are scene furniture, not payload: they
        are excluded, so a jittering piece of clutter cannot stretch every
        settle (and with it every reset) to the tick limit.
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

        The generic head of a task reset: the subclass calls this and then
        distributes its own objects (which is the task-specific part).
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
    def fk_body_pose(
        self, body: str, arm_joints: Dict[str, float]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Forward-kinematics world pose of ``body`` for given arm joint angles.

        Uses a scratch ``MjData``, so the pose comes back in MuJoCo world
        coordinates -- and that origin is *ground-referenced*, not the URDF
        root: the converter raises the welded base until the wheels rest on
        z=0 (``urdf_mujoco``, step 7), which puts the world origin at
        ``base_footprint``, 0.132 m below the Husky's ``base_link``.  A caller
        that treats this as a ``base_link`` pose is off by exactly that much.
        Used by the real-hardware camera to resolve the wrist
        RealSense pose from the *live* arm joint_states when the foxglove bridge
        does not relay the manipulator's static TF (the camera optical frame is
        driver-only tf, absent from the URDF/MuJoCo model anyway).
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

    def tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
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
        # The pads close along the axis; the *face normal* orientation modulo
        # 90 deg is what matters, so the perpendicular works equally.
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

    def render_rgb(self, camera: str, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        w = width or self._render_size[0]
        h = height or self._render_size[1]
        renderer = self._ensure_renderer(w, h)
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera=camera, scene_option=self._scene_option)
        return renderer.render()

    def render_depth(self, camera: str, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        w = width or self._render_size[0]
        h = height or self._render_size[1]
        renderer = self._ensure_renderer(w, h)
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=camera, scene_option=self._scene_option)
        depth = renderer.render()
        renderer.disable_depth_rendering()
        return depth

    def camera_matrix(self, camera: str, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        w = width or self._render_size[0]
        h = height or self._render_size[1]
        return camera_intrinsics(self.model, camera, w, h)

    def camera_pose(self, camera: str) -> Tuple[np.ndarray, np.ndarray]:
        return camera_extrinsics(self.data, self.model, camera)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ #
    # small quaternion helpers (wxyz, matching MuJoCo)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_about_z(angle: float) -> np.ndarray:
        return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])

    def _mat_to_quat(self, mat: np.ndarray) -> np.ndarray:
        quat = np.empty(4)
        self._mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat
