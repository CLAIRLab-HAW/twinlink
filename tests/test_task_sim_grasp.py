"""Proximity-capture grasp mechanic: alignment, capture, carry, release.

The parallel-jaw alignment check, kinematic carry and release all live in ``TwinTaskSim._try_grasp`` /
``_carry_grasped`` / ``_release`` -- none of it is about cubes.  Driving that mechanic indirectly through a real UR5
descend would test the arm; here the same production methods are exercised directly against a tiny, robot- and task-free
MJCF model.

Gravity is off (``gravity="0 0 0"``): nothing here is about resting/settling physics (that stays covered task-side),
only about the capture/carry/release state machine.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.testing import StraightLinkage  # noqa: E402
from twinlink.task_sim import GRASP_MAX_MISALIGN_DEG, RobotSimSpec, TwinTaskSim, _wrap_half, _wrap_quarter  # noqa: E402

#: A slider "arm" with a fixed (non-rotating) gripper orientation -- the
#: TCP's world x sits at ``arm_0_slide + 0.17``, y=0, z=0.08.  A payload box
#: sits directly at the TCP's position for ``arm_0_slide = 0.5`` so tests can
#: focus purely on ORIENTATION (misalignment), not on reach/distance.
SCENE_XML = """
<mujoco model="grasp_mechanic">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="twinlink_ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="arm_0_shoulder_link" pos="0 0 0.08">
      <joint name="arm_0_slide" type="slide" axis="1 0 0" range="-2 2"/>
      <geom name="arm_0_geom" type="box" size="0.05 0.05 0.05"/>
      <body name="rg6_base" pos="0.12 0 0">
        <geom name="rg6_geom" type="box" size="0.04 0.04 0.04"/>
        <body name="rg6_hand_tcp" pos="0.05 0 0">
          <geom name="rg6_tcp_marker" type="sphere" size="0.005"
                contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
    <body name="payload" pos="0.67 0 0.08">
      <freejoint name="payload_free"/>
      <geom name="payload_geom" type="box" size="0.02 0.015 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""

SPEC = RobotSimSpec(
    manipulator_prefixes=("arm_0", "rg6"),
    hand_prefixes=("rg6",),
    gripper_prefixes=("rg6",),
    far_arm_bodies=("arm_0_shoulder_link",),
    gripper_stroke_m=0.156,
    tcp_body="rg6_hand_tcp",
    arm_joints=("arm_0_slide",),
)

#: The slide position that puts the TCP exactly on the payload's centre
#: (TCP.x = slide + 0.17; payload.x = 0.67).
ALIGNED_SLIDE = 0.5
#: Far enough that the payload is well outside GRASP_RADIUS (0.07 m).
FAR_SLIDE = -1.0
#: The gripper's pad-opening axis is fixed by the (non-rotating) scene at
#: world yaw = 90 deg (see ``TwinTaskSim.gripper_pad_yaw``: atan2 of the
#: TCP frame's y-axis, which here is always (0, 1, 0)).
GRIPPER_PAD_YAW_DEG = 90.0


class _GraspSim(TwinTaskSim):
    def register_graspables(self) -> None:
        self.register_graspable("payload", "payload_free", self._body_id("payload"), np.array([0.02, 0.015, 0.02]))


def _build() -> _GraspSim:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return _GraspSim(
        model,
        SPEC,
        scene_prefix="",
        default_span=0.04,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def _set_payload_yaw(sim: _GraspSim, yaw_rad: float) -> None:
    adr = sim._graspable["payload"]["qpos"]
    sim.data.qpos[adr + 3 : adr + 7] = np.array([np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)])
    sim._mujoco.mj_forward(sim.model, sim.data)


def _approach(sim: _GraspSim, slide: float = ALIGNED_SLIDE) -> None:
    sim.set_arm_command({"arm_0_slide": slide})
    sim.step_physics(3)


# --------------------------------------------------------------------- #
# pure helper
# --------------------------------------------------------------------- #
def test_wrap_quarter():
    assert _wrap_quarter(0.0) == pytest.approx(0.0)
    assert _wrap_quarter(np.pi / 2) == pytest.approx(0.0)  # 90deg = symmetric
    assert _wrap_quarter(np.radians(30)) == pytest.approx(np.radians(30))
    assert _wrap_quarter(np.radians(60)) == pytest.approx(np.radians(-30))


# --------------------------------------------------------------------- #
# capture: parallel-jaw alignment gate
# --------------------------------------------------------------------- #
def test_misaligned_payload_is_not_captured():
    """Pads at 90deg (fixed), payload at 90+35deg ─▶ jaws meet edges, no grasp."""
    sim = _build()
    try:
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG + 35))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() is None
    finally:
        sim.close()


def test_aligned_payload_is_captured():
    """Payload yaw exactly matches the pad axis ─▶ captured."""
    sim = _build()
    try:
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() == "payload"
    finally:
        sim.close()


def test_small_misalignment_is_squared_up():
    """Within tolerance the pads square the payload onto the gripper yaw."""
    sim = _build()
    try:
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG + 12))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() == "payload"
        residual = _wrap_quarter(sim.object_yaw("payload") - sim.gripper_pad_yaw())
        assert abs(np.degrees(residual)) < 1.0, "pads should square the payload"
    finally:
        sim.close()


def test_tolerance_boundary_is_rejected():
    sim = _build()
    try:
        beyond = GRASP_MAX_MISALIGN_DEG + 5
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG + beyond))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() is None
    finally:
        sim.close()


def test_gripper_close_without_anything_in_reach():
    sim = _build()
    try:
        sim.set_arm_command({"arm_0_slide": FAR_SLIDE})
        sim.step_physics(3)
        sim.command_gripper(close=True)
        assert sim.grasped_label() is None
    finally:
        sim.close()


# --------------------------------------------------------------------- #
# carry + release
# --------------------------------------------------------------------- #
def test_grasp_carry_release():
    sim = _build()
    try:
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() == "payload"
        # carried: the payload follows the TCP as the arm moves.
        sim.set_arm_command({"arm_0_slide": ALIGNED_SLIDE + 0.2})
        sim.step_physics(3)
        tcp, _mat = sim.tcp_pose()
        payload_pos = sim.object_position("payload")
        assert float(np.linalg.norm(payload_pos - tcp)) < 0.03
        # released: grasp ends and the object's ordinary contacts return.
        gid = sim._graspable["payload"]["geoms"][0]
        assert sim.model.geom_contype[gid] == 0, "sanity: contacts suspended while carried"
        sim.command_gripper(close=False)
        assert sim.grasped_label() is None
        assert sim.model.geom_contype[gid] != 0, "release must restore contacts"
    finally:
        sim.close()


# --------------------------------------------------------------------- #
# Gripper width, publicly readable
#
# The twin drives the fingers to the OBJECT WIDTH when an object is held.  If
# that width only lived inside the private ``_gripper_command``, nobody could
# mirror it onto a real gripper without reaching into the internals -- and
# measured in the container: its RG6 joints stood constantly at 0.0 (fully
# open) throughout an entire cell run while the twin was gripping.  Two
# different hands to check against.
# --------------------------------------------------------------------- #
def test_the_open_gripper_reports_what_the_linkage_can_open_to():
    """The open width comes from the LINKAGE, not from ``gripper_stroke_m``.

    If the sim computes the width back from ``gripper_stroke_m`` and two anchors, both numbers inevitably coincide.  But
    they mean different things: ``gripper_stroke_m`` is the budget a grasp span is measured against
    (``GRASP_SPAN_FRACTION``), while the opening of the hand is a property of the linkage -- on the real RG6 159.0 mm,
    where the sim span carries 156 mm.  Equating them is exactly the confusion that helped carry the gripper bug.
    """
    sim = _build()
    sim.command_gripper(False)
    assert sim.gripper_width_m() == pytest.approx(StraightLinkage().max_width_m)
    assert sim.gripper_width_m() != pytest.approx(SPEC.gripper_stroke_m)


def test_a_grasp_drives_the_joint_the_linkage_asks_for_not_a_line_of_its_own():
    """The actual bug, as a test.

    ``command_gripper`` used to convert the width into a joint value with a straight line of its OWN (``closed * (1 -
    width/stroke)``).  From the outside that went unnoticed because ``gripper_width_m`` walked the same line backwards
    and covered the error up -- it only became visible at the joint, that is, at what is actually written into the model
    and what move_group checks.
    """
    linkage = StraightLinkage()
    sim = _build()
    _approach(sim)
    sim.command_gripper(True, grasp=True)
    span = sim.gripper_width_m()
    assert sim.gripper_command_rad == pytest.approx(linkage.angle_from_width(span))
    # And the old straight line would have returned something else here.
    old_linear = linkage.closed_rad * (1.0 - span / SPEC.gripper_stroke_m)
    assert sim.gripper_command_rad != pytest.approx(old_linear, abs=1e-3)


def test_the_empty_closed_gripper_reports_zero_width():
    sim = _build()
    sim.command_gripper(True)  # nothing to grasp ─▶ fully closed
    assert sim.gripper_width_m() == pytest.approx(0.0, abs=1e-9)


def test_a_grasped_object_sets_the_width_to_its_own_span():
    """The one number this accessor exists for.

    It follows the OBJECT, not some setting: the payload measures 0.04 x 0.03 x 0.04 m, the pads close over its 0.03 m
    edge -- and NOT onto ``default_span`` (0.04), which only applies where no object was measured.  It is exactly this
    dependence on the object that is lost on a gripper with a fixed opening.
    """
    sim = _build()
    _approach(sim)
    sim.command_gripper(True)
    assert sim.grasped_label() == "payload", "ohne Griff prueft der Test nichts"
    assert sim.gripper_width_m() == pytest.approx(0.03, abs=2e-3)
    assert sim.gripper_width_m() != pytest.approx(
        0.04, abs=2e-3
    ), "die Weite haengt an der Vorgabe statt am Objekt -- dann misst sie nichts"


# --------------------------------------------------------------------- #
# Placing down: open only as far as necessary
# --------------------------------------------------------------------- #
def test_releasing_opens_only_as_far_as_the_object_needed():
    """Tearing the hand wide open to release is a choice, not a necessity.

    Measured on the container: after placing down the hand stood at full width RIGHT IN THE GATE, and move_group refused
    the retreat (``2 contact(s) detected : gate_0 - ..._finger_2, gate_1 - ..._finger_1``).  A real RG6 opens for the
    release only as far as the object demands.

    The payload measures 0.03 m across the closed edge; with 0.005 m clearance per side that is 0.04 m -- markedly less
    than the 0.16 m of the linkage.
    """
    sim = _build()
    _approach(sim)
    sim.command_gripper(True)
    assert sim.grasped_label() == "payload", "ohne Griff prueft der Test nichts"

    sim.command_gripper(False)
    assert sim.grasped_label() is None, "das Objekt muss losgelassen sein"
    assert sim.gripper_width_m() == pytest.approx(0.04, abs=2e-3)
    assert (
        sim.gripper_width_m() < StraightLinkage().max_width_m
    ), "die Hand reisst beim Ablegen weiter auf, als das Objekt verlangt"


def test_the_release_opening_is_clamped_to_what_the_linkage_can_do():
    """A clearance larger than the linkage must not invent a value."""
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    sim = _GraspSim(
        model,
        SPEC,
        scene_prefix="",
        default_span=0.04,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
        release_clearance=1.0,
    )
    _approach(sim)
    sim.command_gripper(True)
    sim.command_gripper(False)
    assert sim.gripper_width_m() == pytest.approx(StraightLinkage().max_width_m)


def test_a_released_gripper_does_not_report_itself_closed():
    """The trap that the narrow opening springs.

    ``gripper_closed`` compared the joint value against HALF the closed position.  As long as the hand opened fully to
    release, that went unnoticed; once it only opens to object width plus clearance, it falls below the same threshold
    for small objects and reported itself as closed -- with the real linkage for every span below 40 mm.  Whether the
    hand is closed is said by the command, not by a geometric threshold.
    """
    sim = _build()
    _approach(sim)
    sim.command_gripper(True)
    assert sim.gripper_closed() is True, "die geschlossene Hand muss zu melden"
    sim.command_gripper(False)
    assert sim.gripper_closed() is False


# --------------------------------------------------------------------- #
# The span comes from the geometry BETWEEN the jaws
#
# Read from ``entry["half"]`` -- the bounding-box half extents of the whole
# body -- it is the same thing for a cube and the coarsest conceivable
# abstraction for everything else.  Measured: a lid (180 mm disc, 30 mm knob)
# reported "kein schliessbares Flaechenpaar" on ALL FOUR object rungs --
# 180 mm against a 156 mm jaw travel, independent of the model fidelity.
#
# So alpha_obj could not bind over the GRASP either: the twin read the
# bounding box regardless of what really stood between the jaws -- that
# explains a flat measurement table as a property of the SETUP.
# --------------------------------------------------------------------- #
WIDE_SCENE_XML = SCENE_XML.replace(
    '<geom name="payload_geom" type="box" size="0.02 0.015 0.02"/>',
    # A wide disc BELOW, a narrow knob ABOVE -- the lid in miniature.  The jaws stand at the height of the knob.
    '<geom name="payload_disc" type="box" size="0.09 0.09 0.008"'
    ' pos="0 0 -0.042"/>'
    '<geom name="payload_knob" type="box" size="0.015 0.015 0.042"'
    ' pos="0 0 0.008"/>',
)


class _WideSim(_GraspSim):
    def register_graspables(self) -> None:
        # The bounding box is that of the whole body -- 180 mm wide.  That is exactly the number the capture condition
        # must NO LONGER use.
        self.register_graspable("payload", "payload_free", self._body_id("payload"), np.array([0.09, 0.09, 0.05]))


def _build_wide() -> _WideSim:
    model = mujoco.MjModel.from_xml_string(WIDE_SCENE_XML)
    return _WideSim(
        model,
        SPEC,
        scene_prefix="",
        default_span=0.04,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def test_a_narrow_feature_is_grasped_even_when_the_whole_body_is_wide():
    """The knob fits into the jaws, the body does not -- what is grasped is
    what stands between the jaws."""
    sim = _build_wide()
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload", "die Fangbedingung liest weiterhin den Huellquader"


def test_the_captured_span_is_the_local_one_not_the_bounding_box():
    """The jaw width on release follows what was grasped.

    With the bounding-box span the gripper opened to 180 mm -- wider than its travel, and wider than necessary -- which
    is exactly the observation the release width grew out of.
    """
    sim = _build_wide()
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim._grasp_span == pytest.approx(0.030, abs=1e-3), f"Spanne {sim._grasp_span} -- erwartet der Knauf (30 mm)"


def test_a_body_that_is_wide_everywhere_is_still_refused():
    """The counter-check: without a narrow spot the answer stays no.

    Without it the change could simply have abolished every check, and the test above would still pass.
    """
    xml = SCENE_XML.replace(
        '<geom name="payload_geom" type="box" size="0.02 0.015 0.02"/>',
        '<geom name="payload_geom" type="box" size="0.09 0.09 0.05"/>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    sim = _WideSim(
        model,
        SPEC,
        scene_prefix="",
        default_span=0.04,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() is None


# --------------------------------------------------------------------- #
# Pad squaring for the TILT as well
#
# The twin welded the object fast at whatever tilt it caught it in -- a pen
# leaning 6 degrees in its holder was therefore CARRIED 6 degrees crooked and
# no longer fitted into any cup in move_group (``RRTConnect: Unable to sample
# any valid states for goal tree``).
#
# Flat jaws closing around a slender body straighten it out -- the same
# mechanism as for the yaw angle, only about a different axis.  It belongs in
# the twin because THAT is what holds the truth and move_group takes it over.
# --------------------------------------------------------------------- #
def _set_payload_tilt(sim, tilt_rad: float) -> None:
    """Tilts the payload about the x axis (out of the vertical)."""
    adr = sim._graspable["payload"]["qpos"]
    sim.data.qpos[adr + 3 : adr + 7] = np.array([np.cos(tilt_rad / 2), np.sin(tilt_rad / 2), 0.0, 0.0])
    sim._mujoco.mj_forward(sim.model, sim.data)


def _tilt_of(sim) -> float:
    adr = sim._graspable["payload"]["qpos"]
    w, x, y, z = sim.data.qpos[adr + 3 : adr + 7]
    # Winkel der Koerper-z-Achse gegen die Welt-z-Achse.
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(min(1.0, abs(zz)))))


def test_the_pads_square_a_tilted_object_upright():
    sim = _build()
    _set_payload_tilt(sim, np.radians(8.0))
    assert _tilt_of(sim) == pytest.approx(8.0, abs=0.5)
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    assert _tilt_of(sim) < 1.0, (
        f"nach dem Griff noch {_tilt_of(sim):.1f} Grad schief -- die Backen " f"richten den Koerper nicht auf"
    )


def test_squaring_snaps_to_the_NEAREST_axis_not_to_upright():
    """A body that LIES DOWN stays lying down.

    Snapping stubbornly to the vertical stands a lying pen upright on grasping -- a motion that does not exist, and the
    marker of the pre-study lies down in two out of three cells.
    """
    sim = _build()
    _set_payload_tilt(sim, np.radians(88.0))  # fast waagerecht
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    assert _tilt_of(sim) > 89.0, f"der liegende Koerper wurde auf {_tilt_of(sim):.1f} Grad gedreht"


# --------------------------------------------------------------------- #
# The grasp is CHECKED, not just modelled
#
# ``_try_grasp`` is a MODEL: distance, orientation, span -- after that comes
# the weld.  Whether the pads really touch the body, penetrate it or grab past
# it, that does not check.  For the sufficiency study that is precisely the
# point: there the robot aims according to a COARSE model and meets REALITY,
# and whether the grasp then still holds must not be covered up by the
# generosity of the capture condition (7 cm, 20 degrees).
#
# The check goes through ``mj_geomDistance`` -- a pure distance query.  The
# contacts of the carried body are switched off for good reason; a query
# disturbs none of that.
# --------------------------------------------------------------------- #
def test_a_sound_grasp_has_both_pads_at_the_object():
    sim = _build()
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    spalt = sim.grasp_gap()
    assert spalt is not None, "kein Griff, also kein Spalt"
    assert spalt < 0.01, f"Abstand {spalt*1000:.1f} mm -- die Backen beruehren den Koerper " f"nicht wirklich"


def test_without_a_grasp_there_is_no_gap_to_report():
    """``None`` means "no statement", not "everything is fine"."""
    sim = _build()
    assert sim.grasp_gap() is None


def test_the_check_sees_a_body_the_pads_pass_through():
    """The actual assurance: penetration is noticed.

    Without it a grasp whose pads stand right INSIDE the body reported the same gap as a clean one -- and the study
    counted it as a success.
    """
    sim = _build()
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    # Den Koerper kuenstlich in die Pads schieben.
    adr = sim._graspable["payload"]["qpos"]
    sim.data.qpos[adr + 1] += 0.02
    sim._mujoco.mj_forward(sim.model, sim.data)
    spalt = sim.grasp_gap()
    assert spalt < 0.0, f"Abstand {spalt*1000:+.1f} mm -- eine Durchdringung muss negativ " f"sein"


# --------------------------------------------------------------------- #
# The MISALIGNMENT at capture is reported, not just snapped away
#
# Two things together make a gap:
#   * move_group CANNOT refuse -- jaw contact is explicitly allowed during the
#     grasp (``_target_touchable``), otherwise every grasp would be a start
#     state in collision.
#   * The twin SEES the misalignment (``best_misalign``), snaps it away and
#     throws the number away.  The grasp gap measures afterwards and therefore
#     looks clean.
#
# So of all things the error a coarse model produces was invisible: the robot
# aims at the box, the real body stands differently, and the generosity of the
# capture irons it out.  The number has to come out.
# --------------------------------------------------------------------- #
def test_the_capture_reports_how_far_it_had_to_square_the_object():
    sim = _build()
    _set_payload_yaw(sim, np.radians(12.0))
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    skewed = sim.grasp_misalign_deg()
    assert skewed is not None
    assert abs(skewed) == pytest.approx(12.0, abs=1.5), f"gemeldet {skewed} Grad -- der Fang musste 12 Grad ausbuegeln"


def test_a_square_grasp_reports_nearly_zero():
    sim = _build()
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert abs(sim.grasp_misalign_deg()) < 1.0


def test_without_a_grasp_there_is_no_misalignment_to_report():
    sim = _build()
    assert sim.grasp_misalign_deg() is None


# --------------------------------------------------------------------- #
# The grasp height follows the BODY, not its imagined upright bounding box
#
# Measured: marker/pick reported success with a gap of +12 to +14 mm -- the
# jaws stood more than a centimetre beside the pen.  Cause: the grasp point
# came from ``entry["half"][2]``, half the height of the AABB IN THE BODY
# FRAME.  The marker LIES DOWN, its true upper edge is about 13 mm above its
# centre instead of 70 -- the probe band lay above the object,
# ``_span_between_pads`` found no geom there and fell back on the bounding box.
# --------------------------------------------------------------------- #
def test_the_grip_reference_follows_the_real_body_not_its_upright_hull():
    """The grasp point comes from the WORLD, not from the body-frame AABB.

    The measurement for this stands in the section header above.  What is checked here is the reference point itself:
    the synthetic setup has no jaws that stop at a width and therefore cannot show the gap.
    """
    xml = SCENE_XML.replace(
        '<geom name="payload_geom" type="box" size="0.02 0.015 0.02"/>',
        '<geom name="payload_geom" type="box" size="0.02 0.015 0.005"/>',
    )
    model = mujoco.MjModel.from_xml_string(xml)

    class _FlatSim(_GraspSim):
        def register_graspables(self) -> None:
            # The bounding box claims 7 cm half height, the body has 0.5 cm -- exactly the situation of the LYING
            # marker, whose AABB in the body frame carries its length as its height.
            self.register_graspable("payload", "payload_free", self._body_id("payload"), np.array([0.02, 0.015, 0.07]))

    sim = _FlatSim(
        model,
        SPEC,
        scene_prefix="",
        default_span=0.04,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )
    entry = sim._graspable["payload"]
    ref = sim._grip_reference(entry)
    top = float(sim.data.xpos[entry["body"]][2]) + 0.005
    # The reference point must lie near the body, not 50 mm above it -- that is how far it wandered when it was
    # computed from the upright bounding box (half height 70 mm).
    assert abs(float(ref[2]) - top) < 0.03, (
        f"Greifpunkt bei {float(ref[2]):.3f}, echte Oberkante {top:.3f} -- "
        f"er folgt der aufrechten Huelle statt dem Koerper"
    )
    # ...and then the span measurement finds something there as well.
    assert sim._span_between_pads(entry, ref, 0.0) is not None


def test_a_tipped_object_offers_the_axes_it_really_has():
    """The grasp axes come from the horizontal body axes.

    ``_try_grasp`` formed them from ``obj_yaw`` and ``obj_yaw + 90 degrees``
    -- that presupposes that the z axis of the body stands VERTICAL.  For the
    lying marker its length lies horizontal, and both candidates then point
    past the graspable direction.  For a body standing upright the computation
    coincides with the old one, so the cube tower does not move.
    """
    sim = _build()
    # upright: x and y are horizontal ─▶ exactly the old two axes
    axes = sim._horizontal_axes(sim._graspable["payload"])
    assert len(axes) == 2
    assert min(abs(_wrap_half(a)) for a in axes) < 1e-6

    # tilted 90 degrees about x: now x and z are horizontal
    _set_payload_tilt(sim, np.radians(90.0))
    tipped = sim._horizontal_axes(sim._graspable["payload"])
    assert len(tipped) == 2, f"{len(tipped)} Achsen bei gekipptem Koerper"


# --------------------------------------------------------------------- #
# The straightening has a LIMIT instead of being a repair
#
# Belief ─▶ action ─▶ change of the world ─▶ read back as "truth" ─▶ belief.
# Causally legitimate (a real gripper really does straighten a pen when it
# grabs it), but modelled as a FREE snap the twin turns abstraction errors
# into nothing: the coarse rung looks sufficient because the twin itself
# repaired the consequence of its coarseness.
#
# Compliance is real, but limited: flat pads straighten a slightly crooked
# body, a strongly crooked one slips out.
#
# HONESTLY THOUGH: the measured distribution over 68 container runs ranges
# from 0.0 to 2.1 degrees -- the limit binds NOWHERE today.  It is a guard
# against a failure mode, not a correction of a result.
# --------------------------------------------------------------------- #
def test_a_slightly_tipped_object_is_still_squared_and_grasped():
    from twinlink.task_sim import PAD_SQUARE_LIMIT_DEG

    sim = _build()
    _set_payload_tilt(sim, np.radians(PAD_SQUARE_LIMIT_DEG - 5.0))
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload", (
        "leichte Neigung muss weiter ausgeglichen werden -- " "Nachgiebigkeit ist echt"
    )


def test_the_limit_stays_inside_the_geometric_capture_window():
    """A latch outside the capture window would have no effect."""
    from twinlink.task_sim import GRASP_MAX_MISALIGN_DEG, PAD_SQUARE_LIMIT_DEG

    assert PAD_SQUARE_LIMIT_DEG < GRASP_MAX_MISALIGN_DEG


def test_the_yaw_path_keeps_its_long_standing_tolerance():
    """The YAW ANGLE stays as it was -- and for a reason.

    Tried with 15 degrees: seven existing tests red, among them the golden trace of the cube tower.  The path is old,
    pinned and measured in the study at at most 2.1 degrees; tightening it would change results for a reason that has
    nothing to do with the question.  What gets limited is the NEW path, which had no limit at all.
    """
    from twinlink.task_sim import GRASP_MAX_MISALIGN_DEG

    sim = _build()
    _set_payload_yaw(sim, np.radians(GRASP_MAX_MISALIGN_DEG - 2.0))
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"


def test_a_badly_tipped_object_is_refused_too():
    """The tilt is the longer way -- it was snapped to the nearest
    axis-parallel attitude with no limit at all."""
    from twinlink.task_sim import PAD_SQUARE_LIMIT_DEG

    sim = _build()
    _set_payload_tilt(sim, np.radians(PAD_SQUARE_LIMIT_DEG + 10.0))
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() is None, f"{PAD_SQUARE_LIMIT_DEG + 10:.0f} Grad Neigung wurden weggeschnappt"


# --------------------------------------------------------------------- #
# The TRANSPORT is checked against reality
#
# The carried body has no contacts (for good reason, see
# ``_suspend_object_contacts``), MoveIt checks the BELIEVED body, and the REAL
# one travels unhindered through the real world.  The direction is the
# dangerous one: it is not reported as "reality forbids what belief allows",
# it is counted as a SUCCESS.
#
# The check goes through ``mj_geomDistance``, as it does for the grasp.
# --------------------------------------------------------------------- #
def _with_wall(x: float):
    xml = SCENE_XML.replace(
        "</worldbody>",
        f'<body name="wand" pos="{x} 0 0.08">'
        '<geom name="wand_geom" type="box" size="0.05 0.05 0.05"/>'
        "</body></worldbody>",
    )
    model = mujoco.MjModel.from_xml_string(xml)
    return _GraspSim(
        model,
        SPEC,
        scene_prefix="",
        default_span=0.04,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def test_a_carried_object_clear_of_the_world_reports_a_positive_gap():
    sim = _with_wall(1.4)  # weit weg
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    spalt = sim.carried_world_gap()
    assert spalt is not None and spalt > 0.0, (
        f"frei getragen, aber Abstand {spalt} -- die Pruefung sieht die " f"Welt nicht"
    )


def test_a_carried_object_driven_into_the_world_reports_no_gap_left():
    """The actual case: the real body travels through real stuff.

    Without this assurance the check could simply always report positive, and the test above would still pass.
    """
    # The wall stands ASIDE -- otherwise two overlapping bodies would physically push each other apart before the
    # grasp, and the setup would be unphysical.  What is driven into it is only the CARRIED body, whose contacts are
    # off: exactly the case nobody notices.
    sim = _with_wall(0.80)
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    sim.set_arm_command({"arm_0_slide": 0.63})  # TCP ─▶ 0.80, in die Wand
    sim.step_physics(60)
    assert sim.carried_world_gap() <= 0.0, (
        "der getragene Koerper steckt in der Wand und die Pruefung " "meldet freien Raum"
    )


def test_without_a_carry_there_is_nothing_to_report():
    sim = _build()
    assert sim.carried_world_gap() is None


def test_the_worst_moment_of_the_carry_is_remembered():
    """A distance at the end says nothing about the journey in between.

    The carried body may have driven through an obstacle halfway along and stand free again at the goal -- what has to
    be reported is the WORST moment, not the last one.
    """
    sim = _with_wall(0.80)
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    sim.set_arm_command({"arm_0_slide": 0.63})  # durch die Wand
    sim.step_physics(60)
    sim.set_arm_command({"arm_0_slide": 0.2})  # wieder heraus
    sim.step_physics(60)
    assert sim.carried_world_gap() > 0.0, "am Ende steht er frei"
    assert sim.carried_world_gap_min() <= 0.0, (
        f"schlechtester Moment {sim.carried_world_gap_min()} -- die Fahrt " f"durch die Wand ist vergessen"
    )


def test_without_a_carry_there_is_no_worst_moment():
    sim = _build()
    assert sim.carried_world_gap_min() is None


# --------------------------------------------------------------------- #
# Rotational symmetry: a roll about its own axis is NOT a misalignment
# --------------------------------------------------------------------- #
CYLINDER_XML = SCENE_XML.replace(
    '<geom name="payload_geom" type="box" size="0.02 0.015 0.02"/>',
    # ``zaxis`` puts the cylinder axis onto world y -- the body LIES DOWN.  ``euler`` would be a trap here: MuJoCo
    # reads it in DEGREES, "1.5708" would have tilted the cylinder by 1.6 degrees instead of laying it down.
    '<geom name="payload_geom" type="cylinder" size="0.015 0.05"' ' zaxis="0 1 0"/>',
)


class _CylinderSim(TwinTaskSim):
    def register_graspables(self) -> None:
        self.register_graspable("payload", "payload_free", self._body_id("payload"), np.array([0.015, 0.05, 0.015]))


def _cylinder() -> _CylinderSim:
    model = mujoco.MjModel.from_xml_string(CYLINDER_XML)
    return _CylinderSim(
        model,
        SPEC,
        scene_prefix="",
        default_span=0.04,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def _roles(sim, deg: float) -> None:
    """Rotate the LYING cylinder about its OWN axis (world y)."""
    w = np.radians(deg) / 2.0
    adr = sim._graspable["payload"]["qpos"]
    sim.data.qpos[adr + 3 : adr + 7] = np.array([np.cos(w), 0.0, np.sin(w), 0.0])
    sim._mujoco.mj_forward(sim.model, sim.data)


def test_a_lying_cylinder_rolled_about_its_own_axis_is_still_graspable():
    """A body of revolution has no attitude about its own axis.

    Measured on 2026-08-17 in the sufficiency pre-study: ``marker/pick`` failed on ALL object rungs with "zu stark
    geneigt".  The marker lay correctly (cylinder axis horizontal, world component 0.012) but was rolled by 39.3 degrees
    about its OWN axis -- a symmetry, not a misalignment.  ``_square_tilt`` read the discrete body axes, which is right
    for a box and meaningless for a cylinder.
    """
    sim = _cylinder()
    adr = sim._graspable["payload"]["qpos"]
    for deg in (0.0, 39.3, 75.0):
        _roles(sim, deg)
        assert sim._square_tilt(adr, sim._graspable["payload"]), (
            f"Roll um {deg} Grad um die eigene Achse als Schieflage " f"abgelehnt -- der Koerper ist darum symmetrisch"
        )


# --------------------------------------------------------------------- #
# Visible gripper ramp (for recordings, not for the measurement)
# --------------------------------------------------------------------- #
def _with_ramp(ticks: int):
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return _GraspSim(
        model,
        SPEC,
        scene_prefix="",
        default_span=0.04,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
        gripper_ramp_ticks=ticks,
    )


def test_without_a_ramp_the_fingers_are_at_their_target_after_one_tick():
    """The existing path -- every measurement of the study hangs on it."""
    sim = _with_ramp(0)
    sim.command_gripper(close=True)
    sim.step_physics(1)
    assert sim.gripper_angle_applied() == pytest.approx(sim._gripper_command)


def test_a_ramp_moves_the_fingers_through_intermediate_angles():
    """For a video the closing has to be VISIBLE.

    Without a ramp ``step_physics`` writes the gripper joints to the target value in the first substep -- in the picture
    the hand jumps open and shut binarily.  The ramp changes only the WAY there, not the target.
    """
    sim = _with_ramp(10)
    open = sim.gripper_angle_applied()
    sim.command_gripper(close=True)
    goal = sim._gripper_command
    between = []
    for _ in range(5):
        sim.step_physics(1)
        between.append(sim.gripper_angle_applied())
    assert all(min(open, goal) <= w <= max(open, goal) for w in between)
    assert len(set(round(w, 6) for w in between)) > 1, f"Winkel bleibt stehen: {between} -- keine sichtbare Bewegung"
    assert between[-1] != pytest.approx(goal), "nach der halben Rampe schon da"
    sim.step_physics(6)
    assert sim.gripper_angle_applied() == pytest.approx(goal), "Ziel nicht erreicht"
