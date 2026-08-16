"""Proximity-capture grasp mechanic: alignment, capture, carry, release.

Moved/re-derived from ``hrl.tests.test_grasp_alignment`` and
``hrl.tests.test_scene_sim`` (task-refactor 2026-07-31): the parallel-jaw
alignment check, kinematic carry and release all live in
``TwinTaskSim._try_grasp`` / ``_carry_grasped`` / ``_release`` -- none of it
is about cubes.  The original tests drove this mechanic indirectly through a
real UR5 descend (``ArmMotionPlanner`` + real cube geometry); here the same
production methods (``command_gripper``, ``arm_positions`` via
``set_arm_command`` + ``step_physics``) are exercised directly against a
tiny, robot- and task-free MJCF model, the same pattern
``test_task_sim_clutter.py`` already established in this file.  Gravity is
off (``gravity="0 0 0"``): nothing here is about resting/settling physics
(that stays covered task-side, e.g. hrl's ``test_cubes_rest_stably``), only
about the capture/carry/release state machine.
"""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.task_sim import (  # noqa: E402
    GRASP_MAX_MISALIGN_DEG,
    RobotSimSpec,
    TwinTaskSim,
    _wrap_quarter,
)

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
        self.register_graspable(
            "payload", "payload_free", self._body_id("payload"),
            np.array([0.02, 0.015, 0.02]),
        )


def _build() -> _GraspSim:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return _GraspSim(
        model,
        SPEC,
        scene_prefix="",
        default_span=0.04,
        gripper_follower_factors={},
        gripper_open=0.0,
        gripper_closed=0.6,
        home_pose={"arm_0_slide": 0.0},
    )


def _set_payload_yaw(sim: _GraspSim, yaw_rad: float) -> None:
    adr = sim._graspable["payload"]["qpos"]
    sim.data.qpos[adr + 3 : adr + 7] = np.array(
        [np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)]
    )
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
    """Pads at 90deg (fixed), payload at 90+35deg -> jaws meet edges, no grasp."""
    sim = _build()
    try:
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG + 35))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() is None
    finally:
        sim.close()


def test_aligned_payload_is_captured():
    """Payload yaw exactly matches the pad axis -> captured."""
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
# Greiferweite, oeffentlich lesbar
#
# Der Zwilling faehrt die Finger seit jeher auf die OBJEKTWEITE, wenn ein
# Objekt gefasst ist (``command_gripper``: "the finger command corresponds
# to the object width").  Diese Weite steckte bisher nur im privaten
# ``_gripper_command``.  Wer sie an einen echten Greifer spiegeln will --
# damit move_group dieselbe Hand prueft, die der Zwilling zeigt -- kann sie
# nicht lesen, ohne in die Interna zu greifen.  Am 2026-08-16 im Container
# gemessen: dessen RG6-Gelenke standen waehrend eines ganzen Zellenlaufs
# konstant auf 0.0 (voll offen), waehrend der Zwilling zugriff -- zwei
# verschiedene Haende, gegen die geprueft wurde.
# --------------------------------------------------------------------- #
def test_the_open_gripper_reports_the_full_stroke():
    sim = _build()
    sim.command_gripper(False)
    assert sim.gripper_width_m() == pytest.approx(SPEC.gripper_stroke_m)


def test_the_empty_closed_gripper_reports_zero_width():
    sim = _build()
    sim.command_gripper(True)   # nichts zu fassen -> ganz zu
    assert sim.gripper_width_m() == pytest.approx(0.0, abs=1e-9)


def test_a_grasped_object_sets_the_width_to_its_own_span():
    """Die eine Zahl, um derentwillen der Zugang existiert.

    Sie folgt dem OBJEKT, nicht einer Vorgabe: der Payload misst
    0,04 x 0,03 x 0,04 m, die Pads schliessen ueber seine 0,03-m-Kante --
    und NICHT auf ``default_span`` (0,04), das nur gilt, wo kein Objekt
    vermessen wurde.  Genau diese Objektabhaengigkeit ist es, die an
    einem Greifer mit fester Oeffnung verloren geht.
    """
    sim = _build()
    _approach(sim)
    sim.command_gripper(True)
    assert sim.grasped_label() == "payload", "ohne Griff prueft der Test nichts"
    assert sim.gripper_width_m() == pytest.approx(0.03, abs=2e-3)
    assert sim.gripper_width_m() != pytest.approx(0.04, abs=2e-3), (
        "die Weite haengt an der Vorgabe statt am Objekt -- dann misst sie nichts")
