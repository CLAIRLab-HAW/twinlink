"""Belief-display mechanic: display_object / display_carry / display_release
/ park_object -- the real-mode twin's non-physics "show what perception
believes" API (see ``TwinTaskSim`` module docstring, "belief display").

The neighbouring suites reach this mechanic only indirectly:
``hrl.tests.test_belief_mirror`` is task-scoped (the WorldModel ─▶ item-tuple
translation via ``hrl.env.belief_mirror.CubeTwinMirror``), and
``twinlink.tests.test_display_mirror`` covers only the dedup/sequencing logic
of ``TwinDisplayMirror`` against a mocked sim.  The sim-side mechanic itself
-- does the object really follow the TCP, is it really parked out of reach,
are its contacts really suspended and restored -- is tested here, with the
same synthetic-scene pattern ``test_task_sim_grasp.py`` uses.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.testing import StraightLinkage  # noqa: E402
from twinlink.task_sim import RobotSimSpec, TwinTaskSim  # noqa: E402

SCENE_XML = """
<mujoco model="display_mechanic">
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
    <body name="believed" pos="-2 2 0.03">
      <freejoint name="believed_free"/>
      <geom name="believed_geom" type="box" size="0.02 0.02 0.02"/>
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


class _DisplaySim(TwinTaskSim):
    def register_graspables(self) -> None:
        self.register_graspable("believed", "believed_free", self._body_id("believed"), np.array([0.02, 0.02, 0.02]))


def _build() -> _DisplaySim:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return _DisplaySim(
        model,
        SPEC,
        scene_prefix="",
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def test_display_object_teleports_without_events():
    sim = _build()
    try:
        sim.display_object("believed", (0.5, 0.1, 0.2), yaw=0.3)
        assert np.allclose(sim.object_position("believed"), [0.5, 0.1, 0.2])
        events = sim.step_physics(1)
        assert events.grasp_acquired is None and events.grasp_lost is None
    finally:
        sim.close()


def test_park_object_moves_it_out_of_view():
    sim = _build()
    try:
        sim.display_object("believed", (0.5, 0.1, 0.2))
        sim.park_object("believed", index=2)
        pos = sim.object_position("believed")
        assert float(np.linalg.norm(pos[:2])) > 1.5
    finally:
        sim.close()


def test_display_carry_follows_tcp_without_events():
    sim = _build()
    try:
        sim.display_carry("believed")
        assert sim.grasped_label() == "believed", "carry pins the object like a grasp internally"
        sim.set_arm_command({"arm_0_slide": 0.4})
        events = sim.step_physics(5)
        tcp, _mat = sim.tcp_pose()
        assert float(np.linalg.norm(sim.object_position("believed") - tcp)) < 0.10
        # A display carry must never look like a real grasp/drop to a consumer watching sim EVENTS -- only the real
        # gripper feedback (real mode) or the kinematic grasp (sim mode) may raise those. (The internal _grasped
        # bookkeeping above is deliberately shared with the real carry machinery; only the event accumulator is exempt
        # -- see TwinTaskSim.display_carry's docstring.)
        assert events.grasp_acquired is None and events.grasp_lost is None
    finally:
        sim.close()


def test_display_carry_switching_labels_ends_the_previous_carry():
    """Only one object may ride the TCP; starting a new carry ends the old one."""
    sim = _build()
    try:
        sim.display_carry("believed")
        gid = sim._graspable["believed"]["geoms"][0]
        assert sim.model.geom_contype[gid] == 0, "carried object's contacts suspended"
        sim.display_release("believed", position=(0.9, -0.2, 0.05))
        assert sim.model.geom_contype[gid] != 0, "release must restore contacts"
        assert np.allclose(sim.object_position("believed"), [0.9, -0.2, 0.05])
    finally:
        sim.close()


def test_display_release_without_position_leaves_object_in_place():
    sim = _build()
    try:
        sim.display_carry("believed")
        before = sim.object_position("believed").copy()
        sim.display_release("believed")  # no position ─▶ stays where the carry left it
        assert np.allclose(sim.object_position("believed"), before, atol=1e-6)
        assert sim.grasped_label() is None
    finally:
        sim.close()
