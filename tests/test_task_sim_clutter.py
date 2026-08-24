"""Regression guard: graspable obstacles stay obstacles.

Background: if ``arm_config_collides`` parks *all* registered graspables out of the scratch model -- including those
that are classified as obstacles at the same time (pool slots, scripted clutter that a task declares to be a grasp
target) -- then the gate never sees exactly the objects it exists for; conversely, ``settle`` waits for scene clutter
that is not part of the payload at all.  No suite notices that by itself.

The test builds a tiny, robot- and task-free MJCF model -- no URDF bundle needed, so it also runs in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.mjcf_scene import distractor_body_name, distractor_joint_name  # noqa: E402
from twinlink.testing import StraightLinkage  # noqa: E402
from twinlink.task_sim import RobotSimSpec, TwinTaskSim  # noqa: E402

#: The body prefix of THIS test scene.  Scene and constructor must make the
#: same choice: if the scene names its distractor with the module default
#: (``hrl_distractor_0``) while the constructor is given ``scene_prefix=""``,
#: that only goes unnoticed as long as the classification ignores the
#: constructor prefix.  An own, app-foreign prefix is the more honest choice
#: for a twinlink test anyway: this scene FURNISHES itself, it is not
#: prefix-less.
PREFIX = "test_"

#: A slider "arm" with a gripper child body, a platform, one graspable
#: obstacle (distractor prefix) in +x and a pure payload in -x.
SCENE_XML = f"""
<mujoco model="clutter_gate">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="twinlink_ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="platform_base" pos="0 -1 0.1">
      <geom name="platform_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
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
    <body name="{distractor_body_name(0, prefix=PREFIX)}" pos="0.6 0 0.05">
      <freejoint name="{distractor_joint_name(0, prefix=PREFIX)}"/>
      <geom name="{distractor_body_name(0, prefix=PREFIX)}_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="payload" pos="-0.5 0 0.02">
      <freejoint name="payload_free"/>
      <geom name="payload_geom" type="box" size="0.04 0.04 0.02"/>
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

#: Slider position at which the arm body drives into the graspable distractor.
INTO_CLUTTER = 0.55
#: Slider position at which the arm body drives into the pure payload.
INTO_PAYLOAD = -0.5


class _ClutterSim(TwinTaskSim):
    """Registers both free bodies as graspable -- like a clear task."""

    def register_graspables(self) -> None:
        self.register_graspable(
            "clutter",
            distractor_joint_name(0, prefix=PREFIX),
            self._body_id(distractor_body_name(0, prefix=PREFIX)),
            np.full(3, 0.05),
        )
        self.register_graspable("payload", "payload_free", self._body_id("payload"), np.array([0.04, 0.04, 0.02]))


def _build() -> _ClutterSim:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return _ClutterSim(
        model,
        SPEC,
        scene_prefix=PREFIX,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def test_clutter_is_classified_as_obstacle_although_it_is_graspable():
    sim = _build()
    try:
        assert sim._graspable.keys() == {"clutter", "payload"}
        assert sim._non_obstacle_graspables == ("payload",)
        clutter_geoms = set(sim._graspable["clutter"]["geoms"])
        assert clutter_geoms <= sim._obstacle_geoms
    finally:
        sim.close()


def test_gate_rejects_a_configuration_reaching_into_graspable_clutter():
    """The core of the regression: graspable does not mean invisible to the gate."""
    sim = _build()
    try:
        assert sim.arm_config_collides({"arm_0_slide": INTO_CLUTTER}) is True
        # ... and also over the pure obstacle path (pose pre-check).
        assert sim.arm_config_collides({"arm_0_slide": INTO_CLUTTER}, obstacles_only=True) is True
        assert sim.arm_config_collides({"arm_0_slide": 0.0}) is False

        # Counter-check = the regression: parking ALL graspables away would make exactly this verdict disappear.
        sim._non_obstacle_graspables = tuple(sim._graspable)
        assert sim.arm_config_collides({"arm_0_slide": INTO_CLUTTER}) is False
    finally:
        sim.close()


def test_gate_still_ignores_the_sims_own_payload():
    """The other half of the old semantics: payload is no reason for validity.

    The payload is parked away AND belongs to none of the three checked pair classes anyway -- the two together keep the
    grasp above one's own object valid (when grasping, the gripper inevitably lowers into it).
    """
    sim = _build()
    try:
        assert sim.arm_config_collides({"arm_0_slide": INTO_PAYLOAD}) is False
        assert "payload" in sim._non_obstacle_graspables
    finally:
        sim.close()


def _settle_ticks(sim, max_ticks: int) -> int:
    start = float(sim.data.time)
    sim.settle(max_ticks=max_ticks)
    return round((float(sim.data.time) - start) / sim.control_dt)


def test_settle_does_not_wait_for_scene_clutter():
    sim = _build()
    try:
        dof = sim._graspable["clutter"]["dof"]
        sim.data.qvel[dof : dof + 3] = 5.0  # das Clutter-Objekt fliegt
        assert _settle_ticks(sim, 40) <= 3
    finally:
        sim.close()

    regression = _build()
    try:
        dof = regression._graspable["clutter"]["dof"]
        regression.data.qvel[dof : dof + 3] = 5.0
        # Die Split-Semantik (alle Greifbaren) lief bis ans Limit.
        regression._non_obstacle_graspables = tuple(regression._graspable)
        assert _settle_ticks(regression, 40) == 40
    finally:
        regression.close()
