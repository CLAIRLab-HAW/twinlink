"""Regression pin: only the gripper shells become permeable, not the whole hand.

Carried over from Task 10 (2026-07-31): ``TwinTaskSim._setup_collision_masks``
must key off ``spec.gripper_prefixes`` -- the jaws/housing alone -- not
``spec.hand_prefixes`` (gripper + wrist camera).  Swapping the two back is a
one-word typo that leaves every suite green (nothing exercised the wrist
camera's contact mask), yet it silently disables ``robot_obstacle_collision``
for the camera housing: the fix that motivated this test cost a full
debugging round because of exactly that blind spot.

The scene is a tiny, robot- and task-free MJCF model (no URDF bundle needed --
runs in CI), built the same way ``test_task_sim_clutter.py`` builds its scene.
"""
from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.task_sim import RobotSimSpec, TwinTaskSim  # noqa: E402

#: A slider "arm" carrying a gripper body, which in turn carries a wrist
#: camera body -- the same nesting as the real a200-0553 profile (RG6 with a
#: D435 mounted on it), just renamed to keep the scene self-contained.
SCENE_XML = """
<mujoco model="camera_contact_mask">
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
        <body name="camera_0_link" pos="0.02 0.02 0">
          <geom name="camera_0_geom" type="box" size="0.01 0.01 0.01"/>
        </body>
        <body name="rg6_hand_tcp" pos="0.05 0 0">
          <geom name="rg6_tcp_marker" type="sphere" size="0.005"
                contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

#: hand_prefixes deliberately includes "camera_0" (gripper + wrist sensor, per
#: the RobotSimSpec docstring) while gripper_prefixes does NOT -- exactly the
#: real a200-0553 profile's shape (rg6 + camera_0 vs. rg6 alone).
SPEC = RobotSimSpec(
    manipulator_prefixes=("arm_0", "rg6", "camera_0"),
    hand_prefixes=("rg6", "camera_0"),
    gripper_prefixes=("rg6",),
    far_arm_bodies=("arm_0_shoulder_link",),
    gripper_stroke_m=0.156,
    tcp_body="rg6_hand_tcp",
    arm_joints=("arm_0_slide",),
)


def _build() -> TwinTaskSim:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return TwinTaskSim(
        model,
        SPEC,
        scene_prefix="",
        gripper_follower_factors={},
        gripper_open=0.0,
        gripper_closed=0.6,
        home_pose={"arm_0_slide": 0.0},
    )


def _first_geom_of_body(model, body_id: int) -> int:
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == body_id:
            return gid
    raise AssertionError(f"body {body_id} has no geom")


def test_only_the_gripper_shell_becomes_permeable_the_wrist_camera_does_not():
    sim = _build()
    try:
        grip_gid = _first_geom_of_body(sim.model, sim._body_id("rg6_base"))
        cam_gid = _first_geom_of_body(sim.model, sim._body_id("camera_0_link"))

        # The gripper shell is made permeable to graspables (see
        # TwinTaskSim._setup_collision_masks): contype=4, conaffinity=1.
        assert int(sim.model.geom_contype[grip_gid]) == 4
        assert int(sim.model.geom_conaffinity[grip_gid]) == 1

        # The wrist camera rides on the same hand assembly (it IS in
        # hand_prefixes) but is not a jaw and must keep its ordinary robot
        # contact mask (untouched XML defaults: contype=1, conaffinity=1) --
        # otherwise it goes blind to obstacle contacts.  This is exactly what
        # regresses if _setup_collision_masks is changed to key off
        # spec.hand_prefixes instead of spec.gripper_prefixes.
        assert int(sim.model.geom_contype[cam_gid]) == 1
        assert int(sim.model.geom_conaffinity[cam_gid]) == 1
    finally:
        sim.close()
