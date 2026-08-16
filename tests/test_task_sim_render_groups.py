"""Regression pin: only the ROBOT's collision shells leave the render groups.

``TwinTaskSim._hide_robot_collision_geoms`` used to decide by exclusion --
"a body whose name does not start with the app's ``scene_prefix`` is robot".
An app that places a body under a name of its own (the sufficiency study's
``task_object``) therefore had it moved into the hidden geom group, which
``_ensure_renderer`` switches off for RGB *and* depth: the object was
invisible to every camera while every pose query kept returning the truth.
Nothing failed -- the cameras simply saw the surface behind it.

The rule is positive now (``body_rootid`` == the robot's root), and this test
holds it there.  Robot- and task-free MJCF, no URDF bundle needed.
"""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.testing import StraightLinkage  # noqa: E402
from twinlink.task_sim import RobotSimSpec, TwinTaskSim  # noqa: E402

#: A slider "arm" that carries BOTH a collision shell and a visual-only mesh
#: stand-in (the split the hiding exists for), plus two non-robot bodies: one
#: under the scene prefix, one under a name of the app's own choosing.
SCENE_XML = """
<mujoco model="render_groups">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="twinlink_ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="arm_0_shoulder_link" pos="0 0 0.30">
      <joint name="arm_0_slide" type="slide" axis="1 0 0" range="-2 2"/>
      <geom name="arm_0_collision" type="box" size="0.05 0.05 0.05"/>
      <geom name="arm_0_visual" type="box" size="0.04 0.04 0.04"
            contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
      <body name="rg6_base" pos="0.12 0 0">
        <geom name="rg6_collision" type="box" size="0.04 0.04 0.04"/>
        <body name="rg6_hand_tcp" pos="0.05 0 0">
          <geom name="rg6_tcp_marker" type="sphere" size="0.005"
                contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
    <body name="hrl_table" pos="0.6 0 0.1">
      <geom name="hrl_table_top" type="box" size="0.2 0.2 0.1"/>
    </body>
    <body name="task_object" pos="0.6 0 0.25">
      <freejoint name="task_object_free"/>
      <geom name="task_object_geom0" type="box" size="0.03 0.03 0.03"/>
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


class _ProbeSim(TwinTaskSim):
    def register_graspables(self) -> None:
        self.register_graspable(
            "task_object", "task_object_free",
            self._body_id("task_object"), np.full(3, 0.03),
        )

    def support_geom_names(self):
        return frozenset({"hrl_table_top"})


def _build() -> _ProbeSim:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return _ProbeSim(
        model,
        SPEC,
        scene_prefix="hrl_",
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def _group(sim, geom: str) -> int:
    gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    assert gid >= 0, f"geom {geom!r} not in model"
    return int(sim.model.geom_group[gid])


def test_the_robot_collision_shells_leave_the_render_groups():
    sim = _build()
    try:
        assert _group(sim, "arm_0_collision") == sim._HIDDEN_GEOM_GROUP
        assert _group(sim, "rg6_collision") == sim._HIDDEN_GEOM_GROUP
    finally:
        sim.close()


def test_the_robot_visual_geometry_stays_visible():
    sim = _build()
    try:
        assert _group(sim, "arm_0_visual") != sim._HIDDEN_GEOM_GROUP
        assert _group(sim, "rg6_tcp_marker") != sim._HIDDEN_GEOM_GROUP
    finally:
        sim.close()


def test_a_body_outside_the_robot_stays_visible_whatever_it_is_called():
    """The regression: furniture under the scene prefix was already safe, an
    app's own task body was not -- and nothing in the sim said so."""
    sim = _build()
    try:
        assert _group(sim, "hrl_table_top") != sim._HIDDEN_GEOM_GROUP
        assert _group(sim, "task_object_geom0") != sim._HIDDEN_GEOM_GROUP
        assert _group(sim, "twinlink_ground") != sim._HIDDEN_GEOM_GROUP
    finally:
        sim.close()
