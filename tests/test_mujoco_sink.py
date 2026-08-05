"""MujocoSink no longer hard-codes one robot's body names.

Task 11 (hrl-Extraktion): before this, ``_guess_lookat``, the two ghost-sphere
overlays and ``_body_for_frame`` all matched on literal UR5/RG6/D435 body
names (``arm_0*``, ``camera_0_link``).  These are read from an optional
``RobotSimSpec`` now; without one (every pre-Task-11 caller --
octomap_explorer, the spact-integration-demos scripts/notebooks, hrl's own
dashboard) the exact same literals apply as defaults, so nothing downstream
has to change.

This module only imports ``numpy`` at collection time (mujoco/opencv stay
lazy inside methods), so these tests do not need the ``mujoco`` extra.
"""
from __future__ import annotations

import inspect

from twinlink.sinks import mujoco_sink
from twinlink.task_sim import RobotSimSpec


def test_mujoco_sink_takes_robot_names_from_spec():
    """The source may keep at most one bare "arm_0" literal (the documented
    pre-spec default) -- every other manipulator-body reference must route
    through ``spec``."""
    src = inspect.getsource(mujoco_sink)
    assert src.count('"arm_0') <= 1
    assert "camera_0_link" not in src or "spec" in src


def test_spec_defaults_to_none_and_keeps_todays_literals():
    sink = mujoco_sink.MujocoSink.__new__(mujoco_sink.MujocoSink)
    sink.spec = None
    assert sink._manipulator_prefixes() == ("arm_0",)
    assert sink._camera_link_candidates() == ("camera_0_link", "camera_0_bottom_screw_frame")


def test_spec_overrides_the_manipulator_prefixes():
    spec = RobotSimSpec(
        manipulator_prefixes=("other_arm", "other_hand"),
        hand_prefixes=("other_hand",),
        gripper_prefixes=("other_hand",),
        far_arm_bodies=("other_arm_base",),
        gripper_stroke_m=0.1,
        tcp_body="other_tcp",
        arm_joints=("other_j1",),
    )
    sink = mujoco_sink.MujocoSink.__new__(mujoco_sink.MujocoSink)
    sink.spec = spec
    assert sink._manipulator_prefixes() == ("other_arm", "other_hand")


def test_spec_overrides_the_camera_link_candidates():
    """hand_prefixes drives the wrist-camera fallback: it is documented as
    "gripper + wrist-mounted sensor" (RobotSimSpec docstring), which is
    exactly what _body_for_frame needs when a cloud's optical frame is not a
    model body."""
    spec = RobotSimSpec(
        manipulator_prefixes=("other_arm", "other_cam"),
        hand_prefixes=("other_grip", "other_cam"),
        gripper_prefixes=("other_grip",),
        far_arm_bodies=(),
        gripper_stroke_m=0.1,
        tcp_body="other_tcp",
        arm_joints=(),
    )
    sink = mujoco_sink.MujocoSink.__new__(mujoco_sink.MujocoSink)
    sink.spec = spec
    assert sink._camera_link_candidates() == (
        "other_grip_link",
        "other_grip_bottom_screw_frame",
        "other_cam_link",
        "other_cam_bottom_screw_frame",
    )
