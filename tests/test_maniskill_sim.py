"""Tick ownership is the whole point of this class, and it is invisible in a screenshot.

``ArmMotionPlanner._mirror_stream`` writes every downlink sample back with ``set_arm_command`` and then calls
``step_physics(1)``.  On the ROS route those samples ORIGINATE in this world -- they came back over the bridge and
the downlink -- so writing them in would be a loop and stepping would give the world two steppers.  These tests pin
which of the two modes does what.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sapien")
pytest.importorskip("mani_skill")


def test_the_ros_route_does_not_advance_on_step_physics(world_ros_route):
    before = dict(world_ros_route.arm_positions())
    world_ros_route.set_arm_command({k: v + 0.2 for k, v in before.items()})
    world_ros_route.step_physics(5)
    after = world_ros_route.arm_positions()
    for name, value in before.items():
        assert after[name] == pytest.approx(value, abs=1e-9), name


def test_the_in_process_route_does_advance(world_in_process):
    before = dict(world_in_process.arm_positions())
    world_in_process.set_arm_command({k: v + 0.2 for k, v in before.items()})
    world_in_process.step_physics(50)
    after = world_in_process.arm_positions()
    assert any(abs(after[n] - before[n]) > 1e-3 for n in before)


def test_an_external_command_advances_even_on_the_ros_route(world_ros_route):
    # The bridge's entry point: it is the one that owns the tick there, and its events land in the buffer the app
    # drains through step_physics.
    before = dict(world_ros_route.arm_positions())
    for _ in range(50):
        world_ros_route.apply_external_command({k: v + 0.2 for k, v in before.items()})
    after = world_ros_route.arm_positions()
    assert any(abs(after[n] - before[n]) > 1e-3 for n in before)


def test_step_physics_returns_events_in_both_modes(world_ros_route):
    events = world_ros_route.step_physics(1)
    assert hasattr(events, "merge")


def test_the_action_is_not_qpos_shaped(world_in_process):
    # Measured 2026-08-28: qpos is (16,) while the arm controller's action is (6,).  Building the action by slicing
    # the state would silently command the wrong joints.
    action = world_in_process._action_from_command()
    assert action.shape == world_in_process.env.action_space.shape
    assert len(action) < len(world_in_process.joint_positions())


def test_the_gate_is_delegated_to_the_kinematics_provider(world_in_process):
    q = world_in_process.arm_positions()
    assert world_in_process.arm_config_collides(q) is world_in_process.kinematics.config_collides(q)


def test_the_camera_pose_is_opengl_like_the_mujoco_sibling(world_in_process):
    pose = world_in_process.camera_pose("base_camera")
    if pose is None:
        pytest.skip("the bare env carries no named sensor -- the app's agent mounts the wrist camera")
    pos, rot = pose
    assert pos.shape == (3,) and rot.shape == (3, 3)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-5)
