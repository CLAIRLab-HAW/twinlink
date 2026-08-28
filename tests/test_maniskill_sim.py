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


def test_the_action_matches_the_environments_action_space(world_in_process):
    """One entry per action DIMENSION, not per joint.

    A mimic controller drives several joints from one value: measured 2026-08-28 on the RG6, whose six joints occupy
    exactly one dimension.  Building the vector per joint produced a 12-vector where the environment expected 7, and
    every step raised.
    """
    action = world_in_process._action_from_command()
    assert action.shape == world_in_process.env.action_space.shape


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


def test_the_bridge_can_command_the_gripper_on_the_ros_route(world_ros_route):
    """``command_gripper`` is a no-op there, ``apply_external_gripper`` is not.

    Measured 2026-08-28 against the running stack: the GripperCommand action was accepted, the jaws never moved and
    the reported width stayed put -- because the server called the no-op.  The planner must not command there (it
    would meet a bridge that refuses during a motion), but the bridge must.
    """
    driver = world_ros_route._driver_joint
    before = dict(world_ros_route._command)
    world_ros_route.command_gripper(True)
    assert world_ros_route._command == before, "command_gripper must stay a no-op on the ROS route"
    world_ros_route.apply_external_gripper(True)
    assert world_ros_route._command[driver] == pytest.approx(world_ros_route._linkage.closed_rad)


def test_a_bare_world_reports_no_contact_although_the_gripper_touches_itself(world_in_process):
    """Contact means contact with something that is NOT the robot.

    The RG6 is a four-bar linkage whose struts touch each other by construction.  Measured 2026-08-28 against the
    running stack, ``get_net_contact_forces`` reported 34 kN on a truss arm and 8 kN on the bracket with an EMPTY
    scene, and the collision monitor froze the plant on the gripper's own mechanism.  Pairwise against the scene's
    non-robot actors is what answers the question that was actually asked.
    """
    forces = world_in_process.contact_forces(world_in_process.monitored_links())
    assert forces, "the monitored links must still be reported, with zero force"
    assert max(forces.values()) == 0.0


def test_settling_to_home_actually_holds(world_in_process):
    """Writing ``qpos`` is not enough.

    The PD controller keeps the target it captured at reset and pulls straight back on the next step: measured
    2026-08-28, the world reported the URDF zero pose again a second after being placed at ``ready`` -- and that
    pose puts the open hand into the upper arm, which the reflex guard reported as a predicted self-collision
    every second.
    """
    world_in_process.settle_to_home()
    for _ in range(30):
        world_in_process.step_physics(1)
    at_home = world_in_process.arm_positions()
    for name, want in world_in_process._home_pose.items():
        if name in at_home:
            assert abs(at_home[name] - want) < 0.15, f"{name}: {at_home[name]:.3f} != {want:.3f}"


def test_a_world_without_a_task_states_no_verdict(world_in_process):
    """No verdict is not the same as a failed task, and the runner has to be able to tell them apart.

    The bridge's own world is ``Empty-v1``: it has nothing to succeed at.  Reading a bare ``False`` out of it as a
    task failure would book every channel test as a lost episode.
    """
    assert world_in_process.task_success() is False


def test_the_verdict_is_read_from_the_environment_not_recomputed(world_in_process):
    """One success predicate, in the place that defines the task -- a second one would be a second truth."""

    class _Verdict:
        def evaluate(self):
            return {"success": [True]}

    original = world_in_process.env
    world_in_process.env = _Verdict()
    try:
        assert world_in_process.task_success() is True
    finally:
        world_in_process.env = original
