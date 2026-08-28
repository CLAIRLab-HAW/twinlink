"""The oracle reads the SAME SRDF ``move_group`` loads -- that is the whole point of choosing Pinocchio.

The numbers are the ones measured on 2026-08-28 (M4, pin 4.1.0) against the URDF bundle and the container's
``/clearpath/robot.srdf``.  They are pinned so that a changed URDF or a changed SRDF is a test failure and not a
silently different collision gate.
"""

from __future__ import annotations

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from twinlink.pin_kinematics import BeliefBox, PinocchioKinematics  # noqa: E402

ARM = (
    "arm_0_shoulder_pan_joint",
    "arm_0_shoulder_lift_joint",
    "arm_0_elbow_joint",
    "arm_0_wrist_1_joint",
    "arm_0_wrist_2_joint",
    "arm_0_wrist_3_joint",
)


@pytest.fixture(scope="module")
def kin(urdf_bundle, robot_srdf):
    return PinocchioKinematics(
        str(urdf_bundle / "robot.urdf"),
        str(robot_srdf),
        mesh_dir=str(urdf_bundle),
        joints=ARM,
        tcp_frame="rg6_hand_tcp",
    )


def test_the_srdf_removes_the_neighbour_pairs(kin):
    # The SRDF carries 132 disable_collisions entries; without it the neighbour pairs stay and the gate refuses
    # every configuration.  The exact counts are recorded in the module docstring of pin_kinematics.
    assert kin.n_pairs_before_srdf > kin.n_pairs
    assert kin.n_pairs > 0


def test_the_neutral_configuration_is_collision_free(kin):
    # Without the SRDF the neighbour pairs would report here -- this is the test that it was applied at all.
    assert kin.config_collides({j: 0.0 for j in ARM}) is False


def test_the_camera_link_is_a_frame(kin):
    pos, rot = kin.frame_pose("camera_0_link", {j: 0.0 for j in ARM})
    assert pos.shape == (3,) and rot.shape == (3, 3)


def test_a_belief_box_on_the_hand_collides(kin):
    q = {j: 0.0 for j in ARM}
    pos, _ = kin.frame_pose("arm_0_wrist_2_link", q)
    kin.set_belief_boxes([BeliefBox("obstacle_0", (0.05, 0.05, 0.05), tuple(pos), (0.0, 0.0, 0.0, 1.0))])
    assert kin.config_collides(q, obstacles_only=True) is True


def test_belief_boxes_are_replaced_not_accumulated(kin):
    q = {j: 0.0 for j in ARM}
    kin.set_belief_boxes([])
    assert kin.config_collides(q, obstacles_only=True) is False


def test_ik_returns_to_a_pose_forward_kinematics_produced(kin):
    q = {j: 0.0 for j in ARM}
    q["arm_0_shoulder_lift_joint"] = -1.0
    q["arm_0_elbow_joint"] = 0.9
    target_pos, target_rot = kin.frame_pose("rg6_hand_tcp", q)
    solved = kin.solve_ik(target_pos, target_rot, seed={j: 0.0 for j in ARM})
    assert solved is not None
    got_pos, _ = kin.frame_pose("rg6_hand_tcp", solved)
    assert np.linalg.norm(got_pos - target_pos) < 1e-3
