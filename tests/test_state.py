"""RobotState — the thread-safe in-memory twin core."""
import math
import threading

import numpy as np
import pytest

from twinlink.state import (
    CameraFrame,
    ObstacleCloud,
    PlannedTrajectory,
    RobotState,
    Transform,
)


def test_joint_updates_and_lookup():
    s = RobotState()
    assert math.isnan(s.joint_position("arm_0_elbow_joint"))
    s.update_joint("arm_0_elbow_joint", 1.6, 0.1, 2.0, stamp=42.0)
    j = s.joint("arm_0_elbow_joint")
    assert j.position == 1.6 and j.velocity == 0.1 and j.effort == 2.0
    assert s.joint_position("arm_0_elbow_joint") == 1.6
    assert s.joint_position("missing", default=-1.0) == -1.0
    assert "arm_0_elbow_joint" in s.joint_names()


def test_update_joints_bulk_and_vector():
    s = RobotState()
    s.update_joints(["a", "b"], [1.0, 2.0])
    vec = s.joint_positions(["a", "b", "c"])
    assert vec[0] == 1.0 and vec[1] == 2.0 and math.isnan(vec[2])
    assert set(s.joints().keys()) == {"a", "b"}


def test_revision_bumps_on_every_write():
    s = RobotState()
    r0 = s.revision
    s.update_joint("a", 0.0)
    s.set_base_pose(Transform.identity())
    s.set_extra("k", 1)
    assert s.revision >= r0 + 3


def test_transforms_and_base_pose():
    s = RobotState()
    t = Transform(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        stamp=1.0,
        frame_id="odom",
        child_frame_id="base_link",
    )
    s.set_transform(t)
    got = s.transform("odom", "base_link")
    assert got is not None and got.translation[1] == 2.0
    assert s.transform("nope", "base_link") is None
    assert ("odom", "base_link") in s.transforms()
    s.set_base_pose(t)
    assert s.base_pose().child_frame_id == "base_link"


def test_camera_obstacles_planned_trajectory():
    s = RobotState()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    s.set_camera("cam", CameraFrame(image=img, encoding="rgb8", stamp=0.0,
                                    frame_id="f", width=4, height=4))
    assert s.camera("cam").width == 4
    assert "cam" in s.cameras()

    s.set_obstacles("cloud", ObstacleCloud(points=np.zeros((10, 3)),
                                           frame_id="f", stamp=0.0))
    traj = PlannedTrajectory(
        joint_names=["a"], positions=np.array([[0.0], [1.0]]),
        times=np.array([0.0, 2.5]),
    )
    s.set_planned_trajectory(traj)
    assert s.planned_trajectory().duration() == pytest.approx(2.5)


def test_thread_safety_smoke():
    """Concurrent writers/readers must not crash or lose the last write."""
    s = RobotState()
    errors = []

    def writer(tid: int):
        try:
            for i in range(300):
                s.update_joint(f"j{tid}", float(i))
                s.set_extra(f"e{tid}", i)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def reader():
        try:
            for _ in range(300):
                s.joint_positions([f"j{k}" for k in range(4)])
                s.joints()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for k in range(4):
        assert s.joint_position(f"j{k}") == 299.0
    assert s.revision >= 4 * 300 * 2
