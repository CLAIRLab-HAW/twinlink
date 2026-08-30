"""``WorldFramed`` moves between world and model frame -- in both directions, for all three operations.

The class exists because ``PinocchioKinematics`` answers in the URDF ROOT frame while a world speaks world
coordinates; a 132 mm base offset in the wrong direction is a plausible pose, not an error.  These tests pin the
direction of every conversion, ``set_belief_boxes`` included -- which is the one that was never exercised until
the sufficiency sweep fed it (2026-08-31) and raised on a NamedTuple.
"""

from __future__ import annotations

import numpy as np
import pytest

from twinlink.kinematics import WorldFramed
from twinlink.pin_kinematics import BeliefBox

#: The base offset the a200 actually stands at -- the number the direction of a conversion shows up in.
_BASE_Z_M = 0.13228


class _Inner:
    """Records what reached the model, in the model's own frame."""

    def __init__(self) -> None:
        self.boxes: list[BeliefBox] = []
        self.ik_args: tuple | None = None

    def frame_pose(self, name, joints):
        return np.array([0.0, 0.0, 0.0]), np.eye(3)

    def solve_ik(self, target_pos, target_rot, seed):
        self.ik_args = (np.asarray(target_pos, dtype=float), np.asarray(target_rot, dtype=float))
        return seed

    def set_belief_boxes(self, boxes) -> None:
        self.boxes = list(boxes)


def _root():
    """The model root, raised by the chassis offset and otherwise unrotated."""
    transform = np.eye(4)
    transform[2, 3] = _BASE_Z_M
    return transform


def test_a_belief_box_arrives_in_the_model_frame():
    """A box at 0.30 m in the world stands 0.30 - 0.13228 m above a root that is lifted by that much."""
    inner = _Inner()
    framed = WorldFramed(inner, _root)
    framed.set_belief_boxes(
        [
            BeliefBox(
                name="gate_0",
                half_extents_m=(0.05, 0.05, 0.15),
                position_m=(0.70, 0.20, 0.30),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        ]
    )
    assert len(inner.boxes) == 1
    assert inner.boxes[0].position_m == pytest.approx((0.70, 0.20, 0.30 - _BASE_Z_M))


def test_a_belief_box_keeps_everything_but_its_place():
    """Only the position is reframed -- a conversion that also touched the extents would resize the obstacle."""
    inner = _Inner()
    box = BeliefBox(
        name="gate_1",
        half_extents_m=(0.05, 0.06, 0.15),
        position_m=(0.70, -0.20, 0.30),
        quaternion_xyzw=(0.0, 0.0, 0.3826834, 0.9238795),
    )
    WorldFramed(inner, _root).set_belief_boxes([box])
    moved = inner.boxes[0]
    assert (moved.name, moved.half_extents_m, moved.quaternion_xyzw) == (
        box.name,
        box.half_extents_m,
        box.quaternion_xyzw,
    )


def test_an_ik_target_is_taken_the_other_way():
    """``frame_pose`` answers world; ``solve_ik`` is ASKED in world -- the two directions must not agree."""
    inner = _Inner()
    WorldFramed(inner, _root).solve_ik(np.array([0.7, 0.0, 0.5]), np.eye(3), {"a": 0.0})
    assert inner.ik_args[0] == pytest.approx([0.7, 0.0, 0.5 - _BASE_Z_M])


def test_a_frame_pose_comes_back_in_world():
    """The inner answer sits at the model origin; in world that is the base offset, not zero."""
    framed = WorldFramed(_Inner(), _root)
    position, _rotation = framed.frame_pose("tcp", {})
    assert position == pytest.approx([0.0, 0.0, _BASE_Z_M])
