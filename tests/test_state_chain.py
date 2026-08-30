"""Tests for transform chains in ``RobotState``: known frames, and no identity silently invented for unknown ones."""

import numpy as np
import pytest
from twinlink.state import RobotState, Transform


def _tf(parent, child, xyz, quat=(0.0, 0.0, 0.0, 1.0)):
    return Transform(
        translation=np.asarray(xyz, dtype=float),
        rotation=np.asarray(quat, dtype=float),
        frame_id=parent,
        child_frame_id=child,
    )


def test_identity_for_the_same_frame():
    # The promise is unchanged -- but it holds for a frame the graph actually KNOWS.  If the short-circuit check ran
    # before every look at the edges, this test would run on a completely empty RobotState and prove nothing.
    st = RobotState()
    st.set_transform(_tf("base", "cam", [1.0, 0.0, 0.0]))
    assert np.allclose(st.chain("base", "base"), np.eye(4))
    assert np.allclose(st.chain("cam", "cam"), np.eye(4))


def test_an_unknown_frame_gets_no_identity_even_against_itself():
    # The finding: `chain("nope", "nope")` returned eye(4) although "nope" occurs in no edge.  In
    # `perception.twinlink_camera` that means: a capture whose frame_id accidentally has the same name as the world
    # frame gets exactly the silent identity matrix the docstring promises to refuse -- and the back-projection looks
    # plausible afterwards.
    st = RobotState()
    st.set_transform(_tf("base", "cam", [1.0, 0.0, 0.0]))
    assert st.chain("nope", "nope") is None
    assert RobotState().chain("base", "base") is None


def test_single_forward_edge():
    st = RobotState()
    st.set_transform(_tf("base", "cam", [1.0, 0.0, 0.0]))
    # A point in the camera frame lies 1 m further forward in the base frame.
    M = st.chain("cam", "base")
    assert np.allclose(M @ np.array([0.0, 0.0, 0.0, 1.0]), [1.0, 0.0, 0.0, 1.0])


def test_reverse_direction_is_the_inverse():
    st = RobotState()
    st.set_transform(_tf("base", "cam", [1.0, 0.0, 0.0]))
    fwd = st.chain("cam", "base")
    rev = st.chain("base", "cam")
    assert np.allclose(fwd @ rev, np.eye(4), atol=1e-9)


def test_two_hops_compose():
    st = RobotState()
    st.set_transform(_tf("base", "arm", [1.0, 0.0, 0.0]))
    st.set_transform(_tf("arm", "cam", [0.0, 2.0, 0.0]))
    M = st.chain("cam", "base")
    assert np.allclose(M @ np.array([0.0, 0.0, 0.0, 1.0]), [1.0, 2.0, 0.0, 1.0])


def test_rotation_is_carried_through():
    # 90 deg um z: xyzw
    q = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    st = RobotState()
    st.set_transform(_tf("base", "cam", [0.0, 0.0, 0.0], q))
    M = st.chain("cam", "base")
    assert np.allclose(M @ np.array([1.0, 0.0, 0.0, 1.0]), [0.0, 1.0, 0.0, 1.0], atol=1e-9)


def test_disconnected_frames_return_none_rather_than_identity():
    # The most important test of the task: a silent identity matrix would falsify the back-projection by orders of
    # magnitude while looking like an artefact of the abstraction.
    st = RobotState()
    st.set_transform(_tf("base", "arm", [1.0, 0.0, 0.0]))
    assert st.chain("cam", "base") is None


def test_two_hops_with_rotation_pin_the_composition_order():
    """Multi-hop WITH rotation -- the only case that catches the order.

    Pure translations commute, which is why ``test_two_hops_compose`` also passes with the accumulation swapped.  Here
    the correct order yields [-1, 0, 0], the swapped one [1, 2, 0].
    """
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    st = RobotState()
    st.set_transform(_tf("base", "arm", [1.0, 0.0, 0.0], q90))
    st.set_transform(_tf("arm", "cam", [0.0, 2.0, 0.0]))

    M = st.chain("cam", "base")
    assert np.allclose(M @ np.array([0.0, 0.0, 0.0, 1.0]), [-1.0, 0.0, 0.0, 1.0], atol=1e-9)


def test_a_degenerate_quaternion_raises_instead_of_reading_as_no_rotation():
    st = RobotState()
    st.set_transform(_tf("base", "cam", [1.0, 0.0, 0.0], (0.0, 0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="degenerate quaternion"):
        st.chain("cam", "base")
