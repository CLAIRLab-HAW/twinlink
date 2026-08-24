"""SimEvents — twin physics event records."""

from twinlink.events import SimEvents


def test_grasp_unknown_is_a_third_state_not_a_missing_grasp():
    """``None`` on the grasp means "not determinable", not "nothing grasped".

    Without this field the caller has only two pigeonholes, and a gripper with dead tool IO ends up in the same one as a
    clean empty grasp.
    """
    events = SimEvents(grasp_unknown="green")

    assert events.grasp_acquired is None
    assert events.grasp_unknown == "green"


def test_merge_carries_grasp_unknown():
    a, b = SimEvents(), SimEvents(grasp_unknown="green")
    a.merge(b)
    assert a.grasp_unknown == "green"
