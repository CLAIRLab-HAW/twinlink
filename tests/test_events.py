"""SimEvents — twin physics event records."""

from twinlink.events import SimEvents


def test_grasp_unknown_is_a_third_state_not_a_missing_grasp():
    """``None`` beim Griff heisst „nicht feststellbar", nicht „nichts gegriffen".

    Ohne dieses Feld hat der Aufrufer nur zwei Faecher, und ein Greifer mit
    toter Tool-IO landet im selben wie ein sauberer Leergriff.
    """
    events = SimEvents(grasp_unknown="green")

    assert events.grasp_acquired is None
    assert events.grasp_unknown == "green"


def test_merge_carries_grasp_unknown():
    a, b = SimEvents(), SimEvents(grasp_unknown="green")
    a.merge(b)
    assert a.grasp_unknown == "green"
