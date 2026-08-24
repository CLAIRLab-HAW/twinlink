"""TwinDisplayMirror: Beliefs anzeigen, ohne Jitter."""

from __future__ import annotations

from twinlink.display_mirror import TwinDisplayMirror


class _FakeSim:
    def __init__(self):
        self.calls = []

    def display_object(self, label, position, yaw=0.0):
        self.calls.append(("display", label, tuple(position), round(yaw, 6)))

    def park_object(self, label, index=0):
        self.calls.append(("park", label, index))

    def display_carry(self, label):
        self.calls.append(("carry", label))

    def display_release(self, label, position=None, yaw=0.0):
        self.calls.append(("release", label))


def test_unchanged_beliefs_are_not_rewritten():
    """Der Kern des Mirrors: kein Neu-Teleportieren, kein Jitter."""
    sim = _FakeSim()
    mirror = TwinDisplayMirror(sim)
    items = [("green", ("pose", (0.7, 0.0, 0.22), 0.0))]
    mirror.sync(items)
    mirror.sync(items)
    assert sim.calls == [("display", "green", (0.7, 0.0, 0.22), 0.0)]


def test_state_change_is_written():
    sim = _FakeSim()
    mirror = TwinDisplayMirror(sim)
    mirror.sync([("green", ("pose", (0.7, 0.0, 0.22), 0.0))])
    mirror.sync([("green", ("carry",))])
    mirror.sync([("green", ("park", 1))])
    # carry -> park ends the display carry first (release) before parking --
    # the original _apply's behaviour, load-bearing for the real sim: without
    # it the twin's kinematic carry keeps following the TCP and the "parked"
    # object snaps right back next physics step (see
    # hrl/tests/test_belief_mirror.py::test_mark_lost_parks_the_cube, which
    # asserts the carry has actually ended after such a transition).
    assert [c[0] for c in sim.calls] == ["display", "carry", "release", "park"]


def test_reset_forgets_written_state():
    sim = _FakeSim()
    mirror = TwinDisplayMirror(sim)
    item = [("green", ("pose", (0.7, 0.0, 0.22), 0.0))]
    mirror.sync(item)
    mirror.reset()
    mirror.sync(item)
    assert len(sim.calls) == 2
