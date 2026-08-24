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
    # Die Zusage ist unveraendert -- aber sie gilt fuer ein Frame, das der
    # Graph auch KENNT.  Liefe die Kurzschluss-Pruefung vor jedem Blick in
    # die Kanten, liefe dieser Test auf einem voellig leeren RobotState und
    # belegte nichts.
    st = RobotState()
    st.set_transform(_tf("base", "cam", [1.0, 0.0, 0.0]))
    assert np.allclose(st.chain("base", "base"), np.eye(4))
    assert np.allclose(st.chain("cam", "cam"), np.eye(4))


def test_an_unknown_frame_gets_no_identity_even_against_itself():
    # Der Befund: `chain("nope", "nope")` lieferte eye(4), obwohl "nope" in
    # keiner Kante vorkommt.  In `perception.twinlink_camera` heisst das: eine
    # Aufnahme, deren frame_id versehentlich wie der Weltframe heisst, bekommt
    # genau die stille Einheitsmatrix, die der Docstring zu verweigern
    # verspricht -- und die Rueckprojektion sieht danach plausibel aus.
    st = RobotState()
    st.set_transform(_tf("base", "cam", [1.0, 0.0, 0.0]))
    assert st.chain("nope", "nope") is None
    assert RobotState().chain("base", "base") is None


def test_single_forward_edge():
    st = RobotState()
    st.set_transform(_tf("base", "cam", [1.0, 0.0, 0.0]))
    # Ein Punkt im Kameraframe liegt 1 m weiter vorn im Basisframe.
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
    # Der wichtigste Test der Aufgabe: eine stillschweigende Einheitsmatrix
    # wuerde die Rueckprojektion um Groessenordnungen verfaelschen und dabei
    # wie ein Abstraktionseffekt aussehen.
    st = RobotState()
    st.set_transform(_tf("base", "arm", [1.0, 0.0, 0.0]))
    assert st.chain("cam", "base") is None


def test_two_hops_with_rotation_pin_the_composition_order():
    """Mehr-Hop MIT Drehung -- der einzige Fall, der die Reihenfolge faengt.

    Reine Translationen kommutieren, deshalb besteht
    ``test_two_hops_compose`` auch mit vertauschter Akkumulation.  Hier
    liefert die richtige Reihenfolge [-1, 0, 0], die vertauschte [1, 2, 0].
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
