"""Die wxyz-Algebra: Konvention und Reihenfolge, nicht die Arithmetik.

Die Arithmetik gehoert MuJoCo (``mju_mulQuat`` und Nachbarn) und braucht hier
keinen Test -- sie ist die Referenz, gegen die die Simulation ohnehin rechnet.
Was einen Test braucht, ist das, was beim Umstellen auf sie kaputtgehen kann:

* **die Reihenfolge** -- ``quat_mul_wxyz(a, b)`` dreht erst um ``b``, dann um
  ``a``.  Vertauscht ergibt sie eine andere, ebenso plausible Drehung und
  keinen Fehler; genau so sieht ein falsch gehaltener Wuerfel am Greifer aus.
* **die Anordnung** -- w zuerst.  Wer hier xyzw hineingibt, bekommt Zahlen
  zurueck, keine Ausnahme.

Bis zum 2026-08-23 stand diese Rechnung dreimal von Hand im Workspace
(``task_sim``, ``openvla_stack.env.sim``, ``twin_sufficiency.scenes``).
"""
import numpy as np
import pytest

pytest.importorskip("mujoco", reason="wxyz-Algebra kommt aus MuJoCo")

from twinlink.quaternion import (  # noqa: E402
    mat_to_quat_wxyz,
    quat_about_z_wxyz,
    quat_conj_wxyz,
    quat_mul_wxyz,
)


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_the_first_argument_is_applied_last():
    """``quat_mul_wxyz(a, b)`` ist ``a nach b``, wie ``A @ B`` bei Matrizen."""
    a, b = quat_about_z_wxyz(0.5), quat_about_z_wxyz(0.2)
    # Um dieselbe Achse ist die Reihenfolge egal -- deshalb ueber zwei Achsen
    # pruefen, wo sie es nicht ist.
    tilt = mat_to_quat_wxyz(np.array([[1.0, 0.0, 0.0],
                                      [0.0, 0.0, -1.0],
                                      [0.0, 1.0, 0.0]]))
    forward = quat_mul_wxyz(a, tilt)
    backward = quat_mul_wxyz(tilt, a)
    assert not np.allclose(forward, backward), (
        "Testaufbau taugt nicht: die beiden Drehungen kommutieren")

    # Der Beleg: erst tilt, dann a -- angewandt auf einen Vektor.
    v = np.array([1.0, 0.0, 0.0])
    step_by_step = _rot_z(0.5) @ (np.array([[1.0, 0.0, 0.0],
                                            [0.0, 0.0, -1.0],
                                            [0.0, 1.0, 0.0]]) @ v)
    out = np.empty(3)
    import mujoco
    mujoco.mju_rotVecQuat(out, v, forward)
    assert np.allclose(out, step_by_step)

    # Und b bleibt b: zweimal dasselbe Argument ist keine Identitaet.
    assert np.allclose(quat_mul_wxyz(a, b), quat_about_z_wxyz(0.7))


def test_w_comes_first():
    """Die Anordnung ist wxyz -- eine Drehung um z fuellt die LETZTE Stelle."""
    q = quat_about_z_wxyz(np.pi / 2)
    assert q[0] == pytest.approx(np.cos(np.pi / 4))
    assert q[1] == pytest.approx(0.0)
    assert q[2] == pytest.approx(0.0)
    assert q[3] == pytest.approx(np.sin(np.pi / 4))


def test_the_conjugate_undoes_the_rotation():
    q = quat_mul_wxyz(quat_about_z_wxyz(1.1),
                      mat_to_quat_wxyz(_rot_z(-0.4)))
    assert np.allclose(quat_mul_wxyz(q, quat_conj_wxyz(q)),
                       [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_a_wrong_length_is_an_error_not_a_silent_result():
    """Ein xyzw-Quaternion mit drei Werten faellt auf, ein Tippfehler auch."""
    with pytest.raises(ValueError, match="expected 4 values"):
        quat_mul_wxyz([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="expected 9 values"):
        mat_to_quat_wxyz(np.eye(2))
