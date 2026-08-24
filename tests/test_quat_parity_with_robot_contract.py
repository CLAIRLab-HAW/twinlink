"""``_matrix_to_quat`` gibt es zweimal im Workspace -- sie duerfen nicht driften.

``twinlink.tf_buffer._matrix_to_quat`` und
``robot_contract.twin_protocol.mat_to_quat_xyzw`` sind Zeile fuer Zeile
dieselbe Shepperd-Implementierung, gleiche Konvention (xyzw), gleiche
Verzweigung ueber die groesste Diagonale -- letztere gewaehlt, weil sie bei
180-Grad-Drehungen (Spur -1) stabil bleibt, und das ist JEDE Top-Down-
Greifmatrix.

Die Dopplung ist kein Versehen, sondern der Preis einer bewussten
Schichtentscheidung: ``twinlink`` haengt absichtlich NICHT an
``robot_contract`` (siehe ``task_sim.py``, dort ausdruecklich kommentiert).
Solange diese Entscheidung steht, bleiben beide Fassungen -- aber sie duerfen
sich nicht auseinanderentwickeln, denn dann rechnet der Zwilling anders als
der Draht, und die Abweichung faellt erst an einer schiefen Greifpose auf.

Ein Querverweis im Kommentar waere dafuer kein Mechanismus (vgl.
``deploy/husky-offboard/tests/test_guard_single_source.py``).  Dieser Test
ist einer: er vergleicht die beiden Fassungen an Matrizen, die genau die
heiklen Zweige treffen.

Er ueberspringt sich sauber, wenn ``robot_contract`` fehlt -- in twinlinks
eigener CI ist das der Normalfall und gerade der Punkt der Schichtentscheidung.
"""

import numpy as np
import pytest

from twinlink.tf_buffer import _matrix_to_quat

tp = pytest.importorskip(
    "robot_contract.twin_protocol", reason="Paritaetstest laeuft nur im Workspace, wo beide Schichten liegen"
)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


#: Jeder Eintrag trifft einen anderen Zweig der Verzweigung.
MATRICES = {
    "identitaet (Spur > 0)": np.eye(3),
    "180 um x (m00 groesst)": _rot_x(np.pi),
    "180 um y (m11 groesst)": _rot_y(np.pi),
    "180 um z (m22 groesst)": _rot_z(np.pi),
    "top-down-greifpose": _rot_x(np.pi) @ _rot_z(0.3),
    "schraeg": _rot_z(0.7) @ _rot_y(-0.4) @ _rot_x(1.1),
    "kleinwinklig": _rot_z(1e-4),
}


@pytest.mark.parametrize("name", sorted(MATRICES))
def test_both_implementations_agree(name):
    m = MATRICES[name]
    a = np.asarray(_matrix_to_quat(m), float)
    b = np.asarray(tp.mat_to_quat_xyzw(m), float)
    # q und -q sind dieselbe Drehung -- verglichen wird die Drehung, nicht das
    # Vorzeichen.
    if float(np.dot(a, b)) < 0.0:
        b = -b
    assert a == pytest.approx(b, abs=1e-9), (
        f"{name}: twinlink und robot_contract rechnen auseinander -- " f"twinlink={a}, robot_contract={b}"
    )


@pytest.mark.parametrize("name", sorted(MATRICES))
def test_the_quaternion_is_a_unit_quaternion(name):
    """Beide muessen normiert liefern, sonst ist der Vergleich oben wertlos."""
    for fn in (_matrix_to_quat, tp.mat_to_quat_xyzw):
        q = np.asarray(fn(MATRICES[name]), float)
        assert float(np.linalg.norm(q)) == pytest.approx(1.0, abs=1e-9)
