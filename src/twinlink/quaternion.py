"""Quaternionen-Algebra in MuJoCo-Konvention (wxyz) -- geliehen, nicht gebaut.

Diese Rechnungen standen am 2026-08-23 dreimal von Hand im Workspace: ``twinlink.task_sim``, ``openvla_stack.env.sim``
und ``twin_sufficiency.scenes`` trugen je eine eigene Fassung des Hamilton-Produkts, teils dazu Konjugation und Drehung
um die Hochachse.  Alle drei rechneten dasselbe -- bis eine von ihnen es nicht mehr tut.  Ein Vorzeichenfehler darin
zeigt sich als leicht verdrehtes Objekt am Greifer, nicht als roter Test.

**Nichts davon ist hier neu geschrieben.**  MuJoCo bringt die Operationen
selbst mit (``mju_mulQuat``, ``mju_negQuat``, ``mju_axisAngle2Quat``,
``mju_mat2Quat``), und es ist die Bibliothek, die die Konvention ueberhaupt
definiert -- ihre Fassung kann per Konstruktion nicht von der Simulation
abweichen, gegen die sie gerechnet wird.  Am 2026-08-23 gegengemessen:
``mju_mulQuat`` stimmt mit der bisherigen Handrechnung ueber 2000 zufaellige
Paare auf 2,2e-16 ueberein.  Was hier steht, ist nur die Huelle: Ausgabepuffer
anlegen, Eingaben nach float64 bringen, Ergebnis zurueckgeben.

**Warum nicht ``scipy.spatial.transform.Rotation``:** geprueft und verworfen.
``twinlink`` haengt bewusst nur an ``numpy``, ``pyyaml`` und ``clearlog``; die
MuJoCo-Wege sind ohnehin schon da, wo diese Funktionen gebraucht werden, und
scipy waere eine zweite Konvention neben der von MuJoCo -- genau die Sorte
Wahlmoeglichkeit, aus der die drei Handkopien entstanden sind.

**xyzw ist etwas anderes.**  Der Draht (ROS, ``/twin/*``) spricht xyzw; diese
Funktionen sprechen ausschliesslich wxyz, und der Konventionsname steht
deshalb in JEDEM Funktionsnamen.  Die xyzw-Seite liegt in
``robot_contract.twin_protocol`` (``quat_mul_xyzw`` und Nachbarn) sowie -- fuer
die Schicht, die nicht an ``robot_contract`` haengen darf -- in
``twinlink.tf_buffer``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _mj():
    """MuJoCo, spaet importiert -- ``twinlink`` laeuft auch ohne das Extra."""
    import mujoco

    return mujoco


def _arr(values: Sequence[float], size: int) -> np.ndarray:
    """``values`` als zusammenhaengendes float64-Array der Laenge ``size``.

    Die ``mju_*``-Bindungen schreiben in Puffer und lesen aus Puffern; ein nicht zusammenhaengender Slice oder ein
    float32-Array waere ein Fehler zur Laufzeit, kein falsches Ergebnis -- aber eben auch erst zur Laufzeit.
    """
    out = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if out.size != size:
        raise ValueError(f"expected {size} values, got {out.size}")
    return out


def quat_mul_wxyz(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Hamilton-Produkt zweier wxyz-Quaternionen: erst ``b``, dann ``a``.

    Die Reihenfolge ist der Punkt, an dem diese Funktion falsch benutzt wird -- vertauscht ergibt sie eine andere,
    ebenso plausible Drehung, keinen Fehler.
    """
    out = np.empty(4)
    _mj().mju_mulQuat(out, _arr(a, 4), _arr(b, 4))
    return out


def quat_conj_wxyz(quat: Sequence[float]) -> np.ndarray:
    """Konjugierter wxyz-Quaternion -- die Gegendrehung eines EINHEITS-Quaternions.

    Fuer einen nicht normierten Quaternion ist die Konjugation NICHT die Inverse.
    """
    out = np.empty(4)
    _mj().mju_negQuat(out, _arr(quat, 4))
    return out


def quat_about_z_wxyz(angle: float) -> np.ndarray:
    """Drehung um die Hochachse (rad) als wxyz-Quaternion."""
    out = np.empty(4)
    _mj().mju_axisAngle2Quat(out, np.array([0.0, 0.0, 1.0]), float(angle))
    return out


def mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """3x3-Rotationsmatrix -> wxyz-Quaternion."""
    out = np.empty(4)
    _mj().mju_mat2Quat(out, _arr(mat, 9))
    return out
