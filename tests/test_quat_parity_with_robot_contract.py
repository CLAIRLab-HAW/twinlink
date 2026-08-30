"""``_matrix_to_quat`` exists twice in the workspace -- the two must not drift.

``twinlink.tf_buffer._matrix_to_quat`` and ``robot_contract.twin_protocol.mat_to_quat_xyzw`` are line for line the same
Shepperd implementation, same convention (xyzw), same branching over the largest diagonal element -- the latter chosen
because it stays stable at 180-degree rotations (trace -1), and that is EVERY top-down grasp matrix.

The duplication is not an oversight but the price of a deliberate layering decision: ``twinlink`` deliberately does NOT
depend on ``robot_contract`` (see ``task_sim.py``, commented there explicitly).  As long as that decision stands, both
versions remain -- but they must not develop apart, because then the twin computes differently from the wire, and the
deviation only shows up at a crooked grasp pose.

A cross-reference in a comment would be no mechanism for that (cf.
``deploy/husky-offboard/tests/test_guard_single_source.py``).  This test is one: it compares the two versions at
matrices that hit exactly the delicate branches.

It skips itself cleanly when ``robot_contract`` is missing -- in twinlink's own CI that is the normal case and precisely
the point of the layering decision.
"""

import numpy as np
import pytest

from twinlink.tf_buffer import _matrix_to_quat

tp = pytest.importorskip(
    "robot_contract.twin_protocol", reason="the parity test only runs in the workspace, where both layers live"
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


#: Every entry hits a different branch of the case distinction.
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
    # q and -q are the same rotation -- what is compared is the rotation, not the sign.
    if float(np.dot(a, b)) < 0.0:
        b = -b
    assert a == pytest.approx(b, abs=1e-9), (
        f"{name}: twinlink and robot_contract compute apart -- twinlink={a}, robot_contract={b}"
    )


@pytest.mark.parametrize("name", sorted(MATRICES))
def test_the_quaternion_is_a_unit_quaternion(name):
    """Both must return normalized results, otherwise the comparison above is worthless."""
    for fn in (_matrix_to_quat, tp.mat_to_quat_xyzw):
        q = np.asarray(fn(MATRICES[name]), float)
        assert float(np.linalg.norm(q)) == pytest.approx(1.0, abs=1e-9)
