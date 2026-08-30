"""The wxyz algebra: convention and order, not the arithmetic.

The arithmetic belongs to MuJoCo (``mju_mulQuat`` and neighbours) and needs no test here -- it is the reference the
simulation computes against anyway.  What needs a test is what can break when switching over to it:

* **the order** -- ``quat_mul_wxyz(a, b)`` rotates about ``b`` first, then
  about ``a``.  Swapped it yields a different, equally plausible rotation and
  no error; that is exactly what a wrongly held cube at the gripper looks like.
* **the layout** -- w first.  Feeding xyzw in here returns numbers, not an
  exception.

Until 2026-08-23 this computation stood written out by hand three times in the workspace (``task_sim``,
``openvla_stack.env.sim``, ``twin_sufficiency.scenes``).
"""

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="wxyz-Algebra kommt aus MuJoCo")

from twinlink.quaternion import mat_to_quat_wxyz, quat_about_z_wxyz, quat_conj_wxyz, quat_mul_wxyz  # noqa: E402


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_the_first_argument_is_applied_last():
    """``quat_mul_wxyz(a, b)`` is ``a after b``, like ``A @ B`` for matrices."""
    a, b = quat_about_z_wxyz(0.5), quat_about_z_wxyz(0.2)
    # About the same axis the order does not matter -- so check over two axes, where it does.
    tilt = mat_to_quat_wxyz(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]))
    forward = quat_mul_wxyz(a, tilt)
    backward = quat_mul_wxyz(tilt, a)
    assert not np.allclose(forward, backward), "the test setup is no good: the two rotations commute"

    # The evidence: first tilt, then a -- applied to a vector.
    v = np.array([1.0, 0.0, 0.0])
    step_by_step = _rot_z(0.5) @ (np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]) @ v)
    out = np.empty(3)
    import mujoco

    mujoco.mju_rotVecQuat(out, v, forward)
    assert np.allclose(out, step_by_step)

    # And b stays b: the same argument twice is not an identity.
    assert np.allclose(quat_mul_wxyz(a, b), quat_about_z_wxyz(0.7))


def test_w_comes_first():
    """The layout is wxyz -- a rotation about z fills the LAST slot."""
    q = quat_about_z_wxyz(np.pi / 2)
    assert q[0] == pytest.approx(np.cos(np.pi / 4))
    assert q[1] == pytest.approx(0.0)
    assert q[2] == pytest.approx(0.0)
    assert q[3] == pytest.approx(np.sin(np.pi / 4))


def test_the_conjugate_undoes_the_rotation():
    q = quat_mul_wxyz(quat_about_z_wxyz(1.1), mat_to_quat_wxyz(_rot_z(-0.4)))
    assert np.allclose(quat_mul_wxyz(q, quat_conj_wxyz(q)), [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_a_wrong_length_is_an_error_not_a_silent_result():
    """An xyzw quaternion with three values is caught, and so is a typo."""
    with pytest.raises(ValueError, match="expected 4 values"):
        quat_mul_wxyz([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="expected 9 values"):
        mat_to_quat_wxyz(np.eye(2))


def test_the_yaw_of_a_z_rotation_is_the_angle_itself():
    from twinlink.quaternion import quat_about_z_wxyz, quat_to_yaw_wxyz

    for angle in (0.0, 0.3, -1.2, 2.9):
        assert abs(quat_to_yaw_wxyz(quat_about_z_wxyz(angle)) - angle) < 1e-9
