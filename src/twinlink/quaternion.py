"""Quaternion algebra in MuJoCo convention (wxyz) -- borrowed, not built.

On 2026-08-23 these computations stood written out by hand three times in the workspace: ``twinlink.task_sim``,
``openvla_stack.env.sim`` and ``twin_sufficiency.scenes`` each carried their own version of the Hamilton product, some
of them conjugation and rotation about the vertical axis as well.  All three computed the same thing -- until one of
them stops doing so.  A sign error in there shows up as a slightly twisted object at the gripper, not as a red test.

**None of this is written anew here.**  MuJoCo ships the operations itself
(``mju_mulQuat``, ``mju_negQuat``, ``mju_axisAngle2Quat``, ``mju_mat2Quat``),
and it is the library that defines the convention in the first place -- its
version cannot, by construction, deviate from the simulation it is computed
against.  Cross-measured on 2026-08-23: ``mju_mulQuat`` agrees with the
previous hand computation to 2.2e-16 over 2000 random pairs.  What stands here
is only the shell: allocate the output buffer, bring the inputs to float64,
return the result.

**Why not ``scipy.spatial.transform.Rotation``:** checked and rejected.
``twinlink`` deliberately depends only on ``numpy``, ``pyyaml`` and
``clearlog``; the MuJoCo routines are already there wherever these functions
are needed, and scipy would be a second convention next to MuJoCo's -- exactly
the kind of choice the three hand-written copies grew out of.

**xyzw is something else.**  The wire (ROS, ``/twin/*``) speaks xyzw; these
functions speak wxyz exclusively, and the name of the convention therefore
appears in EVERY function name.  The xyzw side lives in
``robot_contract.twin_protocol`` (``quat_mul_xyzw`` and neighbours) and -- for
the layer that must not depend on ``robot_contract`` -- in
``twinlink.tf_buffer``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _mj():
    """MuJoCo, imported late -- ``twinlink`` also runs without the extra."""
    import mujoco

    return mujoco


def _arr(values: Sequence[float], size: int) -> np.ndarray:
    """``values`` as a contiguous float64 array of length ``size``.

    The ``mju_*`` bindings write into buffers and read from buffers; a non-contiguous slice or a float32 array would be
    an error at runtime, not a wrong result -- but then again only at runtime.
    """
    out = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if out.size != size:
        raise ValueError(f"expected {size} values, got {out.size}")
    return out


def quat_mul_wxyz(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Hamilton product of two wxyz quaternions: first ``b``, then ``a``.

    The order is the point at which this function gets used wrongly -- swapped it yields a different, equally plausible
    rotation, not an error.
    """
    out = np.empty(4)
    _mj().mju_mulQuat(out, _arr(a, 4), _arr(b, 4))
    return out


def quat_conj_wxyz(quat: Sequence[float]) -> np.ndarray:
    """Conjugate wxyz quaternion -- the counter-rotation of a UNIT quaternion.

    For a quaternion that is not normalized the conjugate is NOT the inverse.
    """
    out = np.empty(4)
    _mj().mju_negQuat(out, _arr(quat, 4))
    return out


def quat_about_z_wxyz(angle: float) -> np.ndarray:
    """Rotation about the vertical axis (rad) as a wxyz quaternion."""
    out = np.empty(4)
    _mj().mju_axisAngle2Quat(out, np.array([0.0, 0.0, 1.0]), float(angle))
    return out


def mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix ─▶ wxyz quaternion."""
    out = np.empty(4)
    _mj().mju_mat2Quat(out, _arr(mat, 9))
    return out


def quat_to_yaw_wxyz(quat: Sequence[float]) -> float:
    """In-plane rotation of a ``wxyz`` quaternion (rad).

    Here rather than at the call sites: a world answers ``object_poses`` with quaternions, and every reader that
    wants a yaw would otherwise write this same ``arctan2`` again -- with its own sign convention.

    :param quat: quaternion as ``(w, x, y, z)``.
    :returns: yaw in radians.
    """
    w, x, y, z = _arr(quat, 4)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
