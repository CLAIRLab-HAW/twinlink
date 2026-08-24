"""Offline TFBuffer — chains, interpolation, inverses (numpy-only slerp)."""

import numpy as np
import pytest
from types import SimpleNamespace as NS

from twinlink.tf_buffer import TFBuffer, Transform, _slerp


def _tf_msg(parent, child, xyz, quat=(0, 0, 0, 1)):
    return NS(
        transforms=[
            NS(
                header=NS(frame_id=parent, stamp=NS(sec=0, nanosec=0)),
                child_frame_id=child,
                transform=NS(
                    translation=NS(x=xyz[0], y=xyz[1], z=xyz[2]),
                    rotation=NS(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                ),
            )
        ]
    )


def test_identity_and_missing_path():
    buf = TFBuffer()
    assert np.allclose(buf.lookup("a", "a", 0), np.eye(4))
    assert buf.lookup("a", "b", 0) is None


def test_static_chain_forward_and_reverse():
    """odom->base->camera chain; lookup camera->odom composes + inverts."""
    buf = TFBuffer()
    buf.add_static(_tf_msg("odom", "base", (1.0, 0.0, 0.0)))
    buf.add_static(_tf_msg("base", "camera", (0.0, 2.0, 0.0)))
    buf.finalize()

    M = buf.lookup("camera", "odom", 0)  # points camera -> odom
    p = M @ np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(p[:3], [1.0, 2.0, 0.0])

    M_inv = buf.lookup("odom", "camera", 0)  # reverse direction
    q = M_inv @ p
    assert np.allclose(q[:3], [0.0, 0.0, 0.0], atol=1e-12)


def test_dynamic_interpolation_translation_and_slerp():
    buf = TFBuffer()
    # 90° about z at t=1000, identity at t=0
    buf.add_dynamic(0, _tf_msg("odom", "base", (0.0, 0.0, 0.0)))
    buf.add_dynamic(1000, _tf_msg("odom", "base", (2.0, 0.0, 0.0), quat=(0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))))
    buf.finalize()

    M = buf.lookup("base", "odom", 500)  # halfway
    assert np.allclose(M[:3, 3], [1.0, 0.0, 0.0])
    # halfway rotation = 45° about z: base-x axis maps to (cos45, sin45, 0)
    x_axis = M[:3, :3] @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(x_axis, [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0], atol=1e-9)

    # clamping outside the series
    assert np.allclose(buf.lookup("base", "odom", -5)[:3, 3], [0.0, 0.0, 0.0])
    assert np.allclose(buf.lookup("base", "odom", 9999)[:3, 3], [2.0, 0.0, 0.0])


def test_slerp_endpoints_and_short_arc():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90° z
    assert np.allclose(_slerp(q0, q1, 0.0), q0)
    assert np.allclose(_slerp(q0, q1, 1.0), q1)
    mid = _slerp(q0, q1, 0.5)  # 45° z
    assert np.allclose(mid, [0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)], atol=1e-9)
    # antipodal representation must take the short arc
    assert np.allclose(np.abs(_slerp(q0, -q1, 1.0)), np.abs(q1), atol=1e-9)


def test_transform_inverse_compose_roundtrip():
    t = Transform(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, np.sin(0.3), np.cos(0.3)]))
    eye = t.compose(t.inverse()).as_matrix()
    assert np.allclose(eye, np.eye(4), atol=1e-12)
