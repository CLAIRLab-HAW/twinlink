"""Tests for the MJCF part helpers: primitive compilation, bounding half extents and free joints.

Every claim here is checked against the MuJoCo compiler rather than against the XML this module writes, so the tests
need the ``mujoco`` extra.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.mjcf_parts import (  # noqa: E402  (after the extra check above)
    PRIMITIVES,
    Part,
    add_shape,
    bounding_half_extents,
    free_joint_name,
)


def _model(parts, mass=0.15):
    root = ET.Element("mujoco")
    wb = ET.SubElement(root, "worldbody")
    add_shape(
        wb, "probe", parts, pos=(0.0, 0.0, 1.0), mass=mass, diaginertia=(1e-4, 1e-4, 1e-4), rgba=(0.8, 0.3, 0.1, 1.0)
    )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def test_every_declared_primitive_compiles():
    for kind, size in [
        ("box", (0.03, 0.03, 0.03)),
        ("cylinder", (0.04, 0.05)),
        ("capsule", (0.01, 0.04)),
        ("ellipsoid", (0.04, 0.03, 0.05)),
        ("sphere", (0.04,)),
    ]:
        assert kind in PRIMITIVES
        assert _model([Part(kind, size)]).ngeom >= 1


def test_a_multi_part_body_keeps_its_parts():
    mug = [
        Part("cylinder", (0.04, 0.05)),
        Part("capsule", (0.008,), fromto=(0.04, 0, 0.02, 0.065, 0, 0.02)),
        Part("capsule", (0.008,), fromto=(0.065, 0, 0.02, 0.065, 0, -0.02)),
    ]
    assert _model(mug).ngeom == 3


def test_mass_is_frozen_across_representations():
    # THE test of this task.  Without <inertial> the four-part variant would be heavier than the one-part variant
    # (measured: 0.19 instead of 0.15 kg) and the alpha_obj sweep would silently drag the mass along.
    one = _model([Part("box", (0.05, 0.05, 0.05))])
    many = _model(
        [
            Part("cylinder", (0.04, 0.05)),
            Part("box", (0.01, 0.01, 0.03), pos=(0.05, 0.0, 0.0)),
            Part("box", (0.01, 0.01, 0.03), pos=(-0.05, 0.0, 0.0)),
            Part("sphere", (0.015,), pos=(0.0, 0.0, 0.06)),
        ]
    )
    assert one.body_mass[1] == pytest.approx(0.15)
    assert many.body_mass[1] == pytest.approx(0.15)
    assert np.allclose(one.body_inertia[1], many.body_inertia[1])


def test_concavity_needs_several_parts_and_is_actually_reachable():
    # The drop test from task 1 as a regression: the decomposition has to let the sphere fall into the recess.  If
    # this test fails, alpha_obj is no longer an effective axis and the pre-study measures nothing.
    root = ET.Element("mujoco")
    ET.SubElement(root, "option").set("gravity", "0 0 -9.81")
    wb = ET.SubElement(root, "worldbody")
    add_shape(
        wb,
        "shell",
        [
            Part("box", (0.05, 0.03, 0.01), pos=(0.0, 0.0, -0.04)),
            Part("box", (0.01, 0.03, 0.04), pos=(-0.04, 0.0, 0.01)),
            Part("box", (0.01, 0.03, 0.04), pos=(0.04, 0.0, 0.01)),
        ],
        pos=(0.0, 0.0, 0.0),
        mass=1.0,
        diaginertia=(1e-3, 1e-3, 1e-3),
        rgba=(0.5, 0.5, 0.5, 1.0),
        free=False,
    )
    add_shape(
        wb,
        "ball",
        [Part("sphere", (0.012,))],
        pos=(0.0, 0.0, 0.25),
        mass=0.01,
        diaginertia=(1e-6, 1e-6, 1e-6),
        rgba=(1.0, 0.0, 0.0, 1.0),
    )

    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    for _ in range(4000):
        mujoco.mj_step(m, d)
    assert d.qpos[2] < 0.0, "the sphere has to lie in the recess, not on the hull"


def test_an_unknown_primitive_is_refused():
    with pytest.raises(ValueError, match="mesh"):
        _model([Part("mesh", (0.01,))])


# --------------------------------------------------------------------- #
# Hull dimensions: `size` alone does not carry the extent
# --------------------------------------------------------------------- #
def test_a_fromto_capsule_reports_its_true_length():
    # The finding: `size` of a fromto capsule carries ONLY the radius.  Whoever takes `max(size)` isotropically holds
    # a 0.256 m long capsule to be 8 mm big -- reset() then puts it 12 cm into the table top.
    half = bounding_half_extents([Part("capsule", (0.008,), fromto=(0.0, 0.0, -0.12, 0.0, 0.0, 0.12))])
    assert half == pytest.approx((0.008, 0.008, 0.128))


def test_the_hull_spans_every_part():
    half = bounding_half_extents(
        [Part("cylinder", (0.040, 0.050)), Part("capsule", (0.008,), fromto=(0.040, 0.0, 0.015, 0.052, 0.0, 0.015))]
    )
    assert half == pytest.approx((0.060, 0.040, 0.050))


def test_a_capsule_is_as_long_as_its_caps_reach():
    # Without fromto a capsule stands along z: size = (radius, half cylinder length), and the hemispheres go on top.
    assert bounding_half_extents([Part("capsule", (0.01, 0.04))]) == pytest.approx((0.01, 0.01, 0.05))


def test_the_hull_is_measured_from_the_body_origin():
    # The rest height in reset() sets the body ORIGIN; a hull that only measures the occupied region would sit off by
    # that region's offset.
    assert bounding_half_extents([Part("box", (0.01, 0.01, 0.01), pos=(0.0, 0.0, 0.05))]) == pytest.approx(
        (0.01, 0.01, 0.06)
    )


def test_the_hull_follows_a_rotated_part():
    # A flat box turned by 90 degrees around y lies on its edge.
    half = bounding_half_extents([Part("box", (0.06, 0.02, 0.01), quat=(0.70710678, 0.0, 0.70710678, 0.0))])
    assert half == pytest.approx((0.01, 0.02, 0.06), abs=1e-6)


# --------------------------------------------------------------------- #
# Inputs that used to run past their own guard into the compiler
# --------------------------------------------------------------------- #
def test_a_box_needs_three_size_values():
    with pytest.raises(ValueError, match="3 size value"):
        _model([Part("box", (0.03,))])


def test_a_fromto_capsule_carries_only_a_radius():
    with pytest.raises(ValueError, match="1 size value"):
        _model([Part("capsule", (0.008, 0.04), fromto=(0.0, 0.0, 0.0, 0.0, 0.0, 0.1))])


def test_a_free_capsule_needs_radius_and_half_length():
    with pytest.raises(ValueError, match="2 size value"):
        _model([Part("capsule", (0.008,))])


def test_pos_beside_fromto_is_refused():
    # MuJoCo discards `pos` in this combination silently -- the part then sits somewhere other than where the caller
    # wrote it.
    with pytest.raises(ValueError, match="fromto"):
        _model([Part("capsule", (0.008,), pos=(0.0, 0.0, 0.05), fromto=(0.0, 0.0, 0.0, 0.0, 0.0, 0.1))])


def test_fromto_is_refused_for_a_box():
    with pytest.raises(ValueError, match="fromto"):
        _model([Part("box", (0.01, 0.01, 0.01), fromto=(0.0, 0.0, 0.0, 0.0, 0.0, 0.1))])


# --------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------- #
def test_a_part_can_be_rotated():
    # Up to the closing review only a fromto capsule could be oriented; every box in a decomposition was forcibly
    # axis-aligned, which capped the upper rungs of the object ladder.
    quat = (0.70710678, 0.0, 0.0, 0.70710678)  # wxyz, 90 Grad um z
    m = _model([Part("box", (0.06, 0.02, 0.01), quat=quat)])
    assert np.allclose(m.geom_quat[0], quat, atol=1e-6)


def test_a_rotation_beside_fromto_is_refused():
    with pytest.raises(ValueError, match="fromto"):
        _model([Part("capsule", (0.008,), fromto=(0, 0, 0, 0, 0, 0.1), quat=(1.0, 0.0, 0.0, 0.0))])


def test_a_degenerate_rotation_is_refused():
    with pytest.raises(ValueError, match="quat"):
        _model([Part("box", (0.01, 0.01, 0.01), quat=(0.0, 0.0, 0.0, 0.0))])


def test_the_free_joint_name_has_one_source():
    # The name was encoded independently in two places (here and in env/sim.py); if one drifts, the registration
    # silently fails.
    m = _model([Part("box", (0.01, 0.01, 0.01))])
    assert free_joint_name("probe") == "probe_free"
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, free_joint_name("probe")) >= 0


def test_the_hull_and_a_single_part_read_the_same_fields():
    """``part_bounds`` is public because a caller that coarsens a decomposition must not read ``fromto`` a second time.

    A second reading is what the endpoint form invites: ``size`` carries the RADIUS there, so a naive reader takes a
    0.256 m capsule for an 8 mm one.  Asserted through the public pair rather than by inspecting the XML, so the two
    stay one reading.
    """
    from twinlink.mjcf_parts import part_bounds

    part = Part("cylinder", (0.008,), fromto=(0.0, 0.0, -0.128, 0.0, 0.0, 0.128))
    lo, hi = part_bounds(part)

    assert hi[2] == pytest.approx(0.136)  # half length + radius
    assert bounding_half_extents([part]) == pytest.approx((float(hi[0]), float(hi[1]), float(hi[2])))
    assert lo[2] == pytest.approx(-0.136)
