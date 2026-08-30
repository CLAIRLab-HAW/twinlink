"""
Build a MuJoCo body from a list of convex parts, at frozen mass.

The point of the decomposition is to vary HOW FINELY an object is modelled without changing anything else about it.
Four facts about MuJoCo shape this module, all measured on 3.10 rather than recalled:

* box / cylinder / capsule / ellipsoid / sphere all compile as plain geoms, so no asset pipeline is needed to vary
  object fidelity.
* A single ``mesh`` geom collides as its CONVEX HULL.  A probe dropped onto a U-shaped shell rests 8 cm higher as one
  mesh than as three boxes. Concavity therefore exists only through several geom children -- which is why ``PRIMITIVES``
  excludes ``mesh`` outright instead of ranking it above a decomposition it cannot beat.
* Geom masses ADD UP.  A four-part mug weighs 0.19 kg where a one-part box weighs 0.15 kg, so refining geometry would
  silently refine mass too. The explicit ``<inertial>`` written here overrides the derived values and keeps both mass
  and inertia identical across every rung -- without it a study measures geometry and parameters at once and can
  separate neither.
* ``size`` means something different per geom type, and for the endpoint (``fromto``) form it carries the RADIUS ALONE.
  Reading an extent off, it therefore needs :func:`bounding_half_extents`, not a ``max`` over the numbers; the compiler
  also accepts a wrong ``size`` arity without a word the caller can act on, which is why :func:`add_shape` checks it
  first.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .mjcf_scene import fmt as _fmt

PRIMITIVES = frozenset({"box", "cylinder", "capsule", "ellipsoid", "sphere"})

#: How many numbers MuJoCo reads out of ``size`` per geom type, in the
#: ordinary (``pos``) form.  Measured against the 3.10 compiler; getting this
#: wrong does not raise here, it raises inside the compiler with a message
#: that names an XML element the caller never wrote.
_SIZE_ARITY = {"box": 3, "ellipsoid": 3, "sphere": 1, "capsule": 2, "cylinder": 2}

#: Types that accept the endpoint form -- and there ``size`` carries the
#: radius ALONE, the length comes from the two points.
_FROMTO_TYPES = frozenset({"capsule", "cylinder"})

_ORIGIN = (0.0, 0.0, 0.0)


def free_joint_name(name: str) -> str:
    """Name of the free joint :func:`add_shape` gives ``name``'s body.

    Single source: a caller that registers the body as graspable looks the joint up again by name -- spelled out
    independently on both sides, the two can drift, and a drift there does not crash.  MuJoCo skips an unknown joint
    silently, and the object simply never becomes graspable.
    """
    return f"{name}_free"


@dataclass(frozen=True)
class Part:
    """One convex piece of an object."""

    type: str
    size: tuple[float, ...]
    pos: tuple[float, float, float] = _ORIGIN
    #: Endpoint form for capsules/cylinders; overrides ``pos`` when given.
    fromto: tuple[float, float, float, float, float, float] | None = None
    #: Orientation in the body frame (wxyz, MuJoCo order).  Without it every
    #: box in a decomposition is forced axis-parallel, which caps how finely
    #: an object can be modelled -- the stapler's arm had to be laid flat.
    #: Mutually exclusive with ``fromto``, which already fixes the axis.
    quat: tuple[float, float, float, float] | None = None


def _quat_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    """Rotation matrix of a wxyz quaternion (not assumed normalised)."""
    q = np.asarray(quat, dtype=float)
    norm = float(q @ q)
    if norm <= 0.0:
        raise ValueError(f"quat {tuple(quat)} has zero length")
    w, x, y, z = q * np.sqrt(2.0 / norm)
    return np.array(
        [
            [1.0 - y * y - z * z, x * y - w * z, x * z + w * y],
            [x * y + w * z, 1.0 - x * x - z * z, y * z - w * x],
            [x * z - w * y, y * z + w * x, 1.0 - x * x - y * y],
        ]
    )


def _part_extent(part: Part) -> np.ndarray:
    """Half extents of one part's own axis-aligned box, in its own frame."""
    size = np.asarray(part.size, dtype=float)
    if part.type in ("box", "ellipsoid"):
        return size[:3].copy()
    if part.type == "sphere":
        return np.full(3, size[0])
    if part.type == "cylinder":
        return np.array([size[0], size[0], size[1]])
    # capsule: the two hemispherical caps sit ON TOP of the cylinder length.
    return np.array([size[0], size[0], size[1] + size[0]])


def part_bounds(part: Part) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of one part in the BODY frame.

    Public because a caller that merges or coarsens a decomposition needs the same reading of ``fromto``, ``pos`` and
    ``quat`` that :func:`bounding_half_extents` uses -- a second reading of the same three fields is exactly the drift
    this module exists to prevent.

    :param part: The part to measure.
    :return: ``(lo, hi)``, the lower and upper corner in the body frame.
    """
    if part.fromto is not None:
        radius = float(part.size[0])
        a = np.asarray(part.fromto[:3], dtype=float)
        b = np.asarray(part.fromto[3:], dtype=float)
        return np.minimum(a, b) - radius, np.maximum(a, b) + radius

    extent = _part_extent(part)
    centre = np.asarray(part.pos, dtype=float)
    if part.quat is None:
        return centre - extent, centre + extent

    # Rotated: take the corners of the part's own box through the rotation. An over-approximation for anything but a
    # box, which is the safe side -- a hull that is too small puts the object into the table.
    rot = _quat_matrix(part.quat)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = centre + (signs * extent) @ rot.T
    return corners.min(axis=0), corners.max(axis=0)


def bounding_half_extents(parts: Sequence[Part]) -> tuple[float, float, float]:
    """Half extents of the box around ``parts``, measured from the body origin.

    The fallback for a scene object that declares no ``half_extents`` of its own.  Deliberately not an isotropic
    ``max`` over every part's ``size``: that reads a ``fromto`` capsule as its RADIUS, so a 0.256 m long capsule comes
    out as 8 mm, a caller placing it on a surface sinks it 12 cm into the table-top, and it settles 8 cm BELOW the
    surface -- no error, no warning, an unphysical starting pose in the very axis such a study measures.

    Measured from the body origin, not from the occupied region's centre: a resting height is applied to the body
    ORIGIN, so a hull offset from it would be off by exactly that offset.
    """
    if not parts:
        raise ValueError("a body needs at least one part")

    lows, highs = zip(*(part_bounds(part) for part in parts))
    lo = np.min(np.asarray(lows), axis=0)
    hi = np.max(np.asarray(highs), axis=0)
    half = np.maximum(np.abs(lo), np.abs(hi))
    return float(half[0]), float(half[1]), float(half[2])


def _check(name: str, part: Part) -> None:
    """Refuse a part the MuJoCo compiler would misread or silently truncate."""
    if part.type not in PRIMITIVES:
        raise ValueError(
            f"{name!r}: geom type {part.type!r} is not supported. "
            f"Allowed: {sorted(PRIMITIVES)}. A 'mesh' collides as its "
            "convex hull, so it adds no fidelity a decomposition does "
            "not already provide -- add parts instead."
        )
    if part.fromto is not None:
        if part.type not in _FROMTO_TYPES:
            raise ValueError(f"{name!r}: 'fromto' is only defined for {sorted(_FROMTO_TYPES)}, not for {part.type!r}")
        if tuple(part.pos) != _ORIGIN:
            raise ValueError(
                f"{name!r}: {part.type!r} carries both 'fromto' and 'pos' -- "
                "MuJoCo drops 'pos' silently, so the part would not sit where "
                "it was written. Move the offset into 'fromto'."
            )
        if part.quat is not None:
            raise ValueError(
                f"{name!r}: {part.type!r} carries both 'fromto' and 'quat' -- the two endpoints already fix the axis."
            )
        wanted = 1  # radius only; the length comes from the endpoints
    else:
        wanted = _SIZE_ARITY[part.type]

    if len(part.size) != wanted:
        raise ValueError(
            f"{name!r}: {part.type!r}"
            f"{' with fromto' if part.fromto is not None else ''} needs "
            f"{wanted} size value(s), got {len(part.size)}: {tuple(part.size)}"
        )
    if part.quat is not None:
        if len(part.quat) != 4:
            raise ValueError(f"{name!r}: quat needs 4 values (wxyz), got {len(part.quat)}")
        _quat_matrix(part.quat)  # raises on a zero-length quat


def add_shape(
    worldbody: ET.Element,
    name: str,
    parts: Sequence[Part],
    *,
    pos: tuple[float, float, float],
    mass: float,
    diaginertia: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    free: bool = True,
) -> ET.Element:
    """Append a body made of ``parts`` and return it.

    :param worldbody: The MJCF worldbody element to append the body to.
    :param name: The name of the body.
    :param parts: A sequence of convex parts making up the body.
    :param pos: The 3D position of the body in the world frame.
    :param mass: The frozen mass of the body, overriding derived values.
    :param diaginertia: The diagonal inertia of the body.
    :param rgba: The RGBA color tuple for the body's geoms.
    :param free: Whether to add a free joint to the body.
    :return: The created MJCF body element.
    """
    if not parts:
        raise ValueError(f"{name!r}: a body needs at least one part")
    for part in parts:
        _check(name, part)

    body = ET.SubElement(worldbody, "body")
    body.set("name", name)
    body.set("pos", _fmt(*pos))

    # BEFORE the geoms: mass and inertia are authored, never derived.
    inertial = ET.SubElement(body, "inertial")
    inertial.set("pos", "0 0 0")
    inertial.set("mass", _fmt(mass))
    inertial.set("diaginertia", _fmt(*diaginertia))

    if free:
        ET.SubElement(body, "freejoint").set("name", free_joint_name(name))

    for i, part in enumerate(parts):
        geom = ET.SubElement(body, "geom")
        geom.set("name", f"{name}_geom{i}")
        geom.set("type", part.type)
        geom.set("size", _fmt(*part.size))
        if part.fromto is not None:
            geom.set("fromto", _fmt(*part.fromto))
        else:
            geom.set("pos", _fmt(*part.pos))
            if part.quat is not None:
                geom.set("quat", _fmt(*part.quat))
        geom.set("rgba", _fmt(*rgba))
        geom.set("friction", "1.2 0.02 0.001")

    return body
