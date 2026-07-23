"""Build a MuJoCo model from a generic URDF.

MuJoCo's URDF importer is solid but several things trip it up on real robot
descriptions; this helper papers over all of them so an arbitrary robot URDF
"just loads":

1. ``<gazebo>`` / ``<transmission>`` blocks are stripped.
2. Meshes are routed through a cache with **globally unique names**.  This is
   essential: MuJoCo names a mesh by its file's basename, so two packages that
   both ship ``base_link`` (e.g. a Husky chassis and a gripper base) silently
   collapse into one mesh -- the gripper ends up wearing the chassis.  Unique
   cache names prevent that.
3. ``.dae`` meshes are converted with :mod:`twinlink.collada` (which preserves
   the authored orientation, unlike assimp, and recovers per-material colours);
   ``.gltf``/``.glb`` use ``assimp`` if present; ``.stl``/``.obj`` pass through.
4. Degenerate meshes (zero-volume shells) that MuJoCo refuses are detected and
   dropped, with a fallback to the link's collision geometry where available.
5. Links lacking ``<inertial>`` get a tiny default so mesh-inertia computation
   never fails (the twin is kinematic; values are irrelevant).
6. Optionally a ground plane, a light and a free-joint base are added by
   round-tripping through compiled MJCF.
7. A welded base is *grounded*: mobile-robot root links sit above the floor
   (the Husky's ``base_link`` is 0.132 m up), so welding them at the origin
   sinks the wheels into the ground plane.  The robot is raised so its lowest
   geometry rests on z=0 instead.

Nothing here is robot-specific; it is driven entirely by the URDF.
"""
from __future__ import annotations

import copy
import logging
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

log = logging.getLogger("twinlink.urdf")

_PASSTHROUGH = (".stl", ".obj")
_compile_cache: dict = {}

# A processed mesh expands to one or more (absolute path, rgba-or-None) parts.
MeshParts = List[Tuple[str, Optional[list]]]


# ---------------------------------------------------------------------- #
# mesh processing
# ---------------------------------------------------------------------- #
def _sanitize(rel: str) -> str:
    base = os.path.splitext(rel)[0]
    return base.replace(os.sep, "__").replace("/", "__").replace("\\", "__").replace(".", "_")


def _fresh(dst: str, src: str) -> bool:
    return os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src)


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copyfile(src, dst)


def _assimp_convert(src: str, dst: str) -> bool:
    if shutil.which("assimp") is None:
        log.warning("assimp not found; cannot convert %s (install assimp or drop it)", os.path.basename(src))
        return False
    res = subprocess.run(["assimp", "export", src, dst], capture_output=True)
    return res.returncode == 0 and os.path.exists(dst)


def _mesh_compiles(path: str) -> bool:
    if path in _compile_cache:
        return _compile_cache[path]
    import mujoco

    xml = (
        f'<mujoco><asset><mesh name="m" file="{path}"/></asset>'
        f'<worldbody><body><geom type="mesh" mesh="m"/></body></worldbody></mujoco>'
    )
    try:
        mujoco.MjModel.from_xml_string(xml)
        ok = True
    except Exception:
        ok = False
    _compile_cache[path] = ok
    return ok


def _process_mesh(rel: str, urdf_dir: str, cache_dir: str, colored: bool) -> MeshParts:
    """Resolve a URDF mesh to unique, MuJoCo-loadable cache file(s).

    Returns a list of ``(abspath, rgba)`` parts -- several when a coloured
    ``.dae`` is split per material, otherwise one.  Empty if unusable."""
    rel = rel.lstrip("./")
    src = os.path.normpath(os.path.join(urdf_dir, rel))
    if not os.path.exists(src):
        log.warning("mesh not found: %s", src)
        return []
    low = rel.lower()
    uname = _sanitize(rel)

    if low.endswith(".dae"):
        if colored:
            try:
                from .collada import dae_to_colored_objs

                parts = dae_to_colored_objs(src, cache_dir, uname)
            except Exception as exc:
                log.warning("DAE conversion failed for %s: %s", rel, exc)
                return []
            return [(p, rgba) for p, rgba in parts if _mesh_compiles(p)]
        dst = os.path.join(cache_dir, uname + ".obj")
        if not _fresh(dst, src):
            try:
                from .collada import dae_to_obj

                dae_to_obj(src, dst)
            except Exception as exc:
                log.warning("DAE conversion failed for %s: %s", rel, exc)
                return []
    elif low.endswith((".gltf", ".glb")):
        dst = os.path.join(cache_dir, uname + ".obj")
        if not _fresh(dst, src) and not _assimp_convert(src, dst):
            return []
    elif low.endswith(_PASSTHROUGH):
        dst = os.path.join(cache_dir, uname + os.path.splitext(rel)[1].lower())
        if not _fresh(dst, src):
            _link_or_copy(src, dst)
    else:
        log.warning("unsupported mesh type: %s", rel)
        return []

    dst = os.path.abspath(dst)
    return [(dst, None)] if _mesh_compiles(dst) else []


def _set_material(visual: ET.Element, name: str, rgba: list) -> None:
    rgba = (list(rgba) + [1.0, 1.0, 1.0, 1.0])[:4]
    mat = ET.SubElement(visual, "material")
    mat.set("name", name)
    col = ET.SubElement(mat, "color")
    col.set("rgba", " ".join(f"{c:.4f}" for c in rgba))


def _rewrite_or_drop(link, geoms, urdf_dir, cache_dir, colored, dropped) -> int:
    """Rewrite each geom's mesh to cache path(s); drop unusable mesh geoms.

    Returns the number of geoms kept (mesh geoms that loaded + primitives)."""
    kept = 0
    for g in list(geoms):
        mesh = g.find("geometry/mesh")
        if mesh is None:
            kept += 1  # primitive geometry: keep as-is
            continue
        rel = mesh.get("filename", "")
        parts = _process_mesh(rel, urdf_dir, cache_dir, colored)
        if not parts:
            link.remove(g)
            dropped.append(os.path.basename(rel))
            continue

        if len(parts) == 1:
            path, rgba = parts[0]
            mesh.set("filename", path)
            if colored and rgba is not None and g.find("material") is None:
                _set_material(g, f"twinlink_{_sanitize(rel)}", rgba)
            kept += 1
            continue

        # Coloured DAE split into several materials -> one <visual> each.
        scale = mesh.get("scale")
        origin = g.find("origin")
        link.remove(g)
        for idx, (path, rgba) in enumerate(parts):
            vis = ET.SubElement(link, "visual")
            if origin is not None:
                vis.append(copy.deepcopy(origin))
            geo = ET.SubElement(vis, "geometry")
            sub = ET.SubElement(geo, "mesh")
            sub.set("filename", path)
            if scale:
                sub.set("scale", scale)
            if rgba is not None:
                _set_material(vis, f"twinlink_{_sanitize(rel)}_{idx}", rgba)
            kept += 1
    return kept


def _ensure_inertial(link) -> None:
    if link.find("inertial") is not None:
        return
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass").set("value", "0.1")
    inertia = ET.SubElement(inertial, "inertia")
    for k, v in dict(ixx="1e-3", iyy="1e-3", izz="1e-3", ixy="0", ixz="0", iyz="0").items():
        inertia.set(k, v)


def _prepare_urdf(urdf_path, keep_visual, colored, with_collision, cache_dir) -> Tuple[ET.ElementTree, List[str]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    for tag in ("gazebo", "transmission"):
        for elem in root.findall(tag):
            root.remove(elem)

    dropped: List[str] = []
    for link in root.findall("link"):
        _ensure_inertial(link)
        visuals = list(link.findall("visual"))
        collisions = list(link.findall("collision"))
        if keep_visual:
            kept = _rewrite_or_drop(link, visuals, urdf_dir, cache_dir, colored, dropped)
            if kept == 0:  # fall back to collision geometry for this link
                _rewrite_or_drop(link, collisions, urdf_dir, cache_dir, False, dropped)
            elif with_collision:  # keep collisions too (for physics contact)
                _rewrite_or_drop(link, collisions, urdf_dir, cache_dir, False, dropped)
            else:
                for col in collisions:
                    link.remove(col)
        else:
            for vis in visuals:
                link.remove(vis)
            _rewrite_or_drop(link, collisions, urdf_dir, cache_dir, False, dropped)
    return tree, dropped


# ---------------------------------------------------------------------- #
# scene augmentation (compiled-MJCF round-trip)
# ---------------------------------------------------------------------- #
def _add_scene(mjcf_root: ET.Element, floating_base: bool, base_body_name: Optional[str]) -> Optional[str]:
    worldbody = mjcf_root.find("worldbody")
    asset = mjcf_root.find("asset")
    if asset is None:
        asset = ET.SubElement(mjcf_root, "asset")

    visual = mjcf_root.find("visual")
    if visual is None:
        visual = ET.SubElement(mjcf_root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", "1280")
    glob.set("offheight", "960")
    # A brighter headlight so material colours read well.
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.4 0.4 0.4")
    headlight.set("diffuse", "0.6 0.6 0.6")
    headlight.set("specular", "0.1 0.1 0.1")

    tex = ET.SubElement(asset, "texture")
    tex.set("name", "twinlink_grid_tex")
    tex.set("type", "2d")
    tex.set("builtin", "checker")
    tex.set("rgb1", ".18 .24 .31")
    tex.set("rgb2", ".10 .14 .18")
    tex.set("width", "300")
    tex.set("height", "300")
    mat = ET.SubElement(asset, "material")
    mat.set("name", "twinlink_grid")
    mat.set("texture", "twinlink_grid_tex")
    mat.set("texrepeat", "8 8")
    mat.set("reflectance", ".15")

    light = ET.SubElement(worldbody, "light")
    light.set("pos", "0 0 4")
    light.set("dir", "0 0 -1")
    light.set("directional", "true")
    light.set("diffuse", ".6 .6 .6")

    ground = ET.SubElement(worldbody, "geom")
    ground.set("name", "twinlink_ground")
    ground.set("type", "plane")
    ground.set("size", "8 8 .1")
    ground.set("material", "twinlink_grid")
    ground.set("pos", "0 0 0")

    free_name = None
    if floating_base:
        top_bodies = worldbody.findall("body")
        target = None
        if base_body_name:
            target = next((b for b in top_bodies if b.get("name") == base_body_name), None)
        if target is None and top_bodies:
            target = top_bodies[0]
        if target is not None:
            free_name = "twinlink_base"
            fj = ET.Element("freejoint")
            fj.set("name", free_name)
            target.insert(0, fj)
        else:
            log.warning("floating_base requested but no top-level body found")
    return free_name


def _geom_lowest_z(model, data, gid: int) -> float:
    """Exact lowest world-z of one geom (mesh vertices / analytic primitives)."""
    import mujoco
    import numpy as np

    gtype = int(model.geom_type[gid])
    pos = data.geom_xpos[gid]
    mat = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        verts = model.mesh_vert[adr : adr + num]
        return float((verts @ mat.T)[:, 2].min() + pos[2])
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(pos[2] - np.abs(mat[2, :]) @ size)
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(pos[2] - size[0])
    if gtype in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        c = abs(float(mat[2, 2]))
        s = float(np.sqrt(max(0.0, 1.0 - c * c)))
        drop = size[1] * c + size[0] * (s if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER) else 1.0)
        return float(pos[2] - drop)
    return float(pos[2] - model.geom_rbound[gid])  # conservative fallback


def _ground_welded_robot(worldbody: ET.Element, model, data) -> float:
    """Raise the welded robot so its lowest geom rests on the z=0 ground plane.

    Measures over ALL geoms, not just colliding ones: visual-only loads
    (``keep_visual`` without ``with_collision``) carry no contact geometry at
    all, and for full loads the collision hull bottoms out at the same height
    as the wheels' visual meshes.  Returns the applied shift (0.0 if the model
    already rests on or above the ground).
    """
    import mujoco

    lowest = 0.0
    for gid in range(model.ngeom):
        if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        lowest = min(lowest, _geom_lowest_z(model, data, gid))
    if lowest >= -1e-4:
        return 0.0
    shift = -lowest
    for body in worldbody.findall("body"):
        pos = [float(v) for v in (body.get("pos") or "0 0 0").split()]
        pos[2] += shift
        body.set("pos", " ".join(f"{v:.6g}" for v in pos))
        log.info(
            "grounded welded base %r: raised %.3f m so the robot rests on z=0",
            body.get("name", "?"), shift,
        )
    return shift


def _root_link_name(urdf_path: str) -> Optional[str]:
    root = ET.parse(urdf_path).getroot()
    links = [l.get("name") for l in root.findall("link")]
    children = {j.find("child").get("link") for j in root.findall("joint") if j.find("child") is not None}
    roots = [name for name in links if name not in children]
    return roots[0] if roots else None


# ---------------------------------------------------------------------- #
# public entry point
# ---------------------------------------------------------------------- #
def load_mujoco_from_urdf(
    urdf_path: str,
    *,
    floating_base: bool = False,
    add_ground: bool = True,
    keep_visual: bool = False,
    colored: bool = True,
    with_collision: bool = False,
    mesh_cache_dir: Optional[str] = None,
):
    """Load ``urdf_path`` into a ``mujoco.MjModel``.

    Parameters
    ----------
    floating_base : add a free joint on the base link so the base pose can be
        driven from odometry, or settle under gravity (otherwise the base is
        welded at the origin).
    add_ground : add a checkered ground plane and a light for a usable scene.
        With a welded base the robot is also raised so its lowest geometry
        rests on the plane -- the root link of a mobile base sits above the
        ground, so welding it at the origin would sink the wheels.  Consumers
        can read the applied shift off the compiled model as the base body's
        ``body_pos[...][2]``.
    keep_visual : render the ``<visual>`` meshes instead of ``<collision>``
        geometry.  Falls back to collision geometry per-link when a visual mesh
        cannot be loaded.
    colored : with ``keep_visual``, split ``.dae`` meshes per material and apply
        their diffuse colours (the UR5's blue/grey/black, etc.).
    with_collision : with ``keep_visual``, *also* keep the ``<collision>``
        geometry so the model can be simulated (contacts) while rendering the
        visual meshes.  Needed for the physics demo.
    mesh_cache_dir : where converted/linked meshes are written
        (default: ``<urdf_dir>/.twinlink_meshcache``).
    """
    import mujoco

    urdf_path = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)
    cache_dir = os.path.abspath(mesh_cache_dir or os.path.join(urdf_dir, ".twinlink_meshcache"))
    os.makedirs(cache_dir, exist_ok=True)

    tree, dropped = _prepare_urdf(urdf_path, keep_visual, colored, with_collision, cache_dir)
    if dropped:
        log.info("dropped %d unusable mesh(es): %s", len(dropped), sorted(set(dropped)))

    root = tree.getroot()
    mj = ET.SubElement(root, "mujoco")
    compiler = ET.SubElement(mj, "compiler")
    compiler.set("meshdir", cache_dir)
    compiler.set("balanceinertia", "true")
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    compiler.set("strippath", "false")

    tmp_urdf = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, dir=cache_dir)
    tree.write(tmp_urdf.name)
    tmp_urdf.close()
    try:
        model = mujoco.MjModel.from_xml_path(tmp_urdf.name)
    finally:
        os.unlink(tmp_urdf.name)

    if not (floating_base or add_ground):
        return model

    base_body = _root_link_name(urdf_path)
    tmp_xml = os.path.join(cache_dir, next(tempfile._get_candidate_names()) + ".twinlink.xml")
    try:
        mujoco.mj_saveLastXML(tmp_xml, model)
        mjcf = ET.parse(tmp_xml)
        if add_ground and not floating_base:
            # Welded base: raise the robot so it stands ON the ground plane
            # instead of half-sinking its wheels (a floating base settles under
            # gravity / is driven from odometry and needs no static grounding).
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            _ground_welded_robot(mjcf.getroot().find("worldbody"), model, data)
        _add_scene(mjcf.getroot(), floating_base, base_body)
        mjcf.write(tmp_xml)
        model = mujoco.MjModel.from_xml_path(tmp_xml)
    finally:
        if os.path.exists(tmp_xml):
            os.unlink(tmp_xml)
    return model
