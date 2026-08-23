"""Minimal, dependency-light Collada (``.dae``) reader.

Why this exists: MuJoCo cannot read ``.dae`` meshes, and the obvious tool
(``assimp``) silently *re-orients* some Collada files (it bakes an inconsistent
axis conversion), which lands meshes rotated 90° in the model.  This reader
extracts the **raw** vertex coordinates and the node-transform hierarchy and
writes them straight to OBJ, preserving the authored frame -- exactly what a
URDF (and RViz/Gazebo) expect.  It only needs ``numpy`` + the stdlib XML parser.

It also recovers the **per-material diffuse colours** the meshes carry (e.g. the
UR5's blue/grey/black), because STL and a plain merged OBJ are colourless.  A
single MuJoCo mesh has a single colour, so :func:`dae_to_colored_objs` splits a
``.dae`` into one compact OBJ per material, each tagged with its RGBA.

It deliberately handles just what robot description meshes use: ``<triangles>``
and ``<polylist>`` primitives, ``<vertices>`` POSITION indirection, the
``<node>`` transform chain and the document ``<up_axis>``.  Anything it cannot
parse raises, so the caller can fall back or skip the mesh.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import numpy as np


def _ns(root: ET.Element) -> dict:
    return {"c": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {"c": ""}


def _q(tag: str, ns: dict) -> str:
    return f"c:{tag}" if ns["c"] else tag


def _floats(text: str) -> np.ndarray:
    return np.array(text.split(), dtype=float)


def _node_matrix(node: ET.Element) -> np.ndarray:
    M = np.eye(4)
    for child in node:
        tag = child.tag.split("}")[-1]
        if tag == "matrix":
            M = M @ _floats(child.text).reshape(4, 4)
        elif tag == "translate":
            T = np.eye(4)
            T[:3, 3] = _floats(child.text)[:3]
            M = M @ T
        elif tag == "rotate":
            v = _floats(child.text)
            n = v[:3] / (np.linalg.norm(v[:3]) or 1.0)
            a = np.radians(v[3])
            x, y, z = n
            c, s = np.cos(a), np.sin(a)
            M = M @ np.array(
                [
                    [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s, 0],
                    [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s, 0],
                    [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c), 0],
                    [0, 0, 0, 1],
                ]
            )
        elif tag == "scale":
            S = np.eye(4)
            S[0, 0], S[1, 1], S[2, 2] = _floats(child.text)[:3]
            M = M @ S
    return M


def _diffuse_by_material(root: ET.Element, ns: dict) -> dict:
    """material-id -> rgba, resolved through ``instance_effect`` -> diffuse."""
    eff_color = {}
    for eff in root.findall(f".//{_q('effect', ns)}", ns):
        col = eff.find(f".//{_q('diffuse', ns)}/{_q('color', ns)}", ns)
        if col is not None:
            try:
                eff_color[eff.get("id")] = [float(x) for x in col.text.split()][:4]
            except (ValueError, AttributeError):
                pass
    mat_color = {}
    for mat in root.findall(f".//{_q('material', ns)}", ns):
        ie = mat.find(_q("instance_effect", ns), ns)
        if ie is not None:
            mat_color[mat.get("id")] = eff_color.get(ie.get("url").lstrip("#"))
    return mat_color


def _primitives(mesh: ET.Element, ns: dict):
    """Yield (material_symbol, faces) for each primitive set in a mesh."""
    for prim in list(mesh):
        tag = prim.tag.split("}")[-1]
        if tag not in ("triangles", "polylist", "polygons"):
            continue
        inputs = prim.findall(_q("input", ns), ns)
        stride = max(int(i.get("offset", "0")) for i in inputs) + 1
        v_off = next(int(i.get("offset", "0")) for i in inputs if i.get("semantic") == "VERTEX")
        faces = []
        if tag == "polylist":
            vcounts = _floats(prim.find(_q("vcount", ns), ns).text).astype(int)
            idx = _floats(prim.find(_q("p", ns), ns).text).astype(int).reshape(-1, stride)[:, v_off]
            k = 0
            for vc in vcounts:
                poly = idx[k : k + vc]
                k += vc
                for j in range(1, vc - 1):
                    faces.append((poly[0], poly[j], poly[j + 1]))
        else:
            for pe in prim.findall(_q("p", ns), ns):
                p = _floats(pe.text).astype(int).reshape(-1, stride)[:, v_off]
                for j in range(0, len(p) - 2, 3):
                    faces.append((p[j], p[j + 1], p[j + 2]))
        yield prim.get("material"), np.array(faces, dtype=int)


def _geometry_positions(mesh: ET.Element, ns: dict):
    verts = mesh.find(_q("vertices", ns), ns)
    pos_src = next(
        i.get("source").lstrip("#")
        for i in verts.findall(_q("input", ns), ns)
        if i.get("semantic") == "POSITION"
    )
    for src in mesh.findall(_q("source", ns), ns):
        if src.get("id") == pos_src:
            acc = src.find(f"{_q('technique_common', ns)}/{_q('accessor', ns)}", ns)
            stride = int(acc.get("stride", "3")) if acc is not None else 3
            return _floats(src.find(_q("float_array", ns), ns).text).reshape(-1, stride)[:, :3]
    return None


def _compact(V: np.ndarray, faces: np.ndarray):
    """Keep only the vertices referenced by ``faces`` and remap the indices."""
    if len(faces) == 0:
        return np.zeros((0, 3)), faces
    used = np.unique(faces)
    return V[used], np.searchsorted(used, faces)


def _extract_groups(dae_path: str) -> Tuple[str, List[Tuple[str, Optional[list], np.ndarray, np.ndarray]]]:
    """Return (up_axis, [(material, rgba, V, F), ...]) in Z-up coordinates."""
    root = ET.parse(dae_path).getroot()
    ns = _ns(root)
    up_elem = root.find(f".//{_q('up_axis', ns)}", ns)
    up = up_elem.text.strip() if (up_elem is not None and up_elem.text) else "Y_UP"
    mat_color = _diffuse_by_material(root, ns)

    geoms = {}
    for g in root.findall(f".//{_q('geometry', ns)}", ns):
        mesh = g.find(_q("mesh", ns), ns)
        if mesh is None:
            continue
        pos = _geometry_positions(mesh, ns)
        if pos is None:
            continue
        geoms[g.get("id")] = (pos, list(_primitives(mesh, ns)))

    groups: dict = {}  # material -> [Vlist, Flist, offset]

    def add(material, V, F):
        entry = groups.setdefault(material, [[], [], 0])
        subV, subF = _compact(V, F)
        entry[0].append(subV)
        if len(subF):
            entry[1].append(subF + entry[2])
        entry[2] += len(subV)

    def walk(node, parent_M):
        M = parent_M @ _node_matrix(node)
        for inst in node.findall(_q("instance_geometry", ns), ns):
            gid = inst.get("url").lstrip("#")
            if gid not in geoms:
                continue
            bind = {
                inst_mat.get("symbol"): inst_mat.get("target").lstrip("#")
                for inst_mat in inst.findall(f".//{_q('instance_material', ns)}", ns)
            }
            pos, prims = geoms[gid]
            Vw = (np.c_[pos, np.ones(len(pos))] @ M.T)[:, :3]
            for symbol, faces in prims:
                add(bind.get(symbol, symbol), Vw, faces)
        for child in node.findall(_q("node", ns), ns):
            walk(child, M)

    scenes = root.findall(f".//{_q('visual_scene', ns)}", ns)
    for vs in scenes:
        for node in vs.findall(_q("node", ns), ns):
            walk(node, np.eye(4))
    if not groups:  # no scene graph -> emit geometries untransformed
        for pos, prims in geoms.values():
            for symbol, faces in prims:
                add(symbol, pos, faces)

    out = []
    for material, (Vs, Fs, _) in groups.items():
        if not Vs:
            continue
        V = np.vstack(Vs)
        F = np.vstack(Fs) if Fs else np.zeros((0, 3), int)
        if up == "Y_UP":
            V = V[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
        elif up == "X_UP":
            V = V[:, [2, 1, 0]] * np.array([1.0, 1.0, -1.0])
        out.append((material, mat_color.get(material), V, F))
    if not out:
        raise ValueError(f"no geometry parsed from {dae_path}")
    return up, out


def _write_obj(path: str, V: np.ndarray, F: np.ndarray, comment: str = "") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        if comment:
            f.write(f"# {comment}\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in F:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def dae_to_obj(dae_path: str, obj_path: str) -> tuple:
    """Convert a ``.dae`` to a single merged ``.obj`` (colourless)."""
    _up, groups = _extract_groups(dae_path)
    Vs, Fs, off = [], [], 0
    for _mat, _rgba, V, F in groups:
        Vs.append(V)
        if len(F):
            Fs.append(F + off)
        off += len(V)
    V = np.vstack(Vs)
    F = np.vstack(Fs) if Fs else np.zeros((0, 3), int)
    _write_obj(obj_path, V, F, f"converted from {os.path.basename(dae_path)} by twinlink.collada")
    return len(V), len(F), (V.max(0) - V.min(0))


def dae_to_colored_objs(dae_path: str, out_dir: str, stem: str) -> List[Tuple[str, Optional[list]]]:
    """Convert a ``.dae`` to one OBJ per material.

    Returns ``[(obj_abspath, rgba), ...]`` (rgba may be ``None``).  Results are
    cached: a sidecar JSON lets repeat loads skip re-parsing when up to date.
    """
    sidecar = os.path.join(out_dir, stem + ".colors.json")
    if os.path.exists(sidecar) and os.path.getmtime(sidecar) >= os.path.getmtime(dae_path):
        try:
            data = json.load(open(sidecar))
            if all(os.path.exists(p) for p, _ in data):
                return [(os.path.abspath(p), rgba) for p, rgba in data]
        except Exception:
            pass

    _up, groups = _extract_groups(dae_path)
    result: List[Tuple[str, Optional[list]]] = []
    for i, (_material, rgba, V, F) in enumerate(groups):
        if len(F) == 0:
            continue
        obj = os.path.join(out_dir, f"{stem}__m{i}.obj")
        _write_obj(obj, V, F, f"{os.path.basename(dae_path)} material {i}")
        result.append((os.path.abspath(obj), rgba))
    try:
        json.dump(result, open(sidecar, "w"))
    except Exception:
        pass
    return result
