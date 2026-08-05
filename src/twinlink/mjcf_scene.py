"""Reusable MJCF scene building blocks for task twins (robot-agnostic).

Extracted from ``hrl.env.scene`` (task-refactor 2026-07-23): every task app
that augments the compiled robot model via the save-XML-and-recompile
round-trip needs the same primitives --

* :func:`add_obstacle_pool` -- pre-allocated, runtime-mutated collision boxes
  for *perceived* obstacles (MuJoCo models cannot grow after compilation),
* :func:`add_distractors` -- sim-only authored clutter the perception
  pipeline must discover,
* :func:`camera_intrinsics` / :func:`camera_extrinsics` -- pinhole geometry
  of named scene cameras,
* :func:`fmt` / :func:`lookat_xyaxes` -- XML attribute helpers.

Task furniture (tables, task objects, camera placement) stays app-side; this
module carries only what is identical across tasks.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Sequence, Tuple

import numpy as np

#: Default number of pre-compiled obstacle slots.  Unused slots are inert
#: (parked + contacts off, see the pool owner e.g. hrl's ``StackCubesSim``).
OBSTACLE_POOL_SIZE = 8
#: Default body-name root for every task app that has not been told to use its
#: own (see the ``prefix`` argument of the functions below).  twinlink itself
#: is task-agnostic; this is only the value every pre-Task-11 caller
#: (``hrl.env.scene``, and -- via that -- ``octomap_explorer`` /
#: ``spact-integration-demos``, which never authored their own furniture and
#: so never needed to override it) was built against.
DEFAULT_SCENE_PREFIX = "hrl_"
#: Kept for backward compatibility: the exact prefixes existing consumers may
#: already import and compare against directly (unchanged values).
#: NICHT in twinlink selbst verwenden: ``TwinTaskSim`` leitet seine beiden
#: Hindernis-Präfixe seit 2026-08-01 aus dem Konstruktor-``scene_prefix`` ab.
#: Wer hier wieder vergleicht, nagelt die Bibliothek auf ``hrl_`` fest und
#: macht jede zweite App still blind für ihre Hindernisse.
OBSTACLE_BODY_PREFIX = f"{DEFAULT_SCENE_PREFIX}obstacle_"
DISTRACTOR_BODY_PREFIX = f"{DEFAULT_SCENE_PREFIX}distractor_"
#: Where unused pool slots wait: outside typical camera frustums and clear of
#: scratch parking spots task sims use (hrl parks cubes at (5+, 5+)).
OBSTACLE_PARK = (3.0, -3.0, 0.05)


def obstacle_body_name(index: int, prefix: str = DEFAULT_SCENE_PREFIX) -> str:
    return f"{prefix}obstacle_{index}"


def fmt(*vals: float) -> str:
    """MJCF attribute formatting (compact float list)."""
    return " ".join(f"{v:.6g}" for v in vals)


def lookat_xyaxes(cam_pos: np.ndarray, target: np.ndarray) -> str:
    """xyaxes for a camera at ``cam_pos`` looking at ``target`` (z-up world).

    MuJoCo cameras look along their -z axis with +y as image-up; we build an
    orthonormal frame whose -z points at the target and whose +y stays as
    world-up as possible.
    """
    fwd = target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:  # looking straight up/down
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fmt(*right, *up)


def add_obstacle_pool(
    worldbody: ET.Element, n_slots: int, prefix: str = DEFAULT_SCENE_PREFIX
) -> None:
    """Pre-allocate ``n_slots`` static obstacle boxes (runtime-mutated).

    MuJoCo models cannot grow after compilation, so perceived obstacles are
    written into this fixed pool via ``model.body_pos`` / ``model.geom_size``
    (valid for primitive geoms -- collision and rendering read them every
    step).  Compiled with contacts ON so a sim's geom classification picks
    them up; the pool owner parks and disables them right after indexing.

    ``prefix`` is the calling app's body-name root (default: ``hrl_``, the
    only value ever used before this became a parameter -- twinlink itself
    does not know the app's name).
    """
    px, py, pz = OBSTACLE_PARK
    for i in range(int(n_slots)):
        name = obstacle_body_name(i, prefix=prefix)
        body = ET.SubElement(worldbody, "body")
        body.set("name", name)
        body.set("pos", fmt(px + 0.5 * i, py, pz))
        geom = ET.SubElement(body, "geom")
        geom.set("name", f"{name}_geom")
        geom.set("type", "box")
        geom.set("size", fmt(0.02, 0.02, 0.02))
        geom.set("rgba", "0.85 0.45 0.10 0.55")


def distractor_body_name(index: int, prefix: str = DEFAULT_SCENE_PREFIX) -> str:
    return f"{prefix}distractor_{index}"


def distractor_joint_name(index: int, prefix: str = DEFAULT_SCENE_PREFIX) -> str:
    return f"{prefix}distractor_{index}_free"


def add_distractors(
    worldbody: ET.Element, distractors: Sequence[Dict], prefix: str = DEFAULT_SCENE_PREFIX
) -> None:
    """Author sim-only clutter boxes (``{"position", "size", "yaw", "rgba"}``).

    Boxes the task does NOT know about, standing in for real-world clutter --
    rendered and colliding like real obstacles, so an obstacle-perception
    pipeline can be exercised (and trained against) without hardware.

    ``"dynamic": true`` makes a distractor a free body (optional ``"mass"``,
    default 0.15 kg): it obeys physics and can be *grasped* by a sim that
    registers dynamic distractors as graspable -- the stand-in for "clear
    this obstacle away" tasks (semantic-masking increment 2).  Static
    (default) distractors stay immovable scenery.

    ``prefix`` is the calling app's body-name root (default: ``hrl_``, see
    :func:`add_obstacle_pool`).
    """
    for i, spec in enumerate(distractors):
        pos = [float(v) for v in spec["position"]]
        size = [float(v) for v in spec["size"]]
        yaw = float(spec.get("yaw", 0.0))
        rgba = spec.get("rgba", (0.45, 0.45, 0.50, 1.0))
        name = distractor_body_name(i, prefix=prefix)
        body = ET.SubElement(worldbody, "body")
        body.set("name", name)
        body.set("pos", fmt(*pos))
        if abs(yaw) > 1e-9:
            body.set("quat", fmt(np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)))
        if spec.get("dynamic"):
            free = ET.SubElement(body, "freejoint")
            free.set("name", distractor_joint_name(i, prefix=prefix))
        geom = ET.SubElement(body, "geom")
        geom.set("name", f"{name}_geom")
        geom.set("type", "box")
        geom.set("size", fmt(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0))
        geom.set("rgba", fmt(*rgba))
        if spec.get("dynamic"):
            geom.set("mass", fmt(float(spec.get("mass", 0.15))))
            geom.set("friction", "1.2 0.02 0.001")


def camera_intrinsics(model, camera: str, width: int, height: int) -> np.ndarray:
    """Pinhole K matrix for a named MuJoCo camera at a render resolution."""
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise KeyError(f"camera {camera!r} not in model")
    fovy = np.radians(float(model.cam_fovy[cam_id]))
    fy = (height / 2.0) / np.tan(fovy / 2.0)
    fx = fy  # square pixels
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


def camera_extrinsics(data, model, camera: str) -> Tuple[np.ndarray, np.ndarray]:
    """World pose of a camera: ``(position (3,), rotation (3,3) cam->world)``.

    MuJoCo camera frames look along -z with +y up (columns of the returned
    rotation are the camera's x/y/z axes in world coordinates).
    """
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise KeyError(f"camera {camera!r} not in model")
    return data.cam_xpos[cam_id].copy(), data.cam_xmat[cam_id].reshape(3, 3).copy()
