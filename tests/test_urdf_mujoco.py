"""URDF→MuJoCo smoke against the workspace robot bundle (needs mujoco)."""

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")


def _workspace_urdf():
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "workspace.repos").is_file():
            return cand / "urdf" / "robot.urdf"
    return None


URDF = _workspace_urdf()
pytestmark = pytest.mark.skipif(
    URDF is None or not URDF.exists(),
    reason="urdf/robot.urdf bundle not checked out (standalone twinlink)",
)


def test_load_robot_bundle_collision_geometry():
    from twinlink.urdf_mujoco import load_mujoco_from_urdf

    model = load_mujoco_from_urdf(str(URDF), add_ground=True)
    joint_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    }
    # the a200-0553 bundle: 6 UR joints + 6 RG6 finger joints + 4 wheels = 16
    for j in (
        "arm_0_shoulder_pan_joint",
        "arm_0_wrist_3_joint",
        "rg6_finger_joint",
    ):
        assert j in joint_names, f"{j} missing from MuJoCo model"
    assert model.njnt >= 16
    # model must actually step
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)


def _lowest_nonplane_z(model) -> float:
    from twinlink.urdf_mujoco import _geom_lowest_z

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return min(
        _geom_lowest_z(model, data, gid)
        for gid in range(model.ngeom)
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
    )


def test_welded_base_is_grounded():
    """With a ground plane, the welded robot stands ON it -- the Husky's
    base_link is 0.132 m above the wheels' contact point, so welding it at the
    origin would sink the wheels half into the plane."""
    from twinlink.urdf_mujoco import load_mujoco_from_urdf

    for kwargs in (
        {},  # collision geometry
        {"keep_visual": True, "colored": False},  # visual-only (spact demos)
        {"keep_visual": True, "colored": False, "with_collision": True},
    ):
        model = load_mujoco_from_urdf(str(URDF), add_ground=True, **kwargs)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        assert bid >= 0
        shift = float(model.body_pos[bid][2])
        assert 0.05 < shift < 0.25, f"grounding shift missing ({kwargs}): {shift}"
        assert _lowest_nonplane_z(model) >= -1e-4, f"still sunk with {kwargs}"


def test_floating_base_is_not_shifted():
    """A free base settles under gravity / is driven from odometry -- the
    loader must not apply the static grounding shift there."""
    from twinlink.urdf_mujoco import load_mujoco_from_urdf

    model = load_mujoco_from_urdf(str(URDF), add_ground=True, floating_base=True)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert bid >= 0
    assert abs(float(model.body_pos[bid][2])) < 1e-9
