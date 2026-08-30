"""Regression pin: only the gripper shells become permeable, not the whole hand.

``TwinTaskSim._setup_collision_masks`` must key off ``spec.gripper_prefixes`` -- the jaws/housing alone -- not
``spec.hand_prefixes`` (gripper + wrist camera).  Swapping the two back is a one-word typo
that leaves every suite green (nothing exercised the wrist camera's contact mask), yet it silently disables
``robot_obstacle_collision`` for the camera housing: the fix that motivated this test cost a full debugging round
because of exactly that blind spot.

The scene is a tiny, robot- and task-free MJCF model (no URDF bundle needed -- runs in CI), built the same way
``test_task_sim_clutter.py`` builds its scene.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.mjcf_scene import obstacle_body_name  # noqa: E402
from twinlink.task_sim import RobotSimSpec, TwinTaskSim  # noqa: E402
from twinlink.testing import StraightLinkage  # noqa: E402

#: A slider "arm" carrying a gripper body, which in turn carries a wrist
#: camera body -- the same nesting as the real a200-0553 profile (RG6 with a
#: D435 mounted on it), just renamed to keep the scene self-contained.
SCENE_XML = """
<mujoco model="camera_contact_mask">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="twinlink_ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="platform_base" pos="0 -1 0.1">
      <geom name="platform_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
    <body name="arm_0_shoulder_link" pos="0 0 0.08">
      <joint name="arm_0_slide" type="slide" axis="1 0 0" range="-2 2"/>
      <geom name="arm_0_geom" type="box" size="0.05 0.05 0.05"/>
      <body name="rg6_base" pos="0.12 0 0">
        <geom name="rg6_geom" type="box" size="0.04 0.04 0.04"/>
        <body name="camera_0_link" pos="0.02 0.02 0">
          <geom name="camera_0_geom" type="box" size="0.01 0.01 0.01"/>
        </body>
        <body name="rg6_hand_tcp" pos="0.05 0 0">
          <geom name="rg6_tcp_marker" type="sphere" size="0.005"
                contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

#: hand_prefixes deliberately includes "camera_0" (gripper + wrist sensor, per
#: the RobotSimSpec docstring) while gripper_prefixes does NOT -- exactly the
#: real a200-0553 profile's shape (rg6 + camera_0 vs. rg6 alone).
SPEC = RobotSimSpec(
    manipulator_prefixes=("arm_0", "rg6", "camera_0"),
    hand_prefixes=("rg6", "camera_0"),
    gripper_prefixes=("rg6",),
    far_arm_bodies=("arm_0_shoulder_link",),
    gripper_stroke_m=0.156,
    tcp_body="rg6_hand_tcp",
    arm_joints=("arm_0_slide",),
)


def _build() -> TwinTaskSim:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return TwinTaskSim(
        model,
        SPEC,
        scene_prefix="",
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def _first_geom_of_body(model, body_id: int) -> int:
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == body_id:
            return gid
    raise AssertionError(f"body {body_id} has no geom")


def test_only_the_gripper_shell_becomes_permeable_the_wrist_camera_does_not():
    sim = _build()
    try:
        grip_gid = _first_geom_of_body(sim.model, sim._body_id("rg6_base"))
        cam_gid = _first_geom_of_body(sim.model, sim._body_id("camera_0_link"))

        # The gripper shell is made permeable to graspables (see TwinTaskSim._setup_collision_masks): contype=4,
        # conaffinity=1.
        assert int(sim.model.geom_contype[grip_gid]) == 4
        assert int(sim.model.geom_conaffinity[grip_gid]) == 1

        # The wrist camera rides on the same hand assembly (it IS in hand_prefixes) but is not a jaw and must keep its
        # ordinary robot contact mask (untouched XML defaults: contype=1, conaffinity=1) -- otherwise it goes blind to
        # obstacle contacts.  This is exactly what regresses if _setup_collision_masks is changed to key off
        # spec.hand_prefixes instead of spec.gripper_prefixes.
        assert int(sim.model.geom_contype[cam_gid]) == 1
        assert int(sim.model.geom_conaffinity[cam_gid]) == 1
    finally:
        sim.close()


# --------------------------------------------------------------------- #
# the perceived-obstacle pool is a MIRROR, not a body
# --------------------------------------------------------------------- #
#: Same slider arm, plus one pool slot and one graspable free body sitting
#: well away from the arm.  Gravity off: the only thing that can move the
#: payload here is a contact force, which is exactly what is under test.
POOL_SCENE_XML = f"""
<mujoco model="pool_vs_graspable">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="twinlink_ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="arm_0_shoulder_link" pos="0 0 0.08">
      <joint name="arm_0_slide" type="slide" axis="1 0 0" range="-2 2"/>
      <geom name="arm_0_geom" type="box" size="0.05 0.05 0.05"/>
      <body name="rg6_base" pos="0.12 0 0">
        <geom name="rg6_geom" type="box" size="0.04 0.04 0.04"/>
        <body name="rg6_hand_tcp" pos="0.05 0 0">
          <geom name="rg6_tcp_marker" type="sphere" size="0.005"
                contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
    <body name="{obstacle_body_name(0, prefix="")}" pos="3 -3 0.05">
      <geom name="{obstacle_body_name(0, prefix="")}_geom" type="box"
            size="0.02 0.02 0.02"/>
    </body>
    <body name="payload" pos="1.0 0 0.26">
      <freejoint name="payload_free"/>
      <geom name="payload_geom" type="box" size="0.05 0.025 0.06"/>
    </body>
  </worldbody>
</mujoco>
"""

POOL_SPEC = RobotSimSpec(
    manipulator_prefixes=("arm_0", "rg6"),
    hand_prefixes=("rg6",),
    gripper_prefixes=("rg6",),
    far_arm_bodies=("arm_0_shoulder_link",),
    gripper_stroke_m=0.156,
    tcp_body="rg6_hand_tcp",
    arm_joints=("arm_0_slide",),
)

#: Authored pose/half-extents of the payload in POOL_SCENE_XML.
PAYLOAD_POS = np.array([1.0, 0.0, 0.26])
PAYLOAD_HALF = np.array([0.05, 0.025, 0.06])


class _PoolSim(TwinTaskSim):
    def register_graspables(self) -> None:
        self.register_graspable("payload", "payload_free", self._body_id("payload"), PAYLOAD_HALF)


class _Box:
    """Duck-typed perceived obstacle (see TwinTaskSim.set_obstacles)."""

    def __init__(self, center, size, yaw=0.0):
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        self.yaw = float(yaw)


def _build_pool_sim() -> _PoolSim:
    model = mujoco.MjModel.from_xml_string(POOL_SCENE_XML)
    return _PoolSim(
        model,
        POOL_SPEC,
        scene_prefix="",
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def test_perceived_obstacle_does_not_shove_the_object_it_mirrors():
    """A pool slot is a PERCEPTION of a body -- it must not push that body.

    In sim, every perceived obstacle over a dynamic object is written on top of the very body it was perceived from: the
    depth pipeline sees the authored distractor, the tracker mirrors it into the pool, and the slot lands exactly on the
    free-jointed original.  With ordinary contacts (contype/conaffinity 1/1) the solver resolves that overlap by
    ejecting the real body out from under its own ghost -- on the first physics step after the sync, before the arm has
    moved at all.  Measured 2026-08-12 in ``instructed_demo.py --pre-clear 'orange box'``: the 0.10x0.05x0.12 distractor
    was shot 77 mm sideways and toppled, so ClearObstacleSkill descended onto empty air and reported "grasp failed
    (nothing between the pads)" -- while perception, grounding and pad alignment were all correct.

    Same reasoning as the gripper shells above (``_setup_collision_masks``): a representation must not exert force on
    what it represents.
    """
    sim = _build_pool_sim()
    try:
        before = sim.data.qpos[sim._graspable["payload"]["qpos"] :][:3].copy()
        # Perceive the payload exactly where it stands -- what the obstacle pipeline does every survey.
        assert sim.set_obstacles([_Box(PAYLOAD_POS, 2.0 * PAYLOAD_HALF)]) == 1
        sim.step_physics(20)
        after = sim.data.qpos[sim._graspable["payload"]["qpos"] :][:3].copy()
        moved = float(np.linalg.norm(after - before))
        assert moved < 1e-6, f"the perceived box shoved its own source body by {moved * 1e3:.1f} mm"
    finally:
        sim.close()


def test_active_pool_slot_still_collides_with_the_arm():
    """The pool exists to be hit BY THE ROBOT -- that must survive the fix.

    ``arm_config_collides`` (client-side IK goal gate) and ``robot_obstacle_collision`` both read contacts between
    manipulator geoms and pool slots.  A mask change that silences pool contacts wholesale would fix the shoving above
    and blind the gate at the same time.
    """
    sim = _build_pool_sim()
    try:
        gid = sim._mujoco.mj_name2id(
            sim.model, sim._mujoco.mjtObj.mjOBJ_GEOM, f"{obstacle_body_name(0, prefix='')}_geom"
        )
        # Put the perceived box right on the arm's shoulder link.
        sim.set_obstacles([_Box([0.0, 0.0, 0.08], [0.1, 0.1, 0.1])])
        sim.step_physics(1)
        partners = {
            (
                int(sim.data.contact[i].geom1)
                if int(sim.data.contact[i].geom2) == gid
                else int(sim.data.contact[i].geom2)
            )
            for i in range(int(sim.data.ncon))
            if gid in (int(sim.data.contact[i].geom1), int(sim.data.contact[i].geom2))
        }
        arm_gid = _first_geom_of_body(sim.model, sim._body_id("arm_0_shoulder_link"))
        assert arm_gid in partners, "the arm no longer feels perceived obstacles"
    finally:
        sim.close()
