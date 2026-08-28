"""Fixtures that reach outside the package: the URDF bundle and the container's SRDF.

``twinlink`` is robot-agnostic, so neither artefact belongs in the package -- the tests locate them through the
workspace marker, the same way the ``_bootstrap`` modules do.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "workspace.repos").is_file():
            return parent
    raise RuntimeError("workspace.repos not found above the test file")


@pytest.fixture(scope="session")
def urdf_bundle() -> Path:
    bundle = _workspace_root() / "urdf"
    if not (bundle / "robot.urdf").is_file():
        pytest.skip("urdf/robot.urdf is absent -- run urdf/generate.sh --from-container")
    return bundle


@pytest.fixture(scope="session")
def robot_srdf(tmp_path_factory) -> Path:
    """The SRDF ``move_group`` loads, copied out of the running mock-robot container.

    The oracle must read THIS file: the whole reason for choosing Pinocchio over a second MuJoCo model is that its
    disabled-pair set then agrees with the planner by construction instead of by measurement.
    """
    out = tmp_path_factory.mktemp("srdf") / "robot.srdf"
    copy = subprocess.run(
        ["docker", "cp", "husky-offboard-mock-robot-1:/clearpath/robot.srdf", str(out)],
        capture_output=True,
    )
    if copy.returncode != 0 or not out.is_file():
        pytest.skip("mock-robot container does not answer -- start it with docker compose --profile mock up -d")
    return out


ARM_JOINTS = (
    "arm_0_shoulder_pan_joint",
    "arm_0_shoulder_lift_joint",
    "arm_0_elbow_joint",
    "arm_0_wrist_1_joint",
    "arm_0_wrist_2_joint",
    "arm_0_wrist_3_joint",
)


def _register_test_agent() -> None:
    """Register the smallest ManiSkill agent over the a200 URDF, once.

    It lives here rather than in a sibling module because no package in this workspace ships a ``tests/__init__.py``
    -- without one a sibling import has no package to resolve against, and adding one would put the test suite into
    the wheel.  The real agent lives in ``apps/maniskill-eval``; this one exists so ``ManiSkillTaskSim`` is testable
    without it, which is the whole reason the environment is injected rather than built inside the world.
    """
    global _AGENT_REGISTERED
    if _AGENT_REGISTERED:
        return
    from mani_skill.agents.base_agent import BaseAgent
    from mani_skill.agents.controllers import PDJointPosControllerConfig
    from mani_skill.agents.registration import register_agent

    bundle = _workspace_root() / "urdf"

    @register_agent()
    class A200TestAgent(BaseAgent):
        uid = "a200_test"
        urdf_path = str(bundle / "robot.urdf")
        fix_root_link = True

        @property
        def _controller_configs(self):
            return dict(
                pd_joint_pos=dict(
                    arm=PDJointPosControllerConfig(
                        list(ARM_JOINTS),
                        lower=None,
                        upper=None,
                        stiffness=1000.0,
                        damping=100.0,
                        force_limit=200.0,
                        normalize_action=False,
                    )
                )
            )

    _AGENT_REGISTERED = True


_AGENT_REGISTERED = False


def _build_world(urdf_bundle, robot_srdf, *, owns_tick: bool):
    """A minimal ManiSkill env with the a200 as its agent -- no task objects, those belong to the app.

    The agent is registered in the test package itself: ``twinlink`` may not import ``maniskill_eval``, and the world
    takes its environment injected precisely so the two can be tested apart.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  -- registers Empty-v1

    from twinlink.maniskill_sim import ManiSkillTaskSim
    from twinlink.pin_kinematics import PinocchioKinematics

    _register_test_agent()

    kinematics = PinocchioKinematics(
        str(urdf_bundle / "robot.urdf"),
        str(robot_srdf),
        mesh_dir=str(urdf_bundle),
        joints=ARM_JOINTS,
        tcp_frame="rg6_hand_tcp",
    )
    env = gym.make(
        "Empty-v1",
        robot_uids="a200_test",
        num_envs=1,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
    )
    env.reset(seed=0)
    from robot_contract import load_profile

    gripper = load_profile().gripper
    return ManiSkillTaskSim(
        env,
        arm_joints=ARM_JOINTS,
        kinematics=kinematics,
        control_dt=0.02,
        owns_tick=owns_tick,
        gripper_follower_factors=dict(gripper.follower_factors),
        gripper_linkage=gripper.linkage,
        gripper_driver_joint=gripper.driver_joint,
        home_pose=load_profile().pose("ready"),
    )


def _world_or_skip(urdf_bundle, robot_srdf, *, owns_tick: bool):
    """Build the world, or skip with the reason named.

    SAPIEN needs a Vulkan driver, and macOS has none until ``brew install molten-vk`` provides the ICD.  Without
    ``VK_ICD_FILENAMES`` it raises ``vk::createInstanceUnique: ErrorIncompatibleDriver`` -- a hard error in the
    default root run, on a machine that is simply not set up for this route.  A red test that means "no GPU here"
    trains people to ignore red tests, so it skips and says what to do.
    """
    pytest.importorskip("sapien")
    pytest.importorskip("mani_skill")
    try:
        return _build_world(urdf_bundle, robot_srdf, owns_tick=owns_tick)
    except RuntimeError as exc:
        if "vk::" not in str(exc) and "rendering device" not in str(exc):
            raise
        pytest.skip(
            "no Vulkan device: brew install molten-vk, then export "
            "VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
        )


@pytest.fixture(scope="module")
def world_in_process(urdf_bundle, robot_srdf):
    return _world_or_skip(urdf_bundle, robot_srdf, owns_tick=True)


@pytest.fixture(scope="module")
def world_ros_route(urdf_bundle, robot_srdf):
    return _world_or_skip(urdf_bundle, robot_srdf, owns_tick=False)
