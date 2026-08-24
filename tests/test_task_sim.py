"""RobotSimSpec: twinlink bleibt roboter-agnostisch."""

from __future__ import annotations

import numpy as np
import pytest

from twinlink.testing import StraightLinkage  # noqa: E402
from twinlink.task_sim import RobotSimSpec


def test_spec_is_plain_data():
    spec = RobotSimSpec(
        manipulator_prefixes=("a", "b"),
        hand_prefixes=("b",),
        gripper_prefixes=("b",),
        far_arm_bodies=("a_link",),
        gripper_stroke_m=0.1,
        tcp_body="tcp",
        arm_joints=("j1", "j2"),
    )
    assert spec.manipulator_prefixes == ("a", "b")
    assert spec.gripper_stroke_m == 0.1


def test_gripper_prefixes_are_separate_from_hand_prefixes():
    """The jaws are a DIFFERENT set from the hand assembly.

    The hand may contain sensors riding along with it (wrist camera); the jaws may not, because only they are made
    permeable for graspable objects.  Were both the same field, the camera housing would lose its collision events
    (regression from the sim split of 2026-07-31).
    """
    fields = RobotSimSpec.__dataclass_fields__
    assert "gripper_prefixes" in fields
    assert "hand_prefixes" in fields


def test_twinlink_stands_alone():
    """The package's independence is part of the contract.

    twinlink is a self-contained MIT package with its own CI: neither the robot profile (``robot_contract``) nor its SDK
    (``husky_sdk``) nor the task app (``hrl``) may be imported here.  Robot facts enter exclusively as
    :class:`RobotSimSpec` through the constructor; any task knowledge (cubes, tower, RL) belongs exclusively to hrl,
    never the other way round.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "twinlink"
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text()
        # ``perception`` belongs in the list too: it is the third layer above twinlink (perception/obstacle tracking)
        # and therefore just as much a backwards import as hrl.  Were it the only one missing from this list, it would
        # slip in unnoticed.
        for package in ("robot_contract", "husky_sdk", "hrl", "perception"):
            if f"import {package}" in text or f"from {package}" in text:
                offenders.append(f"{path.name}: {package}")
    assert offenders == []


def test_grasp_registry_is_label_keyed():
    """Graspable objects are carried by labels, not by colours.

    The WHOLE module is checked, not just the class body: task vocabulary otherwise hides in module constants and helper
    functions beside it.

    The app prefix ``hrl_`` in this word list would be green even though ``task_sim`` classifies against exactly that
    prefix: the module IMPORTED the constants (``OBSTACLE_BODY_PREFIX`` &c.) instead of writing the literal, so the text
    search found nothing.  A text scan fundamentally cannot prove this property; the proof is now carried by
    ``test_scene_prefix_drives_classification`` over the VALUES.
    """
    import inspect

    from twinlink import task_sim

    src = inspect.getsource(task_sim)
    # The task vocabulary must no longer reach the mechanism.
    for word in ("cube", "CUBE", "color", "tower"):
        assert word not in src, f"Task-Begriff {word!r} in twinlink.task_sim"


#: Probe scene for the prefix proof: floor, a slider "arm" with a gripper
#: child body, one pool slot (``…obstacle_0``) and one graspable distractor
#: (``…distractor_0``) -- everything ``_classify_geoms`` and
#: ``_index_obstacle_pool`` hang off the prefix.  Robot- and task-free, so that
#: no URDF bundle is needed.
_PROBE_SCENE = """
<mujoco model="scene_prefix_probe">
  <option timestep="0.002"/>
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
    <body name="{p}obstacle_0" pos="1.2 0 0.05">
      <geom name="{p}obstacle_0_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="{p}distractor_0" pos="0.6 0 0.05">
      <freejoint name="{p}distractor_0_free"/>
      <geom name="{p}distractor_0_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

_PROBE_SPEC = RobotSimSpec(
    manipulator_prefixes=("arm_0", "rg6"),
    hand_prefixes=("rg6",),
    gripper_prefixes=("rg6",),
    far_arm_bodies=("arm_0_shoulder_link",),
    gripper_stroke_m=0.156,
    tcp_body="rg6_hand_tcp",
    arm_joints=("arm_0_slide",),
)


def _classification_under_prefix(prefix: str) -> dict:
    """Build the probe scene under ``prefix`` and read back its classification."""
    import mujoco

    from twinlink.task_sim import TwinTaskSim

    class _ProbeSim(TwinTaskSim):
        def register_graspables(self) -> None:
            self.register_graspable(
                "clutter", f"{prefix}distractor_0_free", self._body_id(f"{prefix}distractor_0"), np.full(3, 0.05)
            )

    model = mujoco.MjModel.from_xml_string(_PROBE_SCENE.format(p=prefix))
    sim = _ProbeSim(
        model,
        _PROBE_SPEC,
        scene_prefix=prefix,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )
    try:
        return {
            "obstacle_geoms": len(sim._obstacle_geoms),
            "pool_slots": len(sim._obstacle_slots),
            "non_obstacle_graspables": sim._non_obstacle_graspables,
        }
    finally:
        sim.close()


def test_scene_prefix_drives_classification():
    """The constructor prefix -- not ``hrl_`` -- decides the classification.

    The core promise of the rework: twinlink is app-agnostic and publishable.  It only half holds if ``__init__`` uses
    the ``scene_prefix`` merely for the render split while ``_classify_geoms`` compares against the module constants and
    ``_index_obstacle_pool`` calls ``obstacle_body_name(i)`` without ``prefix=``.  Identical scene, only the prefix
    swapped, then gives:

        prefix 'hrl_' :  obstacle_geoms=2  pool_slots=1  non_obstacle_graspables=()
        prefix 'task_':  obstacle_geoms=0  pool_slots=0  non_obstacle_graspables=('clutter',)

    -- a second consumer would have been silently blind to the entire obstacle
    class (the same blindness that once already cost Task 10 a round of fixes).
    VALUES are checked, not module text: a text scan for ``"hrl_"`` stays green
    as long as the module imports the constants.
    """
    pytest.importorskip("mujoco", reason="mujoco extra not installed")

    native = _classification_under_prefix("hrl_")
    foreign = _classification_under_prefix("task_")

    # Nailed down absolutely, so the comparison does not turn trivially green when both sides classify nothing at all.
    assert native == {
        "obstacle_geoms": 2,  # Pool-Slot + Distraktor
        "pool_slots": 1,
        "non_obstacle_graspables": (),  # the distractor REMAINS an obstacle
    }
    assert foreign == native


def test_hooks_are_abstract_enough_to_subclass():
    from twinlink.task_sim import TwinTaskSim

    assert hasattr(TwinTaskSim, "register_graspables")
    assert hasattr(TwinTaskSim, "support_geom_names")
