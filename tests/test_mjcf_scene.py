"""Scene-furniture body-name prefix is a call-time parameter, not a constant.

twinlink is a robot- AND app-agnostic package -- it must not bake in the
name of any particular task app.  As a module constant, every prefix
(``hrl_obstacle_``, ``hrl_distractor_``) would fit only hrl, and a second app
authoring its own obstacle pool would collide with hrl's body names in a
shared model.
"""
from __future__ import annotations

from twinlink import mjcf_scene


def test_scene_prefix_is_a_parameter_not_a_constant():
    """The lib must not hard-code the app's name."""
    assert mjcf_scene.obstacle_body_name(0) == "hrl_obstacle_0"  # default
    assert mjcf_scene.obstacle_body_name(0, prefix="foo_") == "foo_obstacle_0"


def test_distractor_names_take_the_same_prefix():
    assert mjcf_scene.distractor_body_name(2) == "hrl_distractor_2"
    assert mjcf_scene.distractor_body_name(2, prefix="foo_") == "foo_distractor_2"
    assert mjcf_scene.distractor_joint_name(2) == "hrl_distractor_2_free"
    assert mjcf_scene.distractor_joint_name(2, prefix="foo_") == "foo_distractor_2_free"


def test_default_prefix_reproduces_the_pre_task11_constants():
    """``OBSTACLE_BODY_PREFIX`` / ``DISTRACTOR_BODY_PREFIX`` keep their exact
    values -- existing consumers (octomap_explorer, spact-integration-demos)
    that import and compare against them directly must not notice this
    refactor."""
    assert mjcf_scene.OBSTACLE_BODY_PREFIX == "hrl_obstacle_"
    assert mjcf_scene.DISTRACTOR_BODY_PREFIX == "hrl_distractor_"
    assert mjcf_scene.obstacle_body_name(3) == f"{mjcf_scene.OBSTACLE_BODY_PREFIX}3"
    assert mjcf_scene.distractor_body_name(3) == f"{mjcf_scene.DISTRACTOR_BODY_PREFIX}3"


def test_add_obstacle_pool_honours_a_custom_prefix():
    import xml.etree.ElementTree as ET

    worldbody = ET.Element("worldbody")
    mjcf_scene.add_obstacle_pool(worldbody, 2, prefix="foo_")
    names = [b.get("name") for b in worldbody.findall("body")]
    assert names == ["foo_obstacle_0", "foo_obstacle_1"]


def test_add_distractors_honours_a_custom_prefix():
    import xml.etree.ElementTree as ET

    worldbody = ET.Element("worldbody")
    mjcf_scene.add_distractors(
        worldbody,
        [{"position": (0.0, 0.0, 0.0), "size": (0.1, 0.1, 0.1)}],
        prefix="foo_",
    )
    body = worldbody.find("body")
    assert body.get("name") == "foo_distractor_0"
