"""Both worlds satisfy the seam -- and the seam does not pretend to settle the role."""

from __future__ import annotations

import pytest

from twinlink.task_world import GroundTruth, TaskWorld

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.maniskill_sim import ManiSkillTaskSim  # noqa: E402
from twinlink.task_sim import TwinTaskSim  # noqa: E402


@pytest.mark.parametrize("world", [TwinTaskSim, ManiSkillTaskSim])
def test_both_worlds_carry_the_whole_seam(world):
    """Measured against the CLASSES: the protocols are what the two actually share, not a wish."""
    for protocol in (TaskWorld, GroundTruth):
        missing = [m for m in protocol.__protocol_attrs__ if not hasattr(world, m)]
        assert not missing, f"{world.__name__} is missing {missing} of {protocol.__name__}"


def test_the_protocol_does_not_settle_the_role():
    """A display twin is the SAME class as the world it mirrors, so isinstance cannot tell them apart.

    This is pinned rather than merely written down: somebody will reach for ``isinstance(world, GroundTruth)`` as
    the real-mode check, and it would answer True for a world whose object poses are this app's own beliefs.
    """

    class _World:
        def object_poses(self) -> dict:
            return {}

    twin_used_as_a_mirror = _World()
    assert isinstance(twin_used_as_a_mirror, GroundTruth), "the methods are there -- the role is not a type"
