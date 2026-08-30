"""The task-world seam: what every world does, and what only a world WITH truth does.

Two Protocols and not one, because the simulator has two jobs.  In a real-hardware run the MuJoCo instance is still
built, reset and stepped -- but it is no longer the world: the arm's joint stream is mirrored into it and the object
bodies are written to where perception BELIEVES the objects are.  Asking such a world for an object pose returns
one's own belief, and a single protocol carrying that read would make the ambiguity an interface.  That is the same
defect ``skill_tree.ros_backend`` refuses by construction with its fused ``gripper_state``.

**What ``isinstance`` can and cannot settle here.**  Both protocols are ``runtime_checkable``, so a world can be
CHECKED for the methods -- useful when a fourth world appears and one wants to know what it brings.  It cannot
settle the ROLE: a display twin is the very same class as the world it mirrors and carries the same methods.  The
role is a property of the USE, not of the type, so whoever builds the world decides it and hands ``GroundTruth |
None`` onward.  ``None`` is then the statement "there is nothing here to verify against" -- made once, where it is
known, instead of re-derived at every call site.

The two surfaces are measured, not chosen (2026-08-30): they are what ``twinlink.task_sim.TwinTaskSim`` and
``twinlink.maniskill_sim.ManiSkillTaskSim`` actually share.  ``settle`` and ``close`` are deliberately absent --
only the MuJoCo world has them -- and so is ``grasped_label``, which would exclude ManiSkill from ``GroundTruth``
for a read the study does not take its verdict from.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TaskWorld(Protocol):
    """What every task world does, truth or not: run, be looked at, and say where the robot is."""

    def sim_time_s(self) -> float:
        """Elapsed SIMULATED seconds -- read as a difference, never as an epoch."""

    def step_physics(self, n: int = 1):
        """Advance the world by ``n`` control periods and report what happened."""

    def tcp_pose(self):
        """Where the tool centre point is, as ``(position, orientation)``."""

    def arm_positions(self) -> dict[str, float]:
        """The arm joints as the world currently holds them."""

    def gripper_width_m(self) -> float:
        """Clear width between the jaws [m]."""

    def render_rgb(self, camera: str):
        """RGB image of a named camera, or ``None`` when the world has none."""

    def camera_matrix(self, camera: str):
        """Intrinsics of a named camera, or ``None``."""

    def camera_pose(self, camera: str):
        """Extrinsics of a named camera, or ``None``."""


@runtime_checkable
class GroundTruth(Protocol):
    """The truth a world can give -- for a verdict, never for a planning scene.

    One method, because one is what both worlds share.  A caller holding ``GroundTruth | None`` is being told
    whether there is anything to verify against at all; see the module docstring for why that cannot be an
    ``isinstance`` check.
    """

    def object_poses(self) -> dict:
        """True poses of the task objects -- ground truth, for the verdict only."""
