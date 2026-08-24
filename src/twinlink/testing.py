"""Test doubles for the protocols handed into :mod:`twinlink.task_sim`.

Not production code: what stands here serves to check the WIRING of the sim without presupposing one particular robot.
It lives in the package and not next to the tests because the workspace runs with ``--import-mode=importlib`` -- a
module next to the test files would not be importable from there.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StraightLinkage:
    """A DOUBLE for :class:`twinlink.task_sim.GripperLinkage`, no geometry.

    Deliberately a straight line and deliberately with round numbers: these suites check the wiring of the sim, not the
    kinematics of one particular gripper.  The real mapping is a cosine and lives in the robot profile
    (``robot_contract``, ``gripper.linkage``); rebuilding it here would be a fourth copy of the same formula -- exactly
    what the rework of 2026-08-16 did away with.

    ``width = (closed_rad - q) * 0.2``, so 0 m at q = 0.6 and 0.16 m at q = -0.2.
    """

    closed_rad: float = 0.6
    max_width_m: float = 0.16
    _per_rad: float = 0.2

    @property
    def open_rad(self) -> float:
        return self.closed_rad - self.max_width_m / self._per_rad

    def width_from_angle(self, q: float) -> float:
        return (self.closed_rad - float(q)) * self._per_rad

    def angle_from_width(self, width_m: float) -> float:
        w = min(max(float(width_m), 0.0), self.max_width_m)
        return self.closed_rad - w / self._per_rad
