"""Twin physics events -- the record every task app reads its rewards from.

The event record is twin-layer vocabulary (what happened inside the simulated mirror of
the robot), not task vocabulary -- every manipulation app that steps a twin
wants exactly these fields.  ``grasp_acquired``/``grasp_lost`` carry the
*object identifier* the task layer uses (hrl: the cube colour).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimEvents:
    """What happened during the last physics-step call (for rewards)."""

    robot_table_collision: bool = False
    robot_ground_collision: bool = False
    #: Robot touched a perceived obstacle (pool slot) or a sim distractor.
    robot_obstacle_collision: bool = False
    grasp_acquired: Optional[str] = None  # object id captured this step
    grasp_lost: Optional[str] = None  # object released/dropped this step
    #: Objekt-Id, deren Griff NICHT FESTSTELLBAR war (Tool-DI0 lieferte
    #: nichts).  Dritter Zustand neben acquired/lost -- „unbekannt" darf
    #: nicht als „nicht gegriffen" verbucht werden.
    grasp_unknown: Optional[str] = None

    def merge(self, other: "SimEvents") -> None:
        self.robot_table_collision |= other.robot_table_collision
        self.robot_ground_collision |= other.robot_ground_collision
        self.robot_obstacle_collision |= other.robot_obstacle_collision
        self.grasp_acquired = other.grasp_acquired or self.grasp_acquired
        self.grasp_lost = other.grasp_lost or self.grasp_lost
        self.grasp_unknown = other.grasp_unknown or self.grasp_unknown
