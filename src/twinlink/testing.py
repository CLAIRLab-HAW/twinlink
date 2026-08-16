"""Testdoppel für die Protokolle, die :mod:`twinlink.task_sim` hereingereicht bekommt.

Kein Produktionscode: was hier steht, dient dazu, die VERDRAHTUNG der Sim zu
prüfen, ohne einen bestimmten Roboter vorauszusetzen.  Es liegt im Paket und
nicht neben den Tests, weil der Workspace mit ``--import-mode=importlib``
läuft -- ein Modul neben den Testdateien wäre von dort nicht importierbar.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StraightLinkage:
    """Ein DOPPEL für :class:`twinlink.task_sim.GripperLinkage`, keine Geometrie.

    Bewusst eine Gerade und bewusst mit runden Zahlen: diese Suiten prüfen die
    Verdrahtung der Sim, nicht die Kinematik eines bestimmten Greifers.  Die
    echte Abbildung ist ein Kosinus und steht im Roboterprofil
    (``robot_contract``, ``gripper.linkage``); sie hier nachzubauen wäre eine
    vierte Kopie derselben Formel -- genau das, was der Umbau vom 2026-08-16
    abgeschafft hat.

    ``width = (closed_rad - q) * 0.2``, also 0 m bei q = 0,6 und 0,16 m bei
    q = -0,2.
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
