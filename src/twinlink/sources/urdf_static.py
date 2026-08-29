"""URDF-only mock source.

The simplest possible "mock mode": no recording, no live robot -- just hold a static joint configuration so a digital
twin can be brought up from a URDF alone (e.g. to validate the model loads and the sink renders).
"""

from __future__ import annotations


from .base import StateSource


class UrdfStaticSource(StateSource):
    requires_mapping = False

    def __init__(self, joint_positions: dict[str, float] | None = None) -> None:
        super().__init__()
        self.joint_positions = dict(joint_positions or {})

    def start(self) -> "UrdfStaticSource":
        assert self.state is not None, "bind() before start()"
        for name, value in self.joint_positions.items():
            self.state.update_joint(name, float(value))
        self._running = True
        return self

    def stop(self) -> None:
        self._running = False
