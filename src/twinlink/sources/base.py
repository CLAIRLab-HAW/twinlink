"""Source abstraction: anything that *fills* a :class:`~twinlink.state.RobotState`.

A source is bound to a state and a mapping, then started.  It typically runs in its own thread and feeds messages
through ``mapping.apply(...)``.  Sinks read the resulting state on the main thread, so sources never touch the
simulator.
"""

from __future__ import annotations

import abc


from ..mapping import RobotMapping
from ..state import RobotState


class StateSource(abc.ABC):
    """Base class for all TwinLink sources (live and mock)."""

    #: Whether this source needs a RobotMapping to interpret messages.
    #: A bare-URDF mock writes joints directly and sets this False.
    requires_mapping: bool = True

    def __init__(self) -> None:
        self.state: RobotState | None = None
        self.mapping: RobotMapping | None = None
        self._running = False

    def bind(self, state: RobotState, mapping: RobotMapping | None = None) -> "StateSource":
        self.state = state
        self.mapping = mapping
        return self

    @abc.abstractmethod
    def start(self) -> "StateSource":
        """Begin producing state updates (usually non-blocking)."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop producing and release resources."""

    def is_running(self) -> bool:
        return self._running
