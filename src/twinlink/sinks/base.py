"""Sink abstraction: anything that *consumes* a :class:`RobotState`.

A sink pulls the current state on every tick of the bridge loop and pushes it
into a simulator (MuJoCo, Isaac Sim, ...).  ``update()`` returns ``False`` to
ask the bridge to stop (e.g. the viewer window was closed).
"""
from __future__ import annotations

import abc
from typing import Optional

from ..state import RobotState


class StateSink(abc.ABC):
    def __init__(self) -> None:
        self.state: Optional[RobotState] = None

    def bind(self, state: RobotState) -> "StateSink":
        self.state = state
        return self

    def setup(self) -> None:
        """One-time initialisation (load model, open window, ...)."""

    @abc.abstractmethod
    def update(self) -> bool:
        """Push the latest state into the sim. Return False to request stop."""

    def close(self) -> None:
        """Release resources."""
