"""TwinLink sinks: things that render a RobotState into a simulator."""
from .base import StateSink

__all__ = ["StateSink", "MujocoSink", "IsaacSimSink"]


def __getattr__(name):
    # Simulator backends are optional heavy deps; import lazily.
    if name == "MujocoSink":
        from .mujoco_sink import MujocoSink

        return MujocoSink
    if name == "IsaacSimSink":
        from .isaac_sim import IsaacSimSink

        return IsaacSimSink
    raise AttributeError(name)
