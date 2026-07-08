"""TwinLink sources: things that fill a RobotState (live or mock)."""
from .base import StateSource
from .mcap import McapSource
from .urdf_static import UrdfStaticSource

__all__ = ["StateSource", "McapSource", "UrdfStaticSource", "Ros2Source", "FoxgloveSource", "ZenohSource"]


def __getattr__(name):
    # Optional transports (rclpy / websockets / zenoh) are imported lazily so the
    # package works without ROS or extra deps installed.
    if name == "Ros2Source":
        from .ros2 import Ros2Source

        return Ros2Source
    if name == "FoxgloveSource":
        from .foxglove import FoxgloveSource

        return FoxgloveSource
    if name == "ZenohSource":
        from .zenoh_source import ZenohSource

        return ZenohSource
    raise AttributeError(name)
