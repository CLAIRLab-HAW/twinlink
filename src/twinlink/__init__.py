"""TwinLink -- a robot-agnostic bridge from a real robot to a digital twin.

TwinLink keeps an in-memory :class:`RobotState` that mirrors the live robot and feeds it into simulation environments
(MuJoCo today, Isaac Sim next).  The state can be filled from any of five interchangeable *sources*:

* :class:`~twinlink.sources.ros2.Ros2Source`         -- live ROS 2 topics (needs ``rclpy``),
* :class:`~twinlink.sources.foxglove.FoxgloveSource` -- live through a ``foxglove_bridge`` WebSocket, no ROS,
* :class:`~twinlink.sources.zenoh_source.ZenohSource` -- live as a native Zenoh client, no ROS,
* :class:`McapSource`       -- a recorded MCAP / rosbag2 (mock mode),
* :class:`UrdfStaticSource` -- a bare URDF, no motion (mock mode).

and rendered by one or more *sinks* (``MujocoSink``, ``IsaacSimSink``).  A
:class:`RobotMapping` (usually a YAML file) describes how a particular robot's
topics map into the state, so adapting to a new robot needs no code change.

Quick start (mock mode, MuJoCo)::

    from twinlink import TwinLink, RobotMapping, McapSource
    from twinlink.sinks.mujoco_sink import MujocoSink
    from twinlink.urdf_mujoco import load_mujoco_from_urdf

    mapping = RobotMapping.from_yaml("configs/a200_0553.yaml")  # from the demos project
    model = load_mujoco_from_urdf("urdf/robot.urdf")
    TwinLink(
        mapping=mapping,
        source=McapSource("record_2026_06_25_3", rate=1.0),
        sinks=[MujocoSink(model, show_sensor_camera="camera_0")],
    ).run()

The full runnable example is ``mujoco_mcap_twin.py`` in the sibling ``spact-integration-demos`` project, which also
carries the robot mapping configs.
"""

from __future__ import annotations

from .bridge import TwinLink
from .mapping import CameraMap, RobotMapping
from .sources.mcap import McapSource
from .sources.urdf_static import UrdfStaticSource
from .state import CameraFrame, JointState, RobotState, Transform

__version__ = "0.1.0"
from .task_world import GroundTruth, SceneView, TaskWorld  # noqa: F401

__all__ = [
    "GroundTruth",
    "SceneView",
    "TaskWorld",
    "TwinLink",
    "RobotState",
    "RobotMapping",
    "CameraMap",
    "Transform",
    "JointState",
    "CameraFrame",
    "McapSource",
    "UrdfStaticSource",
    # lazily exported (optional deps): Ros2Source, FoxgloveSource, ZenohSource, ZenohUplink, ZenohPublisher, MujocoSink,
    # IsaacSimSink, load_mujoco_from_urdf
]


def __getattr__(name):
    if name == "Ros2Source":
        from .sources.ros2 import Ros2Source

        return Ros2Source
    if name == "FoxgloveSource":
        from .sources.foxglove import FoxgloveSource

        return FoxgloveSource
    if name in ("ZenohSource", "ZenohUplink", "ZenohPublisher"):
        from .sources import zenoh_source

        return getattr(zenoh_source, name)
    if name == "MujocoSink":
        from .sinks.mujoco_sink import MujocoSink

        return MujocoSink
    if name == "IsaacSimSink":
        from .sinks.isaac_sim import IsaacSimSink

        return IsaacSimSink
    if name == "load_mujoco_from_urdf":
        from .urdf_mujoco import load_mujoco_from_urdf

        return load_mujoco_from_urdf
    raise AttributeError(f"module 'twinlink' has no attribute {name!r}")
