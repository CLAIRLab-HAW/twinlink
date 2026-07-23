# TwinLink

**A robot-agnostic bridge from a real robot to a digital twin in simulation.**

TwinLink keeps an in-memory `RobotState` that mirrors the *live* state of a
robot and feeds it into simulation environments, so you get a digital twin that
moves exactly like the real machine — for visualization, validation, dataset
replay or sim-in-the-loop work.

```
  ┌─────────── SOURCES ───────────┐         ┌──────────── SINKS ────────────┐
  │  Ros2Source     (live topics) │         │  MujocoSink     (implemented) │
  │  McapSource     (recording)   │──┐  ┌──▶│  IsaacSimSink   (scaffold)    │
  │  UrdfStaticSource (URDF only) │  │  │   │  ...your own                  │
  └───────────────────────────────┘  ▼  │   └───────────────────────────────┘
                              ┌──────────────────┐
                              │    RobotState    │   thread-safe, in-memory
                              │ joints · tf ·    │   "ground truth" of the twin
                              │ base_pose · cams │
                              └──────────────────┘
                                       ▲
                              ┌──────────────────┐
                              │   RobotMapping   │   YAML — the only
                              │  (topics→state)  │   robot-specific part
                              └──────────────────┘
```

The same mapping and state model drive both **live** and **mock** modes
unchanged, because `rclpy` and the MCAP reader expose messages through the same
interface.

## Modes

| Mode | Source | Needs ROS? | Use |
|------|--------|-----------|-----|
| **Live (ROS 2)** | `Ros2Source` | yes (`rclpy`) | mirror a running robot in real time |
| **Live (WebSocket)** | `FoxgloveSource` | no | mirror a robot via `foxglove_bridge` (e.g. from macOS) |
| **Live (Zenoh)** | `ZenohSource` | no | native Zenoh client for robots on `rmw_zenoh` (no ROS, robotics-grade) |
| **Uplink (Zenoh)** | `ZenohUplink` / `ZenohPublisher` | no | publish onto `rmw_zenoh` topics (keyexpr discovery via liveliness, rmw attachment) |
| **Mock — recording** | `McapSource` | no | replay an MCAP / rosbag2 recording |
| **Mock — URDF only** | `UrdfStaticSource` | no | bring a twin up from a bare URDF |

`Ros2Source` keeps its middleware specifics in three overridable methods
(`_init_node` / `_subscribe` / `_spin`), so a different transport (e.g. BabyROS)
is a small subclass — the state/mapping/decoder layers are reused.

## Install

The package itself is dependency-light (`numpy`, `pyyaml`). Pick the extras for
what you want to run:

```bash
pip install -e .            # core
pip install -e .[mcap]      # + rosbags  (replay recordings, no ROS needed)
pip install -e .[mujoco]    # + mujoco, opencv-python
pip install -e .[all]       # everything except rclpy (that comes from ROS 2)
```

> On this workstation the **system `python3`** already has `mujoco`, `rosbags`
> and `opencv`, so the demos run out of the box with `python3`.

## Examples

Runnable examples live in the sibling **[spact-integration-demos](../../apps/spact-integration-demos)**
project, which depends on this package — see its README to run them:

- **`mujoco_mcap_twin.py`** — replay an MCAP recording as a MuJoCo twin, with the
  recorded camera shown alongside. `--visual` renders the full `<visual>` meshes
  in their real per-material colours (Clearpath yellow/black, UR5 blue/grey/black)
  instead of the default collision geometry.
- **`urdf_physics_twin.py`** — URDF-only mock mode with physics: the robot is
  spawned above the ground and `MujocoSink(physics=True, spawn_height=…)` steps
  the simulation so it falls under gravity and parks on its wheels.

## Adapting to another robot

Usually **no code** — copy a mapping config (e.g. the demos'
`configs/a200_0553.yaml`) and edit the topic names:

```yaml
base_link: base_link
joint_states_topics: [ /my_robot/joint_states ]
tf_topics: [ /tf ]
odom_topic: /my_robot/odom
cameras:
  - name: head_cam
    image_topic: /my_robot/head/image_raw
    info_topic:  /my_robot/head/camera_info
joint_remap: {}          # ros_joint_name: sim_joint_name, if they differ
```

The decoders in `mapping.py` handle the standard `sensor_msgs` / `nav_msgs` /
`tf2_msgs` types, so any robot publishing those works. `MujocoSink` matches
state joints to model joints **by name**; `joint_remap` covers the rest.

## Package layout

```
twinlink/
  state.py          RobotState — the in-memory twin (thread-safe)
  mapping.py        RobotMapping + ROS→state decoders (YAML-driven)
  bridge.py         TwinLink — wires source → state → sinks
  urdf_mujoco.py    build a MuJoCo model from any URDF (mesh cache, visual/collision)
  collada.py        minimal .dae -> .obj reader (orientation-preserving)
  sources/
    ros2.py         live ROS 2 (lazy rclpy; subclass for other middleware)
    foxglove.py     live via foxglove_bridge WebSocket (FoxgloveSource; no ROS, CDR)
                    + FoxglovePublisher: uplink (client-publish onto a ROS topic)
    zenoh_source.py live via native Zenoh client (rmw_zenoh robots; no ROS)
                    + ZenohUplink/ZenohPublisher: uplink straight into the
                    rmw_zenoh graph (liveliness keyexpr discovery, rmw attachment)
    mcap.py         MCAP / rosbag2 replay (via rosbags, no ROS needed)
    urdf_static.py  URDF-only mock
  sinks/
    mujoco_sink.py  MuJoCo render + live sensor display
    isaac_sim.py    Isaac Sim adapter (run inside Isaac's python)
```

Runnable examples and robot mapping configs live in the separate
[spact-integration-demos](../../apps/spact-integration-demos) project.

## Design notes

- **Threading**: sources run in background threads and only *write* state; sinks
  are ticked from the main thread (OpenGL / macOS windowing must be on the main
  thread). State access is guarded by a single re-entrant lock.
- **Lazy imports**: `import twinlink` pulls neither `rclpy` nor `mujoco`, so the
  core and the MCAP/MuJoCo example run on a laptop without ROS.
- **MuJoCo from URDF**: `urdf_mujoco.py` strips `<gazebo>`/`<transmission>` and
  optionally adds a ground plane and a free-joint base. Two subtleties bite real
  robot descriptions and are handled here:
  - *`.dae` meshes*: MuJoCo can't read Collada, and `assimp` silently
    **re-orients** some files. `twinlink/collada.py` is a small reader that
    preserves the authored frame, so `--visual` meshes land upright. It also
    recovers each mesh's per-material diffuse colours and splits the mesh into
    one geom per material (a MuJoCo mesh has a single colour), so the twin shows
    the real robot colours instead of flat grey.
  - *mesh name clashes*: MuJoCo names a mesh by its file's basename, so two
    packages that both ship `base_link` collapse into one (the gripper would
    wear the Husky's chassis). Every mesh is cached under a **globally unique**
    name to prevent this. Degenerate (zero-volume) meshes are detected and
    dropped, falling back to collision geometry per link.
