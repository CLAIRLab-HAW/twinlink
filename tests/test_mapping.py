"""RobotMapping — YAML loading, topic routing, and the ROS→state decoders.

The decoders are duck-typed (they only read message attributes), so the tests
feed lightweight namespace fakes instead of real ROS messages — no rosbags /
rclpy needed, matching twinlink's dependency-light core.
"""
import struct
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from twinlink import RobotMapping, RobotState
from twinlink.mapping import CameraMap, pointcloud2_to_xyz


def _stamp(sec=1, nanosec=500_000_000):
    return NS(sec=sec, nanosec=nanosec)


def _header(frame="base_link", **kw):
    return NS(stamp=_stamp(**kw), frame_id=frame)


# --------------------------------------------------------------------------- #
# construction + routing
# --------------------------------------------------------------------------- #
def _mapping(**kw):
    defaults = dict(
        joint_states_topics=["/r/joint_states"],
        tf_topics=["/tf"],
        tf_static_topics=["/tf_static"],
        odom_topic="/r/odom",
        base_pose_relative_to_start=True,
        cameras=[CameraMap(name="cam", image_topic="/r/img", info_topic="/r/info")],
    )
    defaults.update(kw)
    return RobotMapping(**defaults)


def test_topics_and_roles():
    m = _mapping()
    topics = m.topics()
    for t in ("/r/joint_states", "/tf", "/tf_static", "/r/odom", "/r/img", "/r/info"):
        assert t in topics
    assert m.role_of("/r/joint_states") == "joint_states"
    assert m.role_of("/tf") in ("tf", "tf_static")
    assert m.role_of("/unknown") is None
    assert m.topic_type("/r/joint_states") == "sensor_msgs/msg/JointState"


def test_from_yaml_loads_the_a200_mapping():
    """The real robot mapping YAML must keep loading (workspace fixture)."""
    ws = Path(__file__).resolve()
    for cand in ws.parents:
        if (cand / "workspace.repos").is_file():
            yaml_path = (
                cand / "apps" / "spact-integration-demos" / "configs" / "a200_0553.yaml"
            )
            break
    else:
        pytest.skip("workspace.repos not found (standalone checkout)")
    if not yaml_path.exists():
        pytest.skip("a200_0553 mapping YAML not checked out")
    m = RobotMapping.from_yaml(str(yaml_path))
    assert "/a200_0553/platform/joint_states" in m.topics()
    assert m.base_link == "base_link"
    assert m.base_pose_relative_to_start is True


# --------------------------------------------------------------------------- #
# decoders (via apply)
# --------------------------------------------------------------------------- #
def test_decode_joint_states_with_remap():
    m = _mapping(joint_remap={"ros_a": "sim_a"})
    s = RobotState()
    msg = NS(
        header=_header(),
        name=["ros_a", "b"],
        position=[0.5, 1.5],
        velocity=[0.1, 0.2],
        effort=[],
    )
    m.apply("/r/joint_states", "sensor_msgs/msg/JointState", msg, s)
    assert s.joint_position("sim_a") == 0.5       # remapped
    assert s.joint_position("b") == 1.5
    assert s.joint("b").velocity == 0.2
    assert s.joint("b").effort is None            # empty effort list
    assert s.joint("b").stamp == pytest.approx(1.5)


def test_decode_tf():
    m = _mapping()
    s = RobotState()
    tr = NS(
        header=_header(frame="odom"),
        child_frame_id="base_link",
        transform=NS(
            translation=NS(x=1.0, y=2.0, z=3.0),
            rotation=NS(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    m.apply("/tf", "tf2_msgs/msg/TFMessage", NS(transforms=[tr]), s)
    got = s.transform("odom", "base_link")
    assert got is not None
    assert got.translation.tolist() == [1.0, 2.0, 3.0]


def test_decode_odom_relative_to_start():
    """UTM-style absolute odometry must be re-zeroed on the first sample."""
    m = _mapping()
    s = RobotState()

    def odom(x, y):
        return NS(
            header=_header(frame="odom"),
            child_frame_id="base_link",
            pose=NS(pose=NS(
                position=NS(x=x, y=y, z=0.0),
                orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0),
            )),
        )

    m.apply("/r/odom", "nav_msgs/msg/Odometry", odom(374903.0, -54651.0), s)
    first = s.base_pose().translation
    assert first.tolist() == [0.0, 0.0, 0.0]      # origin re-zeroed
    m.apply("/r/odom", "nav_msgs/msg/Odometry", odom(374904.0, -54650.0), s)
    second = s.base_pose().translation
    assert second.tolist() == [1.0, 1.0, 0.0]     # motion relative to start


def test_decode_camera_info():
    """Without a frame the K lands in extras; with one, on the CameraFrame."""
    from twinlink.state import CameraFrame

    m = _mapping()
    s = RobotState()
    K = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
    m.apply("/r/info", "sensor_msgs/msg/CameraInfo", NS(header=_header(), k=K), s)
    assert s.extra("camera_info/cam")[0, 0] == 600.0   # parked until a frame exists

    s.set_camera("cam", CameraFrame(image=np.zeros((2, 2, 3), np.uint8),
                                    encoding="rgb8", stamp=0.0, frame_id="f",
                                    width=2, height=2))
    m.apply("/r/info", "sensor_msgs/msg/CameraInfo", NS(header=_header(), k=K), s)
    assert s.camera("cam").intrinsics[0, 0] == 600.0


# --------------------------------------------------------------------------- #
# pointcloud2_to_xyz (the parser under the obstacles/points path)
# --------------------------------------------------------------------------- #
def _pointcloud(points_xyz, extra_nan=0):
    pts = list(points_xyz) + [(float("nan"),) * 3] * extra_nan
    step = 12
    data = b"".join(struct.pack("<fff", *p) for p in pts)
    fields = [NS(name=n, offset=o, datatype=7)  # 7 = FLOAT32
              for n, o in (("x", 0), ("y", 4), ("z", 8))]
    return NS(
        header=_header(frame="camera"),
        fields=fields, point_step=step, width=len(pts), height=1,
        data=np.frombuffer(data, dtype=np.uint8),
    )


def test_pointcloud2_to_xyz_drops_invalid():
    msg = _pointcloud([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], extra_nan=3)
    xyz = pointcloud2_to_xyz(msg)
    assert xyz.shape == (2, 3)
    assert xyz[1].tolist() == [4.0, 5.0, 6.0]


def test_decode_points_into_obstacles():
    m = _mapping(points_topics={"cloud": "/r/points"})
    s = RobotState()
    m.apply("/r/points", "sensor_msgs/msg/PointCloud2",
            _pointcloud([(0.0, 0.0, 1.0)]), s)
    cloud = s.obstacles("cloud")
    assert cloud is not None and cloud.points.shape == (1, 3)
    assert cloud.frame_id == "camera"


# --------------------------------------------------------------------------- #
# lazy CompressedImage decode (CameraMap.lazy_decode -> CameraFrame.ensure_decoded)
# --------------------------------------------------------------------------- #
_COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"


def _jpeg_msg(bgr):
    cv2 = pytest.importorskip("cv2")
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return NS(header=_header(frame="cam_frame"), format="rgb8; jpeg compressed bgr8",
              data=buf.reshape(-1))


def _depth_png_msg(depth_mm):
    cv2 = pytest.importorskip("cv2")
    ok, buf = cv2.imencode(".png", depth_mm)
    assert ok
    header = struct.pack("<Iff", 0, 100.0, 0.0)  # ConfigHeader: enum + depthParam
    return NS(header=_header(frame="depth_frame"),
              format="16UC1; compressedDepth png",
              data=np.frombuffer(header + buf.tobytes(), np.uint8))


def _lazy_mapping():
    return _mapping(cameras=[
        CameraMap(name="cam", image_topic="/r/img", info_topic="/r/info",
                  lazy_decode=True),
        CameraMap(name="depth", image_topic="/r/dimg", is_depth=True,
                  lazy_decode=True),
    ])


def test_lazy_decode_color_roundtrip():
    m, s = _lazy_mapping(), RobotState()
    m.apply("/r/img", _COMPRESSED_TYPE, _jpeg_msg(np.full((4, 6, 3), 128, np.uint8)), s)
    frame = s.camera("cam")
    assert frame.image is None and frame.raw is not None  # ingest did NOT decode
    assert frame.arrival_monotonic > 0.0
    assert frame.ensure_decoded() is True
    assert frame.image.shape == (4, 6, 3) and frame.encoding == "bgr8"
    assert frame.raw is None                              # payload freed
    assert frame.ensure_decoded() is True                 # idempotent


def test_lazy_decode_depth_png_roundtrip():
    m, s = _lazy_mapping(), RobotState()
    depth = np.full((4, 6), 512, np.uint16)
    m.apply("/r/dimg", _COMPRESSED_TYPE, _depth_png_msg(depth), s)
    frame = s.camera("depth")
    assert frame.image is None and frame.is_depth is True
    assert frame.ensure_decoded() is True
    assert frame.encoding == "16uc1"
    assert frame.image.dtype == np.uint16 and int(frame.image[0, 0]) == 512


def test_lazy_rvl_rejected_live():
    """The pure-Python RVL decoder is unusable live: the lazy path refuses it."""
    from twinlink.mapping import decode_compressed_bytes
    from twinlink.state import CameraFrame

    payload = struct.pack("<Iff", 0, 100.0, 0.0) + struct.pack("<II", 2, 2) + b"\x00" * 8
    with pytest.raises(ValueError, match="RVL"):
        decode_compressed_bytes(payload, "16UC1; compressedDepth rvl",
                                is_depth=True, allow_rvl=False)
    frame = CameraFrame(image=None, raw=payload,
                        raw_format="16UC1; compressedDepth rvl", is_depth=True)
    assert frame.ensure_decoded() is False   # logged, not raised
    assert frame.raw is None                 # poisoned payload not retried


def test_camera_info_cached_but_reattaches_after_clear():
    """K is parsed once, but a cleared camera entry gets it re-applied."""
    from twinlink.state import CameraFrame

    m, s = _mapping(), RobotState()
    K = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]

    def frame():
        return CameraFrame(image=np.zeros((2, 2, 3), np.uint8), encoding="rgb8",
                           stamp=0.0, frame_id="f", width=2, height=2)

    s.set_camera("cam", frame())
    m.apply("/r/info", "sensor_msgs/msg/CameraInfo", NS(header=_header(), k=K), s)
    assert s.camera("cam").intrinsics[0, 0] == 600.0
    # New session: entry cleared, then a new frame + the streaming info topic.
    s.clear_camera("cam")
    s.set_camera("cam", frame())
    assert s.camera("cam").intrinsics is None
    m.apply("/r/info", "sensor_msgs/msg/CameraInfo",
            NS(header=_header(), k=[0.0] * 9), s)  # garbage ignored: cached K wins
    assert s.camera("cam").intrinsics[0, 0] == 600.0


def test_recv_stamp_lands_on_frame():
    m, s = _lazy_mapping(), RobotState()
    m.apply("/r/img", _COMPRESSED_TYPE, _jpeg_msg(np.zeros((2, 2, 3), np.uint8)), s,
            recv_stamp=123.456)
    assert s.camera("cam").bridge_recv_stamp == 123.456
