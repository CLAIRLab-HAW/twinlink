"""Live source over the Foxglove WebSocket protocol (no ROS install needed).

This is the no-ROS sibling of :class:`~twinlink.sources.ros2.Ros2Source`: instead of ``rclpy`` it
talks to a ``foxglove_bridge`` (which the robot / offboard container already
runs) over a WebSocket and decodes the CDR payloads itself -- exactly the same
bytes a ROS 2 publisher puts on the wire, deserialized with the same path as the
MCAP source. So a digital twin can mirror a *live* robot from a laptop that has
no ROS 2.

    foxglove_bridge ──ws (CDR)──▶ FoxgloveSource ──▶ RobotState ──▶ MujocoSink

Transport is the ``websockets`` library (its synchronous client) -- the same one
Foxglove Studio and the official Foxglove Python SDK use, so its handshake
(subprotocol + permessage-deflate) is what ``foxglove_bridge`` expects.

Protocol: https://github.com/foxglove/ws-protocol. We offer both subprotocols
``foxglove.websocket.v1`` (classic) and ``foxglove.sdk.v1`` (new SDK-based
bridges, e.g. foxglove_bridge 3.x) -- they share the wire format. The consumer
subset drives :class:`FoxgloveSource` (advertise → subscribe → message data);
:class:`FoxglovePublisher` adds the *uplink* (client advertise → client message
data), so a no-ROS client can also publish onto a ROS topic through the bridge.
"""
from __future__ import annotations

import json
import logging
import re
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

from .base import StateSource

log = logging.getLogger("twinlink.foxglove")

_OP_MESSAGE_DATA = 0x01  # server binary opcode: [op][subId u32][recvTime u64][payload]
_OP_CLIENT_MESSAGE_DATA = 0x01  # client binary opcode: [op][channelId u32][payload]
# Offer both: the classic ws-protocol name and the newer Foxglove SDK name.
# foxglove_bridge 3.x is built on the new SDK and only accepts "foxglove.sdk.v1"
# (rejecting v1-only clients with a misleading "missing subprotocol" 400); older
# bridges speak "foxglove.websocket.v1". Both share the same message wire format.
_SUBPROTOCOLS = ["foxglove.websocket.v1", "foxglove.sdk.v1"]

# Concatenated ros2msg schema for sensor_msgs/JointState — passed on client
# advertise so the bridge knows the type when relaying our uplink onto a ROS topic.
JOINTSTATE_SCHEMA = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def parse_message_data(data: bytes) -> Optional[Tuple[int, float, bytes]]:
    """Parse a binary server frame; return ``(sub_id, recv_stamp, cdr_payload)``.

    ``recv_stamp`` is the bridge's receive timestamp converted to epoch
    seconds (the u64 nanoseconds at bytes 5..13 of a MessageData frame) --
    it dates the message *at the bridge*, so a consumer can split
    sensor->bridge from bridge->client latency.  Returns ``None`` for
    non-``MessageData`` frames. Pure/testable."""
    if not data or data[0] != _OP_MESSAGE_DATA or len(data) < 13:
        return None
    sub_id = struct.unpack_from("<I", data, 1)[0]
    recv_ns = struct.unpack_from("<Q", data, 5)[0]
    return sub_id, recv_ns * 1e-9, data[13:]


def _parse_concatenated_msg(root_type: str, schema: str) -> dict:
    """Parse a foxglove/ROS concatenated ros2msg schema into rosbags types.

    The schema is the root message's fields followed by ``MSG: pkg/Type`` sections
    for each dependency (separated by ``===`` lines). Returns a name->typedef dict
    ready for ``typestore.register(...)`` (handles moveit_msgs etc. that rosbags
    doesn't ship)."""
    from rosbags.typesys import get_types_from_msg

    sections = re.split(r"(?m)^=+\s*$", schema)
    types: dict = {}
    types.update(get_types_from_msg(sections[0], root_type))
    for sec in sections[1:]:
        m = re.match(r"\s*MSG:\s*(\S+)", sec)
        if not m:
            continue
        name = m.group(1)
        if "/msg/" not in name:  # pkg/Type -> pkg/msg/Type
            parts = name.split("/")
            name = f"{parts[0]}/msg/{parts[-1]}"
        types.update(get_types_from_msg(sec[m.end():], name))
    return types


def select_channels(channels: list, wanted: set) -> List[dict]:
    """Pick advertised CDR channels whose topic we care about. Pure/testable."""
    out = []
    for ch in channels:
        if ch.get("topic") in wanted and (ch.get("encoding") or "cdr") == "cdr":
            out.append(ch)
    return out


def discover_channels(url: str, timeout: float = 5.0) -> List[dict]:
    """Connect to a foxglove bridge and return the advertised channels.

    Each channel is a dict with ``topic``, ``schemaName``, ``encoding``, ``id``.
    Handy at a new robot to see what's on offer before configuring a mapping."""
    from websockets.sync.client import connect

    channels: Dict[int, dict] = {}
    with connect(url, subprotocols=_SUBPROTOCOLS, open_timeout=timeout, max_size=None) as ws:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                msg = ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:
                break
            if isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except ValueError:
                    continue
                if data.get("op") == "advertise":
                    for ch in data.get("channels", []):
                        channels[ch.get("id")] = ch
    return list(channels.values())


class FoxgloveSource(StateSource):
    def __init__(
        self,
        url: str = "ws://localhost:8765",
        *,
        topics: Optional[List[str]] = None,
        reconnect: bool = True,
        connect_timeout: float = 5.0,
        store: str = "LATEST",
    ) -> None:
        super().__init__()
        self.url = url
        self.topics = topics
        self.reconnect = reconnect
        self.connect_timeout = connect_timeout
        self.store = store

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ws = None
        self._typestore = None
        self._sub_map: Dict[int, Tuple[str, str, int]] = {}  # subId -> (topic, msgtype, channelId)
        self._next_sub = 0
        self._decode_errors: set = set()  # topics whose first decode failure was logged
        # Ingest telemetry: message counts + last bridge->client lag per topic,
        # logged every _STAT_PERIOD seconds (DEBUG).  A growing lag means the
        # receive loop is falling behind the wire rate.
        self._stat_counts: Dict[str, int] = {}
        self._stat_lag: Dict[str, float] = {}
        self._stat_t0 = time.monotonic()

    _STAT_PERIOD = 5.0

    def _stat_tick(self, topic: str, recv_stamp: float) -> None:
        self._stat_counts[topic] = self._stat_counts.get(topic, 0) + 1
        if recv_stamp:
            self._stat_lag[topic] = time.time() - recv_stamp
        now = time.monotonic()
        elapsed = now - self._stat_t0
        if elapsed < self._STAT_PERIOD:
            return
        if log.isEnabledFor(logging.DEBUG):
            parts = [
                "%s: %.1f/s lag=%.0fms" % (
                    t, n / elapsed, self._stat_lag.get(t, 0.0) * 1e3,
                )
                for t, n in sorted(self._stat_counts.items())
            ]
            log.debug("ingest %s", " | ".join(parts))
        self._stat_counts.clear()
        self._stat_t0 = now

    # ------------------------------------------------------------------ #
    def start(self) -> "FoxgloveSource":
        assert self.state is not None, "bind() before start()"
        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="foxglove-source", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._running = False

    def _wanted_topics(self) -> set:
        if self.topics is not None:
            return set(self.topics)
        return set(self.mapping.topics()) if self.mapping is not None else set()

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            from websockets.sync.client import connect
            from websockets.exceptions import ConnectionClosed
        except ImportError:
            log.error("FoxgloveSource needs the websockets library: pip install 'twinlink[foxglove]'")
            self._running = False
            return
        try:
            from rosbags.typesys import Stores, get_typestore
        except ImportError:
            log.error("FoxgloveSource needs rosbags: pip install 'twinlink[foxglove]'")
            self._running = False
            return
        self._typestore = get_typestore(getattr(Stores, self.store, Stores.LATEST))

        backoff = 0.5
        while not self._stop.is_set():
            try:
                self._ws = connect(
                    self.url, subprotocols=_SUBPROTOCOLS,
                    open_timeout=self.connect_timeout, max_size=None,
                )
                log.info("connected to foxglove bridge at %s", self.url)
                backoff = 0.5
                self._session(ConnectionClosed)
            except Exception as exc:
                log.warning("foxglove connection failed (%s): %s", self.url, exc)
            finally:
                if self._ws is not None:
                    try:
                        self._ws.close()
                    except Exception:
                        pass
                    self._ws = None
                self._sub_map.clear()
                self._next_sub = 0
                self._decode_errors.clear()
            if not self.reconnect or self._stop.is_set():
                break
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 5.0)
        self._running = False
        log.info("foxglove source stopped")

    def _session(self, ConnectionClosed) -> None:
        wanted = self._wanted_topics()
        while not self._stop.is_set():
            try:
                msg = self._ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except ConnectionClosed:
                return
            if isinstance(msg, str):
                self._on_text(msg, wanted)
            else:
                self._on_binary(bytes(msg))

    def _on_text(self, text: str, wanted: set) -> None:
        try:
            msg = json.loads(text)
        except ValueError:
            return
        if msg.get("op") == "advertise":
            for ch in select_channels(msg.get("channels", []), wanted):
                self._subscribe(ch)
        elif msg.get("op") == "unadvertise":
            ids = set(msg.get("channelIds", []))
            if ids:
                self._sub_map = {s: v for s, v in self._sub_map.items() if v[2] not in ids}

    def _subscribe(self, channel: dict) -> None:
        topic = channel["topic"]
        msgtype = channel.get("schemaName", "")
        self._ensure_type(msgtype, channel.get("schema", ""), channel.get("schemaEncoding", "ros2msg"))
        sub_id = self._next_sub
        self._next_sub += 1
        self._sub_map[sub_id] = (topic, msgtype, channel.get("id"))
        self._ws.send(json.dumps({
            "op": "subscribe",
            "subscriptions": [{"id": sub_id, "channelId": channel["id"]}],
        }))
        log.info("subscribed %s [%s]", topic, msgtype)

    def _ensure_type(self, msgtype: str, schema: str, schema_encoding: str) -> None:
        if not msgtype:
            return
        try:
            if msgtype in self._typestore.types:
                return
        except Exception:
            pass
        if schema and schema_encoding == "ros2msg":
            try:
                self._typestore.register(_parse_concatenated_msg(msgtype, schema))
                log.info("registered type %s (+deps) from schema", msgtype)
            except Exception as exc:
                log.debug("could not register %s: %s", msgtype, exc)

    def _on_binary(self, data: bytes) -> None:
        parsed = parse_message_data(data)
        if parsed is None:
            return
        sub_id, recv_stamp, payload = parsed
        entry = self._sub_map.get(sub_id)
        if entry is None:
            return
        topic, msgtype = entry[0], entry[1]
        self._stat_tick(topic, recv_stamp)
        try:
            msg = self._typestore.deserialize_cdr(payload, msgtype)
            self.mapping.apply(topic, msgtype, msg, self.state, recv_stamp=recv_stamp)
        except Exception as exc:
            # Log the first failure per topic at WARNING (silent DEBUG hides
            # misconfigured compressed-image codecs behind "camera not ready"),
            # then stay quiet so a persistently-bad topic doesn't spam.
            if topic not in self._decode_errors:
                self._decode_errors.add(topic)
                log.warning("decode failed on %s [%s]: %s (once)", topic, msgtype, exc)
            else:
                log.debug("decode failed on %s (%s): %s", topic, msgtype, exc)


class FoxglovePublisher:
    """Publish ROS messages onto a topic *through* a foxglove_bridge (uplink).

    The write-side counterpart of :class:`FoxgloveSource`: it advertises one
    client channel and sends CDR-encoded messages on it, so a client with **no
    ROS install** (e.g. a workstation) can publish onto a real ROS topic via the bridge.
    This is what turns the WebSocket transport bidirectional -- e.g. a digital
    twin sending a planning goal back to MoveIt.

        FoxglovePublisher ──ws (client message data)──▶ foxglove_bridge ──▶ ROS topic

    Requires the bridge's ``clientPublish`` capability (on by default). Build
    messages with :attr:`typestore` (``pub.typestore.types[name](...)``), then
    call :meth:`publish`.
    """

    def __init__(
        self,
        url: str,
        topic: str,
        msgtype: str,
        *,
        schema: str = "",
        channel_id: int = 1,
        store: str = "LATEST",
        connect_timeout: float = 5.0,
    ) -> None:
        self.url = url
        self.topic = topic
        self.msgtype = msgtype  # e.g. "sensor_msgs/msg/JointState"
        self.schema = schema
        self.channel_id = channel_id
        self.store = store
        self.connect_timeout = connect_timeout

        self._ws = None
        self._typestore = None
        self._drain: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def typestore(self):
        return self._typestore

    def start(self) -> "FoxglovePublisher":
        from rosbags.typesys import Stores, get_typestore
        from websockets.sync.client import connect

        self._typestore = get_typestore(getattr(Stores, self.store, Stores.LATEST))
        self._ws = connect(
            self.url, subprotocols=_SUBPROTOCOLS,
            open_timeout=self.connect_timeout, max_size=None,
        )
        self._await_server_info()
        self._advertise()
        # drain incoming frames so the library answers pings and the queue can't grow
        self._stop.clear()
        self._drain = threading.Thread(target=self._drain_loop, name="foxglove-pub-drain", daemon=True)
        self._drain.start()
        return self

    def _await_server_info(self, timeout: float = 5.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                msg = self._ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:
                return
            if isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except ValueError:
                    continue
                if data.get("op") == "serverInfo":
                    caps = data.get("capabilities", [])
                    if "clientPublish" not in caps:
                        log.warning("bridge caps %s lack 'clientPublish' — uplink may be rejected", caps)
                    return

    def _advertise(self) -> None:
        self._ws.send(json.dumps({
            "op": "advertise",
            "channels": [{
                "id": self.channel_id,
                "topic": self.topic,
                "encoding": "cdr",
                "schemaName": self.msgtype,
                "schemaEncoding": "ros2msg",
                "schema": self.schema,
            }],
        }))
        log.info("advertised client channel %s [%s]", self.topic, self.msgtype)

    def _drain_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:
                return

    def publish(self, msg) -> None:
        """Serialize a message (built from :attr:`typestore`) and send it."""
        payload = self._typestore.serialize_cdr(msg, self.msgtype)
        self.publish_raw(payload)

    def publish_raw(self, payload: bytes) -> None:
        frame = bytes([_OP_CLIENT_MESSAGE_DATA]) + struct.pack("<I", self.channel_id) + bytes(payload)
        self._ws.send(frame)

    def close(self) -> None:
        self._stop.set()
        if self._drain is not None:
            self._drain.join(timeout=1.0)
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
