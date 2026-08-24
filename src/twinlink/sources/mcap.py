"""MCAP replay source (mock mode).

Replays a recorded rosbag2 / MCAP directory into the state, optionally paced to wall-clock so the twin moves exactly as
the robot did.  Decoding is done by ``rosbags`` so **no ROS installation is required** -- this runs on a laptop.

Only the topics the mapping cares about are read, and ``rosbags`` uses the MCAP message index, so a multi-GB recording
is opened in well under a second.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from .base import StateSource

log = logging.getLogger("twinlink.mcap")


class McapSource(StateSource):
    def __init__(
        self,
        path,
        rate: float = 1.0,
        loop: bool = False,
        start_offset: float = 0.0,
        duration: Optional[float] = None,
        realtime: bool = True,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.rate = float(rate)
        self.loop = bool(loop)
        self.start_offset = float(start_offset)
        self.duration = duration
        self.realtime = bool(realtime)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        #: playback position within the recording, seconds since first message
        self.clock = 0.0
        #: 0..1 fraction of the requested window played
        self.progress = 0.0

    def start(self) -> "McapSource":
        assert self.state is not None and self.mapping is not None, "bind() before start()"
        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="mcap-source", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._running = False

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            from rosbags.highlevel import AnyReader
        except ImportError as exc:  # pragma: no cover - import guard
            log.error("rosbags is required for McapSource: pip install rosbags (%s)", exc)
            self._running = False
            return

        wanted = set(self.mapping.topics())
        try:
            while not self._stop.is_set():
                self._play_once(AnyReader, wanted)
                if not self.loop or self._stop.is_set():
                    break
                if self.mapping is not None:
                    self.mapping._origin = None  # re-anchor relative base pose
        finally:
            self._running = False
            log.info("MCAP playback finished at t=%.2fs", self.clock)

    def _play_once(self, AnyReader, wanted) -> None:
        with AnyReader([self.path]) as reader:
            conns = [c for c in reader.connections if c.topic in wanted]
            if not conns:
                log.warning("No requested topics present in %s.\n  wanted: %s", self.path, sorted(wanted))
                return

            window = None
            if self.duration is not None:
                window = self.start_offset + self.duration

            start_ns: Optional[int] = None
            wall0 = time.monotonic()
            for conn, ts, raw in reader.messages(connections=conns):
                if self._stop.is_set():
                    return
                if start_ns is None:
                    start_ns = ts
                    wall0 = time.monotonic()
                elapsed = (ts - start_ns) / 1e9
                if elapsed < self.start_offset:
                    continue
                if window is not None and elapsed > window:
                    break

                if self.realtime and self.rate > 0:
                    target = (elapsed - self.start_offset) / self.rate
                    sleep = target - (time.monotonic() - wall0)
                    if sleep > 0 and self._stop.wait(sleep):
                        return

                self.clock = elapsed
                if window is not None and window > self.start_offset:
                    self.progress = min(1.0, (elapsed - self.start_offset) / (window - self.start_offset))

                try:
                    msg = reader.deserialize(raw, conn.msgtype)
                    self.mapping.apply(conn.topic, conn.msgtype, msg, self.state)
                except Exception as exc:  # keep replaying despite a bad frame
                    log.debug("decode failed on %s (%s): %s", conn.topic, conn.msgtype, exc)
