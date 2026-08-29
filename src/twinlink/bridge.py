"""The TwinLink orchestrator.

Wires one *source* (live ROS 2 / MCAP / URDF) through a shared
:class:`~twinlink.state.RobotState` into one or more *sinks* (MuJoCo, Isaac Sim, ...).

Threading model: the source runs in its own thread and only writes state; the sinks are ticked from
:meth:`TwinLink.run`, i.e. the **main thread**.  That matters -- OpenGL contexts and macOS windowing must live on the
main thread.
"""

from __future__ import annotations

import logging
import threading
import time

from .mapping import RobotMapping
from .sinks.base import StateSink
from .sources.base import StateSource
from .state import RobotState

log = logging.getLogger("twinlink.bridge")


class TwinLink:
    def __init__(
        self,
        state: RobotState | None = None,
        source: StateSource | None = None,
        sinks: list[StateSink] | None = None,
        mapping: RobotMapping | None = None,
        rate: float = 60.0,
    ) -> None:
        self.state = state or RobotState()
        self.source = source
        self.sinks: list[StateSink] = list(sinks or [])
        self.mapping = mapping
        self.rate = rate
        self._stop = threading.Event()

    def add_sink(self, sink: StateSink) -> TwinLink:
        self.sinks.append(sink)
        return self

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        if self.source is not None:
            if self.mapping is None and self.source.requires_mapping:
                raise ValueError(f"{type(self.source).__name__} needs a RobotMapping; pass mapping=...")
            self.source.bind(self.state, self.mapping)
        for sink in self.sinks:
            sink.bind(self.state)
            sink.setup()

    def run(self, duration: float | None = None, stop_when_source_done: bool = True, heartbeat: float = 0.0) -> None:
        """Block, ticking sinks at ``rate`` Hz, until done or interrupted.

        ``heartbeat`` > 0 logs the state + incoming message rate every N seconds
        -- useful for a live source, so you can tell data is flowing even when
        the robot is standing still."""
        self.setup()
        if self.source is not None:
            self.source.start()
            log.info("source started: %s", type(self.source).__name__)

        period = 1.0 / self.rate if self.rate > 0 else 0.0
        t0 = time.monotonic()
        source_was_running = self.source is not None and self.source.is_running()
        hb_time, hb_rev = t0, self.state.revision
        try:
            while not self._stop.is_set():
                tick = time.monotonic()

                keep_going = True
                for sink in self.sinks:
                    if sink.update() is False:
                        keep_going = False
                if not keep_going:
                    log.info("a sink requested stop")
                    break

                if heartbeat > 0 and (tick - hb_time) >= heartbeat:
                    rev = self.state.revision
                    rate = (rev - hb_rev) / (tick - hb_time)
                    log.info("heartbeat: %s  (%.0f updates/s)", self.state.summary(), rate)
                    hb_time, hb_rev = tick, rev

                if duration is not None and (time.monotonic() - t0) >= duration:
                    break

                if stop_when_source_done and source_was_running and self.source is not None:
                    if not self.source.is_running():
                        # Render one more frame (already done above), then finish.
                        log.info("source drained; stopping")
                        break

                if period:
                    sleep = period - (time.monotonic() - tick)
                    if sleep > 0:
                        time.sleep(sleep)
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self.close()

    def close(self) -> None:
        if self.source is not None:
            try:
                self.source.stop()
            except Exception as exc:  # pragma: no cover
                log.debug("source stop failed: %s", exc)
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as exc:  # pragma: no cover
                log.debug("sink close failed: %s", exc)
        log.info("twin closed. final state: %s", self.state.summary())
