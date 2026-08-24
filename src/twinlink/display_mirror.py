"""Mirror perceived beliefs into the MuJoCo twin (real mode, display only).

The mechanism is robot- and task-agnostic; only *which* world model supplies the states stays with the app layer
(see ``hrl.env.belief_mirror.CubeTwinMirror``).

In ``--real`` runs the sim's object bodies are *not* physics ground truth -- the real objects in the real world are.
Without this mirror the sim bodies simply stay at their random spawn poses and the dashboard twin shows the arm moving
through a fictional scene.  The mirror closes that gap the same way the obstacle layer does for foreign objects: the
twin *shows* what perception believes.

* a localized object is teleported to its believed pose (then obeys physics,
  i.e. settles onto the floor/stack),
* an un-localized object is parked outside every camera frustum instead of
  lying somewhere it is not,
* the grasped object rides under the TCP via an event-free display carry, so
  transports are visible live,
* a placed object rests at its target slot (whatever the caller's belief
  model writes there).

Strictly display: never enabled in sim training (there the object bodies ARE the task physics and rewards read them); an
app only constructs this for its real-camera mode, where every sim-truth check is already skipped.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

log = logging.getLogger("twinlink.display")


class TwinDisplayMirror:
    """Push external beliefs into the twin's display API after every change."""

    def __init__(self, sim) -> None:
        self.sim = sim
        #: Last written state per label ("carry" | "park" | rounded pose) --
        #: unchanged beliefs are not re-teleported (no visual jitter, and a
        #: settled object stays settled).
        self._written: Dict[str, Tuple] = {}

    # ------------------------------------------------------------------ #
    def sync(self, items) -> None:
        """Reconcile every item's twin body with its current believed state.

        ``items`` is a sequence of ``(label, state)`` where ``state`` is ``("carry",)``, ``("park", index)`` or
        ``("pose", position, yaw)``.
        """
        for label, state in items:
            try:
                self._apply(label, state)
            except Exception as exc:  # display must never break the task
                log.warning("twin display mirror failed for %s: %s", label, exc)

    def carry(self, label: str) -> None:
        """Start the display carry the moment the real gripper confirms.

        ``sync`` would only pick the carry up once the caller's belief marks it grasped (after the lift); this shows the
        object at the gripper immediately.
        """
        try:
            self._apply(label, ("carry",))
        except Exception as exc:
            log.warning("twin display mirror failed for %s: %s", label, exc)

    # ------------------------------------------------------------------ #
    def _apply(self, label: str, state: Tuple) -> None:
        key = self._dedup_key(state)
        previous = self._written.get(label)
        if previous == key:
            return
        was_carrying = previous is not None and previous[0] == "carry"
        if state[0] == "carry":
            self.sim.display_carry(label)
        elif state[0] == "pose":
            _, position, yaw = state
            if was_carrying:
                self.sim.display_release(label, position, yaw)
            else:
                self.sim.display_object(label, position, yaw)
        else:  # park
            _, index = state
            if was_carrying:
                self.sim.display_release(label)
            self.sim.park_object(label, index)
        self._written[label] = key

    @staticmethod
    def _dedup_key(state: Tuple) -> Tuple:
        """Round pose states so imperceptible perception noise is not a "change".

        The actual teleport still uses the caller's exact position/yaw (see ``_apply``); only the change-detection key
        is rounded -- a "carry" or "park" marker needs no rounding, it is already exact.
        """
        if state[0] != "pose":
            return state
        _, position, yaw = state
        return ("pose", tuple(round(float(c), 3) for c in position), round(float(yaw), 2))

    def reset(self) -> None:
        self._written = {}
