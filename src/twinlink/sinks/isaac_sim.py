"""NVIDIA Isaac Sim digital-twin sink (integration scaffold).

Isaac Sim ships its own embedded Python (``omni.isaac.*`` / ``isaacsim``) that
cannot be ``pip``-installed alongside this package, so this sink is a thin,
*runnable-inside-Isaac* adapter rather than something we can exercise here.

The state contract is identical to :class:`~twinlink.sinks.mujoco_sink.MujocoSink`, so the mapping/source
layers are reused unchanged.  When run inside Isaac Sim's interpreter, ``setup``
binds an ``ArticulationView`` and ``update`` writes the state's joint positions
to the articulation each tick::

    # inside Isaac Sim's python.sh
    from twinlink import TwinLink, RobotMapping, Ros2Source
    from twinlink.sinks.isaac_sim import IsaacSimSink
    sink = IsaacSimSink(articulation_prim="/World/husky_ur5")
    TwinLink(mapping=RobotMapping.from_yaml("a200_0553.yaml"),
             source=Ros2Source(), sinks=[sink]).run()
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from .base import StateSink

log = logging.getLogger("twinlink.isaac")


class IsaacSimSink(StateSink):
    def __init__(
        self,
        articulation_prim: str,
        *,
        joint_remap: Optional[Dict[str, str]] = None,
        usd_path: Optional[str] = None,
        base_prim: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.articulation_prim = articulation_prim
        self.joint_remap = dict(joint_remap or {})
        self.usd_path = usd_path
        self.base_prim = base_prim

        self._view = None
        self._dof_names: List[str] = []
        self._state_for_dof: List[Optional[str]] = []

    def setup(self) -> None:
        try:
            from omni.isaac.core.articulations import ArticulationView
        except Exception as exc:  # pragma: no cover - only meaningful inside Isaac
            raise NotImplementedError(
                "Isaac Sim runtime (omni.isaac.*) is not importable here. Run this "
                "sink inside Isaac Sim's python.sh. The state/mapping/source layers "
                "are shared with MujocoSink, so only this sink needs the Isaac runtime."
            ) from exc

        self._view = ArticulationView(
            prim_paths_expr=self.articulation_prim, name="twinlink_twin"
        )
        self._view.initialize()
        # Map each articulation DoF to the (possibly remapped) state joint name.
        inverse = {v: k for k, v in self.joint_remap.items()}
        self._dof_names = list(self._view.dof_names)
        self._state_for_dof = [inverse.get(n, n) for n in self._dof_names]
        log.info(
            "IsaacSimSink bound to %s with %d DoFs",
            self.articulation_prim,
            len(self._dof_names),
        )

    def update(self) -> bool:
        if self._view is None:
            raise RuntimeError("IsaacSimSink.setup() must run inside Isaac Sim first")
        import numpy as np

        positions = np.array(
            [
                self.state.joint_position(name, default=math.nan)
                for name in self._state_for_dof
            ],
            dtype=np.float32,
        )
        current = self._view.get_joint_positions()
        if current is not None:
            mask = np.isnan(positions)
            positions[mask] = np.asarray(current).reshape(-1)[mask]
        self._view.set_joint_positions(positions)

        if self.base_prim is not None:
            bp = self.state.base_pose()
            if bp is not None:
                self._set_base_pose(bp)
        return True

    def _set_base_pose(self, base_pose) -> None:  # pragma: no cover - Isaac-only
        from pxr import Gf, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.base_prim)
        if not prim:
            return
        xform = UsdGeom.Xformable(prim)
        t = base_pose.translation
        q = base_pose.rotation  # xyzw
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
        xform.AddOrientOp().Set(
            Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
        )
