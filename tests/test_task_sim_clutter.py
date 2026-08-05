"""Regressionsschutz: greifbare Hindernisse bleiben Hindernisse.

Hintergrund (Sim-Split 2026-07-31, Fix-Runde 1): ``arm_config_collides`` parkte
nach dem Split *alle* registrierten Greifbaren aus dem Scratch-Modell weg --
auch die, die zugleich als Hindernis klassifiziert sind (Pool-Slots, gescriptete
Clutter, die ein Task zum Greifziel erklärt).  Damit konnte das Gate genau die
Objekte nie mehr sehen, für die es existiert; ``settle`` wartete umgekehrt auf
Szenen-Clutter, das gar nicht zur Nutzlast gehört.  Keine Suite hat es bemerkt.

Der Test baut ein winziges, roboter- und task-freies MJCF-Modell -- kein
URDF-Bundle nötig, läuft also auch in der CI.
"""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.mjcf_scene import distractor_body_name, distractor_joint_name  # noqa: E402
from twinlink.task_sim import RobotSimSpec, TwinTaskSim  # noqa: E402

#: Der Körperpräfix DIESER Testszene.  Bis 2026-08-01 stand hier eine
#: Inkonsistenz, die nur deshalb nicht auffiel, weil die Klassifikation den
#: Konstruktor-Präfix ignorierte: die Szene benannte ihren Distraktor mit dem
#: Modul-Default (``hrl_distractor_0``), der Konstruktor bekam aber
#: ``scene_prefix=""``.  Seit die Klassifikation dem Konstruktor-Präfix folgt,
#: müssen beide dieselbe Wahl treffen -- und der eigene, app-fremde Präfix ist
#: für einen twinlink-Test ohnehin die ehrlichere: diese Szene MÖBLIERT sich
#: selbst, sie ist nicht präfixlos.
PREFIX = "test_"

#: Ein Schieber-"Arm" mit Greifer-Kindkörper, eine Plattform, ein greifbares
#: Hindernis (Distraktor-Präfix) in +x und eine reine Nutzlast in -x.
SCENE_XML = f"""
<mujoco model="clutter_gate">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="twinlink_ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="platform_base" pos="0 -1 0.1">
      <geom name="platform_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
    <body name="arm_0_shoulder_link" pos="0 0 0.08">
      <joint name="arm_0_slide" type="slide" axis="1 0 0" range="-2 2"/>
      <geom name="arm_0_geom" type="box" size="0.05 0.05 0.05"/>
      <body name="rg6_base" pos="0.12 0 0">
        <geom name="rg6_geom" type="box" size="0.04 0.04 0.04"/>
        <body name="rg6_hand_tcp" pos="0.05 0 0">
          <geom name="rg6_tcp_marker" type="sphere" size="0.005"
                contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
    <body name="{distractor_body_name(0, prefix=PREFIX)}" pos="0.6 0 0.05">
      <freejoint name="{distractor_joint_name(0, prefix=PREFIX)}"/>
      <geom name="{distractor_body_name(0, prefix=PREFIX)}_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="payload" pos="-0.5 0 0.02">
      <freejoint name="payload_free"/>
      <geom name="payload_geom" type="box" size="0.04 0.04 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""

SPEC = RobotSimSpec(
    manipulator_prefixes=("arm_0", "rg6"),
    hand_prefixes=("rg6",),
    gripper_prefixes=("rg6",),
    far_arm_bodies=("arm_0_shoulder_link",),
    gripper_stroke_m=0.156,
    tcp_body="rg6_hand_tcp",
    arm_joints=("arm_0_slide",),
)

#: Schieberstellung, bei der der Armkörper in den greifbaren Distraktor fährt.
INTO_CLUTTER = 0.55
#: Schieberstellung, bei der der Armkörper in die reine Nutzlast fährt.
INTO_PAYLOAD = -0.5


class _ClutterSim(TwinTaskSim):
    """Registriert beide freien Körper als greifbar -- wie ein Clear-Task."""

    def register_graspables(self) -> None:
        self.register_graspable(
            "clutter", distractor_joint_name(0, prefix=PREFIX),
            self._body_id(distractor_body_name(0, prefix=PREFIX)), np.full(3, 0.05),
        )
        self.register_graspable(
            "payload", "payload_free", self._body_id("payload"),
            np.array([0.04, 0.04, 0.02]),
        )


def _build() -> _ClutterSim:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return _ClutterSim(
        model,
        SPEC,
        scene_prefix=PREFIX,
        gripper_follower_factors={},
        gripper_open=0.0,
        gripper_closed=0.6,
        home_pose={"arm_0_slide": 0.0},
    )


def test_clutter_is_classified_as_obstacle_although_it_is_graspable():
    sim = _build()
    try:
        assert sim._graspable.keys() == {"clutter", "payload"}
        assert sim._non_obstacle_graspables == ("payload",)
        clutter_geoms = set(sim._graspable["clutter"]["geoms"])
        assert clutter_geoms <= sim._obstacle_geoms
    finally:
        sim.close()


def test_gate_rejects_a_configuration_reaching_into_graspable_clutter():
    """Der Kern der Regression: greifbar heißt nicht unsichtbar fürs Gate."""
    sim = _build()
    try:
        assert sim.arm_config_collides({"arm_0_slide": INTO_CLUTTER}) is True
        # ... und auch über den reinen Hindernis-Pfad (Pose-Vorprobe).
        assert sim.arm_config_collides(
            {"arm_0_slide": INTO_CLUTTER}, obstacles_only=True
        ) is True
        assert sim.arm_config_collides({"arm_0_slide": 0.0}) is False

        # Gegenprobe = die Regression: würde man ALLE Greifbaren wegparken,
        # verschwände genau dieses Urteil.
        sim._non_obstacle_graspables = tuple(sim._graspable)
        assert sim.arm_config_collides({"arm_0_slide": INTO_CLUTTER}) is False
    finally:
        sim.close()


def test_gate_still_ignores_the_sims_own_payload():
    """Die andere Hälfte der alten Semantik: Nutzlast ist kein Gültigkeitsgrund.

    Die Nutzlast wird weggeparkt UND gehört ohnehin keiner der drei geprüften
    Paarklassen an -- beides zusammen hält den Griff über dem eigenen Objekt
    gültig (der Greifer senkt beim Zugreifen zwangsläufig in es hinein).
    """
    sim = _build()
    try:
        assert sim.arm_config_collides({"arm_0_slide": INTO_PAYLOAD}) is False
        assert "payload" in sim._non_obstacle_graspables
    finally:
        sim.close()


def _settle_ticks(sim, max_ticks: int) -> int:
    start = float(sim.data.time)
    sim.settle(max_ticks=max_ticks)
    return round((float(sim.data.time) - start) / sim.control_dt)


def test_settle_does_not_wait_for_scene_clutter():
    sim = _build()
    try:
        dof = sim._graspable["clutter"]["dof"]
        sim.data.qvel[dof : dof + 3] = 5.0  # das Clutter-Objekt fliegt
        assert _settle_ticks(sim, 40) <= 3
    finally:
        sim.close()

    regression = _build()
    try:
        dof = regression._graspable["clutter"]["dof"]
        regression.data.qvel[dof : dof + 3] = 5.0
        # Die Split-Semantik (alle Greifbaren) lief bis ans Limit.
        regression._non_obstacle_graspables = tuple(regression._graspable)
        assert _settle_ticks(regression, 40) == 40
    finally:
        regression.close()
