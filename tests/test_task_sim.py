"""RobotSimSpec: twinlink bleibt roboter-agnostisch."""

from __future__ import annotations

import numpy as np
import pytest

from twinlink.testing import StraightLinkage  # noqa: E402
from twinlink.task_sim import RobotSimSpec


def test_spec_is_plain_data():
    spec = RobotSimSpec(
        manipulator_prefixes=("a", "b"),
        hand_prefixes=("b",),
        gripper_prefixes=("b",),
        far_arm_bodies=("a_link",),
        gripper_stroke_m=0.1,
        tcp_body="tcp",
        arm_joints=("j1", "j2"),
    )
    assert spec.manipulator_prefixes == ("a", "b")
    assert spec.gripper_stroke_m == 0.1


def test_gripper_prefixes_are_separate_from_hand_prefixes():
    """Die Backen sind eine ANDERE Menge als die Hand-Baugruppe.

    Die Hand darf mitfahrende Sensorik enthalten (Handgelenkskamera); die
    Backen dürfen es nicht, denn nur sie werden für greifbare Objekte
    durchlässig gemacht.  Wären beide dasselbe Feld, verlöre das Kameragehäuse
    seine Kollisionsereignisse (Regression aus dem Sim-Split 2026-07-31).
    """
    fields = RobotSimSpec.__dataclass_fields__
    assert "gripper_prefixes" in fields
    assert "hand_prefixes" in fields


def test_twinlink_stands_alone():
    """Die Eigenständigkeit des Pakets ist Teil des Vertrags.

    twinlink ist ein eigenständiges MIT-Paket mit eigener CI: weder das
    Roboterprofil (``robot_contract``) noch dessen SDK (``husky_sdk``) noch
    die Task-App (``hrl``) dürfen hier importiert werden.  Roboter-Fakten
    kommen ausschließlich als :class:`RobotSimSpec` in den Konstruktor; jedes
    Task-Wissen (Würfel, Turm, RL) gehört ausschließlich hrl, nie umgekehrt.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "twinlink"
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text()
        # ``perception`` gehört mit in die Liste: es ist die dritte Schicht
        # oberhalb von twinlink (Wahrnehmung/Hindernis-Tracking) und damit
        # genauso ein Rückwärts-Import wie hrl.  Fehlte es als einziges in
        # dieser Liste, rutschte es unbemerkt herein.
        for package in ("robot_contract", "husky_sdk", "hrl", "perception"):
            if f"import {package}" in text or f"from {package}" in text:
                offenders.append(f"{path.name}: {package}")
    assert offenders == []


def test_grasp_registry_is_label_keyed():
    """Greifbare Objekte werden über Labels geführt, nicht über Farben.

    Geprüft wird das GANZE Modul, nicht nur der Klassenrumpf: Task-Vokabular
    versteckt sich sonst in Modulkonstanten und Hilfsfunktionen daneben.

    Der App-Präfix ``hrl_`` in dieser Wortliste wäre grün, obwohl ``task_sim``
    gegen genau diesen Präfix klassifiziert:
    das Modul IMPORTIERTE die Konstanten (``OBSTACLE_BODY_PREFIX`` &c.), statt
    das Literal zu schreiben, also fand die Textsuche nichts.  Ein Textscan
    kann diese Eigenschaft grundsätzlich nicht belegen; den Nachweis führt
    jetzt ``test_scene_prefix_drives_classification`` über die WERTE.
    """
    import inspect

    from twinlink import task_sim

    src = inspect.getsource(task_sim)
    # Der Task-Wortschatz darf die Mechanik nicht mehr erreichen.
    for word in ("cube", "CUBE", "color", "tower"):
        assert word not in src, f"Task-Begriff {word!r} in twinlink.task_sim"


#: Sondenszene für den Präfix-Nachweis: Boden, ein Schieber-"Arm" mit
#: Greifer-Kindkörper, ein Pool-Slot (``…obstacle_0``) und ein greifbarer
#: Distraktor (``…distractor_0``) -- alles, was ``_classify_geoms`` und
#: ``_index_obstacle_pool`` am Präfix festmachen.  Roboter- und task-frei, so
#: dass kein URDF-Bundle nötig ist.
_PROBE_SCENE = """
<mujoco model="scene_prefix_probe">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="twinlink_ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
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
    <body name="{p}obstacle_0" pos="1.2 0 0.05">
      <geom name="{p}obstacle_0_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="{p}distractor_0" pos="0.6 0 0.05">
      <freejoint name="{p}distractor_0_free"/>
      <geom name="{p}distractor_0_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

_PROBE_SPEC = RobotSimSpec(
    manipulator_prefixes=("arm_0", "rg6"),
    hand_prefixes=("rg6",),
    gripper_prefixes=("rg6",),
    far_arm_bodies=("arm_0_shoulder_link",),
    gripper_stroke_m=0.156,
    tcp_body="rg6_hand_tcp",
    arm_joints=("arm_0_slide",),
)


def _classification_under_prefix(prefix: str) -> dict:
    """Baue die Sondenszene unter ``prefix`` und lies ihre Klassifikation."""
    import mujoco

    from twinlink.task_sim import TwinTaskSim

    class _ProbeSim(TwinTaskSim):
        def register_graspables(self) -> None:
            self.register_graspable(
                "clutter",
                f"{prefix}distractor_0_free",
                self._body_id(f"{prefix}distractor_0"),
                np.full(3, 0.05),
            )

    model = mujoco.MjModel.from_xml_string(_PROBE_SCENE.format(p=prefix))
    sim = _ProbeSim(
        model,
        _PROBE_SPEC,
        scene_prefix=prefix,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )
    try:
        return {
            "obstacle_geoms": len(sim._obstacle_geoms),
            "pool_slots": len(sim._obstacle_slots),
            "non_obstacle_graspables": sim._non_obstacle_graspables,
        }
    finally:
        sim.close()


def test_scene_prefix_drives_classification():
    """Der Konstruktor-Präfix -- nicht ``hrl_`` -- bestimmt die Klassifikation.

    Die Kernzusage des Umbaus: twinlink ist app-agnostisch und publizierbar.
    Sie gilt nur zur Hälfte, wenn ``__init__`` den ``scene_prefix`` bloß für
    die Render-Trennung nutzt, während ``_classify_geoms`` gegen die
    Modulkonstanten vergleicht und ``_index_obstacle_pool``
    ``obstacle_body_name(i)`` ohne ``prefix=`` ruft.  Identische Szene, nur der
    Präfix getauscht, ergibt dann:

        prefix 'hrl_' :  obstacle_geoms=2  pool_slots=1  non_obstacle_graspables=()
        prefix 'task_':  obstacle_geoms=0  pool_slots=0  non_obstacle_graspables=('clutter',)

    -- ein zweiter Konsument wäre still blind für die gesamte Hindernisklasse
    gewesen (dieselbe Blindheit, die Task 10 schon einmal eine Fix-Runde
    gekostet hat).  Geprüft werden WERTE, nicht Modultext: ein Textscan nach
    ``"hrl_"`` bleibt grün, solange das Modul die Konstanten importiert.
    """
    pytest.importorskip("mujoco", reason="mujoco extra not installed")

    native = _classification_under_prefix("hrl_")
    foreign = _classification_under_prefix("task_")

    # Absolut festgenagelt, damit der Vergleich nicht trivial grün wird, wenn
    # beide Seiten nichts mehr klassifizieren.
    assert native == {
        "obstacle_geoms": 2,  # Pool-Slot + Distraktor
        "pool_slots": 1,
        "non_obstacle_graspables": (),  # der Distraktor BLEIBT Hindernis
    }
    assert foreign == native


def test_hooks_are_abstract_enough_to_subclass():
    from twinlink.task_sim import TwinTaskSim

    assert hasattr(TwinTaskSim, "register_graspables")
    assert hasattr(TwinTaskSim, "support_geom_names")
