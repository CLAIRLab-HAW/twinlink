"""Proximity-capture grasp mechanic: alignment, capture, carry, release.

The parallel-jaw alignment check, kinematic carry and release all live in
``TwinTaskSim._try_grasp`` / ``_carry_grasped`` / ``_release`` -- none of it is
about cubes.  Driving that mechanic indirectly through a real UR5 descend
would test the arm; here the same production methods are exercised directly
against a tiny, robot- and task-free MJCF model.

Gravity is off (``gravity="0 0 0"``): nothing here is about resting/settling
physics (that stays covered task-side), only about the capture/carry/release
state machine.
"""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco extra not installed")

from twinlink.testing import StraightLinkage  # noqa: E402
from twinlink.task_sim import (  # noqa: E402
    GRASP_MAX_MISALIGN_DEG,
    RobotSimSpec,
    TwinTaskSim,
    _wrap_half,
    _wrap_quarter,
)

#: A slider "arm" with a fixed (non-rotating) gripper orientation -- the
#: TCP's world x sits at ``arm_0_slide + 0.17``, y=0, z=0.08.  A payload box
#: sits directly at the TCP's position for ``arm_0_slide = 0.5`` so tests can
#: focus purely on ORIENTATION (misalignment), not on reach/distance.
SCENE_XML = """
<mujoco model="grasp_mechanic">
  <option timestep="0.002" gravity="0 0 0"/>
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
    <body name="payload" pos="0.67 0 0.08">
      <freejoint name="payload_free"/>
      <geom name="payload_geom" type="box" size="0.02 0.015 0.02"/>
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

#: The slide position that puts the TCP exactly on the payload's centre
#: (TCP.x = slide + 0.17; payload.x = 0.67).
ALIGNED_SLIDE = 0.5
#: Far enough that the payload is well outside GRASP_RADIUS (0.07 m).
FAR_SLIDE = -1.0
#: The gripper's pad-opening axis is fixed by the (non-rotating) scene at
#: world yaw = 90 deg (see ``TwinTaskSim.gripper_pad_yaw``: atan2 of the
#: TCP frame's y-axis, which here is always (0, 1, 0)).
GRIPPER_PAD_YAW_DEG = 90.0


class _GraspSim(TwinTaskSim):
    def register_graspables(self) -> None:
        self.register_graspable(
            "payload", "payload_free", self._body_id("payload"),
            np.array([0.02, 0.015, 0.02]),
        )


def _build() -> _GraspSim:
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return _GraspSim(
        model,
        SPEC,
        scene_prefix="",
        default_span=0.04,
        gripper_follower_factors={},
        gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def _set_payload_yaw(sim: _GraspSim, yaw_rad: float) -> None:
    adr = sim._graspable["payload"]["qpos"]
    sim.data.qpos[adr + 3 : adr + 7] = np.array(
        [np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)]
    )
    sim._mujoco.mj_forward(sim.model, sim.data)


def _approach(sim: _GraspSim, slide: float = ALIGNED_SLIDE) -> None:
    sim.set_arm_command({"arm_0_slide": slide})
    sim.step_physics(3)


# --------------------------------------------------------------------- #
# pure helper
# --------------------------------------------------------------------- #
def test_wrap_quarter():
    assert _wrap_quarter(0.0) == pytest.approx(0.0)
    assert _wrap_quarter(np.pi / 2) == pytest.approx(0.0)  # 90deg = symmetric
    assert _wrap_quarter(np.radians(30)) == pytest.approx(np.radians(30))
    assert _wrap_quarter(np.radians(60)) == pytest.approx(np.radians(-30))


# --------------------------------------------------------------------- #
# capture: parallel-jaw alignment gate
# --------------------------------------------------------------------- #
def test_misaligned_payload_is_not_captured():
    """Pads at 90deg (fixed), payload at 90+35deg -> jaws meet edges, no grasp."""
    sim = _build()
    try:
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG + 35))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() is None
    finally:
        sim.close()


def test_aligned_payload_is_captured():
    """Payload yaw exactly matches the pad axis -> captured."""
    sim = _build()
    try:
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() == "payload"
    finally:
        sim.close()


def test_small_misalignment_is_squared_up():
    """Within tolerance the pads square the payload onto the gripper yaw."""
    sim = _build()
    try:
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG + 12))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() == "payload"
        residual = _wrap_quarter(sim.object_yaw("payload") - sim.gripper_pad_yaw())
        assert abs(np.degrees(residual)) < 1.0, "pads should square the payload"
    finally:
        sim.close()


def test_tolerance_boundary_is_rejected():
    sim = _build()
    try:
        beyond = GRASP_MAX_MISALIGN_DEG + 5
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG + beyond))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() is None
    finally:
        sim.close()


def test_gripper_close_without_anything_in_reach():
    sim = _build()
    try:
        sim.set_arm_command({"arm_0_slide": FAR_SLIDE})
        sim.step_physics(3)
        sim.command_gripper(close=True)
        assert sim.grasped_label() is None
    finally:
        sim.close()


# --------------------------------------------------------------------- #
# carry + release
# --------------------------------------------------------------------- #
def test_grasp_carry_release():
    sim = _build()
    try:
        _set_payload_yaw(sim, np.radians(GRIPPER_PAD_YAW_DEG))
        _approach(sim)
        sim.command_gripper(close=True)
        assert sim.grasped_label() == "payload"
        # carried: the payload follows the TCP as the arm moves.
        sim.set_arm_command({"arm_0_slide": ALIGNED_SLIDE + 0.2})
        sim.step_physics(3)
        tcp, _mat = sim.tcp_pose()
        payload_pos = sim.object_position("payload")
        assert float(np.linalg.norm(payload_pos - tcp)) < 0.03
        # released: grasp ends and the object's ordinary contacts return.
        gid = sim._graspable["payload"]["geoms"][0]
        assert sim.model.geom_contype[gid] == 0, "sanity: contacts suspended while carried"
        sim.command_gripper(close=False)
        assert sim.grasped_label() is None
        assert sim.model.geom_contype[gid] != 0, "release must restore contacts"
    finally:
        sim.close()


# --------------------------------------------------------------------- #
# Greiferweite, oeffentlich lesbar
#
# Der Zwilling faehrt die Finger auf die OBJEKTWEITE, wenn ein Objekt gefasst
# ist.  Steckte diese Weite nur im privaten ``_gripper_command``, koennte sie
# niemand an einen echten Greifer spiegeln, ohne in die Interna zu greifen --
# und im Container gemessen: dessen RG6-Gelenke standen waehrend eines ganzen
# Zellenlaufs konstant auf 0.0 (voll offen), waehrend der Zwilling zugriff.
# Zwei verschiedene Haende, gegen die geprueft wurde.
# --------------------------------------------------------------------- #
def test_the_open_gripper_reports_what_the_linkage_can_open_to():
    """Die offene Weite kommt aus dem GETRIEBE, nicht aus ``gripper_stroke_m``.

    Rechnet die Sim die Weite aus ``gripper_stroke_m`` und zwei Ankern
    zurueck, fallen beide Zahlen zwangslaeufig zusammen.  Sie bedeuten aber
    Verschiedenes: ``gripper_stroke_m`` ist das Budget, an
    dem sich eine Griffspanne misst (``GRASP_SPAN_FRACTION``), waehrend die
    Oeffnung der Hand eine Eigenschaft des Getriebes ist -- am echten RG6
    159,0 mm, wo die Sim-Spanne 156 mm fuehrt.  Sie gleichzusetzen ist genau
    die Verwechslung, die den Greiferfehler mitgetragen hat.
    """
    sim = _build()
    sim.command_gripper(False)
    assert sim.gripper_width_m() == pytest.approx(StraightLinkage().max_width_m)
    assert sim.gripper_width_m() != pytest.approx(SPEC.gripper_stroke_m)


def test_a_grasp_drives_the_joint_the_linkage_asks_for_not_a_line_of_its_own():
    """Der eigentliche Fehler, als Test.

    Vorher rechnete ``command_gripper`` die Weite mit einer EIGENEN Geraden in
    einen Gelenkwert um (``closed * (1 - width/stroke)``).  Von aussen fiel das
    nicht auf, weil ``gripper_width_m`` dieselbe Gerade rueckwaerts ging und
    den Fehler zudeckte -- sichtbar wurde er erst am Gelenk, also an dem, was
    tatsaechlich ins Modell geschrieben wird und was move_group prueft.
    """
    linkage = StraightLinkage()
    sim = _build()
    _approach(sim)
    sim.command_gripper(True, grasp=True)
    span = sim.gripper_width_m()
    assert sim.gripper_command_rad == pytest.approx(linkage.angle_from_width(span))
    # Und die alte Gerade haette hier etwas anderes geliefert.
    old_linear = linkage.closed_rad * (1.0 - span / SPEC.gripper_stroke_m)
    assert sim.gripper_command_rad != pytest.approx(old_linear, abs=1e-3)


def test_the_empty_closed_gripper_reports_zero_width():
    sim = _build()
    sim.command_gripper(True)   # nichts zu fassen -> ganz zu
    assert sim.gripper_width_m() == pytest.approx(0.0, abs=1e-9)


def test_a_grasped_object_sets_the_width_to_its_own_span():
    """Die eine Zahl, um derentwillen der Zugang existiert.

    Sie folgt dem OBJEKT, nicht einer Vorgabe: der Payload misst
    0,04 x 0,03 x 0,04 m, die Pads schliessen ueber seine 0,03-m-Kante --
    und NICHT auf ``default_span`` (0,04), das nur gilt, wo kein Objekt
    vermessen wurde.  Genau diese Objektabhaengigkeit ist es, die an
    einem Greifer mit fester Oeffnung verloren geht.
    """
    sim = _build()
    _approach(sim)
    sim.command_gripper(True)
    assert sim.grasped_label() == "payload", "ohne Griff prueft der Test nichts"
    assert sim.gripper_width_m() == pytest.approx(0.03, abs=2e-3)
    assert sim.gripper_width_m() != pytest.approx(0.04, abs=2e-3), (
        "die Weite haengt an der Vorgabe statt am Objekt -- dann misst sie nichts")


# --------------------------------------------------------------------- #
# Ablegen: nur so weit oeffnen wie noetig
# --------------------------------------------------------------------- #
def test_releasing_opens_only_as_far_as_the_object_needed():
    """Zum Loslassen ganz aufzureissen ist eine Wahl, keine Notwendigkeit.

    Am Container gemessen: nach dem Ablegen stand die Hand auf voller Weite
    MITTEN IM TOR, und move_group verweigerte den Rueckzug (``2 contact(s)
    detected : gate_0 - ..._finger_2, gate_1 - ..._finger_1``).  Ein echter
    RG6 oeffnet zum Loslassen nur so weit, wie das Objekt es verlangt.

    Der Payload misst 0,03 m ueber die geschlossene Kante; mit 0,005 m Spiel
    je Seite sind das 0,04 m -- deutlich weniger als die 0,16 m des Getriebes.
    """
    sim = _build()
    _approach(sim)
    sim.command_gripper(True)
    assert sim.grasped_label() == "payload", "ohne Griff prueft der Test nichts"

    sim.command_gripper(False)
    assert sim.grasped_label() is None, "das Objekt muss losgelassen sein"
    assert sim.gripper_width_m() == pytest.approx(0.04, abs=2e-3)
    assert sim.gripper_width_m() < StraightLinkage().max_width_m, (
        "die Hand reisst beim Ablegen weiter auf, als das Objekt verlangt")


def test_the_release_opening_is_clamped_to_what_the_linkage_can_do():
    """Ein Spiel groesser als das Getriebe darf keinen Wert erfinden."""
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    sim = _GraspSim(
        model, SPEC, scene_prefix="", default_span=0.04,
        gripper_follower_factors={}, gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0}, release_clearance=1.0,
    )
    _approach(sim)
    sim.command_gripper(True)
    sim.command_gripper(False)
    assert sim.gripper_width_m() == pytest.approx(StraightLinkage().max_width_m)


def test_a_released_gripper_does_not_report_itself_closed():
    """Die Falle, die das enge Oeffnen aufreisst.

    ``gripper_closed`` verglich den Gelenkwert mit der HALBEN
    Schliessstellung.  Solange die Hand zum Loslassen ganz aufging, war das
    unauffaellig; oeffnet sie nur noch auf Objektbreite plus Spiel, faellt
    sie bei kleinen Objekten unter dieselbe Schwelle und meldete sich als
    geschlossen -- mit dem echten Getriebe fuer jede Spanne unter 40 mm.
    Ob die Hand zu ist, sagt der Befehl, nicht ein geometrischer Schwellwert.
    """
    sim = _build()
    _approach(sim)
    sim.command_gripper(True)
    assert sim.gripper_closed() is True, "die geschlossene Hand muss zu melden"
    sim.command_gripper(False)
    assert sim.gripper_closed() is False


# --------------------------------------------------------------------- #
# Die Spanne kommt aus der Geometrie ZWISCHEN den Backen
#
# Aus ``entry["half"]`` -- den Huellquader-Halbmassen des ganzen Koerpers --
# gelesen ist sie fuer einen Wuerfel dasselbe, fuer alles andere die groebste
# denkbare Abstraktion.  Gemessen: ein Deckel (180 mm Scheibe, 30 mm Knauf)
# meldete auf ALLEN VIER Objektsprossen "kein schliessbares Flaechenpaar" --
# 180 mm gegen 156 mm Backengang, unabhaengig von der Modellfeinheit.
#
# Damit konnte alpha_obj auch ueber den GRIFF nicht binden: der Zwilling las
# den Huellquader, gleichgueltig was zwischen den Backen wirklich stand -- das
# erklaert eine flache Messtabelle als Eigenschaft des AUFBAUS.
# --------------------------------------------------------------------- #
WIDE_SCENE_XML = SCENE_XML.replace(
    '<geom name="payload_geom" type="box" size="0.02 0.015 0.02"/>',
    # Eine breite Scheibe UNTEN, ein schmaler Knauf OBEN -- der Deckel im
    # Kleinen.  Die Backen stehen auf Hoehe des Knaufs.
    '<geom name="payload_disc" type="box" size="0.09 0.09 0.008"'
    ' pos="0 0 -0.042"/>'
    '<geom name="payload_knob" type="box" size="0.015 0.015 0.042"'
    ' pos="0 0 0.008"/>')


class _WideSim(_GraspSim):
    def register_graspables(self) -> None:
        # Die Huelle ist die des ganzen Koerpers -- 180 mm breit.  Genau
        # diese Zahl darf die Fangbedingung NICHT mehr benutzen.
        self.register_graspable(
            "payload", "payload_free", self._body_id("payload"),
            np.array([0.09, 0.09, 0.05]),
        )


def _build_wide() -> _WideSim:
    model = mujoco.MjModel.from_xml_string(WIDE_SCENE_XML)
    return _WideSim(
        model, SPEC, scene_prefix="", default_span=0.04,
        gripper_follower_factors={}, gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def test_a_narrow_feature_is_grasped_even_when_the_whole_body_is_wide():
    """Der Knauf passt in die Backen, der Koerper nicht -- gegriffen wird
    das, was zwischen den Backen steht."""
    sim = _build_wide()
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload", (
        "die Fangbedingung liest weiterhin den Huellquader")


def test_the_captured_span_is_the_local_one_not_the_bounding_box():
    """Die Backenweite beim Loslassen richtet sich nach dem Gegriffenen.

    Mit der Huellquader-Spanne oeffnete der Greifer auf 180 mm -- weiter
    als sein Gang, und weiter als noetig -- genau die Beobachtung, aus der
    die Freigabeweite entstanden ist.
    """
    sim = _build_wide()
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim._grasp_span == pytest.approx(0.030, abs=1e-3), (
        f"Spanne {sim._grasp_span} -- erwartet der Knauf (30 mm)")


def test_a_body_that_is_wide_everywhere_is_still_refused():
    """Die Gegenprobe: ohne schmale Stelle bleibt es beim Nein.

    Ohne sie koennte die Aenderung schlicht jede Pruefung abgeschafft
    haben, und der Test darueber bestuende trotzdem.
    """
    xml = SCENE_XML.replace(
        '<geom name="payload_geom" type="box" size="0.02 0.015 0.02"/>',
        '<geom name="payload_geom" type="box" size="0.09 0.09 0.05"/>')
    model = mujoco.MjModel.from_xml_string(xml)
    sim = _WideSim(model, SPEC, scene_prefix="", default_span=0.04,
                   gripper_follower_factors={},
                   gripper_linkage=StraightLinkage(),
                   home_pose={"arm_0_slide": 0.0})
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() is None


# --------------------------------------------------------------------- #
# Pad-Squaring auch fuer die NEIGUNG
#
# Der Zwilling schweisste das Objekt in der Neigung fest, in der er es fing --
# ein Stift, der im Koecher 6 Grad lehnt, wurde also 6 Grad schief GETRAGEN
# und passte in move_group in keinen Becher mehr (``RRTConnect: Unable to
# sample any valid states for goal tree``).
#
# Flache Backen, die sich um einen schlanken Koerper schliessen, richten ihn
# auf -- dieselbe Mechanik wie beim Gierwinkel, nur um eine andere Achse.  Sie
# gehoert in den Zwilling, weil DER die Wahrheit haelt und move_group sie
# uebernimmt.
# --------------------------------------------------------------------- #
def _set_payload_tilt(sim, tilt_rad: float) -> None:
    """Kippt die Nutzlast um die x-Achse (aus der Senkrechten)."""
    adr = sim._graspable["payload"]["qpos"]
    sim.data.qpos[adr + 3 : adr + 7] = np.array(
        [np.cos(tilt_rad / 2), np.sin(tilt_rad / 2), 0.0, 0.0])
    sim._mujoco.mj_forward(sim.model, sim.data)


def _tilt_of(sim) -> float:
    adr = sim._graspable["payload"]["qpos"]
    w, x, y, z = sim.data.qpos[adr + 3 : adr + 7]
    # Winkel der Koerper-z-Achse gegen die Welt-z-Achse.
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(min(1.0, abs(zz)))))


def test_the_pads_square_a_tilted_object_upright():
    sim = _build()
    _set_payload_tilt(sim, np.radians(8.0))
    assert _tilt_of(sim) == pytest.approx(8.0, abs=0.5)
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    assert _tilt_of(sim) < 1.0, (
        f"nach dem Griff noch {_tilt_of(sim):.1f} Grad schief -- die Backen "
        f"richten den Koerper nicht auf")


def test_squaring_snaps_to_the_NEAREST_axis_not_to_upright():
    """Ein LIEGENDER Koerper bleibt liegen.

    Wer stur auf senkrecht schnappt, stellt einen liegenden Stift beim
    Griff auf -- eine Bewegung, die es nicht gibt, und der Marker der
    Vorstudie liegt in zwei von drei Zellen.
    """
    sim = _build()
    _set_payload_tilt(sim, np.radians(88.0))     # fast waagerecht
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    assert _tilt_of(sim) > 89.0, (
        f"der liegende Koerper wurde auf {_tilt_of(sim):.1f} Grad gedreht")


# --------------------------------------------------------------------- #
# Der Griff wird GEPRUEFT, nicht nur modelliert
#
# ``_try_grasp`` ist ein MODELL: Abstand, Ausrichtung, Spanne -- danach wird
# geschweisst.  Ob die Pads den Koerper wirklich beruehren, ihn durchdringen
# oder danebengreifen, prueft das nicht.  Fuer die Suffizienz-Studie ist genau
# das der Punkt: dort zielt der Roboter nach einem GROBEN Modell und trifft
# auf die WIRKLICHKEIT, und ob der Griff dann noch sitzt, darf nicht durch die
# Grosszuegigkeit der Fangbedingung (7 cm, 20 Grad) verdeckt werden.
#
# Geprueft wird ueber ``mj_geomDistance`` -- eine reine Abstandsabfrage.  Die
# Kontakte des getragenen Koerpers sind aus gutem Grund abgeschaltet; eine
# Abfrage stoert davon nichts.
# --------------------------------------------------------------------- #
def test_a_sound_grasp_has_both_pads_at_the_object():
    sim = _build()
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    spalt = sim.grasp_gap()
    assert spalt is not None, "kein Griff, also kein Spalt"
    assert spalt < 0.01, (
        f"Abstand {spalt*1000:.1f} mm -- die Backen beruehren den Koerper "
        f"nicht wirklich")


def test_without_a_grasp_there_is_no_gap_to_report():
    """``None`` heisst "keine Aussage", nicht "alles in Ordnung"."""
    sim = _build()
    assert sim.grasp_gap() is None


def test_the_check_sees_a_body_the_pads_pass_through():
    """Die eigentliche Zusicherung: Durchdringung faellt auf.

    Ohne sie meldete ein Griff, bei dem die Pads mitten IM Koerper
    stehen, denselben Spalt wie ein sauberer -- und die Studie zaehlte
    ihn als Erfolg.
    """
    sim = _build()
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    # Den Koerper kuenstlich in die Pads schieben.
    adr = sim._graspable["payload"]["qpos"]
    sim.data.qpos[adr + 1] += 0.02
    sim._mujoco.mj_forward(sim.model, sim.data)
    spalt = sim.grasp_gap()
    assert spalt < 0.0, (
        f"Abstand {spalt*1000:+.1f} mm -- eine Durchdringung muss negativ "
        f"sein")


# --------------------------------------------------------------------- #
# Die SCHIEFLAGE beim Fangen wird gemeldet, nicht nur weggeschnappt
#
# Zwei Dinge zusammen ergeben eine Luecke:
#   * move_group KANN nicht ablehnen -- der Backenkontakt ist beim Griff
#     ausdruecklich freigegeben (``_target_touchable``), sonst waere jeder
#     Griff ein Startzustand in Kollision.
#   * Der Zwilling SIEHT die Schieflage (``best_misalign``), schnappt sie weg
#     und wirft die Zahl fort.  Der Griffspalt misst danach und sieht deshalb
#     sauber aus.
#
# Damit war ausgerechnet der Fehler unsichtbar, den ein grobes Modell erzeugt:
# der Roboter zielt nach dem Quader, der echte Koerper steht anders, und die
# Grosszuegigkeit des Fangs buegelt es aus.  Die Zahl muss heraus.
# --------------------------------------------------------------------- #
def test_the_capture_reports_how_far_it_had_to_square_the_object():
    sim = _build()
    _set_payload_yaw(sim, np.radians(12.0))
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    schief = sim.grasp_misalign_deg()
    assert schief is not None
    assert abs(schief) == pytest.approx(12.0, abs=1.5), (
        f"gemeldet {schief} Grad -- der Fang musste 12 Grad ausbuegeln")


def test_a_square_grasp_reports_nearly_zero():
    sim = _build()
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert abs(sim.grasp_misalign_deg()) < 1.0


def test_without_a_grasp_there_is_no_misalignment_to_report():
    sim = _build()
    assert sim.grasp_misalign_deg() is None


# --------------------------------------------------------------------- #
# Die Greifhoehe folgt dem KOERPER, nicht seiner aufrecht gedachten Huelle
#
# Gemessen: marker/pick meldete Erfolg mit einem Spalt von +12 bis +14 mm --
# die Backen standen ueber einen Zentimeter neben dem Stift.  Ursache: der
# Greifpunkt kam aus ``entry["half"][2]``, der halben Hoehe der AABB IM
# KOERPERFRAME.  Der Marker LIEGT, seine wahre Oberkante ist rund 13 mm ueber
# seiner Mitte statt 70 -- das Pruefband lag ueber dem Objekt,
# ``_span_between_pads`` fand dort kein Geom und fiel auf die Huelle zurueck.
# --------------------------------------------------------------------- #
def test_the_grip_reference_follows_the_real_body_not_its_upright_hull():
    """Der Greifpunkt kommt aus der WELT, nicht aus der Koerperframe-AABB.

    Die Messung dazu steht im Abschnittskopf darueber.  Geprueft wird hier der
    Bezugspunkt selbst: der synthetische Aufbau hat keine Backen, die auf eine
    Weite stoppen, und kann den Spalt deshalb nicht zeigen.
    """
    xml = SCENE_XML.replace(
        '<geom name="payload_geom" type="box" size="0.02 0.015 0.02"/>',
        '<geom name="payload_geom" type="box" size="0.02 0.015 0.005"/>')
    model = mujoco.MjModel.from_xml_string(xml)

    class _FlachSim(_GraspSim):
        def register_graspables(self) -> None:
            # Die Huelle behauptet 7 cm halbe Hoehe, der Koerper hat
            # 0,5 cm -- genau die Lage des LIEGENDEN Markers, dessen AABB
            # im Koerperframe seine Laenge als Hoehe fuehrt.
            self.register_graspable(
                "payload", "payload_free", self._body_id("payload"),
                np.array([0.02, 0.015, 0.07]))

    sim = _FlachSim(model, SPEC, scene_prefix="", default_span=0.04,
                    gripper_follower_factors={},
                    gripper_linkage=StraightLinkage(),
                    home_pose={"arm_0_slide": 0.0})
    entry = sim._graspable["payload"]
    ref = sim._grip_reference(entry)
    oben = float(sim.data.xpos[entry["body"]][2]) + 0.005
    # Der Bezugspunkt muss in der Naehe des Koerpers liegen, nicht 50 mm
    # darueber -- so weit wanderte er, als er aus der aufrechten Huelle
    # (halbe Hoehe 70 mm) gerechnet wurde.
    assert abs(float(ref[2]) - oben) < 0.03, (
        f"Greifpunkt bei {float(ref[2]):.3f}, echte Oberkante {oben:.3f} -- "
        f"er folgt der aufrechten Huelle statt dem Koerper")
    # ...und dann findet die Spannenmessung dort auch etwas.
    assert sim._span_between_pads(entry, ref, 0.0) is not None


def test_a_tipped_object_offers_the_axes_it_really_has():
    """Die Greifachsen kommen aus den waagerechten Koerperachsen.

    ``_try_grasp`` bildete sie aus ``obj_yaw`` und ``obj_yaw + 90 Grad``
    -- das setzt voraus, dass die z-Achse des Koerpers SENKRECHT steht.
    Beim liegenden Marker liegt seine Laenge waagerecht, und beide
    Kandidaten zeigen dann an der greifbaren Richtung vorbei.  Fuer einen
    aufrecht stehenden Koerper faellt die Rechnung mit der alten
    zusammen, der Wuerfelturm bewegt sich also nicht.
    """
    sim = _build()
    # aufrecht: x und y sind waagerecht -> genau die alten zwei Achsen
    achsen = sim._horizontal_axes(sim._graspable["payload"])
    assert len(achsen) == 2
    assert min(abs(_wrap_half(a)) for a in achsen) < 1e-6

    # 90 Grad um x gekippt: jetzt sind x und z waagerecht
    _set_payload_tilt(sim, np.radians(90.0))
    gekippt = sim._horizontal_axes(sim._graspable["payload"])
    assert len(gekippt) == 2, f"{len(gekippt)} Achsen bei gekipptem Koerper"


# --------------------------------------------------------------------- #
# Das Ausrichten hat eine GRENZE statt einer Reparatur
#
# Glaube -> Handlung -> Weltaenderung -> wird als "Wahrheit" zurueckgelesen ->
# Glaube.  Kausal legitim (ein echter Greifer richtet einen Stift beim
# Zupacken wirklich auf), aber als KOSTENLOSER Schnapp modelliert verwandelt
# der Zwilling Abstraktionsfehler in nichts: die grobe Sprosse sieht
# ausreichend aus, weil er die Folge ihrer Grobheit selbst repariert hat.
#
# Nachgiebigkeit ist echt, aber begrenzt: flache Pads richten einen leicht
# schiefen Koerper aus, ein stark schiefer rutscht ab.
#
# EHRLICH DAZU: die gemessene Verteilung ueber 68 Container-Laeufe reicht von
# 0,0 bis 2,1 Grad -- die Grenze bindet heute NIRGENDS.  Sie ist eine Wache
# gegen einen Fehlermodus, keine Korrektur eines Ergebnisses.
# --------------------------------------------------------------------- #
def test_a_slightly_tipped_object_is_still_squared_and_grasped():
    from twinlink.task_sim import PAD_SQUARE_LIMIT_DEG

    sim = _build()
    _set_payload_tilt(sim, np.radians(PAD_SQUARE_LIMIT_DEG - 5.0))
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload", (
        "leichte Neigung muss weiter ausgeglichen werden -- "
        "Nachgiebigkeit ist echt")


def test_the_limit_stays_inside_the_geometric_capture_window():
    """Ein Riegel ausserhalb des Fangfensters waere wirkungslos."""
    from twinlink.task_sim import GRASP_MAX_MISALIGN_DEG, PAD_SQUARE_LIMIT_DEG

    assert PAD_SQUARE_LIMIT_DEG < GRASP_MAX_MISALIGN_DEG


def test_the_yaw_path_keeps_its_long_standing_tolerance():
    """Der GIERWINKEL bleibt, wie er war -- und zwar mit Grund.

    Mit 15 Grad probiert: sieben Bestandstests rot,
    darunter die Golden-Spur des Wuerfelturms.  Der Pfad ist alt,
    gepinnt und in der Studie mit hoechstens 2,1 Grad gemessen; ihn
    enger zu ziehen aenderte Ergebnisse aus einem Grund, der mit der
    Frage nichts zu tun hat.  Begrenzt wird der NEUE Pfad, der gar keine
    Grenze hatte.
    """
    from twinlink.task_sim import GRASP_MAX_MISALIGN_DEG

    sim = _build()
    _set_payload_yaw(sim, np.radians(GRASP_MAX_MISALIGN_DEG - 2.0))
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"


def test_a_badly_tipped_object_is_refused_too():
    """Die Neigung ist der weitere Weg -- sie wurde ganz ohne Grenze
    auf die naechste achsparallele Lage geschnappt."""
    from twinlink.task_sim import PAD_SQUARE_LIMIT_DEG

    sim = _build()
    _set_payload_tilt(sim, np.radians(PAD_SQUARE_LIMIT_DEG + 10.0))
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() is None, (
        f"{PAD_SQUARE_LIMIT_DEG + 10:.0f} Grad Neigung wurden weggeschnappt")


# --------------------------------------------------------------------- #
# Der TRANSPORT wird gegen die Wirklichkeit geprueft
#
# Der getragene Koerper hat keine Kontakte (aus gutem Grund, siehe
# ``_suspend_object_contacts``), MoveIt prueft den GEGLAUBTEN Koerper, und der
# ECHTE faehrt ungehindert durch die echte Welt.  Die Richtung ist die
# gefaehrliche: es wird nicht "Wirklichkeit verbietet, was der Glaube erlaubt"
# gemeldet, sondern als ERFOLG gezaehlt.
#
# Geprueft wird wie beim Griff ueber ``mj_geomDistance``.
# --------------------------------------------------------------------- #
def _mit_wand(x: float):
    xml = SCENE_XML.replace(
        '</worldbody>',
        f'<body name="wand" pos="{x} 0 0.08">'
        '<geom name="wand_geom" type="box" size="0.05 0.05 0.05"/>'
        '</body></worldbody>')
    model = mujoco.MjModel.from_xml_string(xml)
    return _GraspSim(model, SPEC, scene_prefix="", default_span=0.04,
                     gripper_follower_factors={},
                     gripper_linkage=StraightLinkage(),
                     home_pose={"arm_0_slide": 0.0})


def test_a_carried_object_clear_of_the_world_reports_a_positive_gap():
    sim = _mit_wand(1.4)                      # weit weg
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    spalt = sim.carried_world_gap()
    assert spalt is not None and spalt > 0.0, (
        f"frei getragen, aber Abstand {spalt} -- die Pruefung sieht die "
        f"Welt nicht")


def test_a_carried_object_driven_into_the_world_reports_no_gap_left():
    """Der eigentliche Fall: der echte Koerper faehrt durch echtes Zeug.

    Ohne diese Zusicherung koennte die Pruefung schlicht immer positiv
    melden, und der Test darueber bestuende trotzdem.
    """
    # Die Wand steht ABSEITS -- zwei ueberlappende Koerper stiessen sich
    # vor dem Griff sonst physikalisch ab, und der Aufbau waere
    # unphysikalisch.  Hineingefahren wird erst der GETRAGENE Koerper,
    # dessen Kontakte aus sind: genau der Fall, den niemand bemerkt.
    sim = _mit_wand(0.80)
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    sim.set_arm_command({"arm_0_slide": 0.63})   # TCP -> 0.80, in die Wand
    sim.step_physics(60)
    assert sim.carried_world_gap() <= 0.0, (
        "der getragene Koerper steckt in der Wand und die Pruefung "
        "meldet freien Raum")


def test_without_a_carry_there_is_nothing_to_report():
    sim = _build()
    assert sim.carried_world_gap() is None


def test_the_worst_moment_of_the_carry_is_remembered():
    """Ein Abstand am Ende sagt nichts ueber die Fahrt dazwischen.

    Der getragene Koerper kann mitten auf dem Weg durch ein Hindernis
    gefahren sein und am Ziel wieder frei stehen -- gemeldet werden muss
    der SCHLECHTESTE Moment, nicht der letzte.
    """
    sim = _mit_wand(0.80)
    _approach(sim)
    sim.command_gripper(close=True)
    sim.step_physics(30)
    assert sim.grasped_label() == "payload"
    sim.set_arm_command({"arm_0_slide": 0.63})     # durch die Wand
    sim.step_physics(60)
    sim.set_arm_command({"arm_0_slide": 0.2})      # wieder heraus
    sim.step_physics(60)
    assert sim.carried_world_gap() > 0.0, "am Ende steht er frei"
    assert sim.carried_world_gap_min() <= 0.0, (
        f"schlechtester Moment {sim.carried_world_gap_min()} -- die Fahrt "
        f"durch die Wand ist vergessen")


def test_without_a_carry_there_is_no_worst_moment():
    sim = _build()
    assert sim.carried_world_gap_min() is None


# --------------------------------------------------------------------- #
# Rotationssymmetrie: ein Roll um die eigene Achse ist KEINE Schieflage
# --------------------------------------------------------------------- #
ZYLINDER_XML = SCENE_XML.replace(
    '<geom name="payload_geom" type="box" size="0.02 0.015 0.02"/>',
    # ``zaxis`` legt die Zylinderachse auf die Welt-y -- der Koerper LIEGT.
    # ``euler`` waere hier eine Falle: MuJoCo liest es in GRAD, "1.5708"
    # haette den Zylinder um 1,6 Grad gekippt statt ihn hinzulegen.
    '<geom name="payload_geom" type="cylinder" size="0.015 0.05"'
    ' zaxis="0 1 0"/>',
)


class _ZylinderSim(TwinTaskSim):
    def register_graspables(self) -> None:
        self.register_graspable(
            "payload", "payload_free", self._body_id("payload"),
            np.array([0.015, 0.05, 0.015]),
        )


def _zylinder() -> _ZylinderSim:
    model = mujoco.MjModel.from_xml_string(ZYLINDER_XML)
    return _ZylinderSim(
        model, SPEC, scene_prefix="", default_span=0.04,
        gripper_follower_factors={}, gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0},
    )


def _rollen(sim, grad: float) -> None:
    """Den LIEGENDEN Zylinder um seine EIGENE Achse drehen (Welt-y)."""
    w = np.radians(grad) / 2.0
    adr = sim._graspable["payload"]["qpos"]
    sim.data.qpos[adr + 3 : adr + 7] = np.array(
        [np.cos(w), 0.0, np.sin(w), 0.0])
    sim._mujoco.mj_forward(sim.model, sim.data)


def test_a_lying_cylinder_rolled_about_its_own_axis_is_still_graspable():
    """Ein Rotationskoerper hat um seine Achse keine Lage.

    Gemessen am 2026-08-17 in der Suffizienz-Vorstudie: ``marker/pick``
    scheiterte auf ALLEN Objektsprossen mit "zu stark geneigt".  Der
    Marker lag korrekt (Zylinderachse waagerecht, Weltanteil 0,012),
    war aber um 39,3 Grad um seine EIGENE Achse gerollt -- eine
    Symmetrie, keine Schieflage.  ``_square_tilt`` las die diskreten
    Koerperachsen, was fuer einen Quader richtig ist und fuer einen
    Zylinder bedeutungslos.
    """
    sim = _zylinder()
    adr = sim._graspable["payload"]["qpos"]
    for grad in (0.0, 39.3, 75.0):
        _rollen(sim, grad)
        assert sim._square_tilt(adr, sim._graspable["payload"]), (
            f"Roll um {grad} Grad um die eigene Achse als Schieflage "
            f"abgelehnt -- der Koerper ist darum symmetrisch")


# --------------------------------------------------------------------- #
# Sichtbare Greiferrampe (fuer Aufnahmen, nicht fuer die Messung)
# --------------------------------------------------------------------- #
def _mit_rampe(takte: int):
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return _GraspSim(
        model, SPEC, scene_prefix="", default_span=0.04,
        gripper_follower_factors={}, gripper_linkage=StraightLinkage(),
        home_pose={"arm_0_slide": 0.0}, gripper_ramp_ticks=takte,
    )


def test_without_a_ramp_the_fingers_are_at_their_target_after_one_tick():
    """Der Bestandsweg -- jede Messung der Studie haengt daran."""
    sim = _mit_rampe(0)
    sim.command_gripper(close=True)
    sim.step_physics(1)
    assert sim.gripper_angle_applied() == pytest.approx(sim._gripper_command)


def test_a_ramp_moves_the_fingers_through_intermediate_angles():
    """Fuer ein Video muss das Zufahren SICHTBAR sein.

    Ohne Rampe schreibt ``step_physics`` die Greifergelenke im ersten
    Substep auf den Zielwert -- im Bild springt die Hand binaer auf und
    zu.  Die Rampe aendert nur den WEG, nicht das Ziel.
    """
    sim = _mit_rampe(10)
    offen = sim.gripper_angle_applied()
    sim.command_gripper(close=True)
    ziel = sim._gripper_command
    zwischen = []
    for _ in range(5):
        sim.step_physics(1)
        zwischen.append(sim.gripper_angle_applied())
    assert all(min(offen, ziel) <= w <= max(offen, ziel) for w in zwischen)
    assert len(set(round(w, 6) for w in zwischen)) > 1, (
        f"Winkel bleibt stehen: {zwischen} -- keine sichtbare Bewegung")
    assert zwischen[-1] != pytest.approx(ziel), "nach der halben Rampe schon da"
    sim.step_physics(6)
    assert sim.gripper_angle_applied() == pytest.approx(ziel), "Ziel nicht erreicht"
