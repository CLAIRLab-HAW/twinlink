# Changelog — twinlink

Was sich wann geändert hat. Der aktuelle Stand steht in der [README](README.md).

Das Format folgt [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
die Versionierung [Semantic Versioning](https://semver.org/lang/de/).

## 2026-08-23 (Bezeichner auf Englisch)

- **Die Bezeichner dieses Pakets sind englisch**, die Prosa bleibt deutsch —
  dieselbe Konvention wie in `sdk/skill-tree` und wie CLAUDE.md sie vorgibt
  ("Doku ist deutsch"). Umbenannt wurden Funktionen, Klassen, Konstanten,
  Parameter und lokale Variablen; Docstrings und Kommentare NICHT.
- **Was ein Programm AUSGIBT, bleibt deutsch**: Abschnittsmarken, JSON-Feld-
  namen und Log-Meldungen sind der Bericht an den Menschen, nicht Code.
- Umbenannt wurde mit einem `tokenize`-Werkzeug (nur NAME-Token), nicht per
  Regex — deshalb ist kein Kommentar und kein String mitgewandert. Drei
  Stellen, die `tokenize` NICHT sieht, wurden eigens nachgezogen:
  f-String-Interpolationen (unter Python 3.11 ist ein f-String EIN Token),
  die Parameternamen in `pytest.mark.parametrize` und Bezeichner, die
  quelltextlesende Tests als String erwarten.
- Gegengemessen: `uv run pytest` steht unveraendert bei 2465 passed,
  3 skipped — derselbe Stand wie vor der Umbenennung.

## [Unreleased]

- **`_matrix_to_quat` gegen die Drift abgesichert.** Die Funktion steht Zeile
  für Zeile ein zweites Mal im Workspace
  (`robot_contract.twin_protocol.mat_to_quat_xyzw`). Das ist der Preis der
  Schichtentscheidung, dass twinlink nicht an `robot_contract` hängt — die
  Entscheidung bleibt, die stille Drift nicht: `tests/
  test_quat_parity_with_robot_contract.py` vergleicht beide Fassungen an den
  Zweigen, die die Verzweigung über die größte Diagonale wirklich treffen
  (180° um x/y/z, Top-Down-Greifpose, kleinwinklig). Der Test überspringt sich
  sauber, wenn `robot_contract` fehlt — in twinlinks eigener CI der Normalfall
  und gerade der Punkt der Schichtentscheidung.

## [0.2.0] - 2026-08-19 (Doku-Abgleich)

- **Die README kannte `TwinTaskSim` nicht.** Die Paketübersicht listete elf
  Module und ließ `task_sim.py`, `display_mirror.py` und `testing.py` aus —
  ausgerechnet das Modul, um das die Arbeit der letzten Woche kreist.
  Nachgetragen, dazu zwei Design-Notes: dass der Griff kinematisch ist und **an
  den Pads** beurteilt wird, und dass die Weite↔Gelenk-Abbildung aus dem
  Roboterprofil kommt statt aus einer lokalen Interpolation.
- **Historische Bezüge aus den Modul-Docstrings entfernt.** Fünf Dateien
  beantworteten in Zeile 3 „woher kam das" statt „was ist das":
  `display_mirror.py`, `events.py`, `kinematics.py`, `mjcf_scene.py`,
  `task_sim.py` („Aus `hrl.env.*` extrahiert", „task-refactor 2026-07-23") und
  `tf_buffer.py` („Phase-6 refactor"). Ebenso die „Bis 2026-08-01/-08-16/-08-19
  stand hier …"-Absätze in `task_sim.py` und `mjcf_scene.py`.

  Die Begründungen bleiben, im Konjunktiv statt in der Vergangenheit: aus „bis
  2026-08-19 stand hier 12,9 mm" wird „ein so gewonnener Wert ist ein Artefakt
  der Geom-Mischung, keine Padbreite". Gemessene Zahlen behalten ihr Datum.
- Zwei weitere Stellen nachgezogen: `sinks/mujoco_sink.py` („literals this
  sink used to hard-code … before RobotSimSpec existed") und `state.py` („the
  shortcut used to run before any look-up").
- `sources/zenoh_source.py` zeigte im Docstring-Beispiel einen Publisher auf
  `/twin/plan_goal` — ein Kanal, der seit Protokoll v2 stillgelegt ist
  (`RETIRED_CHANNEL_IDS`).

---

**Vor der Einführung von SemVer (2026-08-19)** wurde nach Datum
geführt. Die Abschnitte darunter behalten ihre Datumsüberschrift — ihnen
nachträglich Versionsnummern zu geben, würde eine Release-Historie
erfinden, die es nicht gab.
- **SemVer eingeführt.** Version auf `0.2.0`, dieses Changelog folgt
  [Keep a Changelog](https://keepachangelog.com/de/1.1.0/), Tag `v0.2.0`.
  Ältere Abschnitte behalten ihre Datumsüberschrift — ihnen nachträglich
  Versionsnummern zu geben, würde eine Release-Historie erfinden.
- **README nach dem Workspace-Schema** (readme.so): Features · Tech Stack ·
  Installation · Usage · Running Tests · Related · Versioning · License. Die
  vorhandene Prosa ist erhalten und unter den passenden Abschnitt gewandert.
## 2026-08-13 bis 2026-08-19 (Greifermodell)

Nachgetragen, weil es zwischen dem 2026-08-15 und dem 2026-08-19 in rund
fünfzehn Commits entstand und in keiner README stand. Der Zwilling beurteilt
einen Griff seither an dem, was zwischen den Pads liegt, statt an der Box des
ganzen Körpers:

- Padbreite an den Greifflächen gemessen statt als Median über die ganze Hand;
  Padhöhe dort, wo die Backen schließen, nicht wo sie offen stehen.
- Der Greifspalt wird bei **geschlossenen** Fingern gemessen, nicht davor oder
  danach; die Hand schließt über mehrere Ticks, damit die Bewegung sichtbar ist.
- Die schließenden Pads richten die Kippung eines Objekts auf, nicht nur seine
  Gierung; wie weit dabei gedreht werden musste, steht im Ergebnis statt
  verborgen zu bleiben. Ein rundes Objekt, das nur um seine eigene Achse
  gerollt ist, wird nicht mehr abgelehnt.
- Das getragene Objekt wird gegen die reale Welt geprüft — vorher prüfte das
  nichts. Der schlechteste Moment einer Fahrt und der knappste Abstand werden
  mitgeführt.
- Die Weite↔Gelenk-Abbildung kommt aus dem Roboterprofil; die Sim meldet die
  kommandierte Fingeröffnung in Metern, damit ein echter Greifer halten kann,
  was der Zwilling hält. Ein aktuierter Greifer drückt jetzt tatsächlich auf
  das Objekt, das er umschließt, und öffnet nur so weit, wie das freigegebene
  Objekt es braucht.
- Kontaktnamen sind auf das eingeschränkt, was das Greifermodell hergibt.
- Transformationen lösen über mehrere Frames auf statt nur über direkte Kanten;
  eine Kette im selben Frame wird für einen Frame abgelehnt, den der
  Transformationsgraph nie gesehen hat. Entartete Quaternionen werfen einen
  Fehler, statt still zu passieren.
- `fk_body_pose` beschreibt seinen Frame korrekt: der MuJoCo-Ursprung ist
  bodenreferenziert.
