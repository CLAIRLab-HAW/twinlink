# Changelog — twinlink

What changed when. The current state is described in the [README](README.md).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
the versioning [Semantic Versioning](https://semver.org/).

## 2026-08-29 (the grasp model and its suite are English)

- **The remaining German docstrings and comments in `task_sim.py` are English** -- the three
  `GripperLinkage` properties, the tilt-squaring helpers, `_measure_gap`, and the gripper ramp.
- **Two comments quoted a log line that no longer reads that way.** They cited
  `"kein schliessbares Flaechenpaar"`; what `task_sim` actually logs is
  `"%s in reach but no closable face pair"`, and the citations now match it.
- **`test_task_sim_grasp.py` and four smaller suites state their assertion messages in English**, and the
  German local names (`spalt`) went with them.

## 2026-08-29 (the world states its own clock)

- **`sim_time_s()` is the world's clock in seconds**, counted in control steps. The bridge publishes it as
  `/clock`, which makes two properties load-bearing rather than cosmetic: it never goes backwards, and it counts
  steps rather than wall time. A reset therefore starts a new scene and NOT a new clock -- rclpy does not recover
  from a clock that jumps back, and a study runs many episodes against one graph. A world that renders slowly
  simply produces a slow clock, and everything paced by it slows with it.
- **`step_dt` reads the step from the ENVIRONMENT** (`env.control_timestep`), with the injected `control_dt` only
  as the fallback. The two are different quantities that read alike: `control_dt` is what the motion planner paces
  its samples with. Measured 2026-08-29 against `maniskill-eval` -- 0.02 s against the environment's 0.01 s, so
  taking `control_dt` published a clock at exactly twice the world's rate.
- **`_step_once` is the single place the environment is stepped.** The in-process tick, the bridge's external
  command and the settle at reset all go through it; a second `env.step` elsewhere would advance physics without
  advancing the clock.

## 2026-08-26 (the black section stops repeating the workspace rule)

- **`[tool.black]` carries no copy of the workspace rule any more.** The section itself is unchanged --
  `line-length = 120` and the same `force-exclude` -- but the rationale that stood verbatim in every sub-repo
  is gone. Why the section has to exist is written down once: in the workspace `CLAUDE.md`, and in this file's
  2026-08-25 entry *Black formats this repo the same way from anywhere*.
- **`authors` is indented four spaces**, like every other array in the file.
- **`requires-python` is `>=3.11`.** `contract/robot-contract` and `apps/hrl` import `typing.Self` (PEP 673),
  which does not exist before 3.11. Measured on 3.10.19: `from typing import Self` raises
  `ImportError: cannot import name 'Self' from 'typing'` -- at import time, so the module does not load at all, and
  no `from __future__ import annotations` helps. Thirteen packages depend on `robot-contract`, so a `>=3.10` beside
  it was not resolvable on 3.10 anyway; all 19 workspace packages now carry the same floor.

## 2026-08-25 (.idea/ joins the package-specific ignores)

- **`.gitignore` ignores `.idea/`.** The JetBrains project directory appeared in this repo's working tree; nothing
  under that name is tracked in any of the 24 sub-repos, so ignoring it hides no versioned content. It sits with
  `MUJOCO_LOG.TXT` as a package-specific extra on top of the workspace's lean 8-line base.

## 2026-08-25 (Black formats this repo the same way from anywhere)

- **`[tool.black]` now stands in this repo's `pyproject.toml`.** Black takes the first directory
  containing a `.git` as its project root, so a run from inside this repo fell back to Black's own
  88-column default while the workspace runs at 120. The pre-commit hook was unaffected -- it passes the
  root config explicitly -- but an editor or a bare `black` was not.

## 2026-08-25 (output translated to English)

- **The `task_sim` messages about pad bodies and tilt are English now.**

## 2026-08-24 (.gitignore normalised to the workspace base)

- **`.gitignore` now uses the workspace's lean 8-line base** (`__pycache__/`, `*.py[cod]`, `*.egg-info/`, `build/`, `dist/`, `.venv/`, `.pytest_cache/`, `.DS_Store`); replaces the ~280-line auto-generated toptal.com template (Django/Flask/C/C++ patterns this package never produces). Package-specific extra: `MUJOCO_LOG.TXT`.

## 2026-08-24 (pyproject.toml normalised)

- **`pyproject.toml` follows the workspace's canonical section order now** (`[build-system]`, `[project]`, `[project.optional-dependencies]`, `[project.scripts]`, `[project.urls]`, `[dependency-groups]`, `[tool.uv.sources]`, `[tool.setuptools.*]`, `[tool.pytest.*]`); the `[project]` keys follow PEP 621 order (`name`, `version`, `description`, `readme`, `requires-python`, `authors`, `dependencies`). Pure reordering -- every comment and value is unchanged.

## 2026-08-24 (Author metadata unified)

- **`authors` spacing normalised** to `{ name = "Hannes Philip Voss", email =
  "mail@hannesvoss.de" }` (the value was already right; only the `=` spacing
  differed from the other packages). Metadata only, no behaviour change.

## 2026-08-24 (requires-python aligns to the workspace floor)

- **`requires-python` raised from `>=3.8` to `>=3.10`.** The workspace venv is
  Python 3.11, and the workspace floor is `>=3.10` (CLAUDE.md, "Ein neues Paket
  anlegen"); a `>=3.8` promise is never tested and was unlisted drift.
- No behaviour change; 143 tests still green.

## 2026-08-23 (wxyz-Algebra kommt von MuJoCo)

- **Neues Modul `twinlink.quaternion`** mit `quat_mul_wxyz`,
  `quat_conj_wxyz`, `quat_about_z_wxyz`, `mat_to_quat_wxyz`. Es rechnet
  nichts selbst: es ruft MuJoCos `mju_mulQuat`, `mju_negQuat`,
  `mju_axisAngle2Quat`, `mju_mat2Quat` — die Bibliothek, die die Konvention
  definiert, kann per Konstruktion nicht von der Simulation abweichen, gegen
  die gerechnet wird.
- **`TwinTaskSim` traegt die Algebra nicht mehr selbst.** Dieselben vier
  Methoden standen byte-identisch auch in `openvla_stack.env.sim`, das
  Hamilton-Produkt zusaetzlich in `twin_sufficiency.scenes` — drei Handkopien
  fuer eine Rechnung. Gegen die bisherige Fassung ueber 5000 zufaellige Faelle
  gemessen: 2,2e-16, und ueber ±4π kein einziger Vorzeichenwechsel.
- **`tests/test_quaternion.py` nagelt fest, was beim Umstellen kaputtgehen
  kann**: die Argumentreihenfolge (`quat_mul_wxyz(a, b)` = erst `b`, dann `a`)
  und die w-zuerst-Anordnung. Nicht die Arithmetik — die gehoert MuJoCo.
  Gegengeprueft: vertauscht man die Argumente, faellt dieser Test — und
  **kein einziger** der 186 anderen in twinlink und openvla-stack.
- `scipy.spatial.transform.Rotation` geprueft und verworfen: twinlink haengt
  bewusst nur an `numpy`, `pyyaml`, `clearlog`, und scipy waere eine zweite
  Konvention neben der von MuJoCo — genau die Wahlmoeglichkeit, aus der die
  drei Handkopien entstanden sind. Der Grund steht im Modulkopf.

## 2026-08-23 (fmt ist die einzige Fassung)

- **`mjcf_scene.fmt` ist die einzige MJCF-Zahlenformatierung im Workspace.**
  `hrl.env.geometry` und `openvla_stack.env.scene` trugen eigene Kopien; beide
  rufen jetzt diese hier.
- **Die robustere der drei wurde die gemeinsame**: `float(v)` statt `v` vor
  `:.6g` — hrls Kopie hatte die Umwandlung, die beiden anderen nicht. Damit
  nimmt `fmt` auch numpy-Skalare und Zahlen-Strings, die `f"{v:.6g}"` allein
  zurueckweist. Fuer alles, was schon vorher durchging, aendert sich nichts.

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
