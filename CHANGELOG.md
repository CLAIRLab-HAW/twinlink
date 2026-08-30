# Changelog — twinlink

What changed when. The current state is described in the [README](README.md).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
the versioning [Semantic Versioning](https://semver.org/).

## 2026-08-30 (one name for the inverse kinematics)

- **`ArmIK.solve` is `ArmIK.solve_ik`** — the name the `Kinematics` protocol beside it already declared, and the
  one `PinocchioKinematics` already used. `ArmMotionPlanner` picks the world's own kinematics when it has one and
  then called `solve`, so the seam was half built: the selection worked and the call did not. Found by running
  hrl's task on a SAPIEN world, which fails on that line and nowhere earlier.

## 2026-08-30 (a body out of convex parts is the lib's, not an app's)

- **`twinlink.mjcf_parts` added** -- `Part`, `add_shape`, `bounding_half_extents`, `free_joint_name` and
  `PRIMITIVES`, moved here from `hrl.env.geometry`. It sits beside `mjcf_scene`, whose `fmt` it already used, and
  needs nothing but numpy: writing a MuJoCo body out of convex geoms at an authored mass is a twin concern, not a
  cube-stacking one. Two packages read it -- `hrl` builds its scene with it, `twin-sufficiency` varies object
  fidelity with it -- and the second was reaching across a package boundary into the first to get it.
- **`part_bounds` is public** (was `_part_bounds`). A caller that merges or coarsens a decomposition needs the same
  reading of `fromto`, `pos` and `quat` that `bounding_half_extents` uses; a second reading of those three fields is
  the drift the module exists to prevent -- in the endpoint form `size` carries the RADIUS alone, so a naive reader
  takes a 0.256 m capsule for an 8 mm one. `twin-sufficiency.coarsen` was importing the private name.
- The module docstring says **four** facts, which is how many it lists.

## 2026-08-30 (drawing is the simulator's, not the world's)

- **`SceneView` is a protocol of its own**, and `TaskWorld` no longer carries `render_rgb`, `camera_matrix` or
  `camera_pose`. What a camera sees, at what resolution and through which renderer, is decided by the engine that
  draws it. The evidence was already in the tree, measured 2026-08-30: `TwinTaskSim.render_rgb` takes a `width` and
  a `height`, because MuJoCo renders on demand, while `ManiSkillTaskSim.render_rgb` takes neither, because SAPIEN
  fixes the sensor resolution once at `gym.make`. One protocol covering both makes one of the two a liar -- and it
  already did: `openvla_stack` calls the sized form, which no ManiSkill world can serve. `SceneView` declares only
  the shape both keep, `render_depth` included, and a sized render stays what it is: one engine's own extension.
- **A world without a renderer is still a world**, and a test says so. `TaskWorld` is now what its name promises --
  run, and say where the robot is.

## 2026-08-30 (ruff resolves the same settings from anywhere)

- **`target-version = "py311"` now stands in `[tool.ruff]`.** Ruff infers it from `project.requires-python`
  when absent -- which the virtual workspace root has no `[project]` table to supply, so a run through the ROOT
  config resolved to 3.10 while a run inside this package resolved to 3.11 (measured 2026-08-30,
  `ruff check --show-settings`). The pre-commit hook passes `--config <workspace-root>`, so 3.10 was the
  version every commit got checked against.
- **CI pins `ruff>=0.16.5,<0.17`** -- the minor the lint scope was measured against, the same bound the
  workspace dev group carries. Unpinned, a ruff release can stabilise new rules and turn this CI red without
  a commit of ours.

## 2026-08-30 (the silent paths speak)

- **25 of the 30 exception handlers that neither logged nor re-raised now say what they swallowed.** The five
  that stay silent are all `except TimeoutError: continue` on a 0.5 s poll -- the idle path, which at 2 Hz would
  be noise rather than information. DEBUG calls across the package go from 21 to 56.
- **`mapping.py` had no logger at all.** A topic that matches no role in the mapping was dropped in `apply()`
  without a trace, which is the mechanism behind "the twin does not move"; a stamp missing `sec`/`nanosec` became
  `0.0` and every lag computed from it was then wrong by 56 years rather than absent; a camera `K` that would not
  parse left the camera without intrinsics forever. All three are gated (`clearlog.once`) because the decoders run
  at the wire rate.
- **`state.chain()` names the frame pair it cannot connect.** Returning `None` rather than an identity matrix is
  the documented and correct behaviour, but it was silent, and the caller then drops a point cloud with no reason
  given. `clearlog.on_change` reports each distinct broken pair once, so a new one speaks and a repeating one does
  not.
- **`pin_kinematics.solve_ik` reports the residual it gave up at.** A few millimetres means the seed was poor;
  half a metre means the target is out of reach. Both arrived at the caller as `None`.
- **`sources/foxglove.py` was the worst single file in the workspace** with 15 mute handlers. A publisher that
  never sees `serverInfo` within its window now warns instead of publishing unverified; the drain thread, the
  session end and the discovery cut-off are dated; dropped non-JSON frames are counted rather than vanishing.
- **The teardown paths in `zenoh_source.py` and `mujoco_sink.py` say what would not close.** A zenoh session that
  holds its port makes the *next* start fail somewhere else entirely.
- `task_sim._try_grasp` reports the commonest outcome of all -- nothing within `GRASP_RADIUS` -- which was the one
  branch of the three without a line.

## 2026-08-30 (pytransform3d checked and rejected as well)

- **`pytransform3d` 3.16.0 measured and rejected**: 2000 matrix-to-quaternion conversions took 22.5 ms against
  3.0 ms in `_matrix_to_quat`, a factor of 7.5. It branches the same way this module does rather than solving the
  K matrix, but `check_matrix` -- the orthonormality test it runs on every call -- accounts for 19.2 of those
  22.5 ms, and `strict_check=False` does not skip it: the flag only turns the raise into a warning. It also
  returns wxyz where `tf_buffer` speaks xyzw, and requires `scipy` and `matplotlib` (14 packages resolved) for a
  package that declares three dependencies.
- **The transforms3d figures in the `tf_buffer` docstring were re-measured in that same run** and now read
  13.5 ms against 3.0 ms, a factor of 4.5. Until 2026-08-30 they read 68.4 ms against 20.6 ms, a factor of 3.3;
  those absolute values did not reproduce on 2026-08-30, neither in the workspace venv (numpy 2.4.6) nor in a
  clean one (numpy 2.5.2) -- both put `_matrix_to_quat` at 3.0-3.2 ms. The ranking is unchanged; the docstring now
  quotes one run, so its ratios are comparable among themselves.

## 2026-08-29 (the package docstring lists every source)

- **All five sources are named**: `Ros2Source` (needs `rclpy`), `FoxgloveSource`, `ZenohSource`, `McapSource` and
  `UrdfStaticSource`, each with what it needs to run. The docstring named three.
- **The quick start says where its inputs live.** The robot mapping configs and the runnable `mujoco_mcap_twin.py`
  belong to the sibling `spact-integration-demos` project, not to this repo.

## 2026-08-29 (the package run can measure coverage again)

- **`addopts = "-p no:cov"` is gone from `[tool.pytest.ini_options]`.** It disabled the pytest-cov plugin
  outright, so `pytest --cov` in this package aborted with `unrecognized arguments: --cov`. The package run is
  where a suite is measured on its own, and where the E2E marks are reachable at all -- the root run deselects
  them.
- The reason the line carried applied to the SYSTEM interpreter, whose pytest-cov did not match its coverage.
  The workspace venv is not that interpreter: it carries pytest-cov 5.0.0 against coverage 7.15.1, verified on
  2026-08-29 by measuring this package.
- Coverage of the workspace is configured once, in `[tool.coverage.*]` of the root `pyproject.toml`, and driven
  by the `coverage-report` skill.
- **`.gitignore` gained `.coverage` and `.coverage.*`** -- a justified package extra, not a mass dump: a package
  run with `--cov` writes them into this directory, and until now they would have shown up as untracked. The
  workspace-wide measurement writes to `.coverage-data/` at the root instead.

## 2026-08-29 (one source for the scene prefix)

### Removed

- **`OBSTACLE_BODY_PREFIX` and `DISTRACTOR_BODY_PREFIX` are gone.** They were kept for consumers that might
  import and compare against them directly; no consumer in the workspace did, while the module's own comment
  forbade twinlink from reading them. `DEFAULT_SCENE_PREFIX` plus the `prefix` argument of the name builders
  is the whole surface, and `test_mjcf_scene.py` pins the names against that one constant.

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
