# Changelog — twinlink

Was sich wann geändert hat. Der aktuelle Stand steht in der [README](README.md).

## 2026-08-19 (Doku-Abgleich)

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
- `sources/zenoh_source.py` zeigte im Docstring-Beispiel einen Publisher auf
  `/twin/plan_goal` — ein Kanal, der seit Protokoll v2 stillgelegt ist
  (`RETIRED_CHANNEL_IDS`).

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
