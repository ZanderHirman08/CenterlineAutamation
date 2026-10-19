# Handoff: training from the user's daily edits (2026-10-19)

This adds to `docs/HANDOFF.txt`; read that first for the rest of the project
(constraints, data paths, earlier results). Everything here is committed and
pushed on branch **`claude/lucid-hypatia-mx0f0g`** of
`zanderhirman08/centerlineautamation` (commits `9b1571d`, `1fceabb`).
Note: `HANDOFF.txt` names an older branch (`claude/road-centerline-alignment-albtwy`);
this work is on the branch above.

## Why this was built

* The alignment tool could not beat the county road layer (median ~1.6 ft vs
  1.4 ft against validated roads). Likely cause: its classifier learned
  "road" from pixels within 4 ft of the **county lines**, so it learned the
  county layer's errors. Also, the user often traces the county layer
  (Replace Geometry > Trace), so on traced roads the county layer cannot lose.
* The user wanted to collect training data while editing by hand: one button in
  the morning, one in the evening, then train overnight.

## Constraints the user gave (keep to these)

* On `Roads_working` (the GDOT layer in Phase6Working.gdb) the user can only
  edit **geometry** and **Validated** ("Yes"/"No" text, default "No"). **No
  new fields.** So the tools only *read* the road layer and keep everything in
  a separate training folder.
* Editing a road's geometry sets Validated = "Yes" automatically, and it can't
  be set back. The user can also set "Yes" on a correct road without editing it.
* The user finishes each road before the end of the day (no half-edited "Yes"
  roads). Even if one is collected, the next End of Day replaces it.
* Locked-down PC: ArcGIS Pro's Python only (arcpy, numpy, scipy), Python 3.9,
  no GPU deep learning.

## REVAMP Phase 6 manual (user's PDF, version 2026-05-01-01): what matters here

* Standard: "the road centerline should be adjusted to fall within the
  pavement" (Example 1). So an unedited "Yes" means *meets the standard*, not
  *exact centre*. Don't edit when the reference isn't more accurate than GDOT.
* GIO aerial imagery is the primary source; Google/Esri are reference only.
  New leaf-off 2026 imagery is replacing parts of the state.
* Lines are split at intersections. Snap them simply, without extra vertices;
  T-intersections bend along the turn (Examples 2-4). So near junctions a line
  is **not** a pavement centre.
* Never delete features; abandoned roads get `AbandonedRoad = "Yes"`, private
  roads get the Private Road field = "Yes". Missing roads are added
  (`MissingRoad` = "Yes"). Divided roads: both sides validated separately.
  Roundabouts are drawn as circles; new cul-de-sacs as straight lines.
* Do not change line direction (Attribute Transfer is banned for that reason)
  and don't use curve tools. Keep routes connected.

## What was built

New module `centerline_align/arcgis_training.py`, bundled into
`arcgis/CenterlineAlign.pyt` by `scripts/build_pyt.py`. Three new toolbox tools:

1. **"1. Start of Day: Snapshot Roads"** (`snapshot`) writes `originals.json`
   in the training folder: each road OID's first-seen geometry, never
   overwritten, plus `checked_before` for roads already "Yes" at their first
   snapshot (their true original is unknown). Stored in the road layer's SR
   (WKT saved).
2. **"2. End of Day: Collect My Edits"** (`collect`) writes
   `training_roads.json`, rebuilt from every road matching the SQL
   (default `Validated = 'Yes'`). For each road it stores: parts, `changed`
   (moved ≥ 1 ft from its original), `shift_avg/max`, `day` (kept if the
   geometry is unchanged), `original_known`, and `traced`.
   * Roads with `AbandonedRoad`/`PrivateRoad` = 'Yes' are excluded. Field
     names are matched ignoring case, underscores and spaces.
   * Optional *Reference road layer* (the county layer): `traced` = 1 if the
     road is within 0.5 ft of it along ≥ 80% of its length.
   * Split or added roads (new OIDs) are compared with the nearest originals
     and get `original_known` = False.
   * Optional output feature class with ROAD_OID, DAY, CHANGED, SHIFT_AVG,
     SHIFT_MAX, TRACED.
3. **"3. Overnight: Train From My Edits"** (`train`):
   * Holdout: grids with `crc32(grid id) % 5 == 0`, so the same grids every
     night. If none qualify, it falls back to one grid. Roads are assigned to
     grids by their midpoint along the line.
   * Classifier labels: road = within 4 ft of checked lines **with 40 ft
     trimmed off each end** (`trim_ends`). It uses `sample_training_pixels`
     with trusted half_width = None (new: centre-only labels, no verge).
     Not-road = 45-120 ft from any road (checked lines + originals).
   * **Keep/move model** (`_learn_moves` → `engine.fit_move_model`):
     1. It aligns the original lines of checked roads with a known original
        in the training grids.
     2. For each moved line, gain = (original's mean distance to the checked
        line) − (tool's), with 40 ft ends trimmed.
     3. A logistic regression on shift_avg, shift_max, conf, pct_valid and
        width_med predicts "gain ≥ 0.5".
     4. The threshold is chosen to maximise total gain.

     It needs ≥ 20 moved lines with ≥ 5 of each outcome; otherwise it is
     skipped and every move is applied. It is fitted on the same grids the
     classifier trained on (slightly optimistic); the holdout judges it.
   * Scoring on the holdout grids: the original lines are aligned with the new
     model and with the current model (each with its keep/move model). Both
     are compared with the checked lines (only roads with
     `original_known`), alongside any optional compare layers. It prints the
     Evaluate-style table by cover class, plus a table for **non-traced
     roads only**.
   * Promotion: the new model replaces `models/road_classifier.json` if its
     median is ≤ the current model's, measured on points matched by every
     layer and **not near intersections**. Every night is kept as
     `models/road_classifier_<date>.json`; `models/history.csv` gets one row
     per night.

Changes to existing code:

* **Align tool**: new optional parameter *Trained classifier*
  (`classifier_file`); it overrides the work-folder classifier.
  `RoadClassifier.move_model` (new optional field) is applied in
  `engine.refine_pieces(..., move_model=)`. When it rejects a move, the
  original line is kept, MOVED = 0 and the REASON gets
  `move_unlikely_to_help`. New output field **MOVE_P**.
* `classifier.sample_training_pixels`: in `trusted` good lines, half_width
  None means centre-only labels.
* `tests/fake_arcpy.py`: OIDs are now stable under where-clauses (they were
  renumbered before). `SpatialReference` supports `exportToString` and
  `loadFromString`.

## Tests

`tests/test_arcgis.py::test_training_day_cycle` simulates two days and a night
(abandoned skip, traced detection, unchecked removal, holdout scoring,
promotion and history, Align with a trained model, and a forced "never move"
model). `tests/test_engine.py::test_fit_move_model_keeps_unhelpful_moves`
tests the keep/move model. Full suite: 37 passed, 1 skipped. Run
`python scripts/build_pyt.py` after any package change; a test enforces it.
**Nothing has been run on the user's real data or in real ArcGIS Pro yet.**

## User docs

`docs/daily_training.md` is the step-by-step daily routine with Glynn paths,
including a section on how the training follows the manual. `README.md` and
`docs/HANDOFF.txt` were updated too.

## Open items / what to watch

* First real run: confirm the real field names (`AbandonedRoad`, the private
  road field) are matched; the messages report which fields were used.
  Confirm the snapshot runs quickly on the full layer, and that
  `projectAs` works if the roads' SR differs from the imagery's.
* Scores are noisy until a few hundred roads are checked across many grids.
  Watch `history.csv`: the tool's median vs. the original lines and the
  county layer on non-traced roads.
* If a county's imagery is replaced (2026 leaf-off GIO), use a new training
  folder; the model learns one imagery set.
* Possible next steps:
  * a minimum-move dead band;
  * using `changed`/`traced` to weight training labels;
  * adding cover class features to the keep/move model;
  * comparing a model across counties with the Evaluate tool.
