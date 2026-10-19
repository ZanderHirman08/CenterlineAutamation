# REVAMP Phase 6 Manual: how the tools follow every rule

Checked against manual version 2026-05-01-01. Each rule is in one of these groups:

* **Built in:** the tools do it themselves.
* **Can't be broken:** the tools never make that kind of edit. No tool writes to
  `Roads_working`; they only read it, and everything they save goes to your own
  folders.
* **Not part of the tools:** setup, packaging and submission steps you do in the
  REVAMP template Tasks.
* **Limit:** the rule depends on something the data doesn't record. This says how
  it is covered.

The future editing tools (Suggest / Apply, not built yet) are listed in the last
column. They must meet the same rules before they are built.

## Data sources (2.2.5)

| Rule | Today | Future Suggest / Apply |
|---|---|---|
| GIO imagery is the primary source for validation and digitizing; Google/Esri are reference only | **Built in:** the model is trained and scored on the GIO tiles you point it to. Use the GIO tiles folder, never Google/Esri basemaps. | Same imagery |
| Don't edit GDOT data if the local data is not more accurate per the GIO imagery | **Built in:** the "score to beat" measures the local layer against your validated roads. Roads traced from it are not scored. | Low-confidence stretches keep the original line and are flagged, never guessed |

## Edit & validate (3.1.2)

| Rule | Today | Future |
|---|---|---|
| Roads needing no change: set Validated = Yes | **Built in:** those roads train the model like edited ones | Applies only to roads you accept |
| Geometry edits update Validated and last-edited automatically (attribute rules) | **Can't be broken** | Apply writes the shape only, so the template's attribute rules do the rest, exactly as with Replace Geometry |
| Missing roads: MissingRoad and last-edited set automatically | **Built in:** included, since they also become Validated = Yes | Never adds roads |
| Do NOT delete features; abandoned = Abandoned Road "Yes" | **Can't be broken** | Never deletes |
| Feedback Local field | **Can't be broken** (not read or written) | Not touched |

## Validation standards (3.2)

| Rule | Today | Future |
|---|---|---|
| Abandonment scenarios 1–3 (delete overshoot, delete same-route segment, or flag a whole route) | **Built in:** roads with AbandonedRoad = Yes or an RC_Comments note about abandonment are left out of training and Evaluate. **Can't be broken:** the deletions themselves are yours. | Never deletes or trims. Flags "road seems to end before the line" for your decision. |
| Do NOT use the curve tool | **Can't be broken** | Writes plain vertices only |
| Roundabouts: circles following the pavement; new ones counterclockwise; keep the TEMP route ID | **Built in:** kept for training (they follow the pavement), recognized by ROUTE_ID containing TEMP | Skipped and flagged; route IDs never touched |
| New cul-de-sacs as straight lines; existing circles left as is | **Built in:** circles (loops under 400 ft, not TEMP) are left out, since a circle round the bulb isn't a centreline. For straight ones, the last 40 ft (the bulb) is left out. | Circles skipped; straight ones keep their dead-end point |
| Private roads not validated; Private Road = Yes; gated/private noted in RC_Comments (FAQ) | **Built in:** PrivateRoad = Yes, or RC_Comments mentioning private/gated, are left out | Skipped |
| Roads with medians: both sides validated | **Built in:** each side trains on its own carriageway | Each side follows its own ridge; flagged if both land on one |
| Missing roads: route ID/mile points blank; direction from the major road or S→N, W→E; don't split the existing road at the new junction | **Can't be broken** | Never adds roads or splits |
| Segment in the wrong direction: flip it to match its route | **Can't be broken:** flipping is yours | Never reverses direction; flags a segment that runs against the rest of its route so you can flip it |
| Don't edit Route ID, Mile Points, Function Type, County, System Code, Ownership | **Can't be broken** | Writes the shape only |
| Maintain route connectivity at intersections (except roundabouts, divided/undivided switches, abandoned sections, county borders) | **Can't be broken** | Shared points stay shared; they move only if every connecting segment is applied together |
| GDOT edge-matches county boundaries | **Not part of the tools** | County-border ends are never moved |

## Recommended tools (3.3)

| Rule | Today | Future |
|---|---|---|
| Do NOT use Attribute Transfer (it changes direction) | **Can't be broken** | Not used; direction is checked |
| Replace Geometry / Edit Vertices / Extend-Trim / Reshape | Your edits | Apply is equivalent to Replace Geometry |

## Validation examples (3.4)

| Rule | Today | Future |
|---|---|---|
| Ex 1: centreline within the pavement | **Built in:** this is what the model learns and is scored on | The goal of every suggestion |
| Ex 2: snap at intersections simply, without extra vertices | **Built in:** 40 ft at line ends is left out of training, because lines bend into the snap | Ends stay snapped; straight runs are thinned to few vertices |
| Ex 3: keep route connectivity | **Can't be broken** | As in 3.2 |
| Ex 4: T-intersections follow the roadway | **Built in:** ends masked as in Ex 2 | The line follows the ridge into the turn |
| Ex 5: small islands are not medians (one line through) | **Limit:** islands aren't recorded in any field, so the tools can't tell an island from a median. Such a line may sit on the island; the suspect-roads list in the notebook shows any big disagreement for you to check. | Flagged where the model sees pavement on both sides of the line but not under it |

## QAQC and post-validation (4)

| Rule | Today | Future |
|---|---|---|
| Null shapes / segments under 1 ft | **Built in:** null shapes are skipped; very short segments are too short to score | Skipped and flagged |
| Multipart features, dissolve, m-values, packaging, submission, ITOS feedback, local feedback | **Not part of the tools:** done in the template Tasks after validation | Not touched; Apply keeps single-part lines single-part |

## FAQ

| Rule | Today | Future |
|---|---|---|
| Under construction: capture if possible, note in RC_Comments | **Built in:** RC_Comments mentioning construction are left out, since the imagery may not match | Skipped |
| Gated communities are private; note in RC_Comments | **Built in:** see Private roads above | Skipped |
