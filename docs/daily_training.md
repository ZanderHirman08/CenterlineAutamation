# Training the tool from your daily edits

While you edit the road layer by hand, the toolbox can save your edits as
training data and train the tool overnight. It never writes to the road layer;
it only reads it. Everything it saves goes into a **training folder** of your
own.

```
morning        1. Start of Day: Snapshot Roads        (about a minute)
               ... edit as usual; set Validated = 'Yes' on every road you check ...
end of day     2. End of Day: Collect My Edits        (about a minute)
before you go  3. Overnight: Train From My Edits      (runs while you're away)
```

## How this follows the REVAMP Phase 6 manual

* **Only real, public pavement is learned from.** Roads with `AbandonedRoad` or
  `PrivateRoad` = 'Yes' are left out automatically, even when Validated = 'Yes'
  (an abandoned road may have no pavement at all).
* **Intersections are left out of the road labels.** Lines are split at
  intersections and the manual snaps them simply and bends T-intersections along
  the turn (Examples 2-4), so the last 40 ft of each line is not a pavement-centre
  example. Models are also compared away from intersections.
* **Roads checked without an edit mean "meets the standard"** (the line falls
  within the pavement, Example 1), not "on the exact centre". They teach the
  **keep/move model** when a line is good enough to leave alone, in line with the
  manual's "if the local data is not more accurate, do not edit".
* **Traced roads are tracked.** If you trace the county layer (Replace Geometry >
  Trace), give that layer to *End of Day*. The nightly test then also reports on
  the roads you drew from the imagery, which is the fair comparison with the
  county layer.
* **Use the GIO imagery** (the manual's primary source) for training and for
  *Align*. When new imagery replaces a county's, start a new training folder.
* New roads you add (MissingRoad) are used for training; they have no original,
  so they aren't used for scoring. Direction and route connectivity are never
  touched by the training tools, which only read the layer.

## One-time setup

1. Get the latest `CenterlineAlign.pyt` (see the README) and add it to your project.
   It now has the three tools above as well as *Align Centerlines* and *Evaluate*.
2. Make an empty folder for the training data, on a local drive, e.g.
   `C:\Users\zhirman\Desktop\Centerline\Glynn\training`. Use **the same folder
   every day** for that county. It holds:
   * `originals.json`: each road's shape the first time a snapshot saw it
   * `training_roads.json`: every checked road, how far your edit moved it, and when
   * `models\`: the trained models (`road_classifier.json` is the current one),
     plus `history.csv` with one score line per night
3. Run *Start of Day* **before your first edit**. Roads that were already
   checked before that first snapshot are still used for training, but their
   original shape is unknown, so they aren't used to score the tool.

## Every day

### 1. Start of Day: Snapshot Roads

| Parameter | Value |
|---|---|
| Road layer you edit | `Roads_working` |
| Checked roads are where (SQL) | `Validated = 'Yes'` |
| Training folder | your training folder |

It saves the shape of every road it hasn't seen before. A road's first-seen
shape is never replaced, so taking the snapshot every morning is safe. If you
forget one morning, *End of Day* warns you: roads first seen that day count as
unchanged.

If the layer has a definition query or a selection, only those roads are
read. Leaving it unfiltered is simplest.

### 2. Edit as usual

Set `Validated = 'Yes'` on **every road you check, including roads you didn't
need to move**. Roads that were already right teach the tool when to leave a
line alone, which is where it does worst today.

### 3. End of Day: Collect My Edits

**Save your edits first** (Edit tab > Save). The tool reads the saved data.

Same three parameters as Start of Day, plus:

* *Reference road layer you trace from* (optional): `Local_Roads_Glynn`, if you
  use Trace. Roads that follow it within 0.5 ft along 80% of their length are
  marked `TRACED`.
* *Training roads to look at* (optional, e.g. `...\Results.gdb\Training_Roads`):
  a layer showing each checked road with:

* `CHANGED`: 1 if your edit moved it at least 1 ft (Advanced), 0 if it was already right
* `SHIFT_AVG` / `SHIFT_MAX`: how far it was moved from its original, on average and at most
* `DAY`: when it was last checked or edited
* `TRACED`: 1 if it follows the reference layer

The messages summarise the day, for example *"Today: 23 newly checked and
moved, 9 newly checked and left as they were, 2 edited again."* Roads you set
back to `No` (or delete) are removed from the training data.

### 4. Overnight: Train From My Edits

Start it before you leave and leave Pro open.

| Parameter | Value |
|---|---|
| Imagery | `C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\tiles` |
| Training folder | your training folder |
| Work grids / Grid ID field | `Glynn_Grid_1mi` / `OBJECTID` |
| Also compare these layers (optional) | `Local_Roads_Glynn` |
| Bands | 1, 2, 3, 4 (as for Align) |

What it does:

1. **Sets about 1 grid in 5 aside for testing.** The choice depends only on the
   grid ID, so the same grids are used every night and the scores can be
   compared from night to night. Roads in those grids are never trained on.
2. **Trains the classifier on your checked roads.** Pixels on your lines (minus
   the 40 ft at each end) are labelled road. The classifier no longer learns
   from the county layer's positions, which are themselves a few feet off.
3. **Learns the keep/move model.** It aligns the original lines of your checked
   roads in the training grids and learns, from each move's confidence, size,
   share of pavement found and width, which moves bring a line closer to where
   you put it. Moves it judges won't help keep the original line. It needs at
   least 20 moved lines, with some that helped and some that didn't; until then
   every move is applied and the messages say so.
4. **Tests it.** In the test grids, the tool aligns the *original* lines with
   tonight's model, and with the current model if there is one, and measures
   each against your edits. The same table as *Evaluate* is printed:

   ```
   ALL (812 points)
     layer                 points  mean  median  worst10%  <=2ft  <=5ft
     Original lines           812  ...
     Tool, new model          812  ...
     Tool, current model      812  ...
     Local_Roads_Glynn        812  ...
   ```

   If some roads were traced, a second table shows only the roads you drew from
   the imagery.
5. **Keeps the better model.** If tonight's model is at least as good as the
   current one (median distance to your edits, away from intersections), it becomes
   `models\road_classifier.json`. Every night's model is also kept as
   `models\road_classifier_<date>.json`, and a line is added to `history.csv`.

The scores are only as good as the test set. With a handful of checked roads
the numbers jump around; they settle once a few hundred roads are checked
across many grids.

## Using the trained model

In **Align Centerlines to Imagery**, set *Trained model > Trained classifier*
to `models\road_classifier.json`. The tool then uses that model instead of
training its own from the road layer, and applies the keep/move model. Lines
where it keeps the original get `MOVED = 0` and `REASON` includes
`move_unlikely_to_help`; `MOVE_P` is its estimate (0-1) that the move would
have helped. Use a new work folder, as usual.

## Things to know

* **One training folder per county.** The model learns that county's imagery.
  To try a model on the next county, point *Align* at it and check the result
  with *Evaluate*.
* **Split or merged roads are fine.** A new piece without an original is
  compared with the nearest original lines.
* **Nothing is sent anywhere.** Everything stays on your PC, in the training folder.
