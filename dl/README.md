# Deep-learning centerline model (Google Colab)

The goal is sub-foot accuracy. A U-Net with an ImageNet-pretrained encoder learns
to draw a thin ridge on the pavement centre from your hand-checked roads. Its
crest is the centreline. Training runs on a free Colab GPU. Nothing is
installed on the work PC.

**Before uploading anything:** get your supervisor's or IT's OK to put GIO imagery
and road lines on Google Drive / Colab.

## Daily (in ArcGIS Pro, toolbox `CenterlineAlign.pyt`)

Use the routine in `docs/daily_training.md`:

1. **1. Start of Day: Snapshot Roads**
2. Edit `Roads_working` following the REVAMP manual.
3. **2. End of Day: Collect My Edits** (set *Reference road layer* = `Local_Roads_Glynn`
   so traced roads are recognised).

## When you want a new model (e.g. weekly)

### 1. Export Training Chips (ArcGIS Pro)

| Parameter | Value |
|---|---|
| Checked roads (truth) | `Roads_working` |
| Only checked roads where | `Validated = 'Yes'` |
| Imagery | `C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\tiles` |
| Output folder | `C:\Users\zhirman\Desktop\Centerline\Glynn\dl\chips` |
| Local roads | `Local_Roads_Glynn` (gives the score to beat) |
| Only inside these grids | leave empty (or select grids) |
| Training folder from 'End of Day' | your training folder |
| Only roads I moved | off at first; turn on once you have a few hundred moved roads |

It only reads the inputs. It writes `chips.zip` next to the output folder, and the
messages show the score to beat.

What it does with your data:
* 40 ft at every line end is left out, because lines bend through junctions (manual Examples 2–4).
* Abandoned and private roads are left out.
* Pixels near roads you have **not** checked are left out, so the model is never told "no road" where one is.
* 20% of the ground (1000 ft blocks) is kept for validation. The model never trains on those blocks, so the score is honest.
* Roads you traced from the county layer are used for training. They are left out of the score, because the county layer would win there by default.

### 2. Train in Colab

1. Upload `chips.zip` to Google Drive in `My Drive/Centerline/`.
2. Open `dl/Train_Centerline_Colab.ipynb` in Colab (File > Upload notebook).
3. **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**.
4. Each epoch prints the validation median error and the share within 1 ft. At the
   end it compares the model with the local roads. If Colab disconnects, Run all
   again: it resumes.

Results are written to `My Drive/Centerline/model/`: `best_model.pt` and
`model_info.json` (the scores per epoch).

### 3. What to send back

The last ~15 lines the notebook printed, or `model_info.json`.

## Next (not built yet)

* Prediction in Colab on corridors around every `Roads_working` line, giving heatmap GeoTIFFs.
* An Align option that follows the heatmap crest. It keeps each GDOT segment's
  direction and attributes, keeps ends snapped at junctions, and adds no curves.
  You still review and accept each edit (Replace Geometry > Trace).
