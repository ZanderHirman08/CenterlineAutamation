# Centerline Alignment: REVAMP Phase 6, Glynn County

> **Paused (2026-10-20):** the imagery used for validation (latest GIO/Vexcel, a web tile service) is not available as files, and it does not line up with the downloadable 2023-24 GIO tiles the model trains on (offsets of 2 ft and more, varying by area). See [docs/HANDOFF.md](docs/HANDOFF.md) for what would let this resume.

**Goal:** put GDOT road centerlines (`Roads_working`) on the centre of the pavement in
the GIO imagery to **within a foot**, following the REVAMP Phase 6 manual.

**The plan:** a deep-learning model learns the pavement centre from the roads you
validate by hand. It is trained for free on Google Colab's GPU, because nothing
can be installed on the work PC. When it beats the county road layer on roads it
has never seen, it will draw the centrelines for you to review.

```
 You edit & validate roads ──► Start/End of Day tools record your edits (every day)
                                      │
                                      ▼
                     Export Training Chips ──► chips.zip (weekly)
                                      │
                                      ▼
                  Google Colab trains the model ──► score vs. county layer
                                      │
                                      ▼
        (next) model draws centrelines ──► you review in Pro ──► Validated = 'Yes'
```

---

## 1. Downloads

Branch: **`claude/road-centerline-alignment-albtwy`** (the files are on this branch).
Right-click a download link → **Save link as…**

| File | Download | Save it to |
|---|---|---|
| Toolbox | [CenterlineAlign.pyt](https://github.com/ZanderHirman08/CenterlineAutamation/raw/claude/road-centerline-alignment-albtwy/arcgis/CenterlineAlign.pyt) | `C:\Users\zhirman\Desktop\Centerline\Toolbox_v2\` |
| Colab notebook | [Train_Centerline_Colab.ipynb](https://github.com/ZanderHirman08/CenterlineAutamation/raw/claude/road-centerline-alignment-albtwy/dl/Train_Centerline_Colab.ipynb) | `C:\Users\zhirman\Desktop\Centerline\Glynn\dl\` |
| Imagery copy script (new counties only) | [copy_aoi_tiles.py](https://github.com/ZanderHirman08/CenterlineAutamation/raw/claude/road-centerline-alignment-albtwy/arcgis/copy_aoi_tiles.py) | `C:\Users\zhirman\Desktop\Centerline\` |

You can also open the notebook straight in Colab:
**[Open in Colab](https://colab.research.google.com/github/ZanderHirman08/CenterlineAutamation/blob/claude/road-centerline-alignment-albtwy/dl/Train_Centerline_Colab.ipynb)**.
Colab asks to connect to GitHub the first time; the repository is private.

Check the saved file names end in `.pyt` and `.ipynb`, not `.txt`. In File
Explorer, use View → Show → File name extensions to see them.

Websites: [Google Drive](https://drive.google.com) · [Google Colab](https://colab.research.google.com)

---

## 2. One-time setup

1. **Make folders:**
   ```
   C:\Users\zhirman\Desktop\Centerline\Toolbox_v2
   C:\Users\zhirman\Desktop\Centerline\Glynn\training
   C:\Users\zhirman\Desktop\Centerline\Glynn\dl
   ```
2. **Download** the toolbox and notebook (table above) into those folders.
3. **Add the toolbox in Pro.** In your *work* project (the one with `Roads_working`):
   Catalog pane → Toolboxes → right-click → **Add Toolbox** → `Toolbox_v2\CenterlineAlign.pyt`.
   You should see 5 tools. If not, restart Pro.
4. **Get the OK** from your supervisor/IT to upload imagery chips to Google Drive.
5. **Google:** in [Google Drive](https://drive.google.com), create a folder named `Centerline`.

The old Centerline project and its outputs are not touched. None of the tools
write to `Roads_working`; they only read it.

---

## 3. Every day you edit

| When | Tool | Settings |
|---|---|---|
| ☀️ Before your first edit | **1. Start of Day: Snapshot Roads** | Road layer: `Roads_working` · Checked where: `Validated = 'Yes'` · Training folder: `...\Glynn\training` |
| ✏️ During the day | *(edit normally, per the manual)* | Mark `Validated = 'Yes'` on every road you check |
| 🌙 After **saving** edits | **2. End of Day: Collect My Edits** | Same three settings, plus Reference road layer: `Local_Roads_Glynn` |

**For the best training data:** draw on the **centre of the pavement** in the GIO
imagery, zoomed in, rather than tracing the county layer. Spread your roads over
trees, shadows, wide and narrow roads, and different grids.

---

## 4. Once a week: train the model

### a. Export (in Pro): **3. Export Training Chips (weekly)**

| Setting | Value |
|---|---|
| Checked roads | `Roads_working` |
| Only checked roads where | `Validated = 'Yes'` |
| Imagery | `C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\tiles` |
| Output folder | `C:\Users\zhirman\Desktop\Centerline\Glynn\dl\chips` |
| Local roads | `Local_Roads_Glynn` |
| Training folder from 'End of Day' | `C:\Users\zhirman\Desktop\Centerline\Glynn\training` |
| Only roads I moved | unchecked (turn on once you have a few hundred moved roads) |

It writes **`...\Glynn\dl\chips.zip`** (next to the `chips` folder). Copy the messages;
they include the **score to beat** (the county layer's error).

### b. Upload
Drag `chips.zip` into **Google Drive → Centerline**, replacing last week's.

### c. Train (in Colab)
1. Open the notebook: [Open in Colab](https://colab.research.google.com/github/ZanderHirman08/CenterlineAutamation/blob/claude/road-centerline-alignment-albtwy/dl/Train_Centerline_Colab.ipynb),
   or Colab → File → Upload notebook → `Train_Centerline_Colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all**, and allow Google Drive access.
4. Wait 20–40 min. If it disconnects, **Run all** again; it resumes where it stopped.

Results go to **Google Drive → Centerline → model** (`best_model.pt`, `model_info.json`).

### d. Send back
The final "model vs. local roads" block it prints, and a screenshot of the example chips.

---

## 4b. Adding a county someone else finished

Finished counties have every road validated, which is a lot more training data. Other
editors' lines are less exact than yours, so they count for less (trust), and the
best model is still picked by your Glynn score.

**What you need for the county** (e.g. Camden):
* its finished `Roads_working`, from that county's Phase6Working.gdb
* its original ITOS-provided roads, from the OneDrive zip. These show what each road looked like before validation.
* its GIO imagery tiles, copied with `copy_aoi_tiles.py` to e.g. `C:\Users\zhirman\Desktop\Centerline\Camden\imagery\tiles`
* its local roads layer, optional; it gives that county's score to beat

**1. Export (in Pro): 3. Export Training Chips**

| Setting | Value |
|---|---|
| Checked roads | Camden's finished `Roads_working` |
| Only checked roads where | `Validated = 'Yes'` |
| Imagery | Camden's tiles folder |
| Output folder | `...\Centerline\Camden\d1\chips_camden` (makes `chips_camden.zip`) |
| Local roads | Camden's local roads (optional) |
| Training folder from 'End of Day' | leave empty |
| Only roads I moved | **checked**: only roads the editor actually moved, which are more likely carefully drawn |
| *Another county's finished data* → Original roads before validation | Camden's ITOS-provided roads |
| → County name | `camden` |
| → Trust | `0.5` |
| → Chip limit | `3000`, so a whole county fits in Colab |

**2. Test the current model on it first (Colab)**
1. Upload `chips_camden.zip` to Drive → Centerline.
2. In the notebook settings, set `CHIPS_ZIPS = ["chips_camden.zip"]` and `TEST_ONLY = True`, then **Run all**.
3. It scores your current model on Camden without training, against Camden's local roads.

The **suspect roads list** works here too. It flags Camden lines the model strongly disagrees with: lines marked Yes that may never have been checked.

**3. Train on both**
1. Set `CHIPS_ZIPS = ["chips.zip", "chips_camden.zip"]` (**Glynn first**) and `TEST_ONLY = False`.
2. Delete the `model` folder in Drive (or set `MODEL_FOLDER` to a new folder to keep the old model), then **Run all**.
3. Each epoch line shows Glynn's score and Camden's. The final table compares each county with its local roads.

**Is it helping?** Compare Glynn's score with the previous Glynn-only run. If Glynn got better or stayed the same, keep Camden in. If Glynn got worse, lower Camden's trust (e.g. 0.3) or leave it out.

## 5. Where everything lives

```
Desktop\Centerline\
├── Toolbox_v2\CenterlineAlign.pyt   ← the toolbox
└── Glynn\
    ├── imagery\tiles\               ← GIO imagery (already there)
    ├── training\                    ← Start/End of Day write here
    └── dl\
        ├── Train_Centerline_Colab.ipynb
        ├── chips\                   ← made by Export Training Chips
        └── chips.zip                ← upload this

Google Drive\Centerline\
├── chips.zip                        ← you upload
└── model\                           ← Colab writes the trained model here
```

---

## 6. The tools

| Tool | What it does |
|---|---|
| **1. Start of Day** | Remembers each road's shape before you edit it (stored in the training folder). |
| **2. End of Day** | Records every checked road: whether you moved it, by how much, and whether it was traced from the county layer. It leaves out roads with AbandonedRoad/PrivateRoad = 'Yes'. |
| **3. Export Training Chips** | Imagery chips (R, G, B, NIR) with your centrelines as the target. Details below. |
| **4. Evaluate Against Checked Roads** | Measures any layer's distance to your validated roads (open road, shadow, canopy, intersection). |
| **Align Centerlines to Imagery** | The current imagery-classifier method. It is accurate to about 1.5 ft, which is no better than the county layer. It will be switched to the model once the model wins. |

What the export does with your roads:
* **Ends of lines:** 40 ft at each end is left out, because lines bend through junctions (manual Examples 2–4). Where a road crosses itself (a "P" shape) the crossing is treated the same way.
* **Manual rules:** roads with AbandonedRoad/PrivateRoad = 'Yes', roads whose RC_Comments mention construction, a private/gated road or abandonment, and existing cul-de-sacs left as circles (loops under 400 ft) are left out of training and of Evaluate. Roundabouts (ROUTE_ID with TEMP) follow the pavement and are kept. Missing roads you add are included, since they are set to Validated = 'Yes'.
* **Roads you haven't checked:** pixels near them are left out, so the model is never told "no road" there.
* **Validation:** 20% of the ground (1000 ft blocks) is kept for testing and never trained on, so the score is honest.
* **Traced roads:** used for training but not scored, since the county layer would win there by default.

## 7. Manual rules the project keeps

Every rule in the manual, and how each one is handled: [docs/MANUAL_COMPLIANCE.md](docs/MANUAL_COMPLIANCE.md).


* No Attribute Transfer, and line direction is never changed.
* No curve tools; new cul-de-sacs are straight.
* Route ID, mile points, function type, county, system code and ownership are never touched.
* Routes stay connected at intersections.
* Nothing is deleted (abandoned roads use AbandonedRoad = 'Yes').
* GIO imagery is the reference.
* You review every change and mark Validated = 'Yes' yourself.

## 8. Status

| Step | Status |
|---|---|
| Daily edit capture (Start/End of Day) | ✅ built, not yet run on real data |
| Export Training Chips | ✅ built, not yet run on real data |
| Colab training notebook | ✅ built (tested on synthetic data) |
| First real training run, score vs. county layer | ⏳ next: needs upload approval |
| Model draws centrelines along `Roads_working` → review in Pro | 🔜 after the model beats the county layer |

Results so far, measured on validated roads in grids 7, 11, 16, 92, 284 and 285:
* County layer: median **1.42 ft** from the validated roads.
* Current Align tool: median **1.62 ft**.

That ~1.5 ft limit is why the project moved to a deep-learning model.

---

## For developers

* `centerline_align/`: the package. `scripts/build_pyt.py` bundles it into
  `arcgis/CenterlineAlign.pyt`. Only arcpy, numpy and scipy are allowed, with
  Python 3.9 syntax. Rebuild after any change; a test checks it.
* `dl/train_centerline.py`: the training code. `scripts/build_notebook.py`
  builds the Colab notebook from it. Rebuild after any change; a test checks it.
* Tests: `pip install -r requirements-dev.txt`, then `pytest`. ArcGIS is
  simulated by `tests/fake_arcpy.py` on a synthetic scene.
* `docs/HANDOFF.md`: project summary for picking the work up in a new session.
