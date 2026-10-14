# Setting up a run

## 1. Folder structure

Put everything on a **local drive** (not a network share; the tool reads a
lot of imagery) with a short path and no spaces:

```
D:\Centerline\
├── toolbox\
│   ├── CenterlineAlign.pyt          the toolbox (add it in Pro)
│   └── copy_aoi_tiles.py            helper for step 2
├── imagery\
│   └── 2024\
│       └── tiles\                   tiles covering your grids + their .aux.xml / .ovr / .tfw
├── data\
│   └── Inputs.gdb                   Local_Roads, Grids_1mi (copies, not the originals)
├── work\
│   ├── 2024_test\                   work folder for the 2-3 grid test
│   └── 2024_run1\                   work folder for the full run: start it empty
├── output\
│   └── Results.gdb                  aligned_centerlines (and the dissolved version later)
└── Centerline.aprx                  your Pro project
```

Rules of thumb:

* **One work folder per run.** It holds per-grid progress and the trained
  classifier. If you change settings or imagery, make a new folder
  (`2024_run2`) or tick *Start over*; otherwise finished grids are skipped.
* **Work on copies.** Export the local roads and grids into `Inputs.gdb`
  (right-click → Data → Export Features) so the originals are never touched.
* **Check disk space first.** Each of your tiles is ~380 MB uncompressed and
  covers 5,000 × 5,000 ft (~0.9 sq mi), so expect roughly 1.2-1.5 tiles per
  1-mile grid once edges are counted: 400 grids is ~500-600 tiles,
  **~190-230 GB**. The copy script's dry run gives the exact figure. If you
  don't have the room, skip the copy and point the tool straight at the
  statewide tile folder: it only reads the tiles it needs, but it checks every
  tile when it starts, which is slow for a whole state.

## 2. Getting the imagery for your grids

Your 1-mile grids are the area of interest (AOI), so cut the imagery to the
grids, not the county line. First, how is the state imagery stored? Look at it
in the Catalog pane:

* **A folder of `.tif` tiles** (what you described): use **A**. No clipping;
  the tiles touching your grids are copied as they are.
* **A mosaic dataset** (inside a `.gdb`) or an **image service** (a URL): use **B**.

### A. Copy the tiles that touch your grids (recommended)

Copying whole tiles is exact (no resampling, no new NoData edges) and quick.

1. Open `toolbox\copy_aoi_tiles.py` in Notepad and set the settings at the top:
   ```python
   STATE_TILES = r"D:\Imagery\State_2024\tiles"         # statewide tile folder
   AOI = r"D:\Centerline\data\Inputs.gdb\Grids_1mi"     # your work grids
   BUFFER = 500                                         # in the grids' units (feet for State Plane ft)
   OUT_TILES = r"D:\Centerline\imagery\2024\tiles"
   DRY_RUN = True                                       # count first, copy second
   ```
   The tool reads about 100 ft past each grid edge (search distance + maximum
   shift + a little context), so 500 ft is a comfortable margin. Lower it to
   ~150 if disk space is tight.
2. In Pro: **Analysis** tab → **Python** → *Python Window*. Paste the whole
   file and press Enter. With `DRY_RUN = True` it only reports how many
   tiles touch your grids and how many GB they are, so you can check disk space.
3. Set `DRY_RUN = False`, paste again, and press Enter to copy.
4. Check: add the `tiles` folder to a map with the grids. Every grid should be
   covered, with a margin and no gaps.

If `AOI` is a layer in your map with grids selected, only the selected grids
are used. That's handy for a quick test area. It copies each tile's sidecar
files too (`.aux.xml` often holds the projection, `.ovr` the pyramids); keep
them together.

### B. Clip a mosaic dataset or image service

1. **Buffer the grids:** *Pairwise Buffer*, input = Grids_1mi, distance = 500 Feet,
   dissolve = *Dissolve all*. Save as `Inputs.gdb\Grids_buf500`.
2. **Clip Raster** (Data Management → Raster → Raster Processing):
   * Input raster: the state mosaic dataset / image service layer
   * Output extent: `Grids_buf500`
   * **Leave *Use input features for clipping geometry* unchecked**. This keeps
     a plain rectangle and avoids NoData inside the margin.
   * Output raster: `D:\Centerline\imagery\2024\AOI_2024.crf` (Cloud Raster
     Format: tiled and compressed, and much smaller than a single giant .tif)
   * NoData value: leave empty, or `0`
   * Environments: don't change the cell size or the projection
3. Point the toolbox at the `.crf`.

Image services often limit how much you can export. If the clip fails or
comes out at a lower resolution, ask the state for the tiles instead.

## 3. Run the toolbox

1. Add `CenterlineAlign.pyt` to the project (Catalog → Toolboxes → Add Toolbox).
2. Add `Local_Roads` and `Grids_1mi` to the map. **Select 2-3 grids** for a
   first test, one urban and one rural.
3. Open **Align Centerlines to Imagery**:

   | Parameter | Value |
   |---|---|
   | Imagery | `D:\Centerline\imagery\2024\tiles` (or the `.crf`) |
   | Local road centerlines | `Local_Roads` (layer) |
   | Road ID field | the unique ID you'll join on later |
   | Work grids | `Grids_1mi` (layer) |
   | Grid ID field | the grid's ID |
   | Work folder | `D:\Centerline\work\2024_test` |
   | Output | `D:\Centerline\output\Results.gdb\aligned_test` |
   | Bands | R 1, G 2, B 3, NIR 4 |

4. Read the messages: it reports the imagery's units (all distance settings
   use them), which bands it used, and progress per grid.
5. Look at the output over the imagery. Symbolise `REVIEW` (1 = check) and
   look at a few unflagged lines too.
6. Happy with it? Clear the grid selection and run all the grids with a
   **new** work folder (`work\2024_run1`) and output
   (`Results.gdb\aligned_centerlines`). The classifier is trained on the
   first run in a work folder, from grids spread across whatever is
   selected. A test folder would leave it trained on just your 2-3 test grids.
   Expect about a minute per grid, so plan an overnight run. If it stops,
   run it again with the same work folder and it picks up where it left off.
