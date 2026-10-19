# Handoff: Centerline Alignment (REVAMP Phase 6)

Read `README.md` first. This file adds the background a new session needs.

## Who / what
* The user is a GIS tech at a regional commission working on GDOT REVAMP Phase 6 ("Geometry").
  They validate and edit the state layer `Roads_working`
  (`S:\Analysts\Zhirm\ZREVAMP\Phase6Working.gdb\ReferenceData\Roads_working`) onto the
  pavement centre in GIO orthoimagery (0.5 ft, 4-band RGB+NIR, NAD83(2011) GA East, US ft).
  Glynn County first.
* `Local_Roads_Glynn` is the county layer, an independent source. The user uses it as a
  reference and to avoid private roads, and sometimes traces it (Replace Geometry → Trace).
* On `Roads_working` the user may edit only the geometry and `Validated` ('Yes'/'No').
  No new fields. Tools must only read it.

## Constraints
* Locked-down PC: ArcGIS Pro's Python only (arcpy, numpy, scipy; Python 3.9 syntax). No
  installs, no Image Analyst, no torch. GPU: AMD Radeon Pro W6400 (no CUDA).
* Deep learning runs on Google Colab (free T4). Uploading data needs the supervisor's/IT's OK.
  At last check that approval was still pending.
* The REVAMP manual rules are listed in the README (direction, no curves, no deletes,
  connectivity, Validated).

## Why deep learning
On 6 grids with validated roads, the county layer was 1.42 ft median from the validated
roads, and the classic Align tool (pixel classifier + ribbon DP) was 1.62 ft. LiDAR (1.67 ft)
and a second-pass classifier didn't help. That noise floor of about 1.5 ft can't reach
sub-foot, so the project moved to a U-Net (pretrained ResNet34 encoder,
segmentation_models_pytorch) predicting a Gaussian centreline ridge (σ = 1 ft) from R, G, B, NIR.

## Pipeline
1. `snapshot` / `collect` (`centerline_align/arcgis_training.py`) → `training_folder`
   holds `originals.json` and `training_roads.json` (per OID: changed, shift, traced, day).
2. `export_training_chips` → shards `shard_{train,val}_NNN.npz` (image uint8, heat uint8,
   mask) plus `.json` (scored lines in pixel coords), `manifest.json`, `baseline.json`
   (the county layer's error at the scored points), and `chips.zip`.
   * Split by 1000 ft blocks. Train chips have the mask off in val blocks; val chips are
     scored only on val-block mask pixels.
   * Mask off within 40 ft of any checked-line end (junctions) and near unchecked roads.
   * Traced roads are not scored. AbandonedRoad/PrivateRoad = 'Yes' are excluded.
3. `dl/train_centerline.py`: masked, ridge-weighted BCE; AdamW + OneCycle; checkpoints to
   Drive with resume. The score is the cross-section crest distance (parabolic subpixel)
   on val lines, compared with `baseline.json`.

## Next steps
1. First real export and Colab run; compare with the county layer (val median, within 1 ft).
2. If it wins: export corridor chips around every `Roads_working` line, predict in Colab,
   and write heatmap GeoTIFFs. Then add an Align mode that follows the heatmap crest
   (center-only DP), output 1:1 per GDOT segment, keep direction and junction snapping,
   no curves. Review in Pro, then Evaluate.
3. If not: more and more varied validated roads (trees, shadows, intersections); try
   test-time augmentation, a larger chip, a smaller σ.

## Removed (no longer the direction)
The LiDAR download/processing scripts and features, the second-pass classifier, the
"Overnight: Train From My Edits" tool and keep/move model, the old midpoint refinement,
and the standalone command-line pipeline (geopandas/rasterio, foundation-model
embeddings). They are in git history before this commit if ever needed.
