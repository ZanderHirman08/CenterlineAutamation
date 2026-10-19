"""ArcGIS Pro Python toolbox definition (becomes ``arcgis/CenterlineAlign.pyt``)."""
import arcpy

from .arcgis_impl import evaluate, run
from .arcgis_training import collect, export_training_chips, snapshot
from .config import ClassifierConfig, RefineConfig


class Toolbox:
    def __init__(self):
        self.label = "Centerline Alignment"
        self.alias = "centerline"
        self.tools = [StartOfDay, EndOfDay, ExportTrainingChips, EvaluateAlignment,
                      AlignCenterlines]


def _param(name, label, datatype, direction="Input", required=True, category=None, value=None):
    p = arcpy.Parameter(name=name, displayName=label, datatype=datatype,
                        parameterType="Required" if required else "Optional",
                        direction=direction, category=category)
    if value is not None:
        p.value = value
    return p


# (name, label, default) for the numeric settings, in RefineConfig units
ADVANCED = [
    ("station_spacing", "Cross-section spacing", 10.0),
    ("search_half_width", "Search distance each side of the line", 40.0),
    ("max_shift", "Maximum shift", 30.0),
    ("min_road_width", "Minimum pavement width", 10.0),
    ("max_road_width", "Maximum pavement width", 60.0),
    ("review_max_shift", "Flag lines moved more than", 15.0),
    ("review_min_confidence", "Flag lines with confidence below (0-1)", 0.2),
]


class AlignCenterlines:
    def __init__(self):
        self.label = "Align Centerlines to Imagery"
        self.description = (
            "Moves road centerlines onto the middle of the pavement in orthoimagery, one grid "
            "at a time. Only selected roads and grids are used. Results for each grid are saved "
            "in the work folder, so a stopped run picks up where it left off.")
        self.canRunInBackground = True

    def getParameterInfo(self):
        imagery = _param("imagery", "Imagery (raster, mosaic dataset, or folder of .tif tiles)",
                         ["GPRasterLayer", "DERasterDataset", "DEMosaicDataset", "DEFolder"])
        roads = _param("roads", "Local road centerlines", "GPFeatureLayer")
        roads.filter.list = ["Polyline"]
        road_id = _param("road_id_field", "Road ID field", "Field")
        road_id.parameterDependencies = [roads.name]
        grids = _param("grids", "Work grids (1-mile)", "GPFeatureLayer")
        grids.filter.list = ["Polygon"]
        grid_id = _param("grid_id_field", "Grid ID field", "Field")
        grid_id.parameterDependencies = [grids.name]
        work = _param("work_folder", "Work folder (progress and trained classifier)", "DEFolder")
        out = _param("out_fc", "Output aligned centerlines", "DEFeatureClass", direction="Output")
        params = [imagery, roads, road_id, grids, grid_id, work, out]
        for name, label, default in (("red", "Red band", 1), ("green", "Green band", 2),
                                     ("blue", "Blue band", 3)):
            params.append(_param(name, label, "GPLong", category="Bands", value=default))
        params.append(_param("nir", "Near-infrared band (leave empty if none)", "GPLong",
                             required=False, category="Bands", value=4))
        for name, label, default in ADVANCED:
            params.append(_param(name, label + " (map units)" if "confidence" not in label
                                 else label, "GPDouble", category="Advanced", value=default))
        params.append(_param("keep_flagged",
                             "Keep the original line where the result is flagged for review",
                             "GPBoolean", required=False, category="Advanced", value=True))
        params.append(_param("training_grids", "Grids used to train the classifier", "GPLong",
                             category="Advanced", value=8))
        params.append(_param("fresh", "Start over (discard saved progress and classifier)",
                             "GPBoolean", required=False, category="Advanced", value=False))
        params.append(_param("diagnostics",
                             "Save road-probability rasters per grid (large; for test runs)",
                             "GPBoolean", required=False, category="Advanced", value=False))
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        p = {x.name: x for x in parameters}
        nir = p["nir"].value
        bands = [int(p["red"].value), int(p["green"].value), int(p["blue"].value)]
        if nir not in (None, "", 0):
            bands.append(int(nir))
        rcfg = RefineConfig(keep_flagged=p["keep_flagged"].value is not False,
                            **{name: float(p[name].value) for name, _, _ in ADVANCED})
        ccfg = ClassifierConfig(training_grids=int(p["training_grids"].value))
        run(arcpy,
            imagery=p["imagery"].valueAsText,
            roads_fc=p["roads"].valueAsText,
            road_id_field=p["road_id_field"].valueAsText,
            grids_fc=p["grids"].valueAsText,
            grid_id_field=p["grid_id_field"].valueAsText,
            work_folder=p["work_folder"].valueAsText,
            out_fc=p["out_fc"].valueAsText,
            bands=bands, refine=rcfg, classifier=ccfg,
            fresh=bool(p["fresh"].value),
            diagnostics=bool(p["diagnostics"].value))

    def postExecute(self, parameters):
        return


def _split_multi(text) -> list[str]:
    """Values of a multi-value parameter ('a;b' with optional quotes)."""
    return [v.strip().strip("'\"") for v in (text or "").split(";") if v.strip()]


class EvaluateAlignment:
    def __init__(self):
        self.label = "4. Evaluate Against Checked Roads"
        self.description = (
            "Measures how far road layers (the original roads, aligned outputs) are from roads "
            "you checked by hand, per cover type: open, shadow, partial trees, canopy, "
            "intersections. Only reads the inputs.")
        self.canRunInBackground = True

    def getParameterInfo(self):
        truth = _param("truth", "Checked roads (truth)", "GPFeatureLayer")
        truth.filter.list = ["Polyline"]
        where = _param("truth_where", "Only checked roads where (SQL)", "GPString",
                       required=False, value="Validated = 'Yes'")
        cands = arcpy.Parameter(name="candidates", displayName="Layers to evaluate",
                                datatype="GPFeatureLayer", parameterType="Required",
                                direction="Input", multiValue=True)
        imagery = _param("imagery", "Imagery, for cover types (optional)",
                         ["GPRasterLayer", "DERasterDataset", "DEMosaicDataset", "DEFolder"],
                         required=False)
        grids = _param("grids", "Only inside these grids (optional)", "GPFeatureLayer",
                       required=False)
        out = _param("out_points", "Output check points with errors (optional)",
                     "DEFeatureClass", direction="Output", required=False)
        spacing = _param("spacing", "Check point spacing (map units)", "GPDouble",
                         category="Advanced", value=5.0)
        max_dist = _param("max_dist", "Ignore matches farther than (map units)", "GPDouble",
                          category="Advanced", value=40.0)
        bands = _param("bands", "Imagery bands R,G,B,NIR", "GPString", category="Advanced",
                       value="1,2,3,4")
        return [truth, where, cands, imagery, grids, out, spacing, max_dist, bands]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        import os
        p = {x.name: x for x in parameters}
        cands = [(os.path.basename(v) or v, v) for v in _split_multi(p["candidates"].valueAsText)]
        evaluate(arcpy,
                 truth_fc=p["truth"].valueAsText,
                 truth_where=p["truth_where"].valueAsText,
                 candidates=cands,
                 imagery=p["imagery"].valueAsText,
                 bands=[int(b) for b in (p["bands"].valueAsText or "1,2,3,4").split(",")],
                 grids_fc=p["grids"].valueAsText,
                 out_points=p["out_points"].valueAsText,
                 spacing=float(p["spacing"].value or 5.0),
                 max_dist=float(p["max_dist"].value or 40.0))

    def postExecute(self, parameters):
        return


def _checked_params():
    roads = _param("roads", "Road layer you edit", "GPFeatureLayer")
    roads.filter.list = ["Polyline"]
    where = _param("checked_where", "Checked roads are where (SQL)", "GPString",
                   value="Validated = 'Yes'")
    folder = _param("training_folder", "Training folder (keep the same one every day)",
                    "DEFolder")
    return [roads, where, folder]


class StartOfDay:
    def __init__(self):
        self.label = "1. Start of Day: Snapshot Roads"
        self.description = (
            "Run before you start editing. Remembers the original shape of every road, so "
            "'End of Day' can tell what you changed. Only reads the road layer.")
        self.canRunInBackground = True

    def getParameterInfo(self):
        return _checked_params()

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        p = {x.name: x for x in parameters}
        snapshot(arcpy, roads_fc=p["roads"].valueAsText,
                 checked_where=p["checked_where"].valueAsText,
                 training_folder=p["training_folder"].valueAsText)

    def postExecute(self, parameters):
        return


class EndOfDay:
    def __init__(self):
        self.label = "2. End of Day: Collect My Edits"
        self.description = (
            "Run when you finish editing (save your edits first). Every checked road becomes "
            "training data, with how far you moved it from its original; abandoned and private "
            "roads are left out. Only reads the road layer.")
        self.canRunInBackground = True

    def getParameterInfo(self):
        params = _checked_params()
        ref = _param("reference_fc", "Reference road layer you trace from (optional, e.g. the "
                     "county roads)", "GPFeatureLayer", required=False)
        ref.filter.list = ["Polyline"]
        params.append(ref)
        params.append(_param("out_fc", "Training roads to look at (optional)", "DEFeatureClass",
                             direction="Output", required=False))
        params.append(_param("min_change", "Count a road as moved when an edit moved it at "
                             "least (map units)", "GPDouble", category="Advanced", value=1.0))
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        p = {x.name: x for x in parameters}
        collect(arcpy, roads_fc=p["roads"].valueAsText,
                checked_where=p["checked_where"].valueAsText,
                training_folder=p["training_folder"].valueAsText,
                out_fc=p["out_fc"].valueAsText,
                min_change=float(p["min_change"].value or 1.0),
                reference_fc=p["reference_fc"].valueAsText)

    def postExecute(self, parameters):
        return


class ExportTrainingChips:
    def __init__(self):
        self.label = "3. Export Training Chips (weekly)"
        self.description = (
            "Cuts imagery chips around roads you checked by hand and labels where their true "
            "centreline is, for training a centreline model in Google Colab (see "
            "README.md). Writes a zip to upload. Only reads the inputs.")
        self.canRunInBackground = True

    def getParameterInfo(self):
        truth = _param("truth", "Checked roads (truth)", "GPFeatureLayer")
        truth.filter.list = ["Polyline"]
        where = _param("truth_where", "Only checked roads where (SQL)", "GPString",
                       value="Validated = 'Yes'")
        imagery = _param("imagery", "Imagery (raster, mosaic dataset, or folder of .tif tiles)",
                         ["GPRasterLayer", "DERasterDataset", "DEMosaicDataset", "DEFolder"])
        out = _param("out_folder", "Output folder for the chips", "DEFolder", direction="Output")
        local = _param("local", "Local roads, to measure the score to beat (optional)",
                       "GPFeatureLayer", required=False)
        grids = _param("grids", "Only inside these grids (optional)", "GPFeatureLayer",
                       required=False)
        folder = _param("training_folder", "Training folder from 'End of Day' (optional: "
                        "leaves traced roads out of the score)", "DEFolder", required=False)
        moved = _param("only_moved", "Only roads I moved (needs the training folder)",
                       "GPBoolean", required=False, value=False)
        bands = _param("bands", "Imagery bands R,G,B,NIR", "GPString", category="Advanced",
                       value="1,2,3,4")
        chip = _param("chip_px", "Chip size (pixels)", "GPLong", category="Advanced", value=256)
        step = _param("step", "Distance between chips along a road (map units)", "GPDouble",
                      category="Advanced", value=100.0)
        val = _param("val_fraction", "Share of ground kept for validation (0-1)", "GPDouble",
                     category="Advanced", value=0.2)
        block = _param("block", "Validation block size (map units)", "GPDouble",
                       category="Advanced", value=1000.0)
        original = _param("original_fc", "Original roads before validation (ITOS-provided "
                          "data; optional)", "GPFeatureLayer", required=False,
                          category="Another county's finished data")
        original.filter.list = ["Polyline"]
        source = _param("source", "County name (optional)", "GPString", required=False,
                        category="Another county's finished data")
        trust = _param("trust", "Trust (1 = roads you checked yourself; e.g. 0.5 for another "
                       "editor's)", "GPDouble", category="Another county's finished data",
                       value=1.0)
        max_chips = _param("max_chips", "Chip limit (0 = no limit; e.g. 3000 for a whole "
                           "county)", "GPLong", category="Another county's finished data",
                           value=0)
        return [truth, where, imagery, out, local, grids, folder, moved, bands, chip, step, val,
                block, original, source, trust, max_chips]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        p = {x.name: x for x in parameters}
        export_training_chips(
            arcpy,
            truth_fc=p["truth"].valueAsText,
            truth_where=p["truth_where"].valueAsText,
            imagery=p["imagery"].valueAsText,
            bands=[int(b) for b in (p["bands"].valueAsText or "1,2,3,4").split(",")],
            out_folder=p["out_folder"].valueAsText,
            local_fc=p["local"].valueAsText,
            grids_fc=p["grids"].valueAsText,
            chip_px=int(p["chip_px"].value or 256),
            step=float(p["step"].value or 100.0),
            val_fraction=float(p["val_fraction"].value or 0.2),
            block=float(p["block"].value or 1000.0),
            training_folder=p["training_folder"].valueAsText,
            only_moved=bool(p["only_moved"].value),
            original_fc=p["original_fc"].valueAsText,
            source=p["source"].valueAsText,
            trust=float(p["trust"].value if p["trust"].value is not None else 1.0),
            max_chips=int(p["max_chips"].value or 0))

    def postExecute(self, parameters):
        return
