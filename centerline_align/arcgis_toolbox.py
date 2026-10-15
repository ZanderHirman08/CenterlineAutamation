"""ArcGIS Pro Python toolbox definition (becomes ``arcgis/CenterlineAlign.pyt``)."""
import arcpy

from .arcgis_impl import evaluate, run
from .config import ClassifierConfig, RefineConfig


class Toolbox:
    def __init__(self):
        self.label = "Centerline Alignment"
        self.alias = "centerline"
        self.tools = [AlignCenterlines, EvaluateAlignment]


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
        params.append(_param("lidar_folder",
                             "LiDAR layers folder (optional; from make_lidar_layers.py)",
                             "DEFolder", required=False, category="LiDAR"))
        params.append(_param("retrain_from",
                             "Learn from an earlier result (optional; use a new work folder)",
                             "GPFeatureLayer", required=False, category="Second pass"))
        for name, label, default in (("red", "Red band", 1), ("green", "Green band", 2),
                                     ("blue", "Blue band", 3)):
            params.append(_param(name, label, "GPLong", category="Bands", value=default))
        params.append(_param("nir", "Near-infrared band (leave empty if none)", "GPLong",
                             required=False, category="Bands", value=4))
        for name, label, default in ADVANCED:
            params.append(_param(name, label + " (map units)" if "confidence" not in label
                                 else label, "GPDouble", category="Advanced", value=default))
        method = _param("method", "Refinement method", "GPString", category="Advanced",
                        value="ribbon")
        method.filter.type = "ValueList"
        method.filter.list = ["ribbon", "midpoint"]
        params.append(method)
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
        rcfg = RefineConfig(method=p["method"].valueAsText or "ribbon",
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
            diagnostics=bool(p["diagnostics"].value),
            retrain_from=p["retrain_from"].valueAsText,
            lidar_folder=p["lidar_folder"].valueAsText)

    def postExecute(self, parameters):
        return


def _split_multi(text) -> list[str]:
    """Values of a multi-value parameter ('a;b' with optional quotes)."""
    return [v.strip().strip("'\"") for v in (text or "").split(";") if v.strip()]


class EvaluateAlignment:
    def __init__(self):
        self.label = "Evaluate Against Checked Roads"
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
