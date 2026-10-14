"""ArcGIS Pro Python toolbox definition (becomes ``arcgis/CenterlineAlign.pyt``)."""
import arcpy

from .arcgis_impl import run
from .config import ClassifierConfig, RefineConfig


class Toolbox:
    def __init__(self):
        self.label = "Centerline Alignment"
        self.alias = "centerline"
        self.tools = [AlignCenterlines]


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
        rcfg = RefineConfig(**{name: float(p[name].value) for name, _, _ in ADVANCED})
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
