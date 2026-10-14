"""Run configuration, loaded from a YAML file.

All distances are in the linear units of the working CRS (``crs``), which is
normally your State Plane projection in US feet.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class InputConfig:
    imagery: str                      # a GeoTIFF / VRT, or a folder of GeoTIFF tiles
    roads: str                        # local road centerlines (shp / gpkg)
    grids: str                        # 1-mile grid polygons (shp / gpkg)
    road_id_field: str = "OBJECTID"   # unique id carried to the output
    grid_id_field: str = "GRID_ID"
    roads_layer: str | None = None    # layer name when reading from a gpkg
    grids_layer: str | None = None


@dataclass
class BandConfig:
    # 1-based band indexes in the imagery. nir is optional.
    red: int = 1
    green: int = 2
    blue: int = 3
    nir: int | None = None


@dataclass
class ClassifierConfig:
    # "pixel": spectral + texture features (no GPU needed)
    # "embedding": pixel features + frozen foundation-model embeddings (GPU)
    features: str = "pixel"
    embedding_model: str = "facebook/dinov2-small"
    embedding_device: str = "cuda"
    embedding_tile: int = 518         # pixels per inference tile (multiple of patch size)
    # Training labels come from the local road file itself (distances in CRS units).
    positive_half_width: float = 4.0  # pixels within this distance of a road = road
    negative_min_dist: float = 45.0   # pixels between these distances from any road = not road
    negative_max_dist: float = 120.0
    samples_per_class: int = 20000
    training_grids: int = 8           # how many grids to sample training pixels from
    model_path: str | None = None     # reuse a saved classifier instead of training


@dataclass
class RefineConfig:
    station_spacing: float = 10.0     # distance between cross-sections along a road
    search_half_width: float = 40.0   # how far each side of the line to look
    max_shift: float = 30.0           # never move a line farther than this
    min_road_width: float = 10.0      # plausible pavement widths
    max_road_width: float = 60.0
    prob_threshold: float = 0.5
    smooth_window: int = 7            # stations used for the along-road median filter
    outlier_tolerance: float = 4.0    # station offsets farther than this from the median are dropped
    simplify_tolerance: float = 0.5
    node_snap_tolerance: float = 1.0  # input endpoints closer than this are one node
    node_taper: float = 50.0          # distance over which an end's move into its node fades out
    review_min_confidence: float = 0.6
    review_max_shift: float = 15.0


@dataclass
class OutputConfig:
    folder: str = "output"
    name: str = "aligned_centerlines"
    write_shapefile: bool = True
    write_geopackage: bool = True
    write_stations: bool = False      # per-station QA points (large)
    write_probability: bool = False   # road-probability GeoTIFF per grid


@dataclass
class Config:
    inputs: InputConfig
    crs: str | None = None            # working CRS; defaults to the imagery CRS
    resolution: float | None = None   # working pixel size; defaults to the imagery's own
    grid_ids: list | None = None      # process only these grids
    bands: BandConfig = field(default_factory=BandConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        sections = {
            "inputs": InputConfig,
            "bands": BandConfig,
            "classifier": ClassifierConfig,
            "refine": RefineConfig,
            "output": OutputConfig,
        }
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        kwargs = {}
        for key, value in raw.items():
            if key in sections:
                kwargs[key] = _build(sections[key], value or {}, key)
            else:
                kwargs[key] = value
        if "inputs" not in kwargs:
            raise ValueError("Config needs an 'inputs' section")
        return cls(**kwargs)


def _build(dc, values: dict, section: str):
    known = {f.name for f in fields(dc)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys in '{section}': {sorted(unknown)}")
    return dc(**values)
