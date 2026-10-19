# -*- coding: utf-8 -*-
"""Centerline Alignment - ArcGIS Pro Python toolbox.

GENERATED FILE - do not edit by hand. Built by scripts/build_pyt.py from the
centerline_align package (version 0.1.0).

Add to ArcGIS Pro: Catalog pane > Toolboxes > right-click > Add Toolbox, pick this
file. Needs nothing beyond ArcGIS Pro's own Python (arcpy, numpy, scipy).
"""
from __future__ import annotations

import sys as _sys
import types as _types

# dataclasses look up the defining module in sys.modules; ArcGIS may load a .pyt
# without registering it there, so stand in a placeholder while classes are built.
_placeholder = None
if __name__ not in _sys.modules:
    _placeholder = _sys.modules[__name__] = _types.ModuleType(__name__)


# ======================================================================
# config.py
# ======================================================================

"""Run configuration, loaded from a YAML file.

All distances are in the linear units of the working CRS (``crs``), which is
normally your State Plane projection in US feet.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path


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
    review_min_confidence: float = 0.2
    review_max_shift: float = 15.0
    # "ribbon": fit both pavement edges and a smooth width along the whole line at once
    # (shadows on one edge, cross streets and single bad cross-sections barely move it).
    # "midpoint": the older method - midpoint of each cross-section, then smoothing.
    method: str = "ribbon"
    junction_mask: float = 30.0       # ignore cross-sections this close to an intersection
    offset_stiffness: float = 0.5     # cost of the line bending sideways (per ft^2 per station)
    width_stiffness: float = 0.25     # cost of the width changing between stations
    width_prior: float = 0.1          # pull toward the line's typical width (narrowing)
    stay_prior: float = 0.03          # pull toward the original position (no evidence = no move)
    # Flagged lines (REVIEW = 1) keep their original geometry instead of a doubtful move:
    # checked against hand-validated roads, moves under trees and big jumps made lines
    # worse more often than better.
    keep_flagged: bool = True


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
        import yaml

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


# ======================================================================
# chip.py
# ======================================================================

"""A georeferenced block of imagery, independent of how it was read."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Chip:
    """Bands (B, H, W) float32, validity mask (H, W) and georeference.

    ``x0, y0`` is the top-left corner; pixels are square with size ``res``.
    """
    data: np.ndarray
    valid: np.ndarray
    x0: float
    y0: float
    res: float
    # optional LiDAR layers resampled to this chip: (intensity, ground elevation,
    # surface elevation), heights in metres; aux_valid marks where they exist
    aux: np.ndarray | None = None
    aux_valid: np.ndarray | None = None
    xy_to_m: float = 0.3048006096  # metres per CRS unit (for slopes)

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.shape
        return self.x0, self.y0 - h * self.res, self.x0 + w * self.res, self.y0

    def world_to_pixel(self, x, y):
        """Fractional (row, col) of pixel centres for world coords."""
        col = (np.asarray(x) - self.x0) / self.res - 0.5
        row = (self.y0 - np.asarray(y)) / self.res - 0.5
        return row, col


# ======================================================================
# geom.py
# ======================================================================

"""Polyline helpers on plain (N, 2) coordinate arrays (no GIS library needed)."""

import numpy as np


def as_coords(coords) -> np.ndarray:
    c = np.asarray(coords, dtype=np.float64)[:, :2]
    keep = np.ones(len(c), bool)
    keep[1:] = np.any(np.diff(c, axis=0) != 0, axis=1)  # drop repeated vertices
    return c[keep]


def cumlen(c: np.ndarray) -> np.ndarray:
    """Distance along the line at each vertex."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])


def interpolate(c: np.ndarray, d, s: np.ndarray | None = None) -> np.ndarray:
    """Points at distances ``d`` along the line (clamped to its ends)."""
    s = cumlen(c) if s is None else s
    d = np.clip(np.asarray(d, dtype=np.float64), 0.0, s[-1])
    return np.column_stack([np.interp(d, s, c[:, 0]), np.interp(d, s, c[:, 1])])


def densify(c: np.ndarray, spacing: float) -> np.ndarray:
    s = cumlen(c)
    if s[-1] == 0:
        return c[:1]
    n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
    return interpolate(c, np.linspace(0, s[-1], n), s)


def simplify(c: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker; always keeps both ends."""
    if len(c) <= 2 or tol <= 0:
        return c
    keep = np.zeros(len(c), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(c) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = c[i], c[j]
        ab = b - a
        L = np.hypot(*ab)
        seg = c[i + 1:j] - a
        if L == 0:
            dist = np.hypot(seg[:, 0], seg[:, 1])
        else:
            dist = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return c[keep]


# ======================================================================
# features.py
# ======================================================================

"""Per-pixel features for the road / not-road classifier.

Two sources, which the classifier can combine:

* ``pixel_features`` - spectral and texture measures computed from the bands
  directly. Cheap, needs no GPU.
* ``EmbeddingExtractor`` - patch embeddings from a frozen, pretrained vision
  foundation model (DINOv2 / DINOv3 via Hugging Face). These come out at
  patch resolution (e.g. one vector per 14x14 or 16x16 pixels) and are kept at
  that resolution; the classifier is linear, so its output can be upsampled
  instead of the (much larger) embeddings.
"""

import numpy as np
from scipy import ndimage

EPS = 1e-6


def pixel_feature_names(has_nir: bool) -> list[str]:
    names = ["red", "green", "blue"]
    if has_nir:
        names.append("nir")
    names += ["bright", "sat", "exg"]
    if has_nir:
        names.append("ndvi")
    names += ["bright_s2", "bright_s6", "bright_std3", "sat_s4", "exg_s4"]
    if has_nir:
        names.append("ndvi_s4")
    return names


def scale_bands(img: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Linearly stretch raw band values to roughly 0-1 using training percentiles."""
    lo = np.asarray(lo, dtype=np.float32)[:, None, None]
    hi = np.asarray(hi, dtype=np.float32)[:, None, None]
    return (img.astype(np.float32) - lo) / np.maximum(hi - lo, EPS)


def pixel_features(scaled: np.ndarray, has_nir: bool) -> np.ndarray:
    """Features from scaled bands (R, G, B[, NIR]) shaped (bands, H, W).

    Returns (n_features, H, W) float32, in the order of ``pixel_feature_names``.
    """
    # Clip and lift away from zero so ratios stay bounded in shadows and at the
    # ends of the stretch (dividing by ~0 would give huge, fit-breaking values).
    s = np.clip(scaled, 0.0, 1.5) + 0.05
    r, g, b = s[0], s[1], s[2]
    rgb_max = np.maximum(np.maximum(r, g), b)
    rgb_min = np.minimum(np.minimum(r, g), b)
    bright = (r + g + b) / 3.0
    sat = (rgb_max - rgb_min) / rgb_max          # pavement is grey: low saturation, 0-1
    exg = 2.0 * g - r - b                        # excess green: vegetation without NIR

    feats = [r, g, b]
    if has_nir:
        nir = s[3]
        ndvi = (nir - r) / (nir + r)             # -1 to 1
        feats.append(nir)
    feats += [bright, sat, exg]
    if has_nir:
        feats.append(ndvi)

    blur = ndimage.gaussian_filter
    b2 = blur(bright, 2)
    var3 = blur(bright * bright, 3) - blur(bright, 3) ** 2
    feats += [b2, blur(bright, 6), np.sqrt(np.maximum(var3, 0)), blur(sat, 4), blur(exg, 4)]
    if has_nir:
        feats.append(blur(ndvi, 4))
    return np.stack(feats).astype(np.float32)


def lidar_feature_names(has_nir: bool) -> list[str]:
    names = ["l_int", "l_int_s", "l_height", "l_canopy", "l_slope", "l_rough",
             "x_canopy_bright", "x_canopy_sat"]
    if has_nir:
        names.append("x_canopy_ndvi")
    return names


def lidar_features(chip, img_feats: np.ndarray, has_nir: bool) -> np.ndarray:
    """Features from the chip's LiDAR layers (see ``Chip.aux``); NaN where there is
    no LiDAR. Ground intensity shows asphalt under gaps in the canopy, the ground
    surface is flat and smooth on a road bed, canopy height says where the imagery
    can't see the ground, and the canopy x imagery terms let the (linear)
    classifier trust the imagery less under trees."""
    inten, dtm, dsm = (a.astype(np.float32) for a in chip.aux)
    ok = chip.aux_valid
    if not ok.any():
        return np.full((len(lidar_feature_names(has_nir)),) + chip.shape, np.nan, np.float32)
    for a in (inten, dtm, dsm):  # neutral values where missing, so blurs stay sane
        a[~ok] = np.median(a[ok])
    res_m = chip.res * chip.xy_to_m
    blur = ndimage.gaussian_filter
    height = np.clip(dsm - dtm, 0, 40)
    canopy = np.clip((height - 1.0) / 8.0, 0, 1)
    gy, gx = np.gradient(dtm, res_m)
    slope = np.hypot(gx, gy)
    sig = 1.5 / res_m
    rough = np.sqrt(np.maximum(blur(dtm * dtm, sig) - blur(dtm, sig) ** 2, 0))
    names = pixel_feature_names(has_nir)
    feats = [inten, blur(inten, 1.0 / res_m), height, canopy, slope, rough,
             canopy * img_feats[names.index("bright")], canopy * img_feats[names.index("sat")]]
    if has_nir:
        feats.append(canopy * img_feats[names.index("ndvi")])
    out = np.stack(feats).astype(np.float32)
    out[:, ~ok] = np.nan
    return out


def all_features(chip, scaled: np.ndarray, has_nir: bool, lidar: bool) -> np.ndarray:
    """Pixel features, plus LiDAR features when the classifier uses them."""
    feats = pixel_features(scaled, has_nir)
    if not lidar:
        return feats
    if chip.aux is None:
        raise ValueError("This classifier was trained with LiDAR layers: give the same "
                         "LiDAR folder again (or retrain without LiDAR in a new work folder).")
    return np.concatenate([feats, lidar_features(chip, feats, has_nir)])


def sample_lowres(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch: int) -> np.ndarray:
    """Bilinearly sample a patch-resolution array (C, h, w) at full-res pixel positions.

    Patch ``i`` covers pixels ``i*patch .. (i+1)*patch-1``, so its centre is at
    pixel ``(i + 0.5) * patch - 0.5``.
    """
    pr = (np.asarray(rows, dtype=np.float64) + 0.5) / patch - 0.5
    pc = (np.asarray(cols, dtype=np.float64) + 0.5) / patch - 0.5
    out = [ndimage.map_coordinates(c, [pr, pc], order=1, mode="nearest") for c in grid]
    return np.stack(out)


def upsample_lowres(grid2d: np.ndarray, shape: tuple[int, int], patch: int) -> np.ndarray:
    """Bilinearly upsample a single patch-resolution band to ``shape`` (full res)."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    return sample_lowres(grid2d[None], rows.ravel(), cols.ravel(), patch)[0].reshape(shape)


class EmbeddingExtractor:
    """Frozen foundation-model patch embeddings for an RGB tile.

    Works with Hugging Face vision transformers that return one token per
    patch (optionally after a CLS token and register tokens), e.g.
    ``facebook/dinov2-small`` (Apache-2.0) or a DINOv3 satellite checkpoint
    (gated; accept its licence on Hugging Face first).
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover - depends on optional install
            raise ImportError(
                "Embedding features need PyTorch and transformers: "
                "pip install -r requirements-gpu.txt"
            ) from e

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("embedding_device is 'cuda' but no CUDA GPU is visible to PyTorch")
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        if device == "cuda":
            self.model = self.model.half()
        cfg = self.model.config
        self.patch = int(getattr(cfg, "patch_size", 14))
        self.n_registers = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.dim = int(cfg.hidden_size)
        self.mean, self.std = _normalisation(model_name)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (3, H, W) scaled 0-1. Returns (dim, ceil(H/p), ceil(W/p)) float32."""
        torch = self.torch
        _, h, w = rgb.shape
        p = self.patch
        hp, wp = -(-h // p), -(-w // p)
        x = np.zeros((3, hp * p, wp * p), dtype=np.float32)
        x[:, :h, :w] = np.clip(rgb, 0, 1)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        t = torch.from_numpy(x)[None].to(self.device)
        if self.device == "cuda":
            t = t.half()
        with torch.no_grad():
            tokens = self.model(pixel_values=t).last_hidden_state[0]
        tokens = tokens[-hp * wp:]  # drop CLS / register tokens at the front
        emb = tokens.float().cpu().numpy().T.reshape(self.dim, hp, wp)
        return emb.astype(np.float32)


def _normalisation(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(model_name)
        mean = list(getattr(proc, "image_mean", mean))
        std = list(getattr(proc, "image_std", std))
    except Exception:
        pass
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


# ======================================================================
# classifier.py
# ======================================================================

"""Road / not-road pixel classifier trained from the local road file.

The local centerlines are used as free (if slightly noisy) labels: pixels very
close to a road line are "road", pixels a good distance from every road are
"not road". A logistic regression on those samples gives a road-probability
map that is far more robust than a fixed colour threshold.

The model is linear on standardised features, so the embedding part can be
evaluated at patch resolution and upsampled afterwards (see ``predict``).
It is saved as plain JSON - no pickles. Only numpy and scipy are needed, so it
also runs inside ArcGIS Pro's built-in Python.
"""

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage


log = logging.getLogger("centerline_align")


@dataclass
class RoadClassifier:
    feature_mode: str                 # "pixel" or "embedding"
    has_nir: bool
    band_lo: list
    band_hi: list
    pix_names: list
    pix_mean: list
    pix_std: list
    pix_coef: list
    intercept: float
    embedding_model: str | None = None
    emb_mean: list | None = None
    emb_std: list | None = None
    emb_coef: list | None = None
    train_accuracy: float | None = None
    trained_from: str | None = None   # second pass: the earlier result it learned from
    lidar: bool = False               # uses LiDAR layers as well as the imagery
    move_model: dict | None = None    # keep/move model learned from checked roads

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, path) -> "RoadClassifier":
        with open(path) as f:
            raw = json.load(f)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})

    def predict(self, chip: Chip, embedder: EmbeddingExtractor | None = None) -> np.ndarray:
        """Road probability (H, W) float32 for a chip; 0 where imagery is missing."""
        scaled = scale_bands(chip.data, self.band_lo, self.band_hi)
        feats = all_features(chip, scaled, self.has_nir, self.lidar)
        mean = np.asarray(self.pix_mean, np.float32)[:, None, None]
        if self.lidar:  # no LiDAR here: those features count as average (no effect)
            feats = np.where(np.isnan(feats), mean, feats)
        w = np.asarray(self.pix_coef, np.float32) / np.asarray(self.pix_std, np.float32)
        z = np.tensordot(w, feats - mean, axes=1)
        z += self.intercept
        if self.feature_mode == "embedding":
            if embedder is None:
                raise ValueError("This classifier needs an EmbeddingExtractor")
            emb = embedder(scaled[:3])
            we = np.asarray(self.emb_coef, np.float32) / np.asarray(self.emb_std, np.float32)
            ze = np.tensordot(we, emb - np.asarray(self.emb_mean, np.float32)[:, None, None], axes=1)
            z += upsample_lowres(ze, chip.shape, embedder.patch)
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        prob[~chip.valid] = 0.0
        return prob.astype(np.float32)


def burn_lines(chip: Chip, lines) -> np.ndarray:
    """Boolean raster marking every pixel a line passes through."""
    burned = np.zeros(chip.shape, bool)
    h, w = chip.shape
    for c in lines:
        if c is None or len(c) < 2:
            continue
        pts = densify(np.asarray(c, float)[:, :2], chip.res * 0.5)
        row, col = chip.world_to_pixel(pts[:, 0], pts[:, 1])
        r, k = np.round(row).astype(int), np.round(col).astype(int)
        ok = (r >= 0) & (r < h) & (k >= 0) & (k < w)
        burned[r[ok], k[ok]] = True
    return burned


def distance_to_roads(chip: Chip, lines) -> np.ndarray:
    """Distance (CRS units) from each chip pixel centre to the nearest road line."""
    burned = burn_lines(chip, lines)
    if not burned.any():
        return np.full(chip.shape, np.inf, dtype=np.float32)
    return (ndimage.distance_transform_edt(~burned) * chip.res).astype(np.float32)


def sample_training_pixels(chip: Chip, lines, ccfg, rng, n_per_class: int, trusted=None):
    """Pick (row, col) indexes of positive and negative training pixels in a chip.

    trusted: optional (lines with known pavement width [(coords, half_width)],
    untrusted lines [coords]) from an earlier run (second pass), or hand-checked
    lines with half_width None (labelled road near the line only). Then the road
    labels cover the measured pavement (minus a margin) of trusted lines, and
    the verges just beyond their edges are labelled not-road too, which teaches
    the classifier where pavement ends.
    """
    dist = distance_to_roads(chip, lines)
    ok = chip.valid
    far = (dist > ccfg.negative_min_dist) & (dist <= ccfg.negative_max_dist) & ok
    if trusted is not None:
        good, bad = trusted
        pos_m = np.zeros(chip.shape, bool)
        verge = np.zeros(chip.shape, bool)
        by_width: dict = {}
        for c, hw in good:   # hw None: width unknown (hand-checked centerlines)
            key = None if hw is None else float(np.round(hw * 0.5) * 2)   # 2-unit bins
            by_width.setdefault(key, []).append(c)
        for hw, group in by_width.items():
            d = distance_to_roads(chip, group)
            if hw is None:
                pos_m |= d <= ccfg.positive_half_width
                continue
            pos_m |= d <= max(ccfg.positive_half_width, hw - 2.0)
            verge |= (d > hw + 5.0) & (d <= hw + 15.0)
        others = distance_to_roads(chip, bad) if bad else np.full(chip.shape, np.inf)
        verge &= ~pos_m & (others > ccfg.negative_min_dist) & ok
        pos = np.flatnonzero(pos_m & ok)
        neg = np.flatnonzero(far | verge)
    else:
        pos = np.flatnonzero((dist <= ccfg.positive_half_width) & ok)
        neg = np.flatnonzero(far)
    pos = rng.choice(pos, min(len(pos), n_per_class), replace=False) if len(pos) else pos
    neg = rng.choice(neg, min(len(neg), n_per_class), replace=False) if len(neg) else neg
    idx = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    rows, cols = np.unravel_index(idx, chip.shape)
    return rows, cols, labels


def fit(samples_pix: np.ndarray, samples_emb: np.ndarray | None, labels: np.ndarray,
        *, feature_mode: str, has_nir: bool, band_lo, band_hi,
        embedding_model: str | None, lidar: bool = False) -> RoadClassifier:
    """Fit a balanced logistic regression on standardised features."""
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Training found only one class. Check that roads overlap the imagery and "
            "that negative_min_dist / negative_max_dist suit your CRS units."
        )
    X = [samples_pix]
    if samples_emb is not None:
        X.append(samples_emb)
    X = np.concatenate(X, axis=1).astype(np.float64)
    mean = np.nanmean(X, axis=0)
    mean[np.isnan(mean)] = 0.0
    X = np.where(np.isnan(X), mean, X)   # pixels without LiDAR count as average
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Xs = (X - mean) / std
    coef, intercept = logistic_regression(Xs, labels, C=0.5)
    acc = float((((Xs @ coef + intercept) > 0) == (labels > 0.5)).mean())
    log.info("Classifier trained on %d samples, training accuracy %.3f", len(labels), acc)
    n_pix = samples_pix.shape[1]
    clf = RoadClassifier(
        feature_mode=feature_mode,
        has_nir=has_nir,
        band_lo=[float(v) for v in band_lo],
        band_hi=[float(v) for v in band_hi],
        pix_names=pixel_feature_names(has_nir) + (lidar_feature_names(has_nir) if lidar else []),
        pix_mean=mean[:n_pix].tolist(),
        pix_std=std[:n_pix].tolist(),
        pix_coef=coef[:n_pix].tolist(),
        intercept=float(intercept),
        train_accuracy=acc,
        lidar=lidar,
    )
    if samples_emb is not None:
        clf.embedding_model = embedding_model
        clf.emb_mean = mean[n_pix:].tolist()
        clf.emb_std = std[n_pix:].tolist()
        clf.emb_coef = coef[n_pix:].tolist()
    return clf


def logistic_regression(X: np.ndarray, y: np.ndarray, C: float = 1.0,
                        max_iter: int = 50, tol: float = 1e-8):
    """L2-regularised logistic regression with balanced class weights (Newton's method).

    Minimises ``sum(w_i * logloss_i) + |coef|^2 / (2C)``, the same objective as
    scikit-learn's ``LogisticRegression(C=C, class_weight="balanced")``.
    Returns (coef, intercept).
    """
    n, f = X.shape
    y = np.asarray(y, dtype=np.float64)
    n_pos = y.sum()
    sw = np.where(y > 0.5, n / (2 * n_pos), n / (2 * (n - n_pos)))
    A = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(f + 1)
    reg = np.full(f + 1, 1.0 / C)
    reg[-1] = 0.0  # intercept not penalised
    for _ in range(max_iter):
        z = np.clip(A @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = A.T @ (sw * (p - y)) + reg * beta
        H = (A * (sw * p * (1 - p))[:, None]).T @ A + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.max(np.abs(step)) < tol * (1 + np.max(np.abs(beta))):
            break
    return beta[:-1], float(beta[-1])


def extract_samples(chip: Chip, rows, cols, has_nir, band_lo, band_hi, embedder=None,
                    lidar: bool = False):
    """Feature vectors (n, F) at the given pixels, matching what ``predict`` computes."""
    scaled = scale_bands(chip.data, band_lo, band_hi)
    feats = all_features(chip, scaled, has_nir, lidar)
    pix = feats[:, rows, cols].T
    emb = None
    if embedder is not None:
        low = embedder(scaled[:3])
        emb = sample_lowres(low, rows, cols, embedder.patch).T
    return pix, emb


# ======================================================================
# refine.py
# ======================================================================

"""Move a road line to the middle of the pavement using cross-sections.

At regular stations along the existing line, the road-probability map is
sampled along a short line perpendicular to the road. Two methods turn those
profiles into a new line:

* ``ribbon`` (default): the road is a ribbon with a centre offset and a width
  at every station. The best ribbon for the whole line is found at once
  (dynamic programming): it should cover the "road" pixels and avoid the rest,
  while the centre bends gently and the width stays steady. A shadow that hides
  one edge for a few stations can't drag the centre over, because that would
  need a sudden change in width; cross-sections at intersections (which see the
  cross street) are ignored and bridged by the neighbouring stations. This is
  the "ribbon snake" idea from the road-conflation literature.
* ``midpoint``: the run of "road" pixels nearest the line gives the two edges
  at each station; their midpoint is the centre. Outliers are dropped, gaps
  interpolated and the offsets smoothed.

Where the imagery gives no usable answer (trees, shadows, no data) the line
keeps its original position and is flagged for review.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import ndimage


# Sampler: (x array, y array) -> probability array of the same shape, 0-1.
Sampler = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class RefineResult:
    coords: np.ndarray                # (N, 2) refined line
    shift_avg: float
    shift_max: float
    confidence: float
    pct_valid: float
    width_med: float
    reasons: list = field(default_factory=list)
    stations: dict | None = None
    moved: bool = True                # False: the original geometry was kept

    @property
    def needs_review(self) -> bool:
        return bool(self.reasons)


def station_distances(s: np.ndarray, spacing: float) -> np.ndarray:
    """Regular stations plus every original vertex (``s`` = vertex distances), so
    corners are kept."""
    length = s[-1]
    n = max(2, int(np.ceil(length / spacing)) + 1)
    d = np.unique(np.concatenate([np.linspace(0.0, length, n), s]))
    # drop stations that crowd a vertex
    keep = np.ones(len(d), bool)
    min_gap = spacing * 0.25
    for i in range(1, len(d)):
        if d[i] - d[i - 1] < min_gap and not np.isclose(d[i], length):
            keep[i] = False
    return d[keep]


def _tangents(c: np.ndarray, s: np.ndarray, d: np.ndarray, h: float) -> np.ndarray:
    t = interpolate(c, d + h, s) - interpolate(c, d - h, s)
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return t / norm


def measure_profile(offsets: np.ndarray, prob: np.ndarray, cfg: RefineConfig):
    """Find the pavement run nearest offset 0 in one cross-section profile.

    Returns (centre, width, confidence) or None when no plausible run exists.
    """
    above = prob >= cfg.prob_threshold
    if not above.any():
        return None
    labels, n = ndimage.label(above)
    step = offsets[1] - offsets[0]
    best = None
    for k in range(1, n + 1):
        idx = np.flatnonzero(labels == k)
        i0, i1 = idx[0], idx[-1]
        if i0 == 0 or i1 == len(prob) - 1:
            continue  # edge runs past the search window: width unknown
        left = _crossing(offsets, prob, i0 - 1, i0, cfg.prob_threshold)
        right = _crossing(offsets, prob, i1, i1 + 1, cfg.prob_threshold)
        width = right - left
        if width < cfg.min_road_width * 0.5:
            continue
        centre = 0.5 * (left + right)
        dist = 0.0 if left <= 0 <= right else min(abs(left), abs(right))
        if best is None or dist < best[0]:
            best = (dist, centre, width, i0, i1)
    if best is None:
        return None
    _, centre, width, i0, i1 = best
    if abs(centre) > cfg.max_shift:
        return None
    inside = prob[i0:i1 + 1].mean()
    pad = max(1, int(round(min(width, 10 * step) / step)))
    outside = np.concatenate([prob[max(0, i0 - pad):i0], prob[i1 + 1:i1 + 1 + pad]])
    contrast = float(np.clip(inside - (outside.mean() if len(outside) else 0.0), 0, 1))
    width_ok = cfg.min_road_width <= width <= cfg.max_road_width
    conf = contrast * (1.0 if width_ok else 0.4)
    return centre, width, conf


def _crossing(x, p, i, j, thr):
    """Sub-sample position where the profile crosses ``thr`` between samples i and j."""
    if p[j] == p[i]:
        return 0.5 * (x[i] + x[j])
    t = (thr - p[i]) / (p[j] - p[i])
    return x[i] + t * (x[j] - x[i])


def refine_line(coords, sample: Sampler, cfg: RefineConfig, step: float,
                valid: Sampler | None = None, keep_stations: bool = False,
                junctions: np.ndarray | None = None) -> RefineResult:
    """Refine one line given as (N, 2) coords. ``step`` is the cross-section sample
    spacing (normally the pixel size). ``junctions``: (k, 2) intersection points;
    cross-sections near them are ignored by the ribbon method."""
    c = as_coords(coords)
    s = cumlen(c)
    d = station_distances(s, cfg.station_spacing)
    pts = interpolate(c, d, s)
    tan = _tangents(c, s, d, max(cfg.station_spacing * 0.5, step))
    normal = np.column_stack([-tan[:, 1], tan[:, 0]])

    offs = np.arange(-cfg.search_half_width, cfg.search_half_width + step * 0.5, step)
    xs = pts[:, 0:1] + normal[:, 0:1] * offs[None, :]
    ys = pts[:, 1:2] + normal[:, 1:2] * offs[None, :]
    profiles = sample(xs, ys)

    n = len(d)
    reasons = []
    no_data = False
    has = np.ones(n, bool)
    if valid is not None:
        has = valid(pts[:, 0], pts[:, 1]) > 0.5
        no_data = (~has).mean() > 0.5

    if cfg.method == "ribbon":
        near_junction = np.zeros(n, bool)
        if junctions is not None and len(junctions) and cfg.junction_mask > 0:
            jd = np.min(np.hypot(pts[:, None, 0] - junctions[None, :, 0],
                                 pts[:, None, 1] - junctions[None, :, 1]), axis=1)
            near_junction = jd < cfg.junction_mask
        final, width, conf, used = fit_ribbon(offs, profiles, d, has & ~near_junction, cfg)
        raw = np.where(used, final, np.nan)
    else:
        raw = np.full(n, np.nan)
        width = np.full(n, np.nan)
        conf = np.zeros(n)
        for i in range(n):
            m = measure_profile(offs, profiles[i], cfg)
            if m is not None and m[2] > 0:
                raw[i], width[i], conf[i] = m
        raw[~has] = np.nan
        conf[~has] = 0.0
        final, used = _clean_offsets(d, raw, cfg)
        final = np.clip(final, -cfg.max_shift, cfg.max_shift)

    new_line = simplify(pts + normal * final[:, None], cfg.simplify_tolerance)

    pct_valid = float(used.mean())
    mean_conf = float(conf[used].mean() * pct_valid) if used.any() else 0.0
    shift_abs = np.abs(final)
    if no_data:
        reasons.append("no_imagery")
    if pct_valid < 0.3:
        reasons.append("few_matches")
    if mean_conf < cfg.review_min_confidence:
        reasons.append("low_confidence")
    if shift_abs.max() > cfg.review_max_shift:
        reasons.append("large_shift")

    moved = True
    if reasons and cfg.keep_flagged:
        new_line, moved = c.copy(), False

    stations = None
    if keep_stations:
        stations = dict(x=pts[:, 0], y=pts[:, 1], raw=raw, final=final, conf=conf,
                        width=width, used=used)
    return RefineResult(
        coords=new_line,
        shift_avg=float(shift_abs.mean()),
        shift_max=float(shift_abs.max()),
        confidence=round(mean_conf, 3),
        pct_valid=round(pct_valid, 3),
        width_med=float(np.nanmedian(width[used])) if used.any() else 0.0,
        reasons=reasons,
        stations=stations,
        moved=moved,
    )


def _clean_offsets(d: np.ndarray, raw: np.ndarray, cfg: RefineConfig):
    """Drop outliers, fill gaps along the road, smooth. Returns (offsets, used mask)."""
    good = ~np.isnan(raw)
    if not good.any():
        return np.zeros_like(raw), good
    w = max(1, cfg.smooth_window)
    half = w // 2
    med = np.array([
        np.nanmedian(raw[max(0, i - half):i + half + 1])
        if good[max(0, i - half):i + half + 1].any() else np.nan
        for i in range(len(raw))
    ])
    used = good & (np.abs(raw - med) <= cfg.outlier_tolerance)
    if not used.any():
        return np.zeros_like(raw), used
    filled = np.interp(d, d[used], raw[used])
    if len(filled) >= 3 and w > 1:
        sigma = max(1.0, w / 3.0)
        filled = ndimage.gaussian_filter1d(filled, sigma, mode="nearest")
    return filled, used


# --------------------------------------------------------------------------- ribbon

def _cum_at(cum: np.ndarray, edges: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Cumulative profile values (n, m+1) at positions ``pos`` (any shape), per station.

    Positions outside the search window are clamped to it, so the area beyond it
    counts as neither road nor not-road."""
    step = edges[1] - edges[0]
    m = cum.shape[1] - 1
    f = np.clip((pos - edges[0]) / step, 0, m)
    i0 = np.minimum(f.astype(int), m - 1)
    t = f - i0
    return cum[:, i0] * (1 - t) + cum[:, i0 + 1] * t


def fit_ribbon(offs: np.ndarray, profiles: np.ndarray, d: np.ndarray, data_ok: np.ndarray,
               cfg: RefineConfig):
    """Best ribbon (centre offset + width per station) for a whole line.

    profiles: (n stations, m samples) road probability across the road at ``offs``.
    data_ok: stations whose cross-section may be used (imagery present, not at an
    intersection). Returns (centre offsets, widths, per-station confidence, used mask).
    """
    n = len(d)
    step = offs[1] - offs[0]
    edges = np.concatenate([offs - step / 2, [offs[-1] + step / 2]])
    zero = np.zeros((n, 1))
    gain = np.hstack([zero, np.cumsum((profiles - 0.5) * step, axis=1)])  # + road, - not road
    cum = np.hstack([zero, np.cumsum(profiles * step, axis=1)])

    cstep = max(step, 1.0)
    reach = min(cfg.max_shift, cfg.search_half_width)
    centres = np.arange(-reach, reach + 1e-9, cstep)
    widths = np.arange(cfg.min_road_width, cfg.max_road_width + 1e-9, 2 * cstep)
    left = centres[:, None] - widths[None, :] / 2
    right = centres[:, None] + widths[None, :] / 2
    data = -(_cum_at(gain, edges, right) - _cum_at(gain, edges, left))   # (n, nc, nw)
    data[~data_ok] = 0.0

    ci, wi = _viterbi(data, d, centres, widths, cfg, None)
    ok = data_ok.copy()
    if ok.any():  # second pass: hold the width near this line's typical width
        ci, wi = _viterbi(data, d, centres, widths, cfg, float(np.median(widths[wi[ok]])))
    c, w = centres[ci], widths[wi]

    # confidence: how much more "road" the ribbon holds than a band just outside it
    lo, hi = c - w / 2, c + w / 2
    band = np.maximum(3.0, 0.25 * w)
    rows = np.arange(n)

    def cum_pts(pos):  # cumulative probability at one position per station
        mm = cum.shape[1] - 1
        f = np.clip((pos - edges[0]) / step, 0, mm)
        i0 = np.minimum(f.astype(int), mm - 1)
        t = f - i0
        return cum[rows, i0] * (1 - t) + cum[rows, i0 + 1] * t, np.clip(pos, edges[0], edges[-1])

    def mean_p(a, b):
        ca, pa = cum_pts(a)
        cb, pb = cum_pts(b)
        length = pb - pa
        return np.where(length > step, (cb - ca) / np.maximum(length, 1e-9), np.nan)

    inside = mean_p(lo, hi)
    out_l, out_r = mean_p(lo - band, lo), mean_p(hi, hi + band)
    outside = np.nanmean(np.vstack([out_l, out_r]), axis=0)
    contrast = np.clip(np.nan_to_num(inside - outside), 0, 1)
    conf = np.where(data_ok, contrast, 0.0)
    used = data_ok & (contrast >= 0.2)
    if n >= 3:
        c = ndimage.gaussian_filter1d(c.astype(float), 1.0, mode="nearest")
    return c, w.astype(float), conf, used


def _viterbi(data, d, centres, widths, cfg: RefineConfig, w0):
    """Cheapest path of (centre, width) states through the stations."""
    n, nc, nw = data.shape
    cstep = centres[1] - centres[0] if nc > 1 else 1.0
    wstep = widths[1] - widths[0] if nw > 1 else 1.0
    prior = cfg.stay_prior * (centres ** 2)[:, None] + np.zeros((1, nw))
    if w0 is not None:
        # Shadows and overhanging trees make pavement look narrower, almost never wider,
        # so narrowing below the typical width costs much more than widening (turn lanes).
        dw = widths - w0
        lam = np.where(dw < 0, cfg.width_prior, cfg.width_prior * 0.1)
        prior = prior + (lam * dw ** 2)[None, :]
    kc = max(1, int(round(3.0 / cstep)))     # at most ~3 ft sideways per station
    shifts = [(a, b) for a in range(-kc, kc + 1) for b in (-1, 0, 1)]
    base = np.array([cfg.offset_stiffness * (a * cstep) ** 2 + cfg.width_stiffness * (b * wstep) ** 2
                     for a, b in shifts])
    back = np.zeros((n, nc, nw), np.int16)
    v = data[0] + prior
    inf = np.inf
    for i in range(1, n):
        gap = max(d[i] - d[i - 1], 1e-6)
        scale = float(np.clip(cfg.station_spacing / gap, 0.2, 5.0))
        best = np.full((nc, nw), inf)
        arg = np.zeros((nc, nw), np.int16)
        for k, (a, b) in enumerate(shifts):
            # moved[c, w] = v[c - a, w - b]
            moved = np.full((nc, nw), inf)
            moved[max(a, 0):nc + min(a, 0), max(b, 0):nw + min(b, 0)] = \
                v[max(-a, 0):nc - max(a, 0), max(-b, 0):nw - max(b, 0)]
            moved += base[k] * scale
            better = moved < best
            best[better] = moved[better]
            arg[better] = k
        v = best + data[i] + prior
        back[i] = arg
    ci = np.zeros(n, int)
    wi = np.zeros(n, int)
    ci[-1], wi[-1] = np.unravel_index(np.argmin(v), v.shape)
    for i in range(n - 1, 0, -1):
        a, b = shifts[back[i, ci[i], wi[i]]]
        ci[i - 1], wi[i - 1] = ci[i] - a, wi[i] - b
    return ci, wi


# ======================================================================
# topology.py
# ======================================================================

"""Keep the road network connected after lines are moved independently.

Every input endpoint that was shared by several pieces (an intersection, or
the cut where a road crosses a grid edge) becomes one node. Each piece only
measured its own sideways correction, so the node goes where the corrected
pieces cross: the least-squares point that lies on every piece's end
direction. (For pieces that continue in a straight line, such as a grid-edge
cut, that is ill-posed, and a small pull toward the mean of the ends decides
it.) Each end is then moved to the node, and the move is blended out over
``taper`` distance so the rest of the line keeps its fitted position.
"""

import numpy as np
from scipy.spatial import cKDTree



def snap_nodes(orig_ends: np.ndarray, new_lines: list[np.ndarray], weights: np.ndarray,
               tolerance: float, taper: float = 50.0) -> list[np.ndarray]:
    """orig_ends: (n, 2, 2) original start/end xy per piece; new_lines: (N, 2) coords
    per piece. Returns adjusted coords."""
    n = len(new_lines)
    if n == 0:
        return []
    pts = orig_ends.reshape(-1, 2)
    groups = _components(len(pts), cKDTree(pts).query_pairs(tolerance))
    ends = np.array([[ln[0, :2], ln[-1, :2]] for ln in new_lines]).reshape(-1, 2)
    dirs = np.array([[_end_dir(ln, 0), _end_dir(ln, -1)] for ln in new_lines]).reshape(-1, 2)
    w = np.repeat(np.asarray(weights, float), 2) + 1e-3

    target = ends.copy()
    for g in groups:
        if len(g) >= 2:
            g = np.asarray(g)
            target[g] = _node(ends[g], dirs[g], w[g])
    delta = (target - ends).reshape(n, 2, 2)
    return [_taper_move(ln, delta[i, 0], delta[i, 1], taper) for i, ln in enumerate(new_lines)]


def _node(p: np.ndarray, d: np.ndarray, w: np.ndarray, pull: float = 0.05) -> np.ndarray:
    """Point minimising weighted squared distance to the lines (p_i, d_i), plus a weak
    pull toward the weighted mean so nearly parallel lines still give a unique answer."""
    mean = (p * w[:, None]).sum(0) / w.sum()
    A = pull * w.sum() * np.eye(2)
    b = pull * w.sum() * mean
    for pi, di, wi in zip(p, d, w):
        nrm = np.array([-di[1], di[0]])
        M = wi * np.outer(nrm, nrm)
        A += M
        b += M @ pi
    return np.linalg.solve(A, b)


def _end_dir(ln: np.ndarray, which: int) -> np.ndarray:
    c = ln[:, :2]
    v = c[1] - c[0] if which == 0 else c[-1] - c[-2]
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.array([1.0, 0.0])


def _taper_move(ln: np.ndarray, d0: np.ndarray, d1: np.ndarray, taper: float) -> np.ndarray:
    """Shift the start by d0 and the end by d1, fading each to zero over ``taper``."""
    if not (d0.any() or d1.any()):
        return ln
    c = ln[:, :2]
    s = cumlen(c)
    length = s[-1]
    t0 = min(taper, length) if d0.any() else 0.0
    t1 = min(taper, length) if d1.any() else 0.0
    # add vertices where the fades end, so the line beyond them is untouched
    extra = [x for x in (t0, length - t1) if 0 < x < length]
    if extra:
        s_new = np.unique(np.concatenate([s, extra]))
        c = np.column_stack([np.interp(s_new, s, c[:, 0]), np.interp(s_new, s, c[:, 1])])
        s = s_new
    f0 = np.clip(1 - s / t0, 0, 1) if t0 > 0 else np.zeros_like(s)
    f1 = np.clip(1 - (length - s) / t1, 0, 1) if t1 > 0 else np.zeros_like(s)
    return c + f0[:, None] * d0 + f1[:, None] * d1


def _components(n: int, pairs) -> list[list[int]]:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ======================================================================
# engine.py
# ======================================================================

"""Processing steps shared by the command-line tool and the ArcGIS Pro toolbox.

Everything here works on plain numpy coordinate arrays and an imagery
*reader*, so it runs with only numpy + scipy. A reader needs:

* ``res`` - pixel size in CRS units
* ``has_nir`` - whether the 4th band read is near-infrared
* ``read(x0, y0, width_px, height_px) -> Chip`` - block with top-left corner (x0, y0)
* ``snap(x, y) -> (x, y)`` - nearest pixel corner at or left of x / at or above y
"""

import logging

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


log = logging.getLogger("centerline_align")

PAD_PX = 64  # context pixels read around each tile so blurs/embeddings have no edge effects
TRAIN_TILES_PER_GRID = 4


class ProbabilityMap:
    """Road probability over a region, stored as uint8 to keep memory down."""

    def __init__(self, x0, y0, res, shape):
        self.x0, self.y0, self.res = x0, y0, res
        self.prob = np.zeros(shape, np.uint8)
        self.valid = np.zeros(shape, np.uint8)

    def _rc(self, x, y):
        col = (np.asarray(x) - self.x0) / self.res - 0.5
        row = (self.y0 - np.asarray(y)) / self.res - 0.5
        return row, col

    def sample(self, x, y):
        row, col = self._rc(x, y)
        v = ndimage.map_coordinates(self.prob, [row.ravel(), col.ravel()], order=1, cval=0,
                                    output=np.float32)
        return (v / 255.0).reshape(np.shape(x))

    def sample_valid(self, x, y):
        row, col = self._rc(x, y)
        v = ndimage.map_coordinates(self.valid, [row.ravel(), col.ravel()], order=0, cval=0)
        return v.reshape(np.shape(x)).astype(np.float32)


def region_probability(reader, clf: RoadClassifier, embedder, bounds, lines, rcfg, ccfg
                       ) -> ProbabilityMap:
    """Road probability for ``bounds`` plus a margin, computed only near ``lines``."""
    res = reader.res
    margin = rcfg.search_half_width + rcfg.max_shift + 4 * res
    minx, miny, maxx, maxy = bounds
    x0, y0 = reader.snap(minx - margin, maxy + margin)
    width = int(np.ceil((maxx + margin - x0) / res))
    height = int(np.ceil((y0 - (miny - margin)) / res))
    pmap = ProbabilityMap(x0, y0, res, (height, width))

    tile = ccfg.embedding_tile if embedder is not None else 1024
    core = max(64, tile - 2 * PAD_PX) if embedder is not None else tile
    reach = rcfg.search_half_width + rcfg.max_shift
    pts = np.concatenate([densify(as_coords(c), 5.0) for c in lines]) if lines else np.zeros((0, 2))
    tree = cKDTree(pts) if len(pts) else None
    for r0 in range(0, height, core):
        for c0 in range(0, width, core):
            h, w = min(core, height - r0), min(core, width - c0)
            tx0, ty0 = x0 + c0 * res, y0 - r0 * res
            centre = (tx0 + w * res / 2, ty0 - h * res / 2)
            radius = np.hypot(w, h) * res / 2 + reach + 5.0
            if tree is None or not tree.query_ball_point(centre, radius):
                continue  # nothing to refine near this tile
            chip = reader.read(tx0 - PAD_PX * res, ty0 + PAD_PX * res,
                               w + 2 * PAD_PX, h + 2 * PAD_PX)
            p = clf.predict(chip, embedder)[PAD_PX:PAD_PX + h, PAD_PX:PAD_PX + w]
            v = chip.valid[PAD_PX:PAD_PX + h, PAD_PX:PAD_PX + w]
            pmap.prob[r0:r0 + h, c0:c0 + w] = np.round(p * 255).astype(np.uint8)
            pmap.valid[r0:r0 + h, c0:c0 + w] = v
    return pmap


def pick_spread(items: list, n: int) -> list:
    """Up to ``n`` items spread evenly through the list."""
    if not items:
        return []
    idx = np.unique(np.linspace(0, len(items) - 1, min(n, len(items))).round().astype(int))
    return [items[i] for i in idx]


def alignment_check(chip, lines, has_nir: bool, max_shift: float = 60.0,
                    blur: float = 8.0):
    """How far the road lines sit from grey, non-vegetated pixels in a chip.

    Cross-correlates the road centrelines with a crude pavement index (low
    colour saturation, low NDVI), blurred by ``blur`` so the match peaks at the
    middle of the pavement rather than anywhere inside it, over shifts up to
    ``max_shift``. Returns
    (dx, dy, best_score, unshifted_score): moving the lines by (dx, dy) gives the
    best match. A large, consistent shift means roads and imagery are offset
    from each other; it is a hint, not a measurement.
    """
    if not chip.valid.any():
        return None
    mask = (distance_to_roads(chip, lines) <= chip.res).astype(np.float64)
    if mask.sum() < 50:
        return None
    v = chip.data[:, chip.valid]
    lo, hi = np.percentile(v, 2, axis=1), np.percentile(v, 98, axis=1)
    feats = pixel_features(scale_bands(chip.data, lo, hi), has_nir)
    names = pixel_feature_names(has_nir)
    idx = -feats[names.index("sat")]
    if has_nir:
        idx = idx - feats[names.index("ndvi")]
    idx = np.where(chip.valid, idx, np.nan)
    idx = (idx - np.nanmean(idx)) / (np.nanstd(idx) + 1e-9)
    idx = ndimage.gaussian_filter(np.clip(np.nan_to_num(idx), -3, 3), blur / chip.res)
    h, w = mask.shape
    # corr[s] = sum_x mask[x] * idx[x + s]  (zero-padded, so no wrap-around)
    F = np.fft.rfft2
    pm = np.zeros((2 * h, 2 * w))
    pi = np.zeros((2 * h, 2 * w))
    pm[:h, :w], pi[:h, :w] = mask, idx
    corr = np.fft.irfft2(np.conj(F(pm)) * F(pi), s=pm.shape) / mask.sum()
    k = int(max_shift / chip.res)
    rows = np.r_[np.arange(0, k + 1), np.arange(-k, 0)]
    cols = np.r_[np.arange(0, k + 1), np.arange(-k, 0)]
    sub = corr[np.ix_(rows % (2 * h), cols % (2 * w))]
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    dr, dc = rows[i], cols[j]
    return float(dc * chip.res), float(-dr * chip.res), float(sub[i, j]), float(corr[0, 0])


def train_classifier(reader, cells: list[list], lines_near, ccfg, embedder=None,
                     seed: int = 0, on_chip=None, trusted_near=None) -> RoadClassifier:
    """Train from tiles centred on random road points in each training cell.

    cells: for each training grid, the road coords inside it.
    lines_near(bounds) -> road coords intersecting ``bounds`` (for labelling).
    on_chip(k, chip, lines) is called for every training tile (for diagnostics).
    trusted_near(bounds) -> ([(coords, half_width)], [coords]): second pass, label
    from an earlier run's trusted lines (see ``sample_training_pixels``).
    """
    rng = np.random.default_rng(seed)
    tile = ccfg.embedding_tile if embedder is not None else 1024
    half = tile * reader.res / 2
    centres = []
    for lines in cells:
        lines = [as_coords(c) for c in lines if len(c) >= 2]
        lengths = np.array([cumlen(c)[-1] for c in lines])
        if not len(lines) or lengths.sum() == 0:
            continue
        for _ in range(TRAIN_TILES_PER_GRID):
            li = rng.choice(len(lines), p=lengths / lengths.sum())
            centres.append(interpolate(lines[li], rng.random() * lengths[li])[0])
    if not centres:
        raise ValueError("No grid contains any road; check that the files overlap")

    def read_at(c):
        x0, y0 = reader.snap(c[0] - half, c[1] + half)
        return reader.read(x0, y0, tile, tile)

    # Pass 1: band stretch (2nd-98th percentile) from all training tiles.
    vals = []
    for c in centres:
        chip = read_at(c)
        if chip.valid.any():
            v = chip.data[:, chip.valid]
            vals.append(v[:, rng.choice(v.shape[1], min(v.shape[1], 20000), replace=False)])
    if not vals:
        raise ValueError("The training areas have no imagery; check imagery coverage and CRS")
    allv = np.concatenate(vals, axis=1)
    lo, hi = np.percentile(allv, 2, axis=1), np.percentile(allv, 98, axis=1)

    # Pass 2: labelled samples.
    per_tile = max(50, ccfg.samples_per_class // len(centres))
    pix_s, emb_s, lab_s, shifts = [], [], [], []
    for k, c in enumerate(centres):
        chip = read_at(c)
        bx0, by0, bx1, by1 = chip.bounds
        m = ccfg.negative_max_dist
        near = lines_near((bx0 - m, by0 - m, bx1 + m, by1 + m))
        if on_chip is not None:
            on_chip(k, chip, near)
        a = alignment_check(chip, near, reader.has_nir)
        if a is not None:
            shifts.append(a)
        trusted = trusted_near(chip.bounds) if trusted_near is not None else None
        rows, cols, labels = sample_training_pixels(chip, near, ccfg, rng, per_tile, trusted)
        if len(labels) == 0:
            continue
        pix, emb = extract_samples(chip, rows, cols, reader.has_nir, lo, hi, embedder,
                                   lidar=getattr(reader, "has_lidar", False))
        pix_s.append(pix)
        lab_s.append(labels)
        if emb is not None:
            emb_s.append(emb)
    if not lab_s:
        raise ValueError("No training pixels found; check imagery coverage")
    _report_alignment(shifts)
    pix, labels = np.concatenate(pix_s), np.concatenate(lab_s)
    nb = 4 if reader.has_nir else 3
    names = ["R", "G", "B", "NIR"][:nb]
    for lab, what in ((1, "road"), (0, "not-road")):
        means = pix[labels == lab, :nb].mean(axis=0)
        log.info("Average %s training pixel (0-1 scale): %s", what,
                 ", ".join(f"{n} {m:.2f}" for n, m in zip(names, means)))
    clf = fit(pix, np.concatenate(emb_s) if emb_s else None, labels,
              feature_mode=ccfg.features, has_nir=reader.has_nir, band_lo=lo, band_hi=hi,
              embedding_model=ccfg.embedding_model if embedder is not None else None,
              lidar=getattr(reader, "has_lidar", False))
    if clf.train_accuracy is not None and clf.train_accuracy < 0.75:
        log.warning(
            "Training accuracy is only %.2f: the classifier can barely tell road from not-road "
            "pixels. Results will be poor. Most likely the road lines and the imagery don't line "
            "up (see the alignment check above), or the band numbers are wrong.",
            clf.train_accuracy)
    return clf


def _report_alignment(shifts):
    if not shifts:
        return
    a = np.array(shifts)
    dx, dy = np.median(a[:, 0]), np.median(a[:, 1])
    dist = np.hypot(a[:, 0], a[:, 1])
    log.info("Alignment check on %d training tiles: roads best match grey, non-vegetated "
             "pixels if moved by a median of %+.0f ft east, %+.0f ft north (per tile: %s)",
             len(a), dx, dy, "; ".join(f"{x:+.0f},{y:+.0f}" for x, y in a[:, :2]))
    if np.median(dist) > 15:
        log.warning(
            "The road lines look about %.0f ft away from the pavement in most training tiles. "
            "If that shift is similar everywhere, the roads and imagery are probably in "
            "different places: check both layers' coordinate systems, and that the tool read "
            "the right imagery (compare the diagnostic rasters with the original tiles).",
            float(np.median(dist)))


def find_junctions(lines, tol: float = 1.5) -> np.ndarray:
    """Intersection points of a set of lines: ends shared by 3 or more lines, or an end
    that touches the middle of another line (a T where the through road isn't split).
    Two lines simply continuing each other (a grid-edge cut) are not a junction."""
    lines = [as_coords(c) for c in lines if len(c) >= 2]
    if len(lines) < 2:
        return np.zeros((0, 2))
    dense, owner = [], []
    for k, c in enumerate(lines):
        p = densify(c, tol)
        dense.append(p)
        owner.append(np.full(len(p), k))
    dense, owner = np.concatenate(dense), np.concatenate(owner)
    tree = cKDTree(dense)
    ends = np.array([e for c in lines for e in (c[0], c[-1])])
    end_tree = cKDTree(ends)
    out = []
    for k, c in enumerate(lines):
        for e in (c[0], c[-1]):
            touching = set(owner[tree.query_ball_point(e, tol)]) - {k}
            if len(touching) >= 2:
                out.append(e)
            elif len(touching) == 1:
                # the other line touches here at its own end (continuation) or mid-way (T)
                if len(end_tree.query_ball_point(e, tol * 2)) < 2:
                    out.append(e)
    out += _crossings(lines, tol)
    if not out:
        return np.zeros((0, 2))
    out = np.array(out)
    # one point per junction
    keep = np.ones(len(out), bool)
    t = cKDTree(out)
    for i, j in sorted(t.query_pairs(tol * 2)):
        if keep[i]:
            keep[j] = False
    return out[keep]


def _crossings(lines, tol: float, seg: float = 10.0) -> list:
    """Points where two different lines cross mid-way (neither is split there)."""
    a, b, owner = [], [], []
    for k, c in enumerate(lines):
        c = densify(c, seg)
        a.append(c[:-1])
        b.append(c[1:])
        owner.append(np.full(len(c) - 1, k))
    a, b, owner = np.concatenate(a), np.concatenate(b), np.concatenate(owner)
    out = []
    for i, j in cKDTree((a + b) / 2).query_pairs(seg * 1.01):
        if owner[i] == owner[j]:
            continue
        p, r = a[i], b[i] - a[i]
        q, s_ = a[j], b[j] - a[j]
        den = r[0] * s_[1] - r[1] * s_[0]
        if abs(den) < 1e-12:
            continue
        t = ((q[0] - p[0]) * s_[1] - (q[1] - p[1]) * s_[0]) / den
        u = ((q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]) / den
        if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            x = p + t * r
            # where a line ends, the end-point rules above decide
            if min(np.hypot(*(x - e)) for k in (owner[i], owner[j])
                   for e in (lines[k][0], lines[k][-1])) > tol:
                out.append(x)
    return out


# what the keep/move model looks at for each refined line
MOVE_FEATURES = ("shift_avg", "shift_max", "conf", "pct_valid", "width_med")
MOVE_MIN_GAIN = 0.5   # a move "helps" when it brings the line at least this much closer


def move_probability(model: dict, rec: dict) -> float:
    """Chance (0-1) that applying a refined line's move brings it closer to where a
    person would put it, from the line's own QA values."""
    x = np.array([float(rec[k]) for k in MOVE_FEATURES])
    z = float(np.dot((x - np.asarray(model["mean"])) / np.asarray(model["std"]),
                     np.asarray(model["coef"])) + model["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_move_model(recs: list[dict], gains, min_rows: int = 20):
    """Learn when a move is worth applying.

    recs: refined records whose move was applied (``moved`` = 1); gains: for each,
    how much closer the move brought the line to the hand-checked line (negative =
    farther). Fits a logistic regression for "helps by at least MOVE_MIN_GAIN", then
    picks the probability threshold that maximises the total gain on these lines.
    Returns None when there are too few lines of either kind to learn from.
    """
    gains = np.asarray(gains, float)
    y = (gains >= MOVE_MIN_GAIN).astype(float)
    if len(recs) < min_rows or y.sum() < 5 or (1 - y).sum() < 5:
        return None
    X = np.array([[float(r[k]) for k in MOVE_FEATURES] for r in recs])
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-9] = 1.0
    coef, intercept = logistic_regression((X - mean) / std, y, C=1.0)
    model = dict(features=list(MOVE_FEATURES), mean=mean.tolist(), std=std.tolist(),
                 coef=coef.tolist(), intercept=float(intercept), threshold=0.5)
    p = np.array([move_probability(model, r) for r in recs])
    cands = np.unique(np.concatenate([p, [0.0, 1.01]]))
    total = [gains[p >= t].sum() for t in cands]
    model["threshold"] = float(cands[int(np.argmax(total))])
    model["rows"] = len(recs)
    model["gain_all"] = float(gains.sum())          # applying every move
    model["gain_chosen"] = float(max(total))        # applying the chosen ones
    return model


def refine_pieces(pmap: ProbabilityMap, pieces, rcfg, step: float, keep_stations=False,
                  move_model: dict | None = None):
    """Refine (src_id, coords) pieces. Returns (records, stations) as plain dicts.

    move_model: optional keep/move model (``fit_move_model``); a move it judges
    unlikely to help is not applied (the original line is kept, reason
    ``move_unlikely_to_help``)."""
    records, stations = [], []
    junctions = find_junctions([c for _, c in pieces])
    for src_id, coords in pieces:
        c = as_coords(coords)
        if len(c) < 2:
            continue
        r = refine_line(c, pmap.sample, rcfg, step=step, valid=pmap.sample_valid,
                        keep_stations=keep_stations, junctions=junctions)
        records.append(dict(
            src_id=src_id, shift_avg=round(r.shift_avg, 2), shift_max=round(r.shift_max, 2),
            conf=r.confidence, pct_valid=r.pct_valid, width_med=round(r.width_med, 1),
            review=int(r.needs_review), reason=";".join(r.reasons), moved=int(r.moved),
            o_sx=float(c[0, 0]), o_sy=float(c[0, 1]), o_ex=float(c[-1, 0]), o_ey=float(c[-1, 1]),
            coords=r.coords,
        ))
        rec = records[-1]
        if move_model is not None and rec["moved"]:
            p = move_probability(move_model, rec)
            rec["move_p"] = round(p, 3)
            if p < move_model["threshold"]:
                rec["coords"], rec["moved"] = c.copy(), 0
                rec["reason"] = ";".join(r.reasons + ["move_unlikely_to_help"])
        if r.stations is not None:
            s = r.stations
            for i in range(len(s["x"])):
                stations.append(dict(src_id=src_id, x=float(s["x"][i]), y=float(s["y"][i]),
                                     raw=float(s["raw"][i]), final=float(s["final"][i]),
                                     conf=float(s["conf"][i]), width=float(s["width"][i]),
                                     used=int(s["used"][i])))
    return records, stations


def summarize(records: list[dict]) -> list[str]:
    """Plain-language summary of a run: why lines were flagged, typical values."""
    if not records:
        return ["No road pieces were processed."]
    n = len(records)
    reasons: dict[str, int] = {}
    for r in records:
        for why in filter(None, str(r.get("reason") or "").split(";")):
            reasons[why] = reasons.get(why, 0) + 1
    med = {k: float(np.median([r[k] for r in records]))
           for k in ("conf", "pct_valid", "shift_avg", "shift_max", "width_med")}
    out = [f"Summary of {n} road pieces: median confidence {med['conf']:.2f}, "
           f"median share of cross-sections that found pavement {med['pct_valid']:.0%}, "
           f"median measured width {med['width_med']:.0f}, median average shift "
           f"{med['shift_avg']:.1f}, median largest shift {med['shift_max']:.1f}."]
    if reasons:
        out.append("Flag reasons (a piece can have several): " + ", ".join(
            f"{k} {v} ({v / n:.0%})" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
    return out


def reconnect(records: list[dict], rcfg) -> list[np.ndarray]:
    """Snapped coords for refined records (see ``topology.snap_nodes``)."""
    if not records:
        return []
    ends = np.array([[r["o_sx"], r["o_sy"], r["o_ex"], r["o_ey"]] for r in records],
                    float).reshape(-1, 2, 2)
    return snap_nodes(ends, [np.asarray(r["coords"], float) for r in records],
                      np.array([r["conf"] for r in records], float),
                      rcfg.node_snap_tolerance, rcfg.node_taper)


# ======================================================================
# evaluate.py
# ======================================================================

"""Measure aligned lines against hand-checked "truth" lines.

Truth lines are sampled every few feet. At each sample point, every candidate
layer (the original roads, a tool run, ...) is matched to its nearest line
running in roughly the same direction; the perpendicular distance is that
layer's error there. Points with no such line within ``max_dist`` count as
"not matched" (a road the candidate doesn't have, or one far off) and are kept
out of the distance statistics.

Each point is also given a cover class, so errors can be compared for open
road, shadow, trees and intersections:

* ``intersection`` - within ``junction_dist`` of a junction of the truth lines
* ``canopy`` / ``partial`` - share of vegetation pixels around the point
* ``shadow`` - share of dark pixels around the point
* ``open`` - everything else
"""

import numpy as np
from scipy.spatial import cKDTree


COVER_CLASSES = ["open", "shadow", "partial", "canopy", "intersection", "unknown"]


def sample_lines(lines, spacing: float):
    """Points every ``spacing`` along each line: (points (n, 2), unit tangents (n, 2),
    line index (n,))."""
    pts, tans, idx = [], [], []
    for k, c in enumerate(lines):
        c = as_coords(c)
        s = cumlen(c)
        if len(c) < 2 or s[-1] <= 0:
            continue
        d = np.arange(spacing / 2, s[-1], spacing)
        if not len(d):
            d = np.array([s[-1] / 2])
        h = min(spacing / 2, s[-1] / 4)
        t = interpolate(c, np.minimum(d + h, s[-1]), s) - interpolate(c, np.maximum(d - h, 0), s)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        pts.append(interpolate(c, d, s))
        tans.append(t)
        idx.append(np.full(len(d), k))
    if not pts:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int)
    return np.concatenate(pts), np.concatenate(tans), np.concatenate(idx)


class SegmentIndex:
    """Nearest same-direction segment lookups over a set of lines."""

    def __init__(self, lines, max_seg: float = 20.0):
        a, b = [], []
        for c in lines:
            c = densify(as_coords(c), max_seg)
            if len(c) >= 2:
                a.append(c[:-1])
                b.append(c[1:])
        self.a = np.concatenate(a) if a else np.zeros((0, 2))
        self.b = np.concatenate(b) if b else np.zeros((0, 2))
        self.half = max_seg / 2 + 1e-6
        self.tree = cKDTree((self.a + self.b) / 2) if len(self.a) else None

    def distances(self, pts, tans, max_dist: float = 40.0, max_angle: float = 30.0):
        """Perpendicular distance from each point to the nearest segment within
        ``max_dist`` whose direction is within ``max_angle`` of the point's; NaN if none."""
        out = np.full(len(pts), np.nan)
        if self.tree is None or not len(pts):
            return out
        cos_min = np.cos(np.radians(max_angle))
        cands = self.tree.query_ball_point(pts, max_dist + self.half)
        for i, idx in enumerate(cands):
            if not idx:
                continue
            idx = np.asarray(idx)
            a, b = self.a[idx], self.b[idx]
            ab = b - a
            ln = np.linalg.norm(ab, axis=1)
            ok = ln > 1e-9
            if not ok.any():
                continue
            a, ab, ln = a[ok], ab[ok], ln[ok]
            u = ab / ln[:, None]
            same_dir = np.abs(u @ tans[i]) >= cos_min
            if not same_dir.any():
                continue
            a, ab, ln = a[same_dir], ab[same_dir], ln[same_dir]
            t = np.clip(((pts[i] - a) * ab).sum(1) / (ln * ln), 0, 1)
            d = np.linalg.norm(a + t[:, None] * ab - pts[i], axis=1).min()
            if d <= max_dist:
                out[i] = d
        return out


def cover_from_chip(chip, pts, has_nir: bool, radius: float = 8.0,
                    veg_partial: float = 0.15, veg_canopy: float = 0.5,
                    dark_share: float = 0.3) -> np.ndarray:
    """Cover class per point from the imagery around it (raw band values)."""
    out = np.array(["unknown"] * len(pts), dtype=object)
    if not chip.valid.any():
        return out
    data = chip.data.astype(np.float32)
    r, g, b = data[0], data[1], data[2]
    if has_nir and data.shape[0] > 3:
        nir = data[3]
        veg = (nir - r) / np.maximum(nir + r, 1.0) > 0.2
    else:
        veg = (2 * g - r - b) / np.maximum(r + g + b, 1.0) > 0.05
    bright = (r + g + b) / 3.0
    v = chip.valid
    dark = bright < 0.55 * np.median(bright[v])
    rows, cols = chip.world_to_pixel(pts[:, 0], pts[:, 1])
    k = int(np.ceil(radius / chip.res))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = (yy * yy + xx * xx) * chip.res ** 2 <= radius * radius
    h, w = chip.shape
    for i, (rr, cc) in enumerate(zip(np.round(rows).astype(int), np.round(cols).astype(int))):
        r0, r1, c0, c1 = rr - k, rr + k + 1, cc - k, cc + k + 1
        if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
            continue
        m = disk & v[r0:r1, c0:c1]
        if m.sum() < disk.sum() * 0.5:
            continue
        fv = veg[r0:r1, c0:c1][m].mean()
        fd = dark[r0:r1, c0:c1][m].mean()
        out[i] = ("canopy" if fv >= veg_canopy else "partial" if fv >= veg_partial
                  else "shadow" if fd >= dark_share else "open")
    return out


def near_junctions(pts, junctions, dist: float) -> np.ndarray:
    if not len(junctions) or not len(pts):
        return np.zeros(len(pts), bool)
    d, _ = cKDTree(junctions).query(pts)
    return d < dist


def point_stats(err: np.ndarray) -> dict:
    """Summary of one candidate's errors at a set of points (NaN = not matched)."""
    n = len(err)
    e = err[~np.isnan(err)]
    if not len(e):
        return dict(n=n, matched=0.0, mean=np.nan, median=np.nan, p90=np.nan,
                    within2=np.nan, within5=np.nan)
    return dict(n=n, matched=len(e) / n if n else 0.0, mean=float(e.mean()),
                median=float(np.median(e)), p90=float(np.percentile(e, 90)),
                within2=float((e <= 2).mean()), within5=float((e <= 5).mean()))


def eval_report(errors: dict, cover: np.ndarray, units: str = "ft") -> list[str]:
    """Plain-text tables: per cover class, every candidate's error statistics.

    ``errors``: {candidate name: per-point errors}. The comparison uses only points
    that every candidate matched, so the layers are measured on the same roads."""
    names = list(errors)
    if not names:
        return ["Nothing to compare."]
    allm = np.all([~np.isnan(errors[k]) for k in names], axis=0)
    lines = [f"{len(cover)} check points; {allm.sum()} matched by every layer "
             f"(the comparison below uses only those)."]
    for k in names:
        lines.append(f"  {k}: matched {np.mean(~np.isnan(errors[k])):.0%} of check points")
    width = max(len(k) for k in names)
    head = (f"  {'layer':<{width}}  {'points':>6}  {'mean':>6}  {'median':>6}  {'worst10%':>8}"
            f"  {'<=2' + units:>6}  {'<=5' + units:>6}")
    for cls in ["all"] + COVER_CLASSES:
        sel = allm if cls == "all" else allm & (cover == cls)
        if not sel.any():
            continue
        lines.append("")
        lines.append(f"{cls.upper()} ({sel.sum()} points)")
        lines.append(head)
        for k in names:
            s = point_stats(errors[k][sel])
            lines.append(f"  {k:<{width}}  {s['n']:>6}  {s['mean']:>6.2f}  {s['median']:>6.2f}  "
                         f"{s['p90']:>8.2f}  {s['within2']:>6.0%}  {s['within5']:>6.0%}")
    return lines


# ======================================================================
# chips.py
# ======================================================================

"""Training chips for a deep-learning centreline model.

Each chip is an imagery block (bands, H, W) around a hand-checked road, with:

* ``heat`` - the target: 1 on the checked centreline, falling off as a Gaussian
  (``sigma`` map units) across it. The model learns to draw this ridge; the
  ridge's crest is the centreline.
* ``mask`` - where the loss is counted: near the checked road only, and not near
  roads that haven't been checked (their true centre is unknown, so the model
  must not be told "no road here").

Training and validation are split by square blocks of ground, not by road: roads
join, so one road's chip shows its neighbours. Training chips have the loss
switched off inside validation blocks, and validation chips are scored only
inside them, so the validation score is on pavement the model has not seen.

Line ends are left out of the mask (``end_trim``): lines are split at junctions
and bent through them (REVAMP manual), so near an end a line is not a pavement
centre.
"""

import numpy as np



def pixel_centres(x0: float, y0: float, res: float, h: int, w: int):
    xs = x0 + (np.arange(w) + 0.5) * res
    ys = y0 - (np.arange(h) + 0.5) * res
    return np.meshgrid(xs, ys)


def distance_to_lines(x0, y0, res, shape, lines, max_dist=np.inf) -> np.ndarray:
    """Exact distance (map units) from each pixel centre to the nearest segment."""
    h, w = shape
    X, Y = pixel_centres(x0, y0, res, h, w)
    out = np.full((h, w), np.inf)
    bx0, by1 = x0, y0
    bx1, by0 = x0 + w * res, y0 - h * res
    for c in lines:
        c = as_coords(c)
        if len(c) < 2:
            continue
        a, b = c[:-1], c[1:]
        # segments that can come within max_dist of the chip
        lo = np.minimum(a, b) - max_dist
        hi = np.maximum(a, b) + max_dist
        keep = (hi[:, 0] >= bx0) & (lo[:, 0] <= bx1) & (hi[:, 1] >= by0) & (lo[:, 1] <= by1)
        for (ax, ay), (qx, qy) in zip(a[keep], b[keep]):
            dx, dy = qx - ax, qy - ay
            ll = dx * dx + dy * dy
            if ll == 0:
                d = np.hypot(X - ax, Y - ay)
            else:
                t = np.clip(((X - ax) * dx + (Y - ay) * dy) / ll, 0, 1)
                d = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
            np.minimum(out, d, out=out)
    return out


def make_labels(x0, y0, res, shape, truth, unchecked, sigma=1.0, corridor=30.0,
                unknown_radius=15.0, ends=None, end_trim=0.0, keep=None):
    """(heat uint8 0-255, mask uint8 0/1) for one chip. ``ends``: line end points
    whose surroundings (``end_trim``) are left out; ``keep``: bool (h, w), where
    the mask may be on at all."""
    d_true = distance_to_lines(x0, y0, res, shape, truth, corridor + 3 * sigma)
    heat = np.exp(-0.5 * (d_true / sigma) ** 2)
    mask = d_true <= corridor
    if unchecked:
        d_other = distance_to_lines(x0, y0, res, shape, unchecked, unknown_radius)
        mask &= ~((d_other <= unknown_radius) & (d_true > 3 * sigma))
    if ends is not None and len(ends) and end_trim > 0:
        X, Y = pixel_centres(x0, y0, res, *shape)
        for ex, ey in np.asarray(ends, float):
            mask &= (X - ex) ** 2 + (Y - ey) ** 2 > end_trim ** 2
    if keep is not None:
        mask &= keep
    return np.round(heat * 255).astype(np.uint8), mask.astype(np.uint8)


def block_of(x, y, block: float):
    return f"{int(np.floor(x / block))}_{int(np.floor(y / block))}"


def in_validation(x, y, block: float, fraction: float, seed: int = 0) -> bool:
    return is_validation(block_of(x, y, block), fraction, seed)


def validation_pixels(x0, y0, res, shape, block: float, fraction: float, seed: int = 0):
    """bool (h, w): pixels inside validation blocks."""
    h, w = shape
    xs = x0 + (np.arange(w) + 0.5) * res
    ys = y0 - (np.arange(h) + 0.5) * res
    bx, by = np.floor(xs / block).astype(int), np.floor(ys / block).astype(int)
    vx, vy = np.unique(bx), np.unique(by)
    table = np.array([[is_validation(f"{i}_{j}", fraction, seed) for i in vx] for j in vy])
    return table[np.searchsorted(vy, by)][:, np.searchsorted(vx, bx)]


def is_validation(road_id, fraction: float, seed: int = 0) -> bool:
    """Deterministic train/validation split by road id."""
    h = (hash_id(road_id) + seed * 7919) % 10007
    return h < fraction * 10007


def hash_id(road_id) -> int:
    s = str(road_id)
    v = 2166136261
    for ch in s.encode("utf-8"):
        v = ((v ^ ch) * 16777619) % (2 ** 32)
    return v


def lines_in_pixels(x0, y0, res, lines, pad: float = 0.0):
    """Line coords converted to chip pixel coords (col, row) - for evaluation."""
    out = []
    for c in lines:
        c = densify(as_coords(c), 2 * res)
        col = (c[:, 0] - x0) / res - 0.5
        row = (y0 - c[:, 1]) / res - 0.5
        out.append(np.column_stack([col, row]).round(2).tolist())
    return out


# ======================================================================
# arcgis_impl.py
# ======================================================================

"""ArcGIS Pro side: read imagery / roads / grids with arcpy, write the output.

Uses only what ships with ArcGIS Pro's default Python (arcpy, numpy, scipy),
so nothing has to be installed. Works with a Basic licence: a folder of tiles
is read tile by tile with ``RasterToNumPyArray``; no mosaic dataset is needed.
"""

import json
import logging
import os
import re
import time

import numpy as np
from scipy import ndimage


TILE_EXTS = (".tif", ".tiff", ".img", ".jp2", ".sid")


class ArcTileReader:
    """Reads blocks from one raster or a folder of same-grid tiles."""

    def __init__(self, arcpy, source: str, bands: list[int], nodata=0):
        self.arcpy = arcpy
        self.nodata = nodata
        paths = self._find(source)
        self.tiles = []
        self.skipped = []  # (path, reason): unreadable tiles, or ones that don't match the rest
        for p in paths:
            try:
                r = arcpy.Raster(p)
                e = r.extent
                t = dict(path=p, xmin=e.XMin, ymin=e.YMin, xmax=e.XMax, ymax=e.YMax,
                         res=float(r.meanCellWidth), count=int(r.bandCount),
                         sr=r.spatialReference)
            except Exception as err:
                msg = str(err).strip().splitlines()
                self.skipped.append((p, msg[0] if msg else "could not be read"))
                continue
            if self.tiles:
                ref = self.tiles[0]
                if abs(t["res"] - ref["res"]) > 1e-6 * ref["res"]:
                    self.skipped.append((p, f"cell size {t['res']}, expected {ref['res']}"))
                    continue
                if t["sr"] is None or t["sr"].name != ref["sr"].name:
                    name = t["sr"].name if t["sr"] is not None else "no coordinate system"
                    self.skipped.append((p, f"in {name}, expected {ref['sr'].name}"))
                    continue
            self.tiles.append(t)
        if not self.tiles:
            raise ValueError(f"None of the imagery in {source} could be read" + (
                f" (first problem: {self.skipped[0][0]}: {self.skipped[0][1]})"
                if self.skipped else ""))
        first = self.tiles[0]
        self.sr = first["sr"]
        self.res = first["res"]
        if self.sr is None or self.sr.type != "Projected":
            raise ValueError("The imagery must be in a projected coordinate system "
                             "(e.g. State Plane), not geographic.")
        self.bands = list(bands)
        if max(self.bands) > first["count"]:
            raise ValueError(f"Band {max(self.bands)} requested but the imagery has "
                             f"{first['count']} band(s)")
        self.has_nir = len(self.bands) == 4
        self.gx, self.gy = first["xmin"], first["ymax"]
        self._ext = np.array([[t["xmin"], t["ymin"], t["xmax"], t["ymax"]] for t in self.tiles])

    def _find(self, source: str) -> list[str]:
        # .gdb rasters and Cloud Raster Format (.crf) are folders on disk but one raster
        if os.path.isdir(source) and not source.lower().rstrip("\\/").endswith((".gdb", ".crf")):
            found = []
            for root, _, files in os.walk(source):
                found += [os.path.join(root, f) for f in files if f.lower().endswith(TILE_EXTS)]
            if not found:
                raise ValueError(f"No .tif/.img/.jp2 tiles found in {source}")
            return sorted(found)
        return [source]

    def coverage(self, ext) -> float:
        """Share (0-1) of a rectangle (xmin, ymin, xmax, ymax) covered by tile extents."""
        x0, y0, x1, y1 = ext
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            return 0.0
        e = self._ext
        w = np.clip(np.minimum(e[:, 2], x1) - np.maximum(e[:, 0], x0), 0, None)
        h = np.clip(np.minimum(e[:, 3], y1) - np.maximum(e[:, 1], y0), 0, None)
        return float(min(1.0, (w * h).sum() / area))  # tiles may overlap a little

    @property
    def extent(self):
        return (self._ext[:, 0].min(), self._ext[:, 1].min(),
                self._ext[:, 2].max(), self._ext[:, 3].max())

    def snap(self, x: float, y: float) -> tuple[float, float]:
        r = self.res
        return (self.gx + np.floor((x - self.gx) / r) * r,
                self.gy - np.floor((self.gy - y) / r) * r)

    def read(self, x0: float, y0: float, width_px: int, height_px: int) -> Chip:
        r = self.res
        x1, y1 = x0 + width_px * r, y0 - height_px * r
        data = np.zeros((len(self.bands), height_px, width_px), np.float32)
        valid = np.zeros((height_px, width_px), bool)
        e = self._ext
        hit = np.flatnonzero((e[:, 0] < x1) & (e[:, 2] > x0) & (e[:, 1] < y0) & (e[:, 3] > y1))
        for i in hit:
            t = self.tiles[i]
            left, right = max(x0, t["xmin"]), min(x1, t["xmax"])
            top, bottom = min(y0, t["ymax"]), max(y1, t["ymin"])
            c0, c1 = int(round((left - x0) / r)), int(round((right - x0) / r))
            r0, r1 = int(round((y0 - top) / r)), int(round((y0 - bottom) / r))
            nc, nr = c1 - c0, r1 - r0
            if nc <= 0 or nr <= 0:
                continue
            # a point a quarter-cell inside the block's lower-left cell, so the
            # read starts on that cell however the corner is snapped
            corner = self.arcpy.Point(x0 + c0 * r + r / 4, y0 - r1 * r + r / 4)
            arr = self.arcpy.RasterToNumPyArray(t["path"], corner, nc, nr, self.nodata)
            arr = arr[None] if arr.ndim == 2 else arr
            block = arr[[b - 1 for b in self.bands]].astype(np.float32)
            ok = (block.any(axis=0) if self.nodata == 0
                  else (block != self.nodata).all(axis=0))
            # overlapping tiles: a tile's empty edge never overwrites another's data
            data[:, r0:r1, c0:c1][:, ok] = block[:, ok]
            valid[r0:r1, c0:c1] |= ok
        return Chip(data=data, valid=valid, x0=x0, y0=y0, res=r)


LIDAR_LAYERS = ("intensity", "dtm", "dsm")
LIDAR_NODATA = -9999.0


class LidarLayers:
    """The rasters made by make_lidar_layers.py: ground intensity, ground elevation
    (dtm) and surface elevation (dsm), as tiles in subfolders of one folder."""

    def __init__(self, arcpy, folder: str, sr):
        meta_path = os.path.join(folder, "lidar_layers.json")
        if not os.path.exists(meta_path):
            raise ValueError(f"{folder} is not a LiDAR layers folder (no lidar_layers.json); "
                             "make it with make_lidar_layers.py")
        with open(meta_path) as f:
            meta = json.load(f)
        self.z_to_m = float(meta.get("z_to_m", 1.0))
        self.xy_to_m = float(meta.get("xy_to_m", 0.3048006096))
        self.readers = []
        for name in LIDAR_LAYERS:
            sub = os.path.join(folder, name)
            if not os.path.isdir(sub):
                raise ValueError(f"LiDAR folder {folder} has no '{name}' subfolder")
            r = ArcTileReader(arcpy, sub, [1], nodata=LIDAR_NODATA)
            if r.sr.name != sr.name:
                raise ValueError(f"LiDAR layers are in {r.sr.name} but the imagery is in "
                                 f"{sr.name}; remake them with the imagery's coordinate system")
            self.readers.append(r)

    def sample(self, chip):
        """(3, H, W) LiDAR values at the chip's pixel centres (heights in metres), and
        where all three exist."""
        bx0, by0, bx1, by1 = chip.bounds
        out = np.zeros((3,) + chip.shape, np.float32)
        ok = np.ones(chip.shape, bool)
        h, w = chip.shape
        for k, rd in enumerate(self.readers):
            r = rd.res
            x0, y0 = rd.snap(bx0 - 2 * r, by1 + 2 * r)
            nw = int(np.ceil((bx1 + 2 * r - x0) / r))
            nh = int(np.ceil((y0 - (by0 - 2 * r)) / r))
            blk = rd.read(x0, y0, nw, nh)
            cols = (chip.x0 + (np.arange(w) + 0.5) * chip.res - x0) / r - 0.5
            rows = (y0 - (chip.y0 - (np.arange(h) + 0.5) * chip.res)) / r - 0.5
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            v = ndimage.map_coordinates(blk.data[0], [rr, cc], order=1, mode="nearest")
            good = ndimage.map_coordinates(blk.valid.astype(np.float32), [rr, cc], order=1,
                                           mode="constant", cval=0.0) > 0.99
            out[k] = v * (self.z_to_m if k > 0 else 1.0)
            ok &= good
        return out, ok


class WithLidar:
    """An imagery reader whose chips also carry the LiDAR layers."""

    has_lidar = True

    def __init__(self, reader, lidar: LidarLayers):
        self.base, self.lidar = reader, lidar

    def __getattr__(self, name):
        return getattr(self.base, name)

    def read(self, x0, y0, width_px, height_px):
        chip = self.base.read(x0, y0, width_px, height_px)
        chip.aux, chip.aux_valid = self.lidar.sample(chip)
        chip.xy_to_m = self.lidar.xy_to_m
        return chip


# --------------------------------------------------------------------------- vectors

def read_lines(arcpy, fc, id_field, sr):
    """[(id, Polyline geometry, [(N,2) coords per part], (xmin, ymin, xmax, ymax))]"""
    out = []
    with arcpy.da.SearchCursor(fc, [id_field, "SHAPE@"], spatial_reference=sr) as cur:
        for rid, geom in cur:
            if geom is None:
                continue
            parts = _parts(geom)
            if parts:
                e = geom.extent
                out.append((rid, geom, parts, (e.XMin, e.YMin, e.XMax, e.YMax)))
    return out


def read_grids(arcpy, fc, id_field, sr):
    out = []
    with arcpy.da.SearchCursor(fc, [id_field, "SHAPE@"], spatial_reference=sr) as cur:
        for gid, geom in cur:
            if geom is not None:
                e = geom.extent
                out.append((gid, geom, (e.XMin, e.YMin, e.XMax, e.YMax)))
    return out


def _parts(geom) -> list[np.ndarray]:
    parts = []
    for part in geom:
        pts = [(p.X, p.Y) for p in part if p is not None]
        if len(pts) >= 2:
            parts.append(np.array(pts, float))
    return parts


def _overlaps(a, b) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def clip_to_grid(arcpy, roads, grids, gi, sr) -> list[tuple]:
    """(src_id, coords) pieces of road inside grid ``gi``; a piece on a shared grid
    edge belongs to the lowest-index grid so it is processed once."""
    _, cell, cext = grids[gi]
    out = []
    for rid, geom, _, rext in roads:
        if not _overlaps(rext, cext):
            continue
        piece = geom.intersect(cell, 2)
        if piece is None:
            continue
        for c in _parts(piece):
            s = cumlen(c)
            if s[-1] < 1e-6:
                continue
            mid = interpolate(c, s[-1] / 2, s)[0]
            pt = arcpy.PointGeometry(arcpy.Point(mid[0], mid[1]), sr)
            owner = min((k for k, (_, g, e) in enumerate(grids)
                         if e[0] <= mid[0] <= e[2] and e[1] <= mid[1] <= e[3]
                         and not g.disjoint(pt)), default=gi)
            if owner == gi:
                out.append((rid, c))
    return out


# --------------------------------------------------------------------------- run

class _ArcHandler(logging.Handler):
    def __init__(self, arcpy):
        super().__init__()
        self.arcpy = arcpy

    def emit(self, record):
        msg = self.format(record)
        if record.levelno >= logging.WARNING:
            self.arcpy.AddWarning(msg)
        else:
            self.arcpy.AddMessage(msg)


def _safe(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def run(arcpy, *, imagery, roads_fc, road_id_field, grids_fc, grid_id_field, work_folder,
        out_fc, bands, refine: RefineConfig, classifier: ClassifierConfig, fresh=False,
        diagnostics=False, retrain_from=None, lidar_folder=None, classifier_file=None):
    msg = arcpy.AddMessage
    handler = _ArcHandler(arcpy)
    root = logging.getLogger("centerline_align")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        return _run(arcpy, msg, imagery, roads_fc, road_id_field, grids_fc, grid_id_field,
                    work_folder, out_fc, bands, refine, classifier, fresh, diagnostics,
                    retrain_from, lidar_folder, classifier_file)
    finally:
        root.removeHandler(handler)


def _run(arcpy, msg, imagery, roads_fc, road_id_field, grids_fc, grid_id_field, work_folder,
         out_fc, bands, rcfg, ccfg, fresh, diagnostics=False, retrain_from=None,
         lidar_folder=None, classifier_file=None):
    os.makedirs(work_folder, exist_ok=True)
    grid_dir = os.path.join(work_folder, "grids")
    os.makedirs(grid_dir, exist_ok=True)

    reader = ArcTileReader(arcpy, imagery, bands)
    sr = reader.sr
    msg(f"Imagery: {len(reader.tiles)} raster(s), {sr.name}, cell size {reader.res:g} "
        f"{sr.linearUnitName}. All distance settings are in {sr.linearUnitName}.")
    msg(f"Bands used: {reader.bands} (NIR: {'yes' if reader.has_nir else 'no'})")
    if reader.skipped:
        arcpy.AddWarning(f"{len(reader.skipped)} imagery file(s) were skipped:\n" + "\n".join(
            f"  {p}: {why}" for p, why in reader.skipped[:15])
            + (f"\n  ... and {len(reader.skipped) - 15} more" if len(reader.skipped) > 15 else ""))

    if lidar_folder:
        lidar = LidarLayers(arcpy, lidar_folder, sr)
        reader = WithLidar(reader, lidar)
        counts = ", ".join(f"{len(r.tiles)} {n}" for n, r in zip(LIDAR_LAYERS, lidar.readers))
        msg(f"LiDAR layers: {lidar_folder} (tiles: {counts})")

    roads = read_lines(arcpy, roads_fc, road_id_field, sr)
    grids = read_grids(arcpy, grids_fc, grid_id_field, sr)
    if not roads or not grids:
        raise ValueError("No roads or no grids to process (check selections / definition queries)")
    msg(f"{len(roads)} roads, {len(grids)} grids (only selected features are used)")
    gaps = [(gid, reader.coverage(ext)) for gid, _, ext in grids]
    gaps = [(gid, f) for gid, f in gaps if f < 0.95]
    if gaps:
        arcpy.AddWarning(
            f"{len(gaps)} of {len(grids)} grids are not fully covered by the imagery tiles; "
            "roads there will be flagged no_imagery. Grid (covered %): " + ", ".join(
                f"{g} ({f:.0%})" for g, f in gaps[:60]) + (" ..." if len(gaps) > 60 else "")
            + ". Check the tile copy (copy_aoi_tiles.py reports which grids it couldn't cover).")

    model_path = os.path.join(work_folder, "road_classifier.json")
    if fresh:
        for f in os.listdir(grid_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(grid_dir, f))
        if os.path.exists(model_path):
            os.remove(model_path)
    if classifier_file:
        clf = RoadClassifier.load(classifier_file)
        msg(f"Using the trained classifier {classifier_file}"
            + (f" (trained from {clf.trained_from})" if clf.trained_from else ""))
        if clf.has_nir != reader.has_nir:
            raise ValueError("The trained classifier was made with "
                             f"{'4' if clf.has_nir else '3'} bands; set the same bands here.")
        if clf.lidar and not lidar_folder:
            raise ValueError("The trained classifier uses LiDAR: give the LiDAR folder.")
        if clf.move_model:
            msg("It includes a keep/move model: moves it judges unlikely to help keep the "
                "original line (REASON move_unlikely_to_help; MOVE_P = its estimate).")
    elif os.path.exists(model_path):
        clf = RoadClassifier.load(model_path)
        msg(f"Using saved classifier {model_path}")
        if clf.lidar and not lidar_folder:
            raise ValueError("The saved classifier in this work folder uses LiDAR: give the "
                             "LiDAR folder, or use a new work folder.")
        if lidar_folder and not clf.lidar:
            arcpy.AddWarning("The saved classifier in this work folder was trained without "
                             "LiDAR, so the LiDAR layers are not used. Use a new work folder "
                             "to train with them.")
        if retrain_from and clf.trained_from != str(retrain_from):
            arcpy.AddWarning(
                "This work folder already has a classifier, so the second-pass setting is "
                "ignored. For a second pass, use a new, empty work folder (or tick Start over).")
    elif retrain_from:
        clf = _train_second_pass(arcpy, msg, reader, grids, retrain_from, ccfg, sr)
        clf.save(model_path)
        msg(f"Classifier saved to {model_path} (training accuracy {clf.train_accuracy:.3f})")
    else:
        with_roads = [gi for gi in range(len(grids))
                      if any(_overlaps(r[3], grids[gi][2]) for r in roads)]
        train = pick_spread(with_roads, ccfg.training_grids)
        msg("Training the road classifier on grid(s): "
            + ", ".join(str(grids[g][0]) for g in train))
        cells = [[c for _, c in clip_to_grid(arcpy, roads, grids, gi, sr)] for gi in train]

        def lines_near(b):
            return [c for _, _, parts, e in roads if _overlaps(e, b) for c in parts]

        diag_dir = os.path.join(work_folder, "diagnostics")

        def on_chip(k, chip, lines):
            # first training tile of each of the first two grids
            if k % 4 == 0 and k < 8:
                os.makedirs(diag_dir, exist_ok=True)
                _save_training_diag(arcpy, diag_dir, k // 4 + 1, chip, lines, ccfg, sr)

        clf = train_classifier(reader, cells, lines_near, ccfg, None, on_chip=on_chip)
        msg(f"Diagnostic rasters for the first training tiles: {diag_dir}")
        clf.save(model_path)
        msg(f"Classifier saved to {model_path} (training accuracy {clf.train_accuracy:.3f})")

    arcpy.SetProgressor("step", "Aligning centerlines...", 0, len(grids), 1)
    files = []
    for k, (gid, cell, cext) in enumerate(grids):
        arcpy.SetProgressorLabel(f"Grid {gid} ({k + 1} of {len(grids)})")
        path = os.path.join(grid_dir, f"{_safe(gid)}.json")
        files.append(path)
        if os.path.exists(path):
            msg(f"[{k + 1}/{len(grids)}] grid {gid}: already done, skipping")
            arcpy.SetProgressorPosition()
            continue
        t0 = time.time()
        try:
            pieces = clip_to_grid(arcpy, roads, grids, k, sr)
            records = []
            if pieces:
                pmap = region_probability(reader, clf, None, cext, [c for _, c in pieces],
                                          rcfg, ccfg)
                if diagnostics:
                    prob_dir = os.path.join(work_folder, "diagnostics")
                    os.makedirs(prob_dir, exist_ok=True)
                    h = pmap.prob.shape[0]
                    _save_raster(arcpy, pmap.prob, pmap.x0, pmap.y0 - h * pmap.res, pmap.res,
                                 os.path.join(prob_dir, f"road_prob_{_safe(gid)}.tif"), sr)
                records, _ = refine_pieces(pmap, pieces, rcfg, reader.res,
                                           move_model=clf.move_model)
            for r in records:
                r["grid_id"] = gid
                r["coords"] = np.asarray(r["coords"]).tolist()
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(records, f, default=_json_default)
            os.replace(tmp, path)
            n_rev = sum(r["review"] for r in records)
            msg(f"[{k + 1}/{len(grids)}] grid {gid}: {len(records)} pieces, {n_rev} flagged "
                f"for review ({time.time() - t0:.0f}s)")
        except Exception as e:  # keep going; one bad grid shouldn't stop an overnight run
            arcpy.AddWarning(f"[{k + 1}/{len(grids)}] grid {gid} failed: {e}")
            files.remove(path)
        arcpy.SetProgressorPosition()
    arcpy.ResetProgressor()

    records = []
    for p in files:
        with open(p) as f:
            records += json.load(f)
    coords = reconnect(records, rcfg)
    _write_output(arcpy, out_fc, records, coords, sr, roads_fc, road_id_field)
    n_rev = sum(r["review"] for r in records)
    msg(f"Wrote {len(records)} lines to {out_fc}; {n_rev} flagged for review (REVIEW = 1).")
    for line in summarize(records):
        msg(line)
    return {"lines": len(records), "review": n_rev}


def _train_second_pass(arcpy, msg, reader, grids, earlier_fc, ccfg, sr):
    """Train from an earlier run's output: lines it was confident about (REVIEW = 0)
    give road labels across their measured width; the rest only mark where roads are."""
    good, bad = [], []
    with arcpy.da.SearchCursor(earlier_fc, ["SHAPE@", "REVIEW", "WIDTH_MED"],
                               spatial_reference=sr) as cur:
        for g, review, width in cur:
            if g is None:
                continue
            for c in _parts(g):
                ext = (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())
                if not review and width and width > 0:
                    good.append((c, float(width) / 2, ext))
                else:
                    bad.append((c, ext))
    if not good:
        raise ValueError(f"{earlier_fc} has no lines with REVIEW = 0 to learn from")
    msg(f"Second pass: learning from {len(good)} trusted lines (REVIEW = 0) of {earlier_fc}; "
        f"{len(bad)} flagged lines are only used to keep road pixels out of the not-road labels")
    with_good = [gi for gi, (_, _, ge) in enumerate(grids)
                 if any(_overlaps(e, ge) for _, _, e in good)]
    train = pick_spread(with_good, ccfg.training_grids)
    msg("Training the road classifier on grid(s): " + ", ".join(str(grids[g][0]) for g in train))
    cells = [[c for c, _, e in good if _overlaps(e, grids[gi][2])] for gi in train]

    def lines_near(b):
        return ([c for c, _, e in good if _overlaps(e, b)]
                + [c for c, e in bad if _overlaps(e, b)])

    def trusted_near(b):
        return ([(c, hw) for c, hw, e in good if _overlaps(e, b)],
                [c for c, e in bad if _overlaps(e, b)])

    clf = train_classifier(reader, cells, lines_near, ccfg, None, trusted_near=trusted_near)
    clf.trained_from = str(earlier_fc)
    return clf


def _save_raster(arcpy, arr, x_ll, y_ll, res, path, sr):
    """Write a (rows, cols) or (bands, rows, cols) uint8 array as a GeoTIFF."""
    if arcpy.Exists(path):
        arcpy.management.Delete(path)
    ras = arcpy.NumPyArrayToRaster(np.ascontiguousarray(arr), arcpy.Point(x_ll, y_ll), res, res)
    ras.save(path)
    arcpy.management.DefineProjection(path, sr)


def _save_training_diag(arcpy, folder, n, chip, lines, ccfg, sr):
    """The imagery block the tool read, and where it put its training labels.

    labels_N.tif: 1 = labelled road (near a road line), 2 = labelled not-road,
    0 = not used. Lay both over the original tiles and your roads: if the
    image doesn't match the tiles underneath, the tool misread the imagery; if
    the 1-zone isn't on pavement, the roads and imagery don't line up.
    """
    h = chip.shape[0]
    x_ll, y_ll = chip.x0, chip.y0 - h * chip.res
    img = np.clip(chip.data, 0, 255).astype(np.uint8)
    _save_raster(arcpy, img, x_ll, y_ll, chip.res, os.path.join(folder, f"chip_{n}.tif"), sr)
    dist = distance_to_roads(chip, lines)
    lab = np.zeros(chip.shape, np.uint8)
    lab[dist <= ccfg.positive_half_width] = 1
    lab[(dist > ccfg.negative_min_dist) & (dist <= ccfg.negative_max_dist)] = 2
    lab[~chip.valid] = 0
    _save_raster(arcpy, lab, x_ll, y_ll, chip.res, os.path.join(folder, f"labels_{n}.tif"), sr)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


def _id_field_type(arcpy, fc, field):
    for f in arcpy.ListFields(fc):
        if f.name.lower() == field.lower():
            if f.type in ("Integer", "SmallInteger", "OID", "BigInteger"):
                return "LONG"
            if f.type in ("Double", "Single"):
                return "DOUBLE"
    return "TEXT"


OUT_FIELDS = [("SRC_ID", None), ("GRID_ID", "TEXT"), ("SHIFT_AVG", "DOUBLE"),
              ("SHIFT_MAX", "DOUBLE"), ("CONF", "DOUBLE"), ("PCT_VALID", "DOUBLE"),
              ("WIDTH_MED", "DOUBLE"), ("REVIEW", "SHORT"), ("REASON", "TEXT"),
              ("MOVED", "SHORT"), ("MOVE_P", "DOUBLE")]


def _write_output(arcpy, out_fc, records, coords, sr, roads_fc, road_id_field):
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    folder, name = os.path.split(out_fc)
    arcpy.management.CreateFeatureclass(folder, name, "POLYLINE", spatial_reference=sr)
    id_type = _id_field_type(arcpy, roads_fc, road_id_field)
    for fname, ftype in OUT_FIELDS:
        ftype = ftype or id_type
        if ftype == "TEXT":
            arcpy.management.AddField(out_fc, fname, ftype, field_length=100)
        else:
            arcpy.management.AddField(out_fc, fname, ftype)
    names = [f for f, _ in OUT_FIELDS] + ["SHAPE@"]
    with arcpy.da.InsertCursor(out_fc, names) as cur:
        for r, c in zip(records, coords):
            line = arcpy.Polyline(arcpy.Array([arcpy.Point(float(x), float(y)) for x, y in c]), sr)
            src = r["src_id"] if id_type != "TEXT" else str(r["src_id"])
            cur.insertRow([src, str(r["grid_id"]), r["shift_avg"], r["shift_max"], r["conf"],
                           r["pct_valid"], r["width_med"], int(r["review"]), r["reason"],
                           int(r.get("moved", 1)), r.get("move_p"), line])


# --------------------------------------------------------------------------- evaluation

def evaluate(arcpy, *, truth_fc, truth_where, candidates, imagery=None, bands=(1, 2, 3, 4),
             grids_fc=None, out_points=None, spacing=5.0, max_dist=40.0, junction_dist=40.0):
    """Compare candidate line layers with checked truth lines; see ``evaluate.py``.

    candidates: [(display name, layer or path)]. Nothing is written to any input;
    ``out_points`` (optional) gets one point per check point with each layer's error.
    """
    msg = arcpy.AddMessage
    reader = ArcTileReader(arcpy, imagery, list(bands)) if imagery else None
    sr = reader.sr if reader is not None else arcpy.Describe(candidates[0][1]).spatialReference
    units = sr.linearUnitName or "units"

    every = []          # all truth-layer lines (for finding intersections)
    with arcpy.da.SearchCursor(truth_fc, ["SHAPE@"], spatial_reference=sr) as cur:
        for (g,) in cur:
            if g is not None:
                every += _parts(g)
    truth, oids = [], []
    try:
        with arcpy.da.SearchCursor(truth_fc, ["OID@", "SHAPE@"], where_clause=truth_where or None,
                                   spatial_reference=sr) as cur:
            for oid, g in cur:
                if g is not None:
                    for c in _parts(g):
                        truth.append(c)
                        oids.append(oid)
    except Exception as e:
        raise ValueError(f"Could not read the checked roads with the filter {truth_where!r}: "
                         f"{e}. Check the field name and value (text values need quotes, "
                         f"e.g. Validated = 'Yes').")
    if not truth:
        raise ValueError(f"No checked roads match {truth_where!r}. "
                         + _value_hint(arcpy, truth_fc, truth_where))
    msg(f"{len(set(oids))} checked roads ({len(truth)} parts) from {truth_fc}"
        + (f" where {truth_where}" if truth_where else ""))

    pts, tans, line_idx = sample_lines(truth, spacing)
    if grids_fc:
        grids = [g for _, g, _ in read_grids(arcpy, grids_fc, "OID@", sr)]
        inside = _points_in_polygons(pts, grids)
        msg(f"{inside.sum()} of {len(pts)} check points are inside the work grids")
        pts, tans, line_idx = pts[inside], tans[inside], line_idx[inside]
    if not len(pts):
        raise ValueError("No check points to compare (are the checked roads inside the grids?)")

    errors = {}
    for name, fc in candidates:
        lines = [c for _, _, parts, _ in read_lines(arcpy, fc, "OID@", sr) for c in parts]
        errors[name] = SegmentIndex(lines).distances(pts, tans, max_dist)
        msg(f"{name}: {len(lines)} lines read")

    cover = np.array(["unknown"] * len(pts), dtype=object)
    if reader is not None:
        cover = _cover_classes(reader, pts, line_idx)
    junctions = find_junctions(every)
    cover[near_junctions(pts, junctions, junction_dist)] = "intersection"
    for line in eval_report(errors, cover,
                            "ft" if "foot" in units.lower() or "feet" in units.lower() else ""):
        msg(line)
    if out_points:
        _write_check_points(arcpy, out_points, sr, pts, cover, np.asarray(oids)[line_idx], errors)
        msg(f"Check points written to {out_points}")
    return dict(errors=errors, cover=cover)


def _value_hint(arcpy, fc, where) -> str:
    m = re.match(r'\s*"?([A-Za-z_]\w*)"?', where or "")
    if not m:
        return ""
    field = m.group(1)
    try:
        vals = set()
        with arcpy.da.SearchCursor(fc, [field]) as cur:
            for (v,) in cur:
                vals.add(v)
                if len(vals) > 10:
                    break
        return f"Values found in {field}: " + ", ".join(repr(v) for v in sorted(vals, key=str))
    except Exception:
        return f"Check that the field {field} exists."


def _points_in_polygons(pts, polys) -> np.ndarray:
    """Even-odd rule over all rings of every polygon; vectorised over points."""
    inside = np.zeros(len(pts), bool)
    for g in polys:
        e = g.extent
        cand = np.flatnonzero((pts[:, 0] >= e.XMin) & (pts[:, 0] <= e.XMax)
                              & (pts[:, 1] >= e.YMin) & (pts[:, 1] <= e.YMax))
        if not len(cand):
            continue
        x, y = pts[cand, 0], pts[cand, 1]
        hit = np.zeros(len(cand), bool)
        for ring in _rings(g):
            xa, ya = ring[:-1, 0], ring[:-1, 1]
            xb, yb = ring[1:, 0], ring[1:, 1]
            for x0, y0, x1, y1 in zip(xa, ya, xb, yb):
                crosses = ((y0 > y) != (y1 > y)) & (
                    x < (x1 - x0) * (y - y0) / ((y1 - y0) if y1 != y0 else 1e-12) + x0)
                hit ^= crosses
        inside[cand] |= hit
    return inside


def _rings(geom) -> list[np.ndarray]:
    rings = []
    try:
        for part in geom:
            ring = []
            for p in part:
                if p is None:  # ring separator inside a part
                    if len(ring) >= 3:
                        rings.append(np.array(ring + [ring[0]], float))
                    ring = []
                else:
                    ring.append((p.X, p.Y))
            if len(ring) >= 3:
                rings.append(np.array(ring + [ring[0]], float))
    except TypeError:
        pass
    if not rings:  # stand-ins without ring access
        e = geom.extent
        rings.append(np.array([(e.XMin, e.YMin), (e.XMin, e.YMax), (e.XMax, e.YMax),
                               (e.XMax, e.YMin), (e.XMin, e.YMin)]))
    return rings


def _cover_classes(reader, pts, line_idx, chunk=80, pad=12.0):
    cover = np.array(["unknown"] * len(pts), dtype=object)
    order = np.argsort(line_idx, kind="stable")
    for start in range(0, len(order), chunk):
        sel = order[start:start + chunk]
        p = pts[sel]
        x0, y0 = reader.snap(p[:, 0].min() - pad, p[:, 1].max() + pad)
        w = int(np.ceil((p[:, 0].max() + pad - x0) / reader.res)) + 1
        h = int(np.ceil((y0 - (p[:, 1].min() - pad)) / reader.res)) + 1
        if w * h > 4e7:  # very spread-out chunk: do it point by point
            for i in sel:
                cover[[i]] = _cover_classes(reader, pts[[i]], np.zeros(1, int))
            continue
        chip = reader.read(x0, y0, w, h)
        cover[sel] = cover_from_chip(chip, p, reader.has_nir)
    return cover


def _write_check_points(arcpy, out_fc, sr, pts, cover, oids, errors):
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    folder, name = os.path.split(out_fc)
    arcpy.management.CreateFeatureclass(folder, name, "POINT", spatial_reference=sr)
    arcpy.management.AddField(out_fc, "TRUTH_OID", "LONG")
    arcpy.management.AddField(out_fc, "COVER", "TEXT", field_length=20)
    fields = []
    for k, cand in enumerate(errors, 1):
        f = f"ERR_{k}"
        arcpy.management.AddField(out_fc, f, "DOUBLE", field_alias=f"Error: {cand}"[:255])
        fields.append(f)
    with arcpy.da.InsertCursor(out_fc, ["TRUTH_OID", "COVER"] + fields + ["SHAPE@"]) as cur:
        errs = [errors[k] for k in errors]
        for i, (x, y) in enumerate(pts):
            vals = [None if np.isnan(e[i]) else round(float(e[i]), 2) for e in errs]
            cur.insertRow([int(oids[i]), str(cover[i])] + vals
                          + [arcpy.PointGeometry(arcpy.Point(float(x), float(y)), sr)])


# ======================================================================
# arcgis_training.py
# ======================================================================

"""Collect training data while you edit, and train the classifier from it overnight.

Three toolbox buttons, used in this order:

1. Start of day (``snapshot``): remembers every road's geometry before you edit
   it. The first geometry seen for a road is kept as its *original* and never
   overwritten, so the morning snapshot of a road you edited yesterday doesn't
   replace its real original.
2. End of day (``collect``): every road you have checked (e.g.
   ``Validated = 'Yes'``) becomes a training road, with how far your edit moved
   it from its original. Roads you checked and left alone count too: they
   teach the tool when not to move a line.
3. Overnight (``train``): trains the road classifier on the checked roads (road
   labels on your lines, not on a reference layer that is itself a few feet
   off), then measures on grids held back from training how close the tool gets
   the original lines to your edits, and whether the new model beats the last.

Following the REVAMP Phase 6 manual:

* Roads marked abandoned or private (``AbandonedRoad`` / ``PrivateRoad`` = 'Yes',
  when the layer has those fields) are never used: there may be no pavement.
* The ends of every line are left out of the road labels: lines are split at
  intersections, and the manual snaps them simply and bends T-intersections along
  the turn, so near a junction a line is not a pavement centre.
* Roads traced from a reference layer (the manual's Replace Geometry > Trace) are
  marked, so the nightly test can also report on the roads drawn from the imagery.
* A road checked without an edit means "meets the standard" (the line falls
  within the pavement), not "on the exact centre". Such roads teach the keep/move
  model to leave good lines alone.

Nothing is ever written to the road layer. Everything lives in the training
folder: ``originals.json``, ``training_roads.json`` and ``models\\``.
"""

import json
import logging
import os
import time
import zlib

import numpy as np


ORIGINALS = "originals.json"
TRAINING = "training_roads.json"
MODELS = "models"
CURRENT_MODEL = "road_classifier.json"
HOLDOUT_EVERY = 5      # about 1 grid in 5 with checked roads is kept for testing
SAMPLE_SPACING = 5.0   # how often lines are compared (map units)
MATCH_DIST = 40.0      # an original farther than this from the edited line isn't its match
END_TRIM = 40.0        # line ends left out of the road labels (intersections)
TRACE_TOL = 0.5        # a road this close to the reference layer along...
TRACE_SHARE = 0.8      # ...this share of its length was traced from it
EXCLUDE_FIELDS = ("abandonedroad", "privateroad")   # 'Yes' = not used (no pavement)


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _load(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save(folder, name, data):
    path = os.path.join(folder, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _round(parts) -> list:
    return [np.round(np.asarray(c, float), 3).tolist() for c in parts]


def _same(a, b, tol=0.01) -> bool:
    return len(a) == len(b) and all(
        np.shape(x) == np.shape(y) and np.allclose(x, y, atol=tol) for x, y in zip(a, b))


def _read_roads(arcpy, roads_fc, where=None, sr=None):
    """{oid: [(N, 2) coords per part]} of the road layer (honours selections)."""
    out = {}
    try:
        with arcpy.da.SearchCursor(roads_fc, ["OID@", "SHAPE@"], where_clause=where or None,
                                   spatial_reference=sr) as cur:
            for oid, g in cur:
                if g is not None:
                    parts = _parts(g)
                    if parts:
                        out[int(oid)] = parts
    except Exception as e:
        if not where:
            raise
        raise ValueError(f"Could not read the roads with the filter {where!r}: {e}. Check the "
                         f"field name and value (text values need quotes, e.g. "
                         f"Validated = 'Yes').")
    return out


def _excluded(arcpy, roads_fc):
    """OIDs marked abandoned or private ('Yes'), and the fields that said so."""
    fields = [f.name for f in arcpy.ListFields(roads_fc)
              if f.name.lower().replace("_", "").replace(" ", "") in EXCLUDE_FIELDS]
    out = set()
    if fields:
        with arcpy.da.SearchCursor(roads_fc, ["OID@"] + fields) as cur:
            for row in cur:
                if any(str(v).strip().lower() == "yes" for v in row[1:]):
                    out.add(int(row[0]))
    return out, fields


def _with_log(arcpy, fn, *a, **k):
    handler = _ArcHandler(arcpy)
    root = logging.getLogger("centerline_align")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        return fn(*a, **k)
    finally:
        root.removeHandler(handler)


# --------------------------------------------------------------------------- 1. snapshot

def snapshot(arcpy, *, roads_fc, checked_where, training_folder, today=None):
    """Start of day: add every road not seen before to the originals (read-only on roads)."""
    msg = arcpy.AddMessage
    today = today or _today()
    os.makedirs(training_folder, exist_ok=True)
    sr = arcpy.Describe(roads_fc).spatialReference
    roads = _read_roads(arcpy, roads_fc)
    if not roads:
        raise ValueError(f"No roads read from {roads_fc} (check selections / definition queries)")
    checked = set(_read_roads(arcpy, roads_fc, checked_where)) if checked_where else set()
    data = _load(training_folder, ORIGINALS) or {"sr": sr.exportToString(), "roads": {},
                                                 "snapshots": []}
    known = data["roads"]
    new = 0
    for oid, parts in roads.items():
        if str(oid) not in known:
            # a road already checked before its first snapshot: its true original is unknown
            known[str(oid)] = {"day": today, "parts": _round(parts),
                               "checked_before": oid in checked}
            new += 1
    if today not in data["snapshots"]:
        data["snapshots"].append(today)
    _save(training_folder, ORIGINALS, data)
    msg(f"Snapshot {today}: {len(roads)} roads read, {new} new road(s) remembered "
        f"({len(known)} originals in total). Edit as usual; run 'End of Day' when you're done.")
    if new and new == len(known):
        pre = sum(1 for v in known.values() if v["checked_before"])
        if pre:
            arcpy.AddWarning(
                f"{pre} road(s) were already checked before this first snapshot, so their "
                "original geometry is unknown. They are still used for training, but not for "
                "measuring the tool.")
    return {"roads": len(roads), "new": new}


# --------------------------------------------------------------------------- 2. collect

def collect(arcpy, *, roads_fc, checked_where, training_folder, out_fc=None,
            min_change=1.0, reference_fc=None, today=None):
    """End of day: refresh the training roads from every checked road."""
    msg = arcpy.AddMessage
    today = today or _today()
    orig = _load(training_folder, ORIGINALS)
    if not orig:
        raise ValueError(f"No snapshot in {training_folder}. Run 'Start of Day' before editing.")
    if today not in orig["snapshots"]:
        arcpy.AddWarning(
            f"No snapshot was taken today ({today}). Roads first seen today and edited before "
            "this run have their edited shape as their original, so they count as unchanged.")
    sr = arcpy.Describe(roads_fc).spatialReference
    checked = _read_roads(arcpy, roads_fc, checked_where)
    if not checked:
        raise ValueError(f"No roads match {checked_where!r}. "
                         + _value_hint(arcpy, roads_fc, checked_where))
    skip, fields = _excluded(arcpy, roads_fc)
    skipped = len(set(checked) & skip)
    checked = {k: v for k, v in checked.items() if k not in skip}
    if fields:
        msg(f"Roads with {' or '.join(fields)} = 'Yes' are not used ({skipped} checked "
            "road(s) left out).")
    if not checked:
        raise ValueError("Every checked road is marked abandoned or private.")
    ref = None
    if reference_fc:
        ref = SegmentIndex([c for _, _, parts, _ in read_lines(arcpy, reference_fc, "OID@", sr)
                            for c in parts])
    prev = (_load(training_folder, TRAINING) or {}).get("roads", {})
    originals = {int(k): [np.asarray(c) for c in v["parts"]] for k, v in orig["roads"].items()}
    every = None   # all originals, for roads split or added since their snapshot

    roads = {}
    counts = dict(new_moved=0, new_same=0, re_edited=0, kept=0, no_original=0, traced=0)
    for oid, parts in checked.items():
        pts, tans, _ = sample_lines(parts, SAMPLE_SPACING)
        if oid in originals:
            idx = SegmentIndex(originals[oid])
        else:
            every = every or SegmentIndex([c for p in originals.values() for c in p])
            idx = every
        d = idx.distances(pts, tans, MATCH_DIST)
        found = d[~np.isnan(d)]
        info = orig["roads"].get(str(oid))
        known = info is not None and not info["checked_before"]
        if len(found) < 0.5 * len(d):   # most of it is nowhere near an original line
            shift_avg = shift_max = None
            changed = 1
        else:
            shift_avg, shift_max = round(float(found.mean()), 2), round(float(found.max()), 2)
            changed = int(shift_max >= min_change)
        if oid not in originals:
            known = False   # split or added: compared with the nearest originals only
        traced = 0
        if ref is not None and len(pts):
            dr = ref.distances(pts, tans, MATCH_DIST)
            traced = int(np.mean(dr <= TRACE_TOL) >= TRACE_SHARE)   # NaN counts as no
        elif str(oid) in prev:
            traced = prev[str(oid)].get("traced", 0)
        counts["traced"] += traced
        old = prev.get(str(oid))
        parts_r = _round(parts)
        if old is not None and _same(old["parts"], parts_r):
            day = old["day"]
            counts["kept"] += 1
        else:
            day = today
            if old is not None:
                counts["re_edited"] += 1
            elif changed:
                counts["new_moved"] += 1
            else:
                counts["new_same"] += 1
        if shift_avg is None:
            counts["no_original"] += 1
        roads[str(oid)] = dict(day=day, parts=parts_r, changed=changed, shift_avg=shift_avg,
                               shift_max=shift_max, original_known=bool(known),
                               traced=traced)
    dropped = len(set(prev) - set(roads))
    _save(training_folder, TRAINING, {"sr": sr.exportToString(), "updated": today,
                                      "roads": roads})

    n_changed = sum(r["changed"] for r in roads.values())
    msg(f"Training roads: {len(roads)} checked road(s) in total "
        f"({n_changed} moved by your edits, {len(roads) - n_changed} already right).")
    msg(f"Today ({today}): {counts['new_moved']} newly checked and moved, {counts['new_same']} "
        f"newly checked and left as they were, {counts['re_edited']} edited again.")
    if dropped:
        msg(f"{dropped} road(s) are no longer checked (or were deleted) and were removed from "
            "the training roads.")
    if counts["no_original"]:
        arcpy.AddWarning(f"{counts['no_original']} checked road(s) are more than "
                         f"{MATCH_DIST:g} map units from any original line; they are kept but "
                         "not used for measuring.")
    if ref is not None:
        msg(f"{counts['traced']} checked road(s) follow {reference_fc} within {TRACE_TOL:g} "
            "(traced); the nightly test also reports on the others separately.")
    moved = [r["shift_avg"] for r in roads.values() if r["changed"] and r["shift_avg"] is not None]
    if moved:
        msg(f"Roads you moved were off by a median of {np.median(moved):.1f} on average.")
    if out_fc:
        _write_training_fc(arcpy, out_fc, roads, sr)
        msg(f"Training roads written to {out_fc}")
    return dict(counts, total=len(roads), changed=n_changed, dropped=dropped, skipped=skipped)


TRAIN_FIELDS = [("ROAD_OID", "LONG"), ("DAY", "TEXT"), ("CHANGED", "SHORT"),
                ("SHIFT_AVG", "DOUBLE"), ("SHIFT_MAX", "DOUBLE"), ("TRACED", "SHORT")]


def _write_training_fc(arcpy, out_fc, roads, sr):
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    folder, name = os.path.split(out_fc)
    arcpy.management.CreateFeatureclass(folder, name, "POLYLINE", spatial_reference=sr)
    for fname, ftype in TRAIN_FIELDS:
        if ftype == "TEXT":
            arcpy.management.AddField(out_fc, fname, ftype, field_length=10)
        else:
            arcpy.management.AddField(out_fc, fname, ftype)
    with arcpy.da.InsertCursor(out_fc, [f for f, _ in TRAIN_FIELDS] + ["SHAPE@"]) as cur:
        for oid, r in roads.items():
            for c in r["parts"]:
                line = arcpy.Polyline(arcpy.Array([arcpy.Point(x, y) for x, y in c]), sr)
                cur.insertRow([int(oid), r["day"], r["changed"], r["shift_avg"],
                               r["shift_max"], r.get("traced", 0), line])


# --------------------------------------------------------------------------- 3. train

def _projector(arcpy, sr_text, sr):
    """Function mapping stored parts (in the road layer's coordinates) to ``sr``."""
    src = arcpy.SpatialReference()
    src.loadFromString(sr_text)
    if src.name == sr.name:
        return lambda parts: [np.asarray(c, float) for c in parts]

    def project(parts):
        out = []
        for c in parts:
            g = arcpy.Polyline(arcpy.Array([arcpy.Point(x, y) for x, y in c]), src)
            out += _parts(g.projectAs(sr))
        return out
    return project


def _ext(c):
    return (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())


def _grow(e, d):
    return (e[0] - d, e[1] - d, e[2] + d, e[3] + d)


def trim_ends(c, d: float):
    """The part of a line more than ``d`` from both ends, or None if too short."""
    s = cumlen(c)
    if s[-1] <= 2 * d + 10.0:
        return None
    inner = (s > d) & (s < s[-1] - d)
    ends = interpolate(c, np.array([d, s[-1] - d]), s)
    return np.vstack([ends[:1], c[inner], ends[1:]])


def train(arcpy, *, imagery, training_folder, grids_fc, grid_id_field, bands,
          compare=(), training_grids=30, refine: RefineConfig | None = None, today=None):
    return _with_log(arcpy, _train, arcpy, imagery, training_folder, grids_fc, grid_id_field,
                     bands, list(compare or ()), training_grids, refine or RefineConfig(),
                     today or _today())


def _train(arcpy, imagery, training_folder, grids_fc, grid_id_field, bands, compare,
           training_grids, rcfg, today):
    msg = arcpy.AddMessage
    store = _load(training_folder, TRAINING)
    orig = _load(training_folder, ORIGINALS)
    if not store or not store["roads"] or not orig:
        raise ValueError(f"No training roads in {training_folder}. Use 'Start of Day' and "
                         "'End of Day' while you edit first.")
    reader = ArcTileReader(arcpy, imagery, bands)
    sr = reader.sr
    msg(f"Imagery: {len(reader.tiles)} raster(s), {sr.name}, cell size {reader.res:g}")

    to_sr = _projector(arcpy, store["sr"], sr)
    truth = []   # (oid, coords, ext, record)
    for oid, r in store["roads"].items():
        for c in to_sr(r["parts"]):
            truth.append((int(oid), c, _ext(c), r))
    to_sr = _projector(arcpy, orig["sr"], sr)
    originals = []   # (oid, coords, ext)
    for oid, r in orig["roads"].items():
        for c in to_sr(r["parts"]):
            originals.append((int(oid), c, _ext(c)))

    grids = read_grids(arcpy, grids_fc, grid_id_field, sr)
    mids = np.array([interpolate(c, cumlen(c)[-1] / 2)[0] for _, c, _, _ in truth])
    home = np.full(len(truth), -1)
    for gi, (_, g, _) in enumerate(grids):
        free = home < 0
        if free.any():
            inside = _points_in_polygons(mids[free], [g])
            home[np.flatnonzero(free)[inside]] = gi
    with_truth = sorted(set(home[home >= 0].tolist()))
    if not with_truth:
        raise ValueError("None of the checked roads are inside the grids")
    holdout = [gi for gi in with_truth
               if zlib.crc32(str(grids[gi][0]).encode()) % HOLDOUT_EVERY == 0]
    if not holdout and len(with_truth) >= 2:
        holdout = [min(with_truth, key=lambda gi: zlib.crc32(str(grids[gi][0]).encode()))]
    if len(holdout) == len(with_truth):
        holdout = []
    learn = [gi for gi in with_truth if gi not in holdout]
    msg(f"{len(store['roads'])} checked roads in {len(with_truth)} grid(s): training on "
        f"{len(learn)}, testing on {len(holdout)} ("
        + (", ".join(str(grids[g][0]) for g in holdout) or "none: too few grids yet") + ")")

    ccfg = ClassifierConfig(training_grids=training_grids)
    train_on = pick_spread(learn, training_grids)
    test_set = set(holdout)
    learn_truth = []   # (trimmed coords, ext, home grid): road labels, ends left out
    for (_, c, _, _), h in zip(truth, home):
        t = trim_ends(c, END_TRIM)
        if t is not None and h not in test_set:
            learn_truth.append((t, _ext(t), h))
    cells = [[c for c, _, h in learn_truth if h == gi] for gi in train_on]
    if not any(cells):
        raise ValueError("The training grids have no checked roads longer than "
                         f"{2 * END_TRIM:g} to learn from yet.")

    def lines_near(b):
        return ([c for _, c, e, _ in truth if _overlaps(e, b)]
                + [c for _, c, e in originals if _overlaps(e, b)])

    def trusted_near(b):
        # road labels on your lines only; the originals (and roads in the test grids)
        # just keep road pixels out of the not-road labels
        return ([(c, None) for c, e, _ in learn_truth if _overlaps(e, b)],
                [c for _, c, e in originals if _overlaps(e, b)]
                + [c for _, c, e, _ in truth if _overlaps(e, b)])

    msg("Training the road classifier on grid(s): "
        + ", ".join(str(grids[g][0]) for g in train_on))
    clf = train_classifier(reader, cells, lines_near, ccfg, None, trusted_near=trusted_near)
    clf.trained_from = f"{len(learn_truth)} checked road parts, {today}"
    clf.move_model = _learn_moves(arcpy, msg, reader, grids, train_on, truth, home,
                                  originals, clf, rcfg, ccfg, sr)
    models = os.path.join(training_folder, MODELS)
    os.makedirs(models, exist_ok=True)
    dated = os.path.join(models, f"road_classifier_{today}.json")
    clf.save(dated)
    msg(f"Classifier saved to {dated} (training accuracy {clf.train_accuracy:.3f})")

    current_path = os.path.join(models, CURRENT_MODEL)
    prev = RoadClassifier.load(current_path) if os.path.exists(current_path) else None
    if prev is not None and (prev.has_nir != clf.has_nir or prev.lidar):
        prev = None   # different bands or LiDAR: not comparable

    scores = {}
    if holdout:
        scores = _score(arcpy, msg, reader, grids, holdout, truth, home, originals, clf, prev,
                        compare, rcfg, ccfg, sr)
    new, old = scores.get("new"), scores.get("previous")
    promote = prev is None or new is None or old is None or new <= old
    if promote:
        clf.save(current_path)
        msg(f"This model is now the current one: {current_path}. Point 'Align Centerlines' "
            "at it (Trained classifier).")
    else:
        msg(f"The new model ({new:.2f}) did not beat the current one ({old:.2f}) on the test "
            f"grids, so {current_path} is unchanged. Tonight's model is kept as {dated}.")
    _append_history(models, today, len(store["roads"]), len(train_on), len(holdout), clf,
                    scores, promote)
    return dict(scores, promoted=promote, model=dated, holdout=[grids[g][0] for g in holdout])


def _score(arcpy, msg, reader, grids, holdout, truth, home, originals, clf, prev, compare,
           rcfg, ccfg, sr):
    """Align the original lines in the test grids with each model and compare with your
    edits. Roads whose original is unknown (checked before the first snapshot) are skipped."""
    test = [(c, e, r.get("traced", 0)) for (_, c, e, r), h in zip(truth, home)
            if h in set(holdout) and r.get("original_known")]
    if not test:
        arcpy.AddWarning("The test grids have no checked roads with a known original yet, so "
                         "the models can't be measured.")
        return {}
    near = [(oid, c, e) for oid, c, e in originals
            if any(_overlaps(e, _grow(te, MATCH_DIST)) for _, te, _ in test)]
    roads = [(oid, arcpy.Polyline(arcpy.Array([arcpy.Point(x, y) for x, y in c]), sr), [c], e)
             for oid, c, e in near]
    models = [("Tool, new model", clf)] + ([("Tool, current model", prev)] if prev else [])
    aligned = {name: [] for name, _ in models}
    for gi in holdout:
        pieces = clip_to_grid(arcpy, roads, grids, gi, sr)
        if not pieces:
            continue
        lines = [c for _, c in pieces]
        for name, model in models:
            pmap = region_probability(reader, model, None, grids[gi][2], lines, rcfg, ccfg)
            records, _ = refine_pieces(pmap, pieces, rcfg, reader.res,
                                       move_model=model.move_model)
            aligned[name] += [np.asarray(r["coords"]) for r in records]
        msg(f"Test grid {grids[gi][0]}: {len(pieces)} road pieces aligned")

    lines = [c for c, _, _ in test]
    pts, tans, line_idx = sample_lines(lines, SAMPLE_SPACING)
    errors = {"Original lines": SegmentIndex([c for _, c, _ in near]).distances(
        pts, tans, MATCH_DIST)}
    for name, _ in models:
        errors[name] = SegmentIndex(aligned[name]).distances(pts, tans, MATCH_DIST)
    for fc in compare:
        other = [c for _, _, parts, _ in read_lines(arcpy, fc, "OID@", sr) for c in parts]
        errors[os.path.basename(fc) or fc] = SegmentIndex(other).distances(pts, tans, MATCH_DIST)
    cover = _cover_classes(reader, pts, line_idx)
    cover[near_junctions(pts, find_junctions([c for _, c, _ in near]), 40.0)] = "intersection"
    msg(f"How close each layer is to your edits on {len(lines)} checked road part(s) in the "
        "test grids (smaller is better):")
    units = sr.linearUnitName or ""
    ft = "ft" if "foot" in units.lower() or "feet" in units.lower() else ""
    for line in eval_report(errors, cover, ft):
        msg(line)
    traced = np.array([t for _, _, t in test], bool)[line_idx]
    if traced.any() and not traced.all():
        msg("")
        msg(f"Only roads you did NOT trace from a reference layer ({(~traced).sum()} points): "
            "the fair test against that layer")
        for line in eval_report({k: e[~traced] for k, e in errors.items()}, cover[~traced], ft):
            msg(line)
    # models are compared away from intersections, where the manual's snapping rules
    # (not the pavement centre) decide where a line goes
    use = np.all([~np.isnan(e) for e in errors.values()], axis=0) & (cover != "intersection")
    med = {k: float(np.median(e[use])) if use.any() else None for k, e in errors.items()}
    msg(f"Median distance to your edits away from intersections ({int(use.sum())} points): "
        + ", ".join(f"{k} {v:.2f}" for k, v in med.items() if v is not None))
    return {"original": med["Original lines"], "new": med["Tool, new model"],
            "previous": med.get("Tool, current model"), "points": int(use.sum())}


def _learn_moves(arcpy, msg, reader, grids, train_on, truth, home, originals, clf, rcfg,
                 ccfg, sr):
    """Align the original lines of checked roads in the training grids and learn which
    moves bring a line closer to the checked line (keep/move model)."""
    by_oid: dict = {}
    for (oid, c, _, r), h in zip(truth, home):
        if r.get("original_known") and h in train_on:
            by_oid.setdefault(oid, []).append(c)
    orig = [(oid, c, e) for oid, c, e in originals if oid in by_oid]
    if not orig:
        msg("Keep/move model: no checked roads with a known original in the training grids "
            "yet.")
        return None
    roads = [(oid, arcpy.Polyline(arcpy.Array([arcpy.Point(x, y) for x, y in c]), sr), [c], e)
             for oid, c, e in orig]
    recs, gains = [], []
    for gi in train_on:
        pieces = clip_to_grid(arcpy, roads, grids, gi, sr)
        if not pieces:
            continue
        pmap = region_probability(reader, clf, None, grids[gi][2], [c for _, c in pieces],
                                  rcfg, ccfg)
        records, _ = refine_pieces(pmap, pieces, rcfg, reader.res)
        for (oid, c0), rec in zip([p for p in pieces if len(p[1]) >= 2], records):
            if not rec["moved"]:
                continue
            truth_idx = SegmentIndex(by_oid[oid])
            # compare away from the ends (intersections follow the manual's snapping rules)
            before, after = trim_ends(np.asarray(c0), END_TRIM), trim_ends(
                np.asarray(rec["coords"]), END_TRIM)
            if before is None or after is None:
                continue
            e = []
            for line in (before, after):
                p, t, _ = sample_lines([line], SAMPLE_SPACING)
                d = truth_idx.distances(p, t, MATCH_DIST)
                e.append(float(np.nanmean(d)) if np.mean(~np.isnan(d)) >= 0.5 else np.nan)
            if not np.isnan(e).any():
                recs.append(rec)
                gains.append(e[0] - e[1])
    model = fit_move_model(recs, gains)
    if model is None:
        msg(f"Keep/move model: {len(recs)} moved line(s) to learn from; need at least 20 with "
            "moves that helped and moves that didn't. Every move is applied for now.")
        return None
    kept = sum(move_probability(model, r) < model["threshold"] for r in recs)
    msg(f"Keep/move model learned from {len(recs)} moved lines: it keeps the original for "
        f"{kept} of them. On these lines, applying every move would bring them "
        f"{model['gain_all']:+.0f} closer in total, applying only the chosen ones "
        f"{model['gain_chosen']:+.0f}.")
    return model


def _append_history(models, today, n_roads, n_train, n_test, clf, scores, promoted):
    path = os.path.join(models, "history.csv")
    head = not os.path.exists(path)

    def f(v):
        return "" if v is None else f"{v:.2f}"
    with open(path, "a") as out:
        if head:
            out.write("day,checked_roads,train_grids,test_grids,train_accuracy,"
                      "median_original,median_new,median_current,promoted\n")
        out.write(f"{today},{n_roads},{n_train},{n_test},{clf.train_accuracy:.3f},"
                  f"{f(scores.get('original'))},{f(scores.get('new'))},"
                  f"{f(scores.get('previous'))},{int(promoted)}\n")


# --------------------------------------------------------------------------- deep learning

def export_training_chips(arcpy, *, truth_fc, truth_where, imagery, bands, out_folder,
                          grids_fc=None, local_fc=None, training_folder=None, only_moved=False,
                          chip_px=256, step=100.0, val_fraction=0.2, block=1000.0, sigma=1.0,
                          corridor=30.0, end_trim=END_TRIM, min_valid=0.8, shard_size=200,
                          seed=0, make_zip=True):
    """Chips around hand-checked roads, for training a centreline model elsewhere
    (see ``chips.py`` and ``dl/train_centerline.py``). Only reads the inputs.

    With ``training_folder`` (End of Day's output), roads traced from a reference
    layer are kept for training but not scored, and ``only_moved`` keeps only the
    roads you moved."""
    import shutil
    msg = arcpy.AddMessage
    reader = ArcTileReader(arcpy, imagery, list(bands))
    sr, res = reader.sr, reader.res
    rng = np.random.default_rng(seed)

    with arcpy.da.SearchCursor(truth_fc, ["OID@", "SHAPE@"], spatial_reference=sr) as cur:
        everything = [(int(oid), _parts(g)) for oid, g in cur if g is not None]
    try:
        with arcpy.da.SearchCursor(truth_fc, ["OID@"], where_clause=truth_where or None) as cur:
            checked = {int(oid) for (oid,) in cur}
    except Exception as e:
        raise ValueError(f"Could not read the checked roads with the filter {truth_where!r}: {e}")
    if not checked:
        raise ValueError(f"No checked roads match {truth_where!r}. "
                         + _value_hint(arcpy, truth_fc, truth_where))
    excluded, ex_fields = _excluded(arcpy, truth_fc)
    if excluded & checked:
        msg(f"{len(excluded & checked)} checked road(s) left out: {', '.join(ex_fields)} = 'Yes'")
    info = {}
    if training_folder:
        info = {int(k): v for k, v in _load(training_folder, TRAINING).get("roads", {}).items()}
        if not info:
            raise ValueError(f"No {TRAINING} in {training_folder}: run 'End of Day' first")
    use = checked - excluded
    if only_moved:
        use = {o for o in use if info.get(o, {}).get("changed")}
        msg(f"Using only the {len(use)} checked road(s) you moved")
    truth = [(o, c) for o, parts in everything if o in use for c in parts]
    other = [(o, c) for o, parts in everything if o not in use for c in parts]
    if not truth:
        raise ValueError("No checked roads left to train on")
    traced = {o for o in use if info.get(o, {}).get("traced")}
    if traced:
        msg(f"{len(traced)} checked road(s) were traced from a reference layer: used for "
            "training, left out of the scores")
    grid_polys = [g for _, g, _ in read_grids(arcpy, grids_fc, "OID@", sr)] if grids_fc else None

    os.makedirs(out_folder, exist_ok=True)
    for f in os.listdir(out_folder):  # a fresh export replaces an old one
        if f.startswith("shard_") or f in ("manifest.json", "baseline.json"):
            os.remove(os.path.join(out_folder, f))

    truth_ext = [_ext(c) for _, c in truth]
    other_ext = [_ext(c) for _, c in other]
    ends = np.array([p for _, c in truth for p in (c[0], c[-1])])
    scored = [(o, trim_ends(c, end_trim)) for o, c in truth if o not in traced]
    scored = [(o, c) for o, c in scored if c is not None]
    scored_ext = [_ext(c) for _, c in scored]
    half = chip_px * res / 2
    buffers = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    shards = {"train": 0, "val": 0}

    def flush(split):
        items = buffers[split]
        if not items:
            return
        k = shards[split]
        np.savez_compressed(
            os.path.join(out_folder, f"shard_{split}_{k:03d}.npz"),
            image=np.stack([i["image"] for i in items]),
            heat=np.stack([i["heat"] for i in items]),
            mask=np.stack([i["mask"] for i in items]),
            x0=np.array([i["x0"] for i in items]), y0=np.array([i["y0"] for i in items]),
            road=np.array([str(i["road"]) for i in items]))
        with open(os.path.join(out_folder, f"shard_{split}_{k:03d}.json"), "w") as f:
            json.dump([i["lines"] for i in items], f)
        shards[split] += 1
        buffers[split] = []

    centres = []
    for oid, c in truth:
        pts, _, _ = sample_lines([c], step)
        centres += [(oid, p) for p in pts]
    if grid_polys is not None:
        inside = _points_in_polygons(np.array([p for _, p in centres]), grid_polys)
        centres = [cp for cp, ok in zip(centres, inside) if ok]
    msg(f"{len(use)} checked roads, {len(other)} other road parts; {len(centres)} chip "
        f"positions every {step:g} {sr.linearUnitName}; validation = {val_fraction:.0%} of "
        f"{block:g} {sr.linearUnitName} blocks")
    arcpy.SetProgressor("step", "Exporting training chips...", 0, max(1, len(centres)), 1)
    skipped = 0
    for k, (oid, p) in enumerate(centres):
        arcpy.SetProgressorPosition(k)
        jx, jy = rng.uniform(-0.25, 0.25, 2) * chip_px * res   # the road isn't always centred
        x0, y0 = reader.snap(p[0] - half + jx, p[1] + half + jy)
        split = "val" if in_validation(x0 + half, y0 - half, block, val_fraction, seed) else "train"
        chip = reader.read(x0, y0, chip_px, chip_px)
        if chip.valid.mean() < min_valid:
            skipped += 1
            continue
        b = chip.bounds
        box = _grow(b, corridor + end_trim + 20)
        t_lines = [c for (_, c), e in zip(truth, truth_ext) if _overlaps(e, box)]
        o_lines = [c for (_, c), e in zip(other, other_ext) if _overlaps(e, box)]
        vpix = validation_pixels(x0, y0, res, chip.shape, block, val_fraction, seed)
        keep = (vpix if split == "val" else ~vpix) & chip.valid
        if not keep.any():
            continue
        heat, mask = make_labels(x0, y0, res, chip.shape, t_lines, o_lines, sigma, corridor,
                                 ends=ends[(ends[:, 0] > box[0]) & (ends[:, 0] < box[2])
                                           & (ends[:, 1] > box[1]) & (ends[:, 1] < box[3])],
                                 end_trim=end_trim, keep=keep)
        s_lines = [c for (_, c), e in zip(scored, scored_ext) if _overlaps(e, box)]
        buffers[split].append(dict(
            image=np.clip(chip.data, 0, 255).astype(np.uint8), heat=heat, mask=mask,
            x0=x0, y0=y0, road=oid, lines=lines_in_pixels(x0, y0, res, s_lines)))
        counts[split] += 1
        if len(buffers[split]) >= shard_size:
            flush(split)
    for s_ in ("train", "val"):
        flush(s_)
    arcpy.ResetProgressor()
    if not counts["train"] or not counts["val"]:
        raise ValueError(f"Too few chips (train {counts['train']}, validation {counts['val']}): "
                         "check that the checked roads are inside the grids and the imagery, "
                         "or raise the validation share")

    names = ["red", "green", "blue", "nir"][:len(bands)]
    manifest = dict(channels=names, bands=list(bands), res=res, units=sr.linearUnitName,
                    crs=sr.name, chip_px=chip_px, sigma=sigma, corridor=corridor,
                    end_trim=end_trim, block=block, val_fraction=val_fraction, seed=seed,
                    counts=counts, roads=len(use), traced=len(traced), only_moved=only_moved,
                    created=time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(out_folder, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    msg(f"{counts['train']} training and {counts['val']} validation chips "
        f"({skipped} skipped for missing imagery) in {out_folder}")

    if local_fc:  # the local roads' error at the points the model is scored on
        local = [c for _, _, parts, _ in read_lines(arcpy, local_fc, "OID@", sr) for c in parts]
        idx = SegmentIndex(local)
        pts, tans, _ = sample_lines([c for _, c in scored], SAMPLE_SPACING)
        if len(pts):
            near_end = near_junctions(pts, ends, end_trim)
            val = np.array([in_validation(x, y, block, val_fraction, seed) for x, y in pts])
            ok = ~near_end
            if grid_polys is not None:
                ok &= _points_in_polygons(pts, grid_polys)
        base = {}
        for split, sel in (("train", ~val), ("val", val)) if len(pts) else ():
            e = idx.distances(pts[ok & sel], tans[ok & sel])
            e = e[~np.isnan(e)]
            if len(e):
                base[split] = dict(points=int(len(e)), median=float(np.median(e)),
                                   mean=float(e.mean()), p90=float(np.percentile(e, 90)),
                                   within05=float((e <= 0.5).mean()),
                                   within1=float((e <= 1).mean()), within2=float((e <= 2).mean()))
        with open(os.path.join(out_folder, "baseline.json"), "w") as f:
            json.dump(dict(layer=str(local_fc), **base), f, indent=1)
        if "val" in base:
            v = base["val"]
            msg(f"Local roads in the validation blocks: median {v['median']:.2f}, worst 10% "
                f"{v['p90']:.2f}, within 1 {v['within1']:.0%}, within 2 {v['within2']:.0%} "
                "- the score to beat")
    if make_zip:
        z = shutil.make_archive(out_folder.rstrip("\\/"), "zip", out_folder)
        msg(f"Zipped for upload: {z} ({os.path.getsize(z) / 1024 ** 2:,.0f} MB)")
    return dict(counts=counts, manifest=manifest)


# ======================================================================
# arcgis_toolbox.py
# ======================================================================

"""ArcGIS Pro Python toolbox definition (becomes ``arcgis/CenterlineAlign.pyt``)."""
import arcpy



class Toolbox:
    def __init__(self):
        self.label = "Centerline Alignment"
        self.alias = "centerline"
        self.tools = [AlignCenterlines, EvaluateAlignment, StartOfDay, EndOfDay, TrainOvernight,
                      ExportTrainingChips]


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
        params.append(_param("classifier_file",
                             "Trained classifier (optional; from 'Overnight: Train From My "
                             "Edits')", "DEFile", required=False, category="Trained model"))
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
        rcfg = RefineConfig(method=p["method"].valueAsText or "ribbon",
                            keep_flagged=p["keep_flagged"].value is not False,
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
            lidar_folder=p["lidar_folder"].valueAsText,
            classifier_file=p["classifier_file"].valueAsText)

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


class TrainOvernight:
    def __init__(self):
        self.label = "3. Overnight: Train From My Edits"
        self.description = (
            "Trains the road classifier from the roads you checked, then tests it on grids "
            "kept out of training: how close the tool gets the original lines to your edits, "
            "compared with the current model. The better model becomes the current one.")
        self.canRunInBackground = True

    def getParameterInfo(self):
        imagery = _param("imagery", "Imagery (raster, mosaic dataset, or folder of .tif tiles)",
                         ["GPRasterLayer", "DERasterDataset", "DEMosaicDataset", "DEFolder"])
        folder = _param("training_folder", "Training folder (the one used for Start/End of Day)",
                        "DEFolder")
        grids = _param("grids", "Work grids (1-mile)", "GPFeatureLayer")
        grids.filter.list = ["Polygon"]
        grid_id = _param("grid_id_field", "Grid ID field", "Field")
        grid_id.parameterDependencies = [grids.name]
        compare = arcpy.Parameter(name="compare", displayName="Also compare these layers "
                                  "(optional, e.g. the county roads)", datatype="GPFeatureLayer",
                                  parameterType="Optional", direction="Input", multiValue=True)
        params = [imagery, folder, grids, grid_id, compare]
        for name, label, default in (("red", "Red band", 1), ("green", "Green band", 2),
                                     ("blue", "Blue band", 3)):
            params.append(_param(name, label, "GPLong", category="Bands", value=default))
        params.append(_param("nir", "Near-infrared band (leave empty if none)", "GPLong",
                             required=False, category="Bands", value=4))
        params.append(_param("training_grids", "Grids used to train the classifier", "GPLong",
                             category="Advanced", value=30))
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
        train(arcpy, imagery=p["imagery"].valueAsText,
              training_folder=p["training_folder"].valueAsText,
              grids_fc=p["grids"].valueAsText, grid_id_field=p["grid_id_field"].valueAsText,
              bands=bands, compare=_split_multi(p["compare"].valueAsText),
              training_grids=int(p["training_grids"].value or 30))

    def postExecute(self, parameters):
        return


class ExportTrainingChips:
    def __init__(self):
        self.label = "Export Training Chips (deep learning)"
        self.description = (
            "Cuts imagery chips around roads you checked by hand and labels where their true "
            "centreline is, for training a centreline model in Google Colab (see "
            "dl/README.md). Writes a zip to upload. Only reads the inputs.")
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
        return [truth, where, imagery, out, local, grids, folder, moved, bands, chip, step, val,
                block]

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
            only_moved=bool(p["only_moved"].value))

    def postExecute(self, parameters):
        return



if _placeholder is not None and _sys.modules.get(__name__) is _placeholder:
    del _sys.modules[__name__]
