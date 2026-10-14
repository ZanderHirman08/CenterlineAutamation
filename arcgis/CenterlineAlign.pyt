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

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, path) -> "RoadClassifier":
        with open(path) as f:
            return cls(**json.load(f))

    def predict(self, chip: Chip, embedder: EmbeddingExtractor | None = None) -> np.ndarray:
        """Road probability (H, W) float32 for a chip; 0 where imagery is missing."""
        scaled = scale_bands(chip.data, self.band_lo, self.band_hi)
        feats = pixel_features(scaled, self.has_nir)
        w = np.asarray(self.pix_coef, np.float32) / np.asarray(self.pix_std, np.float32)
        z = np.tensordot(w, feats - np.asarray(self.pix_mean, np.float32)[:, None, None], axes=1)
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


def sample_training_pixels(chip: Chip, lines, ccfg, rng, n_per_class: int):
    """Pick (row, col) indexes of positive and negative training pixels in a chip."""
    dist = distance_to_roads(chip, lines)
    ok = chip.valid
    pos = np.flatnonzero((dist <= ccfg.positive_half_width) & ok)
    neg = np.flatnonzero((dist > ccfg.negative_min_dist) & (dist <= ccfg.negative_max_dist) & ok)
    pos = rng.choice(pos, min(len(pos), n_per_class), replace=False) if len(pos) else pos
    neg = rng.choice(neg, min(len(neg), n_per_class), replace=False) if len(neg) else neg
    idx = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    rows, cols = np.unravel_index(idx, chip.shape)
    return rows, cols, labels


def fit(samples_pix: np.ndarray, samples_emb: np.ndarray | None, labels: np.ndarray,
        *, feature_mode: str, has_nir: bool, band_lo, band_hi,
        embedding_model: str | None) -> RoadClassifier:
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
    mean = X.mean(axis=0)
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
        pix_names=pixel_feature_names(has_nir),
        pix_mean=mean[:n_pix].tolist(),
        pix_std=std[:n_pix].tolist(),
        pix_coef=coef[:n_pix].tolist(),
        intercept=float(intercept),
        train_accuracy=acc,
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


def extract_samples(chip: Chip, rows, cols, has_nir, band_lo, band_hi, embedder=None):
    """Feature vectors (n, F) at the given pixels, matching what ``predict`` computes."""
    scaled = scale_bands(chip.data, band_lo, band_hi)
    feats = pixel_features(scaled, has_nir)
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
sampled along a short line perpendicular to the road. The run of "road"
pixels nearest the current line gives the two pavement edges; their midpoint
is the measured centre. Offsets are cleaned along the road (outliers dropped,
gaps interpolated, smoothed) and the line is rebuilt from the shifted stations.
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
                valid: Sampler | None = None, keep_stations: bool = False) -> RefineResult:
    """Refine one line given as (N, 2) coords. ``step`` is the cross-section sample
    spacing (normally the pixel size)."""
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
    raw = np.full(n, np.nan)
    width = np.full(n, np.nan)
    conf = np.zeros(n)
    for i in range(n):
        m = measure_profile(offs, profiles[i], cfg)
        if m is not None and m[2] > 0:
            raw[i], width[i], conf[i] = m

    reasons = []
    no_data = False
    if valid is not None:
        has = valid(pts[:, 0], pts[:, 1]) > 0.5
        if (~has).mean() > 0.5:
            no_data = True
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
                     seed: int = 0, on_chip=None) -> RoadClassifier:
    """Train from tiles centred on random road points in each training cell.

    cells: for each training grid, the road coords inside it.
    lines_near(bounds) -> road coords intersecting ``bounds`` (for labelling).
    on_chip(k, chip, lines) is called for every training tile (for diagnostics).
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
        rows, cols, labels = sample_training_pixels(chip, near, ccfg, rng, per_tile)
        if len(labels) == 0:
            continue
        pix, emb = extract_samples(chip, rows, cols, reader.has_nir, lo, hi, embedder)
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
              embedding_model=ccfg.embedding_model if embedder is not None else None)
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


def refine_pieces(pmap: ProbabilityMap, pieces, rcfg, step: float, keep_stations=False):
    """Refine (src_id, coords) pieces. Returns (records, stations) as plain dicts."""
    records, stations = [], []
    for src_id, coords in pieces:
        c = as_coords(coords)
        if len(c) < 2:
            continue
        r = refine_line(c, pmap.sample, rcfg, step=step, valid=pmap.sample_valid,
                        keep_stations=keep_stations)
        records.append(dict(
            src_id=src_id, shift_avg=round(r.shift_avg, 2), shift_max=round(r.shift_max, 2),
            conf=r.confidence, pct_valid=r.pct_valid, width_med=round(r.width_med, 1),
            review=int(r.needs_review), reason=";".join(r.reasons),
            o_sx=float(c[0, 0]), o_sy=float(c[0, 1]), o_ex=float(c[-1, 0]), o_ey=float(c[-1, 1]),
            coords=r.coords,
        ))
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


TILE_EXTS = (".tif", ".tiff", ".img", ".jp2")


class ArcTileReader:
    """Reads blocks from one raster or a folder of same-grid tiles."""

    def __init__(self, arcpy, source: str, bands: list[int]):
        self.arcpy = arcpy
        paths = self._find(source)
        self.tiles = []
        for p in paths:
            r = arcpy.Raster(p)
            e = r.extent
            self.tiles.append(dict(path=p, xmin=e.XMin, ymin=e.YMin, xmax=e.XMax, ymax=e.YMax,
                                   res=float(r.meanCellWidth), count=int(r.bandCount),
                                   sr=r.spatialReference))
        first = self.tiles[0]
        self.sr = first["sr"]
        self.res = first["res"]
        if self.sr is None or self.sr.type != "Projected":
            raise ValueError("The imagery must be in a projected coordinate system "
                             "(e.g. State Plane), not geographic.")
        for t in self.tiles[1:]:
            if abs(t["res"] - self.res) > 1e-6 * self.res:
                raise ValueError(f"{t['path']} has cell size {t['res']}, expected {self.res}")
            if t["sr"].name != self.sr.name:
                raise ValueError(f"{t['path']} is in {t['sr'].name}, expected {self.sr.name}")
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
            arr = self.arcpy.RasterToNumPyArray(t["path"], corner, nc, nr, 0)
            arr = arr[None] if arr.ndim == 2 else arr
            block = arr[[b - 1 for b in self.bands]].astype(np.float32)
            data[:, r0:r1, c0:c1] = block
            valid[r0:r1, c0:c1] = block.any(axis=0)
        return Chip(data=data, valid=valid, x0=x0, y0=y0, res=r)


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
        diagnostics=False):
    msg = arcpy.AddMessage
    handler = _ArcHandler(arcpy)
    root = logging.getLogger("centerline_align")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        return _run(arcpy, msg, imagery, roads_fc, road_id_field, grids_fc, grid_id_field,
                    work_folder, out_fc, bands, refine, classifier, fresh, diagnostics)
    finally:
        root.removeHandler(handler)


def _run(arcpy, msg, imagery, roads_fc, road_id_field, grids_fc, grid_id_field, work_folder,
         out_fc, bands, rcfg, ccfg, fresh, diagnostics=False):
    os.makedirs(work_folder, exist_ok=True)
    grid_dir = os.path.join(work_folder, "grids")
    os.makedirs(grid_dir, exist_ok=True)

    reader = ArcTileReader(arcpy, imagery, bands)
    sr = reader.sr
    msg(f"Imagery: {len(reader.tiles)} raster(s), {sr.name}, cell size {reader.res:g} "
        f"{sr.linearUnitName}. All distance settings are in {sr.linearUnitName}.")
    msg(f"Bands used: {reader.bands} (NIR: {'yes' if reader.has_nir else 'no'})")

    roads = read_lines(arcpy, roads_fc, road_id_field, sr)
    grids = read_grids(arcpy, grids_fc, grid_id_field, sr)
    if not roads or not grids:
        raise ValueError("No roads or no grids to process (check selections / definition queries)")
    msg(f"{len(roads)} roads, {len(grids)} grids (only selected features are used)")

    model_path = os.path.join(work_folder, "road_classifier.json")
    if fresh:
        for f in os.listdir(grid_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(grid_dir, f))
        if os.path.exists(model_path):
            os.remove(model_path)
    if os.path.exists(model_path):
        clf = RoadClassifier.load(model_path)
        msg(f"Using saved classifier {model_path}")
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
                records, _ = refine_pieces(pmap, pieces, rcfg, reader.res)
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
              ("WIDTH_MED", "DOUBLE"), ("REVIEW", "SHORT"), ("REASON", "TEXT")]


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
                           r["pct_valid"], r["width_med"], int(r["review"]), r["reason"], line])


# ======================================================================
# arcgis_toolbox.py
# ======================================================================

"""ArcGIS Pro Python toolbox definition (becomes ``arcgis/CenterlineAlign.pyt``)."""
import arcpy



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



if _placeholder is not None and _sys.modules.get(__name__) is _placeholder:
    del _sys.modules[__name__]
