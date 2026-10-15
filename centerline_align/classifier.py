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
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .chip import Chip
from .features import (
    EmbeddingExtractor,
    all_features,
    lidar_feature_names,
    pixel_feature_names,
    sample_lowres,
    scale_bands,
    upsample_lowres,
)
from .geom import densify

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
    untrusted lines [coords]) from an earlier run (second pass). Then the road
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
        by_width: dict[float, list] = {}
        for c, hw in good:
            by_width.setdefault(float(np.round(hw * 0.5) * 2), []).append(c)   # 2-unit bins
        for hw, group in by_width.items():
            d = distance_to_roads(chip, group)
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
