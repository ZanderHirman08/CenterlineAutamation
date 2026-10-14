"""Road / not-road pixel classifier trained from the local road file.

The local centerlines are used as free (if slightly noisy) labels: pixels very
close to a road line are "road", pixels a good distance from every road are
"not road". A logistic regression on those samples gives a road-probability
map that is far more robust than a fixed colour threshold.

The model is linear on standardised features, so the embedding part can be
evaluated at patch resolution and upsampled afterwards (see ``predict``).
It is saved as plain JSON - no pickles.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
from rasterio import features as rfeatures
from rasterio.transform import from_origin
from scipy import ndimage

from .features import (
    EmbeddingExtractor,
    pixel_feature_names,
    pixel_features,
    sample_lowres,
    scale_bands,
    upsample_lowres,
)
from .imagery import Chip

log = logging.getLogger(__name__)


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


def distance_to_roads(chip: Chip, lines) -> np.ndarray:
    """Distance (CRS units) from each chip pixel centre to the nearest road line."""
    transform = from_origin(chip.x0, chip.y0, chip.res, chip.res)
    shapes = [(g, 1) for g in lines if g is not None and not g.is_empty]
    if not shapes:
        return np.full(chip.shape, np.inf, dtype=np.float32)
    burned = rfeatures.rasterize(shapes, out_shape=chip.shape, transform=transform,
                                 fill=0, all_touched=True, dtype="uint8")
    return (ndimage.distance_transform_edt(burned == 0) * chip.res).astype(np.float32)


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
    from sklearn.linear_model import LogisticRegression

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
    model = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
    model.fit(Xs, labels)
    acc = float(model.score(Xs, labels))
    log.info("Classifier trained on %d samples, training accuracy %.3f", len(labels), acc)
    coef = model.coef_[0]
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
        intercept=float(model.intercept_[0]),
        train_accuracy=acc,
    )
    if samples_emb is not None:
        clf.embedding_model = embedding_model
        clf.emb_mean = mean[n_pix:].tolist()
        clf.emb_std = std[n_pix:].tolist()
        clf.emb_coef = coef[n_pix:].tolist()
    return clf


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
