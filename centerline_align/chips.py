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
from __future__ import annotations

import numpy as np

from .geom import as_coords, densify


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
