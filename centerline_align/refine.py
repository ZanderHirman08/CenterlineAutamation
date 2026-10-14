"""Move a road line to the middle of the pavement using cross-sections.

At regular stations along the existing line, the road-probability map is
sampled along a short line perpendicular to the road. The run of "road"
pixels nearest the current line gives the two pavement edges; their midpoint
is the measured centre. Offsets are cleaned along the road (outliers dropped,
gaps interpolated, smoothed) and the line is rebuilt from the shifted stations.
Where the imagery gives no usable answer (trees, shadows, no data) the line
keeps its original position and is flagged for review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString

from .config import RefineConfig

# Sampler: (x array, y array) -> probability array of the same shape, 0-1.
Sampler = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class RefineResult:
    geometry: LineString
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


def station_distances(line: LineString, spacing: float) -> np.ndarray:
    """Regular stations plus every original vertex, so corners are kept."""
    length = line.length
    n = max(2, int(np.ceil(length / spacing)) + 1)
    regular = np.linspace(0.0, length, n)
    vertex_d = np.array([line.project(_pt(c)) for c in line.coords])
    d = np.unique(np.concatenate([regular, vertex_d]))
    # drop stations that crowd a vertex
    keep = np.ones(len(d), bool)
    min_gap = spacing * 0.25
    for i in range(1, len(d)):
        if d[i] - d[i - 1] < min_gap and not np.isclose(d[i], length):
            keep[i] = False
    return d[keep]


def _pt(c):
    from shapely.geometry import Point
    return Point(c[0], c[1])


def _tangents(line: LineString, d: np.ndarray, h: float) -> np.ndarray:
    length = line.length
    a = np.array([line.interpolate(max(0.0, x - h)).coords[0] for x in d])
    b = np.array([line.interpolate(min(length, x + h)).coords[0] for x in d])
    t = b - a
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


def refine_line(line: LineString, sample: Sampler, cfg: RefineConfig, step: float,
                valid: Sampler | None = None, keep_stations: bool = False) -> RefineResult:
    """Refine one LineString. ``step`` is the cross-section sample spacing (pixel size)."""
    d = station_distances(line, cfg.station_spacing)
    pts = np.array([line.interpolate(x).coords[0][:2] for x in d])
    tan = _tangents(line, d, max(cfg.station_spacing * 0.5, step))
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

    new_pts = pts + normal * final[:, None]
    new_line = LineString(new_pts)
    if cfg.simplify_tolerance > 0:
        simplified = new_line.simplify(cfg.simplify_tolerance, preserve_topology=True)
        if not simplified.is_empty and len(simplified.coords) >= 2:
            new_line = simplified

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
        geometry=new_line,
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
