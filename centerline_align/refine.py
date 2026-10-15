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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import ndimage

from .config import RefineConfig
from .geom import as_coords, cumlen, interpolate, simplify

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
