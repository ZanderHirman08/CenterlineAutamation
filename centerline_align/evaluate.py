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
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geom import as_coords, cumlen, densify, interpolate

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
