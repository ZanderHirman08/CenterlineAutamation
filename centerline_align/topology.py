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
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString


def snap_nodes(orig_ends: np.ndarray, new_lines: list[LineString], weights: np.ndarray,
               tolerance: float, taper: float = 50.0) -> list[LineString]:
    """orig_ends: (n, 2, 2) original start/end xy per piece. Returns adjusted lines."""
    n = len(new_lines)
    if n == 0:
        return []
    pts = orig_ends.reshape(-1, 2)
    groups = _components(len(pts), cKDTree(pts).query_pairs(tolerance))
    ends = np.array([[ln.coords[0][:2], ln.coords[-1][:2]] for ln in new_lines]).reshape(-1, 2)
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


def _end_dir(ln: LineString, which: int) -> np.ndarray:
    c = np.asarray(ln.coords)[:, :2]
    v = c[1] - c[0] if which == 0 else c[-1] - c[-2]
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.array([1.0, 0.0])


def _taper_move(ln: LineString, d0: np.ndarray, d1: np.ndarray, taper: float) -> LineString:
    """Shift the start by d0 and the end by d1, fading each to zero over ``taper``."""
    if not (d0.any() or d1.any()):
        return ln
    c = np.asarray(ln.coords)[:, :2]
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
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
    c = c + f0[:, None] * d0 + f1[:, None] * d1
    return LineString(c)


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
