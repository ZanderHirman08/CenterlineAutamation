"""Polyline helpers on plain (N, 2) coordinate arrays (no GIS library needed)."""
from __future__ import annotations

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
