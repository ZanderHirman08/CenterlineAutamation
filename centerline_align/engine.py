"""Processing steps shared by the command-line tool and the ArcGIS Pro toolbox.

Everything here works on plain numpy coordinate arrays and an imagery
*reader*, so it runs with only numpy + scipy. A reader needs:

* ``res`` - pixel size in CRS units
* ``has_nir`` - whether the 4th band read is near-infrared
* ``read(x0, y0, width_px, height_px) -> Chip`` - block with top-left corner (x0, y0)
* ``snap(x, y) -> (x, y)`` - nearest pixel corner at or left of x / at or above y
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .classifier import RoadClassifier, extract_samples, fit, sample_training_pixels
from .geom import as_coords, cumlen, densify, interpolate
from .refine import refine_line
from .topology import snap_nodes

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


def train_classifier(reader, cells: list[list], lines_near, ccfg, embedder=None,
                     seed: int = 0) -> RoadClassifier:
    """Train from tiles centred on random road points in each training cell.

    cells: for each training grid, the road coords inside it.
    lines_near(bounds) -> road coords intersecting ``bounds`` (for labelling).
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
    pix_s, emb_s, lab_s = [], [], []
    for c in centres:
        chip = read_at(c)
        bx0, by0, bx1, by1 = chip.bounds
        m = ccfg.negative_max_dist
        near = lines_near((bx0 - m, by0 - m, bx1 + m, by1 + m))
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
    return fit(np.concatenate(pix_s), np.concatenate(emb_s) if emb_s else None,
               np.concatenate(lab_s), feature_mode=ccfg.features, has_nir=reader.has_nir,
               band_lo=lo, band_hi=hi,
               embedding_model=ccfg.embedding_model if embedder is not None else None)


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


def reconnect(records: list[dict], rcfg) -> list[np.ndarray]:
    """Snapped coords for refined records (see ``topology.snap_nodes``)."""
    if not records:
        return []
    ends = np.array([[r["o_sx"], r["o_sy"], r["o_ex"], r["o_ey"]] for r in records],
                    float).reshape(-1, 2, 2)
    return snap_nodes(ends, [np.asarray(r["coords"], float) for r in records],
                      np.array([r["conf"] for r in records], float),
                      rcfg.node_snap_tolerance, rcfg.node_taper)
