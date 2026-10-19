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

from .classifier import (RoadClassifier, distance_to_roads, extract_samples, fit,
                         logistic_regression, sample_training_pixels)
from .features import pixel_feature_names, pixel_features, scale_bands
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
                     seed: int = 0, on_chip=None, trusted_near=None) -> RoadClassifier:
    """Train from tiles centred on random road points in each training cell.

    cells: for each training grid, the road coords inside it.
    lines_near(bounds) -> road coords intersecting ``bounds`` (for labelling).
    on_chip(k, chip, lines) is called for every training tile (for diagnostics).
    trusted_near(bounds) -> ([(coords, half_width)], [coords]): second pass, label
    from an earlier run's trusted lines (see ``sample_training_pixels``).
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
        trusted = trusted_near(chip.bounds) if trusted_near is not None else None
        rows, cols, labels = sample_training_pixels(chip, near, ccfg, rng, per_tile, trusted)
        if len(labels) == 0:
            continue
        pix, emb = extract_samples(chip, rows, cols, reader.has_nir, lo, hi, embedder,
                                   lidar=getattr(reader, "has_lidar", False))
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
              embedding_model=ccfg.embedding_model if embedder is not None else None,
              lidar=getattr(reader, "has_lidar", False))
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


def find_junctions(lines, tol: float = 1.5) -> np.ndarray:
    """Intersection points of a set of lines: ends shared by 3 or more lines, or an end
    that touches the middle of another line (a T where the through road isn't split).
    Two lines simply continuing each other (a grid-edge cut) are not a junction."""
    lines = [as_coords(c) for c in lines if len(c) >= 2]
    if len(lines) < 2:
        return np.zeros((0, 2))
    dense, owner = [], []
    for k, c in enumerate(lines):
        p = densify(c, tol)
        dense.append(p)
        owner.append(np.full(len(p), k))
    dense, owner = np.concatenate(dense), np.concatenate(owner)
    tree = cKDTree(dense)
    ends = np.array([e for c in lines for e in (c[0], c[-1])])
    end_tree = cKDTree(ends)
    out = []
    for k, c in enumerate(lines):
        for e in (c[0], c[-1]):
            touching = set(owner[tree.query_ball_point(e, tol)]) - {k}
            if len(touching) >= 2:
                out.append(e)
            elif len(touching) == 1:
                # the other line touches here at its own end (continuation) or mid-way (T)
                if len(end_tree.query_ball_point(e, tol * 2)) < 2:
                    out.append(e)
    out += _crossings(lines, tol)
    if not out:
        return np.zeros((0, 2))
    out = np.array(out)
    # one point per junction
    keep = np.ones(len(out), bool)
    t = cKDTree(out)
    for i, j in sorted(t.query_pairs(tol * 2)):
        if keep[i]:
            keep[j] = False
    return out[keep]


def _crossings(lines, tol: float, seg: float = 10.0) -> list:
    """Points where two different lines cross mid-way (neither is split there)."""
    a, b, owner = [], [], []
    for k, c in enumerate(lines):
        c = densify(c, seg)
        a.append(c[:-1])
        b.append(c[1:])
        owner.append(np.full(len(c) - 1, k))
    a, b, owner = np.concatenate(a), np.concatenate(b), np.concatenate(owner)
    out = []
    for i, j in cKDTree((a + b) / 2).query_pairs(seg * 1.01):
        if owner[i] == owner[j]:
            continue
        p, r = a[i], b[i] - a[i]
        q, s_ = a[j], b[j] - a[j]
        den = r[0] * s_[1] - r[1] * s_[0]
        if abs(den) < 1e-12:
            continue
        t = ((q[0] - p[0]) * s_[1] - (q[1] - p[1]) * s_[0]) / den
        u = ((q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]) / den
        if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            x = p + t * r
            # where a line ends, the end-point rules above decide
            if min(np.hypot(*(x - e)) for k in (owner[i], owner[j])
                   for e in (lines[k][0], lines[k][-1])) > tol:
                out.append(x)
    return out


# what the keep/move model looks at for each refined line
MOVE_FEATURES = ("shift_avg", "shift_max", "conf", "pct_valid", "width_med")
MOVE_MIN_GAIN = 0.5   # a move "helps" when it brings the line at least this much closer


def move_probability(model: dict, rec: dict) -> float:
    """Chance (0-1) that applying a refined line's move brings it closer to where a
    person would put it, from the line's own QA values."""
    x = np.array([float(rec[k]) for k in MOVE_FEATURES])
    z = float(np.dot((x - np.asarray(model["mean"])) / np.asarray(model["std"]),
                     np.asarray(model["coef"])) + model["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_move_model(recs: list[dict], gains, min_rows: int = 20):
    """Learn when a move is worth applying.

    recs: refined records whose move was applied (``moved`` = 1); gains: for each,
    how much closer the move brought the line to the hand-checked line (negative =
    farther). Fits a logistic regression for "helps by at least MOVE_MIN_GAIN", then
    picks the probability threshold that maximises the total gain on these lines.
    Returns None when there are too few lines of either kind to learn from.
    """
    gains = np.asarray(gains, float)
    y = (gains >= MOVE_MIN_GAIN).astype(float)
    if len(recs) < min_rows or y.sum() < 5 or (1 - y).sum() < 5:
        return None
    X = np.array([[float(r[k]) for k in MOVE_FEATURES] for r in recs])
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-9] = 1.0
    coef, intercept = logistic_regression((X - mean) / std, y, C=1.0)
    model = dict(features=list(MOVE_FEATURES), mean=mean.tolist(), std=std.tolist(),
                 coef=coef.tolist(), intercept=float(intercept), threshold=0.5)
    p = np.array([move_probability(model, r) for r in recs])
    cands = np.unique(np.concatenate([p, [0.0, 1.01]]))
    total = [gains[p >= t].sum() for t in cands]
    model["threshold"] = float(cands[int(np.argmax(total))])
    model["rows"] = len(recs)
    model["gain_all"] = float(gains.sum())          # applying every move
    model["gain_chosen"] = float(max(total))        # applying the chosen ones
    return model


def refine_pieces(pmap: ProbabilityMap, pieces, rcfg, step: float, keep_stations=False,
                  move_model: dict | None = None):
    """Refine (src_id, coords) pieces. Returns (records, stations) as plain dicts.

    move_model: optional keep/move model (``fit_move_model``); a move it judges
    unlikely to help is not applied (the original line is kept, reason
    ``move_unlikely_to_help``)."""
    records, stations = [], []
    junctions = find_junctions([c for _, c in pieces])
    for src_id, coords in pieces:
        c = as_coords(coords)
        if len(c) < 2:
            continue
        r = refine_line(c, pmap.sample, rcfg, step=step, valid=pmap.sample_valid,
                        keep_stations=keep_stations, junctions=junctions)
        records.append(dict(
            src_id=src_id, shift_avg=round(r.shift_avg, 2), shift_max=round(r.shift_max, 2),
            conf=r.confidence, pct_valid=r.pct_valid, width_med=round(r.width_med, 1),
            review=int(r.needs_review), reason=";".join(r.reasons), moved=int(r.moved),
            o_sx=float(c[0, 0]), o_sy=float(c[0, 1]), o_ex=float(c[-1, 0]), o_ey=float(c[-1, 1]),
            coords=r.coords,
        ))
        rec = records[-1]
        if move_model is not None and rec["moved"]:
            p = move_probability(move_model, rec)
            rec["move_p"] = round(p, 3)
            if p < move_model["threshold"]:
                rec["coords"], rec["moved"] = c.copy(), 0
                rec["reason"] = ";".join(r.reasons + ["move_unlikely_to_help"])
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
