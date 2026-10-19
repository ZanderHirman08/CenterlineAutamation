"""ArcGIS Pro side: read imagery / roads / grids with arcpy, write the output.

Uses only what ships with ArcGIS Pro's default Python (arcpy, numpy, scipy),
so nothing has to be installed. Works with a Basic licence: a folder of tiles
is read tile by tile with ``RasterToNumPyArray``; no mosaic dataset is needed.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import numpy as np

from .chip import Chip
from .classifier import RoadClassifier, distance_to_roads
from .config import ClassifierConfig, RefineConfig
from .engine import (find_junctions, pick_spread, reconnect, refine_pieces,
                     region_probability, summarize, train_classifier)
from .evaluate import (SegmentIndex, cover_from_chip, eval_report, near_junctions,
                       sample_lines)
from .chips import self_crossings
from .geom import cumlen, interpolate

TILE_EXTS = (".tif", ".tiff", ".img", ".jp2", ".sid")


class ArcTileReader:
    """Reads blocks from one raster or a folder of same-grid tiles."""

    def __init__(self, arcpy, source: str, bands: list[int], nodata=0):
        self.arcpy = arcpy
        self.nodata = nodata
        paths = self._find(source)
        self.tiles = []
        self.skipped = []  # (path, reason): unreadable tiles, or ones that don't match the rest
        for p in paths:
            try:
                r = arcpy.Raster(p)
                e = r.extent
                t = dict(path=p, xmin=e.XMin, ymin=e.YMin, xmax=e.XMax, ymax=e.YMax,
                         res=float(r.meanCellWidth), count=int(r.bandCount),
                         sr=r.spatialReference)
            except Exception as err:
                msg = str(err).strip().splitlines()
                self.skipped.append((p, msg[0] if msg else "could not be read"))
                continue
            if self.tiles:
                ref = self.tiles[0]
                if abs(t["res"] - ref["res"]) > 1e-6 * ref["res"]:
                    self.skipped.append((p, f"cell size {t['res']}, expected {ref['res']}"))
                    continue
                if t["sr"] is None or t["sr"].name != ref["sr"].name:
                    name = t["sr"].name if t["sr"] is not None else "no coordinate system"
                    self.skipped.append((p, f"in {name}, expected {ref['sr'].name}"))
                    continue
            self.tiles.append(t)
        if not self.tiles:
            raise ValueError(f"None of the imagery in {source} could be read" + (
                f" (first problem: {self.skipped[0][0]}: {self.skipped[0][1]})"
                if self.skipped else ""))
        first = self.tiles[0]
        self.sr = first["sr"]
        self.res = first["res"]
        if self.sr is None or self.sr.type != "Projected":
            raise ValueError("The imagery must be in a projected coordinate system "
                             "(e.g. State Plane), not geographic.")
        self.bands = list(bands)
        if max(self.bands) > first["count"]:
            raise ValueError(f"Band {max(self.bands)} requested but the imagery has "
                             f"{first['count']} band(s)")
        self.has_nir = len(self.bands) == 4
        self.gx, self.gy = first["xmin"], first["ymax"]
        self._ext = np.array([[t["xmin"], t["ymin"], t["xmax"], t["ymax"]] for t in self.tiles])

    def _find(self, source: str) -> list[str]:
        # .gdb rasters and Cloud Raster Format (.crf) are folders on disk but one raster
        if os.path.isdir(source) and not source.lower().rstrip("\\/").endswith((".gdb", ".crf")):
            found = []
            for root, _, files in os.walk(source):
                found += [os.path.join(root, f) for f in files if f.lower().endswith(TILE_EXTS)]
            if not found:
                raise ValueError(f"No .tif/.img/.jp2 tiles found in {source}")
            return sorted(found)
        return [source]

    def coverage(self, ext) -> float:
        """Share (0-1) of a rectangle (xmin, ymin, xmax, ymax) covered by tile extents."""
        x0, y0, x1, y1 = ext
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            return 0.0
        e = self._ext
        w = np.clip(np.minimum(e[:, 2], x1) - np.maximum(e[:, 0], x0), 0, None)
        h = np.clip(np.minimum(e[:, 3], y1) - np.maximum(e[:, 1], y0), 0, None)
        return float(min(1.0, (w * h).sum() / area))  # tiles may overlap a little

    @property
    def extent(self):
        return (self._ext[:, 0].min(), self._ext[:, 1].min(),
                self._ext[:, 2].max(), self._ext[:, 3].max())

    def snap(self, x: float, y: float) -> tuple[float, float]:
        r = self.res
        return (self.gx + np.floor((x - self.gx) / r) * r,
                self.gy - np.floor((self.gy - y) / r) * r)

    def read(self, x0: float, y0: float, width_px: int, height_px: int) -> Chip:
        r = self.res
        x1, y1 = x0 + width_px * r, y0 - height_px * r
        data = np.zeros((len(self.bands), height_px, width_px), np.float32)
        valid = np.zeros((height_px, width_px), bool)
        e = self._ext
        hit = np.flatnonzero((e[:, 0] < x1) & (e[:, 2] > x0) & (e[:, 1] < y0) & (e[:, 3] > y1))
        for i in hit:
            t = self.tiles[i]
            left, right = max(x0, t["xmin"]), min(x1, t["xmax"])
            top, bottom = min(y0, t["ymax"]), max(y1, t["ymin"])
            c0, c1 = int(round((left - x0) / r)), int(round((right - x0) / r))
            r0, r1 = int(round((y0 - top) / r)), int(round((y0 - bottom) / r))
            nc, nr = c1 - c0, r1 - r0
            if nc <= 0 or nr <= 0:
                continue
            # a point a quarter-cell inside the block's lower-left cell, so the
            # read starts on that cell however the corner is snapped
            corner = self.arcpy.Point(x0 + c0 * r + r / 4, y0 - r1 * r + r / 4)
            arr = self.arcpy.RasterToNumPyArray(t["path"], corner, nc, nr, self.nodata)
            arr = arr[None] if arr.ndim == 2 else arr
            block = arr[[b - 1 for b in self.bands]].astype(np.float32)
            ok = (block.any(axis=0) if self.nodata == 0
                  else (block != self.nodata).all(axis=0))
            # overlapping tiles: a tile's empty edge never overwrites another's data
            data[:, r0:r1, c0:c1][:, ok] = block[:, ok]
            valid[r0:r1, c0:c1] |= ok
        return Chip(data=data, valid=valid, x0=x0, y0=y0, res=r)


# --------------------------------------------------------------------------- vectors

def read_lines(arcpy, fc, id_field, sr):
    """[(id, Polyline geometry, [(N,2) coords per part], (xmin, ymin, xmax, ymax))]"""
    out = []
    with arcpy.da.SearchCursor(fc, [id_field, "SHAPE@"], spatial_reference=sr) as cur:
        for rid, geom in cur:
            if geom is None:
                continue
            parts = _parts(geom)
            if parts:
                e = geom.extent
                out.append((rid, geom, parts, (e.XMin, e.YMin, e.XMax, e.YMax)))
    return out


def read_grids(arcpy, fc, id_field, sr):
    out = []
    with arcpy.da.SearchCursor(fc, [id_field, "SHAPE@"], spatial_reference=sr) as cur:
        for gid, geom in cur:
            if geom is not None:
                e = geom.extent
                out.append((gid, geom, (e.XMin, e.YMin, e.XMax, e.YMax)))
    return out


def _parts(geom) -> list[np.ndarray]:
    parts = []
    for part in geom:
        pts = [(p.X, p.Y) for p in part if p is not None]
        if len(pts) >= 2:
            parts.append(np.array(pts, float))
    return parts


def _overlaps(a, b) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def clip_to_grid(arcpy, roads, grids, gi, sr) -> list[tuple]:
    """(src_id, coords) pieces of road inside grid ``gi``; a piece on a shared grid
    edge belongs to the lowest-index grid so it is processed once."""
    _, cell, cext = grids[gi]
    out = []
    for rid, geom, _, rext in roads:
        if not _overlaps(rext, cext):
            continue
        piece = geom.intersect(cell, 2)
        if piece is None:
            continue
        for c in _parts(piece):
            s = cumlen(c)
            if s[-1] < 1e-6:
                continue
            mid = interpolate(c, s[-1] / 2, s)[0]
            pt = arcpy.PointGeometry(arcpy.Point(mid[0], mid[1]), sr)
            owner = min((k for k, (_, g, e) in enumerate(grids)
                         if e[0] <= mid[0] <= e[2] and e[1] <= mid[1] <= e[3]
                         and not g.disjoint(pt)), default=gi)
            if owner == gi:
                out.append((rid, c))
    return out


# --------------------------------------------------------------------------- run

class _ArcHandler(logging.Handler):
    def __init__(self, arcpy):
        super().__init__()
        self.arcpy = arcpy

    def emit(self, record):
        msg = self.format(record)
        if record.levelno >= logging.WARNING:
            self.arcpy.AddWarning(msg)
        else:
            self.arcpy.AddMessage(msg)


def _safe(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def run(arcpy, *, imagery, roads_fc, road_id_field, grids_fc, grid_id_field, work_folder,
        out_fc, bands, refine: RefineConfig, classifier: ClassifierConfig, fresh=False,
        diagnostics=False):
    msg = arcpy.AddMessage
    handler = _ArcHandler(arcpy)
    root = logging.getLogger("centerline_align")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        return _run(arcpy, msg, imagery, roads_fc, road_id_field, grids_fc, grid_id_field,
                    work_folder, out_fc, bands, refine, classifier, fresh, diagnostics)
    finally:
        root.removeHandler(handler)


def _run(arcpy, msg, imagery, roads_fc, road_id_field, grids_fc, grid_id_field, work_folder,
         out_fc, bands, rcfg, ccfg, fresh, diagnostics=False):
    os.makedirs(work_folder, exist_ok=True)
    grid_dir = os.path.join(work_folder, "grids")
    os.makedirs(grid_dir, exist_ok=True)

    reader = ArcTileReader(arcpy, imagery, bands)
    sr = reader.sr
    msg(f"Imagery: {len(reader.tiles)} raster(s), {sr.name}, cell size {reader.res:g} "
        f"{sr.linearUnitName}. All distance settings are in {sr.linearUnitName}.")
    msg(f"Bands used: {reader.bands} (NIR: {'yes' if reader.has_nir else 'no'})")
    if reader.skipped:
        arcpy.AddWarning(f"{len(reader.skipped)} imagery file(s) were skipped:\n" + "\n".join(
            f"  {p}: {why}" for p, why in reader.skipped[:15])
            + (f"\n  ... and {len(reader.skipped) - 15} more" if len(reader.skipped) > 15 else ""))

    roads = read_lines(arcpy, roads_fc, road_id_field, sr)
    grids = read_grids(arcpy, grids_fc, grid_id_field, sr)
    if not roads or not grids:
        raise ValueError("No roads or no grids to process (check selections / definition queries)")
    msg(f"{len(roads)} roads, {len(grids)} grids (only selected features are used)")
    gaps = [(gid, reader.coverage(ext)) for gid, _, ext in grids]
    gaps = [(gid, f) for gid, f in gaps if f < 0.95]
    if gaps:
        arcpy.AddWarning(
            f"{len(gaps)} of {len(grids)} grids are not fully covered by the imagery tiles; "
            "roads there will be flagged no_imagery. Grid (covered %): " + ", ".join(
                f"{g} ({f:.0%})" for g, f in gaps[:60]) + (" ..." if len(gaps) > 60 else "")
            + ". Check the tile copy (copy_aoi_tiles.py reports which grids it couldn't cover).")

    model_path = os.path.join(work_folder, "road_classifier.json")
    if fresh:
        for f in os.listdir(grid_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(grid_dir, f))
        if os.path.exists(model_path):
            os.remove(model_path)
    if os.path.exists(model_path):
        clf = RoadClassifier.load(model_path)
        msg(f"Using saved classifier {model_path}")
    else:
        with_roads = [gi for gi in range(len(grids))
                      if any(_overlaps(r[3], grids[gi][2]) for r in roads)]
        train = pick_spread(with_roads, ccfg.training_grids)
        msg("Training the road classifier on grid(s): "
            + ", ".join(str(grids[g][0]) for g in train))
        cells = [[c for _, c in clip_to_grid(arcpy, roads, grids, gi, sr)] for gi in train]

        def lines_near(b):
            return [c for _, _, parts, e in roads if _overlaps(e, b) for c in parts]

        diag_dir = os.path.join(work_folder, "diagnostics")

        def on_chip(k, chip, lines):
            # first training tile of each of the first two grids
            if k % 4 == 0 and k < 8:
                os.makedirs(diag_dir, exist_ok=True)
                _save_training_diag(arcpy, diag_dir, k // 4 + 1, chip, lines, ccfg, sr)

        clf = train_classifier(reader, cells, lines_near, ccfg, on_chip=on_chip)
        msg(f"Diagnostic rasters for the first training tiles: {diag_dir}")
        clf.save(model_path)
        msg(f"Classifier saved to {model_path} (training accuracy {clf.train_accuracy:.3f})")

    arcpy.SetProgressor("step", "Aligning centerlines...", 0, len(grids), 1)
    files = []
    for k, (gid, cell, cext) in enumerate(grids):
        arcpy.SetProgressorLabel(f"Grid {gid} ({k + 1} of {len(grids)})")
        path = os.path.join(grid_dir, f"{_safe(gid)}.json")
        files.append(path)
        if os.path.exists(path):
            msg(f"[{k + 1}/{len(grids)}] grid {gid}: already done, skipping")
            arcpy.SetProgressorPosition()
            continue
        t0 = time.time()
        try:
            pieces = clip_to_grid(arcpy, roads, grids, k, sr)
            records = []
            if pieces:
                pmap = region_probability(reader, clf, cext, [c for _, c in pieces], rcfg)
                if diagnostics:
                    prob_dir = os.path.join(work_folder, "diagnostics")
                    os.makedirs(prob_dir, exist_ok=True)
                    h = pmap.prob.shape[0]
                    _save_raster(arcpy, pmap.prob, pmap.x0, pmap.y0 - h * pmap.res, pmap.res,
                                 os.path.join(prob_dir, f"road_prob_{_safe(gid)}.tif"), sr)
                records, _ = refine_pieces(pmap, pieces, rcfg, reader.res)
            for r in records:
                r["grid_id"] = gid
                r["coords"] = np.asarray(r["coords"]).tolist()
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(records, f, default=_json_default)
            os.replace(tmp, path)
            n_rev = sum(r["review"] for r in records)
            msg(f"[{k + 1}/{len(grids)}] grid {gid}: {len(records)} pieces, {n_rev} flagged "
                f"for review ({time.time() - t0:.0f}s)")
        except Exception as e:  # keep going; one bad grid shouldn't stop an overnight run
            arcpy.AddWarning(f"[{k + 1}/{len(grids)}] grid {gid} failed: {e}")
            files.remove(path)
        arcpy.SetProgressorPosition()
    arcpy.ResetProgressor()

    records = []
    for p in files:
        with open(p) as f:
            records += json.load(f)
    coords = reconnect(records, rcfg)
    _write_output(arcpy, out_fc, records, coords, sr, roads_fc, road_id_field)
    n_rev = sum(r["review"] for r in records)
    msg(f"Wrote {len(records)} lines to {out_fc}; {n_rev} flagged for review (REVIEW = 1).")
    for line in summarize(records):
        msg(line)
    return {"lines": len(records), "review": n_rev}


def _save_raster(arcpy, arr, x_ll, y_ll, res, path, sr):
    """Write a (rows, cols) or (bands, rows, cols) uint8 array as a GeoTIFF."""
    if arcpy.Exists(path):
        arcpy.management.Delete(path)
    ras = arcpy.NumPyArrayToRaster(np.ascontiguousarray(arr), arcpy.Point(x_ll, y_ll), res, res)
    ras.save(path)
    arcpy.management.DefineProjection(path, sr)


def _save_training_diag(arcpy, folder, n, chip, lines, ccfg, sr):
    """The imagery block the tool read, and where it put its training labels.

    labels_N.tif: 1 = labelled road (near a road line), 2 = labelled not-road,
    0 = not used. Lay both over the original tiles and your roads: if the
    image doesn't match the tiles underneath, the tool misread the imagery; if
    the 1-zone isn't on pavement, the roads and imagery don't line up.
    """
    h = chip.shape[0]
    x_ll, y_ll = chip.x0, chip.y0 - h * chip.res
    img = np.clip(chip.data, 0, 255).astype(np.uint8)
    _save_raster(arcpy, img, x_ll, y_ll, chip.res, os.path.join(folder, f"chip_{n}.tif"), sr)
    dist = distance_to_roads(chip, lines)
    lab = np.zeros(chip.shape, np.uint8)
    lab[dist <= ccfg.positive_half_width] = 1
    lab[(dist > ccfg.negative_min_dist) & (dist <= ccfg.negative_max_dist)] = 2
    lab[~chip.valid] = 0
    _save_raster(arcpy, lab, x_ll, y_ll, chip.res, os.path.join(folder, f"labels_{n}.tif"), sr)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


def _id_field_type(arcpy, fc, field):
    for f in arcpy.ListFields(fc):
        if f.name.lower() == field.lower():
            if f.type in ("Integer", "SmallInteger", "OID", "BigInteger"):
                return "LONG"
            if f.type in ("Double", "Single"):
                return "DOUBLE"
    return "TEXT"


OUT_FIELDS = [("SRC_ID", None), ("GRID_ID", "TEXT"), ("SHIFT_AVG", "DOUBLE"),
              ("SHIFT_MAX", "DOUBLE"), ("CONF", "DOUBLE"), ("PCT_VALID", "DOUBLE"),
              ("WIDTH_MED", "DOUBLE"), ("REVIEW", "SHORT"), ("REASON", "TEXT"),
              ("MOVED", "SHORT")]


def _write_output(arcpy, out_fc, records, coords, sr, roads_fc, road_id_field):
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    folder, name = os.path.split(out_fc)
    arcpy.management.CreateFeatureclass(folder, name, "POLYLINE", spatial_reference=sr)
    id_type = _id_field_type(arcpy, roads_fc, road_id_field)
    for fname, ftype in OUT_FIELDS:
        ftype = ftype or id_type
        if ftype == "TEXT":
            arcpy.management.AddField(out_fc, fname, ftype, field_length=100)
        else:
            arcpy.management.AddField(out_fc, fname, ftype)
    names = [f for f, _ in OUT_FIELDS] + ["SHAPE@"]
    with arcpy.da.InsertCursor(out_fc, names) as cur:
        for r, c in zip(records, coords):
            line = arcpy.Polyline(arcpy.Array([arcpy.Point(float(x), float(y)) for x, y in c]), sr)
            src = r["src_id"] if id_type != "TEXT" else str(r["src_id"])
            cur.insertRow([src, str(r["grid_id"]), r["shift_avg"], r["shift_max"], r["conf"],
                           r["pct_valid"], r["width_med"], int(r["review"]), r["reason"],
                           int(r.get("moved", 1)), line])


# --------------------------------------------------------------------------- evaluation

def evaluate(arcpy, *, truth_fc, truth_where, candidates, imagery=None, bands=(1, 2, 3, 4),
             grids_fc=None, out_points=None, spacing=5.0, max_dist=40.0, junction_dist=40.0):
    """Compare candidate line layers with checked truth lines; see ``evaluate.py``.

    candidates: [(display name, layer or path)]. Nothing is written to any input;
    ``out_points`` (optional) gets one point per check point with each layer's error.
    """
    msg = arcpy.AddMessage
    reader = ArcTileReader(arcpy, imagery, list(bands)) if imagery else None
    sr = reader.sr if reader is not None else arcpy.Describe(candidates[0][1]).spatialReference
    units = sr.linearUnitName or "units"

    every = []          # all truth-layer lines (for finding intersections)
    with arcpy.da.SearchCursor(truth_fc, ["SHAPE@"], spatial_reference=sr) as cur:
        for (g,) in cur:
            if g is not None:
                every += _parts(g)
    truth, oids = [], []
    try:
        with arcpy.da.SearchCursor(truth_fc, ["OID@", "SHAPE@"], where_clause=truth_where or None,
                                   spatial_reference=sr) as cur:
            for oid, g in cur:
                if g is not None:
                    for c in _parts(g):
                        truth.append(c)
                        oids.append(oid)
    except Exception as e:
        raise ValueError(f"Could not read the checked roads with the filter {truth_where!r}: "
                         f"{e}. Check the field name and value (text values need quotes, "
                         f"e.g. Validated = 'Yes').")
    if not truth:
        raise ValueError(f"No checked roads match {truth_where!r}. "
                         + _value_hint(arcpy, truth_fc, truth_where))
    reasons, temp, _ = road_rules(arcpy, truth_fc)
    small, crossings = loop_rules(list(zip(oids, truth)), temp)
    _summarize_rules(msg, reasons, small, set(oids))
    keep = [k for k, o in enumerate(oids) if o not in reasons and o not in small]
    truth, oids = [truth[k] for k in keep], [oids[k] for k in keep]
    if not truth:
        raise ValueError("Every checked road was left out by the manual rules")
    msg(f"{len(set(oids))} checked roads ({len(truth)} parts) from {truth_fc}"
        + (f" where {truth_where}" if truth_where else ""))

    pts, tans, line_idx = sample_lines(truth, spacing)
    if grids_fc:
        grids = [g for _, g, _ in read_grids(arcpy, grids_fc, "OID@", sr)]
        inside = _points_in_polygons(pts, grids)
        msg(f"{inside.sum()} of {len(pts)} check points are inside the work grids")
        pts, tans, line_idx = pts[inside], tans[inside], line_idx[inside]
    if not len(pts):
        raise ValueError("No check points to compare (are the checked roads inside the grids?)")

    errors = {}
    for name, fc in candidates:
        lines = [c for _, _, parts, _ in read_lines(arcpy, fc, "OID@", sr) for c in parts]
        errors[name] = SegmentIndex(lines).distances(pts, tans, max_dist)
        msg(f"{name}: {len(lines)} lines read")

    cover = np.array(["unknown"] * len(pts), dtype=object)
    if reader is not None:
        cover = _cover_classes(reader, pts, line_idx)
    junctions = find_junctions(every)
    if crossings:  # a road crossing itself is an intersection too
        junctions = np.vstack([junctions.reshape(-1, 2), np.array(crossings)])
    cover[near_junctions(pts, junctions, junction_dist)] = "intersection"
    for line in eval_report(errors, cover,
                            "ft" if "foot" in units.lower() or "feet" in units.lower() else ""):
        msg(line)
    if out_points:
        _write_check_points(arcpy, out_points, sr, pts, cover, np.asarray(oids)[line_idx], errors)
        msg(f"Check points written to {out_points}")
    return dict(errors=errors, cover=cover)


# --------------------------------------------------------------------------- manual rules

def _norm(name) -> str:
    return str(name).lower().replace("_", "").replace(" ", "")


def road_rules(arcpy, fc):
    """Roads the REVAMP manual says not to learn from or score on, and roundabouts.

    Returns ({oid: reason}, {oids with a TEMP route id (roundabouts)}, [fields used]):
    AbandonedRoad / PrivateRoad = 'Yes' (there may be no pavement) and RC_Comments
    mentioning construction (the imagery may not show the road as validated).
    Field names are matched ignoring case, underscores and spaces."""
    names = {_norm(f.name): f.name for f in arcpy.ListFields(fc)}
    yes_fields = [(names[k], why) for k, why in (("abandonedroad", "abandoned"),
                                                 ("privateroad", "private")) if k in names]
    comment = names.get("rccomments")
    route = names.get("routeid")
    fields = [f for f, _ in yes_fields] + [f for f in (comment, route) if f]
    reasons, temp = {}, set()
    if not fields:
        return reasons, temp, fields
    with arcpy.da.SearchCursor(fc, ["OID@"] + fields) as cur:
        for row in cur:
            oid, vals = int(row[0]), dict(zip(fields, row[1:]))
            for f, why in yes_fields:
                if str(vals[f]).strip().lower() == "yes":
                    reasons.setdefault(oid, why)
            if comment and "construct" in str(vals[comment] or "").lower():
                reasons.setdefault(oid, "construction")
            if route and "temp" in str(vals[route] or "").lower():
                temp.add(oid)
    return reasons, temp, fields


LOOP_MAX = 400.0   # a loop shorter than this (map units) is a cul-de-sac circle, not a road


def loop_rules(lines, temp, loop_max=LOOP_MAX):
    """(oids that are small loops - circular cul-de-sacs left as circles, manual 3.2 -
    and points where longer roads cross themselves). Roundabouts (``temp``) are kept
    and their crossings ignored. ``lines``: [(oid, coords)]."""
    small, points = set(), []
    for oid, c in lines:
        if oid in temp:
            continue
        for pt, loop_len in self_crossings(c):
            if loop_len < loop_max:
                small.add(oid)
            else:
                points.append(pt)
    return small, points


def _summarize_rules(msg, reasons, small, checked):
    counts = {}
    for o in checked:
        why = "cul-de-sac circle" if o in small else reasons.get(o)
        if why:
            counts[why] = counts.get(why, 0) + 1
    if counts:
        msg("Left out by the REVAMP manual rules: " + ", ".join(
            f"{n} {why}" for why, n in sorted(counts.items())))


def _value_hint(arcpy, fc, where) -> str:
    m = re.match(r'\s*"?([A-Za-z_]\w*)"?', where or "")
    if not m:
        return ""
    field = m.group(1)
    try:
        vals = set()
        with arcpy.da.SearchCursor(fc, [field]) as cur:
            for (v,) in cur:
                vals.add(v)
                if len(vals) > 10:
                    break
        return f"Values found in {field}: " + ", ".join(repr(v) for v in sorted(vals, key=str))
    except Exception:
        return f"Check that the field {field} exists."


def _points_in_polygons(pts, polys) -> np.ndarray:
    """Even-odd rule over all rings of every polygon; vectorised over points."""
    inside = np.zeros(len(pts), bool)
    for g in polys:
        e = g.extent
        cand = np.flatnonzero((pts[:, 0] >= e.XMin) & (pts[:, 0] <= e.XMax)
                              & (pts[:, 1] >= e.YMin) & (pts[:, 1] <= e.YMax))
        if not len(cand):
            continue
        x, y = pts[cand, 0], pts[cand, 1]
        hit = np.zeros(len(cand), bool)
        for ring in _rings(g):
            xa, ya = ring[:-1, 0], ring[:-1, 1]
            xb, yb = ring[1:, 0], ring[1:, 1]
            for x0, y0, x1, y1 in zip(xa, ya, xb, yb):
                crosses = ((y0 > y) != (y1 > y)) & (
                    x < (x1 - x0) * (y - y0) / ((y1 - y0) if y1 != y0 else 1e-12) + x0)
                hit ^= crosses
        inside[cand] |= hit
    return inside


def _rings(geom) -> list[np.ndarray]:
    rings = []
    try:
        for part in geom:
            ring = []
            for p in part:
                if p is None:  # ring separator inside a part
                    if len(ring) >= 3:
                        rings.append(np.array(ring + [ring[0]], float))
                    ring = []
                else:
                    ring.append((p.X, p.Y))
            if len(ring) >= 3:
                rings.append(np.array(ring + [ring[0]], float))
    except TypeError:
        pass
    if not rings:  # stand-ins without ring access
        e = geom.extent
        rings.append(np.array([(e.XMin, e.YMin), (e.XMin, e.YMax), (e.XMax, e.YMax),
                               (e.XMax, e.YMin), (e.XMin, e.YMin)]))
    return rings


def _cover_classes(reader, pts, line_idx, chunk=80, pad=12.0):
    cover = np.array(["unknown"] * len(pts), dtype=object)
    order = np.argsort(line_idx, kind="stable")
    for start in range(0, len(order), chunk):
        sel = order[start:start + chunk]
        p = pts[sel]
        x0, y0 = reader.snap(p[:, 0].min() - pad, p[:, 1].max() + pad)
        w = int(np.ceil((p[:, 0].max() + pad - x0) / reader.res)) + 1
        h = int(np.ceil((y0 - (p[:, 1].min() - pad)) / reader.res)) + 1
        if w * h > 4e7:  # very spread-out chunk: do it point by point
            for i in sel:
                cover[[i]] = _cover_classes(reader, pts[[i]], np.zeros(1, int))
            continue
        chip = reader.read(x0, y0, w, h)
        cover[sel] = cover_from_chip(chip, p, reader.has_nir)
    return cover


def _write_check_points(arcpy, out_fc, sr, pts, cover, oids, errors):
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    folder, name = os.path.split(out_fc)
    arcpy.management.CreateFeatureclass(folder, name, "POINT", spatial_reference=sr)
    arcpy.management.AddField(out_fc, "TRUTH_OID", "LONG")
    arcpy.management.AddField(out_fc, "COVER", "TEXT", field_length=20)
    fields = []
    for k, cand in enumerate(errors, 1):
        f = f"ERR_{k}"
        arcpy.management.AddField(out_fc, f, "DOUBLE", field_alias=f"Error: {cand}"[:255])
        fields.append(f)
    with arcpy.da.InsertCursor(out_fc, ["TRUTH_OID", "COVER"] + fields + ["SHAPE@"]) as cur:
        errs = [errors[k] for k in errors]
        for i, (x, y) in enumerate(pts):
            vals = [None if np.isnan(e[i]) else round(float(e[i]), 2) for e in errs]
            cur.insertRow([int(oids[i]), str(cover[i])] + vals
                          + [arcpy.PointGeometry(arcpy.Point(float(x), float(y)), sr)])
