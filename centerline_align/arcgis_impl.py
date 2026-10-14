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
from .classifier import RoadClassifier
from .config import ClassifierConfig, RefineConfig
from .engine import pick_spread, reconnect, refine_pieces, region_probability, train_classifier
from .geom import cumlen, interpolate

TILE_EXTS = (".tif", ".tiff", ".img", ".jp2")


class ArcTileReader:
    """Reads blocks from one raster or a folder of same-grid tiles."""

    def __init__(self, arcpy, source: str, bands: list[int]):
        self.arcpy = arcpy
        paths = self._find(source)
        self.tiles = []
        for p in paths:
            r = arcpy.Raster(p)
            e = r.extent
            self.tiles.append(dict(path=p, xmin=e.XMin, ymin=e.YMin, xmax=e.XMax, ymax=e.YMax,
                                   res=float(r.meanCellWidth), count=int(r.bandCount),
                                   sr=r.spatialReference))
        first = self.tiles[0]
        self.sr = first["sr"]
        self.res = first["res"]
        if self.sr is None or self.sr.type != "Projected":
            raise ValueError("The imagery must be in a projected coordinate system "
                             "(e.g. State Plane), not geographic.")
        for t in self.tiles[1:]:
            if abs(t["res"] - self.res) > 1e-6 * self.res:
                raise ValueError(f"{t['path']} has cell size {t['res']}, expected {self.res}")
            if t["sr"].name != self.sr.name:
                raise ValueError(f"{t['path']} is in {t['sr'].name}, expected {self.sr.name}")
        self.bands = list(bands)
        if max(self.bands) > first["count"]:
            raise ValueError(f"Band {max(self.bands)} requested but the imagery has "
                             f"{first['count']} band(s)")
        self.has_nir = len(self.bands) == 4
        self.gx, self.gy = first["xmin"], first["ymax"]
        self._ext = np.array([[t["xmin"], t["ymin"], t["xmax"], t["ymax"]] for t in self.tiles])

    def _find(self, source: str) -> list[str]:
        if os.path.isdir(source) and not source.lower().endswith(".gdb"):
            found = []
            for root, _, files in os.walk(source):
                found += [os.path.join(root, f) for f in files if f.lower().endswith(TILE_EXTS)]
            if not found:
                raise ValueError(f"No .tif/.img/.jp2 tiles found in {source}")
            return sorted(found)
        return [source]

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
            arr = self.arcpy.RasterToNumPyArray(t["path"], corner, nc, nr, 0)
            arr = arr[None] if arr.ndim == 2 else arr
            block = arr[[b - 1 for b in self.bands]].astype(np.float32)
            data[:, r0:r1, c0:c1] = block
            valid[r0:r1, c0:c1] = block.any(axis=0)
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
        out_fc, bands, refine: RefineConfig, classifier: ClassifierConfig, fresh=False):
    msg = arcpy.AddMessage
    handler = _ArcHandler(arcpy)
    root = logging.getLogger("centerline_align")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        return _run(arcpy, msg, imagery, roads_fc, road_id_field, grids_fc, grid_id_field,
                    work_folder, out_fc, bands, refine, classifier, fresh)
    finally:
        root.removeHandler(handler)


def _run(arcpy, msg, imagery, roads_fc, road_id_field, grids_fc, grid_id_field, work_folder,
         out_fc, bands, rcfg, ccfg, fresh):
    os.makedirs(work_folder, exist_ok=True)
    grid_dir = os.path.join(work_folder, "grids")
    os.makedirs(grid_dir, exist_ok=True)

    reader = ArcTileReader(arcpy, imagery, bands)
    sr = reader.sr
    msg(f"Imagery: {len(reader.tiles)} raster(s), {sr.name}, cell size {reader.res:g} "
        f"{sr.linearUnitName}. All distance settings are in {sr.linearUnitName}.")
    msg(f"Bands used: {reader.bands} (NIR: {'yes' if reader.has_nir else 'no'})")

    roads = read_lines(arcpy, roads_fc, road_id_field, sr)
    grids = read_grids(arcpy, grids_fc, grid_id_field, sr)
    if not roads or not grids:
        raise ValueError("No roads or no grids to process (check selections / definition queries)")
    msg(f"{len(roads)} roads, {len(grids)} grids (only selected features are used)")

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

        clf = train_classifier(reader, cells, lines_near, ccfg, None)
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
                pmap = region_probability(reader, clf, None, cext, [c for _, c in pieces],
                                          rcfg, ccfg)
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
    return {"lines": len(records), "review": n_rev}


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
              ("WIDTH_MED", "DOUBLE"), ("REVIEW", "SHORT"), ("REASON", "TEXT")]


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
                           r["pct_valid"], r["width_med"], int(r["review"]), r["reason"], line])
