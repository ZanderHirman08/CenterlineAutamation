"""End-to-end run: train the classifier, refine roads grid by grid, merge outputs.

Each grid is written to ``<output>/grids/<grid_id>.gpkg`` as soon as it is
done, so a long county run can be stopped and resumed. The merge step joins
all finished grids, reconnects shared nodes and writes the deliverables.
"""
from __future__ import annotations

import csv
import logging
import math
import re
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import ndimage
from shapely.geometry import LineString, Point, box

from .classifier import RoadClassifier, extract_samples, fit, sample_training_pixels
from .config import Config
from .imagery import Chip, ImageryReader
from .refine import refine_line
from .topology import snap_nodes

log = logging.getLogger(__name__)

PAD_PX = 64  # context pixels read around each tile so blurs/embeddings have no edge effects


# --------------------------------------------------------------------------- vectors

def load_vectors(cfg: Config, crs):
    ic = cfg.inputs
    roads = gpd.read_file(ic.roads, layer=ic.roads_layer)
    grids = gpd.read_file(ic.grids, layer=ic.grids_layer)
    for name, gdf, fld in (("roads", roads, ic.road_id_field), ("grids", grids, ic.grid_id_field)):
        if fld not in gdf.columns:
            raise ValueError(f"'{fld}' not found in {name}; fields are {list(gdf.columns)}")
        if gdf.crs is None:
            raise ValueError(f"The {name} file has no CRS (.prj missing?)")
    roads = roads.to_crs(crs)
    grids = grids.to_crs(crs)
    roads = roads[[ic.road_id_field, "geometry"]].rename(columns={ic.road_id_field: "src_id"})
    roads["geometry"] = shapely.force_2d(roads.geometry.values)
    roads = roads.explode(index_parts=False)
    roads = roads[roads.geom_type == "LineString"]
    roads = roads[roads.length > 0].reset_index(drop=True)
    grids = grids[[ic.grid_id_field, "geometry"]].rename(columns={ic.grid_id_field: "grid_id"})
    grids = grids.reset_index(drop=True)
    if cfg.grid_ids is not None:
        wanted = {str(g) for g in cfg.grid_ids}
        missing = wanted - set(grids.grid_id.astype(str))
        if missing:
            raise ValueError(f"grid_ids not found in grid file: {sorted(missing)}")
    return roads, grids


def clip_to_grid(roads: gpd.GeoDataFrame, grids: gpd.GeoDataFrame, gi: int) -> list[tuple]:
    """(src_id, LineString) pieces of road inside grid ``gi``.

    A piece lying exactly on a shared grid edge belongs to the lowest-index grid,
    so it is processed once.
    """
    cell = grids.geometry.iloc[gi]
    out = []
    for ri in roads.sindex.query(cell, predicate="intersects"):
        piece = roads.geometry.iloc[ri].intersection(cell)
        for part in getattr(piece, "geoms", [piece]):
            if part.geom_type != "LineString" or part.length < 1e-6:
                continue
            mid = part.interpolate(0.5, normalized=True)
            owners = grids.sindex.query(mid, predicate="intersects")
            if len(owners) and owners.min() != gi:
                continue
            out.append((roads.src_id.iloc[ri], part))
    return out


# --------------------------------------------------------------------------- probability

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


def region_probability(reader: ImageryReader, clf: RoadClassifier, embedder, bounds,
                       lines, cfg: Config) -> ProbabilityMap:
    res = reader.res
    margin = cfg.refine.search_half_width + cfg.refine.max_shift + 4 * res
    minx, miny, maxx, maxy = bounds
    x0 = math.floor((minx - margin) / res) * res
    y0 = math.ceil((maxy + margin) / res) * res
    width = int(math.ceil((maxx + margin - x0) / res))
    height = int(math.ceil((y0 - (miny - margin)) / res))
    pmap = ProbabilityMap(x0, y0, res, (height, width))

    tile = cfg.classifier.embedding_tile if embedder is not None else 1024
    core = max(64, tile - 2 * PAD_PX) if embedder is not None else tile
    tree = shapely.STRtree(lines)
    reach = cfg.refine.search_half_width + cfg.refine.max_shift
    for r0 in range(0, height, core):
        for c0 in range(0, width, core):
            h, w = min(core, height - r0), min(core, width - c0)
            tx0, ty0 = x0 + c0 * res, y0 - r0 * res
            tbox = box(tx0, ty0 - h * res, tx0 + w * res, ty0)
            if not len(tree.query(tbox, predicate="dwithin", distance=reach)):
                continue  # nothing to refine near this tile
            chip = reader.read(tx0 - PAD_PX * res, ty0 + PAD_PX * res,
                               w + 2 * PAD_PX, h + 2 * PAD_PX)
            p = clf.predict(chip, embedder)[PAD_PX:PAD_PX + h, PAD_PX:PAD_PX + w]
            v = chip.valid[PAD_PX:PAD_PX + h, PAD_PX:PAD_PX + w]
            pmap.prob[r0:r0 + h, c0:c0 + w] = np.round(p * 255).astype(np.uint8)
            pmap.valid[r0:r0 + h, c0:c0 + w] = v
    return pmap


# --------------------------------------------------------------------------- training

def make_embedder(cfg: Config):
    if cfg.classifier.features == "pixel":
        return None
    if cfg.classifier.features != "embedding":
        raise ValueError("classifier.features must be 'pixel' or 'embedding'")
    from .features import EmbeddingExtractor
    return EmbeddingExtractor(cfg.classifier.embedding_model, cfg.classifier.embedding_device)


def train_classifier(reader: ImageryReader, roads, grids, cfg: Config, embedder,
                     seed: int = 0) -> RoadClassifier:
    cc = cfg.classifier
    rng = np.random.default_rng(seed)
    with_roads = [gi for gi in range(len(grids))
                  if len(roads.sindex.query(grids.geometry.iloc[gi], predicate="intersects"))]
    if cfg.grid_ids is not None:
        wanted = {str(g) for g in cfg.grid_ids}
        with_roads = [gi for gi in with_roads if str(grids.grid_id.iloc[gi]) in wanted]
    if not with_roads:
        raise ValueError("No grid contains any road; check that the files overlap")
    pick = np.unique(np.linspace(0, len(with_roads) - 1, min(cc.training_grids, len(with_roads)))
                     .round().astype(int))
    train_grids = [with_roads[i] for i in pick]
    log.info("Training on %d grid(s): %s", len(train_grids),
             ", ".join(str(grids.grid_id.iloc[g]) for g in train_grids))

    # Pass 1: band stretch from a coarse look at each training grid.
    vals = []
    for gi in train_grids:
        minx, miny, maxx, maxy = grids.geometry.iloc[gi].bounds
        coarse = max(reader.res, (maxx - minx) / 512)
        n = int(math.ceil((maxx - minx) / coarse)), int(math.ceil((maxy - miny) / coarse))
        chip = reader.read(minx, maxy, n[0], n[1], res=coarse)
        if chip.valid.any():
            vals.append(chip.data[:, chip.valid])
    if not vals:
        raise ValueError("The training grids have no imagery; check imagery coverage and CRS")
    allv = np.concatenate(vals, axis=1)
    lo = np.percentile(allv, 2, axis=1)
    hi = np.percentile(allv, 98, axis=1)

    # Pass 2: labelled samples from tiles centred on random road points.
    tile = cc.embedding_tile if embedder is not None else 1024
    tiles_per_grid = 4
    n_tiles = len(train_grids) * tiles_per_grid
    per_tile = max(50, cc.samples_per_class // n_tiles)
    pix_s, emb_s, lab_s = [], [], []
    half = tile * reader.res / 2
    for gi in train_grids:
        cell = grids.geometry.iloc[gi]
        idx = roads.sindex.query(cell, predicate="intersects")
        lines = roads.geometry.iloc[idx]
        for _ in range(tiles_per_grid):
            li = rng.integers(len(lines))
            ln = lines.iloc[li]
            centre = ln.interpolate(rng.random(), normalized=True)
            chip = reader.read(centre.x - half, centre.y + half, tile, tile)
            near_box = box(centre.x - half - cc.negative_max_dist, centre.y - half - cc.negative_max_dist,
                           centre.x + half + cc.negative_max_dist, centre.y + half + cc.negative_max_dist)
            near = roads.geometry.iloc[roads.sindex.query(near_box, predicate="intersects")]
            rows, cols, labels = sample_training_pixels(chip, list(near), cc, rng, per_tile)
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
               np.concatenate(lab_s), feature_mode=cc.features, has_nir=reader.has_nir,
               band_lo=lo, band_hi=hi, embedding_model=cc.embedding_model if embedder else None)


# --------------------------------------------------------------------------- per grid

def safe_name(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def process_grid(reader, clf, embedder, roads, grids, gi, cfg: Config, out_dir: Path) -> dict:
    grid_id = grids.grid_id.iloc[gi]
    t0 = time.time()
    pieces = clip_to_grid(roads, grids, gi)
    path = out_dir / f"{safe_name(grid_id)}.gpkg"
    summary = dict(grid_id=grid_id, pieces=len(pieces), length=0.0, review=0,
                   review_length=0.0, shift_avg=0.0, status="ok")
    if not pieces:
        summary["status"] = "no_roads"
        _empty_grid(path, roads.crs)
        return summary

    lines = [p[1] for p in pieces]
    pmap = region_probability(reader, clf, embedder, grids.geometry.iloc[gi].bounds, lines, cfg)
    records, station_rows = [], []
    for src_id, ln in pieces:
        r = refine_line(ln, pmap.sample, cfg.refine, step=reader.res, valid=pmap.sample_valid,
                        keep_stations=cfg.output.write_stations)
        (sx, sy), (ex, ey) = ln.coords[0][:2], ln.coords[-1][:2]
        records.append(dict(
            src_id=src_id, grid_id=grid_id, shift_avg=round(r.shift_avg, 2),
            shift_max=round(r.shift_max, 2), conf=r.confidence, pct_valid=r.pct_valid,
            width_med=round(r.width_med, 1), review=int(r.needs_review),
            reason=";".join(r.reasons), o_sx=sx, o_sy=sy, o_ex=ex, o_ey=ey,
            geometry=r.geometry,
        ))
        if r.stations is not None:
            s = r.stations
            for i in range(len(s["x"])):
                station_rows.append(dict(src_id=src_id, grid_id=grid_id,
                                         raw=s["raw"][i], final=s["final"][i], conf=s["conf"][i],
                                         width=s["width"][i], used=int(s["used"][i]),
                                         geometry=Point(s["x"][i], s["y"][i])))
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=roads.crs)
    tmp = path.with_suffix(".tmp.gpkg")
    gdf.to_file(tmp, layer="lines", driver="GPKG")
    if station_rows:
        gpd.GeoDataFrame(station_rows, geometry="geometry", crs=roads.crs).to_file(
            tmp, layer="stations", driver="GPKG")
    tmp.replace(path)
    if cfg.output.write_probability:
        _write_probability(pmap, out_dir / f"{safe_name(grid_id)}_prob.tif", roads.crs)

    lengths = gdf.length
    summary.update(length=round(float(lengths.sum()), 1),
                   review=int(gdf.review.sum()),
                   review_length=round(float(lengths[gdf.review == 1].sum()), 1),
                   shift_avg=round(float(np.average(gdf.shift_avg, weights=lengths)), 2),
                   seconds=round(time.time() - t0, 1))
    return summary


def _empty_grid(path: Path, crs):
    gpd.GeoDataFrame({"src_id": [], "grid_id": []}, geometry=[], crs=crs).to_file(
        path, layer="lines", driver="GPKG")


def _write_probability(pmap: ProbabilityMap, path: Path, crs):
    import rasterio
    from rasterio.transform import from_origin

    h, w = pmap.prob.shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8",
                       crs=crs, transform=from_origin(pmap.x0, pmap.y0, pmap.res, pmap.res),
                       compress="deflate", tiled=True) as dst:
        dst.write(pmap.prob, 1)


# --------------------------------------------------------------------------- merge

FINAL_FIELDS = ["src_id", "grid_id", "shift_avg", "shift_max", "conf", "pct_valid",
                "width_med", "review", "reason"]


def merge(grid_files: list[Path], cfg: Config, crs) -> gpd.GeoDataFrame:
    parts = [gpd.read_file(p, layer="lines") for p in grid_files]
    parts = [p for p in parts if len(p)]
    if not parts:
        return gpd.GeoDataFrame(columns=FINAL_FIELDS + ["geometry"], geometry="geometry", crs=crs)
    gdf = pd.concat(parts, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=parts[0].crs)
    ends = gdf[["o_sx", "o_sy", "o_ex", "o_ey"]].to_numpy(float).reshape(-1, 2, 2)
    snapped = snap_nodes(ends, list(gdf.geometry), gdf.conf.to_numpy(float),
                         cfg.refine.node_snap_tolerance, cfg.refine.node_taper)
    gdf["geometry"] = snapped
    return gdf[FINAL_FIELDS + ["geometry"]]


# --------------------------------------------------------------------------- run

def run(cfg: Config, resume: bool = True, train_only: bool = False) -> dict:
    out = Path(cfg.output.folder)
    grid_dir = out / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    reader = ImageryReader(cfg)
    try:
        roads, grids = load_vectors(cfg, reader.crs)
        log.info("%d road pieces, %d grids, working CRS %s, pixel size %.3g",
                 len(roads), len(grids), reader.crs.to_string(), reader.res)
        embedder = make_embedder(cfg)

        model_path = Path(cfg.classifier.model_path or out / "road_classifier.json")
        if model_path.exists() and (resume or cfg.classifier.model_path):
            clf = RoadClassifier.load(model_path)
            if clf.feature_mode != cfg.classifier.features:
                raise ValueError(f"{model_path} was trained with features="
                                 f"'{clf.feature_mode}'; delete it or change the config")
            log.info("Using saved classifier %s", model_path)
        else:
            clf = train_classifier(reader, roads, grids, cfg, embedder)
            clf.save(model_path)
            log.info("Saved classifier to %s", model_path)
        if train_only:
            return {"model": str(model_path)}

        selected = range(len(grids))
        if cfg.grid_ids is not None:
            wanted = {str(g) for g in cfg.grid_ids}
            selected = [gi for gi in selected if str(grids.grid_id.iloc[gi]) in wanted]

        summaries = []
        files = []
        for k, gi in enumerate(selected, 1):
            grid_id = grids.grid_id.iloc[gi]
            path = grid_dir / f"{safe_name(grid_id)}.gpkg"
            files.append(path)
            if resume and path.exists():
                log.info("[%d/%d] grid %s already done, skipping", k, len(selected), grid_id)
                continue
            try:
                s = process_grid(reader, clf, embedder, roads, grids, gi, cfg, grid_dir)
                log.info("[%d/%d] grid %s: %d pieces, %d to review, avg shift %.2f",
                         k, len(selected), grid_id, s["pieces"], s["review"], s["shift_avg"])
            except Exception as e:  # keep going; one bad grid shouldn't kill an overnight run
                log.exception("[%d/%d] grid %s failed", k, len(selected), grid_id)
                s = dict(grid_id=grid_id, status=f"error: {e}")
                files.remove(path)
            summaries.append(s)
    finally:
        reader.close()

    done = [f for f in files if f.exists()]
    final = merge(done, cfg, roads.crs)
    written = _write_outputs(final, cfg, out)
    _write_summary(summaries, out / "grid_summary.csv")
    log.info("Wrote %d lines (%d flagged for review) to %s",
             len(final), int(final.review.sum()) if len(final) else 0, ", ".join(written))
    return {"lines": len(final), "outputs": written, "grids": summaries}


def _write_outputs(gdf: gpd.GeoDataFrame, cfg: Config, out: Path) -> list[str]:
    written = []
    if cfg.output.write_geopackage:
        p = out / f"{cfg.output.name}.gpkg"
        gdf.to_file(p, layer=cfg.output.name, driver="GPKG")
        written.append(str(p))
    if cfg.output.write_shapefile:
        p = out / f"{cfg.output.name}.shp"
        shp = gdf.rename(columns={c: c.upper() for c in FINAL_FIELDS})
        shp.to_file(p, driver="ESRI Shapefile")
        written.append(str(p))
    return written


def _write_summary(rows: list[dict], path: Path):
    if not rows:
        return
    keys = ["grid_id", "status", "pieces", "length", "review", "review_length", "shift_avg", "seconds"]
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)
