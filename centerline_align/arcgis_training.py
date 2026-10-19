"""Collect training data while you edit, and train the classifier from it overnight.

Three toolbox buttons, used in this order:

1. Start of day (``snapshot``): remembers every road's geometry before you edit
   it. The first geometry seen for a road is kept as its *original* and never
   overwritten, so the morning snapshot of a road you edited yesterday doesn't
   replace its real original.
2. End of day (``collect``): every road you have checked (e.g.
   ``Validated = 'Yes'``) becomes a training road, with how far your edit moved
   it from its original. Roads you checked and left alone count too: they
   teach the tool when not to move a line.
3. Overnight (``train``): trains the road classifier on the checked roads (road
   labels on your lines, not on a reference layer that is itself a few feet
   off), then measures on grids held back from training how close the tool gets
   the original lines to your edits, and whether the new model beats the last.

Nothing is ever written to the road layer. Everything lives in the training
folder: ``originals.json``, ``training_roads.json`` and ``models\\``.
"""
from __future__ import annotations

import json
import logging
import os
import time
import zlib

import numpy as np

from .arcgis_impl import (ArcTileReader, _ArcHandler, _cover_classes, _overlaps, _parts,
                          _points_in_polygons, _value_hint, clip_to_grid, read_grids,
                          read_lines)
from .classifier import RoadClassifier
from .config import ClassifierConfig, RefineConfig
from .engine import (find_junctions, pick_spread, refine_pieces, region_probability,
                     train_classifier)
from .evaluate import SegmentIndex, eval_report, near_junctions, sample_lines
from .geom import cumlen, interpolate

ORIGINALS = "originals.json"
TRAINING = "training_roads.json"
MODELS = "models"
CURRENT_MODEL = "road_classifier.json"
HOLDOUT_EVERY = 5      # about 1 grid in 5 with checked roads is kept for testing
SAMPLE_SPACING = 5.0   # how often lines are compared (map units)
MATCH_DIST = 40.0      # an original farther than this from the edited line isn't its match


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _load(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save(folder, name, data):
    path = os.path.join(folder, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _round(parts) -> list:
    return [np.round(np.asarray(c, float), 3).tolist() for c in parts]


def _same(a, b, tol=0.01) -> bool:
    return len(a) == len(b) and all(
        np.shape(x) == np.shape(y) and np.allclose(x, y, atol=tol) for x, y in zip(a, b))


def _read_roads(arcpy, roads_fc, where=None, sr=None):
    """{oid: [(N, 2) coords per part]} of the road layer (honours selections)."""
    out = {}
    try:
        with arcpy.da.SearchCursor(roads_fc, ["OID@", "SHAPE@"], where_clause=where or None,
                                   spatial_reference=sr) as cur:
            for oid, g in cur:
                if g is not None:
                    parts = _parts(g)
                    if parts:
                        out[int(oid)] = parts
    except Exception as e:
        if not where:
            raise
        raise ValueError(f"Could not read the roads with the filter {where!r}: {e}. Check the "
                         f"field name and value (text values need quotes, e.g. "
                         f"Validated = 'Yes').")
    return out


def _with_log(arcpy, fn, *a, **k):
    handler = _ArcHandler(arcpy)
    root = logging.getLogger("centerline_align")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        return fn(*a, **k)
    finally:
        root.removeHandler(handler)


# --------------------------------------------------------------------------- 1. snapshot

def snapshot(arcpy, *, roads_fc, checked_where, training_folder, today=None):
    """Start of day: add every road not seen before to the originals (read-only on roads)."""
    msg = arcpy.AddMessage
    today = today or _today()
    os.makedirs(training_folder, exist_ok=True)
    sr = arcpy.Describe(roads_fc).spatialReference
    roads = _read_roads(arcpy, roads_fc)
    if not roads:
        raise ValueError(f"No roads read from {roads_fc} (check selections / definition queries)")
    checked = set(_read_roads(arcpy, roads_fc, checked_where)) if checked_where else set()
    data = _load(training_folder, ORIGINALS) or {"sr": sr.exportToString(), "roads": {},
                                                 "snapshots": []}
    known = data["roads"]
    new = 0
    for oid, parts in roads.items():
        if str(oid) not in known:
            # a road already checked before its first snapshot: its true original is unknown
            known[str(oid)] = {"day": today, "parts": _round(parts),
                               "checked_before": oid in checked}
            new += 1
    if today not in data["snapshots"]:
        data["snapshots"].append(today)
    _save(training_folder, ORIGINALS, data)
    msg(f"Snapshot {today}: {len(roads)} roads read, {new} new road(s) remembered "
        f"({len(known)} originals in total). Edit as usual; run 'End of Day' when you're done.")
    if new and new == len(known):
        pre = sum(1 for v in known.values() if v["checked_before"])
        if pre:
            arcpy.AddWarning(
                f"{pre} road(s) were already checked before this first snapshot, so their "
                "original geometry is unknown. They are still used for training, but not for "
                "measuring the tool.")
    return {"roads": len(roads), "new": new}


# --------------------------------------------------------------------------- 2. collect

def collect(arcpy, *, roads_fc, checked_where, training_folder, out_fc=None,
            min_change=1.0, today=None):
    """End of day: refresh the training roads from every checked road."""
    msg = arcpy.AddMessage
    today = today or _today()
    orig = _load(training_folder, ORIGINALS)
    if not orig:
        raise ValueError(f"No snapshot in {training_folder}. Run 'Start of Day' before editing.")
    if today not in orig["snapshots"]:
        arcpy.AddWarning(
            f"No snapshot was taken today ({today}). Roads first seen today and edited before "
            "this run have their edited shape as their original, so they count as unchanged.")
    sr = arcpy.Describe(roads_fc).spatialReference
    checked = _read_roads(arcpy, roads_fc, checked_where)
    if not checked:
        raise ValueError(f"No roads match {checked_where!r}. "
                         + _value_hint(arcpy, roads_fc, checked_where))
    prev = (_load(training_folder, TRAINING) or {}).get("roads", {})
    originals = {int(k): [np.asarray(c) for c in v["parts"]] for k, v in orig["roads"].items()}
    every = None   # all originals, for roads split or added since their snapshot

    roads = {}
    counts = dict(new_moved=0, new_same=0, re_edited=0, kept=0, no_original=0)
    for oid, parts in checked.items():
        pts, tans, _ = sample_lines(parts, SAMPLE_SPACING)
        if oid in originals:
            idx = SegmentIndex(originals[oid])
        else:
            every = every or SegmentIndex([c for p in originals.values() for c in p])
            idx = every
        d = idx.distances(pts, tans, MATCH_DIST)
        found = d[~np.isnan(d)]
        info = orig["roads"].get(str(oid))
        known = info is not None and not info["checked_before"]
        if len(found) < 0.5 * len(d):   # most of it is nowhere near an original line
            shift_avg = shift_max = None
            changed = 1
        else:
            shift_avg, shift_max = round(float(found.mean()), 2), round(float(found.max()), 2)
            changed = int(shift_max >= min_change)
        if oid not in originals:
            known = False   # split or added: compared with the nearest originals only
        old = prev.get(str(oid))
        parts_r = _round(parts)
        if old is not None and _same(old["parts"], parts_r):
            day = old["day"]
            counts["kept"] += 1
        else:
            day = today
            if old is not None:
                counts["re_edited"] += 1
            elif changed:
                counts["new_moved"] += 1
            else:
                counts["new_same"] += 1
        if shift_avg is None:
            counts["no_original"] += 1
        roads[str(oid)] = dict(day=day, parts=parts_r, changed=changed, shift_avg=shift_avg,
                               shift_max=shift_max, original_known=bool(known))
    dropped = len(set(prev) - set(roads))
    _save(training_folder, TRAINING, {"sr": sr.exportToString(), "updated": today,
                                      "roads": roads})

    n_changed = sum(r["changed"] for r in roads.values())
    msg(f"Training roads: {len(roads)} checked road(s) in total "
        f"({n_changed} moved by your edits, {len(roads) - n_changed} already right).")
    msg(f"Today ({today}): {counts['new_moved']} newly checked and moved, {counts['new_same']} "
        f"newly checked and left as they were, {counts['re_edited']} edited again.")
    if dropped:
        msg(f"{dropped} road(s) are no longer checked (or were deleted) and were removed from "
            "the training roads.")
    if counts["no_original"]:
        arcpy.AddWarning(f"{counts['no_original']} checked road(s) are more than "
                         f"{MATCH_DIST:g} map units from any original line; they are kept but "
                         "not used for measuring.")
    moved = [r["shift_avg"] for r in roads.values() if r["changed"] and r["shift_avg"] is not None]
    if moved:
        msg(f"Roads you moved were off by a median of {np.median(moved):.1f} on average.")
    if out_fc:
        _write_training_fc(arcpy, out_fc, roads, sr)
        msg(f"Training roads written to {out_fc}")
    return dict(counts, total=len(roads), changed=n_changed, dropped=dropped)


TRAIN_FIELDS = [("ROAD_OID", "LONG"), ("DAY", "TEXT"), ("CHANGED", "SHORT"),
                ("SHIFT_AVG", "DOUBLE"), ("SHIFT_MAX", "DOUBLE")]


def _write_training_fc(arcpy, out_fc, roads, sr):
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    folder, name = os.path.split(out_fc)
    arcpy.management.CreateFeatureclass(folder, name, "POLYLINE", spatial_reference=sr)
    for fname, ftype in TRAIN_FIELDS:
        if ftype == "TEXT":
            arcpy.management.AddField(out_fc, fname, ftype, field_length=10)
        else:
            arcpy.management.AddField(out_fc, fname, ftype)
    with arcpy.da.InsertCursor(out_fc, [f for f, _ in TRAIN_FIELDS] + ["SHAPE@"]) as cur:
        for oid, r in roads.items():
            for c in r["parts"]:
                line = arcpy.Polyline(arcpy.Array([arcpy.Point(x, y) for x, y in c]), sr)
                cur.insertRow([int(oid), r["day"], r["changed"], r["shift_avg"],
                               r["shift_max"], line])


# --------------------------------------------------------------------------- 3. train

def _projector(arcpy, sr_text, sr):
    """Function mapping stored parts (in the road layer's coordinates) to ``sr``."""
    src = arcpy.SpatialReference()
    src.loadFromString(sr_text)
    if src.name == sr.name:
        return lambda parts: [np.asarray(c, float) for c in parts]

    def project(parts):
        out = []
        for c in parts:
            g = arcpy.Polyline(arcpy.Array([arcpy.Point(x, y) for x, y in c]), src)
            out += _parts(g.projectAs(sr))
        return out
    return project


def _ext(c):
    return (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())


def _grow(e, d):
    return (e[0] - d, e[1] - d, e[2] + d, e[3] + d)


def train(arcpy, *, imagery, training_folder, grids_fc, grid_id_field, bands,
          compare=(), training_grids=30, refine: RefineConfig | None = None, today=None):
    return _with_log(arcpy, _train, arcpy, imagery, training_folder, grids_fc, grid_id_field,
                     bands, list(compare or ()), training_grids, refine or RefineConfig(),
                     today or _today())


def _train(arcpy, imagery, training_folder, grids_fc, grid_id_field, bands, compare,
           training_grids, rcfg, today):
    msg = arcpy.AddMessage
    store = _load(training_folder, TRAINING)
    orig = _load(training_folder, ORIGINALS)
    if not store or not store["roads"] or not orig:
        raise ValueError(f"No training roads in {training_folder}. Use 'Start of Day' and "
                         "'End of Day' while you edit first.")
    reader = ArcTileReader(arcpy, imagery, bands)
    sr = reader.sr
    msg(f"Imagery: {len(reader.tiles)} raster(s), {sr.name}, cell size {reader.res:g}")

    to_sr = _projector(arcpy, store["sr"], sr)
    truth = []   # (oid, coords, ext, record)
    for oid, r in store["roads"].items():
        for c in to_sr(r["parts"]):
            truth.append((int(oid), c, _ext(c), r))
    to_sr = _projector(arcpy, orig["sr"], sr)
    originals = []   # (oid, coords, ext)
    for oid, r in orig["roads"].items():
        for c in to_sr(r["parts"]):
            originals.append((int(oid), c, _ext(c)))

    grids = read_grids(arcpy, grids_fc, grid_id_field, sr)
    mids = np.array([interpolate(c, cumlen(c)[-1] / 2)[0] for _, c, _, _ in truth])
    home = np.full(len(truth), -1)
    for gi, (_, g, _) in enumerate(grids):
        free = home < 0
        if free.any():
            inside = _points_in_polygons(mids[free], [g])
            home[np.flatnonzero(free)[inside]] = gi
    with_truth = sorted(set(home[home >= 0].tolist()))
    if not with_truth:
        raise ValueError("None of the checked roads are inside the grids")
    holdout = [gi for gi in with_truth
               if zlib.crc32(str(grids[gi][0]).encode()) % HOLDOUT_EVERY == 0]
    if not holdout and len(with_truth) >= 2:
        holdout = [min(with_truth, key=lambda gi: zlib.crc32(str(grids[gi][0]).encode()))]
    if len(holdout) == len(with_truth):
        holdout = []
    learn = [gi for gi in with_truth if gi not in holdout]
    msg(f"{len(store['roads'])} checked roads in {len(with_truth)} grid(s): training on "
        f"{len(learn)}, testing on {len(holdout)} ("
        + (", ".join(str(grids[g][0]) for g in holdout) or "none: too few grids yet") + ")")

    ccfg = ClassifierConfig(training_grids=training_grids)
    train_on = pick_spread(learn, training_grids)
    cells = [[c for (_, c, _, _), h in zip(truth, home) if h == gi] for gi in train_on]
    test_set = set(holdout)
    learn_truth = [(c, e) for (_, c, e, _), h in zip(truth, home) if h not in test_set]

    def lines_near(b):
        return ([c for c, e in learn_truth if _overlaps(e, b)]
                + [c for _, c, e in originals if _overlaps(e, b)])

    def trusted_near(b):
        # road labels on your lines only; the originals (and roads in the test grids)
        # just keep road pixels out of the not-road labels
        return ([(c, None) for c, e in learn_truth if _overlaps(e, b)],
                [c for _, c, e in originals if _overlaps(e, b)]
                + [c for (_, c, e, _), h in zip(truth, home) if h in test_set
                   and _overlaps(e, b)])

    msg("Training the road classifier on grid(s): "
        + ", ".join(str(grids[g][0]) for g in train_on))
    clf = train_classifier(reader, cells, lines_near, ccfg, None, trusted_near=trusted_near)
    clf.trained_from = f"{len(learn_truth)} checked road parts, {today}"
    models = os.path.join(training_folder, MODELS)
    os.makedirs(models, exist_ok=True)
    dated = os.path.join(models, f"road_classifier_{today}.json")
    clf.save(dated)
    msg(f"Classifier saved to {dated} (training accuracy {clf.train_accuracy:.3f})")

    current_path = os.path.join(models, CURRENT_MODEL)
    prev = RoadClassifier.load(current_path) if os.path.exists(current_path) else None
    if prev is not None and (prev.has_nir != clf.has_nir or prev.lidar):
        prev = None   # different bands or LiDAR: not comparable

    scores = {}
    if holdout:
        scores = _score(arcpy, msg, reader, grids, holdout, truth, home, originals, clf, prev,
                        compare, rcfg, ccfg, sr)
    new, old = scores.get("new"), scores.get("previous")
    promote = prev is None or new is None or old is None or new <= old
    if promote:
        clf.save(current_path)
        msg(f"This model is now the current one: {current_path}. Point 'Align Centerlines' "
            "at it (Trained classifier).")
    else:
        msg(f"The new model ({new:.2f}) did not beat the current one ({old:.2f}) on the test "
            f"grids, so {current_path} is unchanged. Tonight's model is kept as {dated}.")
    _append_history(models, today, len(store["roads"]), len(train_on), len(holdout), clf,
                    scores, promote)
    return dict(scores, promoted=promote, model=dated, holdout=[grids[g][0] for g in holdout])


def _score(arcpy, msg, reader, grids, holdout, truth, home, originals, clf, prev, compare,
           rcfg, ccfg, sr):
    """Align the original lines in the test grids with each model and compare with your
    edits. Roads whose original is unknown (checked before the first snapshot) are skipped."""
    test = [(c, e) for (_, c, e, r), h in zip(truth, home)
            if h in set(holdout) and r.get("original_known")]
    if not test:
        arcpy.AddWarning("The test grids have no checked roads with a known original yet, so "
                         "the models can't be measured.")
        return {}
    near = [(oid, c, e) for oid, c, e in originals
            if any(_overlaps(e, _grow(te, MATCH_DIST)) for _, te in test)]
    roads = [(oid, arcpy.Polyline(arcpy.Array([arcpy.Point(x, y) for x, y in c]), sr), [c], e)
             for oid, c, e in near]
    models = [("Tool, new model", clf)] + ([("Tool, current model", prev)] if prev else [])
    aligned = {name: [] for name, _ in models}
    for gi in holdout:
        pieces = clip_to_grid(arcpy, roads, grids, gi, sr)
        if not pieces:
            continue
        lines = [c for _, c in pieces]
        for name, model in models:
            pmap = region_probability(reader, model, None, grids[gi][2], lines, rcfg, ccfg)
            records, _ = refine_pieces(pmap, pieces, rcfg, reader.res)
            aligned[name] += [np.asarray(r["coords"]) for r in records]
        msg(f"Test grid {grids[gi][0]}: {len(pieces)} road pieces aligned")

    lines = [c for c, _ in test]
    pts, tans, line_idx = sample_lines(lines, SAMPLE_SPACING)
    errors = {"Original lines": SegmentIndex([c for _, c, _ in near]).distances(
        pts, tans, MATCH_DIST)}
    for name, _ in models:
        errors[name] = SegmentIndex(aligned[name]).distances(pts, tans, MATCH_DIST)
    for fc in compare:
        other = [c for _, _, parts, _ in read_lines(arcpy, fc, "OID@", sr) for c in parts]
        errors[os.path.basename(fc) or fc] = SegmentIndex(other).distances(pts, tans, MATCH_DIST)
    cover = _cover_classes(reader, pts, line_idx)
    cover[near_junctions(pts, find_junctions([c for _, c, _ in near]), 40.0)] = "intersection"
    msg(f"How close each layer is to your edits on {len(lines)} checked road part(s) in the "
        "test grids (smaller is better):")
    units = sr.linearUnitName or ""
    for line in eval_report(errors, cover,
                            "ft" if "foot" in units.lower() or "feet" in units.lower() else ""):
        msg(line)
    allm = np.all([~np.isnan(e) for e in errors.values()], axis=0)
    med = {k: float(np.median(e[allm])) if allm.any() else None for k, e in errors.items()}
    return {"original": med["Original lines"], "new": med["Tool, new model"],
            "previous": med.get("Tool, current model"), "points": int(allm.sum())}


def _append_history(models, today, n_roads, n_train, n_test, clf, scores, promoted):
    path = os.path.join(models, "history.csv")
    head = not os.path.exists(path)

    def f(v):
        return "" if v is None else f"{v:.2f}"
    with open(path, "a") as out:
        if head:
            out.write("day,checked_roads,train_grids,test_grids,train_accuracy,"
                      "median_original,median_new,median_current,promoted\n")
        out.write(f"{today},{n_roads},{n_train},{n_test},{clf.train_accuracy:.3f},"
                  f"{f(scores.get('original'))},{f(scores.get('new'))},"
                  f"{f(scores.get('previous'))},{int(promoted)}\n")
