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

Following the REVAMP Phase 6 manual:

* Roads marked abandoned or private (``AbandonedRoad`` / ``PrivateRoad`` = 'Yes',
  when the layer has those fields) are never used: there may be no pavement.
* The ends of every line are left out of the road labels: lines are split at
  intersections, and the manual snaps them simply and bends T-intersections along
  the turn, so near a junction a line is not a pavement centre.
* Roads traced from a reference layer (the manual's Replace Geometry > Trace) are
  marked, so the nightly test can also report on the roads drawn from the imagery.
* A road checked without an edit means "meets the standard" (the line falls
  within the pavement), not "on the exact centre". Such roads teach the keep/move
  model to leave good lines alone.

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
from .engine import (find_junctions, fit_move_model, move_probability, pick_spread,
                     refine_pieces, region_probability, train_classifier)
from .evaluate import SegmentIndex, eval_report, near_junctions, sample_lines
from .geom import cumlen, interpolate

ORIGINALS = "originals.json"
TRAINING = "training_roads.json"
MODELS = "models"
CURRENT_MODEL = "road_classifier.json"
HOLDOUT_EVERY = 5      # about 1 grid in 5 with checked roads is kept for testing
SAMPLE_SPACING = 5.0   # how often lines are compared (map units)
MATCH_DIST = 40.0      # an original farther than this from the edited line isn't its match
END_TRIM = 40.0        # line ends left out of the road labels (intersections)
TRACE_TOL = 0.5        # a road this close to the reference layer along...
TRACE_SHARE = 0.8      # ...this share of its length was traced from it
EXCLUDE_FIELDS = ("abandonedroad", "privateroad")   # 'Yes' = not used (no pavement)


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


def _excluded(arcpy, roads_fc):
    """OIDs marked abandoned or private ('Yes'), and the fields that said so."""
    fields = [f.name for f in arcpy.ListFields(roads_fc)
              if f.name.lower().replace("_", "").replace(" ", "") in EXCLUDE_FIELDS]
    out = set()
    if fields:
        with arcpy.da.SearchCursor(roads_fc, ["OID@"] + fields) as cur:
            for row in cur:
                if any(str(v).strip().lower() == "yes" for v in row[1:]):
                    out.add(int(row[0]))
    return out, fields


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
            min_change=1.0, reference_fc=None, today=None):
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
    skip, fields = _excluded(arcpy, roads_fc)
    skipped = len(set(checked) & skip)
    checked = {k: v for k, v in checked.items() if k not in skip}
    if fields:
        msg(f"Roads with {' or '.join(fields)} = 'Yes' are not used ({skipped} checked "
            "road(s) left out).")
    if not checked:
        raise ValueError("Every checked road is marked abandoned or private.")
    ref = None
    if reference_fc:
        ref = SegmentIndex([c for _, _, parts, _ in read_lines(arcpy, reference_fc, "OID@", sr)
                            for c in parts])
    prev = (_load(training_folder, TRAINING) or {}).get("roads", {})
    originals = {int(k): [np.asarray(c) for c in v["parts"]] for k, v in orig["roads"].items()}
    every = None   # all originals, for roads split or added since their snapshot

    roads = {}
    counts = dict(new_moved=0, new_same=0, re_edited=0, kept=0, no_original=0, traced=0)
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
        traced = 0
        if ref is not None and len(pts):
            dr = ref.distances(pts, tans, MATCH_DIST)
            traced = int(np.mean(dr <= TRACE_TOL) >= TRACE_SHARE)   # NaN counts as no
        elif str(oid) in prev:
            traced = prev[str(oid)].get("traced", 0)
        counts["traced"] += traced
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
                               shift_max=shift_max, original_known=bool(known),
                               traced=traced)
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
    if ref is not None:
        msg(f"{counts['traced']} checked road(s) follow {reference_fc} within {TRACE_TOL:g} "
            "(traced); the nightly test also reports on the others separately.")
    moved = [r["shift_avg"] for r in roads.values() if r["changed"] and r["shift_avg"] is not None]
    if moved:
        msg(f"Roads you moved were off by a median of {np.median(moved):.1f} on average.")
    if out_fc:
        _write_training_fc(arcpy, out_fc, roads, sr)
        msg(f"Training roads written to {out_fc}")
    return dict(counts, total=len(roads), changed=n_changed, dropped=dropped, skipped=skipped)


TRAIN_FIELDS = [("ROAD_OID", "LONG"), ("DAY", "TEXT"), ("CHANGED", "SHORT"),
                ("SHIFT_AVG", "DOUBLE"), ("SHIFT_MAX", "DOUBLE"), ("TRACED", "SHORT")]


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
                               r["shift_max"], r.get("traced", 0), line])


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


def trim_ends(c, d: float):
    """The part of a line more than ``d`` from both ends, or None if too short."""
    s = cumlen(c)
    if s[-1] <= 2 * d + 10.0:
        return None
    inner = (s > d) & (s < s[-1] - d)
    ends = interpolate(c, np.array([d, s[-1] - d]), s)
    return np.vstack([ends[:1], c[inner], ends[1:]])


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
    test_set = set(holdout)
    learn_truth = []   # (trimmed coords, ext, home grid): road labels, ends left out
    for (_, c, _, _), h in zip(truth, home):
        t = trim_ends(c, END_TRIM)
        if t is not None and h not in test_set:
            learn_truth.append((t, _ext(t), h))
    cells = [[c for c, _, h in learn_truth if h == gi] for gi in train_on]
    if not any(cells):
        raise ValueError("The training grids have no checked roads longer than "
                         f"{2 * END_TRIM:g} to learn from yet.")

    def lines_near(b):
        return ([c for _, c, e, _ in truth if _overlaps(e, b)]
                + [c for _, c, e in originals if _overlaps(e, b)])

    def trusted_near(b):
        # road labels on your lines only; the originals (and roads in the test grids)
        # just keep road pixels out of the not-road labels
        return ([(c, None) for c, e, _ in learn_truth if _overlaps(e, b)],
                [c for _, c, e in originals if _overlaps(e, b)]
                + [c for _, c, e, _ in truth if _overlaps(e, b)])

    msg("Training the road classifier on grid(s): "
        + ", ".join(str(grids[g][0]) for g in train_on))
    clf = train_classifier(reader, cells, lines_near, ccfg, None, trusted_near=trusted_near)
    clf.trained_from = f"{len(learn_truth)} checked road parts, {today}"
    clf.move_model = _learn_moves(arcpy, msg, reader, grids, train_on, truth, home,
                                  originals, clf, rcfg, ccfg, sr)
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
    test = [(c, e, r.get("traced", 0)) for (_, c, e, r), h in zip(truth, home)
            if h in set(holdout) and r.get("original_known")]
    if not test:
        arcpy.AddWarning("The test grids have no checked roads with a known original yet, so "
                         "the models can't be measured.")
        return {}
    near = [(oid, c, e) for oid, c, e in originals
            if any(_overlaps(e, _grow(te, MATCH_DIST)) for _, te, _ in test)]
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
            records, _ = refine_pieces(pmap, pieces, rcfg, reader.res,
                                       move_model=model.move_model)
            aligned[name] += [np.asarray(r["coords"]) for r in records]
        msg(f"Test grid {grids[gi][0]}: {len(pieces)} road pieces aligned")

    lines = [c for c, _, _ in test]
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
    ft = "ft" if "foot" in units.lower() or "feet" in units.lower() else ""
    for line in eval_report(errors, cover, ft):
        msg(line)
    traced = np.array([t for _, _, t in test], bool)[line_idx]
    if traced.any() and not traced.all():
        msg("")
        msg(f"Only roads you did NOT trace from a reference layer ({(~traced).sum()} points): "
            "the fair test against that layer")
        for line in eval_report({k: e[~traced] for k, e in errors.items()}, cover[~traced], ft):
            msg(line)
    # models are compared away from intersections, where the manual's snapping rules
    # (not the pavement centre) decide where a line goes
    use = np.all([~np.isnan(e) for e in errors.values()], axis=0) & (cover != "intersection")
    med = {k: float(np.median(e[use])) if use.any() else None for k, e in errors.items()}
    msg(f"Median distance to your edits away from intersections ({int(use.sum())} points): "
        + ", ".join(f"{k} {v:.2f}" for k, v in med.items() if v is not None))
    return {"original": med["Original lines"], "new": med["Tool, new model"],
            "previous": med.get("Tool, current model"), "points": int(use.sum())}


def _learn_moves(arcpy, msg, reader, grids, train_on, truth, home, originals, clf, rcfg,
                 ccfg, sr):
    """Align the original lines of checked roads in the training grids and learn which
    moves bring a line closer to the checked line (keep/move model)."""
    by_oid: dict = {}
    for (oid, c, _, r), h in zip(truth, home):
        if r.get("original_known") and h in train_on:
            by_oid.setdefault(oid, []).append(c)
    orig = [(oid, c, e) for oid, c, e in originals if oid in by_oid]
    if not orig:
        msg("Keep/move model: no checked roads with a known original in the training grids "
            "yet.")
        return None
    roads = [(oid, arcpy.Polyline(arcpy.Array([arcpy.Point(x, y) for x, y in c]), sr), [c], e)
             for oid, c, e in orig]
    recs, gains = [], []
    for gi in train_on:
        pieces = clip_to_grid(arcpy, roads, grids, gi, sr)
        if not pieces:
            continue
        pmap = region_probability(reader, clf, None, grids[gi][2], [c for _, c in pieces],
                                  rcfg, ccfg)
        records, _ = refine_pieces(pmap, pieces, rcfg, reader.res)
        for (oid, c0), rec in zip([p for p in pieces if len(p[1]) >= 2], records):
            if not rec["moved"]:
                continue
            truth_idx = SegmentIndex(by_oid[oid])
            # compare away from the ends (intersections follow the manual's snapping rules)
            before, after = trim_ends(np.asarray(c0), END_TRIM), trim_ends(
                np.asarray(rec["coords"]), END_TRIM)
            if before is None or after is None:
                continue
            e = []
            for line in (before, after):
                p, t, _ = sample_lines([line], SAMPLE_SPACING)
                d = truth_idx.distances(p, t, MATCH_DIST)
                e.append(float(np.nanmean(d)) if np.mean(~np.isnan(d)) >= 0.5 else np.nan)
            if not np.isnan(e).any():
                recs.append(rec)
                gains.append(e[0] - e[1])
    model = fit_move_model(recs, gains)
    if model is None:
        msg(f"Keep/move model: {len(recs)} moved line(s) to learn from; need at least 20 with "
            "moves that helped and moves that didn't. Every move is applied for now.")
        return None
    kept = sum(move_probability(model, r) < model["threshold"] for r in recs)
    msg(f"Keep/move model learned from {len(recs)} moved lines: it keeps the original for "
        f"{kept} of them. On these lines, applying every move would bring them "
        f"{model['gain_all']:+.0f} closer in total, applying only the chosen ones "
        f"{model['gain_chosen']:+.0f}.")
    return model


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
