"""Copy the imagery tiles that cover your work grids out of a statewide tile folder.

Run in ArcGIS Pro: Analysis tab > Python > Python Window. Edit the settings
below, paste the whole file into the window, and press Enter.

AOI can be any polygon layer: your 1-mile work grids (only selected grids are
used if any are selected), a county boundary, etc. Tiles are copied as they
are (no clipping or resampling), together with their sidecar files (.aux.xml,
.ovr, .tfw, ...), so the copies are exact. Needs only ArcGIS Pro (any licence).

At the end it reports:
  * tiles it could not read (it skips them and carries on),
  * other image formats in the folder that it did not consider,
  * AOI polygons (grids) that are not fully covered by the tiles it found.
Run it again after fixing a problem: tiles already copied are not copied twice.
"""
import os
import shutil

import arcpy

# ---- settings ---------------------------------------------------------------
STATE_TILES = r"D:\Imagery\State_2024\tiles"            # folder of statewide tiles
AOI = r"D:\Centerline\CountyA\data\Inputs.gdb\Grids_1mi"  # the county's work grids (layer or path)
BUFFER = 500          # keep imagery this far past the grid edges (AOI's map units)
OUT_TILES = r"D:\Centerline\CountyA\imagery\tiles"     # where the AOI's tiles go
DRY_RUN = True        # True: only count tiles and size; set False to copy
# -----------------------------------------------------------------------------

TILE_EXTS = (".tif", ".tiff", ".img", ".jp2", ".sid")
# files that travel with a tile rather than being imagery themselves
SIDECAR_EXTS = (".xml", ".ovr", ".tfw", ".tifw", ".tfwx", ".prj", ".aux", ".rrd", ".rde",
                ".sdw", ".j2w", ".jgw", ".jpw", ".pgw", ".cpg", ".lock", ".txt", ".ini",
                ".db", ".dbf", ".shp", ".shx", ".sbn", ".sbx", ".md5", ".htm", ".html")


def aoi_shapes(aoi, buffer):
    """(oid, polygon, buffered polygon, buffered extent) per AOI feature; no slow union."""
    shapes = []
    with arcpy.da.SearchCursor(aoi, ["OID@", "SHAPE@"]) as cur:
        for oid, geom in cur:
            if geom is not None:
                b = geom.buffer(buffer)
                e = b.extent
                shapes.append((oid, geom, b, (e.XMin, e.YMin, e.XMax, e.YMax)))
    if not shapes:
        raise ValueError(f"No features in {aoi} (check selection / definition query)")
    sr = shapes[0][1].spatialReference
    arcpy.AddMessage(f"{len(shapes)} AOI polygon(s), buffered by {buffer} {sr.linearUnitName}")
    return shapes, sr


def find_files(folder):
    """Tile paths, plus a count of other (non-sidecar) file types that were not used."""
    tiles, other = [], {}
    for root, _, files in os.walk(folder):
        for f in files:
            low = f.lower()
            ext = os.path.splitext(low)[1]
            if low.endswith(TILE_EXTS):
                tiles.append(os.path.join(root, f))
            elif ext and ext not in SIDECAR_EXTS:
                other[ext] = other.get(ext, 0) + 1
    return sorted(tiles), other


def footprint(path):
    r = arcpy.Raster(path)
    e = r.extent
    pts = [arcpy.Point(e.XMin, e.YMin), arcpy.Point(e.XMin, e.YMax),
           arcpy.Point(e.XMax, e.YMax), arcpy.Point(e.XMax, e.YMin)]
    return arcpy.Polygon(arcpy.Array(pts), r.spatialReference)


def _overlaps(a, b):
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _extent(g):
    e = g.extent
    return (e.XMin, e.YMin, e.XMax, e.YMax)


def copy_aoi_tiles(state_tiles, aoi, buffer, out_tiles, dry_run=False):
    # Originals are only read, never moved or changed. The output must not be inside
    # the statewide folder, or later runs would pick the copies up as source tiles.
    src = os.path.normcase(os.path.abspath(state_tiles))
    dst = os.path.normcase(os.path.abspath(out_tiles))
    if dst == src or dst.startswith(src + os.sep):
        raise ValueError("OUT_TILES must be outside STATE_TILES; pick a separate folder")
    shapes, sr = aoi_shapes(aoi, buffer)
    if not dry_run:
        os.makedirs(out_tiles, exist_ok=True)
    tiles, other = find_files(state_tiles)
    arcpy.AddMessage(f"Checking {len(tiles)} tiles against the AOI...")
    listing = {}
    matched = []       # footprints of tiles that touch the AOI
    failed = []        # (path, error)
    size = copied = skipped = 0
    for i, path in enumerate(tiles, 1):
        try:
            fp = footprint(path)
            if fp.spatialReference is not None and sr is not None \
                    and fp.spatialReference.name != sr.name:
                fp = fp.projectAs(sr)
        except Exception as e:  # unreadable tile: note it and keep going
            failed.append((path, str(e).strip().splitlines()[0] if str(e).strip() else "error"))
            continue
        e = _extent(fp)
        if not any(_overlaps(e, b) and not fp.disjoint(g) for _, _, g, b in shapes):
            continue
        matched.append((fp, e))
        folder, name = os.path.split(path)
        stem = os.path.splitext(name)[0]
        if folder not in listing:
            listing[folder] = os.listdir(folder)
        try:
            for f in listing[folder]:  # the tile and its sidecars: stem.tif, stem.tif.aux.xml...
                if not f.startswith(stem + "."):
                    continue
                s = os.path.join(folder, f)
                d = os.path.join(out_tiles, f)
                n = os.path.getsize(s)
                size += n
                if dry_run:
                    continue
                if os.path.exists(d) and os.path.getsize(d) == n:
                    skipped += 1   # already copied on an earlier run
                    continue
                shutil.copy2(s, d)
            copied += 1
        except Exception as e:  # e.g. disk full: report it, keep checking the rest
            failed.append((path, f"copy failed: {e}"))
        if i % 200 == 0:
            arcpy.AddMessage(f"  {i}/{len(tiles)} checked, {len(matched)} touch the AOI")

    uncovered = _coverage(shapes, matched)
    _report(tiles, matched, copied, skipped, size, failed, other, uncovered, shapes,
            out_tiles, dry_run)
    return dict(matched=len(matched), copied=copied, failed=failed, other=other,
                uncovered=uncovered)


def _coverage(shapes, matched):
    """[(oid, covered fraction)] for AOI polygons less than 99% covered by tile footprints."""
    out = []
    for oid, geom, _, _ in shapes:
        ge = _extent(geom)
        area = geom.area
        if area <= 0:
            continue
        covered = 0.0
        for fp, e in matched:
            if _overlaps(ge, e):
                part = geom.intersect(fp, 4)
                covered += part.area if part is not None else 0.0
        frac = min(1.0, covered / area)  # tiles can overlap slightly; cap at 100%
        if frac < 0.99:
            out.append((oid, frac))
    return out


def _report(tiles, matched, copied, skipped, size, failed, other, uncovered, shapes,
            out_tiles, dry_run):
    def say(m):
        arcpy.AddMessage(m)
        print(m)

    gb = size / 1024 ** 3
    if dry_run:
        say(f"Dry run: {len(matched)} of {len(tiles)} tiles touch the AOI, {gb:,.1f} GB with "
            f"sidecars. Nothing copied; set DRY_RUN = False to copy.")
    else:
        say(f"Done: {copied} of {len(matched)} tiles copied to {out_tiles} ({gb:,.1f} GB"
            + (f"; {skipped} files were already there" if skipped else "") + ").")
    if failed:
        say(f"WARNING: {len(failed)} tile(s) could not be read or copied and were skipped:")
        for p, e in failed[:15]:
            say(f"  {p}: {e}")
        if len(failed) > 15:
            say(f"  ... and {len(failed) - 15} more")
    if other:
        say("NOTE: other file types in the source folder were not considered: "
            + ", ".join(f"{k} ({v})" for k, v in sorted(other.items(), key=lambda kv: -kv[1]))
            + ". If some imagery is in one of these formats, tell whoever maintains the tool.")
    if uncovered:
        none = [o for o, f in uncovered if f == 0]
        part = [(o, f) for o, f in uncovered if f > 0]
        say(f"WARNING: {len(uncovered)} of {len(shapes)} AOI polygons are not fully covered "
            f"by the tiles found ({len(none)} with no imagery at all, {len(part)} partly).")
        if none:
            say("  No imagery, OBJECTID: " + ", ".join(str(o) for o in none[:60])
                + (" ..." if len(none) > 60 else ""))
        if part:
            say("  Partly covered, OBJECTID (covered %): " + ", ".join(
                f"{o} ({f:.0%})" for o, f in part[:60]) + (" ..." if len(part) > 60 else ""))
        say("  Either the source folder has no tiles there (check the state's tile index), "
            "or those tiles are in the unread list above.")
    elif shapes:
        say("Every AOI polygon is fully covered by the tiles found.")


copy_aoi_tiles(STATE_TILES, AOI, BUFFER, OUT_TILES, DRY_RUN)
