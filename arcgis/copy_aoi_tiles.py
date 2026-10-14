"""Copy the imagery tiles that cover your work grids out of a statewide tile folder.

Run in ArcGIS Pro: Analysis tab > Python > Python Window. Edit the four
settings below, paste the whole file into the window, and press Enter.

AOI can be any polygon layer: your 1-mile work grids (only selected grids are
used if any are selected), a county boundary, etc. Tiles are copied as they
are (no clipping or resampling), together with their sidecar files (.aux.xml,
.ovr, .tfw, ...), so the copies are exact. Needs only ArcGIS Pro (any licence).
"""
import os
import shutil

import arcpy

# ---- settings ---------------------------------------------------------------
STATE_TILES = r"D:\Imagery\State_2024\tiles"            # folder of statewide .tif tiles
AOI = r"D:\Centerline\CountyA\data\Inputs.gdb\Grids_1mi"  # the county's work grids (layer or path)
BUFFER = 500          # keep imagery this far past the grid edges (AOI's map units)
OUT_TILES = r"D:\Centerline\CountyA\imagery\tiles"     # where the AOI's tiles go
DRY_RUN = True        # True: only count tiles and size; set False to copy
# -----------------------------------------------------------------------------

TILE_EXTS = (".tif", ".tiff", ".img", ".jp2")


def aoi_shapes(aoi, buffer):
    """Each AOI polygon, buffered, with its extent - kept separate (no slow union)."""
    shapes = []
    with arcpy.da.SearchCursor(aoi, ["SHAPE@"]) as cur:
        for (geom,) in cur:
            if geom is not None:
                b = geom.buffer(buffer)
                e = b.extent
                shapes.append((b, (e.XMin, e.YMin, e.XMax, e.YMax)))
    if not shapes:
        raise ValueError(f"No features in {aoi} (check selection / definition query)")
    sr = shapes[0][0].spatialReference
    arcpy.AddMessage(f"{len(shapes)} AOI polygon(s), buffered by {buffer} {sr.linearUnitName}")
    return shapes, sr


def footprint(path):
    r = arcpy.Raster(path)
    e = r.extent
    pts = [arcpy.Point(e.XMin, e.YMin), arcpy.Point(e.XMin, e.YMax),
           arcpy.Point(e.XMax, e.YMax), arcpy.Point(e.XMax, e.YMin)]
    return arcpy.Polygon(arcpy.Array(pts), r.spatialReference)


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
    tiles = sorted(os.path.join(root, f) for root, _, files in os.walk(state_tiles)
                   for f in files if f.lower().endswith(TILE_EXTS))
    arcpy.AddMessage(f"Checking {len(tiles)} tiles against the AOI...")
    listing = {}
    copied = 0
    size = 0
    for i, path in enumerate(tiles, 1):
        fp = footprint(path)
        if fp.spatialReference.name != sr.name:
            fp = fp.projectAs(sr)
        e = fp.extent
        hit = any(e.XMin <= b[2] and e.XMax >= b[0] and e.YMin <= b[3] and e.YMax >= b[1]
                  and not fp.disjoint(g) for g, b in shapes)
        if hit:
            folder, name = os.path.split(path)
            stem = os.path.splitext(name)[0]
            if folder not in listing:
                listing[folder] = os.listdir(folder)
            for f in listing[folder]:  # the tile and its sidecars: stem.tif, stem.tif.aux.xml...
                if f.startswith(stem + "."):
                    size += os.path.getsize(os.path.join(folder, f))
                    if not dry_run:
                        shutil.copy2(os.path.join(folder, f), os.path.join(out_tiles, f))
            copied += 1
        if i % 200 == 0:
            arcpy.AddMessage(f"  {i}/{len(tiles)} checked, {copied} copied")
    gb = size / 1024 ** 3
    if dry_run:
        done = (f"Dry run: {copied} of {len(tiles)} tiles touch the AOI, {gb:,.1f} GB with "
                f"sidecars. Nothing copied; set DRY_RUN = False to copy.")
    else:
        done = f"Done: {copied} of {len(tiles)} tiles ({gb:,.1f} GB) copied to {out_tiles}"
    arcpy.AddMessage(done)
    print(done)
    return copied


copy_aoi_tiles(STATE_TILES, AOI, BUFFER, OUT_TILES, DRY_RUN)
