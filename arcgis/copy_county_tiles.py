"""Copy the imagery tiles that cover one county out of a statewide tile folder.

Run in ArcGIS Pro: Analysis tab > Python > Python Window. Edit the four
settings below, paste the whole file into the window, and press Enter.

Tiles are copied as they are (no clipping or resampling), together with their
sidecar files (.aux.xml, .ovr, .tfw, .prj, ...), so the copies are exact.
Needs only ArcGIS Pro (any licence level).
"""
import os
import shutil

import arcpy

# ---- settings ---------------------------------------------------------------
STATE_TILES = r"D:\Imagery\State_2024\tiles"         # folder of statewide .tif tiles
BOUNDARY = r"D:\Centerline\data\Inputs.gdb\County"   # county boundary (layer name or path)
BUFFER = 500          # keep imagery this far past the county line (boundary's map units)
OUT_TILES = r"D:\Centerline\imagery\2024\tiles"      # where the county's tiles go
# -----------------------------------------------------------------------------

TILE_EXTS = (".tif", ".tiff", ".img", ".jp2")


def county_area(boundary, buffer):
    area = None
    with arcpy.da.SearchCursor(boundary, ["SHAPE@"]) as cur:
        for (geom,) in cur:
            area = geom if area is None else area.union(geom)
    if area is None:
        raise ValueError(f"No features in {boundary} (check selection / definition query)")
    arcpy.AddMessage(f"Buffering the county by {buffer} {area.spatialReference.linearUnitName}")
    return area.buffer(buffer)


def footprint(path):
    r = arcpy.Raster(path)
    e = r.extent
    pts = [arcpy.Point(e.XMin, e.YMin), arcpy.Point(e.XMin, e.YMax),
           arcpy.Point(e.XMax, e.YMax), arcpy.Point(e.XMax, e.YMin)]
    return arcpy.Polygon(arcpy.Array(pts), r.spatialReference)


def copy_county_tiles(state_tiles, boundary, buffer, out_tiles):
    area = county_area(boundary, buffer)
    os.makedirs(out_tiles, exist_ok=True)
    tiles = sorted(os.path.join(root, f) for root, _, files in os.walk(state_tiles)
                   for f in files if f.lower().endswith(TILE_EXTS))
    arcpy.AddMessage(f"Checking {len(tiles)} tiles against the county...")
    copied = 0
    for i, path in enumerate(tiles, 1):
        fp = footprint(path)
        if fp.spatialReference.name != area.spatialReference.name:
            fp = fp.projectAs(area.spatialReference)
        if fp.disjoint(area):
            continue
        folder, name = os.path.split(path)
        stem = os.path.splitext(name)[0]
        for f in os.listdir(folder):  # the tile and its sidecars: stem.tif, stem.tif.aux.xml, stem.tfw...
            if f.startswith(stem + "."):
                shutil.copy2(os.path.join(folder, f), os.path.join(out_tiles, f))
        copied += 1
        if i % 200 == 0:
            arcpy.AddMessage(f"  {i}/{len(tiles)} checked, {copied} copied")
    arcpy.AddMessage(f"Done: {copied} of {len(tiles)} tiles copied to {out_tiles}")
    print(f"Done: {copied} of {len(tiles)} tiles copied to {out_tiles}")
    return copied


copy_county_tiles(STATE_TILES, BOUNDARY, BUFFER, OUT_TILES)
