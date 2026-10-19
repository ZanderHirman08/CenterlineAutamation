"""Make the LiDAR layers the alignment toolbox can use, from USGS LAZ tiles.

Run in ArcGIS Pro: Analysis tab > Python > Python Window. Edit the settings
below, paste the whole file into the window, and press Enter.

For every work grid (plus a margin) it makes three rasters, in the imagery's
coordinate system and on a 3 ft grid:

* intensity - laser return strength of ground points (asphalt and concrete
  read differently from grass and dirt, also under gaps in the trees)
* dtm       - ground elevation (road beds are flat and smooth)
* dsm       - top surface elevation (dsm - dtm = tree / building height)

Point it at the toolbox's "LiDAR layers folder" setting (in a NEW work folder,
so the classifier is trained with them).

Some ArcGIS Pro installs can't read .laz directly (a LAS dataset made from the
folder lists 0 files). With CONVERT_LAZ = True the script first converts just
the .laz tiles near the grids to Esri's zLAS format (Convert LAS tool) into
ZLAS_FOLDER, and works from those. Tiles already converted are skipped.

* Needs ArcGIS Pro only (any licence): Create LAS Dataset, LAS Dataset To
  Raster and Project Raster.
* Grids already done are skipped, so it can be stopped and run again. With
  DRY_RUN = True it only checks the inputs and reports what it would make.
* Takes a while (roughly half a minute per grid): run it overnight.
"""
import json
import os
import re
import shutil
import time

import arcpy

# ---- settings ---------------------------------------------------------------
LAZ_FOLDER = r"C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\USGS\LAZ"
GRIDS = "Glynn_Grid_1mi"  # map layer name: only SELECTED grids are used (or a full path)
IMAGERY = r"C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\tiles"  # tile or folder
OUT_FOLDER = r"C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\USGS\lidar_layers"
CELL = 3.0          # cell size, in the imagery's units (feet)
MARGIN = 150.0      # extra around each grid, in the imagery's units
CONVERT_LAZ = True  # convert the .laz tiles near the grids to .zlas first (see above)
# Coordinate system of the LiDAR files, assigned while converting (ArcGIS can't read the
# one stored in these files). GA_Statewide_2018_B18_DRRA: NAD83(2011) / CONUS Albers
# (6350) + NAVD88 height, metres (5703), per the USGS index (.vpc) for the project.
LIDAR_WKID, LIDAR_VERT_WKID = 6350, 5703
ZLAS_FOLDER = r"C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\USGS\LAZ_zlas"
DRY_RUN = True      # True: only check the inputs and count the grids to do
# -----------------------------------------------------------------------------

LAYERS = ("intensity", "dtm", "dsm")
# USGS LiDAR tile names give the tile's position: ..._e<km>n<km>.laz, CONUS Albers (1 km)
LPC_TILE_RE = re.compile(r"_e(\d{3,4})n(\d{3,4})\.la[sz]$", re.IGNORECASE)
ALBERS = 6350
NOISE = (7, 18)     # LAS classes for low / high noise


def say(m):
    arcpy.AddMessage(m)
    print(m)


def first_raster(path):
    if os.path.isdir(path) and not path.lower().rstrip("\\/").endswith((".gdb", ".crf")):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.lower().endswith((".tif", ".tiff", ".img", ".jp2", ".sid")):
                    return os.path.join(root, f)
        raise ValueError(f"No imagery found in {path}")
    return path


def grid_paths(out_folder, oid):
    return {k: os.path.join(out_folder, k, f"g{oid}.tif") for k in LAYERS}


def laz_files(folder):
    return sorted(os.path.join(r, f) for r, _, files in os.walk(folder) for f in files
                  if f.lower().endswith((".laz", ".las", ".zlas")))


def near_grids(paths, cells, margin):
    """LiDAR files that may touch the grids (by tile name; files without a position
    in their name are kept)."""
    sr = arcpy.SpatialReference(ALBERS)
    boxes = []
    for _, g in cells:
        e = g.buffer(margin).projectAs(sr).extent
        boxes.append((e.XMin, e.YMin, e.XMax, e.YMax))
    keep = []
    for p in paths:
        m = LPC_TILE_RE.search(os.path.basename(p))
        if not m:
            keep.append(p)
            continue
        e, n = (int(v) * 1000.0 for v in m.groups())
        x0, y0, x1, y1 = e - 1000, n - 1000, e + 1000, n + 1000   # corner or centre
        if any(b[0] <= x1 and b[2] >= x0 and b[1] <= y1 and b[3] >= y0 for b in boxes):
            keep.append(p)
    return keep


def convert_to_zlas(paths, zlas_folder, wkid=None, vert_wkid=None):
    """Convert .laz files to .zlas (skipping ones already converted). Returns failures."""
    os.makedirs(zlas_folder, exist_ok=True)
    failed = []
    todo = [p for p in paths if p.lower().endswith(".laz") and not os.path.exists(
        os.path.join(zlas_folder, os.path.splitext(os.path.basename(p))[0] + ".zlas"))]
    for i, p in enumerate(todo, 1):
        try:
            if wkid:  # assign the coordinate system the files should have
                sr = (arcpy.SpatialReference(wkid, vert_wkid) if vert_wkid
                      else arcpy.SpatialReference(wkid))
                arcpy.conversion.ConvertLas(p, zlas_folder, compression="ZLAS",
                                            define_coordinate_system="ALL_FILES",
                                            in_coordinate_system=sr)
            else:
                arcpy.conversion.ConvertLas(p, zlas_folder, compression="ZLAS")
        except Exception as err:
            msg = str(err).strip().splitlines()
            failed.append((p, msg[0] if msg else "error"))
            say(f"  could not convert {os.path.basename(p)}: {failed[-1][1]}")
        if i % 10 == 0 or i == len(todo):
            say(f"  converted {i}/{len(todo)}")
    return failed


def make_lidar_layers(laz_folder, grids, imagery, out_folder, cell=3.0, margin=150.0,
                      dry_run=False, convert_laz=False, zlas_folder=None, wkid=None,
                      vert_wkid=None):
    sr_img = arcpy.Describe(first_raster(imagery)).spatialReference
    if sr_img is None or sr_img.type != "Projected":
        raise ValueError("The imagery must be in a projected coordinate system")
    files = laz_files(laz_folder)
    n_laz = len(files)
    if not n_laz:
        raise ValueError(f"No .laz / .las files in {laz_folder}")
    with arcpy.da.SearchCursor(grids, ["OID@", "SHAPE@"], spatial_reference=sr_img) as cur:
        cells = [(oid, g) for oid, g in cur if g is not None]
    todo = [(oid, g) for oid, g in cells
            if not all(os.path.exists(p) for p in grid_paths(out_folder, oid).values())]
    say(f"{n_laz} LiDAR files, {len(cells)} grids; {len(cells) - len(todo)} already done, "
        f"{len(todo)} to do. Output: {out_folder} ({sr_img.name}, {cell:g} "
        f"{sr_img.linearUnitName} cells)")
    need = near_grids(files, todo, margin) if todo else []
    if convert_laz:
        done_z = sum(os.path.exists(os.path.join(zlas_folder, os.path.splitext(
            os.path.basename(p))[0] + ".zlas")) for p in need)
        say(f"{len(need)} LiDAR files are near the grids to do; {done_z} already converted "
            f"to zLAS, {len(need) - done_z} to convert into {zlas_folder}")
    if dry_run:
        mb = len(todo) * 3 * ((5280 + 2 * margin) / cell) ** 2 * 4 / 1024 ** 2
        say(f"Dry run: nothing made. Expect up to about {mb / 1024:,.1f} GB for the grids "
            "to do (less if your grids are smaller than a mile). Set DRY_RUN = False to run.")
        return dict(todo=[o for o, _ in todo], done=[], failed=[])

    if not todo:
        say("Nothing to do: every grid already has its LiDAR layers.")
        return dict(todo=[], done=[], failed=[])
    os.makedirs(out_folder, exist_ok=True)
    source = laz_folder
    if convert_laz:
        say("Converting LiDAR files near the grids to zLAS (Convert LAS tool)...")
        bad = convert_to_zlas(need, zlas_folder, wkid, vert_wkid)
        if bad and len(bad) == len([p for p in need if p.lower().endswith(".laz")]):
            raise ValueError(f"No file could be converted (first: {bad[0][1]}). This ArcGIS "
                             "Pro may not read .laz at all; tell whoever maintains the tool.")
        source = zlas_folder
    lasd = os.path.join(out_folder, "lidar.lasd")
    if os.path.exists(lasd):  # rebuilt each run, so newly converted files are included
        arcpy.management.Delete(lasd)
    say("Creating the LAS dataset (an index of the LiDAR files)...")
    arcpy.management.CreateLasDataset(source, lasd, "RECURSION",
                                      compute_stats="NO_COMPUTE_STATS",
                                      relative_paths="RELATIVE_PATHS")
    desc = arcpy.Describe(lasd)
    if getattr(desc, "fileCount", None) == 0:
        raise ValueError(
            "The LAS dataset holds no files: this version of ArcGIS Pro may not read .laz "
            "directly. Set CONVERT_LAZ = True and run again (it converts the tiles near "
            "the grids to zLAS first).")
    sr_las = desc.spatialReference
    if sr_las is None or sr_las.type != "Projected":
        raise ValueError("The LiDAR files have no projected coordinate system")
    m_img, m_las = sr_img.metersPerUnit, sr_las.metersPerUnit
    cell_las = cell * m_img / m_las
    ground = arcpy.management.MakeLasDatasetLayer(lasd, "cl_ground", class_code=[2])[0]
    surface = arcpy.management.MakeLasDatasetLayer(
        lasd, "cl_surface", class_code=[c for c in range(64) if c not in NOISE])[0]
    with open(os.path.join(out_folder, "lidar_layers.json"), "w") as f:
        json.dump({"z_to_m": m_las, "xy_to_m": m_img, "cell": cell,
                   "source": source}, f)

    tmp = os.path.join(out_folder, "_tmp")
    for k in LAYERS:
        os.makedirs(os.path.join(out_folder, k), exist_ok=True)
    saved = {k: getattr(arcpy.env, k) for k in ("extent", "outputCoordinateSystem",
                                                "pyramid", "rasterStatistics", "compression")}
    arcpy.env.pyramid, arcpy.env.rasterStatistics, arcpy.env.compression = "NONE", "NONE", "LZW"
    done, failed = [], []
    try:
        for i, (oid, g) in enumerate(todo, 1):
            t0 = time.time()
            out = grid_paths(out_folder, oid)
            try:
                os.makedirs(tmp, exist_ok=True)
                e = g.buffer(margin).projectAs(sr_las).extent
                arcpy.env.outputCoordinateSystem = sr_las
                arcpy.env.extent = arcpy.Extent(e.XMin, e.YMin, e.XMax, e.YMax)
                raw = {k: os.path.join(tmp, f"{k}.tif") for k in LAYERS}
                arcpy.conversion.LasDatasetToRaster(ground, raw["intensity"], "INTENSITY",
                                                    "BINNING AVERAGE LINEAR", "FLOAT",
                                                    "CELLSIZE", cell_las)
                arcpy.conversion.LasDatasetToRaster(ground, raw["dtm"], "ELEVATION",
                                                    "BINNING AVERAGE LINEAR", "FLOAT",
                                                    "CELLSIZE", cell_las)
                arcpy.conversion.LasDatasetToRaster(surface, raw["dsm"], "ELEVATION",
                                                    "BINNING MAXIMUM LINEAR", "FLOAT",
                                                    "CELLSIZE", cell_las)
                arcpy.env.extent = None
                arcpy.env.outputCoordinateSystem = None
                for k in LAYERS:  # onto the imagery's coordinate system, on a common grid
                    # positional: cell size, (no) geographic transformation, registration point
                    arcpy.management.ProjectRaster(raw[k], out[k], sr_img, "BILINEAR",
                                                   f"{cell} {cell}", None, "0 0")
                done.append(oid)
                say(f"[{i}/{len(todo)}] grid {oid} done ({time.time() - t0:,.0f} s)")
            except Exception as err:  # keep going; report at the end
                for p in out.values():  # no half-finished grids
                    if os.path.exists(p):
                        arcpy.management.Delete(p)
                msg = str(err).strip().splitlines()
                failed.append((oid, msg[0] if msg else "error"))
                say(f"[{i}/{len(todo)}] grid {oid} FAILED: {failed[-1][1]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    finally:
        for k, v in saved.items():
            setattr(arcpy.env, k, v)
    say(f"Finished: {len(done)} grids made, {len(failed)} failed. LiDAR layers folder: "
        f"{out_folder}")
    if failed:
        say("Run the script again to retry the failed grids.")
    return dict(todo=[o for o, _ in todo], done=done, failed=failed)


make_lidar_layers(LAZ_FOLDER, GRIDS, IMAGERY, OUT_FOLDER, CELL, MARGIN, DRY_RUN, CONVERT_LAZ,
                  ZLAS_FOLDER, LIDAR_WKID, LIDAR_VERT_WKID)
