"""Download USGS 3DEP tiles (1 m DEM GeoTIFFs, LiDAR LAZ) from a product list.

Run in ArcGIS Pro: Analysis tab > Python > Python Window. Edit the settings
below, paste the whole file into the window, and press Enter.

LIST_FILE is the results list exported from The National Map downloader
(.xlsx, .csv or .txt). The script pulls every download link out of it, so the
column layout doesn't matter. No USGS login or API key is needed: these
links are public.

* Only tiles from PROJECT are kept (so different LiDAR collections don't mix).
* With AOI set, 1 m DEM tiles that can't touch your grids are skipped. The
  check errs on the side of downloading: a tile is only skipped if it is
  clearly away from every grid.
* Tiles already downloaded (same size as on the server) are skipped, so the
  script can be stopped and run again. Downloads go to a .part file first and
  are only renamed when complete.
* Failed downloads are retried, then listed at the end.
Needs only ArcGIS Pro's own Python.
"""
import os
import re
import time
import urllib.request
import zipfile

import arcpy

# ---- settings ---------------------------------------------------------------
LIST_FILE = r"C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\USGS\1MDEM_Glynn_data_links.xlsx"
OUT_FOLDER = r"C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\USGS\DEM"
PROJECT = "GA_Statewide_2018_B18_DRRA"   # keep only this project's tiles; None = keep all
AOI = r"C:\Users\zhirman\Desktop\Centerline\Glynn\data\Inputs.gdb\Glynn_Grid_1mi"  # None = no filter
AOI_BUFFER_M = 500                        # metres of margin around the grids
DRY_RUN = True                            # True: only list what would be downloaded
# -----------------------------------------------------------------------------

URL_RE = re.compile(r"https?://[^\s\"'<>,;|]+?\.(?:tif|tiff|laz|las|zip|img)\b", re.IGNORECASE)
# USGS 1 m DEM tile names: USGS_1M_<utm zone>_x<easting/10km>y<northing/10km>_...
DEM_TILE_RE = re.compile(r"_1M_(\d{1,2})_x(\d+)y(\d+)_", re.IGNORECASE)
TILE_M = 10000.0


def say(m):
    arcpy.AddMessage(m)
    print(m)


def read_links(path):
    """Every download link in an .xlsx / .csv / .txt list, in order, without repeats."""
    texts = []
    if path.lower().endswith((".xlsx", ".xlsm")):
        with zipfile.ZipFile(path) as z:  # an .xlsx is a zip of XML files
            for name in z.namelist():
                if name.startswith("xl/") and name.endswith(".xml"):
                    texts.append(z.read(name).decode("utf-8", "replace"))
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            texts.append(f.read())
    seen, links = set(), []
    for t in texts:
        for u in URL_RE.findall(t.replace("&amp;", "&")):
            if u not in seen:
                seen.add(u)
                links.append(u)
    return links


def aoi_boxes_utm(aoi, zone, buffer_m):
    """Bounding box of each AOI polygon in NAD83 / UTM <zone> N, grown by buffer_m."""
    sr = arcpy.SpatialReference(26900 + zone)
    boxes = []
    with arcpy.da.SearchCursor(aoi, ["SHAPE@"]) as cur:
        for (g,) in cur:
            if g is None:
                continue
            e = g.projectAs(sr).extent
            boxes.append((e.XMin - buffer_m, e.YMin - buffer_m,
                          e.XMax + buffer_m, e.YMax + buffer_m))
    if not boxes:
        raise ValueError(f"No features in {aoi} (check selection / definition query)")
    return boxes


def may_touch(url, boxes_by_zone, aoi, buffer_m):
    """False only when a 1 m DEM tile clearly misses every AOI box."""
    m = DEM_TILE_RE.search(os.path.basename(url))
    if not m or aoi is None:
        return True  # not a 1 m DEM tile name we can place: keep it
    zone, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if zone not in boxes_by_zone:
        boxes_by_zone[zone] = aoi_boxes_utm(aoi, zone, buffer_m)
    x0 = x * TILE_M
    # the y number may mark the tile's north or south edge; accept either reading
    y_lo, y_hi = y * TILE_M - TILE_M, y * TILE_M + TILE_M
    return any(b[0] <= x0 + TILE_M and b[2] >= x0 and b[1] <= y_hi and b[3] >= y_lo
               for b in boxes_by_zone[zone])


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n else None


def download(url, dest, size, tries=3):
    part = dest + ".part"
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            got = os.path.getsize(part)
            if size is not None and got != size:
                raise IOError(f"incomplete: {got:,} of {size:,} bytes")
            os.replace(part, dest)
            return
        except Exception as e:
            if os.path.exists(part):
                os.remove(part)
            if attempt == tries:
                raise
            say(f"    retrying ({attempt}/{tries - 1}) after: {e}")
            time.sleep(5 * attempt)


def download_usgs_tiles(list_file, out_folder, project=None, aoi=None, buffer_m=500,
                        dry_run=False):
    links = read_links(list_file)
    say(f"{len(links)} download links found in {os.path.basename(list_file)}")
    if project:
        kept = [u for u in links if project.lower() in u.lower()]
        if len(kept) < len(links):
            say(f"  {len(links) - len(kept)} links from other projects ignored")
        links = kept
    boxes_by_zone = {}
    near = [u for u in links if may_touch(u, boxes_by_zone, aoi, buffer_m)]
    if aoi is not None:
        say(f"  {len(links) - len(near)} tiles skipped: they don't touch the AOI")
    links = near
    if not links:
        say("Nothing to download.")
        return dict(todo=[], done=[], failed=[])

    if not dry_run:
        os.makedirs(out_folder, exist_ok=True)
    todo, done, failed, total = [], [], [], 0
    for url in links:
        dest = os.path.join(out_folder, os.path.basename(url))
        try:
            size = remote_size(url)
        except Exception as e:
            failed.append((url, f"size check failed: {e}"))
            continue
        total += size or 0
        if os.path.exists(dest) and size is not None and os.path.getsize(dest) == size:
            done.append(dest)
        else:
            todo.append((url, dest, size))
    gb = total / 1024 ** 3
    say(f"{len(links)} tiles, {gb:,.1f} GB in total; {len(done)} already downloaded, "
        f"{len(todo)} to download")
    if dry_run:
        for url, _, size in todo[:10]:
            say(f"  {os.path.basename(url)}  ({(size or 0) / 1024 ** 2:,.0f} MB)")
        if len(todo) > 10:
            say(f"  ... and {len(todo) - 10} more")
        say("Dry run: nothing downloaded. Set DRY_RUN = False to download.")
        return dict(todo=todo, done=done, failed=failed)

    for i, (url, dest, size) in enumerate(todo, 1):
        say(f"[{i}/{len(todo)}] {os.path.basename(url)} ({(size or 0) / 1024 ** 2:,.0f} MB)")
        t0 = time.time()
        try:
            download(url, dest, size)
            done.append(dest)
            say(f"    done in {time.time() - t0:,.0f} s")
        except Exception as e:
            failed.append((url, str(e)))
            say(f"    FAILED: {e}")
    say(f"Finished: {len(done)} of {len(links)} tiles in {out_folder}")
    if failed:
        say(f"WARNING: {len(failed)} tile(s) failed. Run the script again to retry them:")
        for url, e in failed[:15]:
            say(f"  {os.path.basename(url)}: {e}")
    return dict(todo=todo, done=done, failed=failed)


download_usgs_tiles(LIST_FILE, OUT_FOLDER, PROJECT, AOI, AOI_BUFFER_M, DRY_RUN)
