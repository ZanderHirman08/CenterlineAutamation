"""Download USGS 3DEP tiles (1 m DEM GeoTIFFs, LiDAR LAZ) from a product list.

Run in ArcGIS Pro: Analysis tab > Python > Python Window. Edit the settings
below, paste the whole file into the window, and press Enter.

LIST_FILE is the results list exported from The National Map downloader
(.xlsx, .csv or .txt). The script pulls every download link out of it, so the
column layout doesn't matter. No USGS login or API key is needed: these
links are public.

* Only tiles from PROJECT are kept (so different LiDAR collections don't mix).
* With AOI set, tiles that can't touch your grids are skipped (1 m DEM tiles
  and LiDAR tiles named like ..._e1372n0996.laz). The check errs on the side
  of downloading: a tile is only skipped if it is clearly away from every grid.
  Tiles whose names don't give a position are kept.
* Tiles already downloaded (same size as on the server) are skipped, so the
  script can be stopped and run again. Downloads go to a .part file first and
  are only renamed when complete.
* Several files download at once (WORKERS); failed downloads are retried,
  then listed at the end.
Needs only ArcGIS Pro's own Python.
"""
import os
import re
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import arcpy

# ---- settings ---------------------------------------------------------------
LIST_FILE = r"C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\USGS\1MDEM_Glynn_data_links.xlsx"
OUT_FOLDER = r"C:\Users\zhirman\Desktop\Centerline\Glynn\imagery\USGS\DEM"
PROJECT = "GA_Statewide_2018_B18_DRRA"   # keep only this project's tiles; None = keep all
AOI = r"C:\Users\zhirman\Desktop\Centerline\Glynn\data\Inputs.gdb\Glynn_Grid_1mi"  # None = no filter
AOI_BUFFER_M = 500                        # metres of margin around the grids
DRY_RUN = True                            # True: only list what would be downloaded
WORKERS = 4                               # files fetched at the same time
# -----------------------------------------------------------------------------

URL_RE = re.compile(r"https?://[^\s\"'<>,;|]+?\.(?:tif|tiff|laz|las|zip|img)\b", re.IGNORECASE)
# Tile names that encode the tile's position, so tiles away from the AOI can be skipped:
#   1 m DEM:  USGS_1M_<utm zone>_x<easting/10 km>y<northing/10 km>_...   (NAD83 / UTM, 10 km)
#   LiDAR:    ..._e<easting km>n<northing km>.laz                        (CONUS Albers, 1 km)
DEM_TILE_RE = re.compile(r"_1M_(\d{1,2})_x(\d+)y(\d+)_", re.IGNORECASE)
LPC_TILE_RE = re.compile(r"_e(\d{3,4})n(\d{3,4})\.la[sz]$", re.IGNORECASE)
ALBERS = 6350  # NAD83(2011) / CONUS Albers


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


def aoi_boxes(aoi, epsg, buffer_m):
    """Bounding box of each AOI polygon in the given (metre) coordinate system, grown by
    buffer_m."""
    sr = arcpy.SpatialReference(epsg)
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


def tile_area(name):
    """(epsg, xmin, ymin, xmax, ymax) that surely contains the tile, or None if the name
    doesn't say where the tile is. Generous on purpose: the numbers may mark a tile's
    corner or its centre, so the area allows for either."""
    m = DEM_TILE_RE.search(name)
    if m:
        zone, x, y = (int(v) for v in m.groups())
        size = 10000.0
        # x is the tile's west edge; y may be its north or south edge, so allow both
        return (26900 + zone, x * size, y * size - size, x * size + size, y * size + size)
    m = LPC_TILE_RE.search(name)
    if m:
        e, n = (int(v) * 1000.0 for v in m.groups())
        return (ALBERS, e - 1000.0, n - 1000.0, e + 1000.0, n + 1000.0)
    return None


def may_touch(url, boxes_by_epsg, aoi, buffer_m):
    """False only when a tile clearly misses every AOI box."""
    area = tile_area(os.path.basename(url)) if aoi is not None else None
    if area is None:
        return True  # can't place it: keep it
    epsg, x0, y0, x1, y1 = area
    if epsg not in boxes_by_epsg:
        boxes_by_epsg[epsg] = aoi_boxes(aoi, epsg, buffer_m)
    return any(b[0] <= x1 and b[2] >= x0 and b[1] <= y1 and b[3] >= y0
               for b in boxes_by_epsg[epsg])


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
            print(f"    retrying {os.path.basename(url)} ({attempt}/{tries - 1}) after: {e}")
            time.sleep(5 * attempt)


def download_usgs_tiles(list_file, out_folder, project=None, aoi=None, buffer_m=500,
                        dry_run=False, workers=4):
    links = read_links(list_file)
    say(f"{len(links)} download links found in {os.path.basename(list_file)}")
    if project:
        kept = [u for u in links if project.lower() in u.lower()]
        if len(kept) < len(links):
            say(f"  {len(links) - len(kept)} links from other projects ignored")
        links = kept
    boxes_by_epsg = {}
    near = [u for u in links if may_touch(u, boxes_by_epsg, aoi, buffer_m)]
    if aoi is not None:
        say(f"  {len(links) - len(near)} tiles skipped: they don't touch the AOI")
    links = near
    if not links:
        say("Nothing to download.")
        return dict(todo=[], done=[], failed=[])

    if not dry_run:
        os.makedirs(out_folder, exist_ok=True)
    todo, done, failed, total = [], [], [], 0
    say(f"Checking file sizes on the server for {len(links)} tiles...")
    # size checks run 8 at a time; messages are only written from this thread
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(remote_size, u): u for u in links}
        for k, fut in enumerate(as_completed(futures), 1):
            url = futures[fut]
            dest = os.path.join(out_folder, os.path.basename(url))
            if k % 100 == 0:
                say(f"  {k}/{len(links)} checked")
            try:
                size = fut.result()
            except Exception as e:  # e.g. the server was busy
                failed.append((url, f"size check failed: {e}"))
                size = None
            total += size or 0
            # downloads are renamed from .part only when complete, so an existing file
            # whose size can't be checked is kept
            if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
                done.append(dest)
            else:
                todo.append((url, dest, size))
    order = {u: i for i, u in enumerate(links)}
    todo.sort(key=lambda t: order[t[0]])
    gb = total / 1024 ** 3
    say(f"{len(links)} tiles, {gb:,.1f} GB in total; {len(done)} already downloaded, "
        f"{len(todo)} to download" + (f" ({len(failed)} of unknown size)" if failed else ""))
    if dry_run:
        for url, _, size in todo[:10]:
            say(f"  {os.path.basename(url)}  ({(size or 0) / 1024 ** 2:,.0f} MB)")
        if len(todo) > 10:
            say(f"  ... and {len(todo) - 10} more")
        if failed:
            say(f"WARNING: the size check failed for {len(failed)} tile(s); "
                "the real run will try to download them anyway:")
            for url, e in failed[:10]:
                say(f"  {os.path.basename(url)}: {e}")
        say("Dry run: nothing downloaded. Set DRY_RUN = False to download.")
        return dict(todo=todo, done=done, failed=failed)

    failed = []  # size-check problems were reported above; the download decides now
    t_start, got_bytes = time.time(), 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(download, url, dest, size): (url, dest, size)
                   for url, dest, size in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            url, dest, size = futures[fut]
            try:
                fut.result()
                done.append(dest)
                got_bytes += size or 0
                rate = got_bytes / 1024 ** 2 / max(1.0, time.time() - t_start)
                say(f"[{i}/{len(todo)}] {os.path.basename(url)} done "
                    f"({got_bytes / 1024 ** 3:,.1f} GB so far, {rate:,.0f} MB/s)")
            except Exception as e:
                failed.append((url, str(e)))
                say(f"[{i}/{len(todo)}] {os.path.basename(url)} FAILED: {e}")
    say(f"Finished: {len(done)} of {len(links)} tiles in {out_folder}")
    if failed:
        say(f"WARNING: {len(failed)} tile(s) failed. Run the script again to retry them:")
        for url, e in failed[:15]:
            say(f"  {os.path.basename(url)}: {e}")
    return dict(todo=todo, done=done, failed=failed)


download_usgs_tiles(LIST_FILE, OUT_FOLDER, PROJECT, AOI, AOI_BUFFER_M, DRY_RUN, WORKERS)
