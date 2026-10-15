"""download_usgs_tiles.py against a local web server and the stand-in arcpy."""
import functools
import http.server
import sys
import threading
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from . import fake_arcpy

ROOT = Path(__file__).resolve().parents[1]
PROJ = "GA_Statewide_2018_B18_DRRA"


def load_script(monkeypatch):
    monkeypatch.setitem(sys.modules, "arcpy", fake_arcpy)
    src = (ROOT / "arcgis" / "download_usgs_tiles.py").read_text()
    call = "download_usgs_tiles(LIST_FILE, OUT_FOLDER, PROJECT, AOI, AOI_BUFFER_M, DRY_RUN)"
    assert src.rstrip().endswith(call)
    ns = {"__name__": "download_usgs_tiles"}
    exec(compile(src.rstrip()[: -len(call)], "download_usgs_tiles.py", "exec"), ns)
    return ns


@pytest.fixture()
def server(tmp_path):
    root = tmp_path / "srv"
    (root / "TIFF").mkdir(parents=True)
    names = {}
    for x, y in [(42, 343), (43, 343), (48, 349)]:
        n = f"USGS_1M_17_x{x}y{y}_{PROJ}.tif"
        (root / "TIFF" / n).write_bytes(bytes([x]) * (1000 + x))
        names[(x, y)] = n
    other = "USGS_1M_17_x42y343_GA_Other_2010.tif"
    (root / "TIFF" / other).write_bytes(b"x" * 10)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/TIFF/"
    yield base, names, other, root
    srv.shutdown()


def write_xlsx(path, rows):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    wb.save(path)


def test_download_from_xlsx_list(monkeypatch, server, tmp_path):
    base, names, other, root = server
    ns = load_script(monkeypatch)
    # messy export: links scattered over columns, plus browse .jpg links to ignore
    rows = [["id", "NED", f"USGS 1 Meter 17 x{x}y{y}", "desc, with, commas", 123, base + n,
             None, base.replace("TIFF", "browse") + n.replace(".tif", ".jpg")]
            for (x, y), n in names.items()]
    rows.append(["id", "NED", "old project", base + other])
    lst = tmp_path / "list.xlsx"
    write_xlsx(lst, rows)
    assert len(ns["read_links"](str(lst))) == 4      # 3 + other project; no .jpg

    # AOI: one grid square inside tile x42 (UTM 17N metres, 420-430 km E, ~3,425 km N)
    grids = tmp_path / "grids.shp"
    gpd.GeoDataFrame({"G": [1]}, geometry=[box(421000, 3424000, 422600, 3425600)],
                     crs="EPSG:26917").to_file(grids)
    out = tmp_path / "DEM"
    dry = ns["download_usgs_tiles"](str(lst), str(out), PROJ, str(grids), 500, dry_run=True)
    assert [Path(d).name for _, d, _ in dry["todo"]] == [names[(42, 343)]]
    assert not out.exists()

    r = ns["download_usgs_tiles"](str(lst), str(out), PROJ, str(grids), 500)
    got = out / names[(42, 343)]
    assert got.read_bytes() == (root / "TIFF" / names[(42, 343)]).read_bytes()
    assert not list(out.glob("*.part")) and not r["failed"]

    # second run: nothing re-downloaded
    r2 = ns["download_usgs_tiles"](str(lst), str(out), PROJ, str(grids), 500)
    assert r2["todo"] == [] and len(r2["done"]) == 1

    # no AOI: every tile of the project
    r3 = ns["download_usgs_tiles"](str(lst), str(tmp_path / "all"), PROJ, None, 500)
    assert len(r3["done"]) == 3


def test_failed_tile_is_reported_not_fatal(monkeypatch, server, tmp_path):
    base, names, other, root = server
    ns = load_script(monkeypatch)
    lst = tmp_path / "list.txt"
    missing = base + f"USGS_1M_17_x99y999_{PROJ}.tif"
    lst.write_text(missing + "\n" + base + names[(43, 343)] + "\n")
    r = ns["download_usgs_tiles"](str(lst), str(tmp_path / "o"), PROJ, None, 500)
    assert len(r["done"]) == 1 and len(r["failed"]) == 1 and "x99y999" in r["failed"][0][0]
