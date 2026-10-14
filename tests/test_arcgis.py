"""Run the generated ArcGIS toolbox (.pyt) against a stand-in arcpy."""
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.windows import Window

from . import fake_arcpy, synthetic
from .test_pipeline import error_to_truth

ROOT = Path(__file__).resolve().parents[1]
PYT = ROOT / "arcgis" / "CenterlineAlign.pyt"


def test_pyt_is_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_pyt.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_pyt_uses_only_arcgis_pro_packages():
    import ast
    allowed = {"__future__", "arcpy", "numpy", "scipy", "json", "logging", "os", "re", "time",
               "math", "dataclasses", "pathlib", "typing", "sys", "types",
               "yaml"}  # yaml: CLI-only, lazy
    tree = ast.parse(PYT.read_text())
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            used |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            used.add(node.module.split(".")[0])
    # transformers/torch are imported lazily only for GPU embeddings, never by the toolbox
    assert used - allowed <= {"torch", "transformers"}, used - allowed


@pytest.fixture()
def toolbox(monkeypatch):
    monkeypatch.setitem(sys.modules, "arcpy", fake_arcpy)
    fake_arcpy.messages.clear()
    fake_arcpy.warnings.clear()
    loader = importlib.machinery.SourceFileLoader("CenterlineAlign", str(PYT))
    spec = importlib.util.spec_from_loader("CenterlineAlign", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiled_scene(tmp_path_factory):
    """4-band scene split into four tiles with .aux.xml sidecars, like a county ortho."""
    base = tmp_path_factory.mktemp("arc")
    paths = synthetic.write_scene(base, nir=True)
    tiles = base / "tiles"
    tiles.mkdir()
    with rasterio.open(paths["imagery"]) as src:
        h, w = src.height, src.width
        for i, (r0, c0) in enumerate([(0, 0), (0, w // 2), (h // 2, 0), (h // 2, w // 2)]):
            win = Window(c0, r0, w - w // 2 if c0 else w // 2, h - h // 2 if r0 else h // 2)
            prof = src.profile.copy()
            prof.update(width=win.width, height=win.height, transform=src.window_transform(win))
            with rasterio.open(tiles / f"tile_{i}.tif", "w", **prof) as dst:
                dst.write(src.read(window=win))
            (tiles / f"tile_{i}.tif.aux.xml").write_text("<PAMDataset></PAMDataset>")
    return dict(paths, tiles=tiles)


def run_tool(mod, scene, tmp_path, **overrides):
    tool = mod.Toolbox().tools[0]()
    params = tool.getParameterInfo()
    p = {x.name: x for x in params}
    p["imagery"].value = str(scene["tiles"])
    p["roads"].value = str(scene["roads"])
    p["road_id_field"].value = "RD_ID"
    p["grids"].value = str(scene["grids"])
    p["grid_id_field"].value = "GRID_ID"
    p["work_folder"].value = str(tmp_path / "work")
    p["out_fc"].value = str(tmp_path / "aligned.shp")
    p["training_grids"].value = 2
    for k, v in overrides.items():
        p[k].value = v
    tool.execute(params, None)
    return gpd.read_file(tmp_path / "aligned.shp")


def test_toolbox_parameters(toolbox):
    params = toolbox.Toolbox().tools[0]().getParameterInfo()
    names = [p.name for p in params]
    assert len(names) == len(set(names))
    assert {"imagery", "roads", "road_id_field", "grids", "grid_id_field", "work_folder",
            "out_fc", "nir", "fresh"} <= set(names)
    assert all(p.datatype for p in params)


def test_toolbox_aligns_roads(toolbox, tiled_scene, tmp_path):
    out = run_tool(toolbox, tiled_scene, tmp_path)
    assert list(out.columns[:9]) == ["SRC_ID", "GRID_ID", "SHIFT_AVG", "SHIFT_MAX", "CONF",
                                     "PCT_VALID", "WIDTH_MED", "REVIEW", "REASON"]
    before = gpd.read_file(tiled_scene["roads"])
    assert set(out.SRC_ID) == set(before.RD_ID)
    mean, p95 = error_to_truth(out.geometry)
    assert mean < 1.0 and p95 < 2.0, (mean, p95)
    assert out.REVIEW.sum() == 0
    assert any("NIR: yes" in m for m in fake_arcpy.messages)
    assert any("Classifier trained" in m for m in fake_arcpy.messages)  # engine logs reach Pro
    assert any("Alignment check" in m for m in fake_arcpy.messages)
    assert any(m.startswith("Summary of") for m in fake_arcpy.messages)
    assert not fake_arcpy.warnings, fake_arcpy.warnings

    # diagnostic rasters: the chip the tool read lines up with the source imagery
    diag = tmp_path / "work" / "diagnostics"
    with rasterio.open(diag / "chip_1.tif") as c, rasterio.open(tiled_scene["imagery"]) as src:
        assert c.crs == src.crs
        win = rasterio.windows.from_bounds(*c.bounds, transform=src.transform)
        expect = src.read(window=win.round_offsets().round_lengths(), boundless=True)
        got = c.read()
        assert got.shape == expect.shape
        assert np.array_equal(got, expect)
    with rasterio.open(diag / "labels_1.tif") as lab, \
            rasterio.open(tiled_scene["imagery"]) as src:
        labels = lab.read(1)
        win = rasterio.windows.from_bounds(*lab.bounds, transform=src.transform)
        blue = src.read(3, window=win.round_offsets().round_lengths(), boundless=True)
        # road-labelled pixels are mostly pavement (blue ~120), not grass (~44)
        assert np.median(blue[labels == 1]) > 90
        assert np.median(blue[labels == 2]) < 70

    # a second run reuses the saved grids and classifier
    fake_arcpy.messages.clear()
    out2 = run_tool(toolbox, tiled_scene, tmp_path)
    assert sum("already done" in m for m in fake_arcpy.messages) == 2
    assert len(out2) == len(out)


def test_toolbox_matches_cli(toolbox, tiled_scene, tmp_path):
    """Same inputs through the toolbox and the command-line pipeline agree closely."""
    from centerline_align import pipeline
    from centerline_align.config import Config

    arc = run_tool(toolbox, tiled_scene, tmp_path / "arc")
    cfg = Config.from_dict({
        "inputs": {"imagery": str(tiled_scene["tiles"]), "roads": str(tiled_scene["roads"]),
                   "grids": str(tiled_scene["grids"]), "road_id_field": "RD_ID",
                   "grid_id_field": "GRID_ID"},
        "bands": {"nir": 4}, "classifier": {"training_grids": 2},
        "output": {"folder": str(tmp_path / "cli")}})
    pipeline.run(cfg)
    cli = gpd.read_file(tmp_path / "cli" / "aligned_centerlines.gpkg")
    a = arc.sort_values(["SRC_ID", "GRID_ID"]).reset_index(drop=True)
    c = cli.sort_values(["src_id", "grid_id"]).reset_index(drop=True)
    assert len(a) == len(c)
    d = [ga.hausdorff_distance(gc) for ga, gc in zip(a.geometry, c.geometry)]
    assert max(d) < 0.05, d


def test_toolbox_skips_bad_tiles_and_warns_about_gaps(toolbox, tiled_scene, tmp_path):
    import shutil
    from shapely.geometry import box

    tiles = tmp_path / "tiles"
    shutil.copytree(tiled_scene["tiles"], tiles)
    (tiles / "aaa_corrupt.tif").write_bytes(b"not a tiff")
    grids = gpd.read_file(tiled_scene["grids"])
    x0, y0 = synthetic.X0, synthetic.Y0
    extra = gpd.GeoDataFrame({"GRID_ID": ["G9"]},
                             geometry=[box(x0 + 5000, y0 - 1000, x0 + 5800, y0)], crs=grids.crs)
    gpath = tmp_path / "grids.shp"
    gpd.GeoDataFrame(pd.concat([grids, extra], ignore_index=True), crs=grids.crs).to_file(gpath)
    out = run_tool(toolbox, dict(tiled_scene, tiles=tiles, grids=gpath), tmp_path / "r")
    warn = "\n".join(fake_arcpy.warnings)
    assert "aaa_corrupt.tif" in warn and "were skipped" in warn
    assert "1 of 3 grids are not fully covered" in warn and "G9 (0%)" in warn
    assert error_to_truth(out.geometry)[0] < 1.0      # the good tiles were still used


def test_toolbox_probability_rasters(toolbox, tiled_scene, tmp_path):
    run_tool(toolbox, tiled_scene, tmp_path, diagnostics=True)
    probs = sorted((tmp_path / "work" / "diagnostics").glob("road_prob_*.tif"))
    assert [p.name for p in probs] == ["road_prob_G1.tif", "road_prob_G2.tif"]
    with rasterio.open(probs[0]) as r:
        assert r.crs is not None and r.read(1).max() > 200


def test_toolbox_no_nir_and_bad_band(toolbox, tiled_scene, tmp_path):
    out = run_tool(toolbox, tiled_scene, tmp_path / "a", nir=None)
    assert any("NIR: no" in m for m in fake_arcpy.messages)
    assert error_to_truth(out.geometry)[0] < 1.0
    with pytest.raises(ValueError, match="Band 5"):
        run_tool(toolbox, tiled_scene, tmp_path / "b", nir=5)


def load_copy_script(monkeypatch):
    monkeypatch.setitem(sys.modules, "arcpy", fake_arcpy)
    src = (ROOT / "arcgis" / "copy_aoi_tiles.py").read_text()
    call = "copy_aoi_tiles(STATE_TILES, AOI, BUFFER, OUT_TILES, DRY_RUN)"
    assert src.rstrip().endswith(call)
    ns = {"__name__": "copy_aoi_tiles"}
    exec(compile(src.rstrip()[: -len(call)], "copy_aoi_tiles.py", "exec"), ns)
    return ns["copy_aoi_tiles"]


def test_copy_aoi_tiles(monkeypatch, tiled_scene, tmp_path):
    """Only tiles touching the buffered AOI polygons are copied, with their sidecars."""
    from shapely.geometry import box

    copy = load_copy_script(monkeypatch)
    # tiles: 800 x 500 ft each; tile_0 top-left, tile_1 top-right, tile_2/3 below
    x0, y0 = synthetic.X0, synthetic.Y0
    grids = tmp_path / "grids.shp"
    cells = [box(x0 + 100, y0 - 300, x0 + 300, y0 - 100),     # inside tile_0
             box(x0 + 100, y0 - 250, x0 + 300, y0 - 50)]      # also tile_0
    gpd.GeoDataFrame({"G": [1, 2]}, geometry=cells, crs=synthetic.CRS).to_file(grids)
    out = tmp_path / "a"
    assert copy(str(tiled_scene["tiles"]), str(grids), 50, str(out), dry_run=True)["matched"] == 1
    assert not out.exists()  # dry run copies nothing
    r = copy(str(tiled_scene["tiles"]), str(grids), 50, str(out))
    assert r["copied"] == 1 and not r["failed"] and not r["uncovered"]
    assert sorted(p.name for p in out.iterdir()) == ["tile_0.tif", "tile_0.tif.aux.xml"]

    # a second, separate grid square in the bottom-right tile adds only that tile
    cells.append(box(x0 + 1200, y0 - 900, x0 + 1400, y0 - 700))
    gpd.GeoDataFrame({"G": [1, 2, 3]}, geometry=cells, crs=synthetic.CRS).to_file(grids)
    out2 = tmp_path / "b"
    assert copy(str(tiled_scene["tiles"]), str(grids), 50, str(out2))["copied"] == 2
    assert {p.name for p in out2.iterdir()} >= {"tile_0.tif", "tile_3.tif"}

    # originals untouched, and the output can't be inside the source folder
    assert sorted(p.name for p in tiled_scene["tiles"].iterdir()) == sorted(
        [f"tile_{i}.tif" for i in range(4)] + [f"tile_{i}.tif.aux.xml" for i in range(4)])
    with pytest.raises(ValueError, match="outside STATE_TILES"):
        copy(str(tiled_scene["tiles"]), str(grids), 50, str(tiled_scene["tiles"] / "sub"))

    # a 250 ft buffer crosses the tile edge at y0 - 500: the first squares pull in
    # the tile below them (tile_2), the third the tile above it (tile_1)
    out3 = tmp_path / "c"
    assert copy(str(tiled_scene["tiles"]), str(grids), 250, str(out3))["copied"] == 4


def test_copy_keeps_subfolders(monkeypatch, tiled_scene, tmp_path):
    """Same-named tiles in different subfolders (e.g. tidal / untidal) both survive."""
    import shutil
    from shapely.geometry import box

    copy = load_copy_script(monkeypatch)
    src = tmp_path / "imagery"
    for sub in ("tidal", "untidal"):
        shutil.copytree(tiled_scene["tiles"], src / sub)
    x0, y0 = synthetic.X0, synthetic.Y0
    grids = tmp_path / "grids.shp"
    gpd.GeoDataFrame({"G": [1]}, geometry=[box(x0 + 100, y0 - 300, x0 + 300, y0 - 100)],
                     crs=synthetic.CRS).to_file(grids)
    out = tmp_path / "out"
    r = copy(str(src), str(grids), 10, str(out))
    assert r["copied"] == 2
    assert (out / "tidal" / "tile_0.tif").exists() and (out / "untidal" / "tile_0.tif").exists()
    assert (out / "tidal" / "tile_0.tif.aux.xml").exists()


def test_copy_reports_problems_and_gaps(monkeypatch, tiled_scene, tmp_path):
    """A bad tile doesn't stop the copy; unread formats and uncovered grids are reported."""
    import shutil
    from shapely.geometry import box

    copy = load_copy_script(monkeypatch)
    src = tmp_path / "state"
    shutil.copytree(tiled_scene["tiles"], src)
    (src / "aaa_corrupt.tif").write_bytes(b"not a tiff")   # sorts first: used to stop the run
    (src / "west.sid").write_bytes(b"")                    # .sid is now considered
    (src / "east.ecw").write_bytes(b"")                    # unknown format: reported
    x0, y0 = synthetic.X0, synthetic.Y0
    grids = tmp_path / "grids.shp"
    cells = [box(x0 + 100, y0 - 300, x0 + 300, y0 - 100),        # covered
             box(x0 + 1500, y0 - 300, x0 + 1700, y0 - 100),      # half off the imagery
             box(x0 + 5000, y0 - 300, x0 + 5200, y0 - 100)]      # no imagery at all
    gpd.GeoDataFrame({"G": [1, 2, 3]}, geometry=cells, crs=synthetic.CRS).to_file(grids)
    fake_arcpy.messages.clear()
    r = copy(str(src), str(grids), 10, str(tmp_path / "out"))
    assert r["copied"] == 2                                  # tile_0 and tile_1, after the bad one
    assert {os.path.basename(p) for p, _ in r["failed"]} == {"aaa_corrupt.tif", "west.sid"}
    assert r["other"] == {".ecw": 1}
    cov = dict(r["uncovered"])
    assert set(cov) == {2, 3} and cov[3] == 0 and 0.4 < cov[2] < 0.6
    text = "\n".join(fake_arcpy.messages)
    assert "could not be read" in text and ".ecw (1)" in text
    assert "No imagery, OBJECTID: 3" in text and "2 (50%)" in text

    # running again after a fix re-copies nothing that is already there
    fake_arcpy.messages.clear()
    copy(str(src), str(grids), 10, str(tmp_path / "out"))
    assert any("already there" in m for m in fake_arcpy.messages)
