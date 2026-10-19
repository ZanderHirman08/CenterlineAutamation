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
               "zlib", "yaml"}  # yaml: CLI-only, lazy
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


def split_tiles(ortho, tiles):
    """Split an ortho into four tiles with .aux.xml sidecars, like a county ortho."""
    tiles.mkdir()
    with rasterio.open(ortho) as src:
        h, w = src.height, src.width
        for i, (r0, c0) in enumerate([(0, 0), (0, w // 2), (h // 2, 0), (h // 2, w // 2)]):
            win = Window(c0, r0, w - w // 2 if c0 else w // 2, h - h // 2 if r0 else h // 2)
            prof = src.profile.copy()
            prof.update(width=win.width, height=win.height, transform=src.window_transform(win))
            with rasterio.open(tiles / f"tile_{i}.tif", "w", **prof) as dst:
                dst.write(src.read(window=win))
            (tiles / f"tile_{i}.tif.aux.xml").write_text("<PAMDataset></PAMDataset>")


@pytest.fixture(scope="module")
def tiled_scene(tmp_path_factory):
    """4-band scene split into four tiles, like a county ortho."""
    base = tmp_path_factory.mktemp("arc")
    paths = synthetic.write_scene(base, nir=True)
    split_tiles(paths["imagery"], base / "tiles")
    return dict(paths, tiles=base / "tiles")


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


def test_copy_checks_output_drive_and_space(monkeypatch, tiled_scene, tmp_path):
    from collections import namedtuple
    from shapely.geometry import box

    copy = load_copy_script(monkeypatch)
    x0, y0 = synthetic.X0, synthetic.Y0
    grids = tmp_path / "grids.shp"
    gpd.GeoDataFrame({"G": [1]}, geometry=[box(x0 + 100, y0 - 300, x0 + 300, y0 - 100)],
                     crs=synthetic.CRS).to_file(grids)
    # an output whose drive can't be used fails at once, even on a dry run
    import shutil as sh
    real = sh.disk_usage

    def broken(path):
        raise PermissionError(21, "The device is not ready")
    monkeypatch.setattr(sh, "disk_usage", broken)
    with pytest.raises(ValueError, match="Change OUT_TILES"):
        copy(str(tiled_scene["tiles"]), str(grids), 10, str(tmp_path / "x"), dry_run=True)

    # not enough space: dry run warns, a real run refuses before copying anything
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(sh, "disk_usage", lambda p: Usage(10, 10, 1000))
    fake_arcpy.warnings.clear()
    copy(str(tiled_scene["tiles"]), str(grids), 10, str(tmp_path / "y"), dry_run=True)
    assert any("Not enough space" in w for w in fake_arcpy.warnings)
    with pytest.raises(ValueError, match="Not enough space"):
        copy(str(tiled_scene["tiles"]), str(grids), 10, str(tmp_path / "y"))
    assert not (tmp_path / "y").exists()
    monkeypatch.setattr(sh, "disk_usage", real)


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


def test_evaluate_tool(toolbox, tiled_scene, tmp_path):
    """Checked roads (truth) vs the original roads and an aligned run."""
    run_tool(toolbox, tiled_scene, tmp_path)
    truth = tmp_path / "checked.shp"
    roads = synthetic.true_roads()
    gpd.GeoDataFrame({"Validated": ["Yes", "Yes", "No"]},
                     geometry=[g for g, _ in roads], crs=synthetic.CRS).to_file(truth)
    tool = toolbox.Toolbox().tools[1]()
    params = tool.getParameterInfo()
    p = {x.name: x for x in params}
    p["truth"].value = str(truth)
    p["candidates"].value = f"{tiled_scene['roads']};'{tmp_path / 'aligned.shp'}'"
    p["imagery"].value = str(tiled_scene["tiles"])
    p["grids"].value = str(tiled_scene["grids"])
    p["out_points"].value = str(tmp_path / "check.shp")
    fake_arcpy.messages.clear()
    tool.execute(params, None)
    text = "\n".join(fake_arcpy.messages)
    assert "2 checked roads" in text and "INTERSECTION" in text and "OPEN" in text

    pts = gpd.read_file(tmp_path / "check.shp")
    assert {"TRUTH_OID", "COVER", "ERR_1", "ERR_2"} <= set(pts.columns)
    both = pts.dropna(subset=["ERR_1", "ERR_2"])
    away = both[both.COVER != "intersection"]
    assert away.ERR_1.mean() > 4.0          # the original roads are off by several feet
    assert away.ERR_2.mean() < 1.0          # the aligned run is on the pavement
    assert set(pts.COVER) <= {"open", "shadow", "partial", "canopy", "intersection"}
    assert (pts.COVER == "intersection").any()
    # only the two checked roads (horizontal, vertical) were sampled, not the curve
    assert len(pts) < (1600 + 1000) / 5 + 10

    with pytest.raises(ValueError, match="Values found in Validated"):
        p["truth_where"].value = "Validated = 'Y'"
        tool.execute(params, None)


def test_second_pass(toolbox, tiled_scene, tmp_path):
    """A second run learns from the first run's trusted lines and still aligns well."""
    import shutil
    run_tool(toolbox, tiled_scene, tmp_path)
    first = tmp_path / "first"
    first.mkdir()
    for f in tmp_path.glob("aligned.*"):
        shutil.copy(f, first / f.name)
    fake_arcpy.messages.clear()
    out = run_tool(toolbox, tiled_scene, tmp_path / "p2", retrain_from=str(first / "aligned.shp"))
    text = "\n".join(fake_arcpy.messages)
    assert "Second pass: learning from" in text, text
    assert error_to_truth(out.geometry)[0] < 1.0
    import json
    model = json.loads((tmp_path / "p2" / "work" / "road_classifier.json").read_text())
    assert model["trained_from"].endswith("aligned.shp")

    # reusing that work folder with another earlier result: warned, not silently ignored
    fake_arcpy.warnings.clear()
    run_tool(toolbox, tiled_scene, tmp_path / "p2", retrain_from=str(tiled_scene["roads"]))
    assert any("second-pass setting is ignored" in w for w in fake_arcpy.warnings)


def test_lidar_sees_road_under_canopy(toolbox, tmp_path):
    """A stretch of road hidden by trees in the imagery: without LiDAR the line can't
    move there; with ground intensity and a smooth road bed it lands on the road."""
    from shapely.geometry import box

    base = tmp_path / "scene"
    paths = synthetic.write_scene(base, nir=True)
    synthetic.hide_under_canopy(paths["imagery"])
    split_tiles(paths["imagery"], base / "tiles")
    scene = dict(paths, tiles=base / "tiles")
    lidar = synthetic.write_lidar(tmp_path / "lidar")
    x0, y0, x1, y1 = synthetic.CANOPY
    hidden = box(x0 + 40, y0, x1 - 40, y1)

    def error_hidden(out):
        from shapely.geometry import LineString
        road = LineString([(synthetic.X0, synthetic.MID), (synthetic.X0 + 1600, synthetic.MID)])
        d = [ln.interpolate(s).distance(road) for ln in out.geometry
             for s in np.arange(0, ln.length, 5.0) if hidden.contains(ln.interpolate(s))]
        assert d
        return float(np.mean(d))

    plain = error_hidden(run_tool(toolbox, scene, tmp_path / "a"))
    fake_arcpy.messages.clear()
    with_lidar = error_hidden(run_tool(toolbox, scene, tmp_path / "b", lidar_folder=str(lidar)))
    assert any("LiDAR layers:" in m for m in fake_arcpy.messages)
    assert plain > 4.0, plain            # stays near the original line (7 ft off)
    assert with_lidar < 2.0, with_lidar

    # a work folder trained with LiDAR refuses to run without it
    with pytest.raises(ValueError, match="uses LiDAR"):
        run_tool(toolbox, scene, tmp_path / "b")


def test_make_lidar_layers_flow(monkeypatch, tiled_scene, tmp_path):
    """Control flow of make_lidar_layers.py with the LAS tools stubbed out: one set of
    rasters per grid, done grids skipped, a failing grid reported and cleaned up."""
    import types
    monkeypatch.setitem(sys.modules, "arcpy", fake_arcpy)
    src = (ROOT / "arcgis" / "make_lidar_layers.py").read_text()
    call = "make_lidar_layers(LAZ_FOLDER, GRIDS, IMAGERY, OUT_FOLDER, CELL, MARGIN, DRY_RUN)"
    assert src.rstrip().endswith(call)
    ns = {"__name__": "make_lidar_layers"}
    exec(compile(src.rstrip()[: -len(call)], "make_lidar_layers.py", "exec"), ns)

    laz = tmp_path / "laz"
    laz.mkdir()
    (laz / "a.laz").write_bytes(b"")
    calls = []
    sr = fake_arcpy.SpatialReference(synthetic.CRS)
    sr.metersPerUnit = 0.3048006096
    real_describe = fake_arcpy.Describe

    def describe(path):
        if str(path).endswith(".lasd"):
            return types.SimpleNamespace(fileCount=1, spatialReference=sr)
        return types.SimpleNamespace(spatialReference=sr)

    def to_raster(lyr, out, field, *a):
        calls.append((field, a[0]))
        if field == "ELEVATION" and fake_arcpy.env.extent.XMin > synthetic.X0 + 500:
            raise RuntimeError("ERROR 999999: something went wrong")   # second grid fails
        open(out, "w").close()

    def project(raw, out, *a):
        open(out, "w").close()

    monkeypatch.setattr(fake_arcpy, "Describe", describe)
    monkeypatch.setattr(fake_arcpy.management, "CreateLasDataset",
                        lambda src_, out, *a, **k: open(out, "w").close(), raising=False)
    monkeypatch.setattr(fake_arcpy.management, "MakeLasDatasetLayer",
                        lambda lasd, name, **k: [name], raising=False)
    monkeypatch.setattr(fake_arcpy.management, "ProjectRaster", project, raising=False)
    monkeypatch.setattr(fake_arcpy, "conversion",
                        types.SimpleNamespace(LasDatasetToRaster=to_raster), raising=False)
    for k in ("extent", "outputCoordinateSystem", "pyramid", "rasterStatistics",
              "compression"):
        monkeypatch.setattr(fake_arcpy.env, k, None, raising=False)
    sr.type = "Projected"

    out = tmp_path / "lidar"
    make = ns["make_lidar_layers"]
    dry = make(str(laz), str(tiled_scene["grids"]), str(tiled_scene["tiles"]), str(out),
               dry_run=True)
    assert dry["todo"] == [1, 2] and not out.exists()
    r = make(str(laz), str(tiled_scene["grids"]), str(tiled_scene["tiles"]), str(out))
    assert r["done"] == [1] and [o for o, _ in r["failed"]] == [2]
    assert sorted(p.name for p in (out / "dtm").iterdir()) == ["g1.tif"]
    assert {c[0] for c in calls} == {"INTENSITY", "ELEVATION"}
    import json
    assert json.loads((out / "lidar_layers.json").read_text())["xy_to_m"] == sr.metersPerUnit
    assert fake_arcpy.env.extent is None          # environment restored
    calls.clear()
    r2 = make(str(laz), str(tiled_scene["grids"]), str(tiled_scene["tiles"]), str(out))
    assert r2["todo"] == [2]                      # grid 1 is skipped
    monkeypatch.setattr(fake_arcpy, "Describe", real_describe)


def _tool(mod, cls_name):
    tool = next(t for t in mod.Toolbox().tools if t.__name__ == cls_name)()
    params = tool.getParameterInfo()
    return tool, params, {x.name: x for x in params}


def _edit(path, geoms=None, validated=None):
    """Edit the road layer like a user would: move lines, set Validated (row order kept)."""
    df = gpd.read_file(path)
    for rid, g in (geoms or {}).items():
        df.loc[df.RD_ID == rid, "geometry"] = g
    for rid, v in (validated or {}).items():
        df.loc[df.RD_ID == rid, "Validated"] = v
    df.to_file(path)


def test_training_day_cycle(toolbox, tiled_scene, tmp_path):
    """Start of day / edit / end of day, twice, then overnight training and its use."""
    import json
    from shapely.geometry import LineString

    x0, mid, y0 = synthetic.X0, synthetic.MID, synthetic.Y0
    true = {"H1": LineString([(x0, mid), (x0 + 400, mid)]),
            "H2": LineString([(x0 + 400, mid), (x0 + 900, mid)]),
            "H3": LineString([(x0 + 900, mid), (x0 + 1600, mid)]),
            "V1": LineString([(x0 + 400, y0), (x0 + 400, mid)]),
            "C1": synthetic.true_roads()[2][0]}
    roads = tmp_path / "federal.gpkg"   # long field names, like the real gdb
    fed = synthetic.local_roads()
    fed["Validated"] = "No"
    fed["AbandonedRoad"] = "No"
    fed.loc[fed.RD_ID == "H2", "geometry"] = true["H2"]     # this one is already right
    fed.to_file(roads)
    county = tmp_path / "county.shp"                         # the layer H3 is traced from
    gpd.GeoDataFrame({"N": [1]}, geometry=[true["H3"]], crs=fed.crs).to_file(county)
    folder = tmp_path / "training"
    folder.mkdir()

    def day(cls, today, **extra):
        tool, params, p = _tool(toolbox, cls)
        p["roads"].value = str(roads)
        p["training_folder"].value = str(folder)
        for k, v in extra.items():
            p[k].value = v
        fake_arcpy.messages.clear()
        mod_fn = {"StartOfDay": "snapshot", "EndOfDay": "collect"}[cls]
        orig = getattr(toolbox, mod_fn)
        setattr(toolbox, mod_fn, lambda *a, **k: orig(*a, today=today, **k))
        try:
            tool.execute(params, None)
        finally:
            setattr(toolbox, mod_fn, orig)
        return "\n".join(fake_arcpy.messages)

    # day 1: move four roads onto the pavement and check them
    assert "6 new road(s)" in day("StartOfDay", "2026-10-19")
    _edit(roads, {k: true[k] for k in ("H1", "V1", "H3", "C1")},
          {k: "Yes" for k in ("H1", "V1", "H3", "C1", "V2")})
    df = gpd.read_file(roads)
    df.loc[df.RD_ID == "V2", "AbandonedRoad"] = "Yes"        # checked, but abandoned
    df.to_file(roads)
    out = day("EndOfDay", "2026-10-19", out_fc=str(tmp_path / "train_roads.shp"),
              reference_fc=str(county))
    assert "4 newly checked and moved, 0 newly checked and left" in out, out
    assert "AbandonedRoad = 'Yes' are not used (1 checked road(s) left out)" in out
    assert "1 checked road(s) follow" in out
    t = gpd.read_file(tmp_path / "train_roads.shp")
    assert sorted(t.ROAD_OID) == [1, 3, 4, 6] and t.CHANGED.all()
    assert t.set_index("ROAD_OID").TRACED.to_dict() == {1: 0, 3: 1, 4: 0, 6: 0}
    assert t.SHIFT_AVG.between(2.5, 8).all(), t.SHIFT_AVG.tolist()

    # day 2: the morning snapshot keeps the true originals; check H2 without moving it,
    # and un-check H1
    assert "0 new road(s)" in day("StartOfDay", "2026-10-20")
    _edit(roads, validated={"H2": "Yes", "H1": "No"})
    out = day("EndOfDay", "2026-10-20")
    assert "0 newly checked and moved, 1 newly checked and left" in out, out
    assert "1 road(s) are no longer checked" in out
    store = json.loads((folder / "training_roads.json").read_text())["roads"]
    assert store["2"]["changed"] == 0 and store["2"]["shift_max"] < 0.5
    assert store["6"]["day"] == "2026-10-19" and store["6"]["original_known"]

    # overnight: train, test on the held-back grid, against the county layer too
    tool, params, p = _tool(toolbox, "TrainOvernight")
    p["imagery"].value = str(tiled_scene["tiles"])
    p["training_folder"].value = str(folder)
    p["grids"].value = str(tiled_scene["grids"])
    p["grid_id_field"].value = "GRID_ID"
    p["compare"].value = str(tiled_scene["roads"])
    fake_arcpy.messages.clear()
    res = toolbox.train(fake_arcpy, imagery=str(tiled_scene["tiles"]),
                        training_folder=str(folder), grids_fc=str(tiled_scene["grids"]),
                        grid_id_field="GRID_ID", bands=[1, 2, 3, 4],
                        compare=[str(tiled_scene["roads"])], today="2026-10-20")
    msgs = "\n".join(fake_arcpy.messages)
    assert len(res["holdout"]) == 1, msgs
    assert res["new"] < 1.0 and res["original"] > 2, res         # tool fixes the held-back roads
    assert res["promoted"] and "local_roads.shp" in msgs
    assert "Keep/move model" in msgs and "away from intersections" in msgs
    assert "did NOT trace" in msgs
    models = folder / "models"
    assert (models / "road_classifier.json").exists()
    assert (models / "road_classifier_2026-10-20.json").exists()

    # the toolbox button runs the same thing; the second night compares with the current model
    fake_arcpy.messages.clear()
    tool.execute(params, None)
    msgs = "\n".join(fake_arcpy.messages)
    assert "Tool, current model" in msgs, msgs
    hist = (models / "history.csv").read_text().splitlines()
    assert len(hist) == 3 and hist[0].startswith("day,")

    # the Align tool uses the trained model instead of training its own
    out = run_tool(toolbox, tiled_scene, tmp_path / "align",
                   classifier_file=str(models / "road_classifier.json"))
    assert any("Using the trained classifier" in m for m in fake_arcpy.messages)
    assert not (tmp_path / "align" / "work" / "road_classifier.json").exists()
    assert error_to_truth(out.geometry)[0] < 1.0

    # a keep/move model that never trusts a move keeps every original line
    clf = json.loads((models / "road_classifier.json").read_text())
    clf["move_model"] = dict(features=["shift_avg", "shift_max", "conf", "pct_valid",
                                       "width_med"], mean=[0] * 5, std=[1] * 5,
                             coef=[0] * 5, intercept=0.0, threshold=0.9)
    (models / "never_move.json").write_text(json.dumps(clf))
    out = run_tool(toolbox, tiled_scene, tmp_path / "never",
                   classifier_file=str(models / "never_move.json"))
    assert (out.MOVED == 0).all() and (out.MOVE_P == 0.5).all()
    assert out.REASON.str.contains("move_unlikely_to_help").all()
    assert error_to_truth(out.geometry)[0] > 4
