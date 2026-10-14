import geopandas as gpd
import numpy as np
import pytest
from shapely.ops import unary_union

from centerline_align import pipeline
from centerline_align.config import Config

from . import synthetic


def make_cfg(paths, out, **extra):
    raw = {
        "inputs": {"imagery": str(paths["imagery"]), "roads": str(paths["roads"]),
                   "grids": str(paths["grids"]), "road_id_field": "RD_ID",
                   "grid_id_field": "GRID_ID"},
        "classifier": {"samples_per_class": 6000, "training_grids": 2},
        "output": {"folder": str(out), "write_stations": True},
    }
    raw.update(extra)
    return Config.from_dict(raw)


def error_to_truth(lines, keep_away_from_nodes=40.0):
    """Mean / 95th pct distance from output vertices to the true centres, ignoring intersections."""
    truth = unary_union([g for g, _ in synthetic.true_roads()])
    nodes = [(synthetic.X0 + 400, synthetic.MID), (synthetic.X0 + 900, synthetic.MID)]
    from shapely.geometry import Point
    d = []
    for ln in lines:
        for s in np.arange(0, ln.length, 5.0):
            p = ln.interpolate(s)
            if min(p.distance(Point(n)) for n in nodes) < keep_away_from_nodes:
                continue
            d.append(p.distance(truth))
    d = np.array(d)
    return d.mean(), np.percentile(d, 95)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return synthetic.write_scene(tmp_path_factory.mktemp("scene"))


def test_lines_move_onto_pavement(scene, tmp_path):
    before = gpd.read_file(scene["roads"])
    b_mean, _ = error_to_truth(before.geometry)
    assert b_mean > 4.0  # the local file really is off

    cfg = make_cfg(scene, tmp_path)
    result = pipeline.run(cfg)
    out = gpd.read_file(tmp_path / "aligned_centerlines.gpkg")
    assert (tmp_path / "aligned_centerlines.shp").exists()
    assert set(out.src_id) == set(before.RD_ID)
    a_mean, a_p95 = error_to_truth(out.geometry)
    assert a_mean < 1.0, (b_mean, a_mean)
    assert a_p95 < 2.0
    assert result["lines"] == len(out)
    assert out.review.sum() == 0, out[["src_id", "conf", "reason"]]


def test_network_stays_connected(scene, tmp_path):
    from scipy.spatial import cKDTree

    pipeline.run(make_cfg(scene, tmp_path))
    out = gpd.read_file(tmp_path / "aligned_centerlines.gpkg")
    before = gpd.read_file(scene["roads"])
    b_ends = np.array([[g.coords[0], g.coords[-1]] for g in before.geometry]).reshape(-1, 2)
    nodes = {tuple(np.round(b_ends[i], 3)) for pair in cKDTree(b_ends).query_pairs(0.5) for i in pair}
    nodes.add((synthetic.X0 + 800, synthetic.MID + 7.0))  # H2 is cut by the grid edge
    ends = np.array([[g.coords[0], g.coords[-1]] for g in out.geometry]).reshape(-1, 2)
    for node in nodes:
        near = ends[np.linalg.norm(ends - np.array(node), axis=1) < 15.0]
        assert len(near) >= 2, node
        assert np.ptp(near, axis=0).max() < 1e-6, (node, near)


def test_resume_skips_finished_grids(scene, tmp_path, caplog):
    cfg = make_cfg(scene, tmp_path, grid_ids=["G1"])
    pipeline.run(cfg)
    caplog.set_level("INFO")
    pipeline.run(cfg)
    assert "already done" in caplog.text


def test_missing_imagery_is_flagged(scene, tmp_path):
    import rasterio
    # blank out the right half of the image (nodata = 0 everywhere)
    src_path = scene["imagery"]
    holed = tmp_path / "holed.tif"
    with rasterio.open(src_path) as src:
        data = src.read()
        prof = src.profile
    data[:, :, data.shape[2] // 2:] = 0
    prof.update(nodata=0)
    with rasterio.open(holed, "w", **prof) as dst:
        dst.write(data)
    paths = dict(scene, imagery=holed)
    cfg = make_cfg(paths, tmp_path / "out", grid_ids=["G2"],
                   classifier={"samples_per_class": 4000, "training_grids": 2,
                               "model_path": None})
    pipeline.run(cfg)
    out = gpd.read_file(tmp_path / "out" / "aligned_centerlines.gpkg")
    assert (out.review == 1).all()
    assert out.reason.str.contains("no_imagery").all()
    # with no evidence, lines are left where they were
    assert out.shift_max.max() < 1e-6


def test_folder_of_tiles(scene, tmp_path):
    """Imagery given as a folder of tiles (with .aux.xml sidecars) is mosaicked."""
    import rasterio
    from rasterio.windows import Window

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    with rasterio.open(scene["imagery"]) as src:
        h, w = src.height, src.width
        for i, (r0, c0) in enumerate([(0, 0), (0, w // 2), (h // 2, 0), (h // 2, w // 2)]):
            win = Window(c0, r0, w - w // 2 if c0 else w // 2, h - h // 2 if r0 else h // 2)
            prof = src.profile.copy()
            prof.update(width=win.width, height=win.height,
                        transform=src.window_transform(win))
            with rasterio.open(tiles / f"tile_{i}.tif", "w", **prof) as dst:
                dst.write(src.read(window=win))
            (tiles / f"tile_{i}.tif.aux.xml").write_text("<PAMDataset></PAMDataset>")
    (tiles / "ignored.sid").write_bytes(b"")
    cfg = make_cfg(dict(scene, imagery=tiles), tmp_path / "out")
    pipeline.run(cfg)
    assert (tmp_path / "out" / "imagery_mosaic.vrt").exists()
    out = gpd.read_file(tmp_path / "out" / "aligned_centerlines.gpkg")
    a_mean, _ = error_to_truth(out.geometry)
    assert a_mean < 1.0


def test_mrsid_gives_clear_error(tmp_path):
    from centerline_align.mosaic import resolve_imagery

    with pytest.raises(ValueError, match="MrSID"):
        resolve_imagery(str(tmp_path / "county.sid"), tmp_path)
    (tmp_path / "only_sid").mkdir()
    (tmp_path / "only_sid" / "a.sid").write_bytes(b"")
    with pytest.raises(ValueError, match="MrSID"):
        resolve_imagery(str(tmp_path / "only_sid"), tmp_path)


def test_embedding_mode_runs(scene, tmp_path):
    """Embedding path end to end with a tiny random DINOv2 (no download needed).

    Random weights carry no meaning, so this checks the plumbing (token layout,
    patch-to-pixel mapping, saved model) rather than accuracy.
    """
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg_m = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                      intermediate_size=64, patch_size=14, image_size=518)
    model_dir = tmp_path / "tiny_dinov2"
    transformers.Dinov2Model(cfg_m).save_pretrained(model_dir)

    cfg = make_cfg(scene, tmp_path / "out", grid_ids=["G2"],
                   classifier={"features": "embedding", "embedding_model": str(model_dir),
                               "embedding_device": "cpu", "embedding_tile": 266,
                               "samples_per_class": 3000, "training_grids": 1})
    pipeline.run(cfg)
    from centerline_align.classifier import RoadClassifier
    clf = RoadClassifier.load(tmp_path / "out" / "road_classifier.json")
    assert clf.feature_mode == "embedding" and len(clf.emb_coef) == 32
    out = gpd.read_file(tmp_path / "out" / "aligned_centerlines.gpkg")
    a_mean, _ = error_to_truth(out.geometry)
    assert a_mean < 1.5  # pixel features still carry this easy scene


def test_linear_upsampling_matches_pixel_sampling():
    """predict() upsamples the embedding logit; training samples embeddings per pixel.
    For a linear model these must agree."""
    from centerline_align.features import sample_lowres, upsample_lowres

    rng = np.random.default_rng(0)
    low = rng.normal(size=(5, 6, 7)).astype(np.float32)
    w = rng.normal(size=5).astype(np.float32)
    shape = (6 * 14, 7 * 14)
    up_logit = upsample_lowres(np.tensordot(w, low, axes=1), shape, 14)
    rows, cols = rng.integers(0, shape[0], 50), rng.integers(0, shape[1], 50)
    per_pixel = w @ sample_lowres(low, rows, cols, 14)
    assert np.allclose(up_logit[rows, cols], per_pixel, atol=1e-4)
