import numpy as np
import pytest
import rasterio

from centerline_align.chip import Chip
from centerline_align.engine import alignment_check

from . import synthetic


@pytest.fixture(scope="module")
def chip(tmp_path_factory):
    path = synthetic.make_image(tmp_path_factory.mktemp("img") / "o.tif", nir=True)
    with rasterio.open(path) as src:
        # a 1024 px block around the main intersection
        col0, row0 = 400, 500
        data = src.read(window=rasterio.windows.Window(col0, row0, 1024, 1024)).astype(np.float32)
    return Chip(data=data, valid=np.ones(data.shape[1:], bool),
                x0=synthetic.X0 + col0 * synthetic.RES, y0=synthetic.Y0 - row0 * synthetic.RES,
                res=synthetic.RES)


def lines(dx, dy):
    return [np.asarray(g.coords) + [dx, dy] for g, _ in synthetic.true_roads()]


@pytest.mark.parametrize("dx,dy", [(0, 0), (25, -10), (-35, 20)])
def test_alignment_check_finds_offset(chip, dx, dy):
    sx, sy, best, zero = alignment_check(chip, lines(dx, dy), has_nir=True)
    # moving the lines by (sx, sy) should undo the offset, within a couple of feet
    assert abs(sx + dx) <= 2 and abs(sy + dy) <= 2, (sx, sy)
    assert best >= zero


def test_ribbon_resists_shadow_and_side_lot():
    """A shadow over one edge and a grey lot beside the road barely move the ribbon
    fit, while the older midpoint method is pulled several feet."""
    from centerline_align.config import RefineConfig
    from centerline_align.refine import refine_line

    rng = np.random.default_rng(0)
    true_off, width = 5.0, 24.0

    def sample(x, y):
        p = np.where(np.abs(y - true_off) <= width / 2, 0.9, 0.1)
        shadow = (x > 200) & (x < 350) & (y > true_off + width / 2 - 9) & (y <= true_off + 16)
        lot = (x > 500) & (x < 560) & (y < true_off - width / 2) & (y > true_off - 27)
        p = np.where(shadow, 0.15, np.where(lot, 0.8, p))
        return np.clip(p + rng.normal(0, 0.1, p.shape), 0, 1)

    line = np.array([[0.0, 0.0], [800.0, 0.0]])
    err = {}
    for method in ("midpoint", "ribbon"):
        r = refine_line(line, sample, RefineConfig(method=method), 0.5, keep_stations=True)
        x, off = r.stations["x"], r.stations["final"]
        bad = ((x > 200) & (x < 350)) | ((x > 500) & (x < 560))
        err[method] = np.abs(off[bad] - true_off).mean()
        assert not r.reasons
    assert err["ribbon"] < 2.0 < err["midpoint"], err


def test_find_junctions():
    from centerline_align.engine import find_junctions

    def a(*pts):
        return np.array(pts, float)
    # grid-edge cut: two lines continuing each other
    assert len(find_junctions([a([0, 0], [10, 0]), a([10, 0], [20, 0])])) == 0
    # T where the through road isn't split
    assert find_junctions([a([0, 0], [20, 0]), a([10, 0], [10, 10])]).tolist() == [[10, 0]]
    # four-way, all split: one junction point
    four = [a([0, 0], [10, 0]), a([10, 0], [20, 0]), a([10, 0], [10, 10]), a([10, -10], [10, 0])]
    assert find_junctions(four).tolist() == [[10, 0]]
    # two unsplit roads crossing
    assert find_junctions([a([0, 0], [20, 0]), a([10, -10], [10, 10])]).tolist() == [[10, 0]]
