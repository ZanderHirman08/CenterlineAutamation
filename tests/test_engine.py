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
    """A shadow over one edge and a grey lot beside the road barely move the ribbon fit."""
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
    r = refine_line(line, sample, RefineConfig(), 0.5, keep_stations=True)
    x, off = r.stations["x"], r.stations["final"]
    bad = ((x > 200) & (x < 350)) | ((x > 500) & (x < 560))
    assert not r.reasons
    assert np.abs(off[bad] - true_off).mean() < 2.5


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


def test_flagged_lines_keep_original_geometry():
    """No usable imagery: the line is flagged and, by default, left exactly where it was."""
    from centerline_align.config import RefineConfig
    from centerline_align.refine import refine_line

    line = np.array([[0.0, 0.0], [300.0, 0.0]])

    def nothing(x, y):
        return np.full(np.shape(x), 0.1)
    r = refine_line(line, nothing, RefineConfig(), 0.5)
    assert r.reasons and not r.moved
    assert np.array_equal(r.coords, line)
    r2 = refine_line(line, nothing, RefineConfig(keep_flagged=False), 0.5)
    assert r2.moved


def test_self_crossings():
    """Closed cul-de-sac circles and P-shaped roads are found, with their loop length."""
    from centerline_align.chips import self_crossings
    t = np.linspace(0, 2 * np.pi, 41)
    circle = np.column_stack([30 * np.cos(t), 30 * np.sin(t)])          # closed, ~188 ft
    (pt, length), = self_crossings(circle)
    assert np.allclose(pt, [30, 0]) and 180 < length < 190
    # a "P": stem up, then a 600 ft loop back across the stem
    loop = np.column_stack([100 * np.cos(t[::-1] + np.pi) + 100, 100 * np.sin(t[::-1] + np.pi)
                            + 300])
    p_road = np.vstack([[[0.0, 0.0]], loop[:30], [[-20.0, 300.0]]])
    hits = self_crossings(p_road)
    assert len(hits) == 1 and hits[0][1] > 400
    assert self_crossings(np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 50.0]])) == []


def test_lines_in_pixels_keeps_only_the_chip():
    """A long road through a chip is stored only where it is in (or near) the chip,
    and a line that leaves and comes back becomes two pieces."""
    from centerline_align.chips import lines_in_pixels
    long_road = np.array([[-5000.0, 50.0], [5000.0, 50.0]])        # 10,000 ft, through the chip
    (piece,) = lines_in_pixels(0.0, 128.0, 0.5, [long_road], (256, 256))
    piece = np.asarray(piece)
    assert piece[:, 0].min() >= -8 and piece[:, 0].max() <= 263 and len(piece) < 300
    full = lines_in_pixels(0.0, 128.0, 0.5, [long_road])[0]
    assert len(full) > 9000                                         # without shape: everything
    u = np.array([[10.0, 150.0], [10.0, 60.0], [400.0, 60.0], [400.0, 100.0], [100.0, 100.0],
                  [100.0, 150.0]])                                 # in, out to the east, back in
    assert len(lines_in_pixels(0.0, 128.0, 0.5, [u], (256, 256))) == 2
