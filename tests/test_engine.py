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
