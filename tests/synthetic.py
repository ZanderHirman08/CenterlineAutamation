"""Synthetic scene: fake ortho with known road centres, and offset 'local' roads.

Also used by ``scripts/make_demo_data.py`` so you can try the tool without
real data.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features as rfeatures
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

CRS = "EPSG:2272"  # PA State Plane South, US feet
X0, Y0 = 2_600_000.0, 250_000.0
RES = 0.5
W_FT, H_FT = 1600.0, 1000.0
MID = Y0 - H_FT / 2


def true_roads():
    """(LineString, pavement width) for the real road centres."""
    t = np.linspace(0, 1, 80)
    curve = LineString(np.column_stack([X0 + 900 + 500 * t,
                                        MID + 380 * t + 40 * np.sin(2 * np.pi * t)]))
    return [
        (LineString([(X0, MID), (X0 + W_FT, MID)]), 24.0),
        (LineString([(X0 + 400, Y0), (X0 + 400, Y0 - H_FT)]), 22.0),
        (curve, 20.0),
    ]


def local_roads():
    """Local file: same network, shifted a few feet, split at intersections."""
    dy, dx = 7.0, -5.0
    vx, hy = X0 + 400 + dx, MID + dy
    t = np.linspace(0, 1, 80)
    curve = np.column_stack([X0 + 900 + 500 * t, MID + 380 * t + 40 * np.sin(2 * np.pi * t)])
    curve[:, 0] += 4.0 * t                       # drifts away from the true line
    curve[:, 1] += dy * (1 - t) - 5.0 * t
    lines = [
        ("H1", LineString([(X0, hy), (vx, hy)])),
        ("H2", LineString([(vx, hy), (X0 + 900, hy)])),
        ("H3", LineString([(X0 + 900, hy), (X0 + W_FT, hy)])),
        ("V1", LineString([(vx, Y0), (vx, hy)])),
        ("V2", LineString([(vx, hy), (vx, Y0 - H_FT)])),
        ("C1", LineString(curve)),
    ]
    return gpd.GeoDataFrame({"RD_ID": [k for k, _ in lines]},
                            geometry=[g for _, g in lines], crs=CRS)


def grids():
    cells = [box(X0, Y0 - H_FT, X0 + 800, Y0), box(X0 + 800, Y0 - H_FT, X0 + W_FT, Y0)]
    return gpd.GeoDataFrame({"GRID_ID": ["G1", "G2"]}, geometry=cells, crs=CRS)


def make_image(path: Path, seed: int = 0, nir: bool = False):
    rng = np.random.default_rng(seed)
    w, h = int(W_FT / RES), int(H_FT / RES)
    transform = from_origin(X0, Y0, RES, RES)
    road = np.zeros((h, w), bool)
    for line, width in true_roads():
        road |= rfeatures.rasterize([(line.buffer(width / 2, cap_style="flat"), 1)],
                                    out_shape=(h, w), transform=transform).astype(bool)
    # grass, a few trees, pavement
    img = np.empty((4 if nir else 3, h, w), np.float32)
    grass = np.array([70, 110, 50, 180])[: img.shape[0]]
    pave = np.array([125, 124, 120, 70])[: img.shape[0]]
    img[:] = grass[:, None, None]
    img[:, road] = pave[:, None]
    yy, xx = np.mgrid[0:h, 0:w]
    tree = np.zeros((h, w), bool)
    for _ in range(60):
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(10, 30)
        if road[max(0, cy - r):cy + r, max(0, cx - r):cx + r].any():
            continue  # keep trees off the pavement in the base scene
        tree |= (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
    tree_col = np.array([35, 60, 30, 150])[: img.shape[0]]
    img[:, tree] = tree_col[:, None]
    img += rng.normal(0, 8, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=img.shape[0],
                       dtype="uint8", crs=CRS, transform=transform) as dst:
        dst.write(img)
    return path


def write_scene(folder: Path, nir: bool = False) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = dict(imagery=folder / "ortho.tif", roads=folder / "local_roads.shp",
                 grids=folder / "grids.shp")
    make_image(paths["imagery"], nir=nir)
    local_roads().to_file(paths["roads"])
    grids().to_file(paths["grids"])
    return paths
