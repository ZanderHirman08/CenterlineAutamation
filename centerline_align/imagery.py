"""Windowed imagery reading in the working CRS and resolution."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

from .chip import Chip
from .config import Config
from .mosaic import resolve_imagery


class ImageryReader:
    def __init__(self, cfg: Config):
        from pathlib import Path

        self.path = resolve_imagery(cfg.inputs.imagery, Path(cfg.output.folder))
        self._src = rasterio.open(self.path)
        self.crs = rasterio.crs.CRS.from_user_input(cfg.crs) if cfg.crs else self._src.crs
        if self.crs is None:
            raise ValueError("Imagery has no CRS; set 'crs' in the config")
        if self.crs.is_geographic:
            raise ValueError(
                "Working CRS is geographic (degrees). Set 'crs' to a projected CRS "
                "such as your State Plane zone so distances are in feet or metres."
            )
        self._ds = self._src if self._src.crs == self.crs else WarpedVRT(
            self._src, crs=self.crs, resampling=Resampling.bilinear
        )
        self.res = float(cfg.resolution or abs(self._ds.res[0]))
        b = cfg.bands
        self.band_indexes = [b.red, b.green, b.blue] + ([b.nir] if b.nir else [])
        self.has_nir = b.nir is not None
        if max(self.band_indexes) > self._ds.count:
            raise ValueError(
                f"Config asks for band {max(self.band_indexes)} but the imagery has "
                f"{self._ds.count} band(s)"
            )

    @property
    def bounds(self):
        return self._ds.bounds

    def snap(self, x: float, y: float) -> tuple[float, float]:
        """Pixel corner at or left of x and at or above y, on the imagery's grid."""
        gx, gy, r = self._ds.transform.c, self._ds.transform.f, self.res
        return gx + np.floor((x - gx) / r) * r, gy - np.floor((gy - y) / r) * r

    def read(self, x0: float, y0: float, width_px: int, height_px: int,
             res: float | None = None) -> Chip:
        """Read a block whose top-left corner is (x0, y0), resampled to ``res``."""
        res = res or self.res
        x1, y1 = x0 + width_px * res, y0 - height_px * res
        window = from_bounds(x0, y1, x1, y0, transform=self._ds.transform)
        kw = dict(window=window, out_shape=(height_px, width_px), boundless=True,
                  resampling=Resampling.bilinear)
        data = self._ds.read(self.band_indexes, fill_value=0, **kw).astype(np.float32)
        mask = self._ds.read_masks(self.band_indexes[0], **kw)
        return Chip(data=data, valid=mask > 0, x0=x0, y0=y0, res=res)

    def close(self):
        if self._ds is not self._src:
            self._ds.close()
        self._src.close()
