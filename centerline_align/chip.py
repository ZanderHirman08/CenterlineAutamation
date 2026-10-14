"""A georeferenced block of imagery, independent of how it was read."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Chip:
    """Bands (B, H, W) float32, validity mask (H, W) and georeference.

    ``x0, y0`` is the top-left corner; pixels are square with size ``res``.
    """
    data: np.ndarray
    valid: np.ndarray
    x0: float
    y0: float
    res: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.shape
        return self.x0, self.y0 - h * self.res, self.x0 + w * self.res, self.y0

    def world_to_pixel(self, x, y):
        """Fractional (row, col) of pixel centres for world coords."""
        col = (np.asarray(x) - self.x0) / self.res - 0.5
        row = (self.y0 - np.asarray(y)) / self.res - 0.5
        return row, col
