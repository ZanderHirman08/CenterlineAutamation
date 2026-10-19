"""Per-pixel features for the road / not-road classifier: spectral and texture
measures computed from the bands directly."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

EPS = 1e-6


def pixel_feature_names(has_nir: bool) -> list[str]:
    names = ["red", "green", "blue"]
    if has_nir:
        names.append("nir")
    names += ["bright", "sat", "exg"]
    if has_nir:
        names.append("ndvi")
    names += ["bright_s2", "bright_s6", "bright_std3", "sat_s4", "exg_s4"]
    if has_nir:
        names.append("ndvi_s4")
    return names


def scale_bands(img: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Linearly stretch raw band values to roughly 0-1 using training percentiles."""
    lo = np.asarray(lo, dtype=np.float32)[:, None, None]
    hi = np.asarray(hi, dtype=np.float32)[:, None, None]
    return (img.astype(np.float32) - lo) / np.maximum(hi - lo, EPS)


def pixel_features(scaled: np.ndarray, has_nir: bool) -> np.ndarray:
    """Features from scaled bands (R, G, B[, NIR]) shaped (bands, H, W).

    Returns (n_features, H, W) float32, in the order of ``pixel_feature_names``.
    """
    # Clip and lift away from zero so ratios stay bounded in shadows and at the
    # ends of the stretch (dividing by ~0 would give huge, fit-breaking values).
    s = np.clip(scaled, 0.0, 1.5) + 0.05
    r, g, b = s[0], s[1], s[2]
    rgb_max = np.maximum(np.maximum(r, g), b)
    rgb_min = np.minimum(np.minimum(r, g), b)
    bright = (r + g + b) / 3.0
    sat = (rgb_max - rgb_min) / rgb_max          # pavement is grey: low saturation, 0-1
    exg = 2.0 * g - r - b                        # excess green: vegetation without NIR

    feats = [r, g, b]
    if has_nir:
        nir = s[3]
        ndvi = (nir - r) / (nir + r)             # -1 to 1
        feats.append(nir)
    feats += [bright, sat, exg]
    if has_nir:
        feats.append(ndvi)

    blur = ndimage.gaussian_filter
    b2 = blur(bright, 2)
    var3 = blur(bright * bright, 3) - blur(bright, 3) ** 2
    feats += [b2, blur(bright, 6), np.sqrt(np.maximum(var3, 0)), blur(sat, 4), blur(exg, 4)]
    if has_nir:
        feats.append(blur(ndvi, 4))
    return np.stack(feats).astype(np.float32)
