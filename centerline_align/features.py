"""Per-pixel features for the road / not-road classifier.

Two sources, which the classifier can combine:

* ``pixel_features`` - spectral and texture measures computed from the bands
  directly. Cheap, needs no GPU.
* ``EmbeddingExtractor`` - patch embeddings from a frozen, pretrained vision
  foundation model (DINOv2 / DINOv3 via Hugging Face). These come out at
  patch resolution (e.g. one vector per 14x14 or 16x16 pixels) and are kept at
  that resolution; the classifier is linear, so its output can be upsampled
  instead of the (much larger) embeddings.
"""
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


def sample_lowres(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch: int) -> np.ndarray:
    """Bilinearly sample a patch-resolution array (C, h, w) at full-res pixel positions.

    Patch ``i`` covers pixels ``i*patch .. (i+1)*patch-1``, so its centre is at
    pixel ``(i + 0.5) * patch - 0.5``.
    """
    pr = (np.asarray(rows, dtype=np.float64) + 0.5) / patch - 0.5
    pc = (np.asarray(cols, dtype=np.float64) + 0.5) / patch - 0.5
    out = [ndimage.map_coordinates(c, [pr, pc], order=1, mode="nearest") for c in grid]
    return np.stack(out)


def upsample_lowres(grid2d: np.ndarray, shape: tuple[int, int], patch: int) -> np.ndarray:
    """Bilinearly upsample a single patch-resolution band to ``shape`` (full res)."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    return sample_lowres(grid2d[None], rows.ravel(), cols.ravel(), patch)[0].reshape(shape)


class EmbeddingExtractor:
    """Frozen foundation-model patch embeddings for an RGB tile.

    Works with Hugging Face vision transformers that return one token per
    patch (optionally after a CLS token and register tokens), e.g.
    ``facebook/dinov2-small`` (Apache-2.0) or a DINOv3 satellite checkpoint
    (gated; accept its licence on Hugging Face first).
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover - depends on optional install
            raise ImportError(
                "Embedding features need PyTorch and transformers: "
                "pip install -r requirements-gpu.txt"
            ) from e

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("embedding_device is 'cuda' but no CUDA GPU is visible to PyTorch")
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        if device == "cuda":
            self.model = self.model.half()
        cfg = self.model.config
        self.patch = int(getattr(cfg, "patch_size", 14))
        self.n_registers = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.dim = int(cfg.hidden_size)
        self.mean, self.std = _normalisation(model_name)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (3, H, W) scaled 0-1. Returns (dim, ceil(H/p), ceil(W/p)) float32."""
        torch = self.torch
        _, h, w = rgb.shape
        p = self.patch
        hp, wp = -(-h // p), -(-w // p)
        x = np.zeros((3, hp * p, wp * p), dtype=np.float32)
        x[:, :h, :w] = np.clip(rgb, 0, 1)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        t = torch.from_numpy(x)[None].to(self.device)
        if self.device == "cuda":
            t = t.half()
        with torch.no_grad():
            tokens = self.model(pixel_values=t).last_hidden_state[0]
        tokens = tokens[-hp * wp:]  # drop CLS / register tokens at the front
        emb = tokens.float().cpu().numpy().T.reshape(self.dim, hp, wp)
        return emb.astype(np.float32)


def _normalisation(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(model_name)
        mean = list(getattr(proc, "image_mean", mean))
        std = list(getattr(proc, "image_std", std))
    except Exception:
        pass
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
