"""Turn each cell into a fixed-length embedding ("fingerprint").

Using each cell's location from detection, its patch is cropped from the original
image and passed through frozen DINOv2, which maps the patch to a vector arranged
so that cells that look alike land close together. DINOv2 is used as-is and never
retrained, which is what lets the downstream head work from few labels.

The cropping is pure NumPy and always available; only the embedding model needs
torch, which is imported lazily so this module can be imported without it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cage.types import Cell


def crop_cells(
    image: np.ndarray,
    cells: Sequence[Cell],
    out_size: int = 64,
    context: float = 2.0,
) -> np.ndarray:
    """Crop a square patch around each cell and resize to ``out_size``.

    The patch is ``context`` times the cell radius on each side, so a little
    surrounding context is included. Returns an array of shape
    ``(n_cells, out_size, out_size)`` for a grayscale image, or with a trailing
    channel axis for a color image.
    """
    image = np.asarray(image)
    h, w = image.shape[:2]
    patches = []
    for cell in cells:
        half = max(1, int(round(cell.radius * context)))
        x0, x1 = int(cell.x) - half, int(cell.x) + half
        y0, y1 = int(cell.y) - half, int(cell.y) + half
        patch = _padded_crop(image, x0, y0, x1, y1, h, w)
        patches.append(_resize_nn(patch, out_size))
    return np.stack(patches) if patches else np.empty((0, out_size, out_size))


def _padded_crop(image, x0, y0, x1, y1, h, w):
    """Crop with zero-padding when the window runs off the image edge."""
    px0, py0 = max(0, x0), max(0, y0)
    px1, py1 = min(w, x1), min(h, y1)
    crop = image[py0:py1, px0:px1]
    pad = [(py0 - y0, y1 - py1), (px0 - x0, x1 - px1)]
    if image.ndim == 3:
        pad.append((0, 0))
    return np.pad(crop, pad, mode="constant")


def _resize_nn(patch: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbor resize to ``size x size`` (no extra dependencies)."""
    ph, pw = patch.shape[:2]
    if ph == 0 or pw == 0:
        shape = (size, size) + patch.shape[2:]
        return np.zeros(shape, dtype=patch.dtype)
    ys = np.clip((np.arange(size) * ph / size).astype(int), 0, ph - 1)
    xs = np.clip((np.arange(size) * pw / size).astype(int), 0, pw - 1)
    return patch[np.ix_(ys, xs)] if patch.ndim == 2 else patch[np.ix_(ys, xs, np.arange(patch.shape[2]))]


class StatsEmbedder:
    """A lightweight, torch-free embedder: simple per-crop intensity statistics.

    Turns each crop into ``[mean, std, min, max]``. It captures nothing like the
    richness of DINOv2, but it needs no model or download, which makes it a
    useful baseline and lets the whole pipeline run end to end without a
    deep-learning stack. :class:`DINOv2Embedder` is the real feature extractor.
    """

    def embed(self, crops: np.ndarray) -> np.ndarray:
        x = np.asarray(crops, dtype=float)
        if len(x) == 0:
            return np.empty((0, 4))
        flat = x.reshape(len(x), -1)
        return np.column_stack([flat.mean(1), flat.std(1), flat.min(1), flat.max(1)])


class DINOv2Embedder:
    """Frozen DINOv2 feature extractor.

    Loads a pretrained DINOv2 backbone once and returns the CLS embedding for
    each crop. The model is never updated here; fine-tuning lives in
    ``classify/lora.py``.
    """

    def __init__(self, model_name: str = "dinov2_vits14", device: str | None = None) -> None:
        self.model_name = model_name
        self._device = device
        self._model = None  # loaded lazily on first use

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch  # lazy: only needed when actually embedding

        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        # DINOv2 is published on torch.hub.
        self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self._model.eval().to(self._device)

    def embed(self, crops: np.ndarray) -> np.ndarray:
        """Return an ``(n, dim)`` array of embeddings for a batch of crops.

        Crops are expected as ``(n, H, W)`` or ``(n, H, W, 3)`` in ``[0, 1]``.
        """
        self._ensure_model()
        torch = self._torch

        x = np.asarray(crops, dtype=np.float32)
        if x.ndim == 3:  # grayscale -> 3 channels
            x = np.repeat(x[..., None], 3, axis=-1)
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        tensor = torch.from_numpy(x).to(self._device)
        with torch.no_grad():
            feats = self._model(tensor)
        return feats.cpu().numpy()
