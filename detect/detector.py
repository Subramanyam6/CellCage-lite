"""Turn a microscope image into a list of cells.

Two backends produce the same output, a list of :class:`Cell`:

- ``"cellpose"`` wraps the pretrained Cellpose package (no training needed) and
  is the practical default.
- ``"flowunet"`` runs the in-repo flow U-Net (``detect/model.py``) once it has
  trained weights.

Both are imported lazily. The label-image-to-cells conversion is pure NumPy and
is what the rest of the pipeline consumes.
"""

from __future__ import annotations

import numpy as np

from cage.types import Cell

from .model import FlowUNet, flows_to_labels


def labels_to_cells(labels: np.ndarray) -> list[Cell]:
    """Reduce an instance-label image to one :class:`Cell` per instance.

    Each cell gets its centroid as center and an effective radius from its area
    (``sqrt(area / pi)``). Every cell starts as ``non-target``; classification
    sets the real label later.
    """
    labels = np.asarray(labels)
    cells: list[Cell] = []
    ids = np.unique(labels)
    ids = ids[ids > 0]
    for new_id, cell_id in enumerate(ids):
        ys, xs = np.nonzero(labels == cell_id)
        area = len(xs)
        if area == 0:
            continue
        cells.append(
            Cell(
                id=new_id,
                x=float(xs.mean()),
                y=float(ys.mean()),
                radius=float(np.sqrt(area / np.pi)),
                label="non-target",
                confidence=1.0,
            )
        )
    return cells


class Detector:
    """Segmentation front-end producing a cell list from an image."""

    def __init__(self, backend: str = "cellpose", **kwargs) -> None:
        if backend not in ("cellpose", "flowunet"):
            raise ValueError(f"unknown backend: {backend}")
        self.backend = backend
        self._kwargs = kwargs
        self._model = None

    def detect(self, image: np.ndarray) -> list[Cell]:
        """Detect cells in a single image and return them as a cell list."""
        if self.backend == "cellpose":
            labels = self._detect_cellpose(image)
        else:
            labels = self._detect_flowunet(image)
        return labels_to_cells(labels)

    def _detect_cellpose(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            from cellpose import models  # lazy

            self._model = models.Cellpose(model_type=self._kwargs.get("model_type", "cyto"))
        diameter = self._kwargs.get("diameter", None)
        masks, *_ = self._model.eval(image, diameter=diameter, channels=[0, 0])
        return np.asarray(masks)

    def _detect_flowunet(self, image: np.ndarray) -> np.ndarray:
        import torch  # lazy

        if self._model is None:
            self._model = FlowUNet(**self._kwargs.get("model_kwargs", {})).build()
            weights = self._kwargs.get("weights")
            if weights is not None:
                self._model.load_state_dict(torch.load(weights, map_location="cpu"))
            self._model.eval()

        x = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(x)[None, None]  # (1, 1, H, W)
        with torch.no_grad():
            out = self._model(tensor)[0].numpy()
        flow, fg_logit = out[:2], out[2]
        foreground = fg_logit > 0
        return flows_to_labels(flow, foreground)
