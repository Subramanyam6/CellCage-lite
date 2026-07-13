"""Tie embeddings and a head together into a cell classifier.

Given an image and the cells detection found, produce the same cells labeled
``target`` or ``non-target``, each with a confidence the cage engine can use to
widen its safety margins on doubtful cells.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from cage.types import Cell

from .embed import DINOv2Embedder, crop_cells
from .head import LinearProbe, NearestClassMean


class CellClassifier:
    """Embed cells, then label them with a lightweight head.

    Pass an ``embedder`` (a :class:`DINOv2Embedder`) to work from images, or call
    the ``*_embeddings`` methods directly with precomputed embeddings (useful for
    testing and for caching DINOv2 features).
    """

    def __init__(
        self,
        head=None,
        embedder: DINOv2Embedder | None = None,
        out_size: int = 64,
        context: float = 2.0,
    ) -> None:
        self.head = head if head is not None else LinearProbe()
        self.embedder = embedder
        self.out_size = out_size
        self.context = context

    # --- image-based API (needs an embedder) ------------------------------- #
    def fit(self, image: np.ndarray, cells: Sequence[Cell], labels: Sequence[str]) -> "CellClassifier":
        return self.fit_embeddings(self._embed(image, cells), labels)

    def predict(self, image: np.ndarray, cells: Sequence[Cell]) -> list[Cell]:
        return self.label_embeddings(cells, self._embed(image, cells))

    def _embed(self, image: np.ndarray, cells: Sequence[Cell]) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("no embedder set; use the *_embeddings methods instead")
        crops = crop_cells(image, cells, out_size=self.out_size, context=self.context)
        return self.embedder.embed(crops)

    # --- embedding-based API (no torch needed) ----------------------------- #
    def fit_embeddings(self, embeddings: np.ndarray, labels: Sequence[str]) -> "CellClassifier":
        self.head.fit(embeddings, np.asarray(labels))
        return self

    def label_embeddings(self, cells: Sequence[Cell], embeddings: np.ndarray) -> list[Cell]:
        """Return the cells relabeled from their embeddings, with confidences."""
        preds = self.head.predict(embeddings)
        confidences = self._confidences(embeddings)
        return [
            replace(cell, label=str(pred), confidence=float(conf))
            for cell, pred, conf in zip(cells, preds, confidences)
        ]

    def _confidences(self, embeddings: np.ndarray) -> np.ndarray:
        if isinstance(self.head, LinearProbe):
            return self.head.predict_proba(embeddings).max(axis=1)
        # A prototypical head reports no calibrated probability; treat its
        # decisions as confident.
        return np.ones(len(embeddings))
