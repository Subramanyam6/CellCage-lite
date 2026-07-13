"""Cell classification for CellCage-lite.

Frozen DINOv2 embeddings feed a lightweight head (a linear probe, or a
prototypical nearest-class-mean classifier for the few-shot case), with an
optional LoRA fine-tune of DINOv2. See README section 2.
"""

from __future__ import annotations

from .classifier import CellClassifier
from .embed import DINOv2Embedder, crop_cells
from .head import LinearProbe, NearestClassMean

__all__ = [
    "CellClassifier",
    "DINOv2Embedder",
    "crop_cells",
    "LinearProbe",
    "NearestClassMean",
]
