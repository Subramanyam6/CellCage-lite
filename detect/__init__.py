"""Cell detection for CellCage-lite.

A Cellpose-style flow U-Net that separates touching cells and handles irregular
shapes, with a pretrained Cellpose backend as the practical default. See README
section 1.
"""

from __future__ import annotations

from .detector import Detector, MaskDetector, labels_to_cells
from .model import FlowUNet, flows_to_labels, masks_to_flows

__all__ = [
    "Detector",
    "MaskDetector",
    "labels_to_cells",
    "FlowUNet",
    "masks_to_flows",
    "flows_to_labels",
]
