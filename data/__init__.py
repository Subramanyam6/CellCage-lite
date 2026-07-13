"""Data generation for CellCage-lite.

LIVECell (real microscopy) feeds the vision stages; the synthetic generators
here feed the cage engine and the tracker, where public ground truth does not
exist.
"""

from __future__ import annotations

from .synthetic import Plate, Sequence, generate_plate, generate_sequence, rasterize_plate

__all__ = ["Plate", "Sequence", "generate_plate", "generate_sequence", "rasterize_plate"]
