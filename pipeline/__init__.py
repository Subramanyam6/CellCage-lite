"""End-to-end pipeline orchestration for CellCage-lite.

Ties detection, classification, cage placement, and tracking into one system
with a single entry point. See ``pipeline/runner.py``.
"""

from __future__ import annotations

from .runner import Pipeline, PipelineResult

__all__ = ["Pipeline", "PipelineResult"]
