"""End-to-end orchestration: detect, classify, place cages, and track.

This is the single entry point that ties the four stages into one system. Each
stage is pluggable: pass a detector and a classifier for the full image-in,
cages-out path, or hand in an already-detected cell list to run just the
placement. Per-stage latency is recorded so a run is easy to profile.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cage import Cage, CageSpec, Cell, place_cages
from track import Tracker


@dataclass
class PipelineResult:
    """Outcome of running the pipeline on one field."""

    cells: list[Cell]  # detected cells, labeled by classification
    cages: list[Cage]  # the placed cages
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def n_targets(self) -> int:
        return sum(1 for c in self.cells if c.label == "target")

    @property
    def coverage(self) -> float:
        """Fraction of targets that received a cage, in ``[0, 1]``."""
        return len(self.cages) / self.n_targets if self.n_targets else 0.0

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latency_ms.values())


class Pipeline:
    """Compose the four stages behind one API.

    Parameters
    ----------
    cage_spec:
        Cage geometry for the placement stage.
    detector:
        Any object with ``detect(image) -> list[Cell]`` (e.g. a
        ``detect.Detector`` or ``detect.MaskDetector``). Required for
        :meth:`run_image`.
    classifier:
        Any object with ``predict(image, cells) -> list[Cell]`` (e.g. a
        ``classify.CellClassifier``). Optional; without it, the detector's labels
        are kept.
    tracker_kwargs:
        Keyword arguments forwarded to the :class:`track.Tracker`.
    """

    def __init__(
        self,
        cage_spec: CageSpec | None = None,
        detector=None,
        classifier=None,
        precision: float = 0.05,
        tracker_kwargs: dict | None = None,
    ) -> None:
        self.cage_spec = cage_spec or CageSpec()
        self.detector = detector
        self.classifier = classifier
        self.precision = precision
        self.tracker_kwargs = tracker_kwargs or {}

    def run_image(self, image: np.ndarray) -> PipelineResult:
        """Full path: detect cells in the image, classify them, place cages."""
        if self.detector is None:
            raise RuntimeError("run_image requires a detector; use run_cells otherwise")
        latency: dict[str, float] = {}

        with _timed(latency, "detect"):
            cells = self.detector.detect(image)
        cells = self._classify(image, cells, latency)
        with _timed(latency, "cage"):
            cages = place_cages(cells, self.cage_spec, self.precision)

        return PipelineResult(cells=cells, cages=cages, latency_ms=latency)

    def run_cells(self, cells: Sequence[Cell], image: np.ndarray | None = None) -> PipelineResult:
        """Placement path when detection is already done.

        If a classifier and image are provided, cells are (re)classified first;
        otherwise their existing labels are used.
        """
        latency: dict[str, float] = {}
        cells = list(cells)
        if image is not None:
            cells = self._classify(image, cells, latency)
        with _timed(latency, "cage"):
            cages = place_cages(cells, self.cage_spec, self.precision)
        return PipelineResult(cells=cells, cages=cages, latency_ms=latency)

    def track_sequence(self, frames_cells: Sequence[Sequence[Cell]]) -> list[np.ndarray]:
        """Track a sequence of per-frame cell lists, returning per-frame track ids.

        ``frames_cells[t]`` is the cells detected in frame ``t``; the returned
        list holds, for each frame, the track id assigned to each of those cells.
        """
        tracker = Tracker(**self.tracker_kwargs)
        labels: list[np.ndarray] = []
        for cells in frames_cells:
            points = np.array([[c.x, c.y] for c in cells], dtype=float).reshape(-1, 2)
            labels.append(tracker.step(points))
        return labels

    def _classify(self, image, cells, latency: dict[str, float]) -> list[Cell]:
        if self.classifier is None or image is None:
            return list(cells)
        with _timed(latency, "classify"):
            return self.classifier.predict(image, cells)


class _timed:
    """Context manager recording elapsed milliseconds into ``store[name]``."""

    def __init__(self, store: dict[str, float], name: str) -> None:
        self.store = store
        self.name = name

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.store[self.name] = (time.perf_counter() - self._start) * 1000.0
        return False
