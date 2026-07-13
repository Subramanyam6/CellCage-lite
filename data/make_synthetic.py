"""CLI: build a synthetic dataset of plates and save it to disk.

    python -m data.make_synthetic --n-plates 20 --n-cells 250 --out data/synthetic

Each plate is stored as a compressed ``.npz`` of its cell arrays. The output
directory is git-ignored: the data is regenerated on demand from the seed, never
versioned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cage.types import Cell
from .synthetic import Plate, generate_plate

_LABELS = ("non-target", "target")


def save_plate(plate: Plate, path: Path) -> None:
    """Serialize a plate to a compressed ``.npz`` file."""
    cells = plate.cells
    np.savez_compressed(
        path,
        width=plate.width,
        height=plate.height,
        id=np.array([c.id for c in cells], dtype=np.int64),
        x=np.array([c.x for c in cells], dtype=np.float64),
        y=np.array([c.y for c in cells], dtype=np.float64),
        radius=np.array([c.radius for c in cells], dtype=np.float64),
        is_target=np.array([c.label == "target" for c in cells], dtype=bool),
        confidence=np.array([c.confidence for c in cells], dtype=np.float64),
    )


def load_plate(path: Path) -> Plate:
    """Load a plate saved by :func:`save_plate`."""
    d = np.load(path)
    cells = [
        Cell(
            id=int(d["id"][i]),
            x=float(d["x"][i]),
            y=float(d["y"][i]),
            radius=float(d["radius"][i]),
            label="target" if bool(d["is_target"][i]) else "non-target",
            confidence=float(d["confidence"][i]),
        )
        for i in range(len(d["id"]))
    ]
    return Plate(width=float(d["width"]), height=float(d["height"]), cells=cells)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic plate dataset.")
    parser.add_argument("--n-plates", type=int, default=20)
    parser.add_argument("--n-cells", type=int, default=250)
    parser.add_argument("--size", type=float, default=512.0)
    parser.add_argument("--target-fraction", type=float, default=0.5)
    parser.add_argument("--clustered", action="store_true", help="crowd cells into clusters")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    total_cells = 0
    total_targets = 0
    for i in range(args.n_plates):
        plate = generate_plate(
            n_cells=args.n_cells,
            size=(args.size, args.size),
            target_fraction=args.target_fraction,
            clustered=args.clustered,
            seed=args.seed + i,
        )
        save_plate(plate, args.out / f"plate_{i:04d}.npz")
        total_cells += len(plate.cells)
        total_targets += len(plate.targets)

    print(
        f"Wrote {args.n_plates} plates to {args.out} "
        f"({total_cells} cells, {total_targets} targets, "
        f"avg {total_cells / args.n_plates:.0f} cells/plate)."
    )


if __name__ == "__main__":
    main()
