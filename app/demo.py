"""Interactive demo for the CellCage-lite pipeline.

    python -m app.demo            # Gradio UI if installed, else render to files
    python -m app.demo --static   # always render static figures to app/output

The Gradio interface lets you dial the plate density, target fraction, and cage
shape and see the cells detected, targets labeled, cages placed, and the
per-field latency. Without Gradio, the same scenarios are rendered to PNG files.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cage import CageSpec, place_cages
from data.synthetic import generate_plate, generate_sequence
from track import run_tracker

from .render import figure_placement, figure_tracking


def run_placement(n_cells: int, target_fraction: float, shape: str, radius: float, seed: int):
    """Generate a plate, place cages, and return the figure plus a latency line."""
    plate = generate_plate(
        n_cells=n_cells, target_fraction=target_fraction, seed=seed, size=(512, 512)
    )
    spec = CageSpec(shape=shape, radius=radius, wall=2.0, clearance=1.0, exclusion_margin=1.0)

    start = time.perf_counter()
    cages = place_cages(plate.cells, spec)
    latency_ms = (time.perf_counter() - start) * 1000.0

    n_targets = len(plate.targets)
    coverage = 100.0 * len(cages) / n_targets if n_targets else 0.0
    caption = (
        f"{len(cages)} {shape} cages over {n_targets} targets "
        f"({coverage:.0f}% coverage) in {latency_ms:.1f} ms"
    )
    fig = figure_placement(plate.cells, cages, spec, title=caption)
    return fig, caption


def _render_static(outdir: Path) -> None:
    """Render a set of scenarios to PNG files (no Gradio required)."""
    outdir.mkdir(parents=True, exist_ok=True)

    for shape in ("circle", "hexagon"):
        fig, caption = run_placement(
            n_cells=250, target_fraction=0.5, shape=shape, radius=18.0, seed=0
        )
        path = outdir / f"placement_{shape}.png"
        fig.savefig(path, dpi=110)
        print(f"{caption}\n  -> {path}")

    seq = generate_sequence(n_cells=25, n_frames=30, seed=1)
    labels = run_tracker(seq.frames, gating_distance=30.0)
    fig = figure_tracking(seq.frames, labels)
    path = outdir / "tracking.png"
    fig.savefig(path, dpi=110)
    print(f"tracking demo -> {path}")


def _launch_gradio() -> bool:
    """Launch the Gradio UI. Returns False if Gradio is not installed."""
    try:
        import gradio as gr
    except ImportError:
        return False

    def _ui(n_cells, target_fraction, shape, radius):
        fig, caption = run_placement(int(n_cells), target_fraction, shape, radius, seed=0)
        return fig, caption

    demo = gr.Interface(
        fn=_ui,
        inputs=[
            gr.Slider(20, 600, value=250, step=10, label="Cells per field"),
            gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Target fraction"),
            gr.Radio(["circle", "hexagon"], value="circle", label="Cage shape"),
            gr.Slider(10, 30, value=18, step=1, label="Cage radius"),
        ],
        outputs=[gr.Plot(label="Placement"), gr.Textbox(label="Result")],
        title="CellCage-lite: cage placement",
        description="Detect cells, label targets, place the maximum valid set of cages.",
    )
    demo.launch()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="CellCage-lite demo.")
    parser.add_argument("--static", action="store_true", help="render to files, skip Gradio")
    parser.add_argument("--out", type=Path, default=Path("app/output"))
    args = parser.parse_args()

    if not args.static and _launch_gradio():
        return
    if not args.static:
        print("Gradio is not installed; rendering static figures instead.")
        print("Install the demo extra with: pip install gradio\n")
    _render_static(args.out)


if __name__ == "__main__":
    main()
