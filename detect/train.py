"""CLI for training the flow U-Net on LIVECell.

    python -m detect.train --data data/livecell --epochs 50

Training minimizes an MSE loss on the flow channels plus a binary cross-entropy
loss on the foreground channel, with flow targets built by
:func:`detect.model.masks_to_flows`. LIVECell does not ship with the repo, so
without a ``--data`` directory this command prints setup guidance and exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train(data_dir: Path, epochs: int, batch_size: int, lr: float) -> None:
    """Train the flow U-Net. Requires torch and a prepared LIVECell directory."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from .model import FlowUNet

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = FlowUNet().build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    flow_loss = nn.MSELoss()
    fg_loss = nn.BCEWithLogitsLoss()

    dataset = _LiveCellFlows(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running = 0.0
        for image, flow_target, fg_target in loader:
            image = image.to(device)
            flow_target = flow_target.to(device)
            fg_target = fg_target.to(device)

            pred = model(image)
            loss = flow_loss(pred[:, :2], flow_target) + fg_loss(pred[:, 2], fg_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch {epoch + 1}/{epochs}  loss {running / max(1, len(loader)):.4f}")

    out = data_dir / "flowunet.pt"
    torch.save(model.state_dict(), out)
    print(f"Saved weights to {out}")


class _LiveCellFlows:
    """Dataset adapter: yields (image, flow_target, foreground) tensors.

    Expects LIVECell images and instance masks under ``data_dir``. This is the
    integration point for the real dataset; the flow targets come from
    :func:`detect.model.masks_to_flows`.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.items = sorted((data_dir / "masks").glob("*.npy")) if (data_dir / "masks").exists() else []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        import numpy as np
        import torch

        from .model import masks_to_flows

        mask_path = self.items[idx]
        image_path = self.data_dir / "images" / (mask_path.stem + ".npy")
        image = np.load(image_path).astype("float32")
        labels = np.load(mask_path)
        flow, foreground = masks_to_flows(labels)
        return (
            torch.from_numpy(image)[None],
            torch.from_numpy(flow),
            torch.from_numpy(foreground.astype("float32")),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the flow U-Net for detection.")
    parser.add_argument("--data", type=Path, default=Path("data/livecell"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if not args.data.exists():
        print(
            f"No dataset at {args.data}. Download LIVECell "
            "(https://sartorius-research.github.io/LIVECell/) and arrange it as "
            f"{args.data}/images/*.npy and {args.data}/masks/*.npy, then rerun.\n"
            "The pretrained Cellpose backend (detect.detector.Detector) needs no "
            "training and works out of the box."
        )
        return
    train(args.data, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
