"""Cellpose-style flow representation and the flow U-Net.

The model predicts, for every pixel, a vector pointing toward the center of that
pixel's own cell, plus a foreground score. Pixels whose vectors flow to the same
center form one cell, which separates touching cells and handles any shape (see
README section 1).

This module holds three pieces:

- :func:`masks_to_flows` turns a labeled mask into the flow targets used to train
  the model.
- :func:`flows_to_labels` turns a predicted flow field back into an instance mask
  by following the vectors to their sinks.
- :class:`FlowUNet` is the network itself (torch, imported lazily).

The two flow functions are pure NumPy so they run and are tested without torch.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from cage.mis import connected_components


def masks_to_flows(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert an instance-label image into flow targets and a foreground mask.

    Each foreground pixel gets a unit vector pointing from itself toward its
    cell's centroid. Returns ``(flow, foreground)`` where ``flow`` has shape
    ``(2, H, W)`` holding ``(dy, dx)`` and ``foreground`` is a boolean ``(H, W)``.

    This uses the centroid as the sink, a light stand-in for Cellpose's
    heat-diffusion center; it is enough to define a well-posed flow that
    :func:`flows_to_labels` can invert.
    """
    labels = np.asarray(labels)
    h, w = labels.shape
    flow = np.zeros((2, h, w), dtype=np.float32)
    foreground = labels > 0

    ys, xs = np.nonzero(foreground)
    ids = labels[ys, xs]
    for cell_id in np.unique(ids):
        sel = ids == cell_id
        cy, cx = ys[sel].mean(), xs[sel].mean()
        dy, dx = cy - ys[sel], cx - xs[sel]
        norm = np.hypot(dy, dx) + 1e-8
        flow[0, ys[sel], xs[sel]] = dy / norm
        flow[1, ys[sel], xs[sel]] = dx / norm
    return flow, foreground


def flows_to_labels(
    flow: np.ndarray,
    foreground: np.ndarray,
    n_steps: int = 200,
    step_size: float = 1.0,
    merge_dist: float = 3.0,
) -> np.ndarray:
    """Recover an instance-label image from a flow field.

    Every foreground pixel is advanced along the flow for ``n_steps`` (Euler
    integration); pixels that converge to the same sink are one cell. Sinks
    within ``merge_dist`` of each other are grouped, and each group becomes a
    labeled instance.
    """
    foreground = np.asarray(foreground, dtype=bool)
    ys, xs = np.nonzero(foreground)
    if len(ys) == 0:
        return np.zeros_like(foreground, dtype=np.int32)

    h, w = foreground.shape
    py = ys.astype(np.float32)
    px = xs.astype(np.float32)
    for _ in range(n_steps):
        iy = np.clip(np.round(py).astype(int), 0, h - 1)
        ix = np.clip(np.round(px).astype(int), 0, w - 1)
        py = py + step_size * flow[0, iy, ix]
        px = px + step_size * flow[1, iy, ix]

    # Group pixels whose sinks are close together.
    sinks = np.column_stack([py, px])
    tree = cKDTree(sinks)
    edges = list(tree.query_pairs(merge_dist))
    components = connected_components(len(ys), edges)

    labels = np.zeros((h, w), dtype=np.int32)
    for new_id, comp in enumerate(components, start=1):
        labels[ys[comp], xs[comp]] = new_id
    return labels


class FlowUNet:
    """A small U-Net that predicts a 2-channel flow field and a foreground score.

    Constructed lazily so importing this module never requires torch. Call
    :meth:`build` to get the underlying ``torch.nn.Module``.
    """

    def __init__(self, base_channels: int = 32) -> None:
        self.base_channels = base_channels

    def build(self):
        import torch
        from torch import nn

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        c = self.base_channels

        class _UNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enc1 = block(1, c)
                self.enc2 = block(c, c * 2)
                self.enc3 = block(c * 2, c * 4)
                self.pool = nn.MaxPool2d(2)
                self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
                self.dec2 = block(c * 4, c * 2)
                self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
                self.dec1 = block(c * 2, c)
                # 3 outputs: flow_y, flow_x, foreground logit.
                self.head = nn.Conv2d(c, 3, 1)

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                return self.head(d1)

        return _UNet()
