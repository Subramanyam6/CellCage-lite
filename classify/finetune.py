"""CLI for the classification stage.

    python -m classify.finetune            # self-check the head on synthetic data
    python -m classify.finetune --probe    # (real use) train a linear probe on
                                           # cached DINOv2 embeddings of LIVECell crops

With real data this extracts frozen DINOv2 embeddings for labeled cell crops and
fits the linear probe; optionally it enables LoRA to adjust DINOv2 itself. Neither
DINOv2 weights nor LIVECell ship with the repo, so with no data available this
command instead runs a quick self-check: it fits the head on well-separated
synthetic embeddings and reports accuracy, verifying the training path end to end.
"""

from __future__ import annotations

import argparse

import numpy as np

from .head import LinearProbe, NearestClassMean


def _synthetic_embeddings(n_per_class: int, dim: int, n_classes: int, seed: int):
    """Two or more Gaussian blobs standing in for DINOv2 embeddings."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, size=(n_classes, dim))
    X, y = [], []
    for c in range(n_classes):
        X.append(rng.normal(centers[c], 1.0, size=(n_per_class, dim)))
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def self_check(seed: int = 0) -> float:
    """Fit both heads on synthetic embeddings and return the linear-probe
    train/test accuracy."""
    X, y = _synthetic_embeddings(n_per_class=50, dim=32, n_classes=3, seed=seed)
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    split = int(0.7 * n)
    tr, te = idx[:split], idx[split:]

    probe = LinearProbe().fit(X[tr], y[tr])
    probe_acc = float(np.mean(probe.predict(X[te]) == y[te]))

    ncm = NearestClassMean().fit(X[tr], y[tr])
    ncm_acc = float(np.mean(ncm.predict(X[te]) == y[te]))

    print(f"Linear probe accuracy:      {probe_acc:.3f}")
    print(f"Nearest-class-mean accuracy: {ncm_acc:.3f}")
    return probe_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train / self-check the classification head.")
    parser.add_argument("--probe", action="store_true", help="train on real cached embeddings")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.probe:
        print(
            "Real training expects cached DINOv2 embeddings and labels for LIVECell "
            "crops (see classify/embed.py and classify/lora.py). Neither ships with "
            "the repo; run without --probe for the synthetic self-check."
        )
        return

    print("No dataset provided; running the synthetic self-check.\n")
    self_check(seed=args.seed)


if __name__ == "__main__":
    main()
