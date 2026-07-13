"""Evaluate frozen DINOv2 embeddings + a lightweight head on LIVECell cell types.

Crops individual cells (by ground-truth bounding box), embeds each with frozen
DINOv2, and classifies the cell line. Reports the linear-probe accuracy and macro
F1, and few-shot prototypical accuracy at k = 1, 5, 10 examples per class. Only
human cell lines are used; BV2 (mouse) is excluded.

Train and test are split by image, so no cell from a training image appears in
the test set.

    python -m bench.eval_classification --per-class-cells 250

Needs the ML stack (torch, transformers) and the downloaded LIVECell data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.eval_detection import HUMAN_LINES, build_image_index, cell_line_of


def collect_crops(
    coco,
    index: dict[str, Path],
    per_class_images: int,
    per_class_cells: int,
    crop_size: int,
    context: float,
    min_cell_size: int,
    seed: int,
):
    """Crop cells from a balanced set of images, one label per cell line.

    Each crop is contrast-normalized on its own (phase-contrast cells are low
    contrast), and cells smaller than ``min_cell_size`` pixels on a side are
    skipped because their upscaled crops carry little morphology.
    """
    import tifffile
    from PIL import Image

    rng = np.random.default_rng(seed)
    by_line: dict[str, list] = defaultdict(list)
    for img in coco.loadImgs(coco.getImgIds()):
        line = cell_line_of(img["file_name"])
        if line in HUMAN_LINES and img["file_name"] in index:
            by_line[line].append(img)

    crops, labels, image_ids = [], [], []
    for line, imgs in by_line.items():
        rng.shuffle(imgs)
        collected = 0
        for img in imgs[:per_class_images]:
            if collected >= per_class_cells:
                break
            raw = tifffile.imread(str(index[img["file_name"]])).astype(np.float32)
            h, w = raw.shape
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img["id"])):
                if collected >= per_class_cells:
                    break
                x, y, bw, bh = ann["bbox"]
                if bw < min_cell_size or bh < min_cell_size:
                    continue
                pad_w, pad_h = bw * (context - 1) / 2, bh * (context - 1) / 2
                x0, x1 = int(max(0, x - pad_w)), int(min(w, x + bw + pad_w))
                y0, y1 = int(max(0, y - pad_h)), int(min(h, y + bh + pad_h))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                patch = raw[y0:y1, x0:x1]
                # Per-crop contrast stretch to 0-255.
                patch = 255.0 * (patch - patch.min()) / (np.ptp(patch) + 1e-8)
                pil = Image.fromarray(patch.astype(np.uint8)).convert("RGB").resize(
                    (crop_size, crop_size)
                )
                crops.append(np.asarray(pil))
                labels.append(line)
                image_ids.append(img["id"])
                collected += 1
    return crops, np.array(labels), np.array(image_ids)


def embed(crops, model_name: str, device: str, batch: int):
    """Return frozen DINOv2 CLS embeddings for a list of crops."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    features = []
    for i in range(0, len(crops), batch):
        chunk = [c for c in crops[i : i + batch]]
        inputs = processor(images=chunk, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        # DINOv2 linear-probe protocol: CLS token concatenated with the mean of
        # the patch tokens.
        cls = out.last_hidden_state[:, 0]
        patch_mean = out.last_hidden_state[:, 1:].mean(dim=1)
        feat = torch.cat([cls, patch_mean], dim=1)
        features.append(feat.cpu().numpy())
        print(f"  embedded {min(i + batch, len(crops))}/{len(crops)}", flush=True)
    return np.concatenate(features)


def linear_probe(X_tr, y_tr, X_te, y_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(X_tr), y_tr)
    pred = clf.predict(scaler.transform(X_te))
    return accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")


def mean_pairwise_binary(X_tr, y_tr, X_te, y_te) -> float:
    """Average two-class linear-probe accuracy over every pair of cell lines.

    This reflects the pipeline's actual job, telling one target type from
    another, rather than the much harder all-at-once fine-grained ID.
    """
    import itertools

    classes = np.unique(y_tr)
    accs = []
    for a, b in itertools.combinations(classes, 2):
        tr = np.isin(y_tr, [a, b])
        te = np.isin(y_te, [a, b])
        if te.sum() == 0 or len(np.unique(y_tr[tr])) < 2:
            continue
        acc, _ = linear_probe(X_tr[tr], y_tr[tr], X_te[te], y_te[te])
        accs.append(acc)
    return float(np.mean(accs)) if accs else 0.0


def few_shot(X_tr, y_tr, X_te, y_te, k: int, seed: int, trials: int = 5) -> float:
    """Prototypical (nearest-class-mean) accuracy with k examples per class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_tr)

    def norm(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)

    Xte = norm(X_te)
    accs = []
    for _ in range(trials):
        protos = []
        for c in classes:
            idx = np.where(y_tr == c)[0]
            chosen = rng.choice(idx, size=min(k, len(idx)), replace=False)
            protos.append(norm(X_tr[chosen]).mean(axis=0))
        protos = norm(np.stack(protos))
        pred = classes[np.argmax(Xte @ protos.T, axis=1)]
        accs.append(float(np.mean(pred == y_te)))
    return float(np.mean(accs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 classification on LIVECell.")
    parser.add_argument("--ann", type=Path, default=Path("data/livecell/livecell_coco_test.json"))
    parser.add_argument("--images", type=Path, default=Path("data/livecell/images"))
    parser.add_argument("--model", type=str, default="facebook/dinov2-small")
    parser.add_argument("--per-class-images", type=int, default=25)
    parser.add_argument("--per-class-cells", type=int, default=250)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--min-cell-size", type=int, default=18)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from pycocotools.coco import COCO

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    coco = COCO(str(args.ann))
    index = build_image_index(args.images)

    print("Cropping cells...")
    crops, labels, image_ids = collect_crops(
        coco, index, args.per_class_images, args.per_class_cells,
        args.crop_size, args.context, args.min_cell_size, args.seed,
    )
    print(f"{len(crops)} cells across {len(set(labels))} human lines "
          f"({dict(zip(*np.unique(labels, return_counts=True)))}).")

    print(f"Embedding with {args.model} on {device}...")
    X = embed(crops, args.model, device, args.batch)

    # Split by image so cells from one image stay together.
    rng = np.random.default_rng(args.seed)
    unique_imgs = np.unique(image_ids)
    rng.shuffle(unique_imgs)
    n_test = max(1, int(0.4 * len(unique_imgs)))
    test_imgs = set(unique_imgs[:n_test].tolist())
    is_test = np.array([i in test_imgs for i in image_ids])

    X_tr, y_tr = X[~is_test], labels[~is_test]
    X_te, y_te = X[is_test], labels[is_test]

    acc, f1 = linear_probe(X_tr, y_tr, X_te, y_te)
    pairwise = mean_pairwise_binary(X_tr, y_tr, X_te, y_te)
    fs = {k: few_shot(X_tr, y_tr, X_te, y_te, k, args.seed) for k in (1, 5, 10)}

    results = {
        "model": args.model,
        "n_cells": len(crops),
        "n_classes": int(len(set(labels))),
        "train_cells": int((~is_test).sum()),
        "test_cells": int(is_test.sum()),
        "linear_probe_accuracy": acc,
        "linear_probe_macro_f1": f1,
        "mean_pairwise_binary_accuracy": pairwise,
        "few_shot_accuracy": fs,
    }
    print("\n=== Classification (DINOv2 + head, LIVECell human lines) ===")
    print(f"Cells: {results['n_cells']}  Classes: {results['n_classes']}  "
          f"(train {results['train_cells']} / test {results['test_cells']})")
    print(f"Linear probe (7-way):      accuracy {acc:.3f}, macro F1 {f1:.3f}")
    print(f"Mean pairwise (2-way):     accuracy {pairwise:.3f}")
    for k, v in fs.items():
        print(f"Few-shot k={k} (7-way):      accuracy {v:.3f}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
