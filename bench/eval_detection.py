"""Evaluate pretrained Cellpose detection on LIVECell (human cell lines only).

Reports the standard instance-segmentation metrics against the LIVECell
ground-truth masks: mean IoU and Dice of matched cells, and average precision
across IoU thresholds. Only human cell lines are used; BV2 (mouse microglia) is
excluded.

    python -m bench.eval_detection --n-images 40

Needs the ML stack (torch, cellpose, pycocotools) and the downloaded LIVECell
data under data/livecell/.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Human LIVECell lines; BV2 (mouse) is deliberately excluded.
HUMAN_LINES = {"a172", "bt474", "huh7", "mcf7", "shsy5y", "skbr3", "skov3"}
EXCLUDED = {"bv2"}


def cell_line_of(file_name: str) -> str:
    """LIVECell encodes the cell line as the filename prefix before the first '_'."""
    return Path(file_name).name.split("_")[0].lower()


def build_image_index(images_root: Path) -> dict[str, Path]:
    """Map each image basename to its path on disk (images.zip nests by split)."""
    index: dict[str, Path] = {}
    for path in images_root.rglob("*.tif"):
        index[path.name] = path
    return index


def gt_instance_mask(coco, img_id: int, height: int, width: int) -> np.ndarray:
    """Rasterize all ground-truth annotations for an image into a label mask."""
    mask = np.zeros((height, width), dtype=np.int32)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    for label, ann in enumerate(coco.loadAnns(ann_ids), start=1):
        m = coco.annToMask(ann).astype(bool)
        mask[m] = label
    return mask


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IoU between every predicted and ground-truth instance.

    Returns ``(iou, pred_ids, gt_ids)`` where ``iou[i, j]`` is the IoU of
    predicted instance ``pred_ids[i]`` with ground-truth instance ``gt_ids[j]``.
    """
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return np.zeros((len(pred_ids), len(gt_ids))), pred_ids, gt_ids

    pred_area = {i: int((pred == i).sum()) for i in pred_ids}
    gt_area = {j: int((gt == j).sum()) for j in gt_ids}

    # Intersections in one pass over foreground pixels.
    fg = (pred > 0) & (gt > 0)
    pairs = pred[fg].astype(np.int64) * (gt.max() + 1) + gt[fg].astype(np.int64)
    counts = np.bincount(pairs)
    pred_pos = {p: i for i, p in enumerate(pred_ids)}
    gt_pos = {g: j for j, g in enumerate(gt_ids)}

    iou = np.zeros((len(pred_ids), len(gt_ids)))
    for code, inter in enumerate(counts):
        if inter == 0:
            continue
        p, g = divmod(code, gt.max() + 1)
        if p in pred_pos and g in gt_pos:
            union = pred_area[p] + gt_area[g] - inter
            iou[pred_pos[p], gt_pos[g]] = inter / union
    return iou, pred_ids, gt_ids


def greedy_match(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Greedily pair predictions to ground truth by descending IoU above a
    threshold (one-to-one)."""
    matches = []
    used_pred, used_gt = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for i, j in order:
        if iou[i, j] < threshold:
            break
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matches.append((int(i), int(j)))
    return matches


def evaluate(ann_path: Path, images_root: Path, n_images: int, seed: int, device: str | None):
    from cellpose import models
    from pycocotools.coco import COCO

    coco = COCO(str(ann_path))
    index = build_image_index(images_root)

    # Group images by (human) cell line, then sample round-robin for balance.
    by_line: dict[str, list] = defaultdict(list)
    for img in coco.loadImgs(coco.getImgIds()):
        line = cell_line_of(img["file_name"])
        if line in HUMAN_LINES and img["file_name"] in index:
            by_line[line].append(img)

    rng = np.random.default_rng(seed)
    for line in by_line:
        rng.shuffle(by_line[line])
    sampled = _round_robin(by_line, n_images)
    print(f"Evaluating {len(sampled)} images across {len(by_line)} human cell lines "
          f"(excluded: {', '.join(sorted(EXCLUDED))}).")

    model = models.CellposeModel(gpu=device != "cpu")

    thresholds = np.arange(0.5, 1.0, 0.05)
    matched_iou, matched_dice = [], []
    ap_hits = {round(t, 2): [0, 0, 0] for t in thresholds}  # tp, fp, fn

    for k, img in enumerate(sampled, start=1):
        image = _load_image(index[img["file_name"]])
        pred, *_ = model.eval(image)
        pred = np.asarray(pred, dtype=np.int32)
        gt = gt_instance_mask(coco, img["id"], img["height"], img["width"])

        iou, pred_ids, gt_ids = iou_matrix(pred, gt)
        for t in thresholds:
            m = greedy_match(iou, t)
            tp = len(m)
            ap_hits[round(t, 2)][0] += tp
            ap_hits[round(t, 2)][1] += len(pred_ids) - tp
            ap_hits[round(t, 2)][2] += len(gt_ids) - tp
            if abs(t - 0.5) < 1e-6:
                for i, j in m:
                    matched_iou.append(iou[i, j])
                    matched_dice.append(2 * iou[i, j] / (1 + iou[i, j]))
        print(f"  [{k}/{len(sampled)}] {img['file_name']}: "
              f"{len(pred_ids)} predicted, {len(gt_ids)} ground truth", flush=True)

    aps = {t: tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
           for t, (tp, fp, fn) in ap_hits.items()}
    results = {
        "n_images": len(sampled),
        "cell_lines": sorted(by_line),
        "mean_iou_matched": float(np.mean(matched_iou)) if matched_iou else 0.0,
        "mean_dice_matched": float(np.mean(matched_dice)) if matched_dice else 0.0,
        "ap_50": aps[0.5],
        "ap_50_95": float(np.mean(list(aps.values()))),
    }
    return results


def _round_robin(by_line: dict[str, list], n: int) -> list:
    out, lines = [], list(by_line)
    i = 0
    while len(out) < n and any(by_line.values()):
        line = lines[i % len(lines)]
        if by_line[line]:
            out.append(by_line[line].pop())
        i += 1
        if i > n * 10:
            break
    return out


def _load_image(path: Path) -> np.ndarray:
    import tifffile

    img = tifffile.imread(str(path)).astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-8)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Cellpose on LIVECell (human lines).")
    parser.add_argument("--ann", type=Path, default=Path("data/livecell/livecell_coco_test.json"))
    parser.add_argument("--images", type=Path, default=Path("data/livecell/images"))
    parser.add_argument("--n-images", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    results = evaluate(args.ann, args.images, args.n_images, args.seed, args.device)
    print("\n=== Detection (Cellpose on LIVECell, human lines) ===")
    print(f"Images:            {results['n_images']}")
    print(f"Mean IoU (matched): {results['mean_iou_matched']:.3f}")
    print(f"Mean Dice (matched):{results['mean_dice_matched']:.3f}")
    print(f"AP @ 0.5:           {results['ap_50']:.3f}")
    print(f"AP @ [0.5:0.95]:    {results['ap_50_95']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
