"""Step 5: per-class error analysis on the test split.

Ultralytics `val` already reports mAP. This script answers the follow-up questions
the internship asks for: which images are false positives / misses, and whether
small damage is missed more often.

Inputs:
  --labels  YOLO ground-truth label folder (data/yolo/labels/test)
  --preds   prediction txt folder from `yolo segment predict save_txt=True save_conf=True`
Matching uses the bounding box of each polygon, IoU >= --iou, same class.

Usage:
  python tools/eval_report.py --labels data/yolo/labels/test \
      --preds runs/segment/predict/labels --out reports/eval
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from common import CLASSES

AREA_BINS = [(0, 0.005, "small (<0.5% of image)"), (0.005, 0.03, "medium"), (0.03, 1.01, "large (>3%)")]


def read_objects(path, with_conf):
    objs = []
    if not path.exists():
        return objs
    for line in path.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        cls = int(v[0])
        if with_conf:
            conf, coords = float(v[-1]), list(map(float, v[1:-1]))
        else:
            conf, coords = 1.0, list(map(float, v[1:]))
        xs, ys = coords[0::2], coords[1::2]
        box = (min(xs), min(ys), max(xs), max(ys))
        objs.append({"cls": cls, "conf": conf, "box": box,
                     "area": (box[2] - box[0]) * (box[3] - box[1])})
    return objs


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def area_bin(a):
    return next(name for lo, hi, name in AREA_BINS if lo <= a < hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    gt_dir, pr_dir, out = Path(args.labels), Path(args.preds), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    by_size = defaultdict(lambda: {"gt": 0, "found": 0})
    per_image = []

    for gt_file in sorted(gt_dir.glob("*.txt")):
        gts = read_objects(gt_file, with_conf=False)
        preds = [p for p in read_objects(pr_dir / gt_file.name, with_conf=True) if p["conf"] >= args.conf]
        preds.sort(key=lambda p: -p["conf"])
        matched = set()
        row = {"image": gt_file.stem, "gt": len(gts), "tp": 0, "fp": 0, "fn": 0}
        for p in preds:
            best, best_iou = None, args.iou
            for k, g in enumerate(gts):
                if k in matched or g["cls"] != p["cls"]:
                    continue
                v = iou(p["box"], g["box"])
                if v >= best_iou:
                    best, best_iou = k, v
            if best is None:
                stats[p["cls"]]["fp"] += 1
                row["fp"] += 1
            else:
                matched.add(best)
                stats[p["cls"]]["tp"] += 1
                row["tp"] += 1
        for k, g in enumerate(gts):
            b = by_size[(g["cls"], area_bin(g["area"]))]
            b["gt"] += 1
            if k in matched:
                b["found"] += 1
            else:
                stats[g["cls"]]["fn"] += 1
                row["fn"] += 1
        row["kind"] = ("negative_image_fp" if row["gt"] == 0 and row["fp"] else
                       "has_miss" if row["fn"] else "has_fp" if row["fp"] else "ok")
        per_image.append(row)

    # Per-class table
    lines = [f"IoU>={args.iou}, conf>={args.conf}, images={len(per_image)}", "",
             f"{'class':22} {'TP':>4} {'FP':>4} {'FN':>4} {'precision':>9} {'recall':>7}"]
    for c, name in enumerate(CLASSES):
        s = stats[c]
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
        lines.append(f"{name:22} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {p:>9.3f} {r:>7.3f}")
    lines += ["", "recall by damage size (box area / image area)"]
    for c, name in enumerate(CLASSES):
        for _, _, bin_name in AREA_BINS:
            b = by_size[(c, bin_name)]
            if b["gt"]:
                lines.append(f"  {name:22} {bin_name:24} {b['found']}/{b['gt']} = {b['found'] / b['gt']:.3f}")
    kinds = defaultdict(int)
    for r in per_image:
        kinds[r["kind"]] += 1
    lines += ["", "images by outcome: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")

    per_image.sort(key=lambda r: (-(r["fn"] + r["fp"]), r["image"]))
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "kind", "gt", "tp", "fp", "fn"])
        w.writeheader(); w.writerows(per_image)
    print(text)
    print(f"\nworst images first -> {out / 'per_image.csv'}")


if __name__ == "__main__":
    main()
