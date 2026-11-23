"""Test-split evaluation for finished runs, plus a look at false positives.

Writes to --runs:
  test_metrics.json   Ultralytics test mAP / P / R per model and class
  conf_sweep.json     TP / FP / FN, precision, recall at several confidence cuts
  fp_gallery_<run>.jpg  crops of the highest-confidence false positives
                        (green = labeled damage, red = prediction with no matching label)

Run from the dataset root (the folder holding yolo/ and tools/):
  python tools/test_eval.py --runs /content/drive/MyDrive/sneaker-damage-seg/runs
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from eval_report import iou, read_objects  # noqa: E402

# name -> imgsz for Ultralytics test val. None = tiled run, evaluated only from stitched predictions.
RUNS = {"base_640": 640, "size_1024": 1024, "tile_640": None, "tile_640_e60": None}
CONFS = [0.25, 0.35, 0.5, 0.65]


def ultralytics_test(runs_dir, data_yaml):
    from ultralytics import YOLO
    out = {}
    for name, imgsz in RUNS.items():
        w = runs_dir / name / "weights" / "best.pt"
        if imgsz is None or not w.exists():
            continue
        r = YOLO(str(w)).val(data=data_yaml, split="test", imgsz=imgsz, plots=False,
                             verbose=False, project=str(runs_dir), name=f"{name}_test", exist_ok=True)
        out[name] = {
            "box_map50": float(r.box.map50), "box_map50_95": float(r.box.map),
            "mask_map50": float(r.seg.map50), "mask_map50_95": float(r.seg.map),
            "mask_p": float(r.seg.mp), "mask_r": float(r.seg.mr),
            "per_class": {n: {"box_map50": float(r.box.ap50[i]), "mask_map50": float(r.seg.ap50[i]),
                              "mask_p": float(r.seg.p[i]), "mask_r": float(r.seg.r[i])}
                          for i, n in r.names.items()},
        }
        print(name, json.dumps(out[name], indent=None)[:200])
    return out


def match(gts, preds, thr=0.5):
    """Return (tp_preds, fp_preds, matched_gt_idx)."""
    preds = sorted(preds, key=lambda p: -p["conf"])
    used, tp, fp = set(), [], []
    for p in preds:
        best, best_iou = None, thr
        for k, g in enumerate(gts):
            if k in used or g["cls"] != p["cls"]:
                continue
            v = iou(p["box"], g["box"])
            if v >= best_iou:
                best, best_iou = k, v
        if best is None:
            fp.append(p)
        else:
            used.add(best)
            tp.append(p)
    return tp, fp, used


def conf_sweep(runs_dir, gt_dir):
    out = {}
    for name in RUNS:
        pr_dir = runs_dir / f"{name}_pred" / "labels"
        if not pr_dir.exists():
            continue
        rows = []
        for c in CONFS:
            tp = fp = fn = 0
            neg_fp_images = 0
            for g in sorted(gt_dir.glob("*.txt")):
                gts = read_objects(g, with_conf=False)
                preds = [p for p in read_objects(pr_dir / g.name, with_conf=True) if p["conf"] >= c]
                t, f, used = match(gts, preds)
                tp, fp, fn = tp + len(t), fp + len(f), fn + len(gts) - len(used)
                neg_fp_images += int(not gts and bool(f))
            rows.append({"conf": c, "tp": tp, "fp": fp, "fn": fn,
                         "precision": tp / (tp + fp) if tp + fp else 0.0,
                         "recall": tp / (tp + fn) if tp + fn else 0.0,
                         "clean_images_with_fp": neg_fp_images})
        out[name] = rows
        for r in rows:
            print(name, r)
    return out


def fp_gallery(runs_dir, gt_dir, img_dir, name, n=16, tile=260):
    pr_dir = runs_dir / f"{name}_pred" / "labels"
    if not pr_dir.exists():
        return
    items = []
    for g in sorted(gt_dir.glob("*.txt")):
        gts = read_objects(g, with_conf=False)
        _, fps, _ = match(gts, [p for p in read_objects(pr_dir / g.name, with_conf=True) if p["conf"] >= 0.25])
        items += [(p["conf"], g.stem, p, gts) for p in fps]
    items.sort(key=lambda x: -x[0])
    items = items[:n]
    names = {0: "abrasions", 1: "discoloration", 2: "scars"}
    cols = 4
    sheet = Image.new("RGB", (cols * tile, ((len(items) + cols - 1) // cols) * (tile + 18)), "white")
    d_sheet = ImageDraw.Draw(sheet)
    for k, (conf, stem, p, gts) in enumerate(items):
        im = Image.open(img_dir / f"{stem}.jpg").convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        lw = max(2, W // 400)
        for g in gts:
            b = g["box"]
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(0, 200, 0), width=lw)
        b = p["box"]
        d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(230, 30, 30), width=lw)
        cx, cy = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
        r = max((b[2] - b[0]) * W, (b[3] - b[1]) * H) * 2.5 + 80
        crop = im.crop((int(cx - r), int(cy - r), int(cx + r), int(cy + r))).resize((tile, tile))
        x, y = (k % cols) * tile, (k // cols) * (tile + 18)
        sheet.paste(crop, (x, y + 18))
        d_sheet.text((x + 4, y + 2), f"{stem} {names.get(p['cls'], p['cls'])} {conf:.2f}", fill="black")
    out = runs_dir / f"fp_gallery_{name}.jpg"
    sheet.save(out, quality=90)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", default="yolo/data.yaml")
    ap.add_argument("--skip-ultralytics", action="store_true")
    args = ap.parse_args()
    runs_dir = Path(args.runs)
    gt_dir, img_dir = Path("yolo/labels/test"), Path("yolo/images/test")

    if not args.skip_ultralytics:
        (runs_dir / "test_metrics.json").write_text(json.dumps(ultralytics_test(runs_dir, args.data), indent=2))
    (runs_dir / "conf_sweep.json").write_text(json.dumps(conf_sweep(runs_dir, gt_dir), indent=2))
    for name in RUNS:
        fp_gallery(runs_dir, gt_dir, img_dir, name)


if __name__ == "__main__":
    main()
