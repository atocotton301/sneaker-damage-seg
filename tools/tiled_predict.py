"""Predict on whole images tile by tile and stitch the results back.

Each test image is cut with the same grid as tile_yolo.py, every tile goes
through the model at full resolution, mask polygons are moved back to
full-image coordinates, and duplicates from overlapping tiles are removed
with per-class box NMS. Output is one YOLO-seg txt per image with a
confidence column, the same format eval_report.py and test_eval.py read.

Usage:
  python tools/tiled_predict.py --weights runs/tile_640/weights/best.pt \
      --images yolo/images/test --out runs/tile_640_pred/labels --tile 640 --overlap 128
"""
import argparse
from pathlib import Path

from PIL import Image

from tile_yolo import windows


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, thr):
    """dets: list of (cls, conf, box, pts). Keep the highest-confidence one per overlapping group."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if all(k[0] != d[0] or box_iou(k[2], d[2]) < thr for k in keep):
            keep.append(d)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--conf", type=float, default=0.1, help="low on purpose; evaluation applies its own cut")
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(args.images).glob("*.jpg"))
    for n, path in enumerate(paths, 1):
        im = Image.open(path).convert("RGB")
        W, H = im.size
        crops, offsets = [], []
        for y in windows(H, args.tile, args.overlap):
            for x in windows(W, args.tile, args.overlap):
                crops.append(im.crop((x, y, min(x + args.tile, W), min(y + args.tile, H))))
                offsets.append((x, y))
        dets = []
        for i in range(0, len(crops), args.batch):
            results = model.predict(crops[i:i + args.batch], imgsz=args.tile, conf=args.conf, verbose=False)
            for r, (ox, oy) in zip(results, offsets[i:i + args.batch]):
                if r.masks is None:
                    continue
                for poly, c, s in zip(r.masks.xy, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    if len(poly) < 3:
                        continue
                    pts = [(float(px) + ox, float(py) + oy) for px, py in poly]
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    dets.append((int(c), float(s), (min(xs), min(ys), max(xs), max(ys)), pts))
        lines = [f"{c} " + " ".join(f"{px / W:.6f} {py / H:.6f}" for px, py in pts) + f" {s:.4f}"
                 for c, s, _, pts in nms(dets, args.nms_iou)]
        (out / f"{path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if n % 50 == 0 or n == len(paths):
            print(f"{n}/{len(paths)} images")
    print(f"predictions -> {out}")


if __name__ == "__main__":
    main()
