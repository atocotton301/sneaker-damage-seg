"""Cut a YOLO-seg dataset into overlapping tiles so small damage keeps its pixels.

Each image is covered by a grid of tile x tile windows (stride = tile - overlap,
last window snapped to the edge). Polygons are clipped to each window
(Sutherland-Hodgman) and renormalized. Slivers smaller than --min-area pixels
are dropped. Tiles with no damage are kept at --empty-ratio so the model still
sees plain leather.

Only train and val are tiled. Test stays whole: tiled_predict.py runs the
model tile by tile and stitches predictions back onto the full image.

Usage:
  python tools/tile_yolo.py --src yolo --out yolo_tiles --tile 640 --overlap 128
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

import yaml_writer


def windows(size, tile, overlap):
    """Start offsets covering [0, size) with windows of length tile."""
    if size <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, size - tile, stride))
    starts.append(size - tile)
    return starts


def clip_polygon(pts, x0, y0, x1, y1):
    """Clip a polygon [(x, y), ...] to the rectangle [x0, x1] x [y0, y1]."""
    def clip(points, inside, cross):
        out = []
        for i, cur in enumerate(points):
            prev = points[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out

    def at_x(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

    def at_y(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)

    for inside, cross in ((lambda p: p[0] >= x0, at_x(x0)), (lambda p: p[0] <= x1, at_x(x1)),
                          (lambda p: p[1] >= y0, at_y(y0)), (lambda p: p[1] <= y1, at_y(y1))):
        pts = clip(pts, inside, cross)
        if not pts:
            return []
    return pts


def area(pts):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))) / 2


def read_labels(path, W, H):
    objs = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) >= 7:
                c = int(v[0])
                xy = list(map(float, v[1:]))
                objs.append((c, [(xy[i] * W, xy[i + 1] * H) for i in range(0, len(xy), 2)]))
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="yolo", help="YOLO dataset with images/ labels/ data.yaml")
    ap.add_argument("--out", default="yolo_tiles")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--empty-ratio", type=float, default=0.3, help="share of damage-free tiles to keep")
    ap.add_argument("--min-area", type=float, default=12.0, help="drop clipped pieces below this many px")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    rng = random.Random(args.seed)
    names = [l.split(":", 1)[1].strip() for l in (src / "data.yaml").read_text(encoding="utf-8").splitlines()
             if l.startswith("  ") and ":" in l]
    shutil.rmtree(out, ignore_errors=True)
    for s in args.splits.split(","):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
        n_img = n_tiles = n_empty = n_objs = 0
        for img_path in sorted((src / "images" / s).glob("*.jpg")):
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
            objs = read_labels(src / "labels" / s / f"{img_path.stem}.txt", W, H)
            n_img += 1
            for y in windows(H, args.tile, args.overlap):
                for x in windows(W, args.tile, args.overlap):
                    tw, th = min(args.tile, W - x), min(args.tile, H - y)
                    lines = []
                    for c, pts in objs:
                        clipped = clip_polygon(pts, x, y, x + tw, y + th)
                        if len(clipped) < 3 or area(clipped) < args.min_area:
                            continue
                        lines.append(f"{c} " + " ".join(f"{(px - x) / tw:.6f} {(py - y) / th:.6f}" for px, py in clipped))
                    if not lines and rng.random() >= args.empty_ratio:
                        continue
                    stem = f"{img_path.stem}__x{x}_y{y}"
                    im.crop((x, y, x + tw, y + th)).save(out / "images" / s / f"{stem}.jpg", quality=92)
                    (out / "labels" / s / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                    n_tiles += 1
                    n_empty += not lines
                    n_objs += len(lines)
        print(f"{s}: {n_img} images -> {n_tiles} tiles ({n_empty} empty), {n_objs} polygon pieces")

    # Validation during training uses tiles too; test is evaluated on whole images separately.
    yaml_writer.write_data_yaml(out / "data.yaml", out.resolve(), names)
    text = (out / "data.yaml").read_text(encoding="utf-8").replace("test: images/test\n", "")
    (out / "data.yaml").write_text(text, encoding="utf-8")
    print(f"dataset -> {out}/data.yaml")


if __name__ == "__main__":
    main()
