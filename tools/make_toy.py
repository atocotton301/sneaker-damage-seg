"""Make a small fake dataset with the problems real data has, to test the tools.

Planted problems: one exact duplicate, one near duplicate under a different shoe id,
one EXIF-rotated photo, one image missing on disk, a misspelled class name,
one file without a shoe id, and images with no damage.
Also writes fake predictions so eval_report.py can be tried without a GPU.

Usage:
  python tools/make_toy.py --out toy
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

from common import save_json

W, H = 640, 480


def shoe_photo(rng, damages):
    # Random cluttered background, like a shop counter.
    bg = Image.new("RGB", (W, H), tuple(rng.randint(120, 230) for _ in range(3)))
    d = ImageDraw.Draw(bg)
    for _ in range(rng.randint(6, 14)):
        x, y = rng.randint(-50, W), rng.randint(-50, H)
        d.rectangle([x, y, x + rng.randint(30, 200), y + rng.randint(30, 200)],
                    fill=tuple(rng.randint(60, 255) for _ in range(3)))
    img = bg
    # Shoe at a random place and size.
    sx, sy = rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0)
    ox, oy = rng.randint(0, int(W * (1 - 0.85 * sx))), rng.randint(0, int(H * (1 - 0.6 * sy)))
    def T(x, y):
        return (ox + x * sx, oy + y * sy)
    body = tuple(rng.randint(30, 120) for _ in range(3))
    d.rounded_rectangle([*T(10, 20), *T(490, 230)], 40, fill=body)          # upper
    d.rectangle([*T(0, 230), *T(500, 280)], fill=(235, 235, 235))           # sole
    polys = []
    for cls in damages:
        if cls == 0:  # heel lining damage: blob near the heel
            cx, cy, r = rng.uniform(40, 130), rng.uniform(60, 180), rng.uniform(12, 40)
            pts = [T(cx + r * dx, cy + r * dy) for dx, dy in ((-1, -.6), (0, -1), (1, -.4), (.8, .9), (-.5, 1))]
            d.polygon(pts, fill=(250, 220, 120))
        else:         # sole separation: thin dark gap between upper and sole
            x0 = rng.uniform(50, 330)
            x1 = x0 + rng.uniform(40, 140)
            pts = [T(x0, 226), T(x1, 226), T(x1, 234), T(x0, 234)]
            d.polygon(pts, fill=(15, 15, 15))
        polys.append((cls, [round(v, 1) for p in pts for v in p]))
    return img, polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="toy")
    ap.add_argument("--shoes", type=int, default=12)
    args = ap.parse_args()
    rng = random.Random(7)
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    img_dir = out / "images"
    img_dir.mkdir(parents=True)

    images, anns = [], []
    def add(name, img, polys, write=True, exif_orient=None):
        if write:
            if exif_orient:
                exif = Image.Exif(); exif[274] = exif_orient
                # Stored rotated, EXIF says how to show it upright.
                img.rotate(90, expand=True).save(img_dir / name, exif=exif)
            else:
                img.save(img_dir / name, quality=92)
        iid = len(images) + 1
        images.append({"id": iid, "file_name": name, "width": W, "height": H})
        for cls, poly in polys:
            xs, ys = poly[0::2], poly[1::2]
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": cls + 1,
                         "segmentation": [poly], "iscrowd": 0,
                         "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                         "area": (max(xs) - min(xs)) * (max(ys) - min(ys))})
        return iid

    first = None
    for s in range(1, args.shoes + 1):
        damages = rng.choice([[], [0], [1], [0, 1], [1, 1]])
        for session in "AB":
            for view in ("heel", "soleL"):
                img, polys = shoe_photo(rng, damages)
                name = f"S{s:03d}_{session}_{view}_01.jpg"
                add(name, img, polys)
                if first is None:
                    first = (img, polys)

    # Planted problems
    img, polys = first
    add("S001_A_heel_01_copy.jpg", img, polys)                                   # exact-ish copy, same shoe
    shutil.copy(img_dir / "S001_A_heel_01.jpg", img_dir / "S001_A_heel_01_copy.jpg")
    add("S099_A_heel_01.jpg", ImageEnhance.Brightness(img).enhance(1.04), polys)  # same shoe, new id
    rimg, rpolys = shoe_photo(rng, [0])
    add("S013_A_heel_01.jpg", rimg, rpolys, exif_orient=6)                         # EXIF rotated
    add("S014_A_heel_01.jpg", *shoe_photo(rng, [1]), write=False)                  # missing on disk
    add("IMG_2041.jpg", *shoe_photo(rng, []))                                      # no shoe id
    Image.new("RGB", (W, H), "white").save(img_dir / "S015_A_top_01.jpg")          # not in COCO

    cats = [{"id": 1, "name": "heel_lining_damage"}, {"id": 2, "name": "Sole Separation "}]
    save_json({"images": images, "annotations": anns, "categories": cats}, out / "annotations.json")
    print(f"toy dataset: {len(images)} COCO images, {len(anns)} polygons -> {out}")


if __name__ == "__main__":
    main()
