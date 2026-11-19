"""Step 3-4: turn split COCO files into a YOLO-segmentation dataset.

Images are copied upright (EXIF orientation applied), so labels and pixels agree.
Polygon coordinates in COCO are assumed to be on the upright image, as CVAT exports them.

Usage:
  python tools/coco2yolo.py --splits data/splits --images data/images --out data/yolo
"""
import argparse
from pathlib import Path

from PIL import Image

import yaml_writer
from common import CLASSES, load_coco, open_upright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="folder with train.json / val.json / test.json")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--max-side", type=int, default=0,
                    help="shrink images so the long side is at most this many pixels (0 = keep). "
                         "Labels are normalized, so they stay valid; use it to cut upload size.")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality of the exported images")
    args = ap.parse_args()

    splits, img_dir, out = Path(args.splits), Path(args.images), Path(args.out)
    skipped = 0
    for s in ("train", "val", "test"):
        coco = load_coco(splits / f"{s}.json")
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        anns = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        written = 0
        for im in coco["images"]:
            src = img_dir / im["file_name"]
            if not src.exists():
                print(f"  missing, skipped: {im['file_name']} (fix it in audit first)")
                continue
            written += 1
            # Keep the original file name (flattened) so results trace back to the source photo.
            stem = Path(im["file_name"]).as_posix().replace("/", "__").rsplit(".", 1)[0]
            img, _ = open_upright(src)
            W, H = img.size  # label normalization uses the original size
            saved = img.convert("RGB")
            if args.max_side and max(W, H) > args.max_side:
                saved.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
            saved.save(out / "images" / s / f"{stem}.jpg", quality=args.quality)
            lines = []
            for a in anns.get(im["id"], []):
                seg = a.get("segmentation")
                if not isinstance(seg, list):
                    skipped += 1
                    continue
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    pts = [min(max(v / (W if i % 2 == 0 else H), 0.0), 1.0) for i, v in enumerate(poly)]
                    lines.append(f"{a['category_id']} " + " ".join(f"{p:.6f}" for p in pts))
            (out / "labels" / s / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"{s}: {written} images")

    yaml_writer.write_data_yaml(out / "data.yaml", out.resolve(), CLASSES)
    if skipped:
        print(f"skipped {skipped} non-polygon annotations (RLE)")
    print(f"dataset -> {out}/data.yaml")


if __name__ == "__main__":
    main()
