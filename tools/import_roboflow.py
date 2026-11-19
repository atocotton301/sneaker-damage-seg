"""Bring a Roboflow "COCO Segmentation" export into this kit's layout.

Roboflow zips have train/ valid/ test/ folders, each with images and
_annotations.coco.json, plus a placeholder category (the project name).
This merges them into one images/ folder and one annotations.json so the
kit can redo the split itself (group-aware, near-duplicate merged), and
writes classes.txt with the real class names.

Roboflow's original split is kept in each image's "rf_split" field for reference.

Usage:
  python tools/import_roboflow.py --zip path/to/export.zip --out data
  (then run the tools from inside --out, or copy classes.txt to where you run them)
"""
import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


def long_path(p):
    """On Windows, prefix \\\\?\\ so paths over 260 characters still work."""
    p = str(Path(p).resolve())
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        return "\\\\?\\" + p
    return p


def snake(name):
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip")
    src.add_argument("--dir", help="already unzipped export folder")
    ap.add_argument("--out", default="data")
    ap.add_argument("--group-regex",
                    help="regex on the ORIGINAL file name whose first group is the object id, "
                         r"e.g. '^(?:back|front|side\d)_(\d+)' for the Shoe Defects set")
    args = ap.parse_args()

    tmp = None
    if args.zip:
        tmp = Path(tempfile.mkdtemp(prefix="rf_"))
        with zipfile.ZipFile(args.zip) as z:
            for m in z.namelist():  # refuse paths that escape the folder
                if Path(m).is_absolute() or ".." in Path(m).parts:
                    raise SystemExit(f"unsafe path in zip: {m}")
            z.extractall(tmp)
        root = tmp
    else:
        root = Path(args.dir)

    ann_files = sorted(root.rglob("_annotations.coco.json"))
    if not ann_files:
        raise SystemExit("no _annotations.coco.json found; export as 'COCO Segmentation'")

    out = Path(args.out)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    names, images, anns = [], [], []
    for af in ann_files:
        split = af.parent.name
        coco = json.loads(af.read_text(encoding="utf-8"))
        # Roboflow adds a placeholder category (the project name) that is the
        # supercategory of every real class; skip it.
        supers = {c.get("supercategory") for c in coco["categories"]}
        real = [c for c in coco["categories"] if c["name"] not in supers]
        local = {}
        for c in real:
            n = snake(c["name"])
            if n not in names:
                names.append(n)
            local[c["id"]] = names.index(n) + 1
        id_map = {}
        for im in coco["images"]:
            new_id = len(images) + 1
            id_map[im["id"]] = new_id
            src_img = af.parent / im["file_name"]
            # Roboflow names are long; with a deep folder they pass the Windows
            # 260-character path limit. Store a short name and keep the original.
            short = f"rf{new_id:05d}{Path(im['file_name']).suffix.lower()}"
            if src_img.exists():
                shutil.copyfile(long_path(src_img), long_path(img_out / short))
            rec = {"id": new_id, "file_name": short, "original_file_name": im["file_name"],
                   "width": im.get("width"), "height": im.get("height"), "rf_split": split}
            if args.group_regex:
                m = re.search(args.group_regex, im["file_name"])
                if m:
                    rec["group"] = f"G{m.group(1)}"
            images.append(rec)
        for a in coco["annotations"]:
            if a["category_id"] not in local:
                continue
            anns.append({**a, "id": len(anns) + 1, "image_id": id_map[a["image_id"]],
                         "category_id": local[a["category_id"]]})
        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    (out / "annotations.json").write_text(
        json.dumps({"images": images, "annotations": anns, "categories": cats}, ensure_ascii=False),
        encoding="utf-8")
    # classes.txt in the working folder makes every tool use these class names.
    Path("classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"classes: {names} (written to ./classes.txt; delete it to go back to the default classes)")
    print(f"{len(images)} images, {len(anns)} annotations -> {out}")


if __name__ == "__main__":
    main()
