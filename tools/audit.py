"""Step 1-2: check images and COCO labels before building a dataset.

Finds: images missing on disk, images with no entry in the COCO file, EXIF-rotated
images, odd sizes, exact and near duplicates (and whether they cross shoe groups),
class-name variants, empty or out-of-bounds polygons, files without a shoe group.

Usage:
  python tools/audit.py --images data/images --coco data/annotations.json --out reports/audit
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from common import (CLASSES, DEFAULT_GROUP_REGEX, IMAGE_EXTS, dhash, group_of,
                    load_coco, near_duplicate_pairs, normalize_class, open_upright,
                    save_json, sha256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--coco", help="COCO json exported from CVAT")
    ap.add_argument("--out", default="reports/audit")
    ap.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX)
    ap.add_argument("--near-dup-distance", type=int, default=20,
                    help="max differing bits out of 256; raise to catch more, lower for fewer false alarms")
    args = ap.parse_args()

    img_dir, out = Path(args.images), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    issues = []  # (severity, kind, file, detail)

    def issue(sev, kind, file, detail=""):
        issues.append({"severity": sev, "kind": kind, "file": file, "detail": detail})

    # A "group" field in the COCO images (set by import_roboflow.py) wins over the file name.
    coco = load_coco(args.coco) if args.coco else None
    coco_group = {}
    if coco:
        coco_group = {im["file_name"].replace("\\", "/"): im["group"]
                      for im in coco["images"] if im.get("group")}

    # --- images on disk
    files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    rows, hashes = [], []
    for p in files:
        rel = p.relative_to(img_dir).as_posix()
        try:
            img, orient = open_upright(p)
        except Exception as e:  # unreadable file
            issue("error", "unreadable_image", rel, str(e))
            continue
        w, h = img.size
        g = coco_group.get(rel) or group_of(rel, args.group_regex)
        rows.append({"file": rel, "width": w, "height": h, "exif_orientation": orient,
                     "group": g or "", "sha256": sha256(p)})
        hashes.append(dhash(img))
        if orient not in (1, None):
            issue("warn", "exif_rotated", rel, f"orientation={orient}; train on the upright image")
        if g is None:
            issue("warn", "no_group", rel, "file name does not match the shoe-id pattern")
        if min(w, h) < 320:
            issue("warn", "small_image", rel, f"{w}x{h}")

    # --- duplicates
    by_sha = defaultdict(list)
    for r in rows:
        by_sha[r["sha256"]].append(r["file"])
    for files_same in by_sha.values():
        if len(files_same) > 1:
            issue("warn", "exact_duplicate", files_same[0], " = ".join(files_same[1:]))

    near = []
    for i, j, d in near_duplicate_pairs(hashes, args.near_dup_distance):
        a, b = rows[i], rows[j]
        if a["sha256"] == b["sha256"]:
            continue
        cross = a["group"] != b["group"]
        near.append({"file_a": a["file"], "file_b": b["file"], "hamming": d,
                     "group_a": a["group"], "group_b": b["group"], "cross_group": cross})
        if cross:
            issue("warn", "near_duplicate_cross_group", a["file"],
                  f"{b['file']} (d={d}); likely the same shoe under two ids -> leakage risk")

    # --- COCO checks
    label_stats = {}
    if coco:
        disk = {r["file"]: r for r in rows}
        coco_names = {im["file_name"].replace("\\", "/") for im in coco["images"]}
        for im in coco["images"]:
            fn = im["file_name"].replace("\\", "/")
            if fn not in disk:
                issue("error", "missing_on_disk", fn, "listed in COCO but not found")
            else:
                r = disk[fn]
                if (im.get("width"), im.get("height")) not in ((r["width"], r["height"]), (None, None)):
                    issue("warn", "size_mismatch", fn,
                          f"COCO {im.get('width')}x{im.get('height')} vs file {r['width']}x{r['height']}")
        for fn in disk:
            if fn not in coco_names:
                issue("warn", "not_in_coco", fn, "image on disk has no COCO entry")

        cat_map = {}
        for c in coco["categories"]:
            canon, key = normalize_class(c["name"])
            cat_map[c["id"]] = canon
            if canon is None:
                issue("error", "unknown_class", "", f"category '{c['name']}' is not in {CLASSES}")
            elif c["name"] != canon:
                issue("warn", "class_name_variant", "", f"'{c['name']}' -> '{canon}'")

        imgs = {im["id"]: im for im in coco["images"]}
        per_img = Counter()
        per_cls = Counter()
        for a in coco["annotations"]:
            im = imgs.get(a["image_id"])
            fn = im["file_name"] if im else f"image_id={a['image_id']}"
            if im is None:
                issue("error", "orphan_annotation", fn, f"annotation {a['id']}")
                continue
            per_img[a["image_id"]] += 1
            per_cls[cat_map.get(a["category_id"]) or "unknown"] += 1
            seg = a.get("segmentation")
            if isinstance(seg, dict):
                issue("warn", "rle_mask", fn, f"annotation {a['id']} is RLE; export polygons instead")
                continue
            if not seg or all(len(s) < 6 for s in seg):
                issue("error", "empty_polygon", fn, f"annotation {a['id']}")
                continue
            W, H = im.get("width"), im.get("height")
            if W and H:
                xs = [v for s in seg for v in s[0::2]]
                ys = [v for s in seg for v in s[1::2]]
                if min(xs) < -1 or min(ys) < -1 or max(xs) > W + 1 or max(ys) > H + 1:
                    issue("error", "polygon_out_of_bounds", fn, f"annotation {a['id']}")
        label_stats = {
            "images_in_coco": len(coco["images"]),
            "images_with_labels": len(per_img),
            "images_without_labels": len(coco["images"]) - len(per_img),
            "annotations_per_class": dict(per_cls),
        }

    # --- write reports
    with open(out / "images.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "width", "height", "exif_orientation", "group", "sha256"])
        w.writeheader(); w.writerows(rows)
    with open(out / "issues.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["severity", "kind", "file", "detail"])
        w.writeheader(); w.writerows(issues)
    with open(out / "near_duplicates.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file_a", "file_b", "hamming", "group_a", "group_b", "cross_group"])
        w.writeheader(); w.writerows(near)

    summary = {
        "images_on_disk": len(rows),
        "groups": len({r["group"] for r in rows if r["group"]}),
        "issues_by_kind": dict(Counter(i["kind"] for i in issues)),
        "errors": sum(i["severity"] == "error" for i in issues),
        "near_duplicate_pairs": len(near),
        **label_stats,
    }
    save_json(summary, out / "summary.json")

    print(f"images {summary['images_on_disk']}, shoe groups {summary['groups']}")
    if label_stats:
        print(f"labeled images {label_stats['images_with_labels']}/{label_stats['images_in_coco']}, "
              f"annotations per class {label_stats['annotations_per_class']}")
    for kind, n in sorted(summary["issues_by_kind"].items()):
        print(f"  {kind}: {n}")
    print(f"reports -> {out}")


if __name__ == "__main__":
    main()
