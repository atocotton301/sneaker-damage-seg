"""Step 3: group-aware train/val/test split of a COCO file.

All photos of one shoe (same group id from the file name) land in one split.
Near-duplicate pairs from audit.py are merged into one group too, so the same
shoe photographed under two ids cannot leak across splits.

Usage:
  python tools/split.py --coco data/annotations.json \
      --near-dups reports/audit/near_duplicates.csv --out data/splits
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import CLASSES, DEFAULT_GROUP_REGEX, group_of, load_coco, normalize_class, save_json


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--near-dups", help="near_duplicates.csv from audit.py")
    ap.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX)
    ap.add_argument("--max-hamming", type=int, default=20,
                    help="only merge near-duplicate pairs at or below this distance; "
                         "check pairs by eye first (10 suited the Shoe Defects set)")
    ap.add_argument("--ratios", default="0.7,0.15,0.15")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = load_coco(args.coco)
    ratios = dict(zip(["train", "val", "test"], map(float, args.ratios.split(","))))

    # Canonical categories, in CLASSES order.
    old_to_canon = {}
    for c in coco["categories"]:
        canon, _ = normalize_class(c["name"])
        if canon is None:
            raise SystemExit(f"unknown class '{c['name']}'. Fix it or add an alias in common.py")
        old_to_canon[c["id"]] = canon
    categories = [{"id": i, "name": n, "supercategory": "damage"} for i, n in enumerate(CLASSES)]
    canon_id = {n: i for i, n in enumerate(CLASSES)}

    # Groups: file-name id, then merge near duplicates.
    uf = UnionFind()
    img_group = {}
    for im in coco["images"]:
        # A "group" field in the COCO image (set by import_roboflow.py) wins over the file name.
        g = im.get("group") or group_of(im["file_name"], args.group_regex) or f"nogroup:{im['file_name']}"
        img_group[im["id"]] = g
        uf.find(g)
    merged = 0
    if args.near_dups:
        name_to_group = {Path(im["file_name"]).as_posix(): img_group[im["id"]] for im in coco["images"]}
        with open(args.near_dups, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if int(row["hamming"]) > args.max_hamming:
                    continue
                ga, gb = name_to_group.get(row["file_a"]), name_to_group.get(row["file_b"])
                if ga and gb and uf.find(ga) != uf.find(gb):
                    uf.union(ga, gb)
                    merged += 1

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # Per-cluster label counts.
    cluster_imgs = defaultdict(list)
    for im in coco["images"]:
        cluster_imgs[uf.find(img_group[im["id"]])].append(im)
    cluster_counts = {}
    for cl, ims in cluster_imgs.items():
        cnt = Counter({"images": len(ims)})
        for im in ims:
            for a in anns_by_img[im["id"]]:
                cnt[old_to_canon[a["category_id"]]] += 1
        cluster_counts[cl] = cnt

    # Greedy: biggest clusters first, each to the split furthest below its target share.
    keys = ["images"] + CLASSES
    totals = Counter()
    for c in cluster_counts.values():
        totals.update(c)
    clusters = list(cluster_counts)
    random.Random(args.seed).shuffle(clusters)
    clusters.sort(key=lambda c: -cluster_counts[c]["images"])
    have = {s: Counter() for s in ratios}
    assign = {}
    for cl in clusters:
        def deficit(s):
            return sum(ratios[s] * totals[k] - have[s][k] for k in keys if totals[k])
        s = max(ratios, key=deficit)
        assign[cl] = s
        have[s].update(cluster_counts[cl])

    # Write split files with canonical category ids.
    out = Path(args.out)
    report = []
    for s in ratios:
        ims = [im for im in coco["images"] if assign[uf.find(img_group[im["id"]])] == s]
        ids = {im["id"] for im in ims}
        anns = []
        for a in coco["annotations"]:
            if a["image_id"] in ids:
                a = dict(a, category_id=canon_id[old_to_canon[a["category_id"]]])
                anns.append(a)
        save_json({"images": ims, "annotations": anns, "categories": categories}, out / f"{s}.json")
        report.append({"split": s, "images": len(ims),
                       **{c: have[s][c] for c in CLASSES},
                       "groups": sum(1 for cl in assign if assign[cl] == s)})

    # Leakage check: every original group must sit in exactly one split.
    seen = defaultdict(set)
    for im in coco["images"]:
        seen[img_group[im["id"]]].add(assign[uf.find(img_group[im["id"]])])
    leaks = [g for g, ss in seen.items() if len(ss) > 1]
    assert not leaks, f"group crosses splits: {leaks}"

    with open(out / "groups.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "cluster", "split"])
        for g in sorted(seen):
            w.writerow([g, uf.find(g), next(iter(seen[g]))])

    print(f"near-duplicate merges: {merged}")
    print(f"{'split':6} {'images':>6} " + " ".join(f"{c:>20}" for c in CLASSES) + f" {'groups':>6}")
    for r in report:
        print(f"{r['split']:6} {r['images']:>6} " + " ".join(f"{r[c]:>20}" for c in CLASSES) + f" {r['groups']:>6}")
    print("no group crosses splits")


if __name__ == "__main__":
    main()
