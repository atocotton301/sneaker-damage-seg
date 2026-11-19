"""Shared helpers: class names, filename groups, image hashing."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DEFAULT_CLASSES = ["heel_lining_damage", "sole_separation"]


def _load_classes():
    """Class list: a classes.txt in the working folder (one name per line) overrides the default."""
    p = Path("classes.txt")
    if p.exists():
        names = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if names:
            return names
    return list(DEFAULT_CLASSES)


CLASSES = _load_classes()

# Known spellings that should map to a canonical class name.
ALIASES = {
    "heel_lining_damage": "heel_lining_damage",
    "heel_lining": "heel_lining_damage",
    "lining_damage": "heel_lining_damage",
    "뒤꿈치안감손상": "heel_lining_damage",
    "뒤꿈치_안감_손상": "heel_lining_damage",
    "sole_separation": "sole_separation",
    "sole_lift": "sole_separation",
    "밑창들뜸": "sole_separation",
    "밑창_들뜸": "sole_separation",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_GROUP_REGEX = r"^(S\d+)_"


def normalize_class(name):
    """Return (canonical_name or None, normalized_key)."""
    key = re.sub(r"[\s\-]+", "_", name.strip().lower())
    if key in CLASSES:
        return key, key
    canon = ALIASES.get(key)
    return (canon if canon in CLASSES else None), key


def group_of(file_name, pattern=DEFAULT_GROUP_REGEX):
    m = re.search(pattern, Path(file_name).name)
    return m.group(1) if m else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def open_upright(path):
    """Open an image with its EXIF orientation applied. Returns (img, exif_orientation)."""
    img = Image.open(path)
    orient = img.getexif().get(274, 1)
    return ImageOps.exif_transpose(img), orient


def dhash(img, size=16):
    """Difference hash with size*size bits, packed into uint64 words."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).view(">u8").astype(np.uint64)


def near_duplicate_pairs(hashes, max_distance=20, chunk=1024):
    """All index pairs (i<j) whose dHash Hamming distance (out of 256 bits) <= max_distance."""
    h = np.stack(hashes).astype(np.uint64)  # (n, words)
    n = len(h)
    pairs = []
    for start in range(0, n, chunk):
        block = h[start:start + chunk]
        dist = np.bitwise_count(block[:, None, :] ^ h[None, :, :]).sum(axis=2)
        ii, jj = np.nonzero(dist <= max_distance)
        for i, j in zip(ii + start, jj):
            if i < j:
                pairs.append((int(i), int(j), int(dist[i - start, j])))
    return pairs


def load_coco(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
