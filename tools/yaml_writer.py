"""Tiny data.yaml writer so the tools need no PyYAML."""


def write_data_yaml(path, root, names):
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(names)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
