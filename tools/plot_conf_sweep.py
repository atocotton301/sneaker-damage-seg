"""Precision vs recall at each confidence cut, one line per run, from test_eval.py's conf_sweep.json.

Usage:
  python tools/plot_conf_sweep.py --sweep runs/conf_sweep.json --out docs/images/conf_sweep.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sweep = json.load(open(args.sweep, encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    for (name, rows), color in zip(sweep.items(), COLORS):
        rows = sorted(rows, key=lambda r: r["conf"])
        xs, ys = [r["recall"] for r in rows], [r["precision"] for r in rows]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=7, mec="white", mew=1.5, label=name)
        for r in rows:
            ax.annotate(f"{r['conf']:.2f}", (r["recall"], r["precision"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color="#555")
    ax.set_xlabel("Recall (842 labeled damages in test)")
    ax.set_ylabel("Precision")
    ax.set_title("Test precision vs recall by confidence cut (IoU 0.5)", loc="left", fontsize=11)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 0.8)
    ax.grid(color="#e5e5e5", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbb")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
