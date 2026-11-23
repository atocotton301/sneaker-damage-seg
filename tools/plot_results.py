"""Compare training runs in one figure.

Panel 1: validation mask mAP50 per epoch (from Ultralytics results.csv).
Panels 2-3: test precision and recall per class (from eval_report.py summary.txt).

Usage:
  python tools/plot_results.py --run base_640 runs/base_640/results.csv reports/base_640/summary.txt \
                               --run size_1024 runs/size_1024/results.csv reports/size_1024/summary.txt \
                               --out reports/compare.png
"""
import argparse
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical order (slot 1 blue, slot 2 orange, slot 3 aqua, slot 4 yellow).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def read_curve(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    key = lambda name: next(k for k in rows[0] if k.strip() == name)
    ep, m = key("epoch"), key("metrics/mAP50(M)")
    return [int(float(r[ep])) for r in rows], [float(r[m]) for r in rows]


def read_report(path):
    """class -> (precision, recall) from eval_report.py summary.txt."""
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*(\S+)\s+\d+\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, runs, classes, idx, title):
    style(ax, title)
    n = len(runs)
    width = 0.8 / n
    for i, (name, _, rep) in enumerate(runs):
        xs = [c + (i - (n - 1) / 2) * width for c in range(len(classes))]
        vals = [rep.get(c, (0, 0))[idx] for c in classes]
        ax.bar(xs, vals, width - 0.04, color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=1)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(classes)), classes)
    top = max(rep.get(c, (0, 0))[idx] for _, _, rep in runs for c in classes)
    ax.set_ylim(0, max(0.1, top * 1.25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=3, action="append", required=True,
                    metavar=("NAME", "RESULTS_CSV", "SUMMARY_TXT"))
    ap.add_argument("--out", default="reports/compare.png")
    args = ap.parse_args()

    runs = [(name, read_curve(csv_path), read_report(txt)) for name, csv_path, txt in args.run]
    classes = sorted({c for _, _, rep in runs for c in rep})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    style(ax, "Validation mask mAP50 by epoch")
    for i, (name, (ep, m), _) in enumerate(runs):
        ax.plot(ep, m, color=SERIES[i], linewidth=2, label=name)
        best = max(range(len(m)), key=m.__getitem__)
        ax.plot(ep[best], m[best], "o", color=SERIES[i], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{name} best {m[best]:.3f} (ep {ep[best]})", (ep[best], m[best]),
                    xytext=(-8, 10), textcoords="offset points", ha="right", fontsize=9, color=INK)
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)

    bars(axes[1], runs, classes, 0, "Test precision per class (IoU 0.5, conf 0.25)")
    bars(axes[2], runs, classes, 1, "Test recall per class (IoU 0.5, conf 0.25)")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
