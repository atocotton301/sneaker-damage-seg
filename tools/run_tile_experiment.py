"""Experiment 3 in one command: tile, train, predict on whole test images, evaluate.

Run from the dataset root (the folder holding yolo/ and tools/):
  python tools/run_tile_experiment.py --runs /content/drive/MyDrive/sneaker-damage-seg/runs

Steps (each skipped if its output already exists, so a broken session can resume):
  1. tile_yolo.py       yolo/ -> yolo_tiles/ (train, val)
  2. yolo train         yolov8n-seg, imgsz = tile size, 50 epochs, seed 0
  3. tiled_predict.py   whole test images, stitched predictions
  4. eval_report.py     per-class TP/FP/FN at IoU 0.5 and 0.3
  5. test_eval.py       confidence sweep and false-positive gallery for all runs
"""
import argparse
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).parent


def run(cmd):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolov8n-seg.pt")
    args = ap.parse_args()

    runs = Path(args.runs)
    name = f"tile_{args.tile}"
    py = sys.executable

    if not Path("yolo_tiles/data.yaml").exists():
        run([py, TOOLS / "tile_yolo.py", "--src", "yolo", "--out", "yolo_tiles",
             "--tile", args.tile, "--overlap", args.overlap])

    weights = runs / name / "weights" / "best.pt"
    if not weights.exists():
        last = runs / name / "weights" / "last.pt"
        if last.exists():  # resume an interrupted run
            run(["yolo", "segment", "train", f"resume={last}"])
        else:
            run(["yolo", "segment", "train", f"model={args.model}", "data=yolo_tiles/data.yaml",
                 f"imgsz={args.tile}", f"epochs={args.epochs}", f"batch={args.batch}", "seed=0",
                 "deterministic=True", f"project={runs}", f"name={name}", "exist_ok=True"])

    pred = runs / f"{name}_pred" / "labels"
    run([py, TOOLS / "tiled_predict.py", "--weights", weights, "--images", "yolo/images/test",
         "--out", pred, "--tile", args.tile, "--overlap", args.overlap])
    for iou, suffix in ((0.5, ""), (0.3, "_iou03")):
        run([py, TOOLS / "eval_report.py", "--labels", "yolo/labels/test", "--preds", pred,
             "--out", runs / f"{name}_report{suffix}", "--iou", iou])
    run([py, TOOLS / "test_eval.py", "--runs", runs, "--skip-ultralytics"])


if __name__ == "__main__":
    main()
