"""
DriveGuard AI - YOLO Training Script
========================================
Trains a YOLO model on the dataset described by training/data.yaml.
This is a thin, well-documented wrapper around Ultralytics' own
training loop - it does not reimplement training itself (no reason
to, Ultralytics' is battle-tested), it just wires it up to this
project's config and produces a `best.pt` in the place
detection/detector.py expects to find it.

IMPORTANT (spec section 27): this script does NOT print or claim any
accuracy/mAP numbers itself. Ultralytics logs its own per-epoch
metrics to the console and to `runs/detect/<name>/results.csv` during
training, and training/evaluate.py computes final precision/recall/
F1/mAP properly afterward - nothing here is invented.

Prerequisites (see datasets/README.md):
    1. A dataset organized under datasets/processed/ in YOLO format.
    2. training/data.yaml pointing at it with the right class names.
    3. `pip install -r requirements.txt` (needs ultralytics + torch).

Usage:
    python training/train.py
    python training/train.py --epochs 50 --imgsz 640 --model yolov8n.pt
    python training/train.py --resume    # continue an interrupted run

This will likely be SLOW or impractical on CPU-only laptops for many
epochs - start with a small --epochs value (e.g. 10-20) to confirm
the pipeline works end-to-end before committing to a long run, and
consider Google Colab's free GPU tier if local training is too slow.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

_DATA_YAML_PATH = os.path.join(os.path.dirname(__file__), "data.yaml")
_DEFAULT_BASE_MODEL = "yolov8n.pt"  # smallest/fastest YOLOv8 checkpoint - good starting point for a laptop


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DriveGuard AI YOLO model")
    parser.add_argument(
        "--model", default=_DEFAULT_BASE_MODEL,
        help=f"Base checkpoint to fine-tune from (default: {_DEFAULT_BASE_MODEL}, "
             f"auto-downloaded by Ultralytics on first use). Use a path to an "
             f"existing .pt file to resume from a specific checkpoint instead."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16; lower it if you run out of memory)")
    parser.add_argument("--device", default="cpu", help="'cpu' or a CUDA device index like '0' (default: cpu)")
    parser.add_argument("--name", default="driveguard_run", help="Name for this training run's output folder")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent interrupted run")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(_DATA_YAML_PATH):
        print(f"[ERROR] data.yaml not found at {_DATA_YAML_PATH}")
        sys.exit(1)

    dataset_root = resolve_path("datasets/processed")
    if not os.path.isdir(dataset_root) or not os.listdir(dataset_root):
        print(f"[ERROR] No dataset found at {dataset_root}")
        print("        Download and organize a dataset first - see datasets/README.md")
        print("        for real, currently-available dataset options and exact steps.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] The 'ultralytics' package is not installed.")
        print("        Run: pip install -r requirements.txt")
        sys.exit(1)

    print("=" * 60)
    print("DriveGuard AI - YOLO Training")
    print("=" * 60)
    print(f"Base model:   {args.model}")
    print(f"Dataset:      {_DATA_YAML_PATH}")
    print(f"Epochs:       {args.epochs}")
    print(f"Image size:   {args.imgsz}")
    print(f"Batch size:   {args.batch}")
    print(f"Device:       {args.device}")
    print(f"Run name:     {args.name}")
    print("=" * 60)

    logger.info(
        f"Starting YOLO training: model={args.model}, epochs={args.epochs}, "
        f"imgsz={args.imgsz}, batch={args.batch}, device={args.device}"
    )

    model = YOLO(args.model)

    try:
        results = model.train(
            data=_DATA_YAML_PATH,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=args.name,
            resume=args.resume,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"[ERROR] Training failed: {e}")
        sys.exit(1)

    # Ultralytics saves the best checkpoint under runs/detect/<name>/weights/best.pt.
    # Copy/point it to models/best.pt so detection/detector.py finds it without
    # any manual file-hunting, but never overwrite a good existing one silently.
    run_dir = getattr(results, "save_dir", None)
    if run_dir is not None:
        candidate_best = os.path.join(str(run_dir), "weights", "best.pt")
        if os.path.exists(candidate_best):
            models_dir = resolve_path("models")
            os.makedirs(models_dir, exist_ok=True)
            target = os.path.join(models_dir, "best.pt")
            print()
            print(f"Best checkpoint saved at: {candidate_best}")
            if os.path.exists(target):
                print(f"[NOTE] {target} already exists - NOT overwriting it automatically.")
                print(f"       Compare the two runs with training/evaluate.py before replacing it, e.g.:")
                print(f"       python training/evaluate.py --weights \"{candidate_best}\"")
            else:
                import shutil
                shutil.copy2(candidate_best, target)
                print(f"Copied to {target} (detection/detector.py will use this automatically)")

    logger.info("Training complete")
    print("=" * 60)
    print("Training complete. Run training/evaluate.py next for real")
    print("precision/recall/F1/mAP numbers on your validation set - do")
    print("not report accuracy without running it.")
    print("=" * 60)


if __name__ == "__main__":
    main()
