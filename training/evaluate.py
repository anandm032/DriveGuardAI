"""
DriveGuard AI - YOLO Evaluation Script
==========================================
Computes REAL precision, recall, F1, and mAP on the validation (or
test) split using Ultralytics' own validation pipeline, and measures
REAL inference speed (FPS) on this machine. Nothing here is invented
or estimated (spec section 27) - every number printed comes directly
from actually running the model against held-out data.

Usage:
    python training/evaluate.py                          # uses models/best.pt against the val split
    python training/evaluate.py --weights models/best.pt --split test
    python training/evaluate.py --weights runs/detect/driveguard_run/weights/best.pt
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

_DATA_YAML_PATH = os.path.join(os.path.dirname(__file__), "data.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the DriveGuard AI YOLO model")
    parser.add_argument(
        "--weights", default=None,
        help="Path to a trained .pt file (default: models/best.pt from config.yaml)"
    )
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate on (default: val)")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size (default: 640)")
    parser.add_argument("--device", default="cpu", help="'cpu' or a CUDA device index like '0' (default: cpu)")
    parser.add_argument("--fps-samples", type=int, default=30, help="Number of inference passes to time for the FPS measurement (default: 30)")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = args.weights or resolve_path("models/best.pt")
    if not os.path.exists(weights_path):
        print(f"[ERROR] No trained model found at: {weights_path}")
        print("        Train one first: python training/train.py")
        sys.exit(1)

    if not os.path.exists(_DATA_YAML_PATH):
        print(f"[ERROR] data.yaml not found at {_DATA_YAML_PATH}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] The 'ultralytics' package is not installed.")
        print("        Run: pip install -r requirements.txt")
        sys.exit(1)

    print("=" * 60)
    print("DriveGuard AI - Model Evaluation")
    print("=" * 60)
    print(f"Weights: {weights_path}")
    print(f"Split:   {args.split}")
    print(f"Device:  {args.device}")
    print("=" * 60)

    logger.info(f"Evaluating {weights_path} on {args.split} split")
    model = YOLO(weights_path)

    # --- Real precision / recall / mAP via Ultralytics' validator ---
    try:
        metrics = model.val(
            data=_DATA_YAML_PATH,
            split=args.split,
            imgsz=args.imgsz,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"[ERROR] Validation failed: {e}")
        print("        Common cause: the dataset has no images for this split, or")
        print("        data.yaml's paths don't match your actual folder layout.")
        sys.exit(1)

    # Ultralytics' DetMetrics exposes precision/recall per class and
    # mAP averaged - .box.mp / .box.mr are mean precision/recall across
    # all classes, matching what most papers report as "P" and "R".
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print()
    print("-" * 60)
    print("REAL metrics (computed just now, not fabricated):")
    print("-" * 60)
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall:           {recall:.4f}")
    print(f"  F1 score:         {f1:.4f}  (derived from precision & recall above)")
    print(f"  mAP@0.5:          {map50:.4f}")
    print(f"  mAP@0.5:0.95:     {map50_95:.4f}")

    # Per-class breakdown, since an aggregate number can hide a class
    # that's performing badly (e.g. if you have few training images
    # for "drinking" compared to "smoking").
    try:
        class_names = model.names
        per_class_map50 = metrics.box.ap50  # array indexed by class
        print()
        print("  Per-class mAP@0.5:")
        for idx, ap in enumerate(per_class_map50):
            name = class_names.get(idx, str(idx)) if isinstance(class_names, dict) else class_names[idx]
            print(f"    {name:25s} {float(ap):.4f}")
    except Exception as e:
        logger.warning(f"Could not compute per-class breakdown: {e}")

    # --- Real FPS measurement on THIS machine (not a claimed/generic number) ---
    print()
    print("-" * 60)
    print(f"Measuring real inference speed on this machine ({args.fps_samples} passes)...")
    print("-" * 60)

    import numpy as np
    dummy_frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)

    # Warm-up pass (first inference includes model/graph setup overhead
    # that wouldn't be representative of steady-state speed)
    model.predict(source=dummy_frame, imgsz=args.imgsz, device=args.device, verbose=False)

    start = time.perf_counter()
    for _ in range(args.fps_samples):
        model.predict(source=dummy_frame, imgsz=args.imgsz, device=args.device, verbose=False)
    elapsed = time.perf_counter() - start

    avg_inference_ms = (elapsed / args.fps_samples) * 1000
    fps = args.fps_samples / elapsed

    print(f"  Average inference time: {avg_inference_ms:.1f} ms/frame")
    print(f"  Approximate FPS:        {fps:.1f}")
    print(f"  (measured on a blank {args.imgsz}x{args.imgsz} frame on device='{args.device}' - ")
    print(f"   real driver footage with more detected objects may be slightly slower)")

    logger.info(
        f"Evaluation complete: P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
        f"mAP50={map50:.4f} mAP50-95={map50_95:.4f} FPS={fps:.1f}"
    )

    print("=" * 60)
    print("Evaluation complete. These are the only accuracy/speed numbers")
    print("this project should ever report - regenerate them any time the")
    print("model, dataset, or hardware changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
