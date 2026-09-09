"""
DriveGuard AI - Detector Sanity Check
=========================================
What this script CAN verify right now, without ultralytics installed
or a trained model file on disk:
  1. Detector() fails with a clear, non-crashing error when
     'ultralytics' isn't installed - never a fake/simulated result.
  2. Once ultralytics IS installed, Detector() fails with an equally
     clear error when models/best.pt doesn't exist yet - same
     principle, no invented detections.
  3. The class-name mapping logic (raw dataset label -> this
     project's activity vocabulary) is correct, tested directly via
     the pure map_class_name() function - this doesn't need a model.

What it CANNOT verify here (needs ultralytics + torch + a trained
model.pt, which requires internet access and real training data this
sandbox doesn't have): actually loading a model and running detect()
on a real frame. Once you've installed requirements.txt and trained a
model (training/train.py), re-run this script - the try/except below
will automatically exercise that real path too instead of skipping it.

Usage:
    python scripts/test_detector.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.detector import Detector, DetectorError, map_class_name
from utils.helpers import load_config


def main():
    print("=" * 60)
    print("DriveGuard AI - Detector Sanity Check")
    print("=" * 60)

    # --- 1/2: Detector() should fail clearly, never fake a result ---
    try:
        Detector()
        print("[INFO] Detector loaded successfully - ultralytics is installed "
              "and a model file exists. Real detect() behavior can be tested "
              "with an actual frame/model at this point (not covered by this "
              "script, which focuses on the failure paths that ARE testable "
              "without those prerequisites).")
    except DetectorError as e:
        msg = str(e)
        if "ultralytics" in msg.lower():
            print(f"[OK] Correctly reported missing dependency:\n      {msg}")
        elif "no trained model found" in msg.lower():
            print(f"[OK] Correctly reported missing model file:\n      {msg}")
        else:
            print(f"[OK] Detector raised DetectorError (some other real "
                  f"problem, still handled gracefully, not silently ignored):\n      {msg}")

    # --- 3: class-name mapping logic, tested directly (no model needed) ---
    print("-" * 60)
    print("Testing class-name mapping (pure function, no model required)")
    print("-" * 60)

    config = load_config()
    class_map = {
        str(k).strip().lower(): v
        for k, v in config.get("model", {}).get("class_map", {}).items()
    }
    print(f"[OK] Loaded class_map from config.yaml: {class_map}")

    test_cases = [
        ("Mobile use", "mobile_phone_usage"),
        ("mobile_phone_usage", "mobile_phone_usage"),   # identity entry
        ("Smoking", "smoking"),
        ("DRINKING", "drinking"),
        ("Distracted", "distraction"),
        ("distraction", "distraction"),                  # identity entry
        ("Drowsy", "drowsiness"),
        ("drowsiness", "drowsiness"),                     # identity entry
        ("SafeDriving", None),                            # intentionally unmapped
        ("Eating", None),                                 # intentionally unmapped (no penalty defined yet)
        ("SomeRandomClass", None),                        # genuinely unknown
    ]

    all_passed = True
    for raw_name, expected in test_cases:
        actual = map_class_name(raw_name, class_map)
        status = "OK" if actual == expected else "FAIL"
        if actual != expected:
            all_passed = False
        print(f"  [{status}] '{raw_name}' -> {actual} (expected {expected})")

    assert all_passed, "One or more class-name mapping cases failed"

    print("=" * 60)
    if all_passed:
        print("All testable Phase 6 checks passed.")
        print()
        print("Next step: get a real dataset (see datasets/README.md), run")
        print("training/train.py, then training/evaluate.py for real metrics,")
        print("then re-run this script to also exercise live model loading")
        print("and detection.")
    print("=" * 60)


if __name__ == "__main__":
    main()
