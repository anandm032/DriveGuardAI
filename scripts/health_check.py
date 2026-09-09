"""
DriveGuard AI - Environment Health Check
==========================================
Run this after setting up the virtual environment and installing
requirements.txt to confirm every core dependency is importable and
working before we build any real functionality on top of it.

Usage:
    python scripts/health_check.py

This script does NOT load a YOLO model file yet (we don't have one
until Phase 6) — it only confirms the libraries themselves are
installed correctly and that OpenCV can talk to a webcam if one is
attached.
"""

import os
import sys
import importlib

# Allow running as `python scripts/health_check.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_python_version():
    print("-" * 60)
    print("Checking Python version...")
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"  Detected: Python {major}.{minor}.{sys.version_info.micro}")
    if (major, minor) < (3, 10):
        print("  [WARN] Python 3.10+ is recommended for this project.")
    else:
        print("  [OK] Python version is suitable.")


def check_import(module_name, friendly_name=None, extra_check=None):
    friendly_name = friendly_name or module_name
    print("-" * 60)
    print(f"Checking {friendly_name}...")
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown version")
        print(f"  [OK] {friendly_name} imported successfully (version: {version})")
        if extra_check:
            extra_check(module)
        return True
    except ImportError as e:
        print(f"  [FAIL] Could not import {friendly_name}: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] {friendly_name} imported but raised an error: {e}")
        return False


def check_opencv_webcam(cv2_module):
    """Try to open the default webcam briefly. Non-fatal if it fails —
    some machines (servers, CI, some laptops) legitimately have no camera
    or block camera access until the app runs interactively."""
    print("  Checking webcam access (index 0)...")
    try:
        cap = cv2_module.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  [OK] Webcam opened and returned a frame of shape {frame.shape}")
            else:
                print("  [WARN] Webcam opened but did not return a frame. "
                      "Check that no other app is using the camera.")
            cap.release()
        else:
            print("  [WARN] Could not open webcam at index 0. "
                  "This is fine for now if you are testing on a machine without a camera; "
                  "the real-time detection module (Phase 7) will need a working webcam.")
    except Exception as e:
        print(f"  [WARN] Webcam check raised an exception: {e}")


def check_torch(torch_module):
    cuda_available = torch_module.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    if not cuda_available:
        print("  [INFO] Running on CPU is expected and fine for this project. "
              "Set model.device to 'cpu' in config/config.yaml.")


def check_project_config():
    """Confirm our own utils/helpers.py can load config.yaml, and that
    the logger sets itself up without error. This checks Phase 2's
    work specifically, not just third-party libraries."""
    print("-" * 60)
    print("Checking project config + logging setup...")
    try:
        from utils.helpers import ConfigError, load_config
        from utils.logger import get_logger

        config = load_config()
        logger = get_logger("health_check")
        logger.info("Health check reached the logger successfully.")
        print(f"  [OK] config.yaml loaded ({len(config)} top-level sections)")
        print("  [OK] Logger initialized (check logs/driveguard.log)")
        return True
    except ConfigError as e:
        print(f"  [FAIL] Config error: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Unexpected error loading project config/logger: {e}")
        return False


def main():
    print("=" * 60)
    print("DriveGuard AI - Environment Health Check")
    print("=" * 60)

    check_python_version()

    results = {}
    results["opencv"] = check_import("cv2", "OpenCV", extra_check=check_opencv_webcam)
    results["torch"] = check_import("torch", "PyTorch", extra_check=check_torch)
    results["torchvision"] = check_import("torchvision", "TorchVision")
    results["ultralytics"] = check_import("ultralytics", "Ultralytics YOLO")
    results["streamlit"] = check_import("streamlit", "Streamlit")
    results["numpy"] = check_import("numpy", "NumPy")
    results["pandas"] = check_import("pandas", "Pandas")
    results["yaml"] = check_import("yaml", "PyYAML")
    results["project_config"] = check_project_config()

    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:15s}: {status}")
        if not ok:
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("All core dependencies are installed correctly.")
        print("You are ready to move on to Phase 2.")
    else:
        print("One or more dependencies failed to import.")
        print("Fix the [FAIL] items above (usually: pip install -r requirements.txt "
              "inside the correct virtual environment) before continuing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
