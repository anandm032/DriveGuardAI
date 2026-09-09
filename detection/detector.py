"""
DriveGuard AI - YOLO Detector
================================
Loads a trained YOLO model (models/best.pt, configurable) and runs
inference on a single frame, returning a list of Detection objects in
OUR project's activity vocabulary (mobile_phone_usage, smoking,
drinking, distraction, drowsiness) - not whatever raw class names the
training dataset happened to use.

This module does NOT open a webcam or loop over video - that's
detection/webcam.py (Phase 7). This is just: frame in, detections out.

IMPORTANT - matches spec section 4/27: this module never invents or
hard-codes fake predictions. If the model file doesn't exist yet (you
haven't trained one - see datasets/README.md and training/train.py),
every method here raises a clear DetectorError instead of pretending
to detect something. If `ultralytics` isn't installed, same thing -
loud and clear, not a silent no-op.

Class name mapping
-------------------
Whatever public/custom dataset you train on will have its own class
names (e.g. "Mobile use", "Texting", "D1"). config.yaml's
`model.class_map` translates those raw names to this project's
internal activity vocabulary, so the scoring engine (Phase 4/5) never
needs to know or care what the underlying dataset called things.
Unmapped raw classes are dropped (logged once) rather than passed
through unrecognized - see datasets/README.md for how to fill this in
once you've picked a dataset.

Usage:

    from detection.detector import Detector

    detector = Detector()  # reads model.path etc. from config.yaml
    detections = detector.detect(frame)  # frame = a BGR numpy array from OpenCV
    for d in detections:
        print(d.activity, d.confidence, d.bbox)
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from utils.helpers import load_config, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


class DetectorError(Exception):
    """Raised when the model can't be loaded or inference can't run -
    missing dependency, missing weights file, or a malformed frame.
    Callers (the webcam loop in Phase 7, this module's own sanity
    check script) should catch this and fail gracefully rather than
    crash the whole app (spec section 19)."""
    pass


@dataclass
class Detection:
    activity: str                       # our internal vocabulary, e.g. "mobile_phone_usage"
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coordinates
    raw_class_name: str                 # whatever the model itself called it, for debugging


class Detector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        inference_size: Optional[int] = None,
        device: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        if not _ULTRALYTICS_AVAILABLE:
            raise DetectorError(
                "The 'ultralytics' package is not installed. Run "
                "'pip install -r requirements.txt' in your activated virtual "
                "environment, then try again."
            )

        self.config = config if config is not None else load_config()
        model_config = self.config.get("model", {})

        self.model_path = resolve_path(model_path or model_config.get("path", "models/best.pt"))
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.get("detection", {}).get("confidence_threshold", 0.70)
        )
        self.inference_size = inference_size or model_config.get("inference_size", 640)
        self.device = device or model_config.get("device", "cpu")
        self.class_map = {
            str(k).strip().lower(): v
            for k, v in model_config.get("class_map", {}).items()
        }

        if not os.path.exists(self.model_path):
            raise DetectorError(
                f"No trained model found at: {self.model_path}\n"
                f"You need to train a model first (see datasets/README.md and "
                f"training/train.py) or point config.yaml's model.path at an "
                f"existing weights file. This project never fabricates "
                f"detections when no real model is available."
            )

        try:
            self._model = YOLO(self.model_path)
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {self.model_path}: {e}")
            raise DetectorError(f"Failed to load YOLO model from {self.model_path}: {e}")

        self._warned_unmapped_classes = set()
        logger.info(
            f"Detector loaded: model={self.model_path}, device={self.device}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"classes={list(self._model.names.values())}"
        )

    def detect(self, frame) -> List[Detection]:
        """Run inference on a single BGR frame (as returned by
        cv2.VideoCapture.read()) and return confirmed-vocabulary
        Detection objects above the configured confidence threshold.
        Raw model classes with no entry in config.yaml's
        model.class_map are dropped (and logged once) rather than
        silently passed through under an unrecognized name."""
        if frame is None:
            raise DetectorError("detect() received a None frame (camera read likely failed)")

        try:
            results = self._model.predict(
                source=frame,
                imgsz=self.inference_size,
                device=self.device,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            raise DetectorError(f"YOLO inference failed: {e}")

        return self._parse_results(results)

    def _parse_results(self, results) -> List[Detection]:
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        names = result.names  # {class_id: raw_name}
        for box in result.boxes:
            class_id = int(box.cls[0])
            raw_name = names.get(class_id, str(class_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            activity = self._map_class_name(raw_name)
            if activity is None:
                continue  # unmapped class - not part of our activity vocabulary

            detections.append(
                Detection(
                    activity=activity,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    raw_class_name=raw_name,
                )
            )
        return detections

    def _map_class_name(self, raw_name: str) -> Optional[str]:
        activity = map_class_name(raw_name, self.class_map)
        if activity is None:
            key = str(raw_name).strip().lower()
            if key not in self._warned_unmapped_classes:
                logger.warning(
                    f"Detected raw class '{raw_name}' has no entry in "
                    f"config.yaml's model.class_map - ignoring it. Add "
                    f"'{key}: <activity_name>' under model.class_map to include it."
                )
                self._warned_unmapped_classes.add(key)
        return activity


# ------------------------------------------------------------
# Pure function (no model/config dependency) - lets Phase 13's unit
# tests, and scripts/test_detector.py in the meantime, exercise the
# class-name translation logic directly without needing ultralytics
# installed or a real trained model file on disk.
# ------------------------------------------------------------
def map_class_name(raw_name: str, class_map: dict) -> Optional[str]:
    key = str(raw_name).strip().lower()
    return class_map.get(key)
