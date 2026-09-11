"""
DriveGuard AI - Violation Confirmation (Temporal Filtering)
================================================================
Turns a raw, per-frame stream of Detection objects (from
detection/detector.py) into CONFIRMED violations, using the same two
mechanisms the original project spec (section 7) and config.yaml's
`detection` section already define:

    min_consecutive_frames (10)     - a behavior must be detected in
                                       this many CONSECUTIVE cycles
                                       before it's confirmed at all.
    violation_cooldown_seconds (15) - once confirmed, the SAME
                                       activity won't be confirmed
                                       again until this many real
                                       seconds have passed, even if it
                                       never stopped being detected.

This is deliberately a pure state machine: no camera, no YOLO, no
database, no scoring engine. It only knows about Detection objects in
and Detection objects out. That split exists so this exact file can
be unit-tested completely (see scripts/test_violation_confirmation.py)
without a webcam, ultralytics, a trained model, or a database - and so
Phase 7.2/7.3 (video capture, orchestration) can be built and swapped
independently without touching this logic.

IMPORTANT: this module does NOT decide whether an activity is
"scoreable" (e.g. it will happily confirm "safe_driving" or
"drowsiness" after 10 consecutive frames, exactly like "smoking") and
it does NOT talk to scoring/safety_score.py or database/database.py
at all. Those decisions belong to SafetyScoreEngine - see its
NON_PENALIZED_ACTIVITIES set and `penalties` config for how
"drowsiness"/"safe_driving"/"eating" actually get excluded from
scoring. Keeping that logic out of this file is what keeps it
independently testable.

Usage:

    from detection.violation_confirmation import ViolationTracker

    tracker = ViolationTracker()  # reads thresholds from config.yaml
    for frame in video_stream:
        detections = detector.detect(frame)
        confirmed = tracker.update(detections)
        for violation in confirmed:
            engine.record_violation(vehicle_id, violation.activity, violation.confidence)
"""

import time
from typing import Callable, Dict, List, Optional

from detection.detector import Detection
from utils.helpers import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


class ViolationTracker:
    def __init__(
        self,
        min_consecutive_frames: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
        config: Optional[dict] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """clock is injectable so tests can simulate elapsed time
        without real sleep() calls - defaults to time.monotonic
        (never affected by system clock adjustments, unlike
        time.time()), which is what the real webcam loop should use."""
        cfg = config if config is not None else load_config()
        detection_cfg = cfg.get("detection", {})

        self.min_consecutive_frames = (
            min_consecutive_frames
            if min_consecutive_frames is not None
            else detection_cfg.get("min_consecutive_frames", 10)
        )
        self.cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else detection_cfg.get("violation_cooldown_seconds", 15)
        )
        self._clock = clock

        # activity -> current consecutive-frame streak count
        self._streaks: Dict[str, int] = {}
        # activity -> clock() value when it was last CONFIRMED (not
        # just detected) - drives the cooldown gate, independent of
        # streak resets (see reset-vs-cooldown note in update()).
        self._last_confirmed_at: Dict[str, float] = {}

        logger.info(
            f"ViolationTracker initialized: min_consecutive_frames="
            f"{self.min_consecutive_frames}, cooldown_seconds={self.cooldown_seconds}"
        )

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Feed one detection cycle's worth of raw Detection objects
        (typically: whatever Detector.detect(frame) returned for the
        current frame) and get back the subset that just became newly
        CONFIRMED this cycle. Usually an empty list; occasionally one
        Detection; rarely more than one if several activities cross
        the threshold in the same cycle.

        Call this once per processed frame, in frame order - it has
        no notion of "catching up" on skipped frames, so if the
        caller is skipping frames (config's detection.frame_skip),
        that's already reflected simply by calling update() less
        often, which naturally takes more wall-clock time to reach
        min_consecutive_frames - no special-casing needed here."""
        now = self._clock()

        # Multiple boxes of the SAME activity within a single frame
        # (e.g. two overlapping "smoking" detections) count as ONE
        # streak increment, not two - keep the highest-confidence
        # instance to represent the activity if/when it's confirmed.
        best_this_cycle: Dict[str, Detection] = {}
        for det in detections:
            existing = best_this_cycle.get(det.activity)
            if existing is None or det.confidence > existing.confidence:
                best_this_cycle[det.activity] = det

        activities_present = set(best_this_cycle.keys())
        # Every activity currently mid-streak, plus anything newly
        # seen this cycle, needs its streak evaluated this round.
        activities_to_evaluate = set(self._streaks.keys()) | activities_present

        confirmed: List[Detection] = []

        for activity in activities_to_evaluate:
            if activity not in activities_present:
                # Missing this cycle -> streak breaks completely, per
                # spec ("a missing activity should reset its
                # consecutive streak"). We do NOT touch
                # _last_confirmed_at here - the cooldown clock keeps
                # running in real time regardless of streak breaks,
                # so briefly looking away and back doesn't bypass it.
                self._streaks.pop(activity, None)
                continue

            self._streaks[activity] = self._streaks.get(activity, 0) + 1
            streak = self._streaks[activity]

            if streak < self.min_consecutive_frames:
                continue  # not enough consecutive frames yet

            last_confirmed = self._last_confirmed_at.get(activity)
            if last_confirmed is not None and (now - last_confirmed) < self.cooldown_seconds:
                continue  # streak qualifies, but still cooling down from the last confirmation

            self._last_confirmed_at[activity] = now
            confirmed.append(best_this_cycle[activity])
            logger.info(
                f"Violation confirmed: {activity} (streak={streak} frames, "
                f"confidence={best_this_cycle[activity].confidence:.2f})"
            )

        return confirmed

    def reset(self):
        """Clear all streaks and cooldowns - e.g. when starting a
        fresh monitoring session for a different vehicle."""
        self._streaks.clear()
        self._last_confirmed_at.clear()

    def get_streak(self, activity: str) -> int:
        """Current consecutive-frame count for an activity (0 if it
        isn't currently streaking). Read-only - mainly for tests and
        any future on-screen "9/10 frames..." debug display."""
        return self._streaks.get(activity, 0)
