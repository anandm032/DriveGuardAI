"""
DriveGuard AI - Violation Confirmation Tracker Tests
========================================================
Covers all 10 required scenarios for Phase 7.1. Pure logic tests -
no camera, no ultralytics, no database, no scoring engine involved.
Uses a fake, manually-advanced clock so cooldown tests run instantly
instead of doing real time.sleep() calls.

Usage:
    python scripts/test_violation_confirmation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.detector import Detection
from detection.violation_confirmation import ViolationTracker
from utils.helpers import load_config


class FakeClock:
    """Manually-advanced stand-in for time.monotonic - lets cooldown
    tests simulate 16 elapsed seconds instantly instead of sleeping."""
    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float):
        self.t += seconds


def make_detection(activity: str, confidence: float = 0.90) -> Detection:
    return Detection(activity=activity, confidence=confidence, bbox=(0.0, 0.0, 10.0, 10.0),
                      raw_class_name=activity)


def new_tracker(min_frames: int = 10, cooldown: float = 15.0, clock=None) -> ViolationTracker:
    return ViolationTracker(
        min_consecutive_frames=min_frames,
        cooldown_seconds=cooldown,
        clock=clock or FakeClock(),
    )


def check(condition: bool, description: str):
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {description}")
    assert condition, f"FAILED: {description}"


# ------------------------------------------------------------
# 1. Fewer than 10 consecutive detections -> no violation
# ------------------------------------------------------------
def test_fewer_than_threshold_no_violation():
    print("Test 1: fewer than 10 consecutive detections -> no violation")
    tracker = new_tracker()
    confirmed = []
    for i in range(9):
        confirmed = tracker.update([make_detection("smoking")])
        check(confirmed == [], f"frame {i + 1}/9: no confirmation yet")
    check(tracker.get_streak("smoking") == 9, "streak sits at 9, one short of the threshold")


# ------------------------------------------------------------
# 2. Exactly 10 consecutive detections -> one violation confirmation
# ------------------------------------------------------------
def test_exactly_threshold_confirms_once():
    print("Test 2: exactly 10 consecutive detections -> one confirmation")
    tracker = new_tracker()
    confirmed = []
    for i in range(10):
        confirmed = tracker.update([make_detection("smoking", 0.85)])
    check(len(confirmed) == 1, "exactly one Detection confirmed on the 10th frame")
    check(confirmed[0].activity == "smoking", "confirmed activity is 'smoking'")
    check(confirmed[0].confidence == 0.85, "confirmed Detection carries the real confidence value")


# ------------------------------------------------------------
# 3. 9 detections + missing frame -> streak resets
# ------------------------------------------------------------
def test_missing_frame_resets_streak():
    print("Test 3: 9 detections then a missing frame -> streak resets to 0")
    tracker = new_tracker()
    for _ in range(9):
        tracker.update([make_detection("smoking")])
    check(tracker.get_streak("smoking") == 9, "streak at 9 before the gap")

    confirmed = tracker.update([])  # activity absent this cycle
    check(confirmed == [], "no confirmation on the empty frame")
    check(tracker.get_streak("smoking") == 0, "streak reset to 0 after a missing frame")

    # Confirm the reset is real, not cosmetic: it takes a full new
    # streak of 10, not just 1 more frame, to confirm again.
    confirmed = tracker.update([make_detection("smoking")])
    check(confirmed == [], "streak restarting at 1, not 10 - no confirmation yet")
    check(tracker.get_streak("smoking") == 1, "streak correctly restarted from 1")


# ------------------------------------------------------------
# 4. Different activity interrupts the streak
# ------------------------------------------------------------
def test_different_activity_interrupts_streak():
    print("Test 4: a different activity appearing doesn't extend the original streak")
    tracker = new_tracker()
    for _ in range(9):
        tracker.update([make_detection("smoking")])
    check(tracker.get_streak("smoking") == 9, "smoking streak at 9")

    confirmed = tracker.update([make_detection("drinking")])  # smoking absent, drinking present
    check(confirmed == [], "no confirmation for either activity yet")
    check(tracker.get_streak("smoking") == 0, "smoking streak reset (it was absent this cycle)")
    check(tracker.get_streak("drinking") == 1, "drinking streak starts fresh at 1")


# ------------------------------------------------------------
# 5. Cooldown prevents immediate duplicate confirmation
# ------------------------------------------------------------
def test_cooldown_blocks_immediate_reconfirmation():
    print("Test 5: cooldown prevents re-confirming the same activity immediately")
    clock = FakeClock()
    tracker = new_tracker(clock=clock)
    confirmed = []
    for _ in range(10):
        confirmed = tracker.update([make_detection("smoking")])
    check(len(confirmed) == 1, "first confirmation happens at frame 10")

    # Activity keeps being detected continuously (streak keeps growing),
    # but zero time has passed - cooldown must block re-confirmation.
    for i in range(5):
        confirmed = tracker.update([make_detection("smoking")])
        check(confirmed == [], f"frame {11 + i}: still in cooldown, no re-confirmation")
    check(tracker.get_streak("smoking") == 15, "streak keeps counting through the cooldown window")


# ------------------------------------------------------------
# 6. Same activity can be confirmed again after cooldown
# ------------------------------------------------------------
def test_reconfirms_after_cooldown_expires():
    print("Test 6: same activity confirms again once cooldown has elapsed")
    clock = FakeClock()
    tracker = new_tracker(cooldown=15.0, clock=clock)
    for _ in range(10):
        tracker.update([make_detection("smoking")])

    clock.advance(16.0)  # cooldown (15s) has now elapsed
    confirmed = tracker.update([make_detection("smoking", 0.77)])
    check(len(confirmed) == 1, "re-confirmed after cooldown expired")
    check(confirmed[0].activity == "smoking", "still reports the correct activity")


# ------------------------------------------------------------
# 7. Different activities can be confirmed independently
# ------------------------------------------------------------
def test_independent_activities_confirm_independently():
    print("Test 7: independent streaks for concurrent activities")
    tracker = new_tracker()

    # Frames 1-10: smoking only. Frames 5-14: drinking joins in
    # (starts 4 frames later), so it reaches its own streak of 10
    # four frames after smoking does - proving the two counters don't
    # share state.
    smoking_confirmed_at = None
    drinking_confirmed_at = None
    for frame_num in range(1, 15):
        dets = [make_detection("smoking")]
        if frame_num >= 5:
            dets.append(make_detection("drinking"))
        confirmed = tracker.update(dets)
        activities = {d.activity for d in confirmed}
        if "smoking" in activities:
            smoking_confirmed_at = frame_num
        if "drinking" in activities:
            drinking_confirmed_at = frame_num

    check(smoking_confirmed_at == 10, f"smoking confirmed on frame 10 (got {smoking_confirmed_at})")
    check(drinking_confirmed_at == 14, f"drinking confirmed on frame 14, independently (got {drinking_confirmed_at})")


# ------------------------------------------------------------
# 8. Multiple detections in a frame (same activity, two boxes)
# ------------------------------------------------------------
def test_multiple_detections_same_activity_one_frame():
    print("Test 8: two boxes of the same activity in one frame count as ONE streak step")
    tracker = new_tracker()
    confirmed = []
    for _ in range(10):
        # Two overlapping "smoking" boxes in the SAME frame, different confidences
        confirmed = tracker.update([
            make_detection("smoking", 0.60),
            make_detection("smoking", 0.93),
        ])
    check(tracker.get_streak("smoking") == 10, "streak advanced by 1 per frame, not 2 (10, not 20)")
    check(len(confirmed) == 1, "still exactly one confirmed Detection, not one per box")
    check(confirmed[0].confidence == 0.93, "the higher-confidence box represents the confirmed activity")


# ------------------------------------------------------------
# 9. SafeDriving passes through like any other activity
# ------------------------------------------------------------
def test_safe_driving_passes_through_unfiltered():
    print("Test 9: 'safe_driving' is tracked/confirmed like any other activity")
    print("        (the tracker does not decide what's scoreable - that's SafetyScoreEngine's job)")
    tracker = new_tracker()
    confirmed = []
    for _ in range(10):
        confirmed = tracker.update([make_detection("safe_driving")])
    check(len(confirmed) == 1, "safe_driving gets confirmed by the tracker, same as any activity")
    check(confirmed[0].activity == "safe_driving", "confirmed activity is 'safe_driving'")


# ------------------------------------------------------------
# 10. Drowsiness passes through like any other activity
# ------------------------------------------------------------
def test_drowsiness_passes_through_unfiltered():
    print("Test 10: 'drowsiness' is tracked/confirmed like any other activity")
    print("         (SafetyScoreEngine.NON_PENALIZED_ACTIVITIES is what keeps it warning-only, not this tracker)")
    tracker = new_tracker()
    confirmed = []
    for _ in range(10):
        confirmed = tracker.update([make_detection("drowsiness")])
    check(len(confirmed) == 1, "drowsiness gets confirmed by the tracker, same as any activity")
    check(confirmed[0].activity == "drowsiness", "confirmed activity is 'drowsiness'")


# ------------------------------------------------------------
# Extra: config.yaml defaults are actually honored
# ------------------------------------------------------------
def test_defaults_come_from_config_yaml():
    print("Extra: constructing with no overrides reads real config.yaml values")
    config = load_config()
    expected_min_frames = config["detection"]["min_consecutive_frames"]
    expected_cooldown = config["detection"]["violation_cooldown_seconds"]

    tracker = ViolationTracker()  # no overrides - must read config.yaml itself
    check(tracker.min_consecutive_frames == expected_min_frames,
          f"min_consecutive_frames defaults to config.yaml's value ({expected_min_frames})")
    check(tracker.cooldown_seconds == expected_cooldown,
          f"cooldown_seconds defaults to config.yaml's value ({expected_cooldown})")


# ------------------------------------------------------------
# Extra: tracker never imports/touches the DB or scoring engine
# ------------------------------------------------------------
def test_tracker_has_no_db_or_scoring_dependency():
    print("Extra: tracker module has zero dependency on database/scoring")
    import inspect
    import detection.violation_confirmation as vc_module

    # Check only the actual import statements, not the whole file -
    # the module's own docstring shows a usage EXAMPLE of how calling
    # code (Phase 7.3) will use this tracker's output with
    # SafetyScoreEngine, which is documentation, not a dependency.
    # Scanning the full source text for method-call substrings like
    # "record_violation(" would false-positive on that example.
    import_lines = [
        line for line in inspect.getsource(vc_module).splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    has_db_import = any("database" in line for line in import_lines)
    has_scoring_import = any("scoring" in line for line in import_lines)
    check(not has_db_import, f"no import of the database layer (imports: {import_lines})")
    check(not has_scoring_import, f"no import of the scoring engine (imports: {import_lines})")

    # And confirm the class itself never stores a db/engine reference.
    tracker = ViolationTracker(min_consecutive_frames=10, cooldown_seconds=15, clock=FakeClock())
    attrs = vars(tracker).keys()
    check(not any("db" in a.lower() or "engine" in a.lower() for a in attrs),
          f"tracker instance holds no db/engine attribute (attributes: {list(attrs)})")


def main():
    print("=" * 60)
    print("DriveGuard AI - Violation Confirmation Tracker Tests")
    print("=" * 60)

    tests = [
        test_fewer_than_threshold_no_violation,
        test_exactly_threshold_confirms_once,
        test_missing_frame_resets_streak,
        test_different_activity_interrupts_streak,
        test_cooldown_blocks_immediate_reconfirmation,
        test_reconfirms_after_cooldown_expires,
        test_independent_activities_confirm_independently,
        test_multiple_detections_same_activity_one_frame,
        test_safe_driving_passes_through_unfiltered,
        test_drowsiness_passes_through_unfiltered,
        test_defaults_come_from_config_yaml,
        test_tracker_has_no_db_or_scoring_dependency,
    ]

    for test_fn in tests:
        test_fn()
        print()

    print("=" * 60)
    print(f"All {len(tests)} test groups passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
