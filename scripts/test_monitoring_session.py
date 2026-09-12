"""
DriveGuard AI - MonitoringSession Tests
============================================
No physical webcam or trained YOLO model required. Strategy:

  - Detector is STUBBED (the only component that genuinely needs
    ultralytics + a trained models/best.pt, neither available here).
  - VideoSource, ViolationTracker, SafetyScoreEngine, and Database are
    all REAL (each already independently tested in Phases 3/4/5/7.1/
    7.2) - wrapped in thin call-recording spies where a test needs to
    verify "was this called, with what" rather than just "did the
    final state come out right."
  - A real temporary SQLite database is used (deleted after each
    test), so DB session start/end is genuinely exercised, not mocked.
  - A deterministic, manually-controlled/auto-advancing fake clock
    replaces time.monotonic so cooldown/recovery-interval behavior
    and max_seconds termination are testable instantly, with no real
    sleep() calls.

Usage:
    python scripts/test_monitoring_session.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from database.database import Database
from detection.detector import Detection, DetectorError
from detection.monitoring_session import ConfirmedViolationRecord, MonitoringResult, MonitoringSession
from detection.violation_confirmation import ViolationTracker
from scoring.safety_score import SafetyScoreEngine


def check(condition: bool, description: str):
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {description}")
    assert condition, f"FAILED: {description}"


def make_test_video(path: str, num_frames: int, width: int = 64, height: int = 48, fps: float = 10.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Test setup failed: could not open VideoWriter for {path}")
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 20) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class AutoAdvanceClock:
    """Each call returns the current time, then advances by `step` -
    simulates `step` seconds passing per clock READ (not per frame -
    MonitoringSession itself reads the clock once per frame for its
    own max_seconds bookkeeping, but a ViolationTracker sharing the
    SAME clock instance would add its own additional reads on top of
    that; keep this clock dedicated to whichever component you're
    actually timing, as the tests below do)."""
    def __init__(self, step: float = 1.0, start: float = 0.0):
        self.t = start
        self.step = step

    def __call__(self) -> float:
        current = self.t
        self.t += self.step
        return current


class StubDetector:
    """Returns a pre-programmed list of Detection objects per call, in
    order. Returns [] once the program is exhausted (instead of
    raising), so a test video longer than the programmed sequence
    doesn't crash - just stops producing detections."""
    def __init__(self, programmed_detections):
        self._program = list(programmed_detections)
        self.call_count = 0
        self.frames_seen = []

    def detect(self, frame):
        self.call_count += 1
        self.frames_seen.append(frame.shape)
        if self._program:
            return self._program.pop(0)
        return []


class RaisingDetector:
    """Raises after a fixed number of calls - used to test that
    exceptions mid-session still release resources correctly."""
    def __init__(self, fail_on_call: int):
        self.fail_on_call = fail_on_call
        self.call_count = 0

    def detect(self, frame):
        self.call_count += 1
        if self.call_count == self.fail_on_call:
            raise RuntimeError("Simulated detector failure")
        return []


class SpyTracker:
    """Wraps a real ViolationTracker, counting calls while delegating
    all actual logic to it - so confirmation behavior is genuinely
    correct (not reimplemented here) but still independently
    verifiable as 'was this called'."""
    def __init__(self, real_tracker: ViolationTracker):
        self._real = real_tracker
        self.call_count = 0

    def update(self, detections):
        self.call_count += 1
        return self._real.update(detections)


class SpyEngine:
    """Wraps a real SafetyScoreEngine, recording calls while
    delegating to it - so DB/score state is genuinely correct (not
    reimplemented here) but call arguments are independently
    verifiable."""
    def __init__(self, real_engine: SafetyScoreEngine):
        self._real = real_engine
        self.record_violation_calls = []
        self.add_monitored_time_calls = []

    def record_violation(self, vehicle_id, activity, confidence, session_id=None):
        self.record_violation_calls.append((vehicle_id, activity, confidence, session_id))
        return self._real.record_violation(vehicle_id, activity, confidence, session_id=session_id)

    def add_monitored_time(self, vehicle_id, seconds):
        self.add_monitored_time_calls.append((vehicle_id, seconds))
        return self._real.add_monitored_time(vehicle_id, seconds)


def smoking_detection(confidence=0.9):
    return Detection(activity="smoking", confidence=confidence, bbox=(0.0, 0.0, 10.0, 10.0),
                      raw_class_name="Smoking")


def drinking_detection(confidence=0.9):
    return Detection(activity="drinking", confidence=confidence, bbox=(0.0, 0.0, 10.0, 10.0),
                      raw_class_name="Drinking")


def make_real_stack(db_path, min_consecutive_frames=3, cooldown_seconds=5.0, clock=None):
    """Builds a real Database + SafetyScoreEngine + ViolationTracker
    (wrapped in spies) sharing one clock, plus a fresh test vehicle.
    Small thresholds (vs. config.yaml's real 10/15) keep test videos
    short."""
    clock = clock or AutoAdvanceClock(step=1.0)
    db = Database(db_path=db_path)
    db.initialize()
    vehicle = db.get_or_create_vehicle(f"MONITOR-TEST-{os.getpid()}-{id(clock)}")

    real_tracker = ViolationTracker(min_consecutive_frames=min_consecutive_frames,
                                     cooldown_seconds=cooldown_seconds, clock=clock)
    real_engine = SafetyScoreEngine(db)

    tracker = SpyTracker(real_tracker)
    engine = SpyEngine(real_engine)
    return db, vehicle, tracker, engine, clock


def session_ended_at(db: Database, session_id: int):
    """Peek directly at the sessions table - acceptable in a TEST
    (verifying end state), even though MonitoringSession itself never
    does raw queries like this. Column names (session_id, end_time)
    match database/schema.sql exactly."""
    row = db._conn.execute(
        "SELECT end_time FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    return row["end_time"] if row else None


# ------------------------------------------------------------
# 1. MonitoringSession can initialize with a video-file source
# ------------------------------------------------------------
def test_initializes_with_video_file_source():
    print("Test 1: MonitoringSession initializes with a video-file source")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=5)

        db, vehicle, tracker, engine, clock = make_real_stack(db_path)
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=StubDetector([]), tracker=tracker, engine=engine, clock=clock,
        )
        check(session.vehicle_id == vehicle.vehicle_id, "vehicle_id stored correctly")
        check(session.source == video_path, "source stored correctly")
        check(session.video_source is None, "video_source not yet created before run()")
        check(session.session_id is None, "session_id not yet set before run()")
        db.close()


# ------------------------------------------------------------
# Extra: default wiring genuinely reaches the real Detector class
# ------------------------------------------------------------
def test_default_detector_is_the_real_one():
    print("Extra: omitting `detector=` wires up the REAL Detector (fails clearly without a model, as expected)")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=1)
        db, vehicle, tracker, engine, clock = make_real_stack(db_path)
        try:
            MonitoringSession(vehicle_id=vehicle.vehicle_id, source=video_path, db=db)
            print("  [INFO] A real trained model was found and loaded - unusual for this sandbox")
        except DetectorError as e:
            check(True, f"real Detector correctly attempted and failed cleanly (no model/ultralytics here): {e}")
        db.close()


# ------------------------------------------------------------
# 2 & 3. VideoSource opens correctly / frames are processed
# ------------------------------------------------------------
def test_video_opens_and_frames_are_processed():
    print("Test 2/3: video opens correctly through the session, and all frames are processed")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=8)

        db, vehicle, tracker, engine, clock = make_real_stack(db_path)
        detector = StubDetector([[] for _ in range(8)])
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine, clock=clock,
        )
        result = session.run()

        check(isinstance(result, MonitoringResult), "run() returns a MonitoringResult")
        check(result.frames_processed == 8, f"all 8 frames processed (got {result.frames_processed})")
        check(result.stopped_reason == "end_of_stream", "stopped naturally at end of the video file")
        db.close()


# ------------------------------------------------------------
# 4. Detector is called
# ------------------------------------------------------------
def test_detector_is_called():
    print("Test 4: Detector.detect() is called once per frame")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=6)

        db, vehicle, tracker, engine, clock = make_real_stack(db_path)
        detector = StubDetector([[] for _ in range(6)])
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine, clock=clock,
        )
        result = session.run()

        check(detector.call_count == 6, f"detector called exactly 6 times, once per frame (got {detector.call_count})")
        check(detector.frames_seen[0] == (48, 64, 3), "detector actually received real frame arrays, not placeholders")
        db.close()


# ------------------------------------------------------------
# 5. ViolationTracker is called
# ------------------------------------------------------------
def test_tracker_is_called():
    print("Test 5: ViolationTracker.update() is called once per frame")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=6)

        db, vehicle, tracker, engine, clock = make_real_stack(db_path)
        detector = StubDetector([[] for _ in range(6)])
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine, clock=clock,
        )
        session.run()

        check(tracker.call_count == 6, f"tracker.update() called exactly 6 times (got {tracker.call_count})")
        db.close()


# ------------------------------------------------------------
# 6. Confirmed violations are sent to SafetyScoreEngine
# ------------------------------------------------------------
def test_confirmed_violations_reach_scoring_engine():
    print("Test 6: a confirmed violation actually reaches SafetyScoreEngine.record_violation()")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=10)

        db, vehicle, tracker, engine, clock = make_real_stack(db_path, min_consecutive_frames=3)
        # "smoking" detected on frames 1-3 (confirms at frame 3, streak=3),
        # then absent for the rest - exactly ONE confirmation expected.
        program = [[smoking_detection()], [smoking_detection()], [smoking_detection()]] + [[] for _ in range(7)]
        detector = StubDetector(program)
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine, clock=clock,
        )
        starting_score = vehicle.current_score
        result = session.run()

        check(len(engine.record_violation_calls) == 1,
              f"record_violation() called exactly once (got {len(engine.record_violation_calls)})")
        called_vehicle_id, called_activity, called_confidence, called_session_id = engine.record_violation_calls[0]
        check(called_vehicle_id == vehicle.vehicle_id, "called with the correct vehicle_id")
        check(called_activity == "smoking", "called with the correct activity")
        check(called_session_id == result.session_id, "called with the session's own session_id")

        check(len(result.confirmed_violations) == 1, "MonitoringResult.confirmed_violations has exactly one entry")
        v = result.confirmed_violations[0]
        check(isinstance(v, ConfirmedViolationRecord), "entry is a ConfirmedViolationRecord")
        check(v.activity == "smoking" and v.frame_number == 3, f"correct activity/frame_number (got {v.activity}, {v.frame_number})")

        # And the score genuinely changed in the real database - not just recorded in a list.
        reloaded = db.get_vehicle_by_id(vehicle.vehicle_id)
        check(reloaded.current_score < starting_score,
              f"vehicle's real score actually decreased in the DB ({starting_score} -> {reloaded.current_score})")
        check(result.final_score == reloaded.current_score, "MonitoringResult.final_score matches the DB")
        db.close()


# ------------------------------------------------------------
# 7. Database session starts and ends correctly
# ------------------------------------------------------------
def test_database_session_starts_and_ends():
    print("Test 7: a DB session is started and cleanly ended")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=4)

        db, vehicle, tracker, engine, clock = make_real_stack(db_path)
        detector = StubDetector([[] for _ in range(4)])
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine, clock=clock,
        )
        check(session_ended_at(db, session_id=999999) is None,
              "sanity: helper correctly returns None for a session_id that doesn't exist")
        result = session.run()

        check(result.session_id is not None, "a real session_id was created")
        check(session_ended_at(db, result.session_id) is not None,
              "the session's ended_at timestamp is set (cleanly ended, not left open)")
        db.close()


# ------------------------------------------------------------
# 8. max_frames / max_seconds allow deterministic termination
# ------------------------------------------------------------
def test_max_frames_terminates_early():
    print("Test 8a: max_frames stops the session before the video ends")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=20)

        db, vehicle, tracker, engine, clock = make_real_stack(db_path)
        detector = StubDetector([[] for _ in range(20)])
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine, clock=clock,
        )
        result = session.run(max_frames=5)

        check(result.frames_processed == 5, f"stopped at exactly 5 frames (got {result.frames_processed})")
        check(result.stopped_reason == "max_frames", f"stopped_reason is 'max_frames' (got {result.stopped_reason})")
        db.close()


def test_max_seconds_terminates_early():
    print("Test 8b: max_seconds stops the session deterministically (fake clock, no real sleep)")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=20)

        # The tracker gets its OWN clock (irrelevant here since the
        # detector never produces a detection, so no cooldown/streak
        # timing is exercised) - deliberately NOT the same clock
        # instance the session uses for its own max_seconds check.
        # Sharing one auto-advancing clock between both would mean
        # each frame consumes 2 ticks (one from the tracker's internal
        # read, one from the session's own), not 1 - making "N ticks
        # elapsed" no longer line up with "N frames processed".
        db, vehicle, tracker, engine, _tracker_clock = make_real_stack(db_path)
        detector = StubDetector([[] for _ in range(20)])
        session_clock = AutoAdvanceClock(step=1.0)  # 1 simulated second per session-level clock read
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine, clock=session_clock,
        )
        result = session.run(max_seconds=5.0)

        check(result.frames_processed == 5, f"stopped after ~5 simulated seconds = 5 frames (got {result.frames_processed})")
        check(result.stopped_reason == "max_seconds", f"stopped_reason is 'max_seconds' (got {result.stopped_reason})")
        db.close()


# ------------------------------------------------------------
# 9. Video resources are released
# ------------------------------------------------------------
def test_video_resources_released():
    print("Test 9: the video source is released after run() completes")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=4)

        db, vehicle, tracker, engine, clock = make_real_stack(db_path)
        detector = StubDetector([[] for _ in range(4)])
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine, clock=clock,
        )
        session.run()

        check(session.video_source is not None, "video_source reference retained for inspection")
        check(not session.video_source.is_opened(), "video_source correctly released (is_opened() == False) after run()")
        db.close()


# ------------------------------------------------------------
# 10. Exceptions do not leave the video source or DB session open
# ------------------------------------------------------------
def test_exception_still_releases_resources():
    print("Test 10: an exception mid-session still releases the video source and ends the DB session")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=10)

        db, vehicle, tracker, engine, clock = make_real_stack(db_path)
        detector = RaisingDetector(fail_on_call=3)  # blows up on the 3rd frame
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine, clock=clock,
        )

        try:
            session.run()
            check(False, "run() should have propagated the RuntimeError from the detector")
        except RuntimeError as e:
            check("Simulated detector failure" in str(e), "the original exception propagated, not swallowed")

        check(session.video_source is not None, "video_source reference still available for inspection")
        check(not session.video_source.is_opened(), "video source was released despite the exception")
        check(session.session_id is not None, "a DB session had been started before the failure")
        check(session_ended_at(db, session.session_id) is not None,
              "the DB session was still cleanly ended despite the exception")

        # Hand-traced exact expectation: RaisingDetector returns []
        # (no violations) for frames 1-2, then raises on its 3rd call
        # (frame 3, before the tracker or anything else runs that
        # iteration). Session clock: start_time reads 0.0 (->1.0);
        # frame1's `now` reads 2.0 (tracker's own read of the SAME
        # shared clock consumes 1.0 first); frame2's `now` reads 4.0
        # (tracker consumes 3.0 first). The crash on frame 3 happens
        # before any further reads. In `finally`, one more read
        # returns 5.0, so leftover = 5.0 - 0.0 = 5.0 - proving the
        # exception path reports exactly the genuine elapsed time up
        # to the crash, nothing more.
        check(len(engine.add_monitored_time_calls) == 1,
              f"exception path still reports leftover time exactly once "
              f"(got {len(engine.add_monitored_time_calls)} calls)")
        called_vehicle_id, called_seconds = engine.add_monitored_time_calls[0]
        check(called_vehicle_id == vehicle.vehicle_id, "leftover report used the correct vehicle_id")
        check(called_seconds == 5.0,
              f"leftover amount is exactly the real elapsed time up to the crash (got {called_seconds})")
        db.close()


# ------------------------------------------------------------
# NEW: leftover monitored time is reported for a session shorter
# than the recovery interval (no periodic report ever fires)
# ------------------------------------------------------------
def test_leftover_time_reported_when_session_shorter_than_interval():
    print("Test 11: leftover clean time is reported when the session never reaches a full interval")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=5)

        # Tracker gets a STATIC clock (never advances) - fully
        # decoupled from the session's own clock reads, so the
        # session-level trace below is exact and doesn't depend on
        # how many times the tracker happens to read time internally.
        db, vehicle, tracker, engine, _ = make_real_stack(db_path, clock=lambda: 0.0)
        detector = StubDetector([[] for _ in range(5)])  # no detections at all -> no violations
        session_clock = AutoAdvanceClock(step=1.0)
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine,
            clock=session_clock, recovery_report_interval_seconds=10.0,  # never reached in 5 simulated seconds
        )
        result = session.run()

        # Hand-traced exact expectation: start_time reads 0.0 (clock
        # advances to 1.0), then one session-level clock read per
        # frame (1.0, 2.0, 3.0, 4.0, 5.0) - elapsed never reaches the
        # 10s interval, so zero periodic reports fire. In `finally`,
        # one more read returns 6.0, so leftover = 6.0 - 0.0 = 6.0.
        check(len(engine.add_monitored_time_calls) == 1,
              f"add_monitored_time() called exactly once, as a leftover report "
              f"(got {len(engine.add_monitored_time_calls)} calls)")
        called_vehicle_id, called_seconds = engine.add_monitored_time_calls[0]
        check(called_vehicle_id == vehicle.vehicle_id, "called with the correct vehicle_id")
        check(called_seconds == 6.0, f"called with the exact real elapsed leftover time (got {called_seconds})")
        check(result.stopped_reason == "end_of_stream", "session ended naturally (not via max_frames/max_seconds)")
        db.close()


# ------------------------------------------------------------
# NEW: partial interval remaining after one or more periodic reports
# ------------------------------------------------------------
def test_leftover_after_periodic_reports():
    print("Test 12: leftover time is reported correctly after periodic reports already fired")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=12)

        db, vehicle, tracker, engine, _ = make_real_stack(db_path, clock=lambda: 0.0)
        detector = StubDetector([[] for _ in range(12)])  # no violations - isolates periodic+leftover interaction
        session_clock = AutoAdvanceClock(step=1.0)
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine,
            clock=session_clock, recovery_report_interval_seconds=5.0,
        )
        session.run()

        # Hand-traced exact expectation: periodic reports fire at
        # simulated t=5.0 (frame 5) and t=10.0 (frame 10), each
        # reporting exactly 5.0s - elapsed since the PREVIOUS report
        # point, never the full time since session start. The session
        # then runs 2 more frames (11, 12) and the video ends, leaving
        # a final leftover of 3.0s reported once in `finally`.
        check(len(engine.add_monitored_time_calls) == 3,
              f"3 total calls: 2 periodic + 1 leftover (got {len(engine.add_monitored_time_calls)})")
        seconds_reported = [s for _, s in engine.add_monitored_time_calls]
        check(seconds_reported == [5.0, 5.0, 3.0],
              f"exact reported amounts are [5.0, 5.0, 3.0] (got {seconds_reported})")
        check(sum(seconds_reported) == 13.0,
              "reported amounts sum to the real total elapsed time (13s), nothing double-counted or lost")
        db.close()


# ------------------------------------------------------------
# NEW: a confirmed violation resets the clean-time window, so
# leftover/periodic reporting never spans across it
# ------------------------------------------------------------
def test_violation_resets_the_recovery_window():
    print("Test 13: a confirmed violation resets the wall-clock window used for recovery reporting")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=5)

        # min_consecutive_frames=1 so the very first detection confirms
        # immediately, on a known frame - makes the trace exact.
        db, vehicle, tracker, engine, _ = make_real_stack(db_path, min_consecutive_frames=1, clock=lambda: 0.0)
        program = [[smoking_detection()]] + [[] for _ in range(4)]  # violation on frame 1 only
        detector = StubDetector(program)
        session_clock = AutoAdvanceClock(step=1.0)
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine,
            clock=session_clock, recovery_report_interval_seconds=100.0,  # large - isolates this from periodic reporting
        )
        session.run()

        check(len(engine.record_violation_calls) == 1, "exactly one violation was confirmed and recorded")

        # Hand-traced exact expectation: the violation on frame 1 resets
        # last_recovery_report_time to 1.0 (not the original 0.0). If
        # the reset did NOT happen (the bug being fixed), leftover would
        # incorrectly be 7.0 (measured from session start); correctly
        # reset, it's 6.0 (measured from just after the violation).
        check(len(engine.add_monitored_time_calls) == 1,
              f"exactly one leftover report (got {len(engine.add_monitored_time_calls)})")
        called_vehicle_id, called_seconds = engine.add_monitored_time_calls[0]
        check(called_seconds == 6.0,
              f"leftover correctly measured from AFTER the violation, not from session start "
              f"(got {called_seconds}; a value of 7.0 would mean the pre-violation window leaked in)")
        db.close()


# ------------------------------------------------------------
# NEW: TWO confirmed violations within the same recovery interval -
# each reset must discard its own pre-violation window, not just the
# first one. Proves this doesn't just work for a single violation.
# ------------------------------------------------------------
def test_multiple_violations_each_reset_the_window():
    print("Test 14: multiple violations within one interval - each reset discards its OWN pre-violation clean time")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        video_path = os.path.join(tmp_dir, "clip.mp4")
        make_test_video(video_path, num_frames=10)

        # Two DIFFERENT activities (independent streaks, no cooldown
        # interaction between them) so both confirm immediately with
        # min_consecutive_frames=1, on known frames:
        #   frame 1: smoking  (violation #1)
        #   frames 2-5: clean (4 frames)
        #   frame 6: drinking (violation #2)
        #   frames 7-10: clean (4 frames)
        db, vehicle, tracker, engine, _ = make_real_stack(db_path, min_consecutive_frames=1, clock=lambda: 0.0)
        program = (
            [[smoking_detection()]] + [[] for _ in range(4)]
            + [[drinking_detection()]] + [[] for _ in range(4)]
        )
        detector = StubDetector(program)
        session_clock = AutoAdvanceClock(step=1.0)
        session = MonitoringSession(
            vehicle_id=vehicle.vehicle_id, source=video_path,
            db=db, detector=detector, tracker=tracker, engine=engine,
            clock=session_clock, recovery_report_interval_seconds=100.0,  # large - isolates from periodic reporting
        )
        session.run()

        check(len(engine.record_violation_calls) == 2, "both violations were confirmed and recorded")
        check(engine.record_violation_calls[0][1] == "smoking", "1st violation is smoking (frame 1)")
        check(engine.record_violation_calls[1][1] == "drinking", "2nd violation is drinking (frame 6)")

        # Hand-traced expectation:
        #   start_time reads 0.0 (clock -> 1.0). last_report = 0.0.
        #   Frame 1: violation confirmed -> record_violation() succeeds
        #     -> last_report reset via clock() -> reads 1.0 (-> 2.0).
        #     `now` read -> 2.0 (-> 3.0). elapsed = 2.0-1.0 = 1.0 < 100.
        #   Frames 2-5 (clean): `now` reads 3.0, 4.0, 5.0, 6.0 in turn -
        #     elapsed since last_report (1.0) grows to 2,3,4,5 - all < 100,
        #     so NO periodic report ever fires for this 5-second window.
        #   Frame 6: violation confirmed -> record_violation() succeeds
        #     -> last_report reset via clock() -> reads 7.0 (-> 8.0).
        #     `now` read -> 8.0 (-> 9.0). elapsed = 1.0 < 100.
        #   Frames 7-10 (clean): `now` reads 9.0, 10.0, 11.0, 12.0 -
        #     elapsed since last_report (7.0) grows to 2,3,4,5 - all < 100.
        #   finally: one more read -> 13.0 (-> 14.0).
        #     leftover = 13.0 - 7.0 (last_report after violation #2) = 6.0.
        #
        # The key thing this proves: the 6 real seconds of clean driving
        # BETWEEN violation #1 (t=1.0) and violation #2 (t=7.0) is NEVER
        # reported to add_monitored_time() at all - violation #2's reset
        # discards it, exactly like Phase 4's "a violation resets the
        # CURRENT (unrewarded) clean-driving period" design intends. Only
        # the clean stretch AFTER the last violation is ever credited.
        check(len(engine.add_monitored_time_calls) == 1,
              f"exactly ONE add_monitored_time() call for the whole session "
              f"(got {len(engine.add_monitored_time_calls)}) - the inter-violation "
              f"clean window is correctly never reported, not accidentally banked")
        called_vehicle_id, called_seconds = engine.add_monitored_time_calls[0]
        check(called_vehicle_id == vehicle.vehicle_id, "leftover report used the correct vehicle_id")
        check(called_seconds == 6.0,
              f"leftover reflects ONLY the clean time after the 2nd (most recent) violation "
              f"(got {called_seconds}; 12.0 would mean both violations' pre-windows leaked in, "
              f"and 11.0 would mean only the 1st reset was honored)")
        db.close()


def main():
    print("=" * 60)
    print("DriveGuard AI - MonitoringSession Tests")
    print("=" * 60)

    tests = [
        test_initializes_with_video_file_source,
        test_default_detector_is_the_real_one,
        test_video_opens_and_frames_are_processed,
        test_detector_is_called,
        test_tracker_is_called,
        test_confirmed_violations_reach_scoring_engine,
        test_database_session_starts_and_ends,
        test_max_frames_terminates_early,
        test_max_seconds_terminates_early,
        test_video_resources_released,
        test_exception_still_releases_resources,
        test_leftover_time_reported_when_session_shorter_than_interval,
        test_leftover_after_periodic_reports,
        test_violation_resets_the_recovery_window,
        test_multiple_violations_each_reset_the_window,
    ]

    for test_fn in tests:
        test_fn()
        print()

    print("=" * 60)
    print(f"All {len(tests)} test groups passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
