"""
DriveGuard AI - Monitoring Session Orchestrator
====================================================
Wires together the already-built, already-tested components into the
real-time pipeline:

    VideoSource -> Detector.detect() -> ViolationTracker.update()
                 -> SafetyScoreEngine.record_violation() / .add_monitored_time()

This module contains NO detection logic, NO temporal-confirmation
logic, NO scoring/risk logic, and NO raw database queries - it only
calls the public methods those existing modules already expose
(detection/video_source.py, detection/detector.py,
detection/violation_confirmation.py, scoring/safety_score.py,
database/database.py). Its only job is sequencing calls between them
correctly and tracking real elapsed time.

Design notes:

- SafeDriving/Drowsiness are NOT special-cased here. ViolationTracker
  confirms them like any other activity (by design, see Phase 7.1);
  SafetyScoreEngine is what decides they don't get penalized (via its
  NON_PENALIZED_ACTIVITIES set and `penalties` config). This module
  simply hands every confirmed activity to record_violation() and
  lets that existing logic decide what happens - it never filters by
  activity name itself.

- Recovery time is reported using REAL elapsed wall-clock seconds
  (via the injectable clock, time.monotonic by default), measured
  between periodic report points during the loop - never inferred
  from frame count, since frame rate can vary or frames can be
  skipped.

- Every dependency (Database, Detector, ViolationTracker,
  SafetyScoreEngine) is optionally injectable, specifically so this
  can be unit-tested with a lightweight stub Detector instead of a
  real YOLO model (see scripts/test_monitoring_session.py).

Usage:

    from detection.monitoring_session import MonitoringSession

    session = MonitoringSession(vehicle_id=1, source="drive_clip.mp4")
    result = session.run(max_frames=500)
    print(result.final_score, result.final_risk, result.frames_processed)
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from database.database import Database
from detection.detector import Detection, Detector
from detection.video_source import VideoSource, VideoSourceError
from detection.violation_confirmation import ViolationTracker
from scoring.safety_score import SafetyScoreEngine
from utils.helpers import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

# How often (real seconds) accumulated clean monitoring time is
# reported to SafetyScoreEngine.add_monitored_time(). Not a
# config.yaml setting (deliberately, per Phase 7.3 scope: avoid
# touching config.yaml) - just a sensible default, overridable per
# MonitoringSession instance for testing.
DEFAULT_RECOVERY_REPORT_INTERVAL_SECONDS = 60.0


@dataclass
class ConfirmedViolationRecord:
    """A lightweight summary of one confirmed violation as it happened
    during a session - not the full ScoreResult object, to keep
    MonitoringResult easy to read/print/serialize."""
    activity: str
    confidence: float
    frame_number: int
    new_score: int
    new_risk_level: str


@dataclass
class MonitoringResult:
    session_id: Optional[int]
    frames_processed: int
    confirmed_violations: List[ConfirmedViolationRecord] = field(default_factory=list)
    final_score: Optional[int] = None
    final_risk: Optional[str] = None
    elapsed_seconds: float = 0.0
    stopped_reason: str = "end_of_stream"  # "end_of_stream" | "max_frames" | "max_seconds"


class MonitoringSession:
    def __init__(
        self,
        vehicle_id: int,
        source: Union[int, str],
        db: Optional[Database] = None,
        detector: Optional[Detector] = None,
        tracker: Optional[ViolationTracker] = None,
        engine: Optional[SafetyScoreEngine] = None,
        config: Optional[dict] = None,
        recovery_report_interval_seconds: float = DEFAULT_RECOVERY_REPORT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        """All of db/detector/tracker/engine are optional purely for
        dependency injection in tests - in normal use, leave them all
        as None and this constructs the real components itself
        (including the real Detector, which requires ultralytics and
        a trained models/best.pt to succeed).

        clock defaults to time.monotonic (immune to system clock
        adjustments) and is injectable so tests can simulate elapsed
        time without real sleep() calls - the same pattern
        ViolationTracker already uses.
        """
        self.vehicle_id = vehicle_id
        self.source = source
        self.config = config if config is not None else load_config()

        self.db = db if db is not None else Database()
        self.db.initialize()  # safe/idempotent - see database.py's own docstring

        self.detector = detector if detector is not None else Detector(config=self.config)
        self._clock = clock
        self.tracker = tracker if tracker is not None else ViolationTracker(
            config=self.config, clock=self._clock
        )
        self.engine = engine if engine is not None else SafetyScoreEngine(self.db, config=self.config)

        self.recovery_report_interval_seconds = recovery_report_interval_seconds

        # Populated during run() - kept as attributes (rather than
        # only local variables) specifically so callers/tests can
        # inspect final state afterward, e.g. to confirm the video
        # source was actually released.
        self.video_source: Optional[VideoSource] = None
        self.session_id: Optional[int] = None

    def run(
        self,
        max_frames: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> MonitoringResult:
        """Run the monitoring loop until the video source is
        exhausted (end of file, or a webcam read failure), or until
        max_frames / max_seconds is reached - whichever comes first.
        Both limits are optional and exist so this can terminate
        deterministically in tests on a small video, instead of
        running forever (spec item: avoid an uncontrollable infinite
        loop).

        Guarantees, even if an exception occurs partway through:
          - the video source is released (never leaked)
          - the database session (if one was started) is ended
          - any monitored clean time accrued since the last periodic
            report is reported to SafetyScoreEngine.add_monitored_time()
            before the session ends, so a partial interval is never
            silently lost - this uses real elapsed wall-clock time,
            never inferred from frame count, and correctly excludes
            any window that already contained a confirmed violation
            (see the reset in the violation-handling loop below)

        Re-raises whatever exception occurred after that cleanup -
        this module doesn't swallow errors, matching how
        VideoSourceError/DetectorError/DatabaseError are already
        handled elsewhere in the project (fail clearly, don't hide
        it)."""
        frames_processed = 0
        confirmed_records: List[ConfirmedViolationRecord] = []
        stopped_reason = "end_of_stream"

        start_time = self._clock()
        last_recovery_report_time = start_time

        self.video_source = VideoSource(self.source)
        self.session_id = None

        try:
            self.video_source.open()
            self.session_id = self.db.start_session(self.vehicle_id)
            logger.info(
                f"Monitoring session started: vehicle_id={self.vehicle_id}, "
                f"session_id={self.session_id}, source={self.source!r}"
            )

            for frame in self.video_source:
                frames_processed += 1

                detections: List[Detection] = self.detector.detect(frame)
                confirmed = self.tracker.update(detections)

                for violation in confirmed:
                    if violation.activity in self.config.get('penalties', {}):
                        score_result = self.engine.record_violation(
                            self.vehicle_id,
                            violation.activity,
                            violation.confidence,
                            session_id=self.session_id,
                        )

                        confirmed_records.append(ConfirmedViolationRecord(
                            activity=violation.activity,
                            confidence=violation.confidence,
                            frame_number=frames_processed,
                            new_score=score_result.new_score,
                            new_risk_level=score_result.new_risk_level,
                        ))

                        # record_violation() already reset the vehicle's
                        # clean-time counter in the DB. Reset our own
                        # wall-clock bookkeeping to match RIGHT NOW.
                        last_recovery_report_time = self._clock()

                    else:
                        # Activities such as safe_driving may be confirmed
                        # by the temporal tracker but must not affect score.
                        current_vehicle = self.db.get_vehicle_by_id(self.vehicle_id)

                        confirmed_records.append(ConfirmedViolationRecord(
                            activity=violation.activity,
                            confidence=violation.confidence,
                            frame_number=frames_processed,
                            new_score=current_vehicle.current_score,
                            new_risk_level=current_vehicle.risk_level,
                        ))
                now = self._clock()
                elapsed_since_report = now - last_recovery_report_time
                if elapsed_since_report >= self.recovery_report_interval_seconds:
                    self.engine.add_monitored_time(self.vehicle_id, elapsed_since_report)
                    last_recovery_report_time = now

                if max_frames is not None and frames_processed >= max_frames:
                    stopped_reason = "max_frames"
                    break
                if max_seconds is not None and (now - start_time) >= max_seconds:
                    stopped_reason = "max_seconds"
                    break

        except Exception:
            logger.exception(
                f"Monitoring session failed for vehicle_id={self.vehicle_id} "
                f"after {frames_processed} frame(s)"
            )
            raise

        finally:
            # Runs on every exit path - normal completion, break, or
            # an exception - so the video source and DB session are
            # never left open/leaked, AND any clean monitored time
            # accrued since the last periodic report (or the last
            # violation, whichever is more recent - see the reset
            # above) is reported before the session closes, so a
            # partial interval is never silently discarded.
            #
            # Uses the SAME last_recovery_report_time bookkeeping the
            # periodic report above uses, so this is exactly "whatever
            # elapsed time hasn't been reported yet" - never inferred
            # from frame count, and never double-counting a window
            # that already contained a violation (that window's start
            # point was moved forward by the reset above).
            #
            # Only runs if a DB session actually started - if open()
            # or start_session() itself failed, self.session_id is
            # still None here, and there's nothing to report against.
            if self.session_id is not None:
                leftover_seconds = self._clock() - last_recovery_report_time
                if leftover_seconds > 0:
                    try:
                        self.engine.add_monitored_time(self.vehicle_id, leftover_seconds)
                    except Exception as e:
                        logger.error(
                            f"Failed to report leftover monitored time for vehicle "
                            f"{self.vehicle_id}: {e}"
                        )

            self.video_source.close()
            if self.session_id is not None:
                self.db.end_session(self.session_id)
                logger.info(f"Monitoring session ended: session_id={self.session_id}")

        elapsed_seconds = self._clock() - start_time
        vehicle = self.db.get_vehicle_by_id(self.vehicle_id)

        result = MonitoringResult(
            session_id=self.session_id,
            frames_processed=frames_processed,
            confirmed_violations=confirmed_records,
            final_score=vehicle.current_score if vehicle is not None else None,
            final_risk=vehicle.risk_level if vehicle is not None else None,
            elapsed_seconds=elapsed_seconds,
            stopped_reason=stopped_reason,
        )
        logger.info(
            f"Monitoring session result: frames={result.frames_processed}, "
            f"violations={len(result.confirmed_violations)}, "
            f"final_score={result.final_score}, final_risk={result.final_risk}, "
            f"stopped_reason={result.stopped_reason}"
        )
        return result

