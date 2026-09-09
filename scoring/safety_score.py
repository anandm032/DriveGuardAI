"""
DriveGuard AI - Vehicle Safety Score Engine
==============================================
This is the heart of the project (spec section 8). It turns a
confirmed violation into a point deduction against a vehicle's
PERSISTENT safety score, using penalty values from config.yaml.

Formula:
    total_penalty = base_penalty + (repeated_offense_extra if this
                    activity has been violated by this vehicle before)
    new_score = max(0, previous_score - total_penalty)

The score never goes below 0, and it is never reset - every call
reads the vehicle's current score from the database and writes the
new one back, so it survives across app restarts.

Risk-level classification (LOW/MEDIUM/HIGH) is a separate concern
built in Phase 5 (scoring/risk_classifier.py). This module preserves
whatever risk_level is already stored on the vehicle when it updates
the score - Phase 5 wires proper reclassification in on top of this.

Drowsiness is intentionally NOT a penalized activity by default (spec
section 10: it's a safety condition/warning, not automatically an
insurance penalty) - it isn't in config.yaml's `penalties` section, so
record_violation() will refuse it with a clear error rather than
silently applying no penalty. Use it for on-screen warnings elsewhere,
not for scoring, unless the project's policy is deliberately changed
by adding it to config.yaml.

RECOVERY (clean-driving reward)
--------------------------------
Unsafe behavior drops the score quickly; consistent safe driving
should slowly earn it back - the same shape as a credit score. Every
`recovery.interval_hours` of MONITORED, violation-free driving time
(tracked in seconds, not wall-clock time - the car doesn't get credit
for time the app wasn't running) adds `recovery.points_awarded_per_interval`
points back, capped at 100.

Clean time accumulates cumulatively across sessions/days (3 hours
today + 3 hours tomorrow = 6 hours credited), which is why it's stored
on the vehicle row (clean_seconds_accumulated) instead of being reset
every time a session ends. A confirmed violation resets ONLY the
current, not-yet-rewarded portion of that counter back to 0 - points
already awarded from past completed intervals are never taken back
by a later violation, since they already became part of the score.

Whatever calls this (the webcam detection loop in Phase 7, or this
sanity-check script in the meantime) is responsible for periodically
reporting elapsed monitored seconds via add_monitored_time() - this
module does not read a clock itself.

Usage:

    from database.database import Database
    from scoring.safety_score import SafetyScoreEngine

    db = Database()
    db.initialize()
    engine = SafetyScoreEngine(db)

    vehicle = db.get_or_create_vehicle("KL-XX-1234")
    result = engine.record_violation(vehicle.vehicle_id, "mobile_phone_usage", confidence=0.91)
    print(result.new_score, result.is_repeat_offense)

    recovery = engine.add_monitored_time(vehicle.vehicle_id, seconds=6 * 3600)
    print(recovery.new_score, recovery.points_awarded)
"""

from dataclasses import dataclass
from typing import Optional

from database.database import Database, DatabaseError
from utils.helpers import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

# Activities that config.yaml's `penalties` section can never contain,
# because the spec treats them as safety conditions rather than
# insurance-scored violations by default.
NON_PENALIZED_ACTIVITIES = {"drowsiness"}


class SafetyScoreError(Exception):
    """Raised when a violation can't be scored (unknown activity,
    activity intentionally excluded from scoring policy, etc.)."""
    pass


@dataclass
class ScoreResult:
    vehicle_id: int
    activity: str
    previous_score: int
    base_penalty: int
    repeat_extra_penalty: int
    total_penalty: int
    new_score: int
    is_repeat_offense: bool
    violation_id: int
    score_history_id: int


@dataclass
class RecoveryResult:
    vehicle_id: int
    seconds_added: int
    previous_score: int
    new_score: int
    points_awarded: int
    intervals_completed: int
    previous_clean_seconds: int
    new_clean_seconds: int


class SafetyScoreEngine:
    def __init__(self, db: Database, config: Optional[dict] = None):
        self.db = db
        self.config = config or load_config()
        self.penalties: dict = self.config.get("penalties", {})
        self.repeat_extra: int = self.penalties.get("repeated_offense_extra", 0)

        if not self.penalties:
            logger.warning(
                "No 'penalties' section found in config.yaml - all violations "
                "will be rejected until penalties are configured."
            )

        recovery_config = self.config.get("recovery", {})
        interval_hours = recovery_config.get("interval_hours", 6)
        self.recovery_interval_seconds: int = int(interval_hours * 3600)
        self.recovery_points_per_interval: int = int(
            recovery_config.get("points_awarded_per_interval", 2)
        )
        if not recovery_config:
            logger.warning(
                "No 'recovery' section found in config.yaml - defaulting to "
                f"{interval_hours}h -> +{self.recovery_points_per_interval} points."
            )

    # ------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------
    def record_violation(
        self,
        vehicle_id: int,
        activity: str,
        confidence: float,
        session_id: Optional[int] = None,
    ) -> ScoreResult:
        """Confirm a violation, deduct points from the vehicle's
        persistent score, and record it in violations + score_history.
        This should only be called AFTER temporal confirmation (Phase 8)
        - never once per raw detected frame."""

        activity = self._normalize_activity(activity)
        base_penalty = self._get_base_penalty(activity)

        vehicle = self.db.get_vehicle_by_id(vehicle_id)
        if vehicle is None:
            raise SafetyScoreError(f"No vehicle found with id {vehicle_id}")

        previous_score = vehicle.current_score

        prior_count = self.db.count_violations_by_activity(vehicle_id, activity)
        is_repeat = prior_count > 0
        repeat_extra_penalty = self.repeat_extra if is_repeat else 0
        total_penalty = base_penalty + repeat_extra_penalty

        new_score = apply_penalty(previous_score, total_penalty)

        try:
            violation_id = self.db.insert_violation(
                vehicle_id, activity, confidence, total_penalty, session_id=session_id
            )

            reason = f"{activity} violation"
            if is_repeat:
                reason += f" (repeat offense #{prior_count + 1})"
            score_history_id = self.db.insert_score_history(
                vehicle_id, previous_score, total_penalty, new_score, reason
            )

            # Risk level is left as-is here; Phase 5's risk_classifier
            # is what actually recomputes it. We still have to pass a
            # value to satisfy update_vehicle_score's signature.
            # A violation also resets the CURRENT (not-yet-rewarded)
            # clean-driving counter back to 0 - any recovery points
            # already banked from earlier completed intervals stay,
            # since they're already baked into new_score's ancestry.
            self.db.update_vehicle_score(
                vehicle_id, new_score, vehicle.risk_level, clean_seconds_accumulated=0
            )

        except DatabaseError as e:
            logger.error(f"Failed to record violation for vehicle {vehicle_id}: {e}")
            raise SafetyScoreError(f"Failed to record violation: {e}")

        logger.info(
            f"Vehicle {vehicle_id}: {activity} confirmed "
            f"(confidence={confidence:.2f}) - penalty={total_penalty} "
            f"({'repeat' if is_repeat else 'first'} offense) "
            f"score {previous_score} -> {new_score}"
        )

        return ScoreResult(
            vehicle_id=vehicle_id,
            activity=activity,
            previous_score=previous_score,
            base_penalty=base_penalty,
            repeat_extra_penalty=repeat_extra_penalty,
            total_penalty=total_penalty,
            new_score=new_score,
            is_repeat_offense=is_repeat,
            violation_id=violation_id,
            score_history_id=score_history_id,
        )

    # ------------------------------------------------------------
    # Recovery (clean-driving reward)
    # ------------------------------------------------------------
    def add_monitored_time(self, vehicle_id: int, seconds: int) -> RecoveryResult:
        """Report additional monitored, violation-free driving time
        for a vehicle. Call this periodically while a session is
        actively monitoring with no unconfirmed/pending violation -
        e.g. once per minute from the webcam loop (Phase 7), or once
        at session end with the session's total clean duration.

        Awards recovery.points_awarded_per_interval points for every
        full recovery.interval_hours of accumulated clean time,
        capped at a score of 100, and carries over any leftover time
        that didn't complete a full interval yet."""
        if seconds < 0:
            raise SafetyScoreError(f"seconds cannot be negative (got {seconds})")

        vehicle = self.db.get_vehicle_by_id(vehicle_id)
        if vehicle is None:
            raise SafetyScoreError(f"No vehicle found with id {vehicle_id}")

        previous_score = vehicle.current_score
        previous_clean_seconds = vehicle.clean_seconds_accumulated

        total_clean_seconds = previous_clean_seconds + int(seconds)
        intervals_completed = total_clean_seconds // self.recovery_interval_seconds
        new_clean_seconds = total_clean_seconds % self.recovery_interval_seconds
        points_awarded = intervals_completed * self.recovery_points_per_interval

        new_score = apply_recovery(previous_score, points_awarded)
        # If the score was capped at 100, the "extra" points are simply
        # not banked anywhere - once at 100 you can't bank recovery for
        # later use. The completed interval's time is still consumed
        # (moves from the counter into new_clean_seconds's remainder)
        # so we don't re-award the same block of time twice.

        try:
            self.db.update_vehicle_score(
                vehicle_id, new_score, vehicle.risk_level,
                clean_seconds_accumulated=new_clean_seconds,
            )
            if points_awarded > 0:
                self.db.insert_score_history(
                    vehicle_id, previous_score, -points_awarded, new_score,
                    f"clean driving recovery ({intervals_completed} x "
                    f"{self.recovery_interval_seconds // 3600}h interval"
                    f"{'s' if intervals_completed != 1 else ''})",
                )
        except DatabaseError as e:
            logger.error(f"Failed to record recovery for vehicle {vehicle_id}: {e}")
            raise SafetyScoreError(f"Failed to record clean-driving recovery: {e}")

        if points_awarded > 0:
            logger.info(
                f"Vehicle {vehicle_id}: +{seconds}s clean driving "
                f"({intervals_completed} interval(s) completed) - "
                f"score {previous_score} -> {new_score}"
            )
        else:
            logger.debug(
                f"Vehicle {vehicle_id}: +{seconds}s clean driving accumulated "
                f"({new_clean_seconds}/{self.recovery_interval_seconds}s toward "
                f"next recovery interval)"
            )

        return RecoveryResult(
            vehicle_id=vehicle_id,
            seconds_added=seconds,
            previous_score=previous_score,
            new_score=new_score,
            points_awarded=points_awarded,
            intervals_completed=intervals_completed,
            previous_clean_seconds=previous_clean_seconds,
            new_clean_seconds=new_clean_seconds,
        )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _normalize_activity(self, activity: str) -> str:
        if not activity or not str(activity).strip():
            raise SafetyScoreError("Activity name cannot be empty")
        return str(activity).strip().lower().replace(" ", "_")

    def _get_base_penalty(self, activity: str) -> int:
        if activity in NON_PENALIZED_ACTIVITIES:
            raise SafetyScoreError(
                f"'{activity}' is intentionally excluded from scoring by project "
                f"policy (it's a safety warning, not an insurance-penalized "
                f"violation). Handle it as a display-only alert instead, or "
                f"add it to config.yaml's penalties section if the policy changes."
            )
        if activity not in self.penalties or activity == "repeated_offense_extra":
            known_activities = [
                k for k in self.penalties.keys() if k != "repeated_offense_extra"
            ]
            raise SafetyScoreError(
                f"No penalty configured for activity '{activity}'. "
                f"Known activities: {known_activities}"
            )
        return int(self.penalties[activity])


# ------------------------------------------------------------
# Pure functions (no DB/config dependency) - kept separate so
# Phase 13's unit tests can exercise the scoring math directly
# without needing a real database.
# ------------------------------------------------------------
def apply_penalty(previous_score: int, penalty: int) -> int:
    """Apply a single penalty to a score, floored at 0. This is the
    'St = max(0, St-1 - penalty)' step of the formula."""
    return max(0, previous_score - penalty)


def apply_recovery(previous_score: int, points: int) -> int:
    """Apply a recovery award to a score, capped at 100."""
    return min(100, previous_score + points)


def calculate_total_penalty(base_penalty: int, is_repeat: bool, repeat_extra: int) -> int:
    return base_penalty + (repeat_extra if is_repeat else 0)


def calculate_recovery_intervals(
    previous_clean_seconds: int, seconds_added: int, interval_seconds: int
) -> tuple:
    """Pure helper: given clean time already accumulated and new
    seconds to add, return (intervals_completed, new_remainder_seconds).
    Kept separate from the DB-touching method so Phase 13's tests can
    exercise the interval math directly."""
    total = previous_clean_seconds + seconds_added
    intervals = total // interval_seconds
    remainder = total % interval_seconds
    return intervals, remainder
