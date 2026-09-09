"""
DriveGuard AI - Recovery (Clean-Driving Reward) Sanity Check
=================================================================
Manual end-to-end check for the recovery feature added to Phase 4:
score recovers +2 for every 6 monitored hours of violation-free
driving, cumulative across separate calls/sessions, reset by a
violation, capped at 100.

Reproduces the exact worked example from the spec:

    Start                           100
    Phone violation                 -15  -> 85
    6h clean driving                 +2  -> 87
    Another 6h clean driving         +2  -> 89
    Smoking violation               -10  -> 79

Usage:
    python scripts/test_recovery.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import Database
from scoring.safety_score import SafetyScoreEngine, SafetyScoreError, calculate_recovery_intervals

ONE_HOUR = 3600


def main():
    print("=" * 60)
    print("DriveGuard AI - Recovery Sanity Check")
    print("=" * 60)

    db = Database()
    db.initialize()
    engine = SafetyScoreEngine(db)

    vehicle = db.get_or_create_vehicle(f"RECOVERY-TEST-{int(time.time())}")
    print(f"[OK] Fresh vehicle: {vehicle.vehicle_number} (score={vehicle.current_score})")

    # --- Reproduce the spec's worked example exactly ---
    r1 = engine.record_violation(vehicle.vehicle_id, "mobile_phone_usage", 0.91)
    print(f"[OK] Phone violation: {r1.previous_score} -> {r1.new_score}")
    assert r1.new_score == 85

    r2 = engine.add_monitored_time(vehicle.vehicle_id, 6 * ONE_HOUR)
    print(f"[OK] 6h clean driving: {r2.previous_score} -> {r2.new_score} "
          f"(+{r2.points_awarded}, {r2.intervals_completed} interval(s))")
    assert r2.new_score == 87, f"Expected 87, got {r2.new_score}"
    assert r2.new_clean_seconds == 0

    r3 = engine.add_monitored_time(vehicle.vehicle_id, 6 * ONE_HOUR)
    print(f"[OK] Another 6h clean driving: {r3.previous_score} -> {r3.new_score} "
          f"(+{r3.points_awarded})")
    assert r3.new_score == 89, f"Expected 89, got {r3.new_score}"

    r4 = engine.record_violation(vehicle.vehicle_id, "smoking", 0.87)
    print(f"[OK] Smoking violation: {r4.previous_score} -> {r4.new_score}")
    assert r4.new_score == 79, f"Expected 79, got {r4.new_score}"

    print()
    print("Spec's exact worked example reproduced correctly: 100 -> 85 -> 87 -> 89 -> 79")
    print()

    # --- Cumulative accumulation across separate calls (3h + 3h = 6h -> +2) ---
    # Start below the max score first, otherwise the +2 award would be
    # invisible behind the 100-point cap and prove nothing.
    v2 = db.get_or_create_vehicle(f"CUMULATIVE-TEST-{int(time.time())}")
    engine.record_violation(v2.vehicle_id, "distraction", 0.75)  # 100 -> 92, headroom to recover into
    step1 = engine.add_monitored_time(v2.vehicle_id, 3 * ONE_HOUR)
    print(f"[OK] 3h clean driving: score unchanged at {step1.new_score}, "
          f"clean_seconds now {step1.new_clean_seconds} (no interval completed yet)")
    assert step1.points_awarded == 0
    assert step1.new_clean_seconds == 3 * ONE_HOUR

    step2 = engine.add_monitored_time(v2.vehicle_id, 3 * ONE_HOUR)
    print(f"[OK] Another 3h clean driving: score {step2.previous_score} -> "
          f"{step2.new_score} (3h + 3h = 6h cumulative -> +2)")
    assert step2.points_awarded == 2
    assert step2.new_score == step1.new_score + 2

    # --- A violation resets the CURRENT (unrewarded) clean counter ---
    v3 = db.get_or_create_vehicle(f"RESET-TEST-{int(time.time())}")
    engine.add_monitored_time(v3.vehicle_id, 4 * ONE_HOUR)  # 4h banked, no award yet
    before_violation = db.get_vehicle_by_id(v3.vehicle_id)
    assert before_violation.clean_seconds_accumulated == 4 * ONE_HOUR
    engine.record_violation(v3.vehicle_id, "distraction", 0.80)
    after_violation = db.get_vehicle_by_id(v3.vehicle_id)
    print(f"[OK] Violation reset clean_seconds_accumulated: "
          f"{before_violation.clean_seconds_accumulated}s -> {after_violation.clean_seconds_accumulated}s")
    assert after_violation.clean_seconds_accumulated == 0

    # --- Score never exceeds 100 (cap check) ---
    v4 = db.get_or_create_vehicle(f"CAP-TEST-{int(time.time())}")
    # v4 starts at 100 already - award recovery anyway and confirm it's a no-op on the score
    capped = engine.add_monitored_time(v4.vehicle_id, 12 * ONE_HOUR)
    print(f"[OK] Recovery at already-max score: {capped.previous_score} -> {capped.new_score} "
          f"(capped, never exceeds 100)")
    assert capped.new_score == 100

    # --- Pure function check (no DB) matching 6h/12h/18h -> +2/+4/+6 from the spec ---
    for hours, expected_points in [(6, 2), (12, 4), (18, 6)]:
        intervals, remainder = calculate_recovery_intervals(0, hours * ONE_HOUR, 6 * ONE_HOUR)
        points = intervals * 2
        assert points == expected_points, f"{hours}h should give +{expected_points}, got +{points}"
        assert remainder == 0
    print("[OK] Pure interval math matches spec table: 6h=+2, 12h=+4, 18h=+6")

    db.close()

    print("=" * 60)
    print("All recovery checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
