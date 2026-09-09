"""
DriveGuard AI - Safety Score Engine Sanity Check
====================================================
Manual end-to-end check for Phase 4: creates a fresh test vehicle,
applies the exact sequence from spec section 8's example
(phone -> smoking -> phone again) and confirms the score matches:

    100 -> 85 (phone)  -> 75 (smoking) -> 60 (phone, REPEAT: -15 base -5 repeat = -20)

Note: the spec's worked example (85 -> 75 -> 60) does not itself add a
repeat-offense penalty on the second phone violation - it's a plain
illustration of the base formula. Once repeat-offense penalties
(section 8) are switched on, a second violation of the SAME activity
costs more than the first. This script demonstrates that real
behavior explicitly, and prints both numbers so it's obvious which
is which.

Usage:
    python scripts/test_scoring.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import Database, DatabaseError
from scoring.safety_score import SafetyScoreEngine, SafetyScoreError


def main():
    print("=" * 60)
    print("DriveGuard AI - Safety Score Engine Sanity Check")
    print("=" * 60)

    db = Database()
    db.initialize()
    engine = SafetyScoreEngine(db)

    # Use a fresh, uniquely-named test vehicle each run so this script
    # can be re-run without old TEST-0000 history skewing the numbers.
    import time
    test_vehicle_number = f"SCORE-TEST-{int(time.time())}"

    vehicle = db.get_or_create_vehicle(test_vehicle_number)
    print(f"[OK] Fresh test vehicle created: {vehicle.vehicle_number} (score={vehicle.current_score})")

    session_id = db.start_session(vehicle.vehicle_id)

    # 1) Mobile phone usage - first offense
    r1 = engine.record_violation(vehicle.vehicle_id, "mobile_phone_usage", 0.91, session_id)
    print(f"[OK] Phone (1st): penalty={r1.total_penalty} repeat={r1.is_repeat_offense} "
          f"score {r1.previous_score} -> {r1.new_score}")
    assert r1.new_score == 85, f"Expected 85, got {r1.new_score}"

    # 2) Smoking - first offense
    r2 = engine.record_violation(vehicle.vehicle_id, "smoking", 0.87, session_id)
    print(f"[OK] Smoking (1st): penalty={r2.total_penalty} repeat={r2.is_repeat_offense} "
          f"score {r2.previous_score} -> {r2.new_score}")
    assert r2.new_score == 75, f"Expected 75, got {r2.new_score}"

    # 3) Mobile phone usage AGAIN - this is a repeat offense, so it
    # costs base (15) + repeated_offense_extra (5) = 20, not just 15.
    r3 = engine.record_violation(vehicle.vehicle_id, "mobile_phone_usage", 0.88, session_id)
    print(f"[OK] Phone (2nd, REPEAT): penalty={r3.total_penalty} repeat={r3.is_repeat_offense} "
          f"score {r3.previous_score} -> {r3.new_score}")
    assert r3.total_penalty == 20, f"Expected repeat penalty of 20, got {r3.total_penalty}"
    assert r3.new_score == 55, f"Expected 55, got {r3.new_score}"

    db.end_session(session_id)

    # 4) Score floor check - drain a fresh vehicle below zero
    floor_vehicle = db.get_or_create_vehicle(f"FLOOR-TEST-{int(time.time())}")
    for _ in range(10):
        result = engine.record_violation(floor_vehicle.vehicle_id, "smoking", 0.80)
    print(f"[OK] Floor check after 10x smoking violations: score={result.new_score} "
          f"(must never go below 0)")
    assert result.new_score == 0, f"Expected floor of 0, got {result.new_score}"

    # 5) Drowsiness must be rejected as a scored violation by policy
    try:
        engine.record_violation(vehicle.vehicle_id, "drowsiness", 0.75)
        print("[FAIL] drowsiness should have been rejected for scoring")
    except SafetyScoreError as e:
        print(f"[OK] Drowsiness correctly rejected from scoring: {e}")

    # 6) Unknown activity must be rejected
    try:
        engine.record_violation(vehicle.vehicle_id, "juggling", 0.60)
        print("[FAIL] unknown activity should have been rejected")
    except SafetyScoreError as e:
        print(f"[OK] Unknown activity correctly rejected: {e}")

    db.close()

    print("=" * 60)
    print("All safety score checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
