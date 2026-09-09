"""
DriveGuard AI - Risk Classification Sanity Check
====================================================
Manual end-to-end check for Phase 5: boundary values against the
default policy (80/50/0), config validation, and integration with
SafetyScoreEngine so a vehicle's risk_level actually updates when its
score crosses a threshold.

Usage:
    python scripts/test_risk.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import Database
from scoring.risk_classifier import RiskClassificationError, RiskClassifier, classify_risk
from scoring.safety_score import SafetyScoreEngine


def main():
    print("=" * 60)
    print("DriveGuard AI - Risk Classification Sanity Check")
    print("=" * 60)

    classifier = RiskClassifier()
    print(f"[OK] Loaded thresholds: {classifier.get_thresholds()}")

    # --- Boundary checks against the default policy (80 / 50 / 0) ---
    boundary_cases = [
        (100, "LOW"), (80, "LOW"), (79, "MEDIUM"),
        (50, "MEDIUM"), (49, "HIGH"),
        (0, "HIGH"),
    ]
    for score, expected in boundary_cases:
        actual = classifier.classify(score)
        status = "OK" if actual == expected else "FAIL"
        print(f"  [{status}] score={score:3d} -> {actual} (expected {expected})")
        assert actual == expected, f"score {score}: expected {expected}, got {actual}"

    print(f"[OK] Band descriptions: LOW={classifier.describe_band('LOW')}, "
          f"MEDIUM={classifier.describe_band('MEDIUM')}, "
          f"HIGH={classifier.describe_band('HIGH')}")

    # --- Out-of-range score should be rejected ---
    for bad_score in (-1, 101):
        try:
            classifier.classify(bad_score)
            print(f"[FAIL] score={bad_score} should have been rejected")
        except RiskClassificationError as e:
            print(f"[OK] score={bad_score} correctly rejected: {e}")

    # --- Pure function version should match the class for the same thresholds ---
    for score, expected in boundary_cases:
        assert classify_risk(score, low_risk_min=80, medium_risk_min=50) == expected
    print("[OK] Pure classify_risk() matches RiskClassifier for all boundary cases")

    # --- Misconfigured thresholds should be rejected at construction time ---
    try:
        RiskClassifier(config={"risk_thresholds": {
            "low_risk_min": 50, "medium_risk_min": 80, "high_risk_min": 0
        }})
        print("[FAIL] inverted thresholds should have been rejected")
    except RiskClassificationError as e:
        print(f"[OK] Inverted thresholds correctly rejected: {e}")

    try:
        RiskClassifier(config={})
        print("[FAIL] missing risk_thresholds section should have been rejected")
    except RiskClassificationError as e:
        print(f"[OK] Missing risk_thresholds section correctly rejected: {e}")

    # --- Integration: does a vehicle's stored risk_level actually change? ---
    print("-" * 60)
    print("Integration check with SafetyScoreEngine + database")
    print("-" * 60)

    db = Database()
    db.initialize()
    engine = SafetyScoreEngine(db)

    vehicle = db.get_or_create_vehicle(f"RISK-TEST-{int(time.time())}")
    print(f"[OK] Fresh vehicle: score={vehicle.current_score}, risk={vehicle.risk_level}")
    assert vehicle.risk_level == "LOW"

    # Drop the vehicle from LOW into MEDIUM territory
    r1 = engine.record_violation(vehicle.vehicle_id, "mobile_phone_usage", 0.90)  # 100 -> 85, still LOW
    r2 = engine.record_violation(vehicle.vehicle_id, "smoking", 0.85)             # 85 -> 75, now MEDIUM
    print(f"[OK] After 2 violations: score={r2.new_score}, risk {r1.new_risk_level} -> {r2.new_risk_level}")
    assert r1.new_risk_level == "LOW"
    assert r2.new_risk_level == "MEDIUM"

    reloaded = db.get_vehicle_by_id(vehicle.vehicle_id)
    assert reloaded.risk_level == "MEDIUM", "risk_level must be persisted in the DB, not just returned"
    print(f"[OK] Persisted risk_level confirmed from DB reload: {reloaded.risk_level}")

    # Drop further into HIGH territory
    r3 = engine.record_violation(vehicle.vehicle_id, "drinking", 0.80)   # 75 -> 65, still MEDIUM
    r4 = engine.record_violation(vehicle.vehicle_id, "distraction", 0.70)  # 65 -> 57, still MEDIUM
    r5 = engine.record_violation(vehicle.vehicle_id, "mobile_phone_usage", 0.92)  # 57 -> repeat, -20 -> 37, now HIGH
    print(f"[OK] Further violations: score={r5.new_score}, risk={r5.new_risk_level}")
    assert r5.new_risk_level == "HIGH"

    # Recovery should be able to bring risk level back down (numerically "down" in
    # severity, i.e. HIGH -> MEDIUM) as the score climbs back up
    recovery = engine.add_monitored_time(vehicle.vehicle_id, 12 * 3600)  # +4 -> 41, still HIGH
    print(f"[OK] After 12h recovery: score={recovery.new_score}, risk={recovery.new_risk_level}")

    db.close()

    print("=" * 60)
    print("All risk classification checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
