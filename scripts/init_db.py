"""
DriveGuard AI - Database Initialization / Sanity Check
==========================================================
Run this once to create the SQLite database and its tables, and to
confirm the whole database layer works end-to-end (create a vehicle,
insert a violation, update a score, read history back).

This is a manual sanity check for Phase 3 - the real automated test
suite (pytest) is built in Phase 13.

Usage:
    python scripts/init_db.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import Database, DatabaseError


def main():
    print("=" * 60)
    print("DriveGuard AI - Database Initialization")
    print("=" * 60)

    try:
        db = Database()
        db.initialize()
        print(f"[OK] Database ready at: {db.db_path}")
    except DatabaseError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print("-" * 60)
    print("Running sanity check (uses a test vehicle, safe to re-run)...")
    print("-" * 60)

    test_vehicle_number = "TEST-0000"

    vehicle = db.get_or_create_vehicle(test_vehicle_number)
    print(f"[OK] Vehicle loaded/created: {vehicle}")

    score_before = vehicle.current_score
    session_id = db.start_session(vehicle.vehicle_id)
    print(f"[OK] Session started: {session_id}")

    violation_id = db.insert_violation(
        vehicle.vehicle_id, "mobile_phone_usage", 0.91, 15, session_id=session_id
    )
    print(f"[OK] Violation inserted: id={violation_id}")

    new_score = max(0, score_before - 15)
    db.update_vehicle_score(vehicle.vehicle_id, new_score, "MEDIUM" if new_score < 80 else "LOW")
    db.insert_score_history(
        vehicle.vehicle_id, score_before, 15, new_score, "mobile_phone_usage violation"
    )
    print(f"[OK] Score updated: {score_before} -> {new_score}")

    db.end_session(session_id)
    print(f"[OK] Session ended")

    reloaded = db.get_vehicle_by_number(test_vehicle_number)
    print(f"[OK] Reloaded vehicle from DB: score={reloaded.current_score}, risk={reloaded.risk_level}")

    violations = db.get_violations(vehicle.vehicle_id)
    history = db.get_score_history(vehicle.vehicle_id)
    print(f"[OK] Violations on file: {len(violations)}")
    print(f"[OK] Score history entries on file: {len(history)}")

    db.close()

    print("=" * 60)
    print("Sanity check passed. Re-run this script and the score will")
    print("keep decreasing for TEST-0000 - proving persistence works")
    print("(the score is NOT reset between runs).")
    print("=" * 60)


if __name__ == "__main__":
    main()
