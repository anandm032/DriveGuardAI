"""
DriveGuard AI - Main Application Launcher
==========================================
Phase 7.4: Real runnable monitoring application.

Run:
    python app.py
"""

import sys

from database.database import Database, DatabaseError
from detection.detector import Detector, DetectorError
from detection.monitoring_session import MonitoringSession
from detection.video_source import VideoSourceError
from detection.violation_confirmation import ViolationTracker
from scoring.safety_score import SafetyScoreEngine, SafetyScoreError
from utils.helpers import ConfigError, load_config
from utils.logger import get_logger


def print_header():
    print("=" * 60)
    print("DriveGuard AI")
    print("Intelligent Driver Behavior Monitoring & Insurance Risk Scoring")
    print("=" * 60)


def get_vehicle_number():
    while True:
        vehicle_number = input("Enter vehicle registration number: ").strip()

        if vehicle_number:
            return vehicle_number

        print("[ERROR] Vehicle registration number cannot be empty.")


def get_video_source():
    print()
    print("Video Source")
    print("------------")
    print("Enter 0 for default webcam.")
    print("Or enter the path to a video file.")

    while True:
        source = input("Video source [0 for webcam]: ").strip()

        if not source:
            source = "0"

        if source == "0":
            return 0

        return source


def print_result(result, vehicle):
    print()
    print("=" * 60)
    print("MONITORING SESSION COMPLETE")
    print("=" * 60)
    print(f"Vehicle       : {vehicle.vehicle_number}")
    print(f"Session ID    : {result.session_id}")
    print(f"Frames        : {result.frames_processed}")
    print(f"Violations    : {len(result.confirmed_violations)}")
    print(f"Final Score   : {result.final_score}/100")
    print(f"Risk Level    : {result.final_risk}")
    print(f"Elapsed Time  : {result.elapsed_seconds:.2f} seconds")
    print(f"Stopped By    : {result.stopped_reason}")

    if result.confirmed_violations:
        print()
        print("Confirmed Violations")
        print("--------------------")

        for index, violation in enumerate(result.confirmed_violations, start=1):
            print(
                f"{index}. {violation.activity} "
                f"(confidence={violation.confidence:.2f}, "
                f"frame={violation.frame_number}) "
                f"-> score={violation.new_score}, "
                f"risk={violation.new_risk_level}"
            )

    print("=" * 60)


def main():
    print_header()

    db = None

    try:
        # ---------------------------------------------------------
        # 1. Load configuration
        # ---------------------------------------------------------
        config = load_config()

        # ---------------------------------------------------------
        # 2. Initialize database
        # ---------------------------------------------------------
        db = Database()

        # ---------------------------------------------------------
        # 3. Get vehicle
        # ---------------------------------------------------------
        vehicle_number = get_vehicle_number()
        vehicle = db.get_or_create_vehicle(vehicle_number)

        print()
        print(f"Vehicle loaded : {vehicle.vehicle_number}")
        print(f"Current score  : {vehicle.current_score}/100")
        print(f"Current risk   : {vehicle.risk_level}")

        # ---------------------------------------------------------
        # 4. Get video source
        # ---------------------------------------------------------
        source = get_video_source()

        print()
        print("Initializing monitoring system...")

        # ---------------------------------------------------------
        # 5. Create real application components
        # ---------------------------------------------------------
        detector = Detector(config=config)
        tracker = ViolationTracker(config=config)
        engine = SafetyScoreEngine(db=db, config=config)

        # ---------------------------------------------------------
        # 6. Create monitoring session
        # ---------------------------------------------------------
        monitoring = MonitoringSession(
            vehicle_id=vehicle.vehicle_id,
            source=source,
            db=db,
            detector=detector,
            tracker=tracker,
            engine=engine,
            config=config,
        )

        print("Monitoring started.")
        print("Press Ctrl+C to stop.")
        print()

        # ---------------------------------------------------------
        # 7. Run complete detection -> confirmation -> scoring loop
        # ---------------------------------------------------------
        result = monitoring.run()

        # ---------------------------------------------------------
        # 8. Display final result
        # ---------------------------------------------------------
        print_result(result, vehicle)

    except ConfigError as e:
        print(f"[ERROR] Configuration error: {e}")
        return 1

    except DatabaseError as e:
        print(f"[ERROR] Database error: {e}")
        return 1

    except DetectorError as e:
        print(f"[ERROR] Detector error: {e}")
        return 1

    except VideoSourceError as e:
        print(f"[ERROR] Video source error: {e}")
        return 1

    except SafetyScoreError as e:
        print(f"[ERROR] Safety scoring error: {e}")
        return 1

    except KeyboardInterrupt:
        print()
        print("[INFO] Monitoring interrupted by user.")
        return 0

    except Exception as e:
        print(f"[ERROR] Unexpected application error: {e}")
        return 1

    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    sys.exit(main())
