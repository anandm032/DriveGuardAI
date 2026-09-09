"""
DriveGuard AI - Database Access Layer
========================================
Every other module (scoring, detection, dashboard, reports) talks to
SQLite through this class - nobody else should write raw SQL. This
keeps query logic in one place and makes it possible to add caching,
swap databases, etc. later without touching the rest of the app.

Usage:

    from database.database import Database
    db = Database()
    db.initialize()                                  # creates tables if they don't exist
    vehicle = db.get_or_create_vehicle("KL-XX-1234")  # loads existing or creates new, score persists
    db.insert_violation(vehicle.vehicle_id, "mobile_phone_usage", 0.91, 15)
"""

import os
import sqlite3
from typing import List, Optional

from database.models import ScoreHistoryEntry, Session, Violation, Vehicle
from utils.helpers import load_config, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}


class DatabaseError(Exception):
    """Raised for any database problem the rest of the app should
    handle gracefully (missing vehicle, connection failure, etc.)
    instead of crashing on a raw sqlite3.Error."""
    pass


class Database:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            config = load_config()
            db_path = resolve_path(config["database"]["path"])
        self.db_path = db_path

        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        try:
            # check_same_thread=False: the Streamlit dashboard (Phase 11) and a
            # background detection loop (Phase 7) may touch the DB from
            # different threads. We serialize access ourselves where it matters.
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database at {self.db_path}: {e}")
            raise DatabaseError(f"Could not open database at {self.db_path}: {e}")

        logger.info(f"Connected to database at {self.db_path}")

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    def initialize(self):
        """Create all tables/indexes if they don't already exist.
        Safe to call every app startup - existing data is untouched."""
        if not os.path.exists(_SCHEMA_PATH):
            raise DatabaseError(f"Schema file not found at {_SCHEMA_PATH}")
        try:
            with open(_SCHEMA_PATH, "r") as f:
                schema_sql = f.read()
            self._conn.executescript(schema_sql)
            self._conn.commit()
            logger.info("Database schema initialized (tables verified/created)")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise DatabaseError(f"Failed to initialize database schema: {e}")

    def close(self):
        self._conn.close()
        logger.info("Database connection closed")

    # ----------------------------------------------------------------
    # Vehicles
    # ----------------------------------------------------------------
    def get_or_create_vehicle(self, vehicle_number: str) -> Vehicle:
        """Look up a vehicle by its registration number. If it exists,
        its persisted score/history is returned as-is (never reset).
        If it doesn't exist, a new vehicle starting at 100 points is
        created."""
        vehicle_number = self._validate_vehicle_number(vehicle_number)

        existing = self.get_vehicle_by_number(vehicle_number)
        if existing is not None:
            logger.info(
                f"Loaded existing vehicle {vehicle_number} "
                f"(score={existing.current_score}, risk={existing.risk_level})"
            )
            return existing

        try:
            cursor = self._conn.execute(
                "INSERT INTO vehicles (vehicle_number, current_score, risk_level) "
                "VALUES (?, 100, 'LOW')",
                (vehicle_number,),
            )
            self._conn.commit()
            logger.info(f"Created new vehicle {vehicle_number} with starting score 100")
        except sqlite3.Error as e:
            logger.error(f"Failed to create vehicle {vehicle_number}: {e}")
            raise DatabaseError(f"Failed to create vehicle '{vehicle_number}': {e}")

        return self.get_vehicle_by_id(cursor.lastrowid)

    def get_vehicle_by_number(self, vehicle_number: str) -> Optional[Vehicle]:
        row = self._conn.execute(
            "SELECT * FROM vehicles WHERE vehicle_number = ?", (vehicle_number,)
        ).fetchone()
        return Vehicle.from_row(row) if row else None

    def get_vehicle_by_id(self, vehicle_id: int) -> Optional[Vehicle]:
        row = self._conn.execute(
            "SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,)
        ).fetchone()
        return Vehicle.from_row(row) if row else None

    def update_vehicle_score(self, vehicle_id: int, new_score: int, risk_level: str):
        """Persist a vehicle's new score and risk level. Called by the
        scoring engine (Phase 4) after a violation is confirmed - this
        method itself does not compute anything, only stores the
        result."""
        if new_score < 0:
            raise DatabaseError(f"new_score cannot be negative (got {new_score})")
        if risk_level not in VALID_RISK_LEVELS:
            raise DatabaseError(
                f"risk_level must be one of {VALID_RISK_LEVELS} (got '{risk_level}')"
            )
        if self.get_vehicle_by_id(vehicle_id) is None:
            raise DatabaseError(f"Cannot update score: vehicle_id {vehicle_id} does not exist")

        try:
            self._conn.execute(
                "UPDATE vehicles SET current_score = ?, risk_level = ?, "
                "updated_at = datetime('now') WHERE vehicle_id = ?",
                (new_score, risk_level, vehicle_id),
            )
            self._conn.commit()
            logger.info(
                f"Vehicle {vehicle_id} score updated to {new_score} ({risk_level})"
            )
        except sqlite3.Error as e:
            logger.error(f"Failed to update score for vehicle {vehicle_id}: {e}")
            raise DatabaseError(f"Failed to update vehicle score: {e}")

    @staticmethod
    def _validate_vehicle_number(vehicle_number: str) -> str:
        if vehicle_number is None or not str(vehicle_number).strip():
            raise DatabaseError("Vehicle ID / registration number cannot be empty")
        return str(vehicle_number).strip().upper()

    # ----------------------------------------------------------------
    # Sessions
    # ----------------------------------------------------------------
    def start_session(self, vehicle_id: int) -> int:
        if self.get_vehicle_by_id(vehicle_id) is None:
            raise DatabaseError(f"Cannot start session: vehicle_id {vehicle_id} does not exist")
        try:
            cursor = self._conn.execute(
                "INSERT INTO sessions (vehicle_id) VALUES (?)", (vehicle_id,)
            )
            self._conn.commit()
            logger.info(f"Started session {cursor.lastrowid} for vehicle {vehicle_id}")
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to start session for vehicle {vehicle_id}: {e}")
            raise DatabaseError(f"Failed to start session: {e}")

    def end_session(self, session_id: int):
        try:
            self._conn.execute(
                "UPDATE sessions SET end_time = datetime('now') WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()
            logger.info(f"Ended session {session_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            raise DatabaseError(f"Failed to end session: {e}")

    # ----------------------------------------------------------------
    # Violations
    # ----------------------------------------------------------------
    def insert_violation(
        self,
        vehicle_id: int,
        activity: str,
        confidence: float,
        penalty: int,
        session_id: Optional[int] = None,
    ) -> int:
        if self.get_vehicle_by_id(vehicle_id) is None:
            raise DatabaseError(f"Cannot insert violation: vehicle_id {vehicle_id} does not exist")
        if not (0.0 <= confidence <= 1.0):
            raise DatabaseError(f"confidence must be between 0 and 1 (got {confidence})")
        if penalty < 0:
            raise DatabaseError(f"penalty cannot be negative (got {penalty})")

        try:
            cursor = self._conn.execute(
                "INSERT INTO violations (vehicle_id, session_id, activity, confidence, penalty) "
                "VALUES (?, ?, ?, ?, ?)",
                (vehicle_id, session_id, activity, confidence, penalty),
            )
            self._conn.commit()
            logger.info(
                f"Violation recorded for vehicle {vehicle_id}: {activity} "
                f"(confidence={confidence:.2f}, penalty=-{penalty})"
            )
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to insert violation for vehicle {vehicle_id}: {e}")
            raise DatabaseError(f"Failed to record violation: {e}")

    def get_violations(self, vehicle_id: int, limit: Optional[int] = None) -> List[Violation]:
        query = "SELECT * FROM violations WHERE vehicle_id = ? ORDER BY timestamp DESC"
        params = [vehicle_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [Violation.from_row(r) for r in rows]

    def count_violations_by_activity(self, vehicle_id: int, activity: str) -> int:
        """Used by the scoring engine (Phase 4) to detect repeated
        offenses of the same activity type."""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM violations WHERE vehicle_id = ? AND activity = ?",
            (vehicle_id, activity),
        ).fetchone()
        return row["cnt"] if row else 0

    # ----------------------------------------------------------------
    # Score history
    # ----------------------------------------------------------------
    def insert_score_history(
        self, vehicle_id: int, previous_score: int, penalty: int, new_score: int, reason: str
    ) -> int:
        if self.get_vehicle_by_id(vehicle_id) is None:
            raise DatabaseError(f"Cannot insert score history: vehicle_id {vehicle_id} does not exist")

        try:
            cursor = self._conn.execute(
                "INSERT INTO score_history (vehicle_id, previous_score, penalty, new_score, reason) "
                "VALUES (?, ?, ?, ?, ?)",
                (vehicle_id, previous_score, penalty, new_score, reason),
            )
            self._conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to insert score history for vehicle {vehicle_id}: {e}")
            raise DatabaseError(f"Failed to record score history: {e}")

    def get_score_history(
        self, vehicle_id: int, limit: Optional[int] = None
    ) -> List[ScoreHistoryEntry]:
        query = "SELECT * FROM score_history WHERE vehicle_id = ? ORDER BY timestamp DESC"
        params = [vehicle_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [ScoreHistoryEntry.from_row(r) for r in rows]
