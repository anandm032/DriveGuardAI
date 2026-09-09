"""
DriveGuard AI - Database Models
==================================
Lightweight dataclasses representing rows from each table. These are
plain data holders (no query logic) - all querying lives in
database.py. Using dataclasses instead of raw sqlite3.Row tuples
gives every other module (scoring, dashboard, reports) autocomplete
and named-field access instead of magic index numbers.
"""

import sqlite3
from dataclasses import dataclass
from typing import Optional


@dataclass
class Vehicle:
    vehicle_id: int
    vehicle_number: str
    current_score: int
    risk_level: str
    clean_seconds_accumulated: int
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Vehicle":
        return cls(
            vehicle_id=row["vehicle_id"],
            vehicle_number=row["vehicle_number"],
            current_score=row["current_score"],
            risk_level=row["risk_level"],
            clean_seconds_accumulated=row["clean_seconds_accumulated"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass
class Session:
    session_id: int
    vehicle_id: int
    start_time: str
    end_time: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Session":
        return cls(
            session_id=row["session_id"],
            vehicle_id=row["vehicle_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
        )


@dataclass
class Violation:
    id: int
    vehicle_id: int
    session_id: Optional[int]
    activity: str
    confidence: float
    penalty: int
    timestamp: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Violation":
        return cls(
            id=row["id"],
            vehicle_id=row["vehicle_id"],
            session_id=row["session_id"],
            activity=row["activity"],
            confidence=row["confidence"],
            penalty=row["penalty"],
            timestamp=row["timestamp"],
        )


@dataclass
class ScoreHistoryEntry:
    id: int
    vehicle_id: int
    previous_score: int
    penalty: int
    new_score: int
    reason: str
    timestamp: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ScoreHistoryEntry":
        return cls(
            id=row["id"],
            vehicle_id=row["vehicle_id"],
            previous_score=row["previous_score"],
            penalty=row["penalty"],
            new_score=row["new_score"],
            reason=row["reason"],
            timestamp=row["timestamp"],
        )
