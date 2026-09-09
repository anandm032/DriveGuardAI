-- ==================================================
-- DriveGuard AI - Database Schema
-- ==================================================
-- SQLite database. Run via database/database.py's initialize()
-- method (or scripts/init_db.py) rather than by hand, but this file
-- is the single source of truth for the schema.

PRAGMA foreign_keys = ON;

-- --------------------------------------------------
-- vehicles
-- One row per physical vehicle. current_score and risk_level persist
-- across app restarts - they are NEVER reset on startup.
-- --------------------------------------------------
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_number  TEXT NOT NULL UNIQUE,          -- e.g. "KL-XX-1234", entered by the user
    current_score   INTEGER NOT NULL DEFAULT 100,
    risk_level      TEXT NOT NULL DEFAULT 'LOW',   -- 'LOW' | 'MEDIUM' | 'HIGH'
    -- Seconds of monitored clean (violation-free) driving accumulated
    -- since the last recovery award or the last violation, whichever
    -- was more recent. Resets to 0 on a confirmed violation, and
    -- resets to a remainder (not necessarily 0) whenever it crosses
    -- a full recovery interval and a +N award is granted. Persists
    -- across sessions/app restarts just like current_score.
    clean_seconds_accumulated INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (current_score >= 0),
    CHECK (clean_seconds_accumulated >= 0),
    CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH'))
);

-- --------------------------------------------------
-- sessions
-- One row per monitoring session (roughly: one "run" of the app for
-- a given vehicle). Violations reference the session they occurred in.
-- --------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id  INTEGER NOT NULL,
    start_time  TEXT NOT NULL DEFAULT (datetime('now')),
    end_time    TEXT,                              -- NULL while the session is still active
    FOREIGN KEY (vehicle_id) REFERENCES vehicles (vehicle_id) ON DELETE CASCADE
);

-- --------------------------------------------------
-- violations
-- One row per CONFIRMED violation (after temporal confirmation -
-- Phase 8). Raw per-frame detections are never stored here.
-- --------------------------------------------------
CREATE TABLE IF NOT EXISTS violations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id  INTEGER NOT NULL,
    session_id  INTEGER,                           -- nullable: allows manual/test inserts without an active session
    activity    TEXT NOT NULL,                      -- e.g. 'mobile_phone_usage', 'smoking'
    confidence  REAL NOT NULL,
    penalty     INTEGER NOT NULL,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles (vehicle_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE SET NULL,
    CHECK (confidence >= 0 AND confidence <= 1),
    CHECK (penalty >= 0)
);

-- --------------------------------------------------
-- score_history
-- Audit trail of every score change for a vehicle - lets the
-- dashboard and reports show "how did we get to this score".
-- --------------------------------------------------
CREATE TABLE IF NOT EXISTS score_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id      INTEGER NOT NULL,
    previous_score  INTEGER NOT NULL,
    penalty         INTEGER NOT NULL,
    new_score       INTEGER NOT NULL,
    reason          TEXT NOT NULL,                  -- e.g. 'mobile_phone_usage violation'
    timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles (vehicle_id) ON DELETE CASCADE,
    CHECK (previous_score >= 0),
    CHECK (new_score >= 0)
);

-- --------------------------------------------------
-- Indexes for the lookups the dashboard/reports will do most often
-- --------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_violations_vehicle_id ON violations (vehicle_id);
CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations (timestamp);
CREATE INDEX IF NOT EXISTS idx_score_history_vehicle_id ON score_history (vehicle_id);
CREATE INDEX IF NOT EXISTS idx_sessions_vehicle_id ON sessions (vehicle_id);
