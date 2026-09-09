"""
DriveGuard AI - Logging Setup
================================
Every module in this project (database, detection, scoring, dashboard,
reports) should get its logger from here instead of configuring
logging itself. This keeps log format, log level, and log file
location consistent across the whole app, and controlled from
config/config.yaml (see section 18 of the project spec: application
startup, model loading, camera startup, detections, violations, score
changes, DB errors, report generation, and shutdown should all be
logged).

Usage in any module:

    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from utils.helpers import load_config

_configured = False


def _configure_root_logger():
    """Set up the root logger exactly once per process, using
    settings from config.yaml. Safe to call multiple times — only
    does work the first time."""
    global _configured
    if _configured:
        return

    config = load_config()
    app_config = config.get("app", {})
    log_file = app_config.get("log_file", "logs/driveguard.log")
    log_level_name = app_config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if this somehow runs twice
    if root_logger.handlers:
        _configured = True
        return

    # Console handler - so you see logs live during development/demo
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler - so log files don't grow forever during
    # a long demo/testing session (5 MB per file, keep 3 backups)
    try:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except OSError as e:
        # Don't crash the whole app if the log file can't be created
        # (e.g. permissions issue) - just warn on console and continue.
        console_handler_only_msg = f"[WARN] Could not create log file at {log_file}: {e}"
        print(console_handler_only_msg)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger. Call this at the top of any module:
    logger = get_logger(__name__)
    """
    _configure_root_logger()
    return logging.getLogger(name)
