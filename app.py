"""
DriveGuard AI - Main Entry Point
==================================
This file will grow into the real application launcher as we move
through the phases. Right now (Phase 2) it uses the shared config
loader and logger from utils/, and demonstrates the error handling
those helpers provide.

Run:
    python app.py
"""

import sys

from utils.helpers import ConfigError, load_config
from utils.logger import get_logger


def main():
    print("=" * 60)
    print("DriveGuard AI")
    print("Intelligent Driver Behavior Monitoring & Insurance Risk Scoring")
    print("=" * 60)

    try:
        config = load_config()
    except ConfigError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    logger = get_logger(__name__)
    logger.info("Application starting up")

    app_name = config.get("app", {}).get("name", "DriveGuard AI")
    logger.info(f"Loaded configuration for: {app_name}")

    print(f"Loaded configuration for: {app_name}")
    print(f"Model path (not yet loaded): {config['model']['path']}")
    print(f"Database path (not yet initialized): {config['database']['path']}")
    print()
    print("Phase 2 setup complete: shared config loading + logging are in place.")
    print("Database, detection, scoring, and the Streamlit dashboard are added")
    print("in the phases that follow.")
    print("Check logs/driveguard.log to see this run logged.")

    logger.info("Application shutdown (Phase 2 skeleton - nothing else to run yet)")


if __name__ == "__main__":
    main()
