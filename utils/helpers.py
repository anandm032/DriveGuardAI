"""
DriveGuard AI - Shared Helpers
=================================
Right now this holds config loading, since every module (database,
detection, scoring, dashboard) needs the same settings from
config/config.yaml. More shared helpers get added here as later
phases need them (e.g. timestamp formatting, path resolution).

Usage:

    from utils.helpers import load_config
    config = load_config()
    threshold = config["detection"]["confidence_threshold"]
"""

import os

import yaml

# Resolve the project root regardless of which directory the caller
# was launched from (so `python app.py` and `streamlit run
# dashboard/dashboard.py` both find the same config file).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")

_cached_configs = {}  # keyed by absolute path, so different paths never share a cache slot


class ConfigError(Exception):
    """Raised when config.yaml is missing or malformed."""
    pass


def load_config(path: str = DEFAULT_CONFIG_PATH, force_reload: bool = False) -> dict:
    """Load and return config.yaml as a dict. Cached per-path after the
    first successful load so we don't re-read the file on every call -
    pass force_reload=True if you need to pick up a change made while
    the app is running.

    Raises ConfigError with a clear message if the file is missing or
    is not valid YAML, instead of letting a cryptic exception bubble
    up from deep inside some other module (section 19: invalid
    configuration must be handled gracefully with a clear message).
    """
    cache_key = os.path.abspath(path)

    if cache_key in _cached_configs and not force_reload:
        return _cached_configs[cache_key]

    if not os.path.exists(path):
        raise ConfigError(
            f"Configuration file not found at: {path}\n"
            f"Make sure you are running commands from the project root, "
            f"and that config/config.yaml exists."
        )

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"config.yaml is not valid YAML: {e}")

    if not isinstance(config, dict):
        raise ConfigError(
            "config.yaml did not parse into a dictionary of settings. "
            "Check the file for formatting mistakes."
        )

    _required_top_level_keys = [
        "model", "detection", "penalties", "risk_thresholds", "database", "app",
    ]
    missing = [k for k in _required_top_level_keys if k not in config]
    if missing:
        raise ConfigError(
            f"config.yaml is missing required section(s): {', '.join(missing)}"
        )

    _cached_configs[cache_key] = config
    return config


def resolve_path(relative_path: str) -> str:
    """Turn a path from config.yaml (which is written relative to the
    project root, e.g. 'data/driveguard.db') into an absolute path, so
    it works no matter which folder a script is launched from."""
    return os.path.join(_PROJECT_ROOT, relative_path)
