"""
DriveGuard AI - Risk Classification
======================================
Turns a vehicle's numeric safety score into a LOW / MEDIUM / HIGH risk
label, using thresholds from config.yaml's `risk_thresholds` section
(spec section 9).

IMPORTANT: these thresholds are PROJECT-DEFINED POLICY for this
demonstration, not an official insurance industry standard. Every
place this shows up in the dashboard/reports (later phases) must make
that clear to whoever's reading it.

Default policy (from config.yaml, all configurable):
    score >= 80              -> LOW
    50 <= score < 80          -> MEDIUM
    0  <= score < 50          -> HIGH

Usage:

    from scoring.risk_classifier import RiskClassifier
    classifier = RiskClassifier()
    level = classifier.classify(72)   # -> "MEDIUM"
"""

from typing import Optional

from utils.helpers import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


class RiskClassificationError(Exception):
    """Raised when risk_thresholds in config.yaml are missing or
    internally inconsistent (e.g. low_risk_min <= medium_risk_min)."""
    pass


class RiskClassifier:
    def __init__(self, config: Optional[dict] = None):
        self.config = config if config is not None else load_config()
        thresholds = self.config.get("risk_thresholds", {})

        if not thresholds:
            raise RiskClassificationError(
                "No 'risk_thresholds' section found in config.yaml - "
                "cannot classify risk without it."
            )

        try:
            self.low_risk_min = int(thresholds["low_risk_min"])
            self.medium_risk_min = int(thresholds["medium_risk_min"])
            self.high_risk_min = int(thresholds["high_risk_min"])
        except KeyError as e:
            raise RiskClassificationError(
                f"config.yaml risk_thresholds is missing required key: {e}"
            )
        except (TypeError, ValueError) as e:
            raise RiskClassificationError(
                f"config.yaml risk_thresholds values must be integers: {e}"
            )

        if not (self.low_risk_min > self.medium_risk_min > self.high_risk_min >= 0):
            raise RiskClassificationError(
                "config.yaml risk_thresholds must satisfy "
                "low_risk_min > medium_risk_min > high_risk_min >= 0 "
                f"(got low={self.low_risk_min}, medium={self.medium_risk_min}, "
                f"high={self.high_risk_min})"
            )

    def classify(self, score: int) -> str:
        """Return 'LOW', 'MEDIUM', or 'HIGH' for a given safety score."""
        if score is None:
            raise RiskClassificationError("score cannot be None")
        if not (0 <= score <= 100):
            raise RiskClassificationError(f"score must be between 0 and 100 (got {score})")

        if score >= self.low_risk_min:
            return "LOW"
        elif score >= self.medium_risk_min:
            return "MEDIUM"
        else:
            return "HIGH"

    def get_thresholds(self) -> dict:
        """Expose the active thresholds - used by the dashboard/reports
        (later phases) to show the policy alongside the classification,
        e.g. 'MEDIUM risk (score 50-79)'."""
        return {
            "low_risk_min": self.low_risk_min,
            "medium_risk_min": self.medium_risk_min,
            "high_risk_min": self.high_risk_min,
        }

    def describe_band(self, level: str) -> str:
        """Human-readable score range for a risk level, e.g. for
        display: 'MEDIUM (50-79)'."""
        if level == "LOW":
            return f"LOW ({self.low_risk_min}-100)"
        elif level == "MEDIUM":
            return f"MEDIUM ({self.medium_risk_min}-{self.low_risk_min - 1})"
        elif level == "HIGH":
            return f"HIGH ({self.high_risk_min}-{self.medium_risk_min - 1})"
        else:
            raise RiskClassificationError(f"Unknown risk level: {level}")


# ------------------------------------------------------------
# Pure function version (no config dependency) - for Phase 13 unit
# tests and any quick one-off use that doesn't need a full engine.
# ------------------------------------------------------------
def classify_risk(
    score: int, low_risk_min: int = 80, medium_risk_min: int = 50
) -> str:
    if score >= low_risk_min:
        return "LOW"
    elif score >= medium_risk_min:
        return "MEDIUM"
    else:
        return "HIGH"
