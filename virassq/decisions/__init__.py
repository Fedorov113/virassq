"""Public entry points for peak-based composite decisions."""

from virassq.decisions.pipeline import decide_peak_actions
from virassq.decisions.thresholds import (
    DEFAULT_PEAK_DECISION_THRESHOLDS,
    PeakDecisionThresholds,
)

__all__ = [
    "DEFAULT_PEAK_DECISION_THRESHOLDS",
    "PeakDecisionThresholds",
    "decide_peak_actions",
]
