"""Public entry points for peak-based composite decisions."""

from virassq.decisions.pipeline import (
    build_peak_decisions,
    decide_peak_actions,
    prepare_peak_decision_table,
)
from virassq.decisions.thresholds import (
    DEFAULT_PEAK_DECISION_THRESHOLDS,
    PeakDecisionThresholds,
)

__all__ = [
    "DEFAULT_PEAK_DECISION_THRESHOLDS",
    "PeakDecisionThresholds",
    "build_peak_decisions",
    "decide_peak_actions",
    "prepare_peak_decision_table",
]
