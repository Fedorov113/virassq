"""Named numeric thresholds used by peak decision rules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PeakDecisionThresholds:
    """Numeric thresholds stored beside every resulting peak decision."""

    max_spanning_target_count: int = 0
    min_end_start_target_count: int = 3
    min_directional_or_overlap_b_score: int = 2
    min_dominant_annotation_fraction: float = 0.8
    min_alternative_union_coverage_fraction: float = 0.4
    max_alternative_gap_nt: int = 300
    min_one_sided_score: int = 10
    min_unsupported_extension_nt: int = 200


DEFAULT_PEAK_DECISION_THRESHOLDS = PeakDecisionThresholds()


def validate_peak_decision_thresholds(
    thresholds: PeakDecisionThresholds,
) -> None:
    """Reject values that cannot express the documented rules."""

    if thresholds.max_spanning_target_count < 0:
        raise ValueError("max_spanning_target_count must be >= 0")
    if thresholds.min_end_start_target_count < 1:
        raise ValueError("min_end_start_target_count must be >= 1")
    if thresholds.min_directional_or_overlap_b_score < 1:
        raise ValueError("min_directional_or_overlap_b_score must be >= 1")
    if not 0 < thresholds.min_dominant_annotation_fraction <= 1:
        raise ValueError("min_dominant_annotation_fraction must be in (0, 1]")
    if not 0 < thresholds.min_alternative_union_coverage_fraction <= 1:
        raise ValueError(
            "min_alternative_union_coverage_fraction must be in (0, 1]"
        )
    if thresholds.max_alternative_gap_nt < 0:
        raise ValueError("max_alternative_gap_nt must be >= 0")
    if thresholds.min_one_sided_score < 1:
        raise ValueError("min_one_sided_score must be >= 1")
    if thresholds.min_unsupported_extension_nt < 1:
        raise ValueError("min_unsupported_extension_nt must be >= 1")
