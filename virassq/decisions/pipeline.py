r"""Build and write peak-based composite decisions.

This is the orchestration layer for already measured tables::

    peak descriptions       alternative layouts       exact Q-T-R links
             \                    |                    /
              `----------- measured values ----------'
                                  |
                                  v
                       independent rule flags
                                  |
                                  v
                     one action per selected peak
                                  |
                                  v
                   one action per representative

``review`` keeps a representative in the mapping FASTA. ``quarantine``
excludes the complete sequence until it is replaced or manually accepted.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from segonaut.reference.composite.decisions.one_sided_pairs import (
    LINK_COLUMNS,
    find_reciprocal_one_sided_pairs,
)
from segonaut.reference.composite.decisions.peak_rules import (
    LAYOUT_COLUMNS,
    add_diamond_annotation_difference_flags,
    add_peak_rule_flags,
    add_threshold_columns,
    combine_peak_rule_results,
    mark_nearby_two_sided_alternative_layouts,
    summarize_nearby_alternative_layouts,
)
from segonaut.reference.composite.decisions.thresholds import (
    DEFAULT_PEAK_DECISION_THRESHOLDS,
    PeakDecisionThresholds,
    validate_peak_decision_thresholds,
)

DESCRIPTION_COLUMNS = {
    "peak_id",
    "query_id",
    "query_length",
    "peak_position",
    "score_geometry",
    "peak_score",
    "end_target_count",
    "start_target_count",
    "directional_b_score",
    "overlap_b_score",
    "spanning_target_count",
    "end_start_best_domain_relation",
    "end_dominant_best_domain_target_count",
    "end_dominant_best_domain_fraction_among_nonmissing",
    "start_dominant_best_domain_target_count",
    "start_dominant_best_domain_fraction_among_nonmissing",
    "end_start_best_viral_family_relation",
    "end_dominant_best_viral_family_target_count",
    "end_dominant_best_viral_family_fraction_among_nonmissing",
    "start_dominant_best_viral_family_target_count",
    "start_dominant_best_viral_family_fraction_among_nonmissing",
    "end_start_viral_protein_role_relation",
    "end_dominant_viral_protein_role_target_count",
    "end_dominant_viral_protein_role_fraction_among_nonmissing",
    "start_dominant_viral_protein_role_target_count",
    "start_dominant_viral_protein_role_fraction_among_nonmissing",
}


def require_columns(
    table: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    """Fail with the concrete missing columns before any scientific rule runs."""

    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{table_name} are missing columns: {missing}")


def build_peak_decisions(
    descriptions: pd.DataFrame,
    layouts: pd.DataFrame,
    links: pd.DataFrame,
    thresholds: PeakDecisionThresholds,
) -> pd.DataFrame:
    """Build one auditable action row for every selected score peak."""

    require_columns(descriptions, DESCRIPTION_COLUMNS, "peak descriptions")
    require_columns(layouts, LAYOUT_COLUMNS, "alternative layouts")
    require_columns(links, LINK_COLUMNS, "alternative links")
    if descriptions["peak_id"].duplicated().any():
        raise ValueError("peak descriptions must contain one row per peak_id")

    marked_layouts = mark_nearby_two_sided_alternative_layouts(layouts, thresholds)
    alternatives = summarize_nearby_alternative_layouts(marked_layouts)
    reciprocal_pairs = find_reciprocal_one_sided_pairs(descriptions, links, thresholds)
    peaks = descriptions.merge(
        alternatives, on="peak_id", how="left", validate="one_to_one"
    ).merge(reciprocal_pairs, on="peak_id", how="left", validate="one_to_one")

    peaks["nearby_two_sided_alternative_count"] = (
        peaks["nearby_two_sided_alternative_count"].fillna(0).astype("int64")
    )
    peaks["reciprocal_one_sided_pair_count"] = (
        peaks["reciprocal_one_sided_pair_count"].fillna(0).astype("int64")
    )
    peaks["maximum_reciprocal_shared_target_count"] = (
        peaks["maximum_reciprocal_shared_target_count"].fillna(0).astype("int64")
    )
    for column in (
        "reciprocal_one_sided_peak_ids",
        "reciprocal_one_sided_query_ids",
    ):
        peaks[column] = peaks[column].apply(
            lambda value: value if isinstance(value, list) else []
        )

    add_diamond_annotation_difference_flags(peaks, thresholds)
    add_peak_rule_flags(peaks, thresholds)
    add_threshold_columns(peaks, thresholds)
    if peaks.empty:
        peaks["matched_quarantine_rules"] = pd.Series(dtype="object")
        peaks["matched_quarantine_blockers"] = pd.Series(dtype="object")
        peaks["matched_review_rules"] = pd.Series(dtype="object")
        peaks["peak_action"] = pd.Series(dtype="string")
        peaks["decision_reason"] = pd.Series(dtype="string")
        return peaks

    assigned = pd.DataFrame(
        [combine_peak_rule_results(row) for _, row in peaks.iterrows()],
        index=peaks.index,
    )
    peaks[assigned.columns] = assigned
    return peaks.sort_values(
        ["query_id", "peak_position", "score_geometry", "peak_id"]
    ).reset_index(drop=True)


def unique_rule_names(values: pd.Series) -> list[str]:
    """Collect sorted rule names from list-valued peak columns."""

    return sorted({name for names in values for name in names})


def build_representative_actions(peak_decisions: pd.DataFrame) -> pd.DataFrame:
    """Collapse peak actions; any quarantine peak excludes the representative."""

    rows = []
    for query_id, peaks in peak_decisions.groupby("query_id", sort=True):
        quarantine = peaks["peak_action"].eq("quarantine")
        action = "quarantine" if quarantine.any() else "review"
        rows.append(
            {
                "query_id": str(query_id),
                "peak_count": len(peaks),
                "quarantine_peak_count": int(quarantine.sum()),
                "review_peak_count": int((~quarantine).sum()),
                "representative_action": action,
                "action_peak_ids": peaks.loc[
                    peaks["peak_action"].eq(action), "peak_id"
                ].tolist(),
                "matched_quarantine_rules": unique_rule_names(
                    peaks["matched_quarantine_rules"]
                ),
                "matched_quarantine_blockers": unique_rule_names(
                    peaks["matched_quarantine_blockers"]
                ),
                "matched_review_rules": unique_rule_names(
                    peaks["matched_review_rules"]
                ),
            }
        )
    columns = [
        "query_id",
        "peak_count",
        "quarantine_peak_count",
        "review_peak_count",
        "representative_action",
        "action_peak_ids",
        "matched_quarantine_rules",
        "matched_quarantine_blockers",
        "matched_review_rules",
    ]
    return pd.DataFrame(rows, columns=columns)


def count_rule_matches(values: pd.Series) -> dict[str, int]:
    """Count peaks matching each named rule in a list-valued column."""

    exploded = values.explode().dropna()
    return {
        str(name): int(count)
        for name, count in exploded.value_counts().sort_index().items()
    }


def decide_peak_actions(
    peak_descriptions_path: Path,
    alternative_layouts_path: Path,
    alternative_links_path: Path,
    output_directory: Path,
    *,
    thresholds: PeakDecisionThresholds = DEFAULT_PEAK_DECISION_THRESHOLDS,
    force: bool = False,
) -> dict[str, object]:
    """Write peak actions, representative actions and quarantine query IDs."""

    validate_peak_decision_thresholds(thresholds)
    for name, path in (
        ("peak descriptions", peak_descriptions_path),
        ("alternative layouts", alternative_layouts_path),
        ("alternative links", alternative_links_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} do not exist: {path}")

    peak_output = output_directory / "peak_decisions.parquet"
    representative_output = output_directory / "representative_actions.parquet"
    quarantine_ids_output = output_directory / "quarantine_query_ids.txt"
    existing = [
        path
        for path in (peak_output, representative_output, quarantine_ids_output)
        if path.exists()
    ]
    if existing and not force:
        raise FileExistsError(f"decision outputs already exist: {existing}")

    descriptions = pd.read_parquet(peak_descriptions_path)
    layouts = pd.read_parquet(
        alternative_layouts_path,
        columns=sorted(LAYOUT_COLUMNS),
    )
    links = pd.read_parquet(alternative_links_path, columns=sorted(LINK_COLUMNS))
    peak_decisions = build_peak_decisions(descriptions, layouts, links, thresholds)
    representative_actions = build_representative_actions(peak_decisions)

    output_directory.mkdir(parents=True, exist_ok=True)
    peak_decisions.to_parquet(peak_output, index=False)
    representative_actions.to_parquet(representative_output, index=False)
    quarantine_ids = representative_actions.loc[
        representative_actions["representative_action"].eq("quarantine"),
        "query_id",
    ].sort_values()
    quarantine_ids_output.write_text(
        "".join(f"{query_id}\n" for query_id in quarantine_ids),
        encoding="utf-8",
    )

    result = {
        "peak_decisions": str(peak_output),
        "representative_actions": str(representative_output),
        "quarantine_query_ids": str(quarantine_ids_output),
        "thresholds": asdict(thresholds),
        "peak_count": len(peak_decisions),
        "peak_action_counts": {
            str(action): int(count)
            for action, count in peak_decisions["peak_action"].value_counts().items()
        },
        "quarantine_rule_match_counts": count_rule_matches(
            peak_decisions["matched_quarantine_rules"]
        ),
        "quarantine_blocker_match_counts": count_rule_matches(
            peak_decisions["matched_quarantine_blockers"]
        ),
        "review_rule_match_counts": count_rule_matches(
            peak_decisions["matched_review_rules"]
        ),
        "representative_count": len(representative_actions),
        "representative_action_counts": {
            str(action): int(count)
            for action, count in representative_actions["representative_action"]
            .value_counts()
            .items()
        },
    }
    (output_directory / "peak_decisions.summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
