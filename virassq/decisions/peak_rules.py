r"""Evaluate independent rules for one selected boundary-score peak.

Rules only add named boolean columns. The final action is composed afterwards:

    any quarantine rule? -- no --------------------------> review
              |
             yes
              |
              v
     spanning target? ------- yes ------------------------> review
              |
              no
              |
              `--------------------------------------------> quarantine

This keeps every measured count, every matched rule and every blocker in the
output row instead of hiding the result in an ordered ``if`` chain.

Current v1 rules use the following measured arrangements. ``R`` is the tested
representative, ``T`` is a target contig, ``b`` is the selected peak position,
and ``r`` is the boundary radius (50 nt by default).

Rule 1: two target groups end and start around the same candidate boundary::

    R       1 [----------- part A -----------|----------- part B -----------] L
                                               b
    end T1      ==============================|
    end T2        ============================|
    end T3          ==========================|
    start T4                                    |============================
    start T5                                    |==========================
    start T6                                    |========================

    end_target_count >= 3
    start_target_count >= 3
    max(directional_b_score, overlap_b_score) >= 2

The dominant DIAMOND value on each side must occur on at least three target
contigs and in at least 80% of the non-missing values for that side. The rule
matches when the two dominant values differ in ``best_domain``,
``best_viral_family`` or ``viral_protein_role``. A target contig with no
DIAMOND hit contributes a missing value; absence is not treated as a distinct
annotation value.

Rule 2: target alignments repeatedly end before a suffix of the representative::

    R       1 [==============================|------ suffix >= 200 nt ------] L
    T1        ===============================|
    T2          =============================|
    ...
    Tn            ===========================|
                                               b

    one_sided_end_score = max(internal ends - internal starts, 0) >= 10

The suffix is not reproduced by the target contigs counted at the end peak;
the rule does not assert that no other target aligns anywhere in the suffix.

Rule 3 is the mirrored prefix arrangement::

    R       1 [------ prefix >= 200 nt ------|==============================] L
    T1                                        |==============================
    T2                                        |============================
    ...
    Tn                                        |==========================
                                               b

    one_sided_start_score = max(internal starts - internal ends, 0) >= 10

Rules 2 and 3 use only alignments against ``R``. They do not require another
representative.

For manual review, the output also records whether an end peak on ``R1`` and a
start peak on ``R2`` share at least three physical target-contig IDs::

    R1      [============== common region ==============|------ suffix X]
    T1-T3     ===========================================|

    R2      [prefix Y ------|============== common region ==============]
    T1-T3                  |=============================================

    R1 end peak -- same T1,T2,T3 --> R2 start peak
    R2 start peak -- same T1,T2,T3 --> R1 end peak

The legacy column name calls this a ``reciprocal_one_sided_pair``. It is a
shared-target join, not a reciprocal sequence-alignment test. This arrangement
does not cause or block quarantine because it can also describe fragmented or
alternative termini.

One blocker overrides every matching quarantine rule: an alignment row from
any target contig can cover the complete boundary window::

                              b-r       b       b+r
                               |--------|--------|
    spanning target      =================================

    spanning_target_count > 0                 -> review

The output additionally records whether the end and start target groups occur
near each other on another representative. The calculation accepts either a
linear gap or a gap across the ends of its FASTA representation::

    alternative R, linear
        [end-target ranges] -- gap <= 300 nt -- [start-target ranges]

    alternative R, across FASTA ends
        [start-target ranges] ................. [end-target ranges]
         ^                                                     ^
         `--------------- gap <= 300 nt across ends -----------'

This descriptor requires at least three end and three start target contigs,
consistent projected orientations, and union coverage of at least 40% of the
alternative representative. It is retained for manual interpretation and does
not override the decision.

The decision composition is therefore::

    any rule 1, 2 or 3
              |
              v
    any target alignment spans the boundary window?
              | yes                         | no
              v                             v
            review                       quarantine

    no quarantine rule -> review
"""

from dataclasses import asdict

import pandas as pd

from segonaut.reference.composite.decisions.thresholds import (
    PeakDecisionThresholds,
)

LAYOUT_COLUMNS = {
    "peak_id",
    "alternative_query_id",
    "end_target_count",
    "start_target_count",
    "alternative_union_coverage_fraction",
    "projected_orientation_relation",
    "oriented_endpoint_order",
    "linear_end_to_start_gap_nt",
    "sequence_end_end_to_start_gap_nt",
}


def annotation_group_is_clear(
    target_count: pd.Series,
    fraction_among_nonmissing: pd.Series,
    thresholds: PeakDecisionThresholds,
) -> pd.Series:
    """Return whether one annotation is repeated by enough target contigs."""

    return target_count.ge(thresholds.min_end_start_target_count) & (
        fraction_among_nonmissing.ge(
            thresholds.min_dominant_annotation_fraction
        )
    )


def add_diamond_annotation_difference_flags(
    peaks: pd.DataFrame,
    thresholds: PeakDecisionThresholds,
) -> None:
    """Compare dominant DIAMOND annotations on the two sides of each peak.

    A value must occur on at least ``min_end_start_target_count`` target
    contigs and dominate the non-missing DIAMOND rows on its side. The literal
    protein role ``unknown`` is a value; a missing role remains missing.
    """

    for name, relation_column, count_stem in (
        ("domain", "end_start_best_domain_relation", "best_domain"),
        (
            "viral_family",
            "end_start_best_viral_family_relation",
            "best_viral_family",
        ),
        (
            "protein_role",
            "end_start_viral_protein_role_relation",
            "viral_protein_role",
        ),
    ):
        end_clear = annotation_group_is_clear(
            peaks[f"end_dominant_{count_stem}_target_count"],
            peaks[f"end_dominant_{count_stem}_fraction_among_nonmissing"],
            thresholds,
        )
        start_clear = annotation_group_is_clear(
            peaks[f"start_dominant_{count_stem}_target_count"],
            peaks[f"start_dominant_{count_stem}_fraction_among_nonmissing"],
            thresholds,
        )
        peaks[f"different_clear_{name}"] = (
            peaks[relation_column].eq("different") & end_clear & start_clear
        )

    peaks["different_clear_diamond_annotation"] = peaks[
        [
            "different_clear_domain",
            "different_clear_viral_family",
            "different_clear_protein_role",
        ]
    ].any(axis=1)


def mark_nearby_two_sided_alternative_layouts(
    layouts: pd.DataFrame,
    thresholds: PeakDecisionThresholds,
) -> pd.DataFrame:
    r"""Mark alternative representatives with nearby end and start groups.

    Two arrangements can describe a different FASTA cut point. Overlapping
    projected ranges are deliberately not interpreted here::

        internal continuation             continuation through FASTA end

        [ end ] -- gap -- [ start ]       [ start ] .... [ end ]
                     <= max gap              `-- end --' <= max gap
    """

    marked = layouts.copy()
    enough_targets = (
        marked["end_target_count"].ge(thresholds.min_end_start_target_count)
        & marked["start_target_count"].ge(
            thresholds.min_end_start_target_count
        )
    )
    enough_union = marked["alternative_union_coverage_fraction"].ge(
        thresholds.min_alternative_union_coverage_fraction
    )
    one_orientation = marked["projected_orientation_relation"].ne("mixed")

    internal_gap = marked["oriented_endpoint_order"].eq("end_before_start") & (
        marked["linear_end_to_start_gap_nt"].between(
            0, thresholds.max_alternative_gap_nt
        )
    )
    fasta_end_gap = marked["oriented_endpoint_order"].eq("start_before_end") & (
        marked["sequence_end_end_to_start_gap_nt"].between(
            0, thresholds.max_alternative_gap_nt
        )
    )
    marked["alternative_gap_kind"] = pd.Series(
        pd.NA, index=marked.index, dtype="string"
    )
    marked.loc[internal_gap, "alternative_gap_kind"] = "internal_gap"
    marked.loc[fasta_end_gap, "alternative_gap_kind"] = "fasta_end_gap"
    marked["alternative_gap_nt"] = pd.Series(
        pd.NA, index=marked.index, dtype="Int64"
    )
    marked.loc[internal_gap, "alternative_gap_nt"] = marked.loc[
        internal_gap, "linear_end_to_start_gap_nt"
    ]
    marked.loc[fasta_end_gap, "alternative_gap_nt"] = marked.loc[
        fasta_end_gap, "sequence_end_end_to_start_gap_nt"
    ]
    marked["is_nearby_two_sided_alternative_layout"] = (
        enough_targets
        & enough_union
        & one_orientation
        & (internal_gap | fasta_end_gap)
    )
    return marked


def summarize_nearby_alternative_layouts(layouts: pd.DataFrame) -> pd.DataFrame:
    """Return one row per peak with the best qualifying alternative."""

    nearby = layouts.loc[
        layouts["is_nearby_two_sided_alternative_layout"]
    ].copy()
    columns = [
        "peak_id",
        "nearby_two_sided_alternative_count",
        "best_nearby_alternative_query_id",
        "best_nearby_alternative_gap_kind",
        "best_nearby_alternative_gap_nt",
        "best_nearby_alternative_union_coverage_fraction",
        "best_nearby_alternative_end_target_count",
        "best_nearby_alternative_start_target_count",
    ]
    if nearby.empty:
        return pd.DataFrame(columns=columns)

    nearby["smaller_alternative_endpoint_target_count"] = nearby[
        ["end_target_count", "start_target_count"]
    ].min(axis=1)
    # Prefer the sequence supported by more target contigs, then greater
    # covered fraction, then the shorter measured gap.
    nearby = nearby.sort_values(
        [
            "peak_id",
            "smaller_alternative_endpoint_target_count",
            "alternative_union_coverage_fraction",
            "alternative_gap_nt",
            "alternative_query_id",
        ],
        ascending=[True, False, False, True, True],
    )
    counts = nearby.groupby("peak_id", sort=True).size()
    best = nearby.drop_duplicates("peak_id", keep="first").copy()
    best["nearby_two_sided_alternative_count"] = best["peak_id"].map(counts)
    return best.rename(
        columns={
            "alternative_query_id": "best_nearby_alternative_query_id",
            "alternative_gap_kind": "best_nearby_alternative_gap_kind",
            "alternative_gap_nt": "best_nearby_alternative_gap_nt",
            "alternative_union_coverage_fraction": (
                "best_nearby_alternative_union_coverage_fraction"
            ),
            "end_target_count": "best_nearby_alternative_end_target_count",
            "start_target_count": "best_nearby_alternative_start_target_count",
        }
    )[columns]


def add_peak_rule_flags(
    peaks: pd.DataFrame,
    thresholds: PeakDecisionThresholds,
) -> None:
    """Calculate every quarantine rule, blocker and review rule independently."""

    peaks["maximum_directional_or_overlap_b_score"] = peaks[
        ["directional_b_score", "overlap_b_score"]
    ].max(axis=1)
    peaks["enough_end_start_targets"] = (
        peaks["end_target_count"].ge(thresholds.min_end_start_target_count)
        & peaks["start_target_count"].ge(
            thresholds.min_end_start_target_count
        )
    )
    peaks["enough_directional_or_overlap_b_score"] = peaks[
        "maximum_directional_or_overlap_b_score"
    ].ge(thresholds.min_directional_or_overlap_b_score)
    peaks["has_spanning_targets"] = peaks["spanning_target_count"].gt(
        thresholds.max_spanning_target_count
    )
    peaks["has_nearby_two_sided_alternative"] = peaks[
        "nearby_two_sided_alternative_count"
    ].gt(0)

    # For an end-dominant peak, the suffix is not represented by targets that
    # stop at the peak. For a start-dominant peak, the prefix is analogous.
    #
    # end:    [ targets =====|---- unsupported suffix ----]
    # start:  [unsupported prefix ----|===== targets ]
    peaks["unsupported_extension_nt"] = 0
    end_peak = peaks["score_geometry"].eq("one_sided_end")
    start_peak = peaks["score_geometry"].eq("one_sided_start")
    peaks.loc[end_peak, "unsupported_extension_nt"] = (
        peaks.loc[end_peak, "query_length"] - peaks.loc[end_peak, "peak_position"]
    )
    peaks.loc[start_peak, "unsupported_extension_nt"] = (
        peaks.loc[start_peak, "peak_position"] - 1
    )

    peaks["rule_targets_span_peak"] = peaks["has_spanning_targets"]
    peaks["rule_nearby_two_sided_alternative"] = peaks[
        "has_nearby_two_sided_alternative"
    ]
    peaks["rule_two_sided_different_diamond_annotations"] = (
        peaks["enough_end_start_targets"]
        & peaks["enough_directional_or_overlap_b_score"]
        & peaks["different_clear_diamond_annotation"]
    )
    strong_one_sided = peaks["peak_score"].ge(thresholds.min_one_sided_score) & (
        peaks["unsupported_extension_nt"].ge(
            thresholds.min_unsupported_extension_nt
        )
    )
    peaks["rule_strong_one_sided_end"] = end_peak & strong_one_sided
    peaks["rule_strong_one_sided_start"] = start_peak & strong_one_sided
    peaks["rule_reciprocal_one_sided_pair"] = peaks[
        "reciprocal_one_sided_pair_count"
    ].gt(0)

    peaks["rule_insufficient_end_or_start_targets"] = ~peaks[
        "enough_end_start_targets"
    ]
    peaks["rule_weak_directional_and_overlap_scores"] = (
        peaks["enough_end_start_targets"]
        & ~peaks["enough_directional_or_overlap_b_score"]
    )
    peaks["rule_no_clear_diamond_annotation_difference"] = (
        peaks["enough_end_start_targets"]
        & peaks["enough_directional_or_overlap_b_score"]
        & ~peaks["different_clear_diamond_annotation"]
    )


def add_threshold_columns(
    peaks: pd.DataFrame,
    thresholds: PeakDecisionThresholds,
) -> None:
    """Store every decision threshold beside the measured peak values."""

    for name, value in asdict(thresholds).items():
        peaks[f"threshold_{name}"] = value


QUARANTINE_RULE_COLUMNS = {
    "two_sided_different_diamond_annotations": (
        "rule_two_sided_different_diamond_annotations"
    ),
    "strong_one_sided_end": "rule_strong_one_sided_end",
    "strong_one_sided_start": "rule_strong_one_sided_start",
}
QUARANTINE_BLOCKER_COLUMNS = {
    "targets_span_peak": "rule_targets_span_peak",
}
REVIEW_RULE_COLUMNS = {
    "insufficient_end_or_start_targets": (
        "rule_insufficient_end_or_start_targets"
    ),
    "weak_directional_and_overlap_scores": (
        "rule_weak_directional_and_overlap_scores"
    ),
    "no_clear_diamond_annotation_difference": (
        "rule_no_clear_diamond_annotation_difference"
    ),
}


def matched_rule_names(
    row: pd.Series,
    rule_columns: dict[str, str],
) -> list[str]:
    """Return every named rule whose explicit boolean column is true."""

    return [name for name, column in rule_columns.items() if bool(row[column])]


def combine_peak_rule_results(row: pd.Series) -> dict[str, object]:
    """Combine independent results; blockers override quarantine rules."""

    quarantine_rules = matched_rule_names(row, QUARANTINE_RULE_COLUMNS)
    blockers = matched_rule_names(row, QUARANTINE_BLOCKER_COLUMNS)
    review_rules = matched_rule_names(row, REVIEW_RULE_COLUMNS)

    if blockers:
        action = "review"
        reason = "quarantine_blocked_by:" + ",".join(blockers)
    elif quarantine_rules:
        action = "quarantine"
        reason = "quarantine_rules:" + ",".join(quarantine_rules)
    else:
        action = "review"
        reason = "review_rules:" + ",".join(review_rules)
    return {
        "matched_quarantine_rules": quarantine_rules,
        "matched_quarantine_blockers": blockers,
        "matched_review_rules": review_rules,
        "peak_action": action,
        "decision_reason": reason,
    }
