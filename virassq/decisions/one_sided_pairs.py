r"""Match reciprocal end-dominant and start-dominant peaks.

This module only joins IDs. It does not assign review or quarantine actions::

    Q_end  -- target T -->  R_start
    Q_end  <-- same T  --  R_start

The same raw target contig ``T`` must connect the two representatives in both
directions. A returned pair therefore remains directly auditable.
"""

import pandas as pd

from virassq.decisions.thresholds import (
    PeakDecisionThresholds,
)

LINK_COLUMNS = {
    "peak_id",
    "source_query_id",
    "target_id",
    "alternative_query_id",
}


def find_reciprocal_one_sided_pairs(
    descriptions: pd.DataFrame,
    links: pd.DataFrame,
    thresholds: PeakDecisionThresholds,
) -> pd.DataFrame:
    """Count exact shared target IDs for reciprocal one-sided peak pairs."""

    output_columns = [
        "peak_id",
        "reciprocal_one_sided_pair_count",
        "maximum_reciprocal_shared_target_count",
        "reciprocal_one_sided_peak_ids",
        "reciprocal_one_sided_query_ids",
    ]
    if descriptions.empty or links.empty:
        empty = descriptions[["peak_id"]].head(0).copy()
        empty["reciprocal_one_sided_pair_count"] = pd.Series(dtype="int64")
        empty["maximum_reciprocal_shared_target_count"] = pd.Series(dtype="int64")
        empty["reciprocal_one_sided_peak_ids"] = pd.Series(dtype="object")
        empty["reciprocal_one_sided_query_ids"] = pd.Series(dtype="object")
        return empty[output_columns]

    peak_types = descriptions[["peak_id", "query_id", "score_geometry"]].copy()
    routes = (
        links[list(LINK_COLUMNS)]
        .drop_duplicates()
        .merge(
            peak_types,
            on="peak_id",
            how="inner",
            validate="many_to_one",
        )
    )
    end_routes = routes.loc[routes["score_geometry"].eq("one_sided_end")].rename(
        columns={
            "peak_id": "end_peak_id",
            "source_query_id": "end_query_id",
            "alternative_query_id": "start_query_id",
        }
    )
    start_routes = routes.loc[routes["score_geometry"].eq("one_sided_start")].rename(
        columns={
            "peak_id": "start_peak_id",
            "source_query_id": "start_query_id",
            "alternative_query_id": "end_query_id",
        }
    )
    shared_targets = end_routes.merge(
        start_routes,
        on=["end_query_id", "start_query_id", "target_id"],
        how="inner",
        suffixes=("_end", "_start"),
    )
    pairs = (
        shared_targets.groupby(
            ["end_peak_id", "start_peak_id", "end_query_id", "start_query_id"],
            sort=True,
        )["target_id"]
        .nunique()
        .rename("reciprocal_shared_target_count")
        .reset_index()
    )
    pairs = pairs.loc[
        pairs["reciprocal_shared_target_count"].ge(
            thresholds.min_end_start_target_count
        )
    ]

    if pairs.empty:
        empty = descriptions[["peak_id"]].head(0).copy()
        empty["reciprocal_one_sided_pair_count"] = pd.Series(dtype="int64")
        empty["maximum_reciprocal_shared_target_count"] = pd.Series(dtype="int64")
        empty["reciprocal_one_sided_peak_ids"] = pd.Series(dtype="object")
        empty["reciprocal_one_sided_query_ids"] = pd.Series(dtype="object")
        return empty[output_columns]

    end_view = pairs.rename(
        columns={
            "end_peak_id": "peak_id",
            "start_peak_id": "reciprocal_peak_id",
            "start_query_id": "reciprocal_query_id",
        }
    )
    start_view = pairs.rename(
        columns={
            "start_peak_id": "peak_id",
            "end_peak_id": "reciprocal_peak_id",
            "end_query_id": "reciprocal_query_id",
        }
    )
    per_peak = pd.concat(
        [
            end_view[
                [
                    "peak_id",
                    "reciprocal_peak_id",
                    "reciprocal_query_id",
                    "reciprocal_shared_target_count",
                ]
            ],
            start_view[
                [
                    "peak_id",
                    "reciprocal_peak_id",
                    "reciprocal_query_id",
                    "reciprocal_shared_target_count",
                ]
            ],
        ],
        ignore_index=True,
    )
    return (
        per_peak.groupby("peak_id", sort=True)
        .agg(
            reciprocal_one_sided_pair_count=("reciprocal_peak_id", "nunique"),
            maximum_reciprocal_shared_target_count=(
                "reciprocal_shared_target_count",
                "max",
            ),
            reciprocal_one_sided_peak_ids=(
                "reciprocal_peak_id",
                lambda values: sorted(set(values)),
            ),
            reciprocal_one_sided_query_ids=(
                "reciprocal_query_id",
                lambda values: sorted(set(values)),
            ),
        )
        .reset_index()[output_columns]
    )
