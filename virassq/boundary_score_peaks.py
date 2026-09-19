"""Select separated peaks from per-position boundary score profiles.

Each score geometry remains an independent observation. Peak selection has
two explicit steps::

    boundary_profiles.parquet
              |
              | 1. find a separated above-threshold score region
              | 2. choose one coordinate inside that region
              v
    Q:local:P001         at 1747
    Q:directional:P001   at 1745
    Q:one_sided_end:P001 at 1748

Nearby peaks from different geometries are not merged. Their relationship can
only be assessed later from the target contigs that produced the counts.

The position where a score first reaches its maximum and the coordinate
chosen inside the region are deliberately stored separately::

    local score plateau       5 5 5 5 5 5 5
                              |             |
    score_max_position -------+             |
                                            |
    directional/overlap maximum ------------+--> peak_position

For a local peak, ``peak_score`` is the local score at ``peak_position``. It
can therefore be smaller than ``score_max_value``. No value is discarded:
both coordinates, both values, the selection method and their displacement
remain in the output row.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from virassq.sql import parquet_columns, quote_sql_string

PROFILE_COLUMNS = {
    "query_id",
    "query_length",
    "boundary_position",
    "radius_nt",
    "local_end_target_count",
    "local_start_target_count",
    "local_b_score",
    "directional_end_target_count",
    "directional_start_target_count",
    "directional_b_score",
    "overlap_end_target_count",
    "overlap_start_target_count",
    "overlap_b_score",
    "internal_end_target_count",
    "internal_start_target_count",
    "one_sided_end_score",
    "one_sided_start_score",
    "spanning_target_count",
}

# Keeping these column names together prevents a peak from accidentally
# receiving counts from another score geometry.
SCORE_GEOMETRIES = {
    "local": {
        "score_column": "local_b_score",
        "end_count_column": "local_end_target_count",
        "start_count_column": "local_start_target_count",
    },
    "directional": {
        "score_column": "directional_b_score",
        "end_count_column": "directional_end_target_count",
        "start_count_column": "directional_start_target_count",
    },
    "overlap": {
        "score_column": "overlap_b_score",
        "end_count_column": "overlap_end_target_count",
        "start_count_column": "overlap_start_target_count",
    },
    "one_sided_end": {
        "score_column": "one_sided_end_score",
        "end_count_column": "internal_end_target_count",
        "start_count_column": "internal_start_target_count",
    },
    "one_sided_start": {
        "score_column": "one_sided_start_score",
        "end_count_column": "internal_end_target_count",
        "start_count_column": "internal_start_target_count",
    },
}

POSITION_SCORE_COLUMNS = [
    "local_end_target_count",
    "local_start_target_count",
    "local_b_score",
    "directional_end_target_count",
    "directional_start_target_count",
    "directional_b_score",
    "overlap_end_target_count",
    "overlap_start_target_count",
    "overlap_b_score",
    "internal_end_target_count",
    "internal_start_target_count",
    "one_sided_end_score",
    "one_sided_start_score",
    "spanning_target_count",
]

BOUNDARY_SCORE_PEAK_COLUMNS = [
    "peak_id",
    "query_id",
    "query_length",
    "radius_nt",
    "score_geometry",
    "peak_index",
    "peak_selection_rank",
    "score_max_position",
    "score_max_value",
    "peak_position",
    "peak_position_method",
    "position_shift_nt",
    "peak_score",
    "end_target_count",
    "start_target_count",
    *POSITION_SCORE_COLUMNS,
    "minimum_peak_distance_nt",
]


def enabled_score_thresholds(
    *,
    min_local_b_score: int | None,
    min_directional_b_score: int | None,
    min_overlap_b_score: int | None,
    min_one_sided_score: int | None,
) -> dict[str, int]:
    """Return the explicitly enabled score thresholds."""

    thresholds = {
        "local": min_local_b_score,
        "directional": min_directional_b_score,
        "overlap": min_overlap_b_score,
        "one_sided_end": min_one_sided_score,
        "one_sided_start": min_one_sided_score,
    }
    enabled = {name: value for name, value in thresholds.items() if value is not None}
    if not enabled:
        raise ValueError("at least one minimum boundary score must be provided")
    if any(value < 1 for value in enabled.values()):
        raise ValueError("minimum boundary scores must be >= 1")
    return enabled


def load_matching_positions(
    profiles: Path,
    thresholds: dict[str, int],
    *,
    threads: int,
) -> pd.DataFrame:
    """Load profile positions passing any enabled score threshold."""

    score_filter = " OR ".join(
        f"{SCORE_GEOMETRIES[name]['score_column']} >= {threshold}"
        for name, threshold in thresholds.items()
    )
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    try:
        columns = parquet_columns(con, profiles)
        missing = sorted(PROFILE_COLUMNS - columns)
        if missing:
            raise ValueError(f"boundary profile is missing columns: {missing}")

        return con.execute(
            f"""
            SELECT *
            FROM read_parquet({quote_sql_string(profiles)})
            WHERE ({score_filter})
            ORDER BY query_id, boundary_position
            """
        ).df()
    finally:
        con.close()


def find_next_score_region(
    remaining_positions: pd.DataFrame,
    score_column: str,
    *,
    minimum_peak_distance_nt: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return the strongest remaining position and its nearby score region.

    ``remaining_positions`` is already restricted to one query, one geometry
    and positions passing that geometry's threshold::

        strongest remaining position
                    |
          [ positions within minimum_peak_distance_nt ]
                    score region

    Equal score maxima are ordered left to right before this function is
    called, so the returned maximum is deterministic.
    """

    score_maximum = remaining_positions.iloc[0]
    score_max_position = int(score_maximum["boundary_position"])
    inside_region = (
        remaining_positions["boundary_position"]
        .sub(score_max_position)
        .abs()
        .le(minimum_peak_distance_nt)
    )
    return score_maximum, remaining_positions.loc[inside_region]


def choose_position_within_score_region(
    score_region: pd.DataFrame,
    score_maximum: pd.Series,
    geometry: str,
    *,
    score_column: str,
) -> tuple[pd.Series, str]:
    """Choose one coordinate inside an already detected score region.

    The region was detected only by ``score_column``. Coordinate choice then
    follows the rule recorded in ``peak_position_method``::

        local       max(directional_b_score, overlap_b_score)
        one-sided   score-weighted centre
        other       own score maximum

    The returned row retains every score and target count at the selected
    coordinate; the caller separately stores the original maximum.
    """

    if geometry == "local":
        # Local detects the region; directional/overlap locate the most
        # geometrically consistent coordinate inside it. local_b_score is the
        # tie-breaker, followed by the leftmost coordinate.
        chosen = score_region.sort_values(
            [
                "_paired_b_score",
                "local_b_score",
                "boundary_position",
            ],
            ascending=[False, False, True],
        ).iloc[0].copy()
        return chosen, "directional_or_overlap_within_local_region"

    if geometry.startswith("one_sided_"):
        # Endpoints at 850 with radius=50 vote over a broad 800--900
        # region. Its weighted centre returns the selected coordinate to the
        # observed endpoint pileup instead of the region edge.
        weighted_centre = round(
            (score_region["boundary_position"] * score_region[score_column]).sum()
            / score_region[score_column].sum()
        )
        chosen = (
            score_region.assign(
                _distance_to_centre=score_region["boundary_position"]
                .sub(weighted_centre)
                .abs()
            )
            .sort_values(
                [
                    "_distance_to_centre",
                    score_column,
                    "boundary_position",
                ],
                ascending=[True, False, True],
            )
            .iloc[0]
            .copy()
        )
        return chosen, "weighted_center"

    # Directional and overlap already encode a direction around b, so their
    # own maximum is also their selected coordinate.
    return score_maximum.copy(), "score_maximum"


def select_separated_score_peaks(
    query_positions: pd.DataFrame,
    geometry: str,
    threshold: int,
    *,
    minimum_peak_distance_nt: int,
) -> pd.DataFrame:
    """Iteratively find a score region, choose a coordinate and suppress it.

    A moving-window score produces a broad above-threshold region around the
    same endpoint pileup. The strongest remaining position starts one region.
    Only positions of this geometry that pass its threshold participate::

        positions on Q:  1 --------- b1 ------------------ b2 -------- L
        score >= limit:      [==========]                  [========]
                              region 1                      region 2

    Coordinate choice inside one region depends on the score geometry::

        local       detect by local; locate by max(directional, overlap)
        directional locate by directional maximum
        overlap     locate by overlap maximum
        one-sided   locate by the score-weighted centre of the region

    Local coordinate refinement is intentional. ``local_b_score`` says that
    starts and ends accumulate somewhere within a radius-wide neighborhood,
    while directional and overlap counts can place the transition more
    precisely inside that broad local region.

    After a coordinate is chosen, positions within
    ``minimum_peak_distance_nt`` of either the score maximum or chosen
    coordinate are removed from this geometry. This prevents one broad region
    from producing several nearby rows while retaining distant peaks on the
    same query::

        score maximum       chosen coordinate
             |--------------------|
          remove around both positions          keep distant region
          [=======================]              [============]

    Different geometries are never removed against each other here. A local
    and a one-sided peak may therefore remain close and be compared later by
    their concrete target IDs.
    """

    score_column = SCORE_GEOMETRIES[geometry]["score_column"]
    matching = query_positions.loc[query_positions[score_column].ge(threshold)].copy()
    if matching.empty:
        return matching

    matching["_paired_b_score"] = matching[
        ["directional_b_score", "overlap_b_score"]
    ].max(axis=1)
    remaining = matching.sort_values(
        [score_column, "boundary_position"],
        ascending=[False, True],
    )

    selected_rows = []
    while not remaining.empty:
        score_maximum, score_region = find_next_score_region(
            remaining,
            score_column,
            minimum_peak_distance_nt=minimum_peak_distance_nt,
        )
        score_max_position = int(score_maximum["boundary_position"])
        chosen, peak_position_method = choose_position_within_score_region(
            score_region,
            score_maximum,
            geometry,
            score_column=score_column,
        )

        chosen["score_max_position"] = score_max_position
        chosen["score_max_value"] = int(score_maximum[score_column])
        chosen["peak_position"] = int(chosen["boundary_position"])
        chosen["peak_position_method"] = peak_position_method
        chosen["position_shift_nt"] = (
            int(chosen["boundary_position"]) - score_max_position
        )
        selected_rows.append(chosen)

        # Refinement can move peak_position away from score_max_position.
        # Suppressing around both avoids rediscovering the same broad region
        # from the portion left behind by that shift.
        chosen_position = int(chosen["boundary_position"])
        near_score_maximum = remaining["boundary_position"].sub(
            score_max_position
        ).abs().le(
            minimum_peak_distance_nt
        )
        near_chosen = remaining["boundary_position"].sub(chosen_position).abs().le(
            minimum_peak_distance_nt
        )
        remaining = remaining.loc[~(near_score_maximum | near_chosen)]

    selected = pd.DataFrame(selected_rows)
    selected["peak_selection_rank"] = range(1, len(selected) + 1)
    return selected.drop(
        columns=["_paired_b_score", "_distance_to_centre"],
        errors="ignore",
    )


def build_boundary_score_peaks(
    matching_positions: pd.DataFrame,
    thresholds: dict[str, int],
) -> pd.DataFrame:
    """Return one row per separated peak and score geometry.

    Output coordinates have distinct meanings::

        score_max_position   leftmost maximum that started the region
        score_max_value      this geometry's score at that maximum
        peak_position        coordinate selected inside the region
        peak_score           this geometry's score at peak_position
        position_shift_nt    peak_position - score_max_position

    ``peak_index`` orders peaks from left to right within one query and
    geometry. ``peak_selection_rank`` retains the greedy strongest-first
    selection order.
    """

    selected_groups = []
    for _, query_positions in matching_positions.groupby("query_id", sort=True):
        radius_nt = int(query_positions["radius_nt"].iloc[0])
        minimum_distance = 2 * radius_nt
        for geometry, threshold in thresholds.items():
            selected = select_separated_score_peaks(
                query_positions,
                geometry,
                threshold,
                minimum_peak_distance_nt=minimum_distance,
            )
            if selected.empty:
                continue
            geometry_columns = SCORE_GEOMETRIES[geometry]
            score_column = geometry_columns["score_column"]
            end_column = geometry_columns["end_count_column"]
            start_column = geometry_columns["start_count_column"]
            selected["score_geometry"] = geometry
            selected["peak_score"] = selected[score_column]
            selected["end_target_count"] = selected[end_column]
            selected["start_target_count"] = selected[start_column]
            selected["minimum_peak_distance_nt"] = minimum_distance
            selected_groups.append(selected)

    if not selected_groups:
        return pd.DataFrame(columns=BOUNDARY_SCORE_PEAK_COLUMNS)

    peaks = pd.concat(selected_groups, ignore_index=True).sort_values(
        ["query_id", "score_geometry", "peak_position"]
    )
    peaks["peak_index"] = peaks.groupby(["query_id", "score_geometry"]).cumcount().add(1)
    peaks["peak_id"] = peaks.apply(
        lambda row: (
            f"{row['query_id']}:{row['score_geometry']}:P{int(row['peak_index']):03d}"
        ),
        axis=1,
    )
    return peaks[BOUNDARY_SCORE_PEAK_COLUMNS].reset_index(drop=True)


def select_boundary_score_peaks(
    profiles: Path,
    output_directory: Path,
    *,
    min_local_b_score: int | None = None,
    min_directional_b_score: int | None = None,
    min_overlap_b_score: int | None = None,
    min_one_sided_score: int | None = None,
    threads: int = 8,
    force: bool = False,
) -> dict[str, object]:
    """Write separated boundary score peaks without merging geometries."""

    if not profiles.is_file():
        raise FileNotFoundError(f"boundary profile does not exist: {profiles}")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    thresholds = enabled_score_thresholds(
        min_local_b_score=min_local_b_score,
        min_directional_b_score=min_directional_b_score,
        min_overlap_b_score=min_overlap_b_score,
        min_one_sided_score=min_one_sided_score,
    )
    output = output_directory / "boundary_score_peaks.parquet"
    manifest = output_directory / "boundary_score_peaks.manifest.json"
    existing = [str(path) for path in (output, manifest) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "boundary score peak outputs already exist:\n" + "\n".join(existing)
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    matching = load_matching_positions(
        profiles,
        thresholds,
        threads=threads,
    )
    peaks = build_boundary_score_peaks(matching, thresholds)
    pq.write_table(
        pa.Table.from_pandas(peaks, preserve_index=False),
        output,
        compression="zstd",
    )

    result: dict[str, object] = {
        "schema": "virassq.boundary_score_peaks.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "profiles": str(profiles),
        "boundary_score_peaks": str(output),
        "min_local_b_score": min_local_b_score,
        "min_directional_b_score": min_directional_b_score,
        "min_overlap_b_score": min_overlap_b_score,
        "min_one_sided_score": min_one_sided_score,
        "representatives_with_peaks": int(peaks["query_id"].nunique()),
        "selected_score_peaks": len(peaks),
        "matching_profile_rows": len(matching),
    }
    manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
