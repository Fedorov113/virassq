"""Calculate simple boundary-support profiles for one query contig.

``local_b_score`` asks whether stacks of starts and ends occur near the same
position. Small assembly-end jitter is allowed on either side of ``b``::

              b - radius       b       b + radius
                   |-----------|-----------|
    local ends     [    query_to anywhere  ]
    local starts   [   query_from anywhere ]

    local_b_score = min(local ends, local starts)

``directional_b_score`` asks the stricter geometric question: do alignments
end to the left of ``b`` and start to its right?::

              b - radius       b       b + radius
                   |-----------|-----------|
    directional    [ query_to  ]
                               [ query_from]

    directional_b_score = min(directional ends, directional starts)

``overlap_b_score`` mirrors that geometry. It asks whether alignments start to
the left of ``b`` and other alignments end to its right::

              b - radius       b       b + radius
                   |-----------|-----------|
    overlap         [ query_from]
                                [ query_to ]

    overlap_b_score = min(overlap starts, overlap ends)

This catches a short overlap that has no coordinate satisfying
``end <= b <= start``::

    left alignment   ====================| end
                                  start |==================== right alignment
                                        ^ b inside overlap

``spanning_target_count`` counts targets with one alignment covering the full
neighborhood around ``b``::

              b - radius       b       b + radius
                   |-----------|-----------|
    spanning  =====================================

Two separate alignments of the same target on opposite sides do not count as
one spanning alignment.

Two additional scores retain strongly asymmetric internal endpoint
accumulations. Before counting, starts within one radius of query position 1
and ends within one radius of ``query_length`` are removed. The scores then
count the excess on one side of the same local window::

    one_sided_end_score   = max(internal ends - internal starts, 0)
    one_sided_start_score = max(internal starts - internal ends, 0)

For example, 12 alignments ending near ``b`` and none starting there give an
end score of 12. A 30-to-3 imbalance gives 27, while a balanced 12-to-12
boundary gives zero for both one-sided scores and remains represented by the
existing local/directional/overlap scores.

Nearly all alignments naturally start near position 1 or end near
``query_length``; those physical contig ends are not internal boundary events::

    query  1 |------ internal positions eligible ------| query_length
             < r nt                              r nt >
             ignored                            ignored

The scores count alignment starts and ends; they are not decisions. This
module does not select candidate events or classify a query as composite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _group_row_indices_by_target(target_ids: pd.Series) -> list[np.ndarray]:
    """Return alignment-row indices for each distinct physical target contig.

    Factorization preserves the first-seen target order. Stable sorting then
    places rows of one target together without constructing a DataFrame for
    every target::

        target IDs:        A  B  A  C  B
        row groups:       [0,2] [1,4] [3]
    """

    target_codes, _ = pd.factorize(target_ids, sort=False)
    ordered_rows = np.argsort(target_codes, kind="stable")
    ordered_rows = ordered_rows[target_codes[ordered_rows] >= 0]
    group_boundaries = np.flatnonzero(np.diff(target_codes[ordered_rows])) + 1
    return np.split(ordered_rows, group_boundaries)


def _count_targets_in_vote_intervals(
    coordinates: np.ndarray,
    target_row_groups: list[np.ndarray],
    *,
    query_length: int,
    left_offset: int,
    right_offset: int,
    eligible_alignment_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Count distinct targets voting for each query coordinate.

    One alignment start or end votes for every integer position in its interval::

        left                              right
          [ +1 -------------------------- ] -1

    ``target_row_groups`` contains integer indices into ``coordinates``. The
    same groups are reused by all score geometries, avoiding repeated Pandas
    groupby operations and temporary one-target DataFrames.

    Cumulative summation reconstructs the count without scanning every
    alignment at every position. Intervals are merged per target first, so one
    target still contributes at most one vote to a coordinate::

        target T, block 1       [----------]
        target T, block 2            [----------]
        merged target vote      [---------------]   count T once
    """
    difference = np.zeros(query_length + 2, dtype=np.int32)

    for rows in target_row_groups:
        if eligible_alignment_rows is not None:
            rows = rows[eligible_alignment_rows[rows]]
        if len(rows) == 0:
            continue

        positions = (
            coordinates[rows]
            if len(rows) == 1
            else np.unique(coordinates[rows])
        )
        previous_left = None
        previous_right = None
        for position in positions:
            left = max(1, int(position) + left_offset)
            right = min(query_length, int(position) + right_offset)
            if previous_left is not None and left <= previous_right + 1:
                previous_right = max(previous_right, right)
                continue
            if previous_left is not None:
                difference[previous_left] += 1
                difference[previous_right + 1] -= 1
            previous_left = left
            previous_right = right

        difference[previous_left] += 1
        difference[previous_right + 1] -= 1

    return np.cumsum(difference, dtype=np.int32)[1 : query_length + 1]


def _count_targets_spanning_windows(
    query_from: np.ndarray,
    query_to: np.ndarray,
    target_row_groups: list[np.ndarray],
    *,
    query_length: int,
    radius_nt: int,
) -> np.ndarray:
    """Count distinct targets spanning each full boundary neighborhood.

    An alignment from ``query_from`` to ``query_to`` spans every boundary
    position in this interval::

        query_from       query_from+r       query_to-r       query_to
             |-----------------|===============|-----------------|
                               valid b positions

    Intervals are merged per target before counting, so alternative alignments
    cannot make one target contribute twice at the same coordinate.
    """

    difference = np.zeros(query_length + 2, dtype=np.int32)

    for rows in target_row_groups:
        lefts = query_from[rows] + radius_nt
        rights = query_to[rows] - radius_nt
        valid = lefts <= rights
        if not valid.any():
            continue

        previous_left = None
        previous_right = None
        for left, right in sorted(zip(lefts[valid], rights[valid])):
            left = int(left)
            right = int(right)
            if previous_left is not None and left <= previous_right + 1:
                previous_right = max(previous_right, right)
                continue
            if previous_left is not None:
                difference[previous_left] += 1
                difference[previous_right + 1] -= 1
            previous_left = left
            previous_right = right

        difference[previous_left] += 1
        difference[previous_right + 1] -= 1

    return np.cumsum(difference, dtype=np.int32)[1 : query_length + 1]


def score_boundary_positions(
    alignments: pd.DataFrame,
    *,
    radius_nt: int = 50,
) -> pd.DataFrame:
    """Return local, directional and overlap support at every query coordinate."""
    query_id = str(alignments["query_id"].iloc[0])
    query_length = int(alignments["query_length"].iloc[0])
    query_from = alignments["query_from"].to_numpy(dtype=np.int64, copy=False)
    query_to = alignments["query_to"].to_numpy(dtype=np.int64, copy=False)
    target_row_groups = _group_row_indices_by_target(alignments["target_id"])

    # Local support allows both starts and ends on either side of b.
    local_end_target_count = _count_targets_in_vote_intervals(
        query_to,
        target_row_groups,
        query_length=query_length,
        left_offset=-radius_nt,
        right_offset=radius_nt,
    )
    local_start_target_count = _count_targets_in_vote_intervals(
        query_from,
        target_row_groups,
        query_length=query_length,
        left_offset=-radius_nt,
        right_offset=radius_nt,
    )

    # Directional support keeps ends left of b and starts right of b.
    directional_end_target_count = _count_targets_in_vote_intervals(
        query_to,
        target_row_groups,
        query_length=query_length,
        left_offset=0,
        right_offset=radius_nt,
    )
    directional_start_target_count = _count_targets_in_vote_intervals(
        query_from,
        target_row_groups,
        query_length=query_length,
        left_offset=-radius_nt,
        right_offset=0,
    )

    # Overlap support is the mirror image of directional support:
    #
    #     start <= b <= end
    #
    #     right target   |====================
    #     left target        ====================|
    #                         ^ possible b
    #
    # It intentionally remains a separate score. Combining it with the
    # directional score would hide whether the inferred geometry is a gap or
    # an overlap.
    overlap_end_target_count = _count_targets_in_vote_intervals(
        query_to,
        target_row_groups,
        query_length=query_length,
        left_offset=-radius_nt,
        right_offset=0,
    )
    overlap_start_target_count = _count_targets_in_vote_intervals(
        query_from,
        target_row_groups,
        query_length=query_length,
        left_offset=0,
        right_offset=radius_nt,
    )
    spanning_target_count = _count_targets_spanning_windows(
        query_from,
        query_to,
        target_row_groups,
        query_length=query_length,
        radius_nt=radius_nt,
    )

    # The weaker side controls each paired score. A stack of 20 ends and one
    # start therefore has balanced support of one, not 21.
    # One-sided events are built only from internal endpoints. Filtering the
    # endpoint rows themselves matters: a query_from=1 endpoint votes through
    # position 51 when radius=50, even if position 51 is called "internal".
    internal_end_target_count = _count_targets_in_vote_intervals(
        query_to,
        target_row_groups,
        query_length=query_length,
        left_offset=-radius_nt,
        right_offset=radius_nt,
        eligible_alignment_rows=query_to < query_length - radius_nt,
    )
    internal_start_target_count = _count_targets_in_vote_intervals(
        query_from,
        target_row_groups,
        query_length=query_length,
        left_offset=-radius_nt,
        right_offset=radius_nt,
        eligible_alignment_rows=query_from > 1 + radius_nt,
    )
    one_sided_end_score = np.maximum(
        internal_end_target_count - internal_start_target_count,
        0,
    )
    one_sided_start_score = np.maximum(
        internal_start_target_count - internal_end_target_count,
        0,
    )

    # Query termini are assembly ends, not inferred internal boundaries. This
    # mask prevents the routine start-at-1 and end-at-L stacks from dominating
    # every representative's one-sided maximum.
    positions = np.arange(1, query_length + 1, dtype=np.int32)
    internal_position = (positions > 1 + radius_nt) & (
        positions < query_length - radius_nt
    )
    one_sided_end_score = np.where(internal_position, one_sided_end_score, 0)
    one_sided_start_score = np.where(internal_position, one_sided_start_score, 0)

    return pd.DataFrame(
        {
            "query_id": query_id,
            "query_length": query_length,
            "boundary_position": positions,
            "radius_nt": radius_nt,
            "local_end_target_count": local_end_target_count,
            "local_start_target_count": local_start_target_count,
            "local_b_score": np.minimum(
                local_end_target_count,
                local_start_target_count,
            ),
            "directional_end_target_count": directional_end_target_count,
            "directional_start_target_count": directional_start_target_count,
            "directional_b_score": np.minimum(
                directional_end_target_count,
                directional_start_target_count,
            ),
            "overlap_end_target_count": overlap_end_target_count,
            "overlap_start_target_count": overlap_start_target_count,
            "overlap_b_score": np.minimum(
                overlap_end_target_count,
                overlap_start_target_count,
            ),
            "internal_end_target_count": internal_end_target_count,
            "internal_start_target_count": internal_start_target_count,
            "one_sided_end_score": one_sided_end_score,
            "one_sided_start_score": one_sided_start_score,
            "spanning_target_count": spanning_target_count,
        }
    )
