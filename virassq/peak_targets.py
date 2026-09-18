r"""Map every selected boundary score peak to its raw target contigs.

This is a mechanical coordinate transformation. It does not use DIAMOND and
does not decide whether a representative is composite::

    boundary_score_peaks.parquet + normalized_alignments.parquet
                              |
                              v
                     peak_targets.parquet
                     one row per peak_id + target_id

For each target, boolean columns state exactly which per-position counts it
contributes to. Several MMseqs2 alignment rows from one target still count as
one physical target contig::

    alignment rows
          |
          | join by query_id to every selected peak on that query
          v
    coordinate flags for each alignment row
          |
          | group by peak_id + target_id
          | BOOL_OR: did at least one row contribute?
          v
    one peak_id x target_id row
          |
          | sum target flags and compare with boundary score counts
          v
    audited peak_targets.parquet

This module performs only that mechanical join and aggregation. It does not
compare annotations and does not decide whether a query should be kept,
reviewed or quarantined.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from segonaut.sql import parquet_columns, quote_sql_string

PEAK_COLUMNS = {
    "peak_id",
    "query_id",
    "query_length",
    "peak_position",
    "radius_nt",
    "score_geometry",
    "local_end_target_count",
    "local_start_target_count",
    "directional_end_target_count",
    "directional_start_target_count",
    "overlap_end_target_count",
    "overlap_start_target_count",
    "internal_end_target_count",
    "internal_start_target_count",
    "spanning_target_count",
}

ALIGNMENT_COLUMNS = {
    "query_id",
    "target_id",
    "query_from",
    "query_to",
}

# Each flag below must reproduce the distinct-target count already stored in
# the peak row. This makes coordinate drift between scoring and target
# assignment a hard error instead of a silent scientific discrepancy.
COUNT_FLAG_COLUMNS = {
    "local_start_target_count": "contributes_local_start",
    "local_end_target_count": "contributes_local_end",
    "directional_start_target_count": "contributes_directional_start",
    "directional_end_target_count": "contributes_directional_end",
    "overlap_start_target_count": "contributes_overlap_start",
    "overlap_end_target_count": "contributes_overlap_end",
    "internal_start_target_count": "contributes_internal_start",
    "internal_end_target_count": "contributes_internal_end",
    "spanning_target_count": "has_spanning_alignment",
}


def write_peak_targets(
    connection: duckdb.DuckDBPyConnection,
    peaks_path: Path,
    alignments_path: Path,
    output_path: Path,
) -> None:
    """Aggregate alignment rows directly into one Parquet row per peak and target.

    All coordinates are positions on the query representative::

                         b-r          b          b+r
                          |-----------|-----------|
    local endpoints       [-----------------------]
    directional ends      [-----------]
    directional starts                [-----------]
    overlap starts        [-----------]
    overlap ends                      [-----------]

    One target may have several MMseqs2 alignment rows. ``BOOL_OR`` answers
    whether at least one row contributes to a count. In particular, spanning
    requires one row to cover the complete ``[b-r, b+r]`` interval::

        target T, row 1     ========|
        target T, row 2                 |========
        boundary window         [--- b ---]

        crosses_peak_position may be true after aggregation
        has_spanning_alignment remains false

    The SQL has two mechanical levels. ``flags`` retains one row per MMseqs2
    alignment and evaluates coordinate predicates. The outer GROUP BY
    collapses those rows to one physical target contig per peak.
    """

    peaks = quote_sql_string(peaks_path)
    alignments = quote_sql_string(alignments_path)
    output = quote_sql_string(output_path)
    connection.execute(
        f"""
        COPY (
          WITH flags AS (
            SELECT
              p.peak_id,
              p.query_id,
              p.query_length,
              p.peak_position,
              p.radius_nt,
              p.score_geometry,
              a.target_id,
              a.query_from,
              a.query_to,
              a.query_from BETWEEN p.peak_position - p.radius_nt
                               AND p.peak_position + p.radius_nt AS local_start,
              a.query_to BETWEEN p.peak_position - p.radius_nt
                             AND p.peak_position + p.radius_nt AS local_end,
              a.query_from BETWEEN p.peak_position
                               AND p.peak_position + p.radius_nt AS directional_start,
              a.query_to BETWEEN p.peak_position - p.radius_nt
                             AND p.peak_position AS directional_end,
              a.query_from BETWEEN p.peak_position - p.radius_nt
                               AND p.peak_position AS overlap_start,
              a.query_to BETWEEN p.peak_position
                             AND p.peak_position + p.radius_nt AS overlap_end,
              local_start AND a.query_from > 1 + p.radius_nt AS internal_start,
              local_end AND a.query_to < p.query_length - p.radius_nt AS internal_end,
              a.query_to <= p.peak_position AS is_left,
              a.query_from >= p.peak_position AS is_right,
              a.query_from < p.peak_position
                AND a.query_to > p.peak_position AS is_crossing,
              a.query_from <= p.peak_position - p.radius_nt
                AND a.query_to >= p.peak_position + p.radius_nt AS is_spanning
            FROM read_parquet({peaks}) p
            JOIN read_parquet({alignments}) a USING (query_id)
          ),
          per_target AS (
            SELECT
              peak_id,
              query_id,
              query_length,
              peak_position,
              radius_nt,
              score_geometry,
              target_id,
              count(*)::BIGINT AS alignment_count,
              min(query_from)::BIGINT AS min_query_from,
              max(query_to)::BIGINT AS max_query_to,
              min(abs(query_from - peak_position))::BIGINT
                AS closest_start_distance_nt,
              min(abs(query_to - peak_position))::BIGINT
                AS closest_end_distance_nt,
              count_if(local_start)::BIGINT AS local_start_alignment_count,
              count_if(local_end)::BIGINT AS local_end_alignment_count,
              bool_or(local_start) AS contributes_local_start,
              bool_or(local_end) AS contributes_local_end,
              bool_or(directional_start) AS contributes_directional_start,
              bool_or(directional_end) AS contributes_directional_end,
              bool_or(overlap_start) AS contributes_overlap_start,
              bool_or(overlap_end) AS contributes_overlap_end,
              bool_or(internal_start) AS contributes_internal_start,
              bool_or(internal_end) AS contributes_internal_end,
              count_if(is_left)::BIGINT AS left_alignment_count,
              count_if(is_right)::BIGINT AS right_alignment_count,
              count_if(is_crossing)::BIGINT AS crossing_alignment_count,
              bool_or(is_left) AS has_left_alignment,
              bool_or(is_right) AS has_right_alignment,
              bool_or(is_crossing) AS crosses_peak_position,
              max(
                CASE WHEN is_crossing
                  THEN least(peak_position - query_from, query_to - peak_position)
                  ELSE 0
                END
              )::BIGINT AS max_symmetric_peak_flank_nt,
              bool_or(is_spanning) AS has_spanning_alignment
            FROM flags
            GROUP BY ALL
          )
          SELECT
            *,
            CASE score_geometry
              WHEN 'local' THEN contributes_local_end
              WHEN 'directional' THEN contributes_directional_end
              WHEN 'overlap' THEN contributes_overlap_end
              WHEN 'one_sided_end' THEN contributes_internal_end
              WHEN 'one_sided_start' THEN contributes_internal_end
            END AS enters_peak_end_count,
            CASE score_geometry
              WHEN 'local' THEN contributes_local_start
              WHEN 'directional' THEN contributes_directional_start
              WHEN 'overlap' THEN contributes_overlap_start
              WHEN 'one_sided_end' THEN contributes_internal_start
              WHEN 'one_sided_start' THEN contributes_internal_start
            END AS enters_peak_start_count
          FROM per_target
          ORDER BY query_id, peak_id, target_id
        ) TO {output} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def count_peak_target_mismatches(
    connection: duckdb.DuckDBPyConnection,
    peaks_path: Path,
    peak_targets_path: Path,
) -> int:
    """Count peak measurements that disagree with stored profile counts.

    For each of the eight endpoint counts and the spanning count::

        boundary_score_peaks.expected_target_count
                              ==
        sum(peak_targets.contributes_to_that_count)

    A mismatch means scoring and target assignment used different coordinate
    definitions. The caller deletes the output instead of allowing that drift
    into the annotation or decision layer.
    """

    peaks = quote_sql_string(peaks_path)
    peak_targets = quote_sql_string(peak_targets_path)
    comparisons = []
    for count_column, flag_column in COUNT_FLAG_COLUMNS.items():
        comparisons.append(
            f"SELECT p.peak_id "
            f"FROM read_parquet({peaks}) p "
            f"LEFT JOIN ("
            f"  SELECT peak_id, sum({flag_column}::INTEGER) AS observed_count "
            f"  FROM read_parquet({peak_targets}) GROUP BY peak_id"
            f") o USING (peak_id) "
            f"WHERE p.{count_column} != coalesce(o.observed_count, 0)"
        )
    query = " UNION ALL ".join(comparisons)
    return int(connection.execute(f"SELECT count(*) FROM ({query})").fetchone()[0])


def assign_peak_targets(
    peaks_path: Path,
    alignments_path: Path,
    output_path: Path,
    *,
    threads: int = 8,
    force: bool = False,
) -> dict[str, object]:
    """Write peak-target measurements after exact count agreement checks."""

    if not peaks_path.is_file():
        raise FileNotFoundError(f"boundary score peak table does not exist: {peaks_path}")
    if not alignments_path.is_file():
        raise FileNotFoundError(f"normalized alignment table does not exist: {alignments_path}")
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path}")
    if threads < 1:
        raise ValueError("threads must be >= 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as connection:
        connection.execute(f"PRAGMA threads={threads}")
        missing_peaks = sorted(PEAK_COLUMNS - parquet_columns(connection, peaks_path))
        if missing_peaks:
            raise ValueError(
                f"boundary score peak table is missing columns: {missing_peaks}"
            )
        missing_alignments = sorted(
            ALIGNMENT_COLUMNS - parquet_columns(connection, alignments_path)
        )
        if missing_alignments:
            raise ValueError(
                f"alignment table is missing columns: {missing_alignments}"
            )
        duplicate_peak_count = int(
            connection.execute(
                f"""
                SELECT count(*) - count(DISTINCT peak_id)
                FROM read_parquet({quote_sql_string(peaks_path)})
                """
            ).fetchone()[0]
        )
        if duplicate_peak_count:
            raise ValueError("peak_id values must be unique")

        write_peak_targets(connection, peaks_path, alignments_path, output_path)
        mismatch_count = count_peak_target_mismatches(
            connection,
            peaks_path,
            output_path,
        )
        if mismatch_count:
            output_path.unlink(missing_ok=True)
            raise ValueError(
                f"peak target flags disagree with {mismatch_count} stored counts"
            )

        peak_count, peak_with_targets_count = connection.execute(
            f"""
            SELECT
              (SELECT count(*) FROM read_parquet({quote_sql_string(peaks_path)})),
              count(DISTINCT peak_id)
            FROM read_parquet({quote_sql_string(output_path)})
            """
        ).fetchone()
        peak_target_count = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({quote_sql_string(output_path)})"
            ).fetchone()[0]
        )
        selected_query_alignment_count = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM read_parquet({quote_sql_string(alignments_path)}) a
                SEMI JOIN (
                  SELECT DISTINCT query_id
                  FROM read_parquet({quote_sql_string(peaks_path)})
                ) p USING (query_id)
                """
            ).fetchone()[0]
        )

    return {
        "output": str(output_path),
        "peak_count": int(peak_count),
        "peak_with_targets_count": int(peak_with_targets_count),
        "peak_target_count": peak_target_count,
        "selected_query_alignment_count": selected_query_alignment_count,
        "audited_count_columns": list(COUNT_FLAG_COLUMNS),
        "count_mismatch_count": mismatch_count,
    }
