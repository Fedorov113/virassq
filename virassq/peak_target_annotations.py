r"""Attach existing contig-level DIAMOND hits to targets counted at each peak.

This module is a mechanical join. It does not compare annotations and does not
decide whether a representative is composite::

    boundary_score_peaks.parquet     peak_targets.parquet
                  \                       /
                   \                     /
                    v                   v
                 targets counted by each selected peak
                              |
                              | target_id = qseqid
                              v
                   contig_best_hits.parquet
                              |
                              v
                 peak_target_annotations.parquet

Only targets counted by the selected score geometry, plus targets spanning the
whole boundary window, are retained. Missing DIAMOND hits remain explicit rows
with ``has_diamond_hit = false``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from virassq.sql import parquet_columns, quote_sql_string

PEAK_COLUMNS = {
    "peak_id",
    "end_target_count",
    "start_target_count",
    "spanning_target_count",
}

PEAK_TARGET_COLUMNS = {
    "peak_id",
    "query_id",
    "score_geometry",
    "target_id",
    "contributes_local_end",
    "contributes_local_start",
    "contributes_directional_end",
    "contributes_directional_start",
    "contributes_overlap_end",
    "contributes_overlap_start",
    "contributes_internal_end",
    "contributes_internal_start",
    "has_spanning_alignment",
    "enters_peak_end_count",
    "enters_peak_start_count",
}


def write_peak_target_annotations(
    connection: duckdb.DuckDBPyConnection,
    peak_targets_path: Path,
    contig_best_hits_path: Path,
    output_path: Path,
) -> None:
    """Write one row per target contig relevant to one selected score peak.

    The selected score geometry determines which endpoint flags are counted::

        local          local_end          | b |          local_start
        directional    directional_end    | b |    directional_start
        overlap          overlap_start    | b |      overlap_end
        one_sided_end    internal_end     | b |      internal_start
        one_sided_start  internal_end     | b |      internal_start

    A target spanning ``[b - radius_nt, b + radius_nt]`` is retained even when
    it contributes to neither selected endpoint count. It is the concrete
    target contig behind ``spanning_target_count``.
    """

    peak_targets = quote_sql_string(peak_targets_path)
    hits = quote_sql_string(contig_best_hits_path)
    output = quote_sql_string(output_path)
    connection.execute(
        f"""
        COPY (
          WITH relevant AS (
            SELECT *
            FROM read_parquet({peak_targets})
            WHERE enters_peak_end_count
               OR enters_peak_start_count
               OR has_spanning_alignment
          )
          SELECT
            r.*,
            h.qseqid IS NOT NULL AS has_diamond_hit,
            h.* EXCLUDE (qseqid)
          FROM relevant r
          LEFT JOIN read_parquet({hits}) h
            ON r.target_id = h.qseqid
          ORDER BY r.query_id, r.peak_id, r.target_id
        ) TO {output} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def count_annotation_audit_mismatches(
    connection: duckdb.DuckDBPyConnection,
    peaks_path: Path,
    annotations_path: Path,
) -> int:
    """Count peaks whose retained target rows do not reproduce stored counts."""

    peaks = quote_sql_string(peaks_path)
    annotations = quote_sql_string(annotations_path)
    return int(
        connection.execute(
            f"""
            WITH observed AS (
              SELECT
                peak_id,
                sum(enters_peak_end_count::INTEGER)::BIGINT AS end_count,
                sum(enters_peak_start_count::INTEGER)::BIGINT AS start_count,
                sum(has_spanning_alignment::INTEGER)::BIGINT AS spanning_count
              FROM read_parquet({annotations})
              GROUP BY peak_id
            )
            SELECT count(*)
            FROM read_parquet({peaks}) p
            LEFT JOIN observed o USING (peak_id)
            WHERE p.end_target_count != coalesce(o.end_count, 0)
               OR p.start_target_count != coalesce(o.start_count, 0)
               OR p.spanning_target_count != coalesce(o.spanning_count, 0)
            """
        ).fetchone()[0]
    )


def annotate_peak_targets_with_diamond_hits(
    peaks_path: Path,
    peak_targets_path: Path,
    contig_best_hits_path: Path,
    output_path: Path,
    *,
    threads: int = 8,
    force: bool = False,
    validate_input_tables: bool = True,
) -> dict[str, object]:
    """Join peak targets to DIAMOND hits and audit all retained target counts.

    Disable schema and ID checks only for tables already checked or generated
    by the screening workflow. Output-count audits remain enabled.
    """

    for description, path in (
        ("boundary score peak table", peaks_path),
        ("peak target table", peak_targets_path),
        ("contig DIAMOND best-hit table", contig_best_hits_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path}")
    if threads < 1:
        raise ValueError("threads must be >= 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.parquet")
    partial_path.unlink(missing_ok=True)

    try:
        with duckdb.connect() as connection:
            connection.execute(f"PRAGMA threads={threads}")

            if validate_input_tables:
                missing_peaks = sorted(
                    PEAK_COLUMNS - parquet_columns(connection, peaks_path)
                )
                if missing_peaks:
                    raise ValueError(f"peak table is missing columns: {missing_peaks}")

                missing_targets = sorted(
                    PEAK_TARGET_COLUMNS - parquet_columns(connection, peak_targets_path)
                )
                if missing_targets:
                    raise ValueError(
                        f"peak target table is missing columns: {missing_targets}"
                    )

                hit_columns = parquet_columns(connection, contig_best_hits_path)
                missing_hits = sorted({"qseqid", "has_viral_hit"} - hit_columns)
                if missing_hits:
                    raise ValueError(
                        f"contig DIAMOND best-hit table is missing columns: {missing_hits}"
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

                duplicate_peak_target_count = int(
                    connection.execute(
                        f"""
                        SELECT count(*) - count(DISTINCT (peak_id, target_id))
                        FROM read_parquet({quote_sql_string(peak_targets_path)})
                        """
                    ).fetchone()[0]
                )
                if duplicate_peak_target_count:
                    raise ValueError("peak_id + target_id pairs must be unique")

                duplicate_hit_count = int(
                    connection.execute(
                        f"""
                        SELECT count(*) - count(DISTINCT qseqid)
                        FROM read_parquet({quote_sql_string(contig_best_hits_path)})
                        """
                    ).fetchone()[0]
                )
                if duplicate_hit_count:
                    raise ValueError("qseqid values in contig best hits must be unique")

            write_peak_target_annotations(
                connection,
                peak_targets_path,
                contig_best_hits_path,
                partial_path,
            )
            mismatch_count = count_annotation_audit_mismatches(
                connection,
                peaks_path,
                partial_path,
            )
            if mismatch_count:
                raise ValueError(
                    f"annotated target rows disagree with {mismatch_count} peak counts"
                )

            row = connection.execute(
                f"""
                SELECT
                  count(*)::BIGINT,
                  count(DISTINCT peak_id)::BIGINT,
                  count(DISTINCT target_id)::BIGINT,
                  coalesce(count_if(has_diamond_hit), 0)::BIGINT,
                  coalesce(count_if(coalesce(has_viral_hit, false)), 0)::BIGINT
                FROM read_parquet({quote_sql_string(partial_path)})
                """
            ).fetchone()

        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    return {
        "output": str(output_path),
        "peak_target_annotation_count": int(row[0]),
        "peak_count": int(row[1]),
        "distinct_target_count": int(row[2]),
        "diamond_hit_row_count": int(row[3]),
        "viral_diamond_hit_row_count": int(row[4]),
        "count_mismatch_count": 0,
    }
