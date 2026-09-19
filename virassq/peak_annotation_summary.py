r"""Summarize target-contig DIAMOND fields separately for each side of a peak.

This module describes what the target contigs annotate. It does not decide
whether different annotations imply a composite representative::

    peak_target_annotations.parquet
                  |
                  +--> enters_peak_end_count   --> end_* columns
                  +--> enters_peak_start_count --> start_* columns
                  `--> has_spanning_alignment  --> spanning_* counts
                                                   |
                                                   v
                                    peak_annotation_summary.parquet
                                    one row per peak_id

A short target may enter both endpoint counts. It then contributes once to the
end summary and once to the start summary; it is still one physical target in
``peak_target_annotations.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from virassq.sql import parquet_columns, quote_sql_string
from virassq.viral_protein_titles import classify_viral_protein_title

# These are descriptive DIAMOND fields. ``best_*`` describes the top hit
# overall; ``best_viral_*`` preserves the top viral hit even when a cellular
# hit has a larger bitscore.
DIAMOND_ANNOTATION_COLUMNS = (
    "best_domain",
    "best_family",
    "best_genus",
    "best_species",
    "best_title",
    "best_viral_family",
    "best_viral_genus",
    "best_viral_species",
    "best_viral_title",
)

ANNOTATION_COLUMNS = (
    *DIAMOND_ANNOTATION_COLUMNS,
    "viral_protein_role",
    "viral_protein_named_tokens",
)

PEAK_COLUMNS = {
    "peak_id",
    "end_target_count",
    "start_target_count",
    "spanning_target_count",
}

ANNOTATED_TARGET_COLUMNS = {
    "peak_id",
    "target_id",
    "enters_peak_end_count",
    "enters_peak_start_count",
    "has_spanning_alignment",
    "has_diamond_hit",
    "has_viral_hit",
    "best_hit_is_viral",
    *DIAMOND_ANNOTATION_COLUMNS,
}

MISSING_ANNOTATION_LABELS = ("", "NA", "N/A", "NONE", "NULL", "UNKNOWN")


def classify_distinct_viral_titles(
    connection: duckdb.DuckDBPyConnection,
    annotations_path: Path,
) -> pd.DataFrame:
    """Classify each distinct viral DIAMOND title once.

    The exact title remains the join key and stays in the output. Functional
    roles and explicit VP/NS/L/M/S tokens are derived only from words present
    in that title::

        "RNA-dependent RNA polymerase [Virus X]" -> role=rdrp
        "hypothetical protein [Virus X]"          -> role=unknown
        "segment L protein [Virus X]"             -> token=L

    This small in-memory table avoids running the same regular expressions for
    every peak-target occurrence of one database title.
    """

    titles = connection.execute(
        f"""
        SELECT DISTINCT best_viral_title
        FROM read_parquet({quote_sql_string(annotations_path)})
        WHERE best_viral_title IS NOT NULL
          AND trim(best_viral_title) != ''
        ORDER BY best_viral_title
        """
    ).df()
    rows = []
    for title in titles["best_viral_title"]:
        labels = classify_viral_protein_title(title)
        named_tokens = labels["viral_protein_named_tokens"]
        rows.append(
            {
                "best_viral_title": title,
                "viral_protein_role": labels["viral_protein_role"],
                "viral_protein_named_tokens": (
                    "|".join(named_tokens) if named_tokens else None
                ),
                "viral_protein_role_ambiguous": labels["viral_protein_role_ambiguous"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "best_viral_title",
            "viral_protein_role",
            "viral_protein_named_tokens",
            "viral_protein_role_ambiguous",
        ],
    )


def annotation_values_sql() -> str:
    """Return the explicit long-form projection of DIAMOND text columns."""

    rows = ",\n".join(
        f"              ('{column}', CAST(t.{column} AS VARCHAR))"
        for column in ANNOTATION_COLUMNS
    )
    return f"""CROSS JOIN LATERAL (VALUES
{rows}
            ) annotations(annotation_name, annotation_value)"""


def dominant_annotation_columns_sql(alias: str, side: str) -> str:
    """Return final columns for one side of one peak.

    Ties are already resolved alphabetically in ``ranked_annotations``. The
    fraction denominator is the number of targets with a non-missing value for
    that exact DIAMOND column, not all targets at the peak.
    """

    columns = []
    for name in ANNOTATION_COLUMNS:
        columns.extend(
            [
                f"{alias}.dominant_{name} AS {side}_dominant_{name}",
                (
                    f"coalesce({alias}.dominant_{name}_target_count, 0)::BIGINT "
                    f"AS {side}_dominant_{name}_target_count"
                ),
                (
                    f"coalesce({alias}.dominant_{name}_fraction, 0.0)::DOUBLE "
                    f"AS {side}_dominant_{name}_fraction_among_nonmissing"
                ),
                (
                    f"coalesce({alias}.distinct_{name}_count, 0)::BIGINT "
                    f"AS {side}_distinct_{name}_count"
                ),
            ]
        )
    return ",\n            ".join(columns)


def dominant_annotation_pivot_sql() -> str:
    """Pivot the strongest value for every DIAMOND column back to one row."""

    columns = []
    for name in ANNOTATION_COLUMNS:
        columns.extend(
            [
                (
                    f"max(annotation_value) FILTER "
                    f"(WHERE annotation_name = '{name}') AS dominant_{name}"
                ),
                (
                    f"max(annotation_value_target_count) FILTER "
                    f"(WHERE annotation_name = '{name}') "
                    f"AS dominant_{name}_target_count"
                ),
                (
                    f"max(annotation_value_fraction) FILTER "
                    f"(WHERE annotation_name = '{name}') "
                    f"AS dominant_{name}_fraction"
                ),
                (
                    f"max(distinct_annotation_value_count) FILTER "
                    f"(WHERE annotation_name = '{name}') "
                    f"AS distinct_{name}_count"
                ),
            ]
        )
    return ",\n              ".join(columns)


def write_peak_annotation_summary(
    connection: duckdb.DuckDBPyConnection,
    peaks_path: Path,
    annotations_path: Path,
    viral_title_roles: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write one descriptive DIAMOND summary row per selected score peak."""

    peaks = quote_sql_string(peaks_path)
    targets = quote_sql_string(annotations_path)
    output = quote_sql_string(output_path)
    missing_labels = ", ".join(f"'{value}'" for value in MISSING_ANNOTATION_LABELS)

    values_sql = annotation_values_sql()
    pivot_sql = dominant_annotation_pivot_sql()
    end_columns = dominant_annotation_columns_sql("ea", "end")
    start_columns = dominant_annotation_columns_sql("sa", "start")
    connection.register("viral_title_roles", viral_title_roles)

    connection.execute(
        f"""
        COPY (
          -- One target can appear on both sides. UNION ALL is deliberate:
          -- each side is summarized independently in the following CTEs.
          WITH side_targets AS (
            SELECT 'end' AS peak_side, t.*, r.* EXCLUDE (best_viral_title)
            FROM read_parquet({targets}) t
            LEFT JOIN viral_title_roles r USING (best_viral_title)
            WHERE t.enters_peak_end_count
            UNION ALL
            SELECT 'start' AS peak_side, t.*, r.* EXCLUDE (best_viral_title)
            FROM read_parquet({targets}) t
            LEFT JOIN viral_title_roles r USING (best_viral_title)
            WHERE t.enters_peak_start_count
          ),
          side_counts AS (
            SELECT
              peak_id,
              peak_side,
              count(*)::BIGINT AS target_count,
              count_if(has_diamond_hit)::BIGINT AS diamond_hit_target_count,
              count_if(NOT has_diamond_hit)::BIGINT AS no_diamond_hit_target_count,
              count_if(coalesce(has_viral_hit, false))::BIGINT
                AS viral_hit_target_count,
              count_if(coalesce(best_hit_is_viral, false))::BIGINT
                AS best_hit_viral_target_count,
              count_if(
                has_diamond_hit AND NOT coalesce(best_hit_is_viral, false)
              )::BIGINT AS best_hit_not_viral_target_count,
              count_if(coalesce(viral_protein_role_ambiguous, false))::BIGINT
                AS ambiguous_viral_protein_role_target_count
            FROM side_targets
            GROUP BY peak_id, peak_side
          ),
          -- Convert the nine named DIAMOND columns to the same three-column
          -- shape so one transparent counting rule handles every field.
          annotation_values AS (
            SELECT
              t.peak_id,
              t.peak_side,
              t.target_id,
              annotations.annotation_name,
              trim(annotations.annotation_value) AS annotation_value
            FROM side_targets t
            {values_sql}
            WHERE annotations.annotation_value IS NOT NULL
              AND (
                annotations.annotation_name = 'viral_protein_role'
                OR upper(trim(annotations.annotation_value))
                    NOT IN ({missing_labels})
              )
          ),
          annotation_counts AS (
            SELECT
              peak_id,
              peak_side,
              annotation_name,
              annotation_value,
              count(DISTINCT target_id)::BIGINT AS annotation_value_target_count
            FROM annotation_values
            GROUP BY peak_id, peak_side, annotation_name, annotation_value
          ),
          ranked_annotations AS (
            SELECT
              *,
              annotation_value_target_count
                / sum(annotation_value_target_count) OVER (
                    PARTITION BY peak_id, peak_side, annotation_name
                  )::DOUBLE AS annotation_value_fraction,
              count(*) OVER (
                PARTITION BY peak_id, peak_side, annotation_name
              )::BIGINT AS distinct_annotation_value_count,
              row_number() OVER (
                PARTITION BY peak_id, peak_side, annotation_name
                ORDER BY annotation_value_target_count DESC, annotation_value ASC
              ) AS value_rank
            FROM annotation_counts
          ),
          dominant_annotations AS (
            SELECT
              peak_id,
              peak_side,
              {pivot_sql}
            FROM ranked_annotations
            WHERE value_rank = 1
            GROUP BY peak_id, peak_side
          ),
          spanning_counts AS (
            SELECT
              peak_id,
              count(*)::BIGINT AS target_count,
              count_if(has_diamond_hit)::BIGINT AS diamond_hit_target_count,
              count_if(NOT has_diamond_hit)::BIGINT AS no_diamond_hit_target_count,
              count_if(coalesce(has_viral_hit, false))::BIGINT
                AS viral_hit_target_count
            FROM read_parquet({targets})
            WHERE has_spanning_alignment
            GROUP BY peak_id
          )
          SELECT
            p.*,
            coalesce(ec.target_count, 0)::BIGINT AS end_observed_target_count,
            coalesce(ec.diamond_hit_target_count, 0)::BIGINT
              AS end_diamond_hit_target_count,
            coalesce(ec.no_diamond_hit_target_count, 0)::BIGINT
              AS end_no_diamond_hit_target_count,
            coalesce(ec.viral_hit_target_count, 0)::BIGINT
              AS end_viral_hit_target_count,
            coalesce(ec.best_hit_viral_target_count, 0)::BIGINT
              AS end_best_hit_viral_target_count,
            coalesce(ec.best_hit_not_viral_target_count, 0)::BIGINT
              AS end_best_hit_not_viral_target_count,
            coalesce(ec.ambiguous_viral_protein_role_target_count, 0)::BIGINT
              AS end_ambiguous_viral_protein_role_target_count,
            {end_columns},
            coalesce(sc.target_count, 0)::BIGINT AS start_observed_target_count,
            coalesce(sc.diamond_hit_target_count, 0)::BIGINT
              AS start_diamond_hit_target_count,
            coalesce(sc.no_diamond_hit_target_count, 0)::BIGINT
              AS start_no_diamond_hit_target_count,
            coalesce(sc.viral_hit_target_count, 0)::BIGINT
              AS start_viral_hit_target_count,
            coalesce(sc.best_hit_viral_target_count, 0)::BIGINT
              AS start_best_hit_viral_target_count,
            coalesce(sc.best_hit_not_viral_target_count, 0)::BIGINT
              AS start_best_hit_not_viral_target_count,
            coalesce(sc.ambiguous_viral_protein_role_target_count, 0)::BIGINT
              AS start_ambiguous_viral_protein_role_target_count,
            {start_columns},
            coalesce(sp.target_count, 0)::BIGINT AS spanning_observed_target_count,
            coalesce(sp.diamond_hit_target_count, 0)::BIGINT
              AS spanning_diamond_hit_target_count,
            coalesce(sp.no_diamond_hit_target_count, 0)::BIGINT
              AS spanning_no_diamond_hit_target_count,
            coalesce(sp.viral_hit_target_count, 0)::BIGINT
              AS spanning_viral_hit_target_count
          FROM read_parquet({peaks}) p
          LEFT JOIN side_counts ec
            ON p.peak_id = ec.peak_id AND ec.peak_side = 'end'
          LEFT JOIN side_counts sc
            ON p.peak_id = sc.peak_id AND sc.peak_side = 'start'
          LEFT JOIN dominant_annotations ea
            ON p.peak_id = ea.peak_id AND ea.peak_side = 'end'
          LEFT JOIN dominant_annotations sa
            ON p.peak_id = sa.peak_id AND sa.peak_side = 'start'
          LEFT JOIN spanning_counts sp ON p.peak_id = sp.peak_id
          ORDER BY p.query_id, p.score_geometry, p.peak_position
        ) TO {output} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def count_summary_mismatches(
    connection: duckdb.DuckDBPyConnection,
    summary_path: Path,
) -> int:
    """Count summaries that do not reproduce the three stored target counts."""

    summary = quote_sql_string(summary_path)
    return int(
        connection.execute(
            f"""
            SELECT count(*)
            FROM read_parquet({summary})
            WHERE end_target_count != end_observed_target_count
               OR start_target_count != start_observed_target_count
               OR spanning_target_count != spanning_observed_target_count
               OR end_observed_target_count
                    != end_diamond_hit_target_count
                     + end_no_diamond_hit_target_count
               OR start_observed_target_count
                    != start_diamond_hit_target_count
                     + start_no_diamond_hit_target_count
               OR spanning_observed_target_count
                    != spanning_diamond_hit_target_count
                     + spanning_no_diamond_hit_target_count
               OR end_diamond_hit_target_count
                    != end_best_hit_viral_target_count
                     + end_best_hit_not_viral_target_count
               OR start_diamond_hit_target_count
                    != start_best_hit_viral_target_count
                     + start_best_hit_not_viral_target_count
            """
        ).fetchone()[0]
    )


def summarize_peak_target_annotations(
    peaks_path: Path,
    annotations_path: Path,
    output_path: Path,
    *,
    threads: int = 8,
    force: bool = False,
) -> dict[str, object]:
    """Write and audit per-peak end/start/spanning DIAMOND summaries."""

    for description, path in (
        ("boundary score peak table", peaks_path),
        ("peak target annotation table", annotations_path),
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
            missing_peaks = sorted(
                PEAK_COLUMNS - parquet_columns(connection, peaks_path)
            )
            if missing_peaks:
                raise ValueError(f"peak table is missing columns: {missing_peaks}")

            missing_targets = sorted(
                ANNOTATED_TARGET_COLUMNS - parquet_columns(connection, annotations_path)
            )
            if missing_targets:
                raise ValueError(
                    f"peak target annotation table is missing columns: {missing_targets}"
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

            write_peak_annotation_summary(
                connection,
                peaks_path,
                annotations_path,
                classify_distinct_viral_titles(connection, annotations_path),
                partial_path,
            )
            mismatch_count = count_summary_mismatches(connection, partial_path)
            if mismatch_count:
                raise ValueError(
                    f"DIAMOND summaries disagree with {mismatch_count} peak counts"
                )

            (
                peak_count,
                different_domain_count,
                different_viral_family_count,
                different_viral_protein_role_count,
                named_protein_token_peak_count,
            ) = connection.execute(
                f"""
                    SELECT
                      count(*)::BIGINT,
                      coalesce(count_if(
                        end_dominant_best_domain IS NOT NULL
                        AND start_dominant_best_domain IS NOT NULL
                        AND end_dominant_best_domain != start_dominant_best_domain
                      ), 0)::BIGINT,
                      coalesce(count_if(
                        end_dominant_best_viral_family IS NOT NULL
                        AND start_dominant_best_viral_family IS NOT NULL
                        AND end_dominant_best_viral_family
                            != start_dominant_best_viral_family
                      ), 0)::BIGINT,
                      coalesce(count_if(
                        end_dominant_viral_protein_role IS NOT NULL
                        AND start_dominant_viral_protein_role IS NOT NULL
                        AND end_dominant_viral_protein_role
                            != start_dominant_viral_protein_role
                      ), 0)::BIGINT,
                      coalesce(count_if(
                        end_dominant_viral_protein_named_tokens IS NOT NULL
                        OR start_dominant_viral_protein_named_tokens IS NOT NULL
                      ), 0)::BIGINT
                    FROM read_parquet({quote_sql_string(partial_path)})
                    """
            ).fetchone()

        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    return {
        "output": str(output_path),
        "peak_count": int(peak_count),
        "different_dominant_best_domain_count": int(different_domain_count),
        "different_dominant_best_viral_family_count": int(different_viral_family_count),
        "different_dominant_viral_protein_role_count": int(
            different_viral_protein_role_count
        ),
        "named_protein_token_peak_count": int(named_protein_token_peak_count),
        "count_mismatch_count": 0,
    }
