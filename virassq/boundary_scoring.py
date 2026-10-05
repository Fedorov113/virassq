r"""Score alignment starts and ends for a selected set of NT representatives.

The batch stage keeps selection, scoring and later classification separate::

    selected representatives        normalized alignments
               \                         /
                \ query_id join         /
                 v                     v
              alignments split at representative boundaries
                         /       |       \
                    worker 1  worker 2  worker N
                         \       |       /
                          ordered Parquet parts
                                  |
                                  v
              one full position profile + compact summary

Each worker calls ``score_boundary_positions()`` independently. The full
profiles are written in bounded batches and merged into one public Parquet.
This module does not choose a composite threshold.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from virassq.boundaries import (
    score_boundary_positions,
)
from virassq.sql import parquet_columns, quote_sql_string

QUERY_COLUMNS = {"representative_contig_id"}
ALIGNMENT_COLUMNS = {
    "query_id",
    "target_id",
    "query_length",
    "query_from",
    "query_to",
}

BOUNDARY_PROFILE_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("query_length", pa.int64()),
        ("boundary_position", pa.int32()),
        ("radius_nt", pa.int64()),
        ("local_end_target_count", pa.int32()),
        ("local_start_target_count", pa.int32()),
        ("local_b_score", pa.int32()),
        ("directional_end_target_count", pa.int32()),
        ("directional_start_target_count", pa.int32()),
        ("directional_b_score", pa.int32()),
        ("overlap_end_target_count", pa.int32()),
        ("overlap_start_target_count", pa.int32()),
        ("overlap_b_score", pa.int32()),
        ("internal_end_target_count", pa.int32()),
        ("internal_start_target_count", pa.int32()),
        ("one_sided_end_score", pa.int32()),
        ("one_sided_start_score", pa.int32()),
        ("spanning_target_count", pa.int32()),
    ]
)

# On Linux, forked workers inherit this read-only frame without serializing it.
# The table is much smaller than the generated position profiles: for the
# PRJNA605178 benchmark, 583 thousand alignment rows produce 106 million
# profile rows. Workers therefore receive only integer row ranges.
_WORKER_ALIGNMENTS: pd.DataFrame | None = None


def log(message: str) -> None:
    """Write progress immediately during the per-representative loop."""

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_selected_alignments(
    con: duckdb.DuckDBPyConnection,
    queries: Path,
    alignments: Path,
) -> tuple[list[str], pd.DataFrame]:
    """Load five scoring columns for all selected representatives in one scan."""

    query_ids = [
        str(row[0])
        for row in con.execute(
            f"""
            SELECT representative_contig_id
            FROM read_parquet({quote_sql_string(queries)})
            ORDER BY representative_contig_id
            """
        ).fetchall()
    ]

    # Only these columns enter score_boundary_positions. Annotation columns
    # remain in the query table and can be joined to the compact summary later.
    selected = con.execute(
        f"""
        SELECT
          a.query_id,
          a.target_id,
          a.query_length,
          a.query_from,
          a.query_to
        FROM read_parquet({quote_sql_string(alignments)}) a
        JOIN read_parquet({quote_sql_string(queries)}) q
          ON a.query_id = q.representative_contig_id
        ORDER BY a.query_id, a.query_from, a.query_to, a.target_id
        """
    ).df()
    return query_ids, selected


def build_boundary_summary(
    query_ids: list[str],
    selected_alignments: pd.DataFrame,
    *,
    radius_nt: int,
) -> pd.DataFrame:
    """Describe scored alignments and retain selected representatives without hits."""

    if selected_alignments.empty:
        aligned = pd.DataFrame(
            columns=[
                "representative_contig_id",
                "query_length",
                "alignment_row_count",
                "target_contig_count",
            ]
        )
    else:
        aligned = (
            selected_alignments.groupby("query_id", sort=False)
            .agg(
                query_length=("query_length", "first"),
                alignment_row_count=("target_id", "size"),
                target_contig_count=("target_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"query_id": "representative_contig_id"})
        )
    aligned["radius_nt"] = radius_nt
    aligned["has_nonself_alignments"] = True

    scored_ids = set(aligned["representative_contig_id"])
    unaligned = pd.DataFrame(
        {
            "representative_contig_id": [
                query_id for query_id in query_ids if query_id not in scored_ids
            ],
            "query_length": None,
            "radius_nt": radius_nt,
            "has_nonself_alignments": False,
            "alignment_row_count": 0,
            "target_contig_count": 0,
        }
    )
    columns = [
        "representative_contig_id",
        "query_length",
        "radius_nt",
        "has_nonself_alignments",
        "alignment_row_count",
        "target_contig_count",
    ]
    return pd.concat([aligned[columns], unaligned[columns]], ignore_index=True)


def write_profile_batch(
    writer: pq.ParquetWriter | None,
    profiles: list[pd.DataFrame],
    output: Path,
) -> pq.ParquetWriter:
    """Append a bounded group of position profiles to one Parquet file."""

    table = pa.Table.from_pandas(
        pd.concat(profiles, ignore_index=True), preserve_index=False
    )
    if writer is None:
        writer = pq.ParquetWriter(output, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def write_empty_profiles(output: Path) -> None:
    """Write the normal profile schema when no query has an alignment."""

    empty = pa.Table.from_batches([], schema=BOUNDARY_PROFILE_SCHEMA)
    pq.write_table(empty, output, compression="zstd")


def write_summary(summary: pd.DataFrame, output: Path) -> None:
    """Write the compact one-row-per-representative table."""

    summary = summary.sort_values("representative_contig_id")
    integer_columns = [
        "query_length",
        "radius_nt",
        "alignment_row_count",
        "target_contig_count",
    ]
    summary[integer_columns] = summary[integer_columns].astype("Int64")
    pq.write_table(
        pa.Table.from_pandas(summary, preserve_index=False),
        output,
        compression="zstd",
    )


def split_alignment_row_ranges(
    selected_alignments: pd.DataFrame,
    *,
    workers: int,
) -> list[tuple[int, int]]:
    """Split complete representatives into similarly sized profile parts.

    Runtime is driven mainly by the number of query coordinates, not by the
    number of alignment rows. Splits therefore follow cumulative query length::

        Q1:  700 nt  ┐
        Q2: 1200 nt  ├── part 1, about the same profile rows
        Q3:  900 nt  ┘
        Q4: 2800 nt  ─── part 2

    A representative is never split between workers.
    """

    if selected_alignments.empty:
        return []

    query_ids = selected_alignments["query_id"].to_numpy()
    query_starts = np.r_[0, np.flatnonzero(query_ids[1:] != query_ids[:-1]) + 1]
    query_ends = np.r_[query_starts[1:], len(selected_alignments)]
    query_lengths = selected_alignments.iloc[query_starts]["query_length"].to_numpy(
        dtype=np.int64,
        copy=False,
    )
    part_count = min(workers, len(query_starts))
    if part_count == 1:
        return [(0, len(selected_alignments))]

    cumulative_rows = np.cumsum(query_lengths)
    desired_rows = cumulative_rows[-1] / part_count
    split_queries = np.searchsorted(
        cumulative_rows,
        desired_rows * np.arange(1, part_count),
        side="right",
    )
    split_queries = np.unique(np.r_[0, split_queries, len(query_starts)])
    return [
        (int(query_starts[left]), int(query_ends[right - 1]))
        for left, right in pairwise(split_queries)
        if left < right
    ]


def write_profile_part(
    row_range: tuple[int, int],
    output: Path,
    *,
    radius_nt: int,
    profile_batch_rows: int,
) -> tuple[int, int]:
    """Score one complete group of representatives and write one Parquet part."""

    if _WORKER_ALIGNMENTS is None:
        raise RuntimeError("parallel boundary-scoring input is not initialized")

    row_start, row_end = row_range
    alignments = _WORKER_ALIGNMENTS.iloc[row_start:row_end]
    profile_frames: list[pd.DataFrame] = []
    buffered_profile_rows = 0
    profile_rows = 0
    representative_count = 0
    writer: pq.ParquetWriter | None = None

    for _, query_alignments in alignments.groupby("query_id", sort=False):
        scores = score_boundary_positions(query_alignments, radius_nt=radius_nt)
        profile_frames.append(scores)
        buffered_profile_rows += len(scores)
        profile_rows += len(scores)
        representative_count += 1

        if buffered_profile_rows >= profile_batch_rows:
            writer = write_profile_batch(writer, profile_frames, output)
            profile_frames = []
            buffered_profile_rows = 0

    if profile_frames:
        writer = write_profile_batch(writer, profile_frames, output)
    if writer is not None:
        writer.close()
    return representative_count, profile_rows


def merge_profile_parts(parts: list[Path], output: Path) -> None:
    """Merge ordered worker parts into the single public profile Parquet."""

    writer: pq.ParquetWriter | None = None
    try:
        for part in parts:
            parquet = pq.ParquetFile(part)
            if writer is None:
                writer = pq.ParquetWriter(
                    output, parquet.schema_arrow, compression="zstd"
                )
            for row_group_index in range(parquet.num_row_groups):
                writer.write_table(parquet.read_row_group(row_group_index))
    finally:
        if writer is not None:
            writer.close()


def score_profiles_in_parallel(
    selected_alignments: pd.DataFrame,
    output: Path,
    *,
    radius_nt: int,
    workers: int,
    profile_batch_rows: int,
    temporary_directory: Path,
) -> tuple[int, int, int]:
    """Score independent representatives in worker processes."""

    global _WORKER_ALIGNMENTS

    row_ranges = split_alignment_row_ranges(selected_alignments, workers=workers)
    if not row_ranges:
        write_empty_profiles(output)
        return 0, 0, 0
    if len(row_ranges) == 1:
        _WORKER_ALIGNMENTS = selected_alignments
        try:
            representative_count, profile_rows = write_profile_part(
                row_ranges[0],
                output,
                radius_nt=radius_nt,
                profile_batch_rows=profile_batch_rows,
            )
            return representative_count, profile_rows, 1
        finally:
            _WORKER_ALIGNMENTS = None

    parts = [
        temporary_directory / f"profile-part-{index:03d}.parquet"
        for index in range(len(row_ranges))
    ]
    _WORKER_ALIGNMENTS = selected_alignments
    try:
        # Fork preserves one read-only copy of the selected alignment table.
        # Only row ranges and output paths cross process boundaries.
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=len(row_ranges), mp_context=context
        ) as pool:
            futures = [
                pool.submit(
                    write_profile_part,
                    row_range,
                    part,
                    radius_nt=radius_nt,
                    profile_batch_rows=profile_batch_rows,
                )
                for row_range, part in zip(row_ranges, parts, strict=True)
            ]
            results = [future.result() for future in futures]
    finally:
        _WORKER_ALIGNMENTS = None

    merge_profile_parts(parts, output)
    representative_count, profile_rows = (
        sum(values) for values in zip(*results, strict=True)
    )
    return representative_count, profile_rows, len(row_ranges)


def score_selected_representatives(
    queries: Path,
    alignments: Path,
    output_directory: Path,
    *,
    radius_nt: int = 50,
    threads: int = 32,
    workers: int = 8,
    profile_batch_rows: int = 500_000,
    force: bool = False,
    validate_input_tables: bool = True,
) -> dict[str, object]:
    """Score selected representatives and write profiles plus alignment counts.

    Disable schema and ID checks only for tables already checked or generated
    by the screening workflow. Output-count audits remain enabled.
    """

    inputs = {"selected representatives": queries, "normalized alignments": alignments}
    missing_files = [
        f"{name}: {path}" for name, path in inputs.items() if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing_files))
    if radius_nt < 1:
        raise ValueError("radius_nt must be >= 1")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if profile_batch_rows < 1:
        raise ValueError("profile_batch_rows must be >= 1")

    profiles = output_directory / "boundary_profiles.parquet"
    summary = output_directory / "boundary_summary.parquet"
    manifest = output_directory / "boundary_scoring.manifest.json"
    outputs = [profiles, summary, manifest]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "boundary scoring outputs already exist:\n" + "\n".join(existing)
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    profile_partial = profiles.with_name(f"{profiles.name}.partial")
    summary_partial = summary.with_name(f"{summary.name}.partial")
    for path in (profile_partial, summary_partial):
        if path.exists():
            path.unlink()

    started = time.monotonic()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    try:
        if validate_input_tables:
            missing_queries = sorted(QUERY_COLUMNS - parquet_columns(con, queries))
            missing_alignments = sorted(
                ALIGNMENT_COLUMNS - parquet_columns(con, alignments)
            )
            if missing_queries or missing_alignments:
                raise ValueError(
                    "input schema mismatch: "
                    f"queries={missing_queries}, alignments={missing_alignments}"
                )
            duplicate_queries = con.execute(
                f"""
                SELECT count(*) - count(DISTINCT representative_contig_id)
                FROM read_parquet({quote_sql_string(queries)})
                """
            ).fetchone()[0]
            if duplicate_queries:
                raise ValueError(
                    f"selected representative table has {duplicate_queries} duplicate IDs"
                )
        query_ids, selected_alignments = load_selected_alignments(
            con, queries, alignments
        )
    finally:
        con.close()

    log(
        f"loaded {len(selected_alignments):,} alignments for "
        f"{selected_alignments['query_id'].nunique():,} of {len(query_ids):,} representatives"
    )

    summary_frame = build_boundary_summary(
        query_ids,
        selected_alignments,
        radius_nt=radius_nt,
    )
    representatives_with_alignments = int(summary_frame["has_nonself_alignments"].sum())
    log(
        f"scoring {representatives_with_alignments:,} representatives with "
        f"up to {workers:,} worker processes"
    )
    with tempfile.TemporaryDirectory(
        prefix="boundary-profile-parts-",
        dir=output_directory,
    ) as temporary_directory:
        scored_representatives, profile_rows, worker_parts = score_profiles_in_parallel(
            selected_alignments,
            profile_partial,
            radius_nt=radius_nt,
            workers=workers,
            profile_batch_rows=profile_batch_rows,
            temporary_directory=Path(temporary_directory),
        )
    if scored_representatives != representatives_with_alignments:
        raise RuntimeError(
            "parallel scoring lost representatives: "
            f"expected={representatives_with_alignments}, scored={scored_representatives}"
        )
    expected_profile_rows = int(summary_frame["query_length"].sum())
    if profile_rows != expected_profile_rows:
        raise RuntimeError(
            "parallel scoring produced the wrong number of profile rows: "
            f"expected={expected_profile_rows}, scored={profile_rows}"
        )
    write_summary(summary_frame, summary_partial)

    profile_partial.replace(profiles)
    summary_partial.replace(summary)
    elapsed_seconds = round(time.monotonic() - started, 3)
    result: dict[str, object] = {
        "schema": "virassq.boundary_scores.v4",
        "created_at": datetime.now(UTC).isoformat(),
        "queries": str(queries),
        "alignments": str(alignments),
        "profiles": str(profiles),
        "summary": str(summary),
        "radius_nt": radius_nt,
        "threads": threads,
        "workers": worker_parts,
        "selected_representatives": len(query_ids),
        "representatives_with_alignments": representatives_with_alignments,
        "representatives_without_alignments": (
            len(query_ids) - representatives_with_alignments
        ),
        "selected_alignment_rows": len(selected_alignments),
        "profile_rows": profile_rows,
        "elapsed_seconds": elapsed_seconds,
    }
    manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log(
        f"wrote {len(summary_frame):,} summary rows and "
        f"{result['profile_rows']:,} profile rows in {elapsed_seconds}s"
    )
    return result
