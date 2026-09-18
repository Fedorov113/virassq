"""Import raw-contig MMseqs2 alignments for pre-clustering composite review.

The dataset run script searches one raw-contig FASTA against itself. Each TSV
row projects one target contig onto the coordinate system of one query contig::

    query contig   1 ================================================== Q
    target contig             ===================
                              |<-- alignment -->|
    query interval            qstart          qend
    target interval           tstart          tend

With the current MMseqs2 ``cov-mode 1`` search, the coverage threshold applies
to the target. This favors targets that fit almost completely into some region
of the query::

    query:    ===========================================================
    target:              ==================
                         |<-- >=80% of target -->|

This module owns only the mechanical file boundary::

    fixed 14-column MMseqs2 TSV
                 |
                 | explicit numeric types; no row filtering
                 v
    compressed raw_alignments.parquet

Identity, alignment-length and coverage thresholds are applied by MMseqs2 when
the TSV is produced. Later composite stages apply their own named guards. This
adapter deliberately does not remove self hits, normalize coordinates, collapse
clusters, or classify a contig.
"""

from pathlib import Path

import duckdb

# This order must match ``--format-output`` in the MMseqs2 run command. Keeping
# the contract here makes headerless historical TSV files unambiguous.
RAW_ALIGNMENT_COLUMNS = [
    "query_id",
    "target_id",
    "identity_pct",
    "aligned_length",
    "query_start",
    "query_end",
    "query_length",
    "target_start",
    "target_end",
    "target_length",
    "query_coverage",
    "target_coverage",
    "evalue",
    "bitscore",
]

MMSEQS_FORMAT_OUTPUT = (
    "query,target,pident,alnlen,qstart,qend,qlen,tstart,tend,tlen,"
    "qcov,tcov,evalue,bits"
)


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def alignment_input_sql(path: Path) -> str:
    """Return a DuckDB source expression for the raw-alignment contract.

    Parquet already carries column names and types. Headerless TSV does not, so
    it is read with :data:`RAW_ALIGNMENT_COLUMNS` in their fixed output order.
    The returned expression is used by both the importer and later large-table
    scans; it does not execute a query by itself.
    """
    if path.suffix == ".parquet":
        return f"read_parquet('{_sql_path(path)}')"

    # Read TSV fields as text first. Explicit casts during import make malformed
    # numeric values fail instead of being silently guessed from a huge file.
    columns = ",".join(f"'{name}':'VARCHAR'" for name in RAW_ALIGNMENT_COLUMNS)
    return (
        f"read_csv('{_sql_path(path)}',delim='\\t',header=false,"
        f"columns={{{columns}}})"
    )


def import_raw_alignments(
    input_path: Path,
    output_path: Path,
    *,
    threads: int = 32,
) -> dict[str, int]:
    """Convert a fixed-format MMseqs2 TSV to typed, compressed Parquet.

    Parameters
    ----------
    input_path
        Headerless 14-column MMseqs2 TSV, or an existing Parquet table with the
        same column names.
    output_path
        Destination Parquet file. Parent directories are created as needed.
    threads
        DuckDB worker count used for parsing and compression.

    Returns
    -------
    dict
        ``{"alignments": N}``, where ``N`` is the number of imported rows.

    Notes
    -----
    This is a lossless schema conversion at the row level. Raw coordinates and
    query/target direction are retained exactly because later stages need them
    to interpret alignment geometry. Invalid numeric text fails during an
    explicit cast instead of being silently coerced.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    source = alignment_input_sql(input_path)
    con.execute(
        f"""
        COPY (
          SELECT
            query_id::VARCHAR AS query_id,
            target_id::VARCHAR AS target_id,
            identity_pct::DOUBLE AS identity_pct,
            aligned_length::INTEGER AS aligned_length,
            query_start::INTEGER AS query_start,
            query_end::INTEGER AS query_end,
            query_length::INTEGER AS query_length,
            target_start::INTEGER AS target_start,
            target_end::INTEGER AS target_end,
            target_length::INTEGER AS target_length,
            query_coverage::DOUBLE AS query_coverage,
            target_coverage::DOUBLE AS target_coverage,
            evalue::DOUBLE AS evalue,
            bitscore::DOUBLE AS bitscore
          FROM {source}
        ) TO '{_sql_path(output_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    count = con.execute(
        f"SELECT count(*) FROM read_parquet('{_sql_path(output_path)}')"
    ).fetchone()[0]
    return {"alignments": int(count)}


def normalize_raw_alignments(
    input_path: Path,
    output_path: Path,
    *,
    threads: int = 32,
) -> dict[str, int]:
    """Remove self hits and add complete, strand-aware interval geometry.

    This is a mechanical pass over the imported MMseqs2 rows. It applies no
    additional identity, length, coverage, cluster, or composite thresholds.
    Raw directional coordinates are retained next to normalized intervals::

        same orientation (+1)
        query    100 --------------------> 700
        target    20 --------------------> 620

        reverse orientation (-1)
        query    100 --------------------> 700
        target   620 <-------------------- 20

    ``relative_orientation`` describes the target relative to the query. It is
    not a biological strand assignment because assembled contigs may themselves
    be stored in either orientation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    source = alignment_input_sql(input_path)

    counts = con.execute(
        f"""
        SELECT
          count(*)::BIGINT AS input_alignments,
          count(*) FILTER (WHERE query_id = target_id)::BIGINT AS self_alignments
        FROM {source}
        """
    ).fetchone()
    con.execute(
        f"""
        COPY (
          SELECT
            query_id,
            target_id,
            identity_pct,
            aligned_length,
            query_start,
            query_end,
            query_length,
            target_start,
            target_end,
            target_length,
            query_coverage,
            target_coverage,
            evalue,
            bitscore,
            least(query_start, query_end)::INTEGER AS query_from,
            greatest(query_start, query_end)::INTEGER AS query_to,
            least(target_start, target_end)::INTEGER AS target_from,
            greatest(target_start, target_end)::INTEGER AS target_to,
            CASE
              -- Comparing directions avoids INT32 overflow on long contigs;
              -- multiplying the two coordinate differences is equivalent but
              -- can exceed 2^31 even when every coordinate itself is valid.
              WHEN (query_end >= query_start) = (target_end >= target_start)
                THEN 1
              ELSE -1
            END::SMALLINT AS relative_orientation
          FROM {source}
          WHERE query_id <> target_id
        ) TO '{_sql_path(output_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return {
        "input_alignments": int(counts[0]),
        "self_alignments_removed": int(counts[1]),
        "normalized_alignments": int(counts[0] - counts[1]),
    }
