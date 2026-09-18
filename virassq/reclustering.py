r"""Recluster an MMseqs2 search result after removing quarantined contigs.

The expensive nucleotide search is not repeated. MMseqs2 databases use stable
numeric sequence keys, so the sequence and alignment databases can be reduced
with the official ``createsubdb`` and ``filterdb`` commands::

    sequenceDB                         raw_to_raw resultDB
        |                                      |
        | keep allowed sequence keys           | keep allowed query keys
        v                                      v
    filtered sequenceDB                 allowed query entries
                                               |
                                               | remove quarantine target rows
                                               v
                                        filtered resultDB
        \                                      /
         `--------------- clust --------------'
                           |
                           v
             clusters.tsv + representatives.fasta

This module performs only mechanical database filtering and clustering. The
quarantine IDs must come from a separate, explicit decision stage.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from segonaut.reference.composite.search import mmseqs_database_exists


def read_contig_ids(path: Path) -> list[str]:
    """Read a unique, non-empty contig-ID list while preserving input order."""

    if not path.is_file():
        raise FileNotFoundError(f"missing quarantine contig-ID file: {path}")
    contig_ids = [
        line.strip() for line in path.read_text().splitlines() if line.strip()
    ]
    if len(contig_ids) != len(set(contig_ids)):
        raise ValueError(f"quarantine contig-ID file contains duplicates: {path}")
    return contig_ids


def split_mmseqs_keys(
    lookup: Path,
    quarantine_contig_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Split sequence keys using FASTA identifiers from an MMseqs2 lookup."""

    if not lookup.is_file():
        raise FileNotFoundError(f"missing MMseqs2 sequence lookup: {lookup}")

    retained_keys = []
    quarantine_keys = []
    found_quarantine_ids = set()
    with lookup.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 2:
                continue
            sequence_key, contig_id = row[0], row[1]
            if contig_id in quarantine_contig_ids:
                quarantine_keys.append(sequence_key)
                found_quarantine_ids.add(contig_id)
            else:
                retained_keys.append(sequence_key)

    missing = quarantine_contig_ids - found_quarantine_ids
    if missing:
        raise ValueError(
            f"{len(missing)} quarantine contig IDs are absent from {lookup}: "
            f"{sorted(missing)[:3]}"
        )
    return retained_keys, quarantine_keys


def write_key_file(keys: list[str], output: Path) -> None:
    """Write one MMseqs2 numeric sequence key per line."""

    output.write_text("".join(f"{key}\n" for key in keys), encoding="utf-8")


def build_reclustering_commands(
    sequence_database: Path,
    result_database: Path,
    output_directory: Path,
    *,
    has_quarantine_contigs: bool,
    mmseqs: str,
    threads: int,
    cluster_mode: int,
) -> tuple[list[list[str]], dict[str, Path]]:
    """Build official MMseqs2 commands for filtered reclustering."""

    database_directory = output_directory / "db"
    retained_keys = output_directory / "retained_sequence_keys.txt"
    quarantine_keys = output_directory / "quarantine_sequence_keys.txt"
    filtered_sequences = database_directory / "contigs"
    allowed_query_alignments = database_directory / "allowed_query_alignments"
    filtered_alignments = database_directory / "raw_to_raw"
    clusters = database_directory / "clusters"
    representatives = database_directory / "representatives"
    clusters_tsv = output_directory / "clusters.tsv"
    representatives_fasta = output_directory / "representatives.fasta"

    commands = [
        [
            mmseqs,
            "createsubdb",
            str(retained_keys),
            str(sequence_database),
            str(filtered_sequences),
        ]
    ]
    if has_quarantine_contigs:
        commands.extend(
            [
                [
                    mmseqs,
                    "createsubdb",
                    str(retained_keys),
                    str(result_database),
                    str(allowed_query_alignments),
                ],
                [
                    mmseqs,
                    "filterdb",
                    str(allowed_query_alignments),
                    str(filtered_alignments),
                    "--filter-file",
                    str(quarantine_keys),
                    "--filter-column",
                    "1",
                    "--positive-filter",
                    "0",
                    "--threads",
                    str(threads),
                ],
            ]
        )
    else:
        commands.append(
            [
                mmseqs,
                "createsubdb",
                str(retained_keys),
                str(result_database),
                str(filtered_alignments),
            ]
        )

    commands.extend(
        [
            [
                mmseqs,
                "clust",
                str(filtered_sequences),
                str(filtered_alignments),
                str(clusters),
                "--cluster-mode",
                str(cluster_mode),
                "--threads",
                str(threads),
            ],
            [
                mmseqs,
                "createtsv",
                str(filtered_sequences),
                str(filtered_sequences),
                str(clusters),
                str(clusters_tsv),
                "--threads",
                str(threads),
            ],
            [
                mmseqs,
                "createsubdb",
                str(clusters),
                str(filtered_sequences),
                str(representatives),
            ],
            [
                mmseqs,
                "convert2fasta",
                str(representatives),
                str(representatives_fasta),
            ],
        ]
    )
    outputs = {
        "retained_sequence_keys": retained_keys,
        "quarantine_sequence_keys": quarantine_keys,
        "sequence_database": filtered_sequences,
        "result_database": filtered_alignments,
        "cluster_database": clusters,
        "representative_database": representatives,
        "clusters_tsv": clusters_tsv,
        "representatives_fasta": representatives_fasta,
    }
    return commands, outputs


def count_lines(path: Path) -> int:
    """Count rows in a compact text artifact without loading it into memory."""

    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def run_reclustering_after_quarantine(
    sequence_database: Path,
    result_database: Path,
    quarantine_contig_ids_path: Path,
    output_directory: Path,
    *,
    mmseqs: str = "mmseqs",
    threads: int = 32,
    cluster_mode: int = 2,
) -> dict[str, object]:
    """Filter quarantine queries/targets and rerun MMseqs2 clustering."""

    if not mmseqs_database_exists(sequence_database):
        raise FileNotFoundError(
            f"missing MMseqs2 sequence database: {sequence_database}"
        )
    if not mmseqs_database_exists(result_database):
        raise FileNotFoundError(f"missing MMseqs2 result database: {result_database}")
    if shutil.which(mmseqs) is None:
        raise FileNotFoundError(f"MMseqs2 executable not found: {mmseqs}")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    if cluster_mode not in {0, 1, 2, 3}:
        raise ValueError("cluster_mode must be one of 0, 1, 2, or 3")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"reclustering output directory is not empty: {output_directory}"
        )

    quarantine_contig_ids = read_contig_ids(quarantine_contig_ids_path)
    lookup = Path(f"{sequence_database}.lookup")
    retained_keys, quarantine_keys = split_mmseqs_keys(
        lookup,
        set(quarantine_contig_ids),
    )
    if not retained_keys:
        raise ValueError("quarantine removes every sequence from the MMseqs2 database")

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "db").mkdir(parents=True, exist_ok=True)
    commands, outputs = build_reclustering_commands(
        sequence_database,
        result_database,
        output_directory,
        has_quarantine_contigs=bool(quarantine_keys),
        mmseqs=mmseqs,
        threads=threads,
        cluster_mode=cluster_mode,
    )
    write_key_file(retained_keys, outputs["retained_sequence_keys"])
    write_key_file(quarantine_keys, outputs["quarantine_sequence_keys"])

    for command in commands:
        # MMseqs2 output remains visible to tmux, Snakemake, or a scheduler.
        subprocess.run(command, check=True)

    if not outputs["clusters_tsv"].is_file():
        raise RuntimeError("MMseqs2 did not produce the reclustered membership TSV")
    if not outputs["representatives_fasta"].is_file():
        raise RuntimeError(
            "MMseqs2 did not produce the reclustered representative FASTA"
        )

    # FASTA sequences are currently one line in MMseqs2 output, but count the
    # headers directly so the manifest remains correct if wrapping changes.
    with outputs["representatives_fasta"].open(encoding="utf-8") as handle:
        cluster_count = sum(1 for line in handle if line.startswith(">"))

    result: dict[str, object] = {
        "schema": "segonaut.quarantine_reclustering.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_sequence_database": str(sequence_database),
        "source_result_database": str(result_database),
        "quarantine_contig_ids": str(quarantine_contig_ids_path),
        "quarantine_contig_count": len(quarantine_keys),
        "retained_contig_count": len(retained_keys),
        "cluster_count": cluster_count,
        "cluster_member_count": count_lines(outputs["clusters_tsv"]),
        "cluster_mode": cluster_mode,
        "threads": threads,
        "commands": commands,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest = output_directory / "quarantine_reclustering.run.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["manifest"] = str(manifest)
    return result
