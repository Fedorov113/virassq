"""Build one persisted MMseqs2 search for clustering and alignment review.

The expensive nucleotide search is shared by two downstream views::

    raw contigs FASTA
            |
            | createdb; search raw against raw
            v
      persisted resultDB
            |
            +--> clust ----------------> clusters and representatives
            |
            `--> convertalis ----------> raw alignment rows

Keeping ``resultDB`` is the important contract. Clustering does not create the
exported alignment rows; both clustering and ``convertalis`` consume the same
search result. Composite decisions belong to later modules.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from segonaut.reference.composite.alignments import MMSEQS_FORMAT_OUTPUT


def mmseqs_database_exists(prefix: Path) -> bool:
    """Return whether an MMseqs2 database prefix has its required type file."""
    return Path(f"{prefix}.dbtype").is_file()


def build_raw_clustering_commands(
    contigs: Path,
    output_directory: Path,
    temporary_directory: Path,
    *,
    mmseqs: str = "mmseqs",
    threads: int = 32,
    min_sequence_identity: float = 0.90,
    min_alignment_length: int = 300,
    min_target_coverage: float = 0.80,
    sensitivity: float = 7.5,
    max_hits_per_query: int = 1_000,
    max_sequence_length: int = 1_000_000,
) -> tuple[list[list[str]], dict[str, Path]]:
    """Build the explicit persisted-search clustering workflow.

    ``cov-mode 1`` means that the coverage threshold applies to each target.
    A short target may therefore support one local region of a longer query.
    ``cluster-mode 2`` then greedily selects representatives by sequence length
    from the already filtered search result.
    """
    database_directory = output_directory / "db"
    sequence_database = database_directory / "contigs"
    result_database = database_directory / "raw_to_raw"
    cluster_database = database_directory / "clusters"
    representative_database = database_directory / "representatives"
    clusters_tsv = output_directory / "clusters.tsv"
    representatives_fasta = output_directory / "representatives.fasta"

    commands = [
        [
            mmseqs,
            "createdb",
            str(contigs),
            str(sequence_database),
            "--threads",
            str(threads),
        ],
        [
            mmseqs,
            "search",
            str(sequence_database),
            str(sequence_database),
            str(result_database),
            str(temporary_directory / "search"),
            "--search-type",
            "3",
            "--alignment-mode",
            "3",
            "--min-seq-id",
            str(min_sequence_identity),
            "--min-aln-len",
            str(min_alignment_length),
            "-c",
            str(min_target_coverage),
            "--cov-mode",
            "1",
            "-s",
            str(sensitivity),
            "--max-seqs",
            str(max_hits_per_query),
            "--max-seq-len",
            str(max_sequence_length),
            "--threads",
            str(threads),
            "--remove-tmp-files",
            "1",
        ],
        [
            mmseqs,
            "clust",
            str(sequence_database),
            str(result_database),
            str(cluster_database),
            "--cluster-mode",
            "2",
            "--threads",
            str(threads),
        ],
        [
            mmseqs,
            "createtsv",
            str(sequence_database),
            str(sequence_database),
            str(cluster_database),
            str(clusters_tsv),
            "--threads",
            str(threads),
        ],
        [
            mmseqs,
            "createsubdb",
            str(cluster_database),
            str(sequence_database),
            str(representative_database),
        ],
        [
            mmseqs,
            "convert2fasta",
            str(representative_database),
            str(representatives_fasta),
        ],
    ]
    outputs = {
        "sequence_database": sequence_database,
        "result_database": result_database,
        "cluster_database": cluster_database,
        "representative_database": representative_database,
        "clusters_tsv": clusters_tsv,
        "representatives_fasta": representatives_fasta,
    }
    return commands, outputs


def run_raw_clustering(
    contigs: Path,
    output_directory: Path,
    temporary_directory: Path,
    *,
    mmseqs: str = "mmseqs",
    threads: int = 32,
    min_sequence_identity: float = 0.90,
    min_alignment_length: int = 300,
    min_target_coverage: float = 0.80,
    sensitivity: float = 7.5,
    max_hits_per_query: int = 1_000,
    max_sequence_length: int = 1_000_000,
) -> dict[str, object]:
    """Run raw clustering while retaining the search and cluster databases."""
    if not contigs.is_file():
        raise FileNotFoundError(f"missing raw-contig FASTA: {contigs}")
    if shutil.which(mmseqs) is None:
        raise FileNotFoundError(f"MMseqs2 executable not found: {mmseqs}")

    commands, outputs = build_raw_clustering_commands(
        contigs,
        output_directory,
        temporary_directory,
        mmseqs=mmseqs,
        threads=threads,
        min_sequence_identity=min_sequence_identity,
        min_alignment_length=min_alignment_length,
        min_target_coverage=min_target_coverage,
        sensitivity=sensitivity,
        max_hits_per_query=max_hits_per_query,
        max_sequence_length=max_sequence_length,
    )
    sequence_database_exists = mmseqs_database_exists(outputs["sequence_database"])
    if outputs["sequence_database"].exists() and not sequence_database_exists:
        raise FileExistsError(
            f"incomplete MMseqs2 sequence database: {outputs['sequence_database']}"
        )

    # A failed search may leave a complete, expensive sequenceDB behind. It is
    # safe to reuse because createdb has no scientific thresholds; all search
    # and clustering outputs must still be absent before this workflow starts.
    downstream_outputs = [
        path
        for name, path in outputs.items()
        if name != "sequence_database"
        and (path.exists() or mmseqs_database_exists(path))
    ]
    if downstream_outputs:
        raise FileExistsError(
            f"raw clustering outputs already exist: {downstream_outputs[:3]}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "db").mkdir(parents=True, exist_ok=True)
    temporary_directory.mkdir(parents=True, exist_ok=True)
    commands_to_run = commands[1:] if sequence_database_exists else commands
    for command in commands_to_run:
        # Output is inherited so Snakemake, tmux, or a scheduler can own logs.
        subprocess.run(command, check=True)

    if not outputs["clusters_tsv"].is_file():
        raise RuntimeError("MMseqs2 did not produce the cluster membership TSV")
    if not outputs["representatives_fasta"].is_file():
        raise RuntimeError("MMseqs2 did not produce the representative FASTA")

    manifest = {
        "schema": "segonaut.persisted_raw_clustering.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "contigs": str(contigs),
        "commands": commands,
        "reused_sequence_database": sequence_database_exists,
        "thresholds": {
            "min_sequence_identity": min_sequence_identity,
            "min_alignment_length_nt": min_alignment_length,
            "min_target_coverage": min_target_coverage,
            "sensitivity": sensitivity,
            "max_hits_per_query": max_hits_per_query,
            "max_sequence_length_nt": max_sequence_length,
            "coverage_mode": 1,
            "cluster_mode": 2,
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest_path = output_directory / "raw_clustering.run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path)}


def build_export_search_alignments_command(
    sequence_database: Path,
    result_database: Path,
    output_tsv: Path,
    *,
    mmseqs: str = "mmseqs",
    threads: int = 32,
) -> list[str]:
    """Build ``convertalis`` for the fixed raw-alignment table contract."""
    return [
        mmseqs,
        "convertalis",
        str(sequence_database),
        str(sequence_database),
        str(result_database),
        str(output_tsv),
        "--format-output",
        MMSEQS_FORMAT_OUTPUT,
        "--threads",
        str(threads),
    ]


def export_search_alignments(
    sequence_database: Path,
    result_database: Path,
    output_tsv: Path,
    *,
    mmseqs: str = "mmseqs",
    threads: int = 32,
) -> dict[str, object]:
    """Export every retained query-target row from a persisted resultDB."""
    if not mmseqs_database_exists(sequence_database):
        raise FileNotFoundError(
            f"missing MMseqs2 sequence database: {sequence_database}"
        )
    if not mmseqs_database_exists(result_database):
        raise FileNotFoundError(f"missing MMseqs2 result database: {result_database}")
    if output_tsv.exists():
        raise FileExistsError(f"alignment TSV already exists: {output_tsv}")
    if shutil.which(mmseqs) is None:
        raise FileNotFoundError(f"MMseqs2 executable not found: {mmseqs}")

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    command = build_export_search_alignments_command(
        sequence_database,
        result_database,
        output_tsv,
        mmseqs=mmseqs,
        threads=threads,
    )
    subprocess.run(command, check=True)
    if not output_tsv.is_file() or output_tsv.stat().st_size == 0:
        raise RuntimeError(f"MMseqs2 produced no alignment rows: {output_tsv}")

    manifest = {
        "schema": "segonaut.mmseqs_alignment_export.v1",
        "sequence_database": str(sequence_database),
        "result_database": str(result_database),
        "output_tsv": str(output_tsv),
        "command": command,
        "output_bytes": output_tsv.stat().st_size,
    }
    manifest_path = Path(f"{output_tsv}.run.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path)}
