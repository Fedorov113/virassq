"""Thin Click entry point for screening prepared contig tables."""

from __future__ import annotations

from pathlib import Path

import click
import duckdb

from virassq.workflow import screen_representatives


@click.group()
@click.version_option(package_name="virassq")
def cli() -> None:
    """Check viral representatives using contig-clustering alignments."""


@cli.command("screen")
@click.option(
    "--representatives",
    "representatives_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
    help="FASTA, one-ID-per-line .txt/.ids (also .gz), or representative Parquet.",
)
@click.option(
    "--alignments",
    "alignments_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
    help="14-column MMseqs2 .tsv/.tsv.gz, or normalized .parquet without self hits.",
)
@click.option(
    "--annotations",
    "contig_best_hits_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
    help="Contig-level DIAMOND best-hit Parquet table keyed by qseqid.",
)
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="New or empty directory; existing results are never overwritten.",
)
@click.option("--threads", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--workers", type=click.IntRange(min=1), default=1, show_default=True)
@click.option(
    "--radius-nt",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help="Window radius around each representative coordinate, in nucleotides.",
)
@click.option(
    "--min-two-sided-peak-score",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Minimum local, directional, and overlap score for peak selection.",
)
@click.option(
    "--min-one-sided-peak-score",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Minimum one-sided score for peak selection, not a quarantine rule.",
)
def screen_command(
    representatives_path: Path,
    alignments_path: Path,
    contig_best_hits_path: Path,
    output_directory: Path,
    threads: int,
    workers: int,
    radius_nt: int,
    min_two_sided_peak_score: int,
    min_one_sided_peak_score: int,
) -> None:
    """Screen contigs using MMseqs2 TSV or normalized Parquet alignments.

    TSV coordinates are normalized automatically and self hits removed.
    Representatives may be FASTA, TXT or Parquet; DIAMOND annotations are Parquet.
    Report candidate boundaries and quarantine IDs. Input sequences are not
    changed, trimmed or split, and reclustering is not performed.
    """

    try:
        result = screen_representatives(
            representatives_path,
            alignments_path,
            contig_best_hits_path,
            output_directory,
            radius_nt=radius_nt,
            min_two_sided_peak_score=min_two_sided_peak_score,
            min_one_sided_peak_score=min_one_sided_peak_score,
            threads=threads,
            workers=workers,
        )
    except (OSError, ValueError, duckdb.Error) as error:
        raise click.ClickException(str(error)) from error

    counts = result["representative_action_counts"]
    for action in ("no_selected_boundary", "review", "quarantine"):
        click.echo(f"{action}: {counts.get(action, 0)} representatives")
    click.echo(f"Representative actions: {result['representative_actions']}")
    click.echo(f"Quarantine IDs: {result['quarantine_query_ids']}")
    click.echo(f"Run manifest: {output_directory / 'screen.manifest.json'}")
