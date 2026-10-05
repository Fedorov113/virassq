"""Run representative screening from MMseqs2 alignments and DIAMOND tables.

This module connects existing stages; it does not define quarantine rules::

    selected representatives + normalized MMseqs2 alignments
                              |
                              v
                    position profiles -> peaks
                                          |
    contig DIAMOND best hits -> peak targets -> annotation summaries
                                                    |
                                                    v
                                           review / quarantine

Representatives without a peak remain in the final table. The separate
``has_nonself_alignments`` column distinguishes an unexamined representative
from one whose measured profile contained no selected boundary.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from virassq.alignments import import_raw_alignments, normalize_raw_alignments
from virassq.boundary_score_peaks import (
    enabled_score_thresholds,
    select_boundary_score_peaks,
)
from virassq.boundary_scoring import (
    ALIGNMENT_COLUMNS,
    QUERY_COLUMNS,
    score_selected_representatives,
)
from virassq.decisions import (
    DEFAULT_PEAK_DECISION_THRESHOLDS,
    PeakDecisionThresholds,
    decide_peak_actions,
)
from virassq.decisions.thresholds import validate_peak_decision_thresholds
from virassq.peak_annotation_summary import (
    DIAMOND_ANNOTATION_COLUMNS,
    summarize_peak_target_annotations,
)
from virassq.peak_target_annotations import annotate_peak_targets_with_diamond_hits
from virassq.peak_targets import assign_peak_targets
from virassq.representatives import prepare_representatives, representative_input_format
from virassq.sql import parquet_columns, quote_sql_string


def validate_screen_inputs(
    representatives_path: Path,
    alignments_path: Path,
    contig_best_hits_path: Path,
    *,
    alignment_format: str = "normalized-parquet",
    representative_format: str = "parquet",
) -> None:
    """Check prepared schemas and identifiers before generating full profiles."""

    inputs = (
        (
            "representatives",
            representatives_path,
            QUERY_COLUMNS if representative_format == "parquet" else set(),
        ),
        (
            "MMseqs2 alignments",
            alignments_path,
            ALIGNMENT_COLUMNS if alignment_format == "normalized-parquet" else set(),
        ),
        (
            "contig DIAMOND best hits",
            contig_best_hits_path,
            {
                "qseqid",
                "has_viral_hit",
                "best_hit_is_viral",
                *DIAMOND_ANNOTATION_COLUMNS,
            },
        ),
    )
    with duckdb.connect() as connection:
        for name, path, required in inputs:
            if not path.is_file():
                raise FileNotFoundError(f"{name} do not exist: {path}")
            missing = (
                sorted(required - parquet_columns(connection, path)) if required else []
            )
            if missing:
                raise ValueError(f"{name} are missing columns: {missing}")
        for name, path, identifier in (
            ("representatives", representatives_path, "representative_contig_id"),
            ("contig DIAMOND best hits", contig_best_hits_path, "qseqid"),
        ):
            if name == "representatives" and representative_format != "parquet":
                continue
            invalid_count = connection.execute(
                f"SELECT count(*) - count(DISTINCT {identifier}) "
                f"FROM read_parquet({quote_sql_string(path)})"
            ).fetchone()[0]
            if invalid_count:
                raise ValueError(f"{name} contain duplicate or missing {identifier}")
        if alignment_format == "normalized-parquet":
            self_count = connection.execute(
                f"SELECT count(*) FROM read_parquet({quote_sql_string(alignments_path)}) "
                "WHERE query_id = target_id"
            ).fetchone()[0]
            if self_count:
                raise ValueError(
                    "normalized alignments contain self hits; "
                    "run normalize_raw_alignments before screening"
                )


def prepare_screen_alignments(
    input_path: Path,
    tables_directory: Path,
    *,
    alignment_format: str,
    threads: int,
) -> tuple[Path, dict[str, object]]:
    """Reuse normalized Parquet, or type and normalize a 14-column MMseqs2 TSV."""

    if alignment_format == "normalized-parquet":
        return input_path, {
            "input": str(input_path.resolve()),
            "output": str(input_path.resolve()),
            "input_format": alignment_format,
            "normalization_performed": False,
        }

    tables_directory.mkdir(parents=True, exist_ok=True)
    output_path = tables_directory / "normalized_alignments.parquet"
    partial_path = tables_directory / "normalized_alignments.partial.parquet"
    if output_path.exists() or partial_path.exists():
        raise FileExistsError(
            f"alignment preparation outputs already exist in {tables_directory}"
        )
    # The TSV importer casts coordinates before min/max calculations. Comparing
    # text coordinates directly would put, for example, '100' before '90'.
    try:
        with tempfile.TemporaryDirectory(
            prefix="alignment-import-", dir=tables_directory
        ) as temporary:
            raw_path = Path(temporary) / "raw_alignments.parquet"
            import_raw_alignments(input_path, raw_path, threads=threads)
            counts = normalize_raw_alignments(raw_path, partial_path, threads=threads)
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return output_path, {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "input_format": alignment_format,
        "normalization_performed": True,
        **counts,
    }


def write_all_representative_actions(
    boundary_summary_path: Path,
    peak_representative_actions_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Join peak actions to all selected IDs, retaining alignment availability."""

    representatives = pd.read_parquet(boundary_summary_path).rename(
        columns={"representative_contig_id": "query_id"}
    )
    actions = pd.read_parquet(peak_representative_actions_path)
    complete = representatives.merge(
        actions, on="query_id", how="left", validate="one_to_one"
    )
    no_peak = complete["peak_count"].isna()
    for column in ("peak_count", "quarantine_peak_count", "review_peak_count"):
        complete[column] = complete[column].fillna(0).astype("int64")
    for column in (
        "action_peak_ids",
        "matched_quarantine_rules",
        "matched_quarantine_blockers",
        "matched_review_rules",
    ):
        complete.loc[no_peak, column] = pd.Series(
            [[] for _ in range(int(no_peak.sum()))],
            index=complete.index[no_peak],
            dtype="object",
        )
    # No peak is not a certificate of a correct sequence, especially without
    # non-self alignments. Keep that distinction in the same output row.
    complete["representative_action"] = complete["representative_action"].fillna(
        "no_selected_boundary"
    )
    complete.sort_values("query_id").to_parquet(output_path, index=False)
    return {
        str(action): int(count)
        for action, count in complete["representative_action"].value_counts().items()
    }


def screen_representatives(
    representatives_path: Path,
    alignments_path: Path,
    contig_best_hits_path: Path,
    output_directory: Path,
    *,
    radius_nt: int = 50,
    min_two_sided_peak_score: int = 2,
    min_one_sided_peak_score: int = 10,
    decision_thresholds: PeakDecisionThresholds = DEFAULT_PEAK_DECISION_THRESHOLDS,
    threads: int = 8,
    workers: int = 1,
) -> dict[str, object]:
    """Screen MMseqs2 TSV or normalized Parquet without running a new search.

    Representatives are FASTA, one-ID-per-line text, or Parquet with
    ``representative_contig_id``. Only the selected IDs are read from FASTA;
    sequence bodies are not aligned or checked. Alignments are
    either headerless 14-column MMseqs2 ``.tsv`` (also ``.tsv.gz``) following
    ``MMSEQS_FORMAT_OUTPUT``, or normalized ``.parquet``. TSV coordinates are
    typed and normalized, and self hits removed. No extra identity, length or
    coverage filtering is applied. DIAMOND input contains one row per contig keyed
    by ``qseqid`` and the fields required by ``peak_annotation_summary``.
    Contigs without DIAMOND hits may be absent from the annotation table;
    they remain counted as nucleotide targets.

    Output must be a new or empty directory. The run retains positional
    profiles, exact target IDs, annotations, peak decisions, actions for all
    representatives, quarantine IDs, and a manifest with numerical thresholds.
    It does not modify inputs, remove sequences, or perform reclustering.
    """

    if radius_nt < 1 or threads < 1 or workers < 1:
        raise ValueError("radius_nt, threads and workers must be >= 1")
    enabled_score_thresholds(
        min_local_b_score=min_two_sided_peak_score,
        min_directional_b_score=min_two_sided_peak_score,
        min_overlap_b_score=min_two_sided_peak_score,
        min_one_sided_score=min_one_sided_peak_score,
    )
    validate_peak_decision_thresholds(decision_thresholds)
    representative_format = representative_input_format(representatives_path)
    if alignments_path.suffix.lower() == ".parquet":
        alignment_format = "normalized-parquet"
    elif alignments_path.name.lower().endswith((".tsv", ".tsv.gz")):
        alignment_format = "tsv"
    else:
        raise ValueError("alignment input must end in .tsv, .tsv.gz or .parquet")
    if output_directory.exists() and (
        not output_directory.is_dir() or any(output_directory.iterdir())
    ):
        raise FileExistsError(
            f"screen output directory is not empty: {output_directory}"
        )
    validate_screen_inputs(
        representatives_path,
        alignments_path,
        contig_best_hits_path,
        alignment_format=alignment_format,
        representative_format=representative_format,
    )

    scores_directory = output_directory / "boundary_scores"
    peaks_directory = output_directory / "boundary_score_peaks"
    tables_directory = output_directory / "tables"
    decisions_directory = output_directory / "decisions"
    peaks_path = peaks_directory / "boundary_score_peaks.parquet"
    targets_path = tables_directory / "peak_targets.parquet"
    annotations_path = tables_directory / "peak_target_annotations.parquet"
    summary_path = tables_directory / "peak_annotation_summary.parquet"
    actions_path = output_directory / "representative_actions.parquet"

    representatives_table_path, representative_preparation = prepare_representatives(
        representatives_path,
        tables_directory / "representatives.parquet",
    )
    normalized_alignments_path, alignment_preparation = prepare_screen_alignments(
        alignments_path,
        tables_directory,
        alignment_format=alignment_format,
        threads=threads,
    )
    # Original Parquet schemas and IDs were checked before preparation. Text
    # importers produce the same columns and reject duplicate representative
    # IDs. Later tables come from the preceding stage, so do not rescan their
    # schemas and IDs; each stage still audits its calculated target counts.
    scoring = score_selected_representatives(
        representatives_table_path,
        normalized_alignments_path,
        scores_directory,
        radius_nt=radius_nt,
        threads=threads,
        workers=workers,
        validate_input_tables=False,
    )
    peaks = select_boundary_score_peaks(
        scores_directory / "boundary_profiles.parquet",
        peaks_directory,
        min_local_b_score=min_two_sided_peak_score,
        min_directional_b_score=min_two_sided_peak_score,
        min_overlap_b_score=min_two_sided_peak_score,
        min_one_sided_score=min_one_sided_peak_score,
        threads=threads,
    )
    targets = assign_peak_targets(
        peaks_path,
        normalized_alignments_path,
        targets_path,
        threads=threads,
        validate_input_tables=False,
    )
    annotations = annotate_peak_targets_with_diamond_hits(
        peaks_path,
        targets_path,
        contig_best_hits_path,
        annotations_path,
        threads=threads,
        validate_input_tables=False,
    )
    summary = summarize_peak_target_annotations(
        peaks_path,
        annotations_path,
        summary_path,
        threads=threads,
        validate_input_tables=False,
    )
    decisions = decide_peak_actions(
        summary_path, decisions_directory, thresholds=decision_thresholds
    )
    action_counts = write_all_representative_actions(
        scores_directory / "boundary_summary.parquet",
        decisions_directory / "representative_actions.parquet",
        actions_path,
    )
    result = {
        "schema": "virassq.screen.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "representatives": str(representatives_path.resolve()),
        "representative_input_format": representative_format,
        "prepared_representatives": str(representatives_table_path.resolve()),
        "alignments": str(alignments_path.resolve()),
        "alignment_input_format": alignment_format,
        "normalized_alignments": str(normalized_alignments_path.resolve()),
        "contig_best_hits": str(contig_best_hits_path.resolve()),
        "radius_nt": radius_nt,
        "min_two_sided_peak_score": min_two_sided_peak_score,
        "min_one_sided_peak_score": min_one_sided_peak_score,
        "decision_thresholds": asdict(decision_thresholds),
        "threads": threads,
        "workers": workers,
        "representative_actions": str(actions_path),
        "representative_action_counts": action_counts,
        "quarantine_query_ids": decisions["quarantine_query_ids"],
        "stages": {
            "representative_preparation": representative_preparation,
            "alignment_preparation": alignment_preparation,
            "scoring": scoring,
            "peaks": peaks,
            "targets": targets,
            "annotations": annotations,
            "annotation_summary": summary,
            "decisions": decisions,
        },
    }
    (output_directory / "screen.manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
