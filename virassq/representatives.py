"""Read selected representative IDs without reselecting or clustering contigs."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd


def representative_input_format(path: Path) -> str:
    """Recognise Parquet, FASTA, and one-ID-per-line text by file extension."""

    if path.suffix.lower() == ".parquet":
        return "parquet"
    name = path.name.lower().removesuffix(".gz")
    if name.endswith((".fa", ".fasta", ".fna")):
        return "fasta"
    if name.endswith((".txt", ".ids")):
        return "txt"
    raise ValueError(
        "representatives must be .fa, .fasta, .fna, .txt, .ids "
        "(optionally .gz), or .parquet"
    )


def read_representative_ids(path: Path, input_format: str) -> list[str]:
    """Read FASTA header IDs or bare text IDs; reject duplicate identifiers."""

    if input_format not in ("fasta", "txt"):
        raise ValueError("text representative input format must be fasta or txt")
    identifiers = []
    first_line_by_id = {}
    opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            if input_format == "fasta":
                if not value.startswith(">"):
                    if not identifiers:
                        raise ValueError(
                            f"FASTA sequence before the first header at line {line_number}: {path}"
                        )
                    continue
                fields = value[1:].split()
                if not fields:
                    raise ValueError(
                        f"empty FASTA header at line {line_number}: {path}"
                    )
                # MMseqs2 aligns by the first whitespace-delimited header ID,
                # not the description or sequence body.
                identifier = fields[0]
            else:
                if value.startswith(">") or len(value.split()) != 1:
                    raise ValueError(
                        f"expected one bare contig ID at line {line_number}: {path}"
                    )
                identifier = value
            if identifier in first_line_by_id:
                raise ValueError(
                    f"duplicate representative ID {identifier!r} at line {line_number} "
                    f"(first at line {first_line_by_id[identifier]}): {path}"
                )
            first_line_by_id[identifier] = line_number
            identifiers.append(identifier)
    return identifiers


def prepare_representatives(
    input_path: Path,
    output_path: Path,
) -> tuple[Path, dict[str, object]]:
    """Reuse a Parquet table or write exactly the selected FASTA/TXT IDs."""

    input_format = representative_input_format(input_path)
    if input_format == "parquet":
        return input_path, {
            "input": str(input_path.resolve()),
            "output": str(input_path.resolve()),
            "input_format": input_format,
            "preparation_performed": False,
        }
    if not input_path.is_file():
        raise FileNotFoundError(f"representatives do not exist: {input_path}")
    partial_path = output_path.with_name(f"{output_path.stem}.partial.parquet")
    if output_path.exists() or partial_path.exists():
        raise FileExistsError(
            f"representative preparation outputs already exist: {output_path}"
        )
    identifiers = read_representative_ids(input_path, input_format)
    table = pd.DataFrame(
        {"representative_contig_id": pd.Series(identifiers, dtype="string")}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_parquet(partial_path, index=False)
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return output_path, {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "input_format": input_format,
        "preparation_performed": True,
        "representative_count": len(identifiers),
    }
