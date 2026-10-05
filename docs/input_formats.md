# Input formats and outputs

`virassq screen` takes three existing inputs. It does not run MMseqs2 or DIAMOND.

## Representatives

Accepted formats are FASTA (`.fa`, `.fasta`, `.fna`), a plain `.txt` or `.ids` file with one contig ID per line, or a Parquet table with `representative_contig_id`. FASTA and text files can be gzip-compressed.

FASTA IDs are taken up to the first whitespace; descriptions and sequence bodies are not used. Text IDs must not contain whitespace or a `>` prefix. Blank lines are ignored and duplicate IDs are rejected. IDs must match the query IDs in the MMseqs2 alignments.

For FASTA or text input, the selected IDs are saved in `tables/representatives.parquet`. VirAssQ does not select a new representative or filter the supplied list by annotation.

## MMseqs2 alignments

Use a headerless 14-column `.tsv` (or `.tsv.gz`), exported from an existing MMseqs2 nucleotide search:

```bash
mmseqs convertalis contigs_db contigs_db alignments_db alignments.tsv \
  --format-output 'query,target,pident,alnlen,qstart,qend,qlen,tstart,tend,tlen,qcov,tcov,evalue,bits'
```

Replace the database names with your existing database paths. The default 12-column BLAST-like TSV is not sufficient.

Coordinates are treated as 1-based and inclusive. TSV coordinates are normalized automatically, with original coordinates retained, and self alignments removed. Identity, alignment-length and target-coverage filters must already have been applied during the MMseqs2 search; VirAssQ does not change them. The normalized table is saved as `tables/normalized_alignments.parquet`, and the number of removed self alignments is recorded in `screen.manifest.json`.

Alternatively, provide the normalized Parquet output of `virassq.alignments.normalize_raw_alignments`, without self alignments. This table is reused rather than copied or normalized again.

## Contig annotations

Annotations currently require a prepared Parquet table, not raw DIAMOND output. There must be one row per annotated contig, with the following columns:

```text
qseqid
has_viral_hit, best_hit_is_viral
best_domain, best_family, best_genus, best_species, best_title
best_viral_family, best_viral_genus, best_viral_species, best_viral_title
```

`qseqid` is the contig ID, not an ORF ID. The two flags are boolean: `has_viral_hit` means that at least one DIAMOND hit was viral; `best_hit_is_viral` means that the best overall hit was viral. `best_*` fields describe the best overall hit, and `best_viral_*` describe the best viral hit even when a cellular hit ranked above it. Taxonomic fields and titles are strings; missing values can be null, but the columns must be present.

Contigs without hits may be absent from this table. They remain counted as nucleotide targets, with missing annotations. Protein roles are derived from `best_viral_title` inside VirAssQ.

## Outputs

The output directory must be new or empty. Existing results are not overwritten.

- `representative_actions.parquet` contains all selected representatives and their `no_selected_boundary`, `review` or `quarantine` action. Check `has_nonself_alignments` as well: a representative without alignments could not be examined.
- `decisions/peak_decisions.parquet` contains peak coordinates, measured counts, thresholds and decision reasons. Positional profiles and per-peak target IDs and annotations are retained in `boundary_scores/`, `boundary_score_peaks/` and `tables/`.
- `decisions/quarantine_query_ids.txt` lists representatives marked for quarantine.
- `screen.manifest.json` records input paths, input preparation, parameters and stage counts.

The command does not modify sequences or perform reclustering. Quarantine IDs are a separate output for subsequent catalogue reconstruction.
