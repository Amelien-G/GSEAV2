# Auto-generated stub — do not edit
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class TermRecord:
    """Enrichment data for one GO term in one mutant."""
    term_name: str
    go_id: str
    nes: float
    fdr: float
    nom_pval: float
    size: int
    ...

@dataclass
class MutantProfile:
    """Complete enrichment profile for one mutant (merged pos + neg)."""
    mutant_id: str
    records: dict[str, TermRecord]
    ...

@dataclass
class CohortData:
    """All enrichment data for the entire mutant cohort."""
    mutant_ids: list[str]
    profiles: dict[str, MutantProfile]
    all_term_names: set[str]
    all_go_ids: set[str]
    ...

class DataIngestionError(Exception):
    """Raised when input data violates structural expectations."""
    ...

def discover_mutant_folders(data_dir: Path) -> list[tuple[str, Path]]:
    """Discover level-1 mutant subfolders and extract mutant identifiers.

    Returns list of (mutant_id, folder_path) sorted alphabetically by mutant_id.
    """
    ...

def locate_report_files(mutant_folder: Path, mutant_id: str) -> tuple[Path, Path]:
    """Locate exactly one pos and one neg TSV file in a mutant subfolder.

    Returns (pos_file_path, neg_file_path).
    Raises DataIngestionError if zero or more than one match for either pattern.
    """
    ...

def parse_gsea_report(tsv_path: Path) -> list[TermRecord]:
    """Parse a single GSEA preranked TSV report file.

    Extracts GO ID and term name from NAME column. Handles HTML artifact in
    column headers and trailing tabs. Skips rows without valid GO ID with warning.
    """
    ...

def merge_pos_neg(pos_records: list[TermRecord], neg_records: list[TermRecord]) -> dict[str, TermRecord]:
    """Merge positive and negative report records into a single profile dict keyed by term_name.

    If a term appears in both pos and neg records, the entry with the smaller
    nominal p-value is retained (conflict resolution per spec Section 6.2 Step 1).
    """
    ...

def ingest_data(data_dir: Path) -> CohortData:
    """Top-level ingestion entry point. Discovers folders, validates, parses, and merges.

    Raises DataIngestionError on structural violations including fewer than 2 mutant lines.
    """
    ...
assert data_dir.is_dir(), 'data_dir must be an existing directory'
assert len(cohort.mutant_ids) >= 2, 'At least 2 mutant lines are required'
assert len(cohort.mutant_ids) == len(cohort.profiles), 'Every mutant_id must have a corresponding profile'
assert cohort.mutant_ids == sorted(cohort.mutant_ids), 'mutant_ids must be in alphabetical order'
assert all((rec.term_name == rec.term_name.upper() and (not rec.term_name.startswith('GO:')) for profile in cohort.profiles.values() for rec in profile.records.values())), 'All term names must be uppercase with GO ID prefix stripped'
assert all((rec.go_id.startswith('GO:') and len(rec.go_id) == 10 for profile in cohort.profiles.values() for rec in profile.records.values())), 'All GO IDs must match GO:NNNNNNN format'
