# Auto-generated stub — do not edit
from pathlib import Path
from dataclasses import dataclass
from gsea_tool.data_ingestion import CohortData
from gsea_tool.configuration import CherryPickCategory

@dataclass
class CategoryGroup:
    """A named group of GO terms for dot plot rendering."""
    category_name: str
    term_names: list[str]
    ...

class MappingFileError(Exception):
    """Raised when the category mapping file cannot be parsed."""
    ...

def parse_category_mapping(mapping_path: Path) -> dict[str, str]:
    """Parse the user-supplied category mapping file (TSV fallback path).

    Returns dict mapping term_name (uppercase) -> category_name.
    Raises MappingFileError if the file cannot be parsed.
    """
    ...

def select_cherry_picked_terms(cohort: CohortData, term_to_category: dict[str, str]) -> list[CategoryGroup]:
    """Select and group GO terms for Figure 1 based on user-supplied category mapping (TSV fallback path).

    Terms are included if they appear in both the mapping and the GSEA results.
    Within each category, terms are sorted by mean absolute NES descending.
    Categories are returned in the order they first appear in the mapping dict.
    """
    ...

def get_all_descendants(parent_go_id: str, obo_path: Path) -> set[str]:
    """Resolve all descendant GO IDs of a parent GO term using the OBO ontology.

    Parses the OBO file, builds a children map (inverting the is_a parent relationships),
    and performs a breadth-first traversal from parent_go_id to collect all descendants.
    The parent GO ID itself is included in the result set.

    Returns set of GO IDs (including the parent).
    """
    ...

def resolve_categories_from_ontology(cohort: CohortData, categories: list[CherryPickCategory], obo_path: Path) -> list[CategoryGroup]:
    """Select and group GO terms for Figure 1 using ontology-based category resolution.

    For each configured category:
    1. Resolve all descendant GO IDs of the parent GO ID via the OBO hierarchy.
    2. Intersect descendants with GO IDs present in the GSEA results (cohort.all_go_ids).
    3. Map matching GO IDs back to term names via the cohort data.
    4. Sort terms within each category by mean absolute NES across all mutants, descending.

    A GO term matching multiple categories appears in all of them.
    Categories with zero matching terms are silently omitted.
    Categories are returned in the order specified in the config list.
    """
    ...
assert mapping_path.is_file(), 'Mapping file path must point to an existing file'
assert obo_path.is_file(), 'OBO file path must point to an existing file'
assert all((re.match('GO:\\d{7}$', c.go_id) for c in categories)), 'Each category go_id must be valid'
assert all((len(group.term_names) > 0 for group in groups)), 'Empty categories are omitted'
