# Auto-generated stub — do not edit
from pathlib import Path
from dataclasses import dataclass
from gsea_tool.data_ingestion import CohortData

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
    """Parse the user-supplied category mapping file.

    Returns dict mapping term_name (uppercase) -> category_name.
    Raises MappingFileError if the file cannot be parsed.
    """
    ...

def select_cherry_picked_terms(cohort: CohortData, term_to_category: dict[str, str]) -> list[CategoryGroup]:
    """Select and group GO terms for Figure 1 based on user-supplied category mapping.

    Terms are included if they appear in both the mapping and the GSEA results.
    Within each category, terms are sorted by mean absolute NES descending.
    Categories are returned in the fixed order: Mitochondria, Translation, GPCR, Synapse.
    """
    ...
assert mapping_path.is_file(), 'Mapping file path must point to an existing file'
assert all((group.category_name in ('Mitochondria', 'Translation', 'GPCR', 'Synapse') for group in groups)), 'All category names must be one of the four specified categories'
assert len(groups) <= 4, 'At most four category groups are returned'
assert all((len(group.term_names) > 0 for group in groups)), 'Empty categories are omitted'
