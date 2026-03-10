"""Unit 3 -- Cherry-Picked Term Selection."""

from pathlib import Path
from dataclasses import dataclass
from collections import deque

from gsea_tool.data_ingestion import CohortData
from gsea_tool.configuration import CherryPickCategory


@dataclass
class CategoryGroup:
    """A named group of GO terms for dot plot rendering."""
    category_name: str
    term_names: list[str]


class MappingFileError(Exception):
    """Raised when the category mapping file cannot be parsed."""
    ...


def _compute_mean_abs_nes(term_name: str, cohort: CohortData) -> float:
    """Compute mean absolute NES for a term across all mutants.

    Uses NES=0 for mutants where the term is absent.
    """
    total = 0.0
    for mutant_id in cohort.mutant_ids:
        profile = cohort.profiles[mutant_id]
        rec = profile.records.get(term_name)
        if rec is not None:
            total += abs(rec.nes)
    return total / len(cohort.mutant_ids)


def parse_category_mapping(mapping_path: Path) -> dict[str, str]:
    """Parse the user-supplied category mapping file (TSV fallback path).

    Returns dict mapping term_name (uppercase) -> category_name.
    Raises MappingFileError if the file cannot be parsed.
    """
    if not mapping_path.is_file():
        raise MappingFileError(f"Mapping file not found: {mapping_path}")

    result: dict[str, str] = {}
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                # Skip empty lines and comments
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split("\t")
                if len(parts) != 2:
                    raise MappingFileError(
                        f"Line {line_num}: expected two tab-separated columns, "
                        f"got {len(parts)}"
                    )
                term_name = parts[0].strip().upper()
                category_name = parts[1].strip()
                if not term_name or not category_name:
                    raise MappingFileError(
                        f"Line {line_num}: empty term name or category name"
                    )
                result[term_name] = category_name
    except MappingFileError:
        raise
    except Exception as e:
        raise MappingFileError(f"Failed to parse mapping file: {e}") from e

    return result


def select_cherry_picked_terms(
    cohort: CohortData,
    term_to_category: dict[str, str],
) -> list[CategoryGroup]:
    """Select and group GO terms for Figure 1 based on user-supplied category mapping (TSV fallback path).

    Terms are included if they appear in both the mapping and the GSEA results.
    Within each category, terms are sorted by mean absolute NES descending.
    Categories are returned in the order they first appear in the mapping file.
    Categories with zero matching terms are silently omitted.
    """
    # Build category -> list of matching term names, preserving insertion order
    category_terms: dict[str, list[str]] = {}

    # The mapping keys are uppercase; cohort.all_term_names are also uppercase (from Unit 1)
    # Match case-insensitively by uppercasing both sides
    cohort_terms_upper: dict[str, str] = {}
    for term_name in cohort.all_term_names:
        cohort_terms_upper[term_name.upper()] = term_name

    for mapped_term_upper, category_name in term_to_category.items():
        # Check if this term exists in the cohort (case-insensitive)
        actual_term = cohort_terms_upper.get(mapped_term_upper)
        if actual_term is None:
            continue

        if category_name not in category_terms:
            category_terms[category_name] = []
        category_terms[category_name].append(actual_term)

    # Build result in insertion order, sorting terms within each category
    groups: list[CategoryGroup] = []
    for cat_name, terms in category_terms.items():
        terms.sort(key=lambda t: _compute_mean_abs_nes(t, cohort), reverse=True)
        if terms:
            groups.append(CategoryGroup(category_name=cat_name, term_names=terms))

    return groups


def get_all_descendants(parent_go_id: str, obo_path: Path) -> set[str]:
    """Resolve all descendant GO IDs of a parent GO term using the OBO ontology.

    Parses the OBO file, builds a children map (inverting the is_a parent relationships),
    and performs a breadth-first traversal from parent_go_id to collect all descendants.
    The parent GO ID itself is included in the result set.

    Returns set of GO IDs (including the parent).
    """
    if not obo_path.is_file():
        raise FileNotFoundError(f"OBO file not found: {obo_path}")

    # Parse OBO file to extract is_a relationships
    # Build children map: parent -> set of children
    children_map: dict[str, set[str]] = {}
    all_term_ids: set[str] = set()

    current_id = None
    in_term = False

    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[Term]":
                in_term = True
                current_id = None
            elif stripped.startswith("[") and stripped.endswith("]"):
                in_term = False
                current_id = None
            elif in_term:
                if stripped.startswith("id: "):
                    current_id = stripped[4:].strip()
                    all_term_ids.add(current_id)
                elif stripped.startswith("is_a: ") and current_id is not None:
                    # Format: "is_a: GO:XXXXXXX ! description"
                    parts = stripped[6:].strip().split("!", 1)
                    parent_id = parts[0].strip()
                    if parent_id not in children_map:
                        children_map[parent_id] = set()
                    children_map[parent_id].add(current_id)

    # Validate parent_go_id exists in the ontology
    if parent_go_id not in all_term_ids:
        raise ValueError(
            f"Parent GO ID '{parent_go_id}' not found in the OBO ontology"
        )

    # BFS from parent_go_id
    result: set[str] = set()
    queue: deque[str] = deque([parent_go_id])
    while queue:
        node = queue.popleft()
        if node in result:
            continue
        result.add(node)
        for child in children_map.get(node, set()):
            if child not in result:
                queue.append(child)

    return result


def resolve_categories_from_ontology(
    cohort: CohortData,
    categories: list[CherryPickCategory],
    obo_path: Path,
) -> list[CategoryGroup]:
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
    # Build a mapping from GO ID -> term name(s) from the cohort data
    go_id_to_term_name: dict[str, str] = {}
    for profile in cohort.profiles.values():
        for rec in profile.records.values():
            if rec.go_id not in go_id_to_term_name:
                go_id_to_term_name[rec.go_id] = rec.term_name

    groups: list[CategoryGroup] = []
    for category in categories:
        descendants = get_all_descendants(category.go_id, obo_path)
        # Intersect with cohort GO IDs
        matching_go_ids = descendants & cohort.all_go_ids
        # Map GO IDs to term names
        term_names: list[str] = []
        seen_terms: set[str] = set()
        for go_id in matching_go_ids:
            term_name = go_id_to_term_name.get(go_id)
            if term_name is not None and term_name not in seen_terms:
                term_names.append(term_name)
                seen_terms.add(term_name)

        if not term_names:
            continue

        # Sort by mean absolute NES descending
        term_names.sort(key=lambda t: _compute_mean_abs_nes(t, cohort), reverse=True)
        groups.append(CategoryGroup(category_name=category.label, term_names=term_names))

    return groups
