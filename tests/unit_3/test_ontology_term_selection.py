"""
Test suite for Unit 3 -- Ontology-Based Cherry-Picked Term Selection.

Tests cover the ontology resolution functions: get_all_descendants and
resolve_categories_from_ontology. These functions use the GO OBO hierarchy
to resolve descendant GO terms for configured parent categories, then
intersect with GSEA results to produce CategoryGroup objects.

Synthetic Data Assumptions (module level):
  - DATA ASSUMPTION: OBO files use standard Gene Ontology OBO format with
    [Term] stanzas containing id:, name:, and is_a: fields.
  - DATA ASSUMPTION: GO IDs follow GO:NNNNNNN format. Placeholder IDs like
    GO:0000001 through GO:0000020 are used for test hierarchies.
  - DATA ASSUMPTION: Test ontology hierarchies are small (5-15 terms) with
    clear parent-child is_a relationships for verifiable transitive closure.
  - DATA ASSUMPTION: NES values range from -3.0 to +3.0, typical for GSEA.
  - DATA ASSUMPTION: CherryPickCategory uses go_id and label fields per Unit 2.
  - DATA ASSUMPTION: CohortData.all_go_ids is populated with GO IDs present
    in the enrichment data, enabling intersection with ontology descendants.
  - DATA ASSUMPTION: Each TermRecord has a meaningful go_id for ontology tests.
  - DATA ASSUMPTION: nom_pval=0.05, fdr=0.01, size=100 are placeholders;
    this unit only uses term_name, go_id, and nes from TermRecord.
"""

from pathlib import Path

import pytest

from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.configuration import CherryPickCategory
from gsea_tool.cherry_picked import CategoryGroup

# These functions are specified in the blueprint but may not yet be in the stub.
# Import them and skip the entire module if they are not available yet.
try:
    from gsea_tool.cherry_picked import get_all_descendants, resolve_categories_from_ontology
except ImportError:
    pytest.skip(
        "get_all_descendants and resolve_categories_from_ontology not yet in stub",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers for building synthetic test data
# ---------------------------------------------------------------------------

def _make_term_record(
    term_name: str,
    go_id: str,
    nes: float,
    fdr: float = 0.01,
    nom_pval: float = 0.05,
    size: int = 100,
) -> TermRecord:
    """Build a TermRecord with sensible defaults."""
    return TermRecord(
        term_name=term_name,
        go_id=go_id,
        nes=nes,
        fdr=fdr,
        nom_pval=nom_pval,
        size=size,
    )


def _make_cohort_with_go_ids(
    mutant_term_map: dict[str, list[tuple[str, str, float]]],
) -> CohortData:
    """Create CohortData from {mutant_id: [(term_name, go_id, nes), ...], ...}.

    Unlike the TSV-path helper, this one requires explicit go_id per term
    to support ontology-based intersection.
    """
    profiles: dict[str, MutantProfile] = {}
    all_term_names: set[str] = set()
    all_go_ids: set[str] = set()
    for mid, term_list in mutant_term_map.items():
        records: dict[str, TermRecord] = {}
        for tname, gid, nes in term_list:
            rec = _make_term_record(tname, gid, nes)
            records[tname] = rec
            all_term_names.add(tname)
            all_go_ids.add(gid)
        profiles[mid] = MutantProfile(mutant_id=mid, records=records)
    mutant_ids = sorted(profiles.keys())
    return CohortData(
        mutant_ids=mutant_ids,
        profiles=profiles,
        all_term_names=all_term_names,
        all_go_ids=all_go_ids,
    )


def _write_obo_file(tmp_path: Path, stanzas: list[dict], filename: str = "go.obo") -> Path:
    """Write a minimal OBO file from a list of term stanza dicts.

    Each stanza dict should have keys: id, name, and optionally is_a (list of parent IDs).
    Returns the path to the written OBO file.
    """
    lines = [
        "format-version: 1.2",
        "ontology: go",
        "",
    ]
    for stanza in stanzas:
        lines.append("[Term]")
        lines.append(f"id: {stanza['id']}")
        lines.append(f"name: {stanza['name']}")
        for parent_id in stanza.get("is_a", []):
            lines.append(f"is_a: {parent_id} ! parent term")
        lines.append("")

    obo_path = tmp_path / filename
    obo_path.write_text("\n".join(lines) + "\n")
    return obo_path


# ---------------------------------------------------------------------------
# Standard test OBO hierarchy:
#
#   GO:0000001 (root_process)
#   ├── GO:0000002 (child_a)
#   │   ├── GO:0000004 (grandchild_a1)
#   │   └── GO:0000005 (grandchild_a2)
#   └── GO:0000003 (child_b)
#       └── GO:0000006 (grandchild_b1)
#
#   GO:0000010 (other_root) -- separate hierarchy
#   └── GO:0000011 (other_child)
#       └── GO:0000005 (grandchild_a2) -- ALSO child of GO:0000002 (multi-parent)
# ---------------------------------------------------------------------------

STANDARD_OBO_STANZAS = [
    {"id": "GO:0000001", "name": "root_process"},
    {"id": "GO:0000002", "name": "child_a", "is_a": ["GO:0000001"]},
    {"id": "GO:0000003", "name": "child_b", "is_a": ["GO:0000001"]},
    {"id": "GO:0000004", "name": "grandchild_a1", "is_a": ["GO:0000002"]},
    {"id": "GO:0000005", "name": "grandchild_a2", "is_a": ["GO:0000002", "GO:0000011"]},
    {"id": "GO:0000006", "name": "grandchild_b1", "is_a": ["GO:0000003"]},
    {"id": "GO:0000010", "name": "other_root"},
    {"id": "GO:0000011", "name": "other_child", "is_a": ["GO:0000010"]},
]


# ---------------------------------------------------------------------------
# Tests for get_all_descendants
# ---------------------------------------------------------------------------

class TestGetAllDescendantsBasic:
    """Test basic descendant resolution from OBO hierarchy."""

    def test_root_returns_all_descendants(self, tmp_path):
        """Contract 2: Transitive closure of is_a relationships from root.

        GO:0000001 should yield itself plus all descendants:
        GO:0000002, GO:0000003, GO:0000004, GO:0000005, GO:0000006.
        """
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        result = get_all_descendants("GO:0000001", obo_path)

        expected = {"GO:0000001", "GO:0000002", "GO:0000003",
                    "GO:0000004", "GO:0000005", "GO:0000006"}
        assert result == expected

    def test_parent_included_in_result(self, tmp_path):
        """Contract 2: The parent GO ID itself is included in the result set."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        result = get_all_descendants("GO:0000002", obo_path)
        assert "GO:0000002" in result

    def test_intermediate_node_returns_subtree(self, tmp_path):
        """Contract 2: Intermediate node returns its subtree only."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        result = get_all_descendants("GO:0000002", obo_path)

        expected = {"GO:0000002", "GO:0000004", "GO:0000005"}
        assert result == expected

    def test_leaf_node_returns_only_itself(self, tmp_path):
        """Contract 2: Leaf node has no children, returns singleton set."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        result = get_all_descendants("GO:0000004", obo_path)

        assert result == {"GO:0000004"}

    def test_separate_hierarchy_root(self, tmp_path):
        """Contract 2: Separate root resolves its own subtree only."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        result = get_all_descendants("GO:0000010", obo_path)

        # GO:0000005 is also a child of GO:0000011, so it's a descendant of GO:0000010
        expected = {"GO:0000010", "GO:0000011", "GO:0000005"}
        assert result == expected

    def test_returns_set_type(self, tmp_path):
        """Signature: get_all_descendants returns a set of GO ID strings."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        result = get_all_descendants("GO:0000004", obo_path)
        assert isinstance(result, set)
        for item in result:
            assert isinstance(item, str)


class TestGetAllDescendantsMultiParent:
    """Test handling of terms with multiple parents (multi-parent GO terms)."""

    def test_multi_parent_term_reachable_from_both_parents(self, tmp_path):
        """Contract 6: GO:0000005 has parents GO:0000002 and GO:0000011.
        It should be reachable as descendant from both roots.
        """
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)

        from_root1 = get_all_descendants("GO:0000001", obo_path)
        from_root2 = get_all_descendants("GO:0000010", obo_path)

        assert "GO:0000005" in from_root1
        assert "GO:0000005" in from_root2


class TestGetAllDescendantsErrors:
    """Test error conditions for get_all_descendants."""

    def test_missing_obo_file_raises_file_not_found(self, tmp_path):
        """Error condition: FileNotFoundError when OBO file does not exist."""
        nonexistent = tmp_path / "nonexistent.obo"
        with pytest.raises(FileNotFoundError):
            get_all_descendants("GO:0000001", nonexistent)

    def test_invalid_parent_go_id_raises_value_error(self, tmp_path):
        """Error condition: ValueError when parent GO ID not found in ontology."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        with pytest.raises(ValueError):
            get_all_descendants("GO:9999999", obo_path)


class TestGetAllDescendantsEdgeCases:
    """Test edge cases for descendant resolution."""

    def test_single_term_ontology(self, tmp_path):
        """Edge case: OBO with only one term, no children."""
        stanzas = [{"id": "GO:0000001", "name": "lonely_term"}]
        obo_path = _write_obo_file(tmp_path, stanzas)
        result = get_all_descendants("GO:0000001", obo_path)
        assert result == {"GO:0000001"}

    def test_deep_chain(self, tmp_path):
        """Edge case: Linear chain of is_a relationships (depth=4)."""
        stanzas = [
            {"id": "GO:0000001", "name": "level0"},
            {"id": "GO:0000002", "name": "level1", "is_a": ["GO:0000001"]},
            {"id": "GO:0000003", "name": "level2", "is_a": ["GO:0000002"]},
            {"id": "GO:0000004", "name": "level3", "is_a": ["GO:0000003"]},
            {"id": "GO:0000005", "name": "level4", "is_a": ["GO:0000004"]},
        ]
        obo_path = _write_obo_file(tmp_path, stanzas)
        result = get_all_descendants("GO:0000001", obo_path)
        expected = {"GO:0000001", "GO:0000002", "GO:0000003",
                    "GO:0000004", "GO:0000005"}
        assert result == expected

    def test_diamond_hierarchy(self, tmp_path):
        """Edge case: Diamond pattern where child has two parents that share a grandparent.

           GO:0000001
           /        \\
        GO:0000002  GO:0000003
           \\        /
           GO:0000004
        """
        stanzas = [
            {"id": "GO:0000001", "name": "top"},
            {"id": "GO:0000002", "name": "left", "is_a": ["GO:0000001"]},
            {"id": "GO:0000003", "name": "right", "is_a": ["GO:0000001"]},
            {"id": "GO:0000004", "name": "bottom", "is_a": ["GO:0000002", "GO:0000003"]},
        ]
        obo_path = _write_obo_file(tmp_path, stanzas)
        result = get_all_descendants("GO:0000001", obo_path)
        expected = {"GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"}
        assert result == expected


# ---------------------------------------------------------------------------
# Tests for resolve_categories_from_ontology
# ---------------------------------------------------------------------------

class TestResolveCategoriesBasic:
    """Test basic ontology-based category resolution."""

    def test_single_category_with_matching_terms(self, tmp_path):
        """Contract 2: Resolve descendants, intersect with cohort, return CategoryGroup.

        Category 'Root Process' maps to GO:0000001, whose descendants include
        GO:0000002 and GO:0000004. Cohort has terms with those GO IDs.
        """
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("CHILD A PROCESS", "GO:0000002", 2.0),
                ("GRANDCHILD A1 PROCESS", "GO:0000004", 1.5),
            ],
            "mutB": [
                ("CHILD A PROCESS", "GO:0000002", -1.0),
                ("GRANDCHILD A1 PROCESS", "GO:0000004", 0.5),
            ],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root Process")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert len(result) == 1
        assert result[0].category_name == "Root Process"
        assert isinstance(result[0], CategoryGroup)
        assert len(result[0].term_names) == 2

    def test_returns_list_of_category_groups(self, tmp_path):
        """Signature: resolve_categories_from_ontology returns list[CategoryGroup]."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [("CHILD A PROCESS", "GO:0000002", 1.0)],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Test")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert isinstance(result, list)
        for group in result:
            assert isinstance(group, CategoryGroup)

    def test_multiple_categories(self, tmp_path):
        """Contract 5: Multiple categories returned in config list order."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("CHILD A PROCESS", "GO:0000002", 2.0),
                ("OTHER CHILD PROCESS", "GO:0000011", 1.0),
            ],
        })
        categories = [
            CherryPickCategory(go_id="GO:0000010", label="Other"),
            CherryPickCategory(go_id="GO:0000001", label="Root"),
        ]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert len(result) == 2
        assert result[0].category_name == "Other"
        assert result[1].category_name == "Root"


class TestResolveCategoriesSorting:
    """Test that terms within each category are sorted by mean abs NES descending."""

    def test_terms_sorted_by_mean_abs_nes_descending(self, tmp_path):
        """Contract 4: Within each category, terms sorted by mean |NES| descending."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        # GO:0000002, GO:0000004, GO:0000005 are all descendants of GO:0000001
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("LOW NES TERM", "GO:0000002", 0.5),
                ("HIGH NES TERM", "GO:0000004", 3.0),
                ("MID NES TERM", "GO:0000005", 1.5),
            ],
            "mutB": [
                ("LOW NES TERM", "GO:0000002", 0.5),
                ("HIGH NES TERM", "GO:0000004", 3.0),
                ("MID NES TERM", "GO:0000005", 1.5),
            ],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert len(result) == 1
        assert result[0].term_names[0] == "HIGH NES TERM"
        assert result[0].term_names[1] == "MID NES TERM"
        assert result[0].term_names[2] == "LOW NES TERM"

    def test_negative_nes_uses_absolute_value_for_sorting(self, tmp_path):
        """Contract 4: Sorting uses absolute NES, so -3.0 ranks higher than 1.0."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("NEGATIVE TERM", "GO:0000002", -3.0),
                ("POSITIVE TERM", "GO:0000004", 1.0),
            ],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert result[0].term_names[0] == "NEGATIVE TERM"
        assert result[0].term_names[1] == "POSITIVE TERM"

    def test_mean_nes_uses_zero_for_absent_mutants(self, tmp_path):
        """Contract 4: NES=0 used for mutants where term is absent.

        Term A: mutA=6.0, mutB=absent -> mean |NES| = (6.0 + 0) / 2 = 3.0
        Term B: mutA=2.0, mutB=2.0    -> mean |NES| = (2.0 + 2.0) / 2 = 2.0
        So Term A should rank first.
        """
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("TERM A", "GO:0000002", 6.0),
                ("TERM B", "GO:0000004", 2.0),
            ],
            "mutB": [
                ("TERM B", "GO:0000004", 2.0),
            ],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert result[0].term_names[0] == "TERM A"
        assert result[0].term_names[1] == "TERM B"


class TestResolveCategoriesFiltering:
    """Test filtering behavior: intersection with cohort data."""

    def test_descendants_not_in_cohort_silently_dropped(self, tmp_path):
        """Contract 3: GO terms in ontology but absent from all mutant profiles are dropped."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        # Only GO:0000002 is in cohort; GO:0000004, GO:0000005 are descendants but not in data
        cohort = _make_cohort_with_go_ids({
            "mutA": [("CHILD A PROCESS", "GO:0000002", 1.0)],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert len(result) == 1
        assert result[0].term_names == ["CHILD A PROCESS"]

    def test_empty_category_silently_omitted(self, tmp_path):
        """Contract 5: Categories with zero matching terms are silently omitted."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        # Cohort only has terms under GO:0000010, not GO:0000001
        cohort = _make_cohort_with_go_ids({
            "mutA": [("OTHER CHILD PROCESS", "GO:0000011", 1.0)],
        })
        categories = [
            CherryPickCategory(go_id="GO:0000001", label="Root"),
            CherryPickCategory(go_id="GO:0000010", label="Other"),
        ]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert len(result) == 1
        assert result[0].category_name == "Other"

    def test_all_categories_empty_returns_empty_list(self, tmp_path):
        """Contract 5: If no categories have matching terms, return empty list."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        # Cohort has a GO ID that's not a descendant of any configured parent
        cohort = _make_cohort_with_go_ids({
            "mutA": [("UNRELATED TERM", "GO:0099999", 1.0)],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert result == []


class TestResolveCategoriesMultiCategory:
    """Test GO terms appearing in multiple categories."""

    def test_term_in_multiple_categories(self, tmp_path):
        """Contract 6: A GO term descending from multiple configured parents
        appears in all matching CategoryGroup objects.

        GO:0000005 is a descendant of both GO:0000001 (via GO:0000002)
        and GO:0000010 (via GO:0000011).
        """
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [("SHARED TERM", "GO:0000005", 2.0)],
        })
        categories = [
            CherryPickCategory(go_id="GO:0000001", label="Root Process"),
            CherryPickCategory(go_id="GO:0000010", label="Other Root"),
        ]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert len(result) == 2
        category_names = [g.category_name for g in result]
        assert "Root Process" in category_names
        assert "Other Root" in category_names

        for group in result:
            assert "SHARED TERM" in group.term_names


class TestResolveCategoriesOrder:
    """Test that categories are returned in config list order."""

    def test_categories_returned_in_config_order(self, tmp_path):
        """Contract 5: Categories returned in the order specified in config list."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("CHILD B PROCESS", "GO:0000003", 1.0),
                ("OTHER CHILD PROCESS", "GO:0000011", 2.0),
                ("CHILD A PROCESS", "GO:0000002", 1.5),
            ],
        })
        # Config order: Other first, then Root
        categories = [
            CherryPickCategory(go_id="GO:0000010", label="Other"),
            CherryPickCategory(go_id="GO:0000001", label="Root"),
        ]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert result[0].category_name == "Other"
        assert result[1].category_name == "Root"

    def test_config_order_reversed(self, tmp_path):
        """Contract 5: Reversing config order reverses output order."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("CHILD B PROCESS", "GO:0000003", 1.0),
                ("OTHER CHILD PROCESS", "GO:0000011", 2.0),
            ],
        })
        categories = [
            CherryPickCategory(go_id="GO:0000001", label="Root"),
            CherryPickCategory(go_id="GO:0000010", label="Other"),
        ]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert result[0].category_name == "Root"
        assert result[1].category_name == "Other"


class TestResolveCategoriesErrors:
    """Test error conditions for resolve_categories_from_ontology."""

    def test_missing_obo_file_raises_file_not_found(self, tmp_path):
        """Error condition: FileNotFoundError when OBO file does not exist."""
        nonexistent = tmp_path / "nonexistent.obo"
        cohort = _make_cohort_with_go_ids({
            "mutA": [("TERM", "GO:0000001", 1.0)],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        with pytest.raises(FileNotFoundError):
            resolve_categories_from_ontology(cohort, categories, nonexistent)

    def test_invalid_parent_go_id_raises_value_error(self, tmp_path):
        """Error condition: ValueError when configured parent GO ID not in ontology."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [("TERM", "GO:0000001", 1.0)],
        })
        categories = [CherryPickCategory(go_id="GO:9999999", label="Nonexistent")]

        with pytest.raises(ValueError):
            resolve_categories_from_ontology(cohort, categories, obo_path)


class TestResolveCategoriesGoIdMapping:
    """Test that GO IDs are correctly mapped back to term names via cohort data."""

    def test_go_id_mapped_to_term_name_via_cohort(self, tmp_path):
        """Contract 2: GO IDs mapped back to term names via cohort's TermRecord.go_id."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("MITOCHONDRIAL TRANSPORT", "GO:0000002", 2.0),
                ("RIBOSOME BIOGENESIS", "GO:0000004", 1.0),
            ],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert len(result) == 1
        assert "MITOCHONDRIAL TRANSPORT" in result[0].term_names
        assert "RIBOSOME BIOGENESIS" in result[0].term_names


class TestResolveCategoriesInvariants:
    """Test post-condition invariants."""

    def test_no_empty_groups_in_output(self, tmp_path):
        """Invariant: All returned groups have len(term_names) > 0."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [("CHILD A PROCESS", "GO:0000002", 1.0)],
        })
        categories = [
            CherryPickCategory(go_id="GO:0000001", label="Root"),
            CherryPickCategory(go_id="GO:0000010", label="Other"),  # No matching terms
        ]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        for group in result:
            assert len(group.term_names) > 0, (
                f"Category '{group.category_name}' has empty term_names"
            )

    def test_output_type_is_category_group(self, tmp_path):
        """Contract 7: Output uses CategoryGroup type for renderer compatibility."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [("CHILD A PROCESS", "GO:0000002", 1.0)],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        for group in result:
            assert isinstance(group, CategoryGroup)
            assert isinstance(group.category_name, str)
            assert isinstance(group.term_names, list)
            for name in group.term_names:
                assert isinstance(name, str)


class TestResolveCategoriesComprehensive:
    """End-to-end scenario tests for ontology-based resolution."""

    def test_full_scenario_multiple_categories_and_mutants(self, tmp_path):
        """Comprehensive test: multiple categories, multiple mutants, sorting, filtering.

        Category 'A Lineage': parent GO:0000002 -> descendants GO:0000002, GO:0000004, GO:0000005
        Category 'B Lineage': parent GO:0000003 -> descendants GO:0000003, GO:0000006

        Cohort has terms for GO:0000002, GO:0000004, GO:0000006.
        GO:0000005 and GO:0000003 are in ontology but not in cohort -> silently dropped.

        mutA: GO:0000002 (NES=2.0), GO:0000004 (NES=1.0), GO:0000006 (NES=3.0)
        mutB: GO:0000002 (NES=1.0), GO:0000004 (NES=3.0), GO:0000006 (NES=1.0)

        'A Lineage' terms:
          - GO:0000002 CHILD_A: mean |NES| = (2.0+1.0)/2 = 1.5
          - GO:0000004 GRANDCHILD_A1: mean |NES| = (1.0+3.0)/2 = 2.0
          -> sorted: GRANDCHILD_A1 first, CHILD_A second

        'B Lineage' terms:
          - GO:0000006 GRANDCHILD_B1: mean |NES| = (3.0+1.0)/2 = 2.0
        """
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("CHILD_A", "GO:0000002", 2.0),
                ("GRANDCHILD_A1", "GO:0000004", 1.0),
                ("GRANDCHILD_B1", "GO:0000006", 3.0),
            ],
            "mutB": [
                ("CHILD_A", "GO:0000002", 1.0),
                ("GRANDCHILD_A1", "GO:0000004", 3.0),
                ("GRANDCHILD_B1", "GO:0000006", 1.0),
            ],
        })
        categories = [
            CherryPickCategory(go_id="GO:0000002", label="A Lineage"),
            CherryPickCategory(go_id="GO:0000003", label="B Lineage"),
        ]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert len(result) == 2

        # Config order preserved
        assert result[0].category_name == "A Lineage"
        assert result[1].category_name == "B Lineage"

        # A Lineage: sorted by mean |NES| descending
        assert result[0].term_names == ["GRANDCHILD_A1", "CHILD_A"]

        # B Lineage: single term
        assert result[1].term_names == ["GRANDCHILD_B1"]

    def test_empty_cohort_returns_empty(self, tmp_path):
        """Edge case: Cohort with no terms returns empty list."""
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = CohortData(
            mutant_ids=[],
            profiles={},
            all_term_names=set(),
            all_go_ids=set(),
        )
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert result == []


# ---------------------------------------------------------------------------
# Coverage gap tests -- added by coverage review
# ---------------------------------------------------------------------------


class TestResolveCategoriesEmptyInput:
    """Test behavior when the categories list is empty."""

    def test_empty_categories_list_returns_empty(self, tmp_path):
        """Edge case: An empty categories list should return an empty list.

        DATA ASSUMPTION: A valid cohort with terms but no configured categories
        means no resolution is performed, yielding an empty result.
        """
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [("CHILD A PROCESS", "GO:0000002", 1.0)],
        })
        categories: list[CherryPickCategory] = []

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert result == []


class TestResolveCategoriesMixedSignNES:
    """Test sorting in ontology path when a term has mixed positive/negative NES
    across different mutants.

    Contract 4 states terms are sorted by mean absolute NES. When a term
    has positive NES in one mutant and negative in another, both should
    contribute their absolute values to the mean.
    """

    def test_mixed_positive_negative_nes_sorting(self, tmp_path):
        """Contract 4: A term with NES=+2.0 in mutA and NES=-3.0 in mutB
        should have mean |NES| = (2.0 + 3.0)/2 = 2.5, ranking higher than
        a term with consistent NES=1.5 (mean |NES| = 1.5).

        DATA ASSUMPTION: Two terms under GO:0000001 descendants. MIXED TERM
        has opposite-sign NES across mutants; CONSISTENT TERM has same-sign.
        Mean |NES| for MIXED TERM = (2.0 + 3.0)/2 = 2.5
        Mean |NES| for CONSISTENT TERM = (1.5 + 1.5)/2 = 1.5
        MIXED TERM should rank first.
        """
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("MIXED TERM", "GO:0000002", 2.0),
                ("CONSISTENT TERM", "GO:0000004", 1.5),
            ],
            "mutB": [
                ("MIXED TERM", "GO:0000002", -3.0),
                ("CONSISTENT TERM", "GO:0000004", 1.5),
            ],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert len(result) == 1
        assert result[0].term_names == ["MIXED TERM", "CONSISTENT TERM"]


class TestResolveCategoriesParentTermInData:
    """Test that the parent GO ID itself is included in the intersection
    when it appears in the cohort data.

    Contract 2 states the parent GO ID itself is included in the descendant set.
    If the cohort contains a term with the parent GO ID, it should appear in
    the resulting CategoryGroup.
    """

    def test_parent_go_id_term_included_in_result(self, tmp_path):
        """Contract 2: The parent GO ID itself is included in descendants.
        If the cohort has a term with that GO ID, it should be in the result.

        DATA ASSUMPTION: Cohort has a term with GO:0000001 (the parent itself)
        and a term with GO:0000002 (a child). Both should appear in the result.
        """
        obo_path = _write_obo_file(tmp_path, STANDARD_OBO_STANZAS)
        cohort = _make_cohort_with_go_ids({
            "mutA": [
                ("ROOT PROCESS TERM", "GO:0000001", 3.0),
                ("CHILD A PROCESS", "GO:0000002", 1.0),
            ],
        })
        categories = [CherryPickCategory(go_id="GO:0000001", label="Root")]

        result = resolve_categories_from_ontology(cohort, categories, obo_path)

        assert len(result) == 1
        assert "ROOT PROCESS TERM" in result[0].term_names
        assert "CHILD A PROCESS" in result[0].term_names
        # Parent term has higher |NES|, should be first
        assert result[0].term_names[0] == "ROOT PROCESS TERM"
