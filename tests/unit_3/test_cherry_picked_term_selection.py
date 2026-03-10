"""
Test suite for Unit 3 -- Cherry-Picked Term Selection.

Tests cover the category mapping file parser and the term selection/grouping
logic for Figure 1 (hypothesis-driven figure). The unit reads a two-column TSV
mapping file that assigns GO term names to one of four biological categories
(Mitochondria, Translation, GPCR, Synapse), filters cohort enrichment data
to matching terms, sorts within each category by mean absolute NES descending,
and returns ordered CategoryGroup objects in a fixed category order.

Synthetic Data Assumptions (module level):
  - DATA ASSUMPTION: GO term names are short uppercase English strings
    (e.g., "OXIDATIVE PHOSPHORYLATION"), representative of typical GO
    biological process term naming conventions.
  - DATA ASSUMPTION: NES values range roughly from -3.0 to +3.0, which is
    typical for GSEA normalized enrichment scores.
  - DATA ASSUMPTION: FDR values are set to 0.01 by default (not used by this
    unit, but required by TermRecord).
  - DATA ASSUMPTION: Mutant IDs are short alphanumeric strings like "mutA",
    "mutB", representing simplified mutant identifiers.
  - DATA ASSUMPTION: Category mapping files use tab-separated values with
    term names in the first column and category names in the second column,
    per the blueprint specification.
  - DATA ASSUMPTION: go_id values follow the GO:NNNNNNN format (10 chars),
    using placeholder IDs like "GO:0000001" since this unit does not use go_id.
  - DATA ASSUMPTION: nom_pval is set to 0.05 as a placeholder since this unit
    does not use nom_pval.
  - DATA ASSUMPTION: size is set to 100 as a placeholder since this unit
    does not use size.
"""

import inspect
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.cherry_picked import (
    CategoryGroup,
    MappingFileError,
    parse_category_mapping,
    select_cherry_picked_terms,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic test data
# ---------------------------------------------------------------------------

def _make_term_record(
    term_name: str, nes: float, fdr: float = 0.01,
    go_id: str = "GO:0000001", nom_pval: float = 0.05, size: int = 100,
) -> TermRecord:
    """Build a TermRecord with sensible defaults.

    DATA ASSUMPTION: FDR defaults to 0.01, go_id to placeholder "GO:0000001",
    nom_pval to 0.05, and size to 100. These are plausible but arbitrary values;
    Unit 3 only uses term_name and nes from TermRecord.
    """
    return TermRecord(
        term_name=term_name,
        go_id=go_id,
        nes=nes,
        fdr=fdr,
        nom_pval=nom_pval,
        size=size,
    )


def _make_cohort(
    mutant_term_map: dict[str, dict[str, float]],
) -> CohortData:
    """Create a CohortData from {mutant_id: {TERM_NAME: nes, ...}, ...}.

    DATA ASSUMPTION: All terms get default FDR=0.01, go_id="GO:0000001",
    nom_pval=0.05, size=100. Unit 3 only uses term_name and nes.
    """
    profiles: dict[str, MutantProfile] = {}
    all_term_names: set[str] = set()
    all_go_ids: set[str] = set()
    for mid, terms in mutant_term_map.items():
        records: dict[str, TermRecord] = {}
        for tname, nes in terms.items():
            rec = _make_term_record(tname, nes)
            records[tname] = rec
            all_go_ids.add(rec.go_id)
        profiles[mid] = MutantProfile(mutant_id=mid, records=records)
        all_term_names.update(records.keys())
    mutant_ids = sorted(profiles.keys())
    return CohortData(
        mutant_ids=mutant_ids,
        profiles=profiles,
        all_term_names=all_term_names,
        all_go_ids=all_go_ids,
    )


def _write_mapping_file(tmp_path: Path, lines: list[str]) -> Path:
    """Write a category mapping TSV file and return its path.

    DATA ASSUMPTION: Mapping files use tab as separator per the blueprint
    two-column TSV specification.
    """
    mapping_file = tmp_path / "category_mapping.tsv"
    mapping_file.write_text("\n".join(lines) + "\n")
    return mapping_file


# ---------------------------------------------------------------------------
# Signature and structure tests
# ---------------------------------------------------------------------------

class TestCategoryGroupStructure:
    """Verify the CategoryGroup dataclass structure matches the blueprint."""

    def test_is_dataclass(self):
        """CategoryGroup should be a dataclass."""
        assert hasattr(CategoryGroup, "__dataclass_fields__"), (
            "CategoryGroup should be a dataclass"
        )

    def test_has_category_name_field(self):
        """CategoryGroup should have a 'category_name' field of type str."""
        field_names = {f.name for f in dataclass_fields(CategoryGroup)}
        assert "category_name" in field_names

    def test_has_term_names_field(self):
        """CategoryGroup should have a 'term_names' field of type list[str]."""
        field_names = {f.name for f in dataclass_fields(CategoryGroup)}
        assert "term_names" in field_names

    def test_instantiation(self):
        """CategoryGroup can be instantiated with category_name and term_names."""
        # DATA ASSUMPTION: Simple strings for testing instantiation.
        group = CategoryGroup(category_name="Mitochondria", term_names=["TERM_A"])
        assert group.category_name == "Mitochondria"
        assert group.term_names == ["TERM_A"]


class TestMappingFileErrorStructure:
    """Verify MappingFileError is a proper exception class."""

    def test_is_exception(self):
        """MappingFileError should be a subclass of Exception."""
        assert issubclass(MappingFileError, Exception)

    def test_can_raise_and_catch(self):
        """MappingFileError can be raised and caught."""
        with pytest.raises(MappingFileError):
            raise MappingFileError("test error")


class TestParseCategoryMappingSignature:
    """Verify parse_category_mapping has the correct signature."""

    def test_accepts_path_parameter(self):
        """parse_category_mapping should accept a 'mapping_path' parameter."""
        sig = inspect.signature(parse_category_mapping)
        assert "mapping_path" in sig.parameters

    def test_is_callable(self):
        """parse_category_mapping should be callable."""
        assert callable(parse_category_mapping)


class TestSelectCherryPickedTermsSignature:
    """Verify select_cherry_picked_terms has the correct signature."""

    def test_accepts_cohort_parameter(self):
        """select_cherry_picked_terms should accept a 'cohort' parameter."""
        sig = inspect.signature(select_cherry_picked_terms)
        assert "cohort" in sig.parameters

    def test_accepts_term_to_category_parameter(self):
        """select_cherry_picked_terms should accept a 'term_to_category' parameter."""
        sig = inspect.signature(select_cherry_picked_terms)
        assert "term_to_category" in sig.parameters

    def test_is_callable(self):
        """select_cherry_picked_terms should be callable."""
        assert callable(select_cherry_picked_terms)


# ---------------------------------------------------------------------------
# parse_category_mapping tests
# ---------------------------------------------------------------------------

class TestParseCategoryMappingBasic:
    """Test basic parsing of well-formed category mapping files."""

    def test_simple_two_column_tsv(self, tmp_path):
        """Contract 1: The mapping file is parsed as a two-column TSV.

        DATA ASSUMPTION: A simple mapping with 3 terms and 2 categories.
        Tab-separated, one entry per line.
        """
        lines = [
            "Oxidative phosphorylation\tMitochondria",
            "Ribosome biogenesis\tTranslation",
            "GPCR signaling\tGPCR",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        result = parse_category_mapping(mapping_file)

        # All term names should be uppercased
        assert "OXIDATIVE PHOSPHORYLATION" in result
        assert "RIBOSOME BIOGENESIS" in result
        assert "GPCR SIGNALING" in result

    def test_returns_dict_str_to_str(self, tmp_path):
        """Contract 1: Returns dict mapping term_name (uppercase) -> category_name."""
        lines = [
            "Synaptic vesicle cycle\tSynapse",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        result = parse_category_mapping(mapping_file)

        assert isinstance(result, dict)
        assert result["SYNAPTIC VESICLE CYCLE"] == "Synapse"

    def test_term_names_uppercased(self, tmp_path):
        """Contract 1: Term names are matched case-insensitively after uppercasing.

        DATA ASSUMPTION: Mixed-case term names in the mapping file should be
        uppercased in the returned dictionary keys.
        """
        lines = [
            "Mitochondrial Translation\tMitochondria",
            "ribosomal rna processing\tTranslation",
            "G PROTEIN COUPLED RECEPTOR\tGPCR",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        result = parse_category_mapping(mapping_file)

        assert "MITOCHONDRIAL TRANSLATION" in result
        assert "RIBOSOMAL RNA PROCESSING" in result
        assert "G PROTEIN COUPLED RECEPTOR" in result

    def test_category_names_preserved(self, tmp_path):
        """Contract 1: Category names should be preserved as-is from the file.

        DATA ASSUMPTION: Category names are one of the four fixed categories
        and should not be modified.
        """
        lines = [
            "Term A\tMitochondria",
            "Term B\tTranslation",
            "Term C\tGPCR",
            "Term D\tSynapse",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        result = parse_category_mapping(mapping_file)

        assert result["TERM A"] == "Mitochondria"
        assert result["TERM B"] == "Translation"
        assert result["TERM C"] == "GPCR"
        assert result["TERM D"] == "Synapse"


class TestParseCategoryMappingComments:
    """Test handling of comment and blank lines in mapping files."""

    def test_comment_lines_skipped(self, tmp_path):
        """Contract 8: Lines starting with '#' are treated as comments and skipped.

        DATA ASSUMPTION: Comment lines start with '#' as per typical TSV conventions.
        """
        lines = [
            "# This is a header comment",
            "Term A\tMitochondria",
            "# Another comment",
            "Term B\tTranslation",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        result = parse_category_mapping(mapping_file)

        assert len(result) == 2
        assert "TERM A" in result
        assert "TERM B" in result

    def test_empty_lines_skipped(self, tmp_path):
        """Contract 8: Empty lines are skipped.

        DATA ASSUMPTION: Empty lines (whitespace only or truly empty) should
        not cause parsing errors.
        """
        lines = [
            "Term A\tMitochondria",
            "",
            "Term B\tTranslation",
            "   ",
            "Term C\tGPCR",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        result = parse_category_mapping(mapping_file)

        assert "TERM A" in result
        assert "TERM B" in result
        assert "TERM C" in result

    def test_mixed_comments_and_empty_lines(self, tmp_path):
        """Contract 8: Mix of comments and empty lines among valid entries.

        DATA ASSUMPTION: File with interspersed comments, blanks, and data.
        """
        lines = [
            "# Category Mapping for Figure 1",
            "",
            "# Mitochondria category",
            "Oxidative phosphorylation\tMitochondria",
            "",
            "# Translation category",
            "Ribosome biogenesis\tTranslation",
            "",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        result = parse_category_mapping(mapping_file)

        assert len(result) == 2

    def test_file_with_only_comments(self, tmp_path):
        """Contract 8: File with only comments should result in empty dict.

        DATA ASSUMPTION: A mapping file with only comment lines and no data
        produces an empty mapping.
        """
        lines = [
            "# This is all comments",
            "# No actual data",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        result = parse_category_mapping(mapping_file)
        assert isinstance(result, dict)
        assert len(result) == 0


class TestParseCategoryMappingErrors:
    """Test error conditions for parse_category_mapping."""

    def test_unparseable_file_raises_mapping_file_error(self, tmp_path):
        """Error condition: MappingFileError raised for unparseable files.

        DATA ASSUMPTION: A line with only one column (no tab separator) is
        considered unparseable for a two-column TSV format.
        """
        lines = [
            "This line has no tab separator",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        with pytest.raises(MappingFileError):
            parse_category_mapping(mapping_file)

    def test_three_columns_raises_mapping_file_error(self, tmp_path):
        """Error condition: MappingFileError raised for more than two columns.

        DATA ASSUMPTION: A line with three tab-separated columns does not
        conform to the expected two-column format.
        """
        lines = [
            "Term A\tMitochondria\textra_column",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        with pytest.raises(MappingFileError):
            parse_category_mapping(mapping_file)

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Pre-condition: mapping_path must point to an existing file.

        DATA ASSUMPTION: A non-existent path should raise an error (either
        MappingFileError or FileNotFoundError or AssertionError).
        """
        nonexistent = tmp_path / "does_not_exist.tsv"
        with pytest.raises((MappingFileError, FileNotFoundError, AssertionError)):
            parse_category_mapping(nonexistent)

    def test_mixed_valid_and_invalid_lines_raises_error(self, tmp_path):
        """Error condition: A file with a mix of valid and malformed lines.

        DATA ASSUMPTION: If any non-comment, non-empty line is malformed,
        the file should be considered unparseable.
        """
        lines = [
            "Term A\tMitochondria",
            "This line has no tab",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        with pytest.raises(MappingFileError):
            parse_category_mapping(mapping_file)


# ---------------------------------------------------------------------------
# select_cherry_picked_terms tests
# ---------------------------------------------------------------------------

class TestSelectCherryPickedTermsBasic:
    """Test basic behavior of select_cherry_picked_terms."""

    def test_returns_list_of_category_groups(self):
        """Basic return type check: should return list[CategoryGroup].

        DATA ASSUMPTION: Two mutants with 4 terms mapped to 2 categories.
        """
        cohort = _make_cohort({
            "mutA": {
                "OXIDATIVE PHOSPHORYLATION": 2.0,
                "RIBOSOME BIOGENESIS": 1.5,
            },
            "mutB": {
                "OXIDATIVE PHOSPHORYLATION": 1.8,
                "RIBOSOME BIOGENESIS": 1.2,
            },
        })
        term_to_category = {
            "OXIDATIVE PHOSPHORYLATION": "Mitochondria",
            "RIBOSOME BIOGENESIS": "Translation",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert isinstance(result, list)
        assert all(isinstance(g, CategoryGroup) for g in result)

    def test_simple_two_category_selection(self):
        """Contract 1: Terms in both mapping and GSEA results are included.

        DATA ASSUMPTION: Two terms, each in a different category, present
        in both the mapping and the GSEA results.
        """
        cohort = _make_cohort({
            "mutA": {
                "MITOCHONDRIAL ELECTRON TRANSPORT": 2.5,
                "TRANSLATION INITIATION": 1.8,
            },
            "mutB": {
                "MITOCHONDRIAL ELECTRON TRANSPORT": 2.0,
                "TRANSLATION INITIATION": 1.5,
            },
        })
        term_to_category = {
            "MITOCHONDRIAL ELECTRON TRANSPORT": "Mitochondria",
            "TRANSLATION INITIATION": "Translation",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        # Should have two groups
        assert len(result) == 2
        # Each group should contain one term
        all_terms = []
        for g in result:
            all_terms.extend(g.term_names)
        assert "MITOCHONDRIAL ELECTRON TRANSPORT" in all_terms
        assert "TRANSLATION INITIATION" in all_terms


class TestSelectCherryPickedTermsFiltering:
    """Test filtering behavior: terms present in mapping vs GSEA results."""

    def test_terms_in_mapping_but_not_results_silently_dropped(self):
        """Contract 2: GO terms present in the mapping but absent from all
        mutant profiles in the cohort are silently dropped.

        DATA ASSUMPTION: "MISSING TERM" is in the mapping but not in any
        mutant's GSEA results. It should be silently dropped.
        """
        cohort = _make_cohort({
            "mutA": {"OXIDATIVE PHOSPHORYLATION": 2.0},
            "mutB": {"OXIDATIVE PHOSPHORYLATION": 1.5},
        })
        term_to_category = {
            "OXIDATIVE PHOSPHORYLATION": "Mitochondria",
            "MISSING TERM": "Translation",  # not in GSEA results
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        # Only Mitochondria category should appear (Translation has no matching terms)
        assert len(result) == 1
        assert result[0].category_name == "Mitochondria"
        assert "OXIDATIVE PHOSPHORYLATION" in result[0].term_names

    def test_terms_in_results_but_not_mapping_silently_ignored(self):
        """Contract 3: GO terms present in the GSEA results but absent from
        the mapping are silently ignored.

        DATA ASSUMPTION: "EXTRA TERM" is in the GSEA results but not in the
        mapping. It should be silently ignored.
        """
        cohort = _make_cohort({
            "mutA": {
                "OXIDATIVE PHOSPHORYLATION": 2.0,
                "EXTRA TERM NOT IN MAPPING": 3.0,
            },
            "mutB": {
                "OXIDATIVE PHOSPHORYLATION": 1.5,
                "EXTRA TERM NOT IN MAPPING": 2.5,
            },
        })
        term_to_category = {
            "OXIDATIVE PHOSPHORYLATION": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        # Only the mapped term should appear
        assert len(result) == 1
        all_terms = []
        for g in result:
            all_terms.extend(g.term_names)
        assert "EXTRA TERM NOT IN MAPPING" not in all_terms
        assert "OXIDATIVE PHOSPHORYLATION" in all_terms

    def test_no_overlap_returns_empty(self):
        """Contracts 2, 3: When there is zero overlap between mapping and
        GSEA results, the output should be an empty list.

        DATA ASSUMPTION: Mapping and GSEA results have completely disjoint
        term sets.
        """
        cohort = _make_cohort({
            "mutA": {"TERM_X": 2.0},
            "mutB": {"TERM_Y": 1.5},
        })
        term_to_category = {
            "TERM_Z": "Mitochondria",
            "TERM_W": "Translation",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert result == []

    def test_case_insensitive_matching(self):
        """Contract 1: Matching is case-insensitive (both sides uppercased).

        DATA ASSUMPTION: The mapping has lowercase term names, GSEA data has
        uppercase (as per upstream invariant). Matching should work because
        both sides are uppercased.
        """
        cohort = _make_cohort({
            "mutA": {"OXIDATIVE PHOSPHORYLATION": 2.0},
        })
        # Term in mapping is already uppercase because parse_category_mapping
        # uppercases it. The dict passed to select_cherry_picked_terms should
        # have uppercase keys.
        term_to_category = {
            "OXIDATIVE PHOSPHORYLATION": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert len(result) == 1
        assert "OXIDATIVE PHOSPHORYLATION" in result[0].term_names


class TestSelectCherryPickedTermsSorting:
    """Test within-category sorting by mean absolute NES descending."""

    def test_terms_sorted_by_mean_abs_nes_descending(self):
        """Contract 4: Within each category, terms are sorted by mean absolute
        NES across all mutants, descending.

        DATA ASSUMPTION: Three terms in the Mitochondria category with known
        NES values across two mutants. Mean |NES| determines the sort order.
        """
        # TERM_HIGH: mutA=3.0, mutB=1.0 -> mean |NES| = (3+1)/2 = 2.0
        # TERM_MED:  mutA=1.0, mutB=2.0 -> mean |NES| = (1+2)/2 = 1.5
        # TERM_LOW:  mutA=0.5, mutB=0.5 -> mean |NES| = (0.5+0.5)/2 = 0.5
        cohort = _make_cohort({
            "mutA": {
                "TERM_HIGH": 3.0,
                "TERM_MED": 1.0,
                "TERM_LOW": 0.5,
            },
            "mutB": {
                "TERM_HIGH": 1.0,
                "TERM_MED": 2.0,
                "TERM_LOW": 0.5,
            },
        })
        term_to_category = {
            "TERM_HIGH": "Mitochondria",
            "TERM_MED": "Mitochondria",
            "TERM_LOW": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert len(result) == 1
        group = result[0]
        assert group.category_name == "Mitochondria"
        assert group.term_names == ["TERM_HIGH", "TERM_MED", "TERM_LOW"]

    def test_negative_nes_uses_absolute_value(self):
        """Contract 4: Mean absolute NES uses absolute values, so negative NES
        values contribute positively to the sort criterion.

        DATA ASSUMPTION: A term with NES=-3.0 has |NES|=3.0, which should
        rank higher than a term with NES=2.0 (|NES|=2.0).
        """
        # NEG_TERM: mutA=-3.0, mutB=-1.0 -> mean |NES| = (3+1)/2 = 2.0
        # POS_TERM: mutA=1.5, mutB=0.5 -> mean |NES| = (1.5+0.5)/2 = 1.0
        cohort = _make_cohort({
            "mutA": {
                "NEG_TERM": -3.0,
                "POS_TERM": 1.5,
            },
            "mutB": {
                "NEG_TERM": -1.0,
                "POS_TERM": 0.5,
            },
        })
        term_to_category = {
            "NEG_TERM": "Synapse",
            "POS_TERM": "Synapse",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert len(result) == 1
        assert result[0].term_names[0] == "NEG_TERM"
        assert result[0].term_names[1] == "POS_TERM"

    def test_mean_nes_uses_zero_for_missing_mutants(self):
        """Contract 4: The mean is computed over all mutants, using NES=0 for
        mutants where the term is absent.

        DATA ASSUMPTION: TERM_A appears in only one of two mutants with NES=4.0.
        Mean |NES| = (4.0 + 0.0)/2 = 2.0.
        TERM_B appears in both mutants with NES=1.5 each.
        Mean |NES| = (1.5 + 1.5)/2 = 1.5.
        TERM_A should rank higher than TERM_B.
        """
        cohort = _make_cohort({
            "mutA": {
                "TERM_A": 4.0,
                "TERM_B": 1.5,
            },
            "mutB": {
                "TERM_B": 1.5,
                # TERM_A is absent from mutB
            },
        })
        term_to_category = {
            "TERM_A": "GPCR",
            "TERM_B": "GPCR",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert len(result) == 1
        assert result[0].term_names == ["TERM_A", "TERM_B"]

    def test_mean_nes_with_three_mutants_one_missing(self):
        """Contract 4: Mean NES with 3 mutants, term absent in one.

        DATA ASSUMPTION: TERM_A has NES=3.0 in mutA, NES=1.5 in mutB,
        absent in mutC. Mean |NES| = (3.0 + 1.5 + 0.0)/3 = 1.5.
        TERM_B has NES=1.0 in all three. Mean |NES| = (1+1+1)/3 = 1.0.
        """
        cohort = _make_cohort({
            "mutA": {"TERM_A": 3.0, "TERM_B": 1.0},
            "mutB": {"TERM_A": 1.5, "TERM_B": 1.0},
            "mutC": {"TERM_B": 1.0},
        })
        term_to_category = {
            "TERM_A": "Translation",
            "TERM_B": "Translation",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert len(result) == 1
        assert result[0].term_names == ["TERM_A", "TERM_B"]


class TestSelectCherryPickedTermsCategoryOrder:
    """Test category ordering follows first-appearance order from mapping."""

    def test_category_order_follows_mapping_insertion_order(self):
        """Contract 5: Categories are returned in the order they first
        appear in the mapping dictionary (which reflects TSV file order).

        DATA ASSUMPTION: One term per category, all present in GSEA results.
        Output order should follow the mapping's insertion order.
        """
        cohort = _make_cohort({
            "mutA": {
                "SYNAPSE TERM": 1.0,
                "GPCR TERM": 1.5,
                "TRANSLATION TERM": 2.0,
                "MITO TERM": 2.5,
            },
        })
        # Supply in a specific order — output should match
        term_to_category = {
            "SYNAPSE TERM": "Synapse",
            "GPCR TERM": "GPCR",
            "TRANSLATION TERM": "Translation",
            "MITO TERM": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 4
        assert result[0].category_name == "Synapse"
        assert result[1].category_name == "GPCR"
        assert result[2].category_name == "Translation"
        assert result[3].category_name == "Mitochondria"

    def test_order_with_subset_of_categories(self):
        """Contract 5: When only some categories have matching terms, the
        returned order follows mapping first-appearance order with absent
        categories omitted.

        DATA ASSUMPTION: Terms only in GPCR and Mitochondria categories.
        Output order follows which category appears first in the mapping.
        """
        cohort = _make_cohort({
            "mutA": {
                "GPCR SIGNALING": 2.0,
                "ELECTRON TRANSPORT": 1.5,
            },
        })
        term_to_category = {
            "GPCR SIGNALING": "GPCR",
            "ELECTRON TRANSPORT": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 2
        assert result[0].category_name == "GPCR"
        assert result[1].category_name == "Mitochondria"

    def test_order_translation_and_synapse(self):
        """Contract 5: Ordering follows mapping insertion order.

        DATA ASSUMPTION: Synapse appears before Translation in mapping.
        """
        cohort = _make_cohort({
            "mutA": {
                "SYNAPTIC VESICLE": 2.5,
                "RIBOSOME ASSEMBLY": 1.0,
            },
        })
        term_to_category = {
            "SYNAPTIC VESICLE": "Synapse",
            "RIBOSOME ASSEMBLY": "Translation",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 2
        assert result[0].category_name == "Synapse"
        assert result[1].category_name == "Translation"

    def test_single_category_returned(self):
        """Contract 5: When only one category has terms, a single-element list
        is returned.

        DATA ASSUMPTION: All terms in the Synapse category.
        """
        cohort = _make_cohort({
            "mutA": {"SYN_A": 2.0, "SYN_B": 1.0},
        })
        term_to_category = {
            "SYN_A": "Synapse",
            "SYN_B": "Synapse",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 1
        assert result[0].category_name == "Synapse"


class TestSelectCherryPickedTermsEmptyCategoryOmission:
    """Test that empty categories are omitted from output."""

    def test_empty_categories_omitted(self):
        """Contract 5 + Invariant: Categories with zero matching terms after
        filtering are omitted from the output.

        DATA ASSUMPTION: Mapping has all four categories but GSEA results only
        contain terms for two of them.
        """
        cohort = _make_cohort({
            "mutA": {
                "MITO TERM": 2.0,
                "SYNAPSE TERM": 1.5,
            },
        })
        term_to_category = {
            "MITO TERM": "Mitochondria",
            "SYNAPSE TERM": "Synapse",
            "MISSING TRANS": "Translation",  # not in GSEA results
            "MISSING GPCR": "GPCR",  # not in GSEA results
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 2
        category_names = [g.category_name for g in result]
        assert "Mitochondria" in category_names
        assert "Synapse" in category_names
        assert "Translation" not in category_names
        assert "GPCR" not in category_names


class TestSelectCherryPickedTermsInvariants:
    """Test post-condition invariants from the blueprint."""

    def test_all_category_names_match_mapping(self):
        """Invariant: All returned category names come from the mapping.

        DATA ASSUMPTION: A cohort with terms in four categories.
        """
        cohort = _make_cohort({
            "mutA": {
                "MITO_A": 2.0,
                "TRANS_A": 1.5,
                "GPCR_A": 1.0,
                "SYN_A": 0.5,
            },
        })
        term_to_category = {
            "MITO_A": "Mitochondria",
            "TRANS_A": "Translation",
            "GPCR_A": "GPCR",
            "SYN_A": "Synapse",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        expected_categories = set(term_to_category.values())
        for group in result:
            assert group.category_name in expected_categories, (
                f"Category '{group.category_name}' is not in the mapping"
            )

    def test_arbitrary_category_names_accepted(self):
        """Categories are not restricted to a fixed set -- any name works.

        DATA ASSUMPTION: A cohort with terms mapped to custom categories.
        """
        cohort = _make_cohort({
            "mutA": {
                "PROTEASOME COMPLEX": 2.0,
                "RIBOSOME ASSEMBLY": 1.5,
            },
        })
        term_to_category = {
            "PROTEASOME COMPLEX": "Proteasome",
            "RIBOSOME ASSEMBLY": "Translation",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert len(result) == 2
        category_names = [g.category_name for g in result]
        assert "Proteasome" in category_names
        assert "Translation" in category_names

    def test_group_count_matches_distinct_categories(self):
        """Number of groups matches the number of distinct categories with matching terms.

        DATA ASSUMPTION: Multiple terms per category.
        """
        cohort = _make_cohort({
            "mutA": {
                "M1": 3.0, "M2": 2.5,
                "T1": 2.0, "T2": 1.5,
                "G1": 1.0, "G2": 0.5,
                "S1": 0.8, "S2": 0.3,
            },
        })
        term_to_category = {
            "M1": "Mitochondria", "M2": "Mitochondria",
            "T1": "Translation", "T2": "Translation",
            "G1": "GPCR", "G2": "GPCR",
            "S1": "Synapse", "S2": "Synapse",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert len(result) == 4

    def test_no_empty_groups(self):
        """Invariant: Empty categories are omitted (all groups have > 0 terms).

        DATA ASSUMPTION: A variety of terms across categories.
        """
        cohort = _make_cohort({
            "mutA": {
                "MITO_1": 2.0,
                "TRANS_1": 1.5,
            },
        })
        term_to_category = {
            "MITO_1": "Mitochondria",
            "TRANS_1": "Translation",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        for group in result:
            assert len(group.term_names) > 0, (
                f"Group '{group.category_name}' has no terms -- "
                "empty categories should be omitted"
            )


class TestSelectCherryPickedTermsInterfaceCompatibility:
    """Test that output is compatible with the dot plot renderer (Unit 5)."""

    def test_output_uses_category_group_type(self):
        """Contract 6: The CategoryGroup data structure is the same type
        consumed by the dot plot renderer (Unit 5).

        DATA ASSUMPTION: Basic cohort with one category.
        """
        cohort = _make_cohort({
            "mutA": {"TERM_A": 2.0},
        })
        term_to_category = {"TERM_A": "Mitochondria"}
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 1
        group = result[0]
        assert isinstance(group, CategoryGroup)
        assert isinstance(group.category_name, str)
        assert isinstance(group.term_names, list)
        assert all(isinstance(t, str) for t in group.term_names)


class TestSelectCherryPickedTermsComprehensive:
    """Comprehensive integration-style tests combining multiple contracts."""

    def test_full_scenario_four_categories(self):
        """Full scenario: Four categories, multiple mutants, some terms missing
        from some mutants. Verify order, sorting, and filtering all at once.

        DATA ASSUMPTION: 8 terms across 4 categories, 3 mutants, with some
        terms absent from some mutants. One extra GSEA term not in mapping.
        One mapping term not in GSEA.
        """
        cohort = _make_cohort({
            "mutA": {
                # Mitochondria terms
                "MITO HIGH": 3.0,
                "MITO LOW": 1.0,
                # Translation terms
                "TRANS HIGH": 2.5,
                "TRANS LOW": 0.5,
                # GPCR terms
                "GPCR ONLY": 2.0,
                # Synapse terms
                "SYN HIGH": 1.5,
                "SYN LOW": 0.8,
                # Extra term not in mapping
                "UNMAPPED TERM": 5.0,
            },
            "mutB": {
                "MITO HIGH": 2.0,
                "MITO LOW": 0.5,
                "TRANS HIGH": 3.0,
                "TRANS LOW": 1.0,
                "GPCR ONLY": 1.5,
                "SYN HIGH": 2.0,
                # SYN LOW absent -> NES=0 for this mutant
                "UNMAPPED TERM": 4.0,
            },
            "mutC": {
                "MITO HIGH": 1.0,
                # MITO LOW absent -> NES=0
                "TRANS HIGH": 1.5,
                "TRANS LOW": 0.8,
                "GPCR ONLY": 0.5,
                "SYN HIGH": 1.0,
                "SYN LOW": 0.3,
                "UNMAPPED TERM": 3.0,
            },
        })
        term_to_category = {
            "MITO HIGH": "Mitochondria",
            "MITO LOW": "Mitochondria",
            "TRANS HIGH": "Translation",
            "TRANS LOW": "Translation",
            "GPCR ONLY": "GPCR",
            "SYN HIGH": "Synapse",
            "SYN LOW": "Synapse",
            "MISSING FROM GSEA": "Synapse",  # Not in GSEA, silently dropped
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        # All four categories have at least one matching term
        assert len(result) == 4

        # Fixed order check
        assert result[0].category_name == "Mitochondria"
        assert result[1].category_name == "Translation"
        assert result[2].category_name == "GPCR"
        assert result[3].category_name == "Synapse"

        # Mitochondria: MITO HIGH mean |NES| = (3+2+1)/3 = 2.0
        #               MITO LOW mean |NES| = (1+0.5+0)/3 = 0.5
        # -> MITO HIGH before MITO LOW
        assert result[0].term_names == ["MITO HIGH", "MITO LOW"]

        # Translation: TRANS HIGH mean |NES| = (2.5+3+1.5)/3 = 2.333
        #              TRANS LOW mean |NES| = (0.5+1+0.8)/3 = 0.767
        # -> TRANS HIGH before TRANS LOW
        assert result[1].term_names == ["TRANS HIGH", "TRANS LOW"]

        # GPCR: only GPCR ONLY
        assert result[2].term_names == ["GPCR ONLY"]

        # Synapse: SYN HIGH mean |NES| = (1.5+2+1)/3 = 1.5
        #          SYN LOW mean |NES| = (0.8+0+0.3)/3 = 0.367
        # -> SYN HIGH before SYN LOW
        # MISSING FROM GSEA was silently dropped
        assert result[3].term_names == ["SYN HIGH", "SYN LOW"]

        # Unmapped term should not appear anywhere
        all_terms = []
        for g in result:
            all_terms.extend(g.term_names)
        assert "UNMAPPED TERM" not in all_terms
        assert "MISSING FROM GSEA" not in all_terms

    def test_all_mapping_terms_absent_from_results(self):
        """Edge case: All terms in the mapping are absent from GSEA results.

        DATA ASSUMPTION: Complete disjoint between mapping and GSEA data.
        """
        cohort = _make_cohort({
            "mutA": {"RESULT_TERM_1": 2.0, "RESULT_TERM_2": 1.0},
        })
        term_to_category = {
            "MAPPED_TERM_1": "Mitochondria",
            "MAPPED_TERM_2": "Translation",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert result == []

    def test_single_term_single_mutant(self):
        """Edge case: One term, one mutant.

        DATA ASSUMPTION: Minimal scenario with a single term and mutant.
        """
        cohort = _make_cohort({
            "mutA": {"SINGLE TERM": 2.5},
        })
        term_to_category = {"SINGLE TERM": "Mitochondria"}
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 1
        assert result[0].category_name == "Mitochondria"
        assert result[0].term_names == ["SINGLE TERM"]

    def test_multiple_terms_same_mean_abs_nes(self):
        """Edge case: Multiple terms in same category with identical mean |NES|.

        DATA ASSUMPTION: Two terms with identical NES values across all mutants.
        Both should appear in the output (order may vary but both must be present).
        """
        cohort = _make_cohort({
            "mutA": {
                "ALPHA TERM": 2.0,
                "BETA TERM": 2.0,
            },
            "mutB": {
                "ALPHA TERM": 1.0,
                "BETA TERM": 1.0,
            },
        })
        term_to_category = {
            "ALPHA TERM": "Mitochondria",
            "BETA TERM": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 1
        assert result[0].category_name == "Mitochondria"
        assert len(result[0].term_names) == 2
        assert set(result[0].term_names) == {"ALPHA TERM", "BETA TERM"}

    def test_many_terms_in_one_category(self):
        """Test with many terms in a single category.

        DATA ASSUMPTION: 5 terms in Mitochondria category with clear NES
        ranking across 2 mutants.
        """
        # Mean |NES| for each term:
        # T1: (5+3)/2 = 4.0
        # T2: (4+2)/2 = 3.0
        # T3: (3+1)/2 = 2.0
        # T4: (2+0.5)/2 = 1.25
        # T5: (1+0.5)/2 = 0.75
        cohort = _make_cohort({
            "mutA": {
                "T1": 5.0, "T2": 4.0, "T3": 3.0, "T4": 2.0, "T5": 1.0,
            },
            "mutB": {
                "T1": 3.0, "T2": 2.0, "T3": 1.0, "T4": 0.5, "T5": 0.5,
            },
        })
        term_to_category = {
            "T1": "Mitochondria",
            "T2": "Mitochondria",
            "T3": "Mitochondria",
            "T4": "Mitochondria",
            "T5": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 1
        assert result[0].category_name == "Mitochondria"
        assert result[0].term_names == ["T1", "T2", "T3", "T4", "T5"]


class TestParseCategoryMappingIntegration:
    """Integration tests combining parse_category_mapping with
    select_cherry_picked_terms."""

    def test_parse_then_select(self, tmp_path):
        """Integration: Parse a mapping file then use the result for selection.

        DATA ASSUMPTION: A realistic mapping file with terms that match
        the cohort data. Verifies the full pipeline from file to groups.
        """
        lines = [
            "# Category mapping for GSEA analysis",
            "Oxidative phosphorylation\tMitochondria",
            "Electron transport chain\tMitochondria",
            "Ribosome biogenesis\tTranslation",
            "GPCR signaling pathway\tGPCR",
            "Synaptic vesicle cycle\tSynapse",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)

        # Parse the mapping
        term_to_category = parse_category_mapping(mapping_file)

        # Verify parsing produced uppercase keys
        assert "OXIDATIVE PHOSPHORYLATION" in term_to_category
        assert "ELECTRON TRANSPORT CHAIN" in term_to_category
        assert "RIBOSOME BIOGENESIS" in term_to_category
        assert "GPCR SIGNALING PATHWAY" in term_to_category
        assert "SYNAPTIC VESICLE CYCLE" in term_to_category

        # Create cohort with matching terms (uppercase per upstream invariant)
        cohort = _make_cohort({
            "mutA": {
                "OXIDATIVE PHOSPHORYLATION": 2.5,
                "ELECTRON TRANSPORT CHAIN": 2.0,
                "RIBOSOME BIOGENESIS": 1.5,
                "GPCR SIGNALING PATHWAY": 1.0,
                "SYNAPTIC VESICLE CYCLE": 0.8,
            },
            "mutB": {
                "OXIDATIVE PHOSPHORYLATION": 2.0,
                "ELECTRON TRANSPORT CHAIN": 1.5,
                "RIBOSOME BIOGENESIS": 2.0,
                "GPCR SIGNALING PATHWAY": 0.5,
                "SYNAPTIC VESICLE CYCLE": 1.2,
            },
        })

        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 4
        assert result[0].category_name == "Mitochondria"
        assert result[1].category_name == "Translation"
        assert result[2].category_name == "GPCR"
        assert result[3].category_name == "Synapse"

        # Mitochondria: OX PHOS mean |NES| = (2.5+2)/2 = 2.25
        #               ETC mean |NES| = (2+1.5)/2 = 1.75
        assert result[0].term_names == [
            "OXIDATIVE PHOSPHORYLATION",
            "ELECTRON TRANSPORT CHAIN",
        ]

    def test_parse_and_select_with_missing_terms(self, tmp_path):
        """Integration: Mapping has terms not in GSEA, GSEA has terms not in
        mapping. Both should be silently handled.

        DATA ASSUMPTION: Partial overlap between mapping and GSEA results.
        """
        lines = [
            "Oxidative phosphorylation\tMitochondria",
            "Missing from gsea\tTranslation",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        term_to_category = parse_category_mapping(mapping_file)

        cohort = _make_cohort({
            "mutA": {
                "OXIDATIVE PHOSPHORYLATION": 2.0,
                "NOT IN MAPPING": 3.0,
            },
        })

        result = select_cherry_picked_terms(cohort, term_to_category)

        # Only Mitochondria should appear
        assert len(result) == 1
        assert result[0].category_name == "Mitochondria"
        assert result[0].term_names == ["OXIDATIVE PHOSPHORYLATION"]


class TestEdgeCasesAndBoundaries:
    """Additional edge case tests."""

    def test_empty_cohort_no_mutants(self):
        """Edge case: Cohort with no mutants but non-empty mapping.

        DATA ASSUMPTION: An empty cohort (no mutants, no profiles). All mapping
        terms are absent from results, so output should be empty.
        """
        cohort = CohortData(
            mutant_ids=[],
            profiles={},
            all_term_names=set(),
            all_go_ids=set(),
        )
        term_to_category = {
            "SOME TERM": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)
        assert result == []

    def test_empty_mapping(self):
        """Edge case: Empty mapping dict means no terms selected.

        DATA ASSUMPTION: An empty mapping should result in no groups.
        """
        cohort = _make_cohort({
            "mutA": {"TERM_A": 2.0},
        })
        result = select_cherry_picked_terms(cohort, {})
        assert result == []

    def test_term_present_in_only_some_mutants(self):
        """Edge case: A mapped term exists in some mutants but not others.
        It should still be included (it exists in at least one mutant profile).

        DATA ASSUMPTION: MITO_A only in mutA, not in mutB. Still selected.
        Mean |NES| = (2.5 + 0.0)/2 = 1.25.
        """
        cohort = _make_cohort({
            "mutA": {"MITO_A": 2.5, "MITO_B": 1.0},
            "mutB": {"MITO_B": 1.5},
        })
        term_to_category = {
            "MITO_A": "Mitochondria",
            "MITO_B": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 1
        assert result[0].category_name == "Mitochondria"
        # MITO_A mean |NES| = (2.5+0)/2 = 1.25
        # MITO_B mean |NES| = (1.0+1.5)/2 = 1.25
        # Both have same mean |NES|, both should be present
        assert len(result[0].term_names) == 2
        assert set(result[0].term_names) == {"MITO_A", "MITO_B"}

    def test_all_nes_zero(self):
        """Edge case: All NES values are 0.0 (mean |NES| = 0.0 for all terms).

        DATA ASSUMPTION: Terms with NES=0 are valid; they should still be
        selected if they appear in both mapping and results.
        """
        cohort = _make_cohort({
            "mutA": {"TERM_A": 0.0, "TERM_B": 0.0},
        })
        term_to_category = {
            "TERM_A": "Mitochondria",
            "TERM_B": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 1
        assert len(result[0].term_names) == 2

    def test_mapping_file_with_trailing_whitespace(self, tmp_path):
        """Edge case: Mapping file entries may have trailing whitespace.

        DATA ASSUMPTION: Trailing spaces/tabs in term names or category names
        should be handled gracefully (stripped or matched).
        """
        lines = [
            "Term A  \tMitochondria  ",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        # This should either parse successfully (stripping whitespace)
        # or raise MappingFileError. The key is it should not crash.
        try:
            result = parse_category_mapping(mapping_file)
            # If it parses, verify the term is there (stripped and uppercased)
            assert "TERM A" in result
        except MappingFileError:
            # Acceptable if the implementation considers this malformed
            pass

    def test_large_number_of_terms_per_category(self):
        """Stress test: Many terms in a single category still sort correctly.

        DATA ASSUMPTION: 20 terms in Translation category with linearly
        decreasing NES values. All should be sorted correctly.
        """
        terms = {f"TERM_{i:02d}": float(20 - i) for i in range(20)}
        cohort = _make_cohort({"mutA": terms})
        term_to_category = {name: "Translation" for name in terms}

        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 1
        assert result[0].category_name == "Translation"
        assert len(result[0].term_names) == 20
        # First term should have highest mean |NES|
        assert result[0].term_names[0] == "TERM_00"
        # Last term should have lowest mean |NES|
        assert result[0].term_names[-1] == "TERM_19"


# ---------------------------------------------------------------------------
# Coverage gap tests -- added by coverage review
# ---------------------------------------------------------------------------


class TestParseCategoryMappingEmptyFields:
    """Test that lines with empty term name or category name raise MappingFileError.

    The blueprint error condition states MappingFileError is raised when the file
    does not conform to two-column TSV format. A line that has two tab-separated
    columns but one column is empty (e.g., '\\tMitochondria' or 'TermA\\t') does
    not conform to meaningful two-column TSV.
    """

    def test_empty_term_name_raises_mapping_file_error(self, tmp_path):
        """Error condition: A line with an empty first column (term name) should
        raise MappingFileError.

        DATA ASSUMPTION: A mapping line like '\\tMitochondria' has two columns
        but the term name is empty, which is invalid.
        """
        lines = [
            "\tMitochondria",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        with pytest.raises(MappingFileError):
            parse_category_mapping(mapping_file)

    def test_empty_category_name_raises_mapping_file_error(self, tmp_path):
        """Error condition: A line with an empty second column (category name)
        should raise MappingFileError.

        DATA ASSUMPTION: A mapping line like 'Term A\\t' has two columns
        but the category name is empty, which is invalid.
        """
        lines = [
            "Term A\t",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        with pytest.raises(MappingFileError):
            parse_category_mapping(mapping_file)


class TestParseCategoryMappingNonexistentFileSpecific:
    """Test that nonexistent mapping_path specifically raises MappingFileError.

    The blueprint invariant states mapping_path must be an existing file, and
    the error condition specifies MappingFileError. This test verifies the
    specific exception type.
    """

    def test_nonexistent_file_raises_mapping_file_error_specifically(self, tmp_path):
        """Invariant: mapping_path must be an existing file. The blueprint error
        condition is MappingFileError.

        DATA ASSUMPTION: A path that does not correspond to any file on disk.
        """
        nonexistent = tmp_path / "nonexistent_mapping.tsv"
        with pytest.raises(MappingFileError):
            parse_category_mapping(nonexistent)


class TestEndToEndCaseInsensitiveMatching:
    """Test that case-insensitive matching works end-to-end from file parsing
    through term selection.

    Contract 1 specifies that the first column (GO term name) is matched
    case-insensitively against GSEA data after uppercasing both sides.
    """

    def test_lowercase_mapping_matches_uppercase_cohort(self, tmp_path):
        """Contract 1: A mapping file with lowercase term names should match
        uppercase terms in the cohort after uppercasing.

        DATA ASSUMPTION: The mapping file uses lowercase 'oxidative phosphorylation',
        while the cohort has the term as 'OXIDATIVE PHOSPHORYLATION' (uppercase,
        as produced by upstream Unit 1). Both sides are uppercased for matching.
        """
        lines = [
            "oxidative phosphorylation\tMitochondria",
            "ribosome biogenesis\tTranslation",
        ]
        mapping_file = _write_mapping_file(tmp_path, lines)
        term_to_category = parse_category_mapping(mapping_file)

        cohort = _make_cohort({
            "mutA": {
                "OXIDATIVE PHOSPHORYLATION": 2.5,
                "RIBOSOME BIOGENESIS": 1.5,
            },
            "mutB": {
                "OXIDATIVE PHOSPHORYLATION": 2.0,
                "RIBOSOME BIOGENESIS": 1.0,
            },
        })
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 2
        all_terms = []
        for g in result:
            all_terms.extend(g.term_names)
        assert "OXIDATIVE PHOSPHORYLATION" in all_terms
        assert "RIBOSOME BIOGENESIS" in all_terms


class TestMixedSignNESSorting:
    """Test sorting when a single term has both positive and negative NES
    values across different mutants.

    Contract 4 states terms are sorted by mean absolute NES. When a term
    has positive NES in one mutant and negative in another, both should
    contribute their absolute values to the mean.
    """

    def test_mixed_positive_negative_nes_within_same_term(self):
        """Contract 4: A term with NES=+2.0 in mutA and NES=-3.0 in mutB
        should have mean |NES| = (2.0 + 3.0)/2 = 2.5.

        DATA ASSUMPTION: TERM_MIXED has positive NES in mutA and negative in mutB.
        TERM_CONSISTENT has moderate positive NES in both mutants.
        Mean |NES| for TERM_MIXED = (2.0 + 3.0)/2 = 2.5
        Mean |NES| for TERM_CONSISTENT = (1.5 + 1.5)/2 = 1.5
        TERM_MIXED should rank first.
        """
        cohort = _make_cohort({
            "mutA": {
                "TERM_MIXED": 2.0,
                "TERM_CONSISTENT": 1.5,
            },
            "mutB": {
                "TERM_MIXED": -3.0,
                "TERM_CONSISTENT": 1.5,
            },
        })
        term_to_category = {
            "TERM_MIXED": "Mitochondria",
            "TERM_CONSISTENT": "Mitochondria",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 1
        assert result[0].term_names == ["TERM_MIXED", "TERM_CONSISTENT"]


class TestArbitraryCategoryNamesInOutput:
    """Test that any user-defined category name is accepted in the output.

    Category names are not restricted to a fixed set. Any category name
    from the mapping file is valid and will appear in the output if it
    has matching terms.
    """

    def test_custom_category_name_appears_in_output(self):
        """Any category name from the mapping is accepted.

        DATA ASSUMPTION: TERM_A is mapped to 'Mitochondria',
        TERM_B is mapped to 'CellCycle' (a custom category).
        Both should appear in the output.
        """
        cohort = _make_cohort({
            "mutA": {
                "TERM_A": 2.0,
                "TERM_B": 3.0,
            },
        })
        term_to_category = {
            "TERM_A": "Mitochondria",
            "TERM_B": "CellCycle",
        }
        result = select_cherry_picked_terms(cohort, term_to_category)

        assert len(result) == 2
        category_names = [g.category_name for g in result]
        assert "Mitochondria" in category_names
        assert "CellCycle" in category_names
