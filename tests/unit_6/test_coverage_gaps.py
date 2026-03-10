"""
Coverage gap tests for Unit 6: Meta-Analysis Computation.

These tests address behavioral contracts from the blueprint that are not
covered by the existing test suite in test_meta_analysis.py.

Synthetic Data Assumptions
==========================
DATA ASSUMPTION: GO IDs follow the standard Gene Ontology format "GO:NNNNNNN"
    (e.g., GO:0008150, GO:0003674). These are realistic identifiers.

DATA ASSUMPTION: Nominal p-values are in [0.0, 1.0], representing statistical
    significance from GSEA preranked analyses.

DATA ASSUMPTION: A cohort contains at least 2 mutant lines, each with enrichment
    profiles for various GO terms.

DATA ASSUMPTION: Pseudocount is a small positive number (default 1e-10) used to
    replace exact-zero p-values before log transformation.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.configuration import FisherConfig
from gsea_tool.meta_analysis import (
    FisherResult,
    build_pvalue_dict_per_mutant,
    build_pvalue_matrix,
    compute_fisher_combined,
    run_fisher_analysis,
    write_pvalue_matrix_tsv,
    write_fisher_results_tsv,
)


# ---------------------------------------------------------------------------
# Helper factories (same pattern as existing test suite)
# ---------------------------------------------------------------------------

def _make_term_record(
    term_name: str = "SIGNAL TRANSDUCTION",
    go_id: str = "GO:0007165",
    nes: float = 1.5,
    fdr: float = 0.01,
    nom_pval: float = 0.005,
    size: int = 150,
) -> TermRecord:
    return TermRecord(
        term_name=term_name,
        go_id=go_id,
        nes=nes,
        fdr=fdr,
        nom_pval=nom_pval,
        size=size,
    )


def _make_mutant_profile(
    mutant_id: str,
    records: dict[str, TermRecord] | None = None,
) -> MutantProfile:
    if records is None:
        records = {}
    return MutantProfile(mutant_id=mutant_id, records=records)


def _make_cohort(
    mutant_terms: dict[str, dict[str, tuple[str, float]]],
) -> CohortData:
    """Build a CohortData from a concise specification.

    mutant_terms: {mutant_id: {term_name: (go_id, nom_pval)}}
    """
    profiles: dict[str, MutantProfile] = {}
    all_term_names: set[str] = set()
    all_go_ids: set[str] = set()
    mutant_ids = sorted(mutant_terms.keys())

    for mid in mutant_ids:
        terms = mutant_terms[mid]
        records: dict[str, TermRecord] = {}
        for term_name, (go_id, nom_pval) in terms.items():
            rec = _make_term_record(
                term_name=term_name,
                go_id=go_id,
                nom_pval=nom_pval,
            )
            records[term_name] = rec
            all_term_names.add(term_name)
            all_go_ids.add(go_id)
        profiles[mid] = _make_mutant_profile(mid, records)

    return CohortData(
        mutant_ids=mutant_ids,
        profiles=profiles,
        all_term_names=all_term_names,
        all_go_ids=all_go_ids,
    )


# ===========================================================================
# Gap 1: All-1.0 p-values yield combined p = 1.0 (Contract 3 + 4)
#
# Blueprint: "Missing entries (GO term absent from a mutant) are imputed as
# p = 1.0, contributing 0 to the Fisher statistic since ln(1.0) = 0."
# When all entries are 1.0, Fisher stat = 0, and chi2.sf(0, df) = 1.0.
# ===========================================================================


class TestImputedValuesContributeZero:
    """Verify that imputed p=1.0 contributes nothing to Fisher statistic."""

    def test_all_ones_yield_combined_pvalue_of_one(self):
        """When all p-values are 1.0, Fisher stat is 0, combined p = 1.0."""
        # DATA ASSUMPTION: 2 GO terms, 3 mutants, all p=1.0 (fully imputed)
        matrix = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        result = compute_fisher_combined(matrix, n_mutants=3)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)

    def test_imputed_entries_do_not_change_fisher_stat(self):
        """A GO term with one real p-value and one imputed (1.0) should produce
        the same Fisher statistic as if the imputed entry contributed 0.

        Contract 4: X^2 = -2 * sum(ln(p_i)); since ln(1.0)=0, the imputed
        entry contributes exactly 0 to the statistic.
        """
        # DATA ASSUMPTION: Single GO term, 2 mutants. One has p=0.01, other
        # is imputed as 1.0. The Fisher stat should equal -2*ln(0.01) + 0.
        p_real = 0.01
        matrix = np.array([[p_real, 1.0]])
        n_mutants = 2

        result = compute_fisher_combined(matrix, n_mutants=n_mutants)

        # Manual calculation: X^2 = -2*(ln(0.01) + ln(1.0)) = -2*ln(0.01)
        expected_chi2 = -2.0 * (math.log(p_real) + math.log(1.0))
        expected_pval = chi2.sf(expected_chi2, df=2 * n_mutants)
        assert result[0] == pytest.approx(expected_pval, rel=1e-6)


# ===========================================================================
# Gap 2: Degrees of freedom constant at 2k for ALL GO terms (Contract 4)
#
# Blueprint: "The degrees of freedom are 2k, constant across all GO terms
# due to imputation."
# ===========================================================================


class TestConstantDegreesOfFreedom:
    """Verify df = 2k is used for all GO terms regardless of imputation pattern."""

    def test_df_constant_across_rows_with_different_imputation(self):
        """GO terms with different numbers of real vs imputed p-values should
        all use the same df = 2k.

        DATA ASSUMPTION: 2 GO terms across 3 mutants. GO:0000001 has all 3
        real p-values; GO:0000002 has only 1 real p-value and 2 imputed.
        Both should use df=6.
        """
        # Row 0: all real
        # Row 1: one real (0.01), two imputed (1.0, 1.0)
        matrix = np.array([
            [0.01, 0.02, 0.03],
            [0.01, 1.0,  1.0],
        ])
        n_mutants = 3

        result = compute_fisher_combined(matrix, n_mutants=n_mutants)

        # Verify both rows used df=2k=6
        df = 2 * n_mutants

        # Row 0
        chi2_stat_0 = -2.0 * np.sum(np.log(matrix[0, :]))
        expected_0 = chi2.sf(chi2_stat_0, df)
        assert result[0] == pytest.approx(expected_0, rel=1e-6)

        # Row 1
        chi2_stat_1 = -2.0 * np.sum(np.log(matrix[1, :]))
        expected_1 = chi2.sf(chi2_stat_1, df)
        assert result[1] == pytest.approx(expected_1, rel=1e-6)


# ===========================================================================
# Gap 3: pvalue_matrix.tsv content verification (Contract 9)
#
# Blueprint: "Columns are mutant IDs, rows are GO IDs with an additional
# column for GO term name."
# ===========================================================================


class TestPvalueMatrixTsvContent:
    """Verify the content of pvalue_matrix.tsv matches Contract 9."""

    def test_header_contains_mutant_ids_as_columns(self, tmp_path):
        """The TSV header should include mutant IDs as column names."""
        # DATA ASSUMPTION: 1 GO term, 2 mutants
        matrix = np.array([[0.01, 0.02]])
        go_id_order = ["GO:0000001"]
        go_id_to_name = {"GO:0000001": "SIGNAL TRANSDUCTION"}
        mutant_ids = ["mutantA", "mutantB"]

        path = write_pvalue_matrix_tsv(matrix, go_id_order, go_id_to_name, mutant_ids, tmp_path)
        content = path.read_text()
        header = content.strip().split("\n")[0]

        assert "mutantA" in header
        assert "mutantB" in header

    def test_data_rows_contain_go_id_and_term_name(self, tmp_path):
        """Each data row should start with GO ID and term name."""
        # DATA ASSUMPTION: 2 GO terms, 2 mutants
        matrix = np.array([[0.01, 0.02], [0.03, 0.04]])
        go_id_order = ["GO:0000001", "GO:0000002"]
        go_id_to_name = {"GO:0000001": "CELL CYCLE", "GO:0000002": "APOPTOTIC PROCESS"}
        mutant_ids = ["mutantA", "mutantB"]

        path = write_pvalue_matrix_tsv(matrix, go_id_order, go_id_to_name, mutant_ids, tmp_path)
        content = path.read_text()
        lines = content.strip().split("\n")

        # Data rows (skip header)
        data_lines = lines[1:]
        assert len(data_lines) == 2

        # First data row should contain GO:0000001 and CELL CYCLE
        fields_0 = data_lines[0].split("\t")
        assert fields_0[0] == "GO:0000001"
        assert fields_0[1] == "CELL CYCLE"

        # Second data row should contain GO:0000002 and APOPTOTIC PROCESS
        fields_1 = data_lines[1].split("\t")
        assert fields_1[0] == "GO:0000002"
        assert fields_1[1] == "APOPTOTIC PROCESS"

    def test_returned_path_is_pvalue_matrix_tsv(self, tmp_path):
        """write_pvalue_matrix_tsv should return a path named pvalue_matrix.tsv."""
        matrix = np.array([[0.01, 0.02]])
        go_id_order = ["GO:0000001"]
        go_id_to_name = {"GO:0000001": "TERM_A"}
        mutant_ids = ["mutantA", "mutantB"]

        path = write_pvalue_matrix_tsv(matrix, go_id_order, go_id_to_name, mutant_ids, tmp_path)
        assert path.name == "pvalue_matrix.tsv"


# ===========================================================================
# Gap 4: fisher_combined_pvalues.tsv content verification (Contract 10)
#
# Blueprint: "containing GO ID, GO term name, combined p-value, and number
# of contributing lines. No cluster assignment column is included."
# ===========================================================================


class TestFisherResultsTsvContent:
    """Verify the content of fisher_combined_pvalues.tsv matches Contract 10."""

    def _make_fisher_result(self):
        """Create a FisherResult for testing TSV content.

        DATA ASSUMPTION: 2 GO terms, 2 mutants with known combined p-values.
        """
        return FisherResult(
            go_ids=["GO:0000001", "GO:0000002"],
            go_id_to_name={"GO:0000001": "CELL CYCLE", "GO:0000002": "APOPTOTIC PROCESS"},
            combined_pvalues={"GO:0000001": 0.001, "GO:0000002": 0.05},
            n_contributing={"GO:0000001": 2, "GO:0000002": 1},
            pvalue_matrix=np.array([[0.01, 0.02], [0.03, 1.0]]),
            mutant_ids=["mutantA", "mutantB"],
            go_id_order=["GO:0000001", "GO:0000002"],
            n_mutants=2,
            corrected_pvalues=None,
        )

    def test_header_contains_expected_columns(self, tmp_path):
        """The header should include GO ID, GO term name, combined p-value,
        and n_contributing columns."""
        fr = self._make_fisher_result()
        path = write_fisher_results_tsv(fr, tmp_path)
        content = path.read_text()
        header = content.strip().split("\n")[0].lower()

        # Should contain identifying information for each expected column
        header_fields = content.strip().split("\n")[0].split("\t")
        # Expect exactly 4 columns (no cluster assignment)
        assert len(header_fields) == 4, (
            f"Expected 4 columns (GO ID, term name, combined pvalue, n_contributing), "
            f"got {len(header_fields)}: {header_fields}"
        )

    def test_data_rows_contain_go_id_and_term_name(self, tmp_path):
        """Each data row should include the GO ID and its term name."""
        fr = self._make_fisher_result()
        path = write_fisher_results_tsv(fr, tmp_path)
        content = path.read_text()
        lines = content.strip().split("\n")
        data_lines = lines[1:]

        assert len(data_lines) == 2

        # Check first row has GO:0000001 and CELL CYCLE
        fields_0 = data_lines[0].split("\t")
        assert "GO:0000001" in fields_0[0]
        assert "CELL CYCLE" in fields_0[1]

    def test_no_cluster_assignment_column(self, tmp_path):
        """Contract 10: No cluster assignment column should be present."""
        fr = self._make_fisher_result()
        path = write_fisher_results_tsv(fr, tmp_path)
        content = path.read_text()
        header = content.strip().split("\n")[0].lower()

        # "cluster" should not appear in the header
        assert "cluster" not in header, (
            f"fisher_combined_pvalues.tsv should not contain a cluster column, "
            f"but header is: {header}"
        )

    def test_data_rows_contain_combined_pvalue_and_n_contributing(self, tmp_path):
        """Each data row should include the combined p-value and n_contributing."""
        fr = self._make_fisher_result()
        path = write_fisher_results_tsv(fr, tmp_path)
        content = path.read_text()
        lines = content.strip().split("\n")
        data_lines = lines[1:]

        # First row: GO:0000001, combined_p=0.001, n_contributing=2
        fields_0 = data_lines[0].split("\t")
        assert float(fields_0[2]) == pytest.approx(0.001)
        assert int(fields_0[3]) == 2

    def test_returned_path_is_fisher_combined_pvalues_tsv(self, tmp_path):
        """write_fisher_results_tsv should return a path named fisher_combined_pvalues.tsv."""
        fr = self._make_fisher_result()
        path = write_fisher_results_tsv(fr, tmp_path)
        assert path.name == "fisher_combined_pvalues.tsv"


# ===========================================================================
# Gap 5: go_id_to_name values are correct term names (Contract 11)
#
# Blueprint: "The go_id_to_name mapping allows downstream units and TSV
# outputs to display human-readable term names alongside GO IDs."
# ===========================================================================


class TestGoIdToNameValues:
    """Verify go_id_to_name contains correct mappings from cohort data."""

    def test_go_id_to_name_has_correct_values(self, tmp_path):
        """go_id_to_name values should match the term_name from the TermRecords."""
        # DATA ASSUMPTION: 2 mutants sharing the same GO terms
        cohort = _make_cohort({
            "mutantA": {
                "SIGNAL TRANSDUCTION": ("GO:0007165", 0.01),
                "CELL CYCLE": ("GO:0007049", 0.05),
            },
            "mutantB": {
                "SIGNAL TRANSDUCTION": ("GO:0007165", 0.02),
                "CELL CYCLE": ("GO:0007049", 0.03),
            },
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert result.go_id_to_name["GO:0007165"] == "SIGNAL TRANSDUCTION"
        assert result.go_id_to_name["GO:0007049"] == "CELL CYCLE"

    def test_go_id_to_name_covers_all_go_ids(self, tmp_path):
        """go_id_to_name should have an entry for every GO ID in the result."""
        # DATA ASSUMPTION: 2 mutants with disjoint GO terms
        cohort = _make_cohort({
            "mutantA": {
                "TERM_X": ("GO:0000001", 0.01),
            },
            "mutantB": {
                "TERM_Y": ("GO:0000002", 0.02),
            },
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        for go_id in result.go_ids:
            assert go_id in result.go_id_to_name, (
                f"go_id_to_name missing entry for {go_id}"
            )
        assert result.go_id_to_name["GO:0000001"] == "TERM_X"
        assert result.go_id_to_name["GO:0000002"] == "TERM_Y"


# ===========================================================================
# Gap 6: build_pvalue_dict_per_mutant extracts from go_id field (Contract 1)
#
# Blueprint: "Per-mutant p-value dictionaries are keyed by GO ID (not term
# name), extracted from the go_id field of TermRecord."
#
# Existing tests check keys are GO IDs, but do not confirm that when a
# mutant has two different GO IDs with distinct term names, both GO IDs
# appear as keys (verifying go_id is used, not term_name).
# ===========================================================================


class TestPvalueDictKeyedByGoIdNotTermName:
    """Ensure dictionaries are keyed by GO ID, not by term name."""

    def test_distinct_go_ids_as_keys(self):
        """When records have different go_id values, all appear as dict keys."""
        # DATA ASSUMPTION: 2 mutants, 3 distinct GO IDs in mutantA
        cohort = _make_cohort({
            "mutantA": {
                "TERM_ALPHA": ("GO:0000001", 0.01),
                "TERM_BETA": ("GO:0000002", 0.05),
                "TERM_GAMMA": ("GO:0000003", 0.10),
            },
            "mutantB": {
                "TERM_ALPHA": ("GO:0000001", 0.02),
            },
        })
        result = build_pvalue_dict_per_mutant(cohort, 1e-10)

        # mutantA should have 3 GO IDs as keys, not term names
        assert set(result["mutantA"].keys()) == {"GO:0000001", "GO:0000002", "GO:0000003"}
        # Verify term names are NOT used as keys
        assert "TERM_ALPHA" not in result["mutantA"]
        assert "TERM_BETA" not in result["mutantA"]
        assert "TERM_GAMMA" not in result["mutantA"]


# ===========================================================================
# Gap 7: Imputation at p=1.0 means ln(1.0)=0, contributing 0 to Fisher
# stat -- verified through build_pvalue_matrix (Contract 3)
# ===========================================================================


class TestImputationValueIsOne:
    """Verify that all missing entries in the matrix are exactly 1.0."""

    def test_all_missing_entries_are_exactly_one(self):
        """When mutants have completely disjoint GO terms, the cross-entries
        should be exactly 1.0 (not NaN, not 0, not any other value)."""
        # DATA ASSUMPTION: 3 mutants, each with a unique GO term
        per_mutant = {
            "m1": {"GO:0000001": 0.01},
            "m2": {"GO:0000002": 0.02},
            "m3": {"GO:0000003": 0.03},
        }
        mutant_ids = ["m1", "m2", "m3"]
        matrix, go_id_order = build_pvalue_matrix(per_mutant, mutant_ids)

        # matrix shape: 3 GO terms x 3 mutants
        assert matrix.shape == (3, 3)

        for i, go_id in enumerate(go_id_order):
            for j, mid in enumerate(mutant_ids):
                if go_id in per_mutant[mid]:
                    assert matrix[i, j] == per_mutant[mid][go_id]
                else:
                    assert matrix[i, j] == 1.0, (
                        f"Expected 1.0 for missing ({go_id}, {mid}), got {matrix[i, j]}"
                    )


# ===========================================================================
# Gap 8: n_contributing = 0 for a GO term that only has imputed values
#
# This is an edge case of Contract 8: when a GO term has p < 1.0 in zero
# mutant lines (all imputed), n_contributing should be 0.
#
# Note: This scenario cannot naturally happen because a GO term only enters
# the union if at least one mutant has it. But it's still good to verify
# the counting logic in the compute path. We test this through
# build_pvalue_matrix + run_fisher_analysis.
# ===========================================================================


class TestNContributingEdgeCases:
    """Edge cases for n_contributing counting."""

    def test_n_contributing_for_term_in_only_one_mutant(self, tmp_path):
        """A GO term present in exactly 1 out of 3 mutants should have
        n_contributing = 1."""
        # DATA ASSUMPTION: 3 mutants, GO:0000099 only in mutantA
        cohort = _make_cohort({
            "mutantA": {
                "TERM_UNIQUE": ("GO:0000099", 0.01),
                "TERM_SHARED": ("GO:0000001", 0.05),
            },
            "mutantB": {
                "TERM_SHARED": ("GO:0000001", 0.03),
            },
            "mutantC": {
                "TERM_SHARED": ("GO:0000001", 0.02),
            },
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert result.n_contributing["GO:0000099"] == 1
        assert result.n_contributing["GO:0000001"] == 3


# ===========================================================================
# Gap 9: Pre-condition -- at least 2 mutant lines required
#
# Blueprint invariant: "Fisher's method requires at least 2 mutant lines"
# Implementation: assert len(cohort.mutant_ids) >= 2
# ===========================================================================


class TestPreConditionMinimumMutants:
    """Verify that run_fisher_analysis enforces the minimum mutant count."""

    def test_single_mutant_raises_assertion_error(self, tmp_path):
        """Providing only 1 mutant line should raise AssertionError.

        DATA ASSUMPTION: A cohort with a single mutant, which violates the
        precondition that Fisher's method requires at least 2 mutant lines.
        """
        cohort = _make_cohort({
            "mutantA": {
                "TERM_X": ("GO:0000001", 0.01),
            },
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        with pytest.raises(AssertionError, match="at least 2"):
            run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)


# ===========================================================================
# Gap 10: Pre-condition -- pseudocount must be positive
#
# Blueprint invariant: "Pseudocount must be positive"
# Implementation: assert config.pseudocount > 0
# ===========================================================================


class TestPreConditionPositivePseudocount:
    """Verify that run_fisher_analysis enforces positive pseudocount."""

    def test_zero_pseudocount_raises_assertion_error(self, tmp_path):
        """A pseudocount of 0.0 should raise AssertionError.

        DATA ASSUMPTION: 2 mutants with valid data but pseudocount=0.0,
        violating the precondition.
        """
        cohort = _make_cohort({
            "mutantA": {
                "TERM_X": ("GO:0000001", 0.01),
            },
            "mutantB": {
                "TERM_X": ("GO:0000001", 0.02),
            },
        })
        config = FisherConfig(pseudocount=0.0, apply_fdr=False)
        with pytest.raises(AssertionError, match="[Pp]seudocount"):
            run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

    def test_negative_pseudocount_raises_assertion_error(self, tmp_path):
        """A negative pseudocount should raise AssertionError.

        DATA ASSUMPTION: 2 mutants with valid data but pseudocount=-1e-10,
        violating the precondition.
        """
        cohort = _make_cohort({
            "mutantA": {
                "TERM_X": ("GO:0000001", 0.01),
            },
            "mutantB": {
                "TERM_X": ("GO:0000001", 0.02),
            },
        })
        config = FisherConfig(pseudocount=-1e-10, apply_fdr=False)
        with pytest.raises(AssertionError, match="[Pp]seudocount"):
            run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)


# ===========================================================================
# Gap 11: Pre-condition -- output_dir must exist
#
# Blueprint invariant: "Output directory must exist"
# Implementation: assert output_dir.is_dir()
# ===========================================================================


class TestPreConditionOutputDirExists:
    """Verify that run_fisher_analysis enforces output_dir existence."""

    def test_nonexistent_output_dir_raises_assertion_error(self):
        """Providing a non-existent output directory should raise AssertionError.

        DATA ASSUMPTION: 2 mutants with valid data but output_dir points to
        a directory that does not exist.
        """
        cohort = _make_cohort({
            "mutantA": {
                "TERM_X": ("GO:0000001", 0.01),
            },
            "mutantB": {
                "TERM_X": ("GO:0000001", 0.02),
            },
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        nonexistent = Path("/tmp/nonexistent_dir_for_unit6_test_xyz")
        with pytest.raises(AssertionError, match="[Oo]utput directory"):
            run_fisher_analysis(cohort, config, nonexistent, clustering_enabled=False)


# ===========================================================================
# Gap 12: pvalue_matrix.tsv numeric data verification (Contract 9)
#
# Existing tests verify header and structural aspects of pvalue_matrix.tsv
# but do not verify that the actual numeric p-values in the TSV match the
# matrix values.
# ===========================================================================


class TestPvalueMatrixTsvNumericValues:
    """Verify that numeric p-values in pvalue_matrix.tsv match the matrix."""

    def test_data_values_match_matrix(self, tmp_path):
        """The numeric values in pvalue_matrix.tsv rows should match the
        corresponding matrix entries.

        DATA ASSUMPTION: 2 GO terms, 2 mutants with known p-values.
        """
        matrix = np.array([[0.01, 0.02], [0.03, 0.04]])
        go_id_order = ["GO:0000001", "GO:0000002"]
        go_id_to_name = {"GO:0000001": "TERM_A", "GO:0000002": "TERM_B"}
        mutant_ids = ["mutantA", "mutantB"]

        path = write_pvalue_matrix_tsv(matrix, go_id_order, go_id_to_name, mutant_ids, tmp_path)
        content = path.read_text()
        lines = content.strip().split("\n")
        data_lines = lines[1:]

        # First data row: GO:0000001, TERM_A, 0.01, 0.02
        fields_0 = data_lines[0].split("\t")
        assert float(fields_0[2]) == pytest.approx(0.01)
        assert float(fields_0[3]) == pytest.approx(0.02)

        # Second data row: GO:0000002, TERM_B, 0.03, 0.04
        fields_1 = data_lines[1].split("\t")
        assert float(fields_1[2]) == pytest.approx(0.03)
        assert float(fields_1[3]) == pytest.approx(0.04)
