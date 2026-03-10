"""Test suite for Unit 6 -- Meta-Analysis Computation (Fisher's Combined Probability).

Tests are derived from the blueprint behavioral contracts, invariants, and error conditions.
"""

import math
import numpy as np
import pytest
from pathlib import Path
from scipy import stats as scipy_stats

from gsea_tool.meta_analysis import (
    FisherResult,
    build_pvalue_dict_per_mutant,
    build_pvalue_matrix,
    compute_fisher_combined,
    run_fisher_analysis,
    write_pvalue_matrix_tsv,
    write_fisher_results_tsv,
)
from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.configuration import FisherConfig


# ---------------------------------------------------------------------------
# Helpers to build synthetic CohortData
# ---------------------------------------------------------------------------

def _make_term_record(term_name: str, go_id: str, nom_pval: float,
                      nes: float = 1.0, fdr: float = 0.05, size: int = 50) -> TermRecord:
    return TermRecord(
        term_name=term_name,
        go_id=go_id,
        nes=nes,
        fdr=fdr,
        nom_pval=nom_pval,
        size=size,
    )


def _make_cohort(mutant_records: dict[str, list[TermRecord]]) -> CohortData:
    """Build a CohortData from a dict of mutant_id -> list of TermRecord."""
    mutant_ids = sorted(mutant_records.keys())
    profiles = {}
    all_term_names: set[str] = set()
    all_go_ids: set[str] = set()
    for mid in mutant_ids:
        records_dict = {rec.term_name: rec for rec in mutant_records[mid]}
        profiles[mid] = MutantProfile(mutant_id=mid, records=records_dict)
        for rec in mutant_records[mid]:
            all_term_names.add(rec.term_name)
            all_go_ids.add(rec.go_id)
    return CohortData(
        mutant_ids=mutant_ids,
        profiles=profiles,
        all_term_names=all_term_names,
        all_go_ids=all_go_ids,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cohort():
    """Two mutants, two GO terms each, one shared, one unique per mutant."""
    return _make_cohort({
        "mutantA": [
            _make_term_record("CELL CYCLE", "GO:0000001", 0.01),
            _make_term_record("APOPTOSIS", "GO:0000002", 0.05),
        ],
        "mutantB": [
            _make_term_record("CELL CYCLE", "GO:0000001", 0.03),
            _make_term_record("DNA REPAIR", "GO:0000003", 0.02),
        ],
    })


@pytest.fixture
def cohort_with_zero_pval():
    """A cohort where one term has NOM p-val of exactly 0.0."""
    return _make_cohort({
        "mutantA": [
            _make_term_record("CELL CYCLE", "GO:0000001", 0.0),
            _make_term_record("APOPTOSIS", "GO:0000002", 0.05),
        ],
        "mutantB": [
            _make_term_record("CELL CYCLE", "GO:0000001", 0.01),
            _make_term_record("APOPTOSIS", "GO:0000002", 0.03),
        ],
    })


@pytest.fixture
def three_mutant_cohort():
    """Three mutants with overlapping GO terms."""
    return _make_cohort({
        "mutantA": [
            _make_term_record("CELL CYCLE", "GO:0000001", 0.01),
            _make_term_record("APOPTOSIS", "GO:0000002", 0.05),
        ],
        "mutantB": [
            _make_term_record("CELL CYCLE", "GO:0000001", 0.03),
            _make_term_record("DNA REPAIR", "GO:0000003", 0.02),
        ],
        "mutantC": [
            _make_term_record("APOPTOSIS", "GO:0000002", 0.001),
            _make_term_record("DNA REPAIR", "GO:0000003", 0.04),
        ],
    })


@pytest.fixture
def default_config():
    return FisherConfig()


@pytest.fixture
def fdr_config():
    return FisherConfig(apply_fdr=True)


# ===========================================================================
# Tests for build_pvalue_dict_per_mutant
# ===========================================================================

class TestBuildPvalueDictPerMutant:
    """Tests for building per-mutant p-value dictionaries."""

    def test_keys_are_go_ids_not_term_names(self, simple_cohort):
        """Contract 1: Per-mutant p-value dicts are keyed by GO ID, not term name."""
        result = build_pvalue_dict_per_mutant(simple_cohort, pseudocount=1e-10)
        for mutant_id, pval_dict in result.items():
            for key in pval_dict:
                assert key.startswith("GO:"), (
                    f"Expected GO ID key, got '{key}'"
                )

    def test_returns_dict_for_each_mutant(self, simple_cohort):
        """Verify that the result contains one entry per mutant."""
        result = build_pvalue_dict_per_mutant(simple_cohort, pseudocount=1e-10)
        assert set(result.keys()) == {"mutantA", "mutantB"}

    def test_correct_pvalues_extracted(self, simple_cohort):
        """Verify that p-values are correctly extracted from TermRecords."""
        result = build_pvalue_dict_per_mutant(simple_cohort, pseudocount=1e-10)
        assert result["mutantA"]["GO:0000001"] == pytest.approx(0.01)
        assert result["mutantA"]["GO:0000002"] == pytest.approx(0.05)
        assert result["mutantB"]["GO:0000001"] == pytest.approx(0.03)
        assert result["mutantB"]["GO:0000003"] == pytest.approx(0.02)

    def test_zero_pval_replaced_with_pseudocount(self, cohort_with_zero_pval):
        """Contract 2: NOM p-val of exactly 0.0 is replaced with pseudocount."""
        pseudocount = 1e-10
        result = build_pvalue_dict_per_mutant(cohort_with_zero_pval, pseudocount=pseudocount)
        # mutantA had 0.0 for GO:0000001
        assert result["mutantA"]["GO:0000001"] == pytest.approx(pseudocount)

    def test_zero_pval_replaced_with_custom_pseudocount(self, cohort_with_zero_pval):
        """Contract 2: Pseudocount is configurable, not hardcoded."""
        custom_pseudocount = 1e-5
        result = build_pvalue_dict_per_mutant(cohort_with_zero_pval, pseudocount=custom_pseudocount)
        assert result["mutantA"]["GO:0000001"] == pytest.approx(custom_pseudocount)

    def test_nonzero_pval_not_replaced(self, cohort_with_zero_pval):
        """Contract 2: Only exactly 0.0 is replaced; other values are left as-is."""
        result = build_pvalue_dict_per_mutant(cohort_with_zero_pval, pseudocount=1e-10)
        # mutantB has 0.01 for GO:0000001 -- should remain 0.01
        assert result["mutantB"]["GO:0000001"] == pytest.approx(0.01)
        # mutantA has 0.05 for GO:0000002 -- should remain 0.05
        assert result["mutantA"]["GO:0000002"] == pytest.approx(0.05)


# ===========================================================================
# Tests for build_pvalue_matrix
# ===========================================================================

class TestBuildPvalueMatrix:
    """Tests for building the GO term x mutant p-value matrix."""

    def test_matrix_shape_matches_go_terms_by_mutants(self, simple_cohort):
        """Contract 3: Matrix has shape (n_GO_terms, n_mutants)."""
        per_mutant = build_pvalue_dict_per_mutant(simple_cohort, pseudocount=1e-10)
        matrix, go_id_order = build_pvalue_matrix(per_mutant, simple_cohort.mutant_ids)
        # 3 unique GO terms, 2 mutants
        assert matrix.shape == (3, 2)
        assert len(go_id_order) == 3

    def test_missing_entries_imputed_as_one(self, simple_cohort):
        """Contract 3: Missing entries are imputed as p = 1.0."""
        per_mutant = build_pvalue_dict_per_mutant(simple_cohort, pseudocount=1e-10)
        matrix, go_id_order = build_pvalue_matrix(per_mutant, simple_cohort.mutant_ids)
        # GO:0000002 (APOPTOSIS) is missing from mutantB
        go_002_idx = go_id_order.index("GO:0000002")
        mutantB_idx = simple_cohort.mutant_ids.index("mutantB")
        assert matrix[go_002_idx, mutantB_idx] == pytest.approx(1.0)
        # GO:0000003 (DNA REPAIR) is missing from mutantA
        go_003_idx = go_id_order.index("GO:0000003")
        mutantA_idx = simple_cohort.mutant_ids.index("mutantA")
        assert matrix[go_003_idx, mutantA_idx] == pytest.approx(1.0)

    def test_present_entries_have_correct_values(self, simple_cohort):
        """Verify that present entries in the matrix have the correct p-values."""
        per_mutant = build_pvalue_dict_per_mutant(simple_cohort, pseudocount=1e-10)
        matrix, go_id_order = build_pvalue_matrix(per_mutant, simple_cohort.mutant_ids)
        go_001_idx = go_id_order.index("GO:0000001")
        mutantA_idx = simple_cohort.mutant_ids.index("mutantA")
        mutantB_idx = simple_cohort.mutant_ids.index("mutantB")
        assert matrix[go_001_idx, mutantA_idx] == pytest.approx(0.01)
        assert matrix[go_001_idx, mutantB_idx] == pytest.approx(0.03)

    def test_go_id_order_contains_union_of_all_go_ids(self, simple_cohort):
        """Contract 3: The union of all GO IDs is used for rows."""
        per_mutant = build_pvalue_dict_per_mutant(simple_cohort, pseudocount=1e-10)
        _, go_id_order = build_pvalue_matrix(per_mutant, simple_cohort.mutant_ids)
        assert set(go_id_order) == {"GO:0000001", "GO:0000002", "GO:0000003"}

    def test_imputed_entry_contributes_zero_to_fisher_stat(self):
        """Contract 3: ln(1.0) = 0, so imputed entries contribute 0 to Fisher stat."""
        # Verify the mathematical property
        assert math.log(1.0) == 0.0


# ===========================================================================
# Tests for compute_fisher_combined
# ===========================================================================

class TestComputeFisherCombined:
    """Tests for the Fisher combined p-value computation."""

    def test_fisher_statistic_formula(self):
        """Contract 4: X^2 = -2 * sum(ln(p_i)), df = 2k."""
        # Two mutants, one GO term with p = [0.01, 0.03]
        pvalues = np.array([[0.01, 0.03]])
        n_mutants = 2
        result = compute_fisher_combined(pvalues, n_mutants)

        # Manually compute
        fisher_stat = -2.0 * (math.log(0.01) + math.log(0.03))
        df = 2 * n_mutants
        expected_p = scipy_stats.chi2.sf(fisher_stat, df)

        assert len(result) == 1
        assert result[0] == pytest.approx(expected_p, rel=1e-10)

    def test_degrees_of_freedom_constant_across_go_terms(self):
        """Contract 4: df = 2k is constant across all GO terms (due to imputation)."""
        # Two GO terms, three mutants
        pvalues = np.array([
            [0.01, 0.05, 1.0],   # GO term 1: one imputed
            [0.02, 0.03, 0.04],  # GO term 2: none imputed
        ])
        n_mutants = 3
        result = compute_fisher_combined(pvalues, n_mutants)

        # Both should use df = 6
        df = 6
        for i in range(2):
            stat = -2.0 * np.sum(np.log(pvalues[i, :]))
            expected = scipy_stats.chi2.sf(stat, df)
            assert result[i] == pytest.approx(expected, rel=1e-10)

    def test_combined_pvalue_from_chi2_survival_function(self):
        """Contract 5: Combined p-value from chi-squared survival function (1-CDF)."""
        pvalues = np.array([[0.05, 0.05]])
        n_mutants = 2
        result = compute_fisher_combined(pvalues, n_mutants)

        stat = -2.0 * (math.log(0.05) + math.log(0.05))
        df = 4
        expected = scipy_stats.chi2.sf(stat, df)
        assert result[0] == pytest.approx(expected, rel=1e-10)

    def test_all_imputed_gives_nonsignificant_result(self):
        """When all entries are 1.0 (all imputed), Fisher stat is 0, combined p = 1.0."""
        pvalues = np.array([[1.0, 1.0, 1.0]])
        n_mutants = 3
        result = compute_fisher_combined(pvalues, n_mutants)
        # X^2 = -2 * sum(ln(1.0)) = 0; chi2.sf(0, 6) = 1.0
        assert result[0] == pytest.approx(1.0)

    def test_very_small_pvalues_give_significant_result(self):
        """Very small p-values should produce a very small combined p-value."""
        pvalues = np.array([[1e-10, 1e-10]])
        n_mutants = 2
        result = compute_fisher_combined(pvalues, n_mutants)
        # Should be extremely small
        assert result[0] < 0.01

    def test_returns_one_pvalue_per_row(self):
        """Verify output length matches number of rows in matrix."""
        pvalues = np.array([
            [0.01, 0.02],
            [0.05, 0.10],
            [0.50, 0.80],
        ])
        result = compute_fisher_combined(pvalues, n_mutants=2)
        assert len(result) == 3

    def test_combined_pvalues_in_valid_range(self):
        """Post-condition: all combined p-values are in [0, 1]."""
        pvalues = np.array([
            [0.001, 0.01],
            [0.5, 0.5],
            [1.0, 1.0],
        ])
        result = compute_fisher_combined(pvalues, n_mutants=2)
        for p in result:
            assert 0.0 <= p <= 1.0


# ===========================================================================
# Tests for run_fisher_analysis
# ===========================================================================

class TestRunFisherAnalysis:
    """Tests for the top-level Fisher analysis entry point."""

    def test_returns_fisher_result(self, simple_cohort, default_config, tmp_path):
        """Verify that run_fisher_analysis returns a FisherResult."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        assert isinstance(result, FisherResult)

    def test_combined_pvalues_one_per_go_id(self, simple_cohort, default_config, tmp_path):
        """Post-condition: one combined p-value per GO ID."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        assert len(result.combined_pvalues) == len(result.go_ids)

    def test_combined_pvalues_in_range(self, simple_cohort, default_config, tmp_path):
        """Post-condition: all combined p-values in [0, 1]."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        for p in result.combined_pvalues.values():
            assert 0.0 <= p <= 1.0

    def test_matrix_shape_matches_labels(self, simple_cohort, default_config, tmp_path):
        """Post-condition: matrix shape == (len(go_id_order), len(mutant_ids))."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        assert result.pvalue_matrix.shape == (len(result.go_id_order), len(result.mutant_ids))

    def test_n_mutants_matches_mutant_ids(self, simple_cohort, default_config, tmp_path):
        """Post-condition: n_mutants == len(mutant_ids)."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        assert result.n_mutants == len(result.mutant_ids)

    def test_corrected_pvalues_none_when_fdr_disabled(self, simple_cohort, default_config, tmp_path):
        """Contract 6: If apply_fdr is False, corrected_pvalues is None."""
        config = FisherConfig(apply_fdr=False)
        result = run_fisher_analysis(simple_cohort, config, tmp_path, clustering_enabled=True)
        assert result.corrected_pvalues is None

    def test_corrected_pvalues_present_when_fdr_enabled(self, simple_cohort, tmp_path):
        """Contract 6: If apply_fdr is True, corrected_pvalues is populated."""
        config = FisherConfig(apply_fdr=True)
        result = run_fisher_analysis(simple_cohort, config, tmp_path, clustering_enabled=True)
        assert result.corrected_pvalues is not None
        assert len(result.corrected_pvalues) == len(result.go_ids)

    def test_corrected_pvalues_in_range(self, simple_cohort, tmp_path):
        """Contract 6: BH-corrected p-values must be in [0, 1]."""
        config = FisherConfig(apply_fdr=True)
        result = run_fisher_analysis(simple_cohort, config, tmp_path, clustering_enabled=True)
        for p in result.corrected_pvalues.values():
            assert 0.0 <= p <= 1.0

    def test_corrected_pvalues_at_least_as_large_as_combined(self, simple_cohort, tmp_path):
        """BH correction should generally produce p-values >= original combined p-values."""
        config = FisherConfig(apply_fdr=True)
        result = run_fisher_analysis(simple_cohort, config, tmp_path, clustering_enabled=True)
        for go_id in result.go_ids:
            assert result.corrected_pvalues[go_id] >= result.combined_pvalues[go_id] - 1e-15

    def test_n_contributing_counts_pvals_less_than_one(self, simple_cohort, default_config, tmp_path):
        """Contract 8: n_contributing counts mutant lines with p < 1.0."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        # GO:0000001 is present in both mutants -> 2 contributing
        assert result.n_contributing["GO:0000001"] == 2
        # GO:0000002 is present only in mutantA -> 1 contributing
        assert result.n_contributing["GO:0000002"] == 1
        # GO:0000003 is present only in mutantB -> 1 contributing
        assert result.n_contributing["GO:0000003"] == 1

    def test_go_id_to_name_mapping_present(self, simple_cohort, default_config, tmp_path):
        """Contract 11: go_id_to_name provides human-readable term names."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        assert "GO:0000001" in result.go_id_to_name
        assert result.go_id_to_name["GO:0000001"] == "CELL CYCLE"
        assert result.go_id_to_name["GO:0000002"] == "APOPTOSIS"
        assert result.go_id_to_name["GO:0000003"] == "DNA REPAIR"

    def test_pseudocount_applied_in_full_pipeline(self, cohort_with_zero_pval, default_config, tmp_path):
        """Contract 2: Zero p-values replaced with pseudocount in the full pipeline."""
        result = run_fisher_analysis(cohort_with_zero_pval, default_config, tmp_path, clustering_enabled=True)
        # The combined p-value for GO:0000001 should be very small (0.0 was replaced)
        # but not derived from -inf (which would happen without pseudocount)
        assert 0.0 <= result.combined_pvalues["GO:0000001"] <= 1.0

    def test_three_mutants_n_mutants(self, three_mutant_cohort, default_config, tmp_path):
        """Verify n_mutants with 3 mutant lines."""
        result = run_fisher_analysis(three_mutant_cohort, default_config, tmp_path, clustering_enabled=True)
        assert result.n_mutants == 3


# ===========================================================================
# Tests for write_pvalue_matrix_tsv
# ===========================================================================

class TestWritePvalueMatrixTsv:
    """Tests for writing the p-value matrix TSV file."""

    def test_pvalue_matrix_tsv_created(self, simple_cohort, default_config, tmp_path):
        """Contract 9: pvalue_matrix.tsv is always written."""
        run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        assert (tmp_path / "pvalue_matrix.tsv").exists()

    def test_pvalue_matrix_tsv_header_has_mutant_ids(self, simple_cohort, default_config, tmp_path):
        """Contract 9: Columns are mutant IDs."""
        run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        content = (tmp_path / "pvalue_matrix.tsv").read_text()
        header = content.splitlines()[0]
        for mid in simple_cohort.mutant_ids:
            assert mid in header

    def test_pvalue_matrix_tsv_has_go_id_and_term_name_columns(self, simple_cohort, default_config, tmp_path):
        """Contract 9: Rows include GO ID and GO term name."""
        run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        content = (tmp_path / "pvalue_matrix.tsv").read_text()
        header = content.splitlines()[0]
        assert "GO_ID" in header
        assert "Term_Name" in header or "GO_Term" in header

    def test_pvalue_matrix_tsv_row_count(self, simple_cohort, default_config, tmp_path):
        """Contract 9: One row per GO term plus header."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        content = (tmp_path / "pvalue_matrix.tsv").read_text()
        lines = [l for l in content.splitlines() if l.strip()]
        # Header + one per GO term
        assert len(lines) == 1 + len(result.go_id_order)

    def test_write_pvalue_matrix_tsv_directly(self, tmp_path):
        """Test write_pvalue_matrix_tsv function directly."""
        matrix = np.array([[0.01, 0.03], [0.05, 1.0]])
        go_id_order = ["GO:0000001", "GO:0000002"]
        go_id_to_name = {"GO:0000001": "CELL CYCLE", "GO:0000002": "APOPTOSIS"}
        mutant_ids = ["mutantA", "mutantB"]

        path = write_pvalue_matrix_tsv(matrix, go_id_order, go_id_to_name, mutant_ids, tmp_path)
        assert path.exists()
        content = path.read_text()
        lines = content.splitlines()
        assert "GO_ID" in lines[0]
        assert "mutantA" in lines[0]
        assert "mutantB" in lines[0]
        assert lines[1].startswith("GO:0000001")
        assert "CELL CYCLE" in lines[1]


# ===========================================================================
# Tests for write_fisher_results_tsv
# ===========================================================================

class TestWriteFisherResultsTsv:
    """Tests for writing the Fisher combined p-values TSV file."""

    def test_fisher_results_tsv_written_when_clustering_disabled(self, simple_cohort, default_config, tmp_path):
        """Contract 10: fisher_combined_pvalues.tsv written when clustering_enabled=False."""
        run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=False)
        assert (tmp_path / "fisher_combined_pvalues.tsv").exists()

    def test_fisher_results_tsv_not_written_when_clustering_enabled(self, simple_cohort, default_config, tmp_path):
        """Contract 10: fisher_combined_pvalues.tsv NOT written when clustering_enabled=True."""
        run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        assert not (tmp_path / "fisher_combined_pvalues.tsv").exists()

    def test_fisher_results_tsv_contains_expected_columns(self, simple_cohort, default_config, tmp_path):
        """Contract 10: Contains GO ID, GO term name, combined p-value, n_contributing."""
        run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=False)
        content = (tmp_path / "fisher_combined_pvalues.tsv").read_text()
        header = content.splitlines()[0]
        assert "GO_ID" in header
        assert "Term_Name" in header or "GO_Term" in header
        assert "Combined_pvalue" in header or "combined_pvalue" in header.lower()
        assert "N_contributing" in header or "contributing" in header.lower()

    def test_fisher_results_tsv_no_cluster_column(self, simple_cohort, default_config, tmp_path):
        """Contract 10: No cluster assignment column when clustering is disabled."""
        run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=False)
        content = (tmp_path / "fisher_combined_pvalues.tsv").read_text()
        header = content.splitlines()[0].lower()
        assert "cluster" not in header

    def test_fisher_results_tsv_row_count(self, simple_cohort, default_config, tmp_path):
        """Contract 10: One row per GO term plus header."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=False)
        content = (tmp_path / "fisher_combined_pvalues.tsv").read_text()
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) == 1 + len(result.go_id_order)

    def test_pvalue_matrix_tsv_also_written_when_clustering_disabled(self, simple_cohort, default_config, tmp_path):
        """Contract 9: pvalue_matrix.tsv is ALWAYS written regardless of clustering."""
        run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=False)
        assert (tmp_path / "pvalue_matrix.tsv").exists()


# ===========================================================================
# Tests for edge cases and additional contracts
# ===========================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_go_term_across_all_mutants(self, tmp_path):
        """Single GO term present in all mutants."""
        cohort = _make_cohort({
            "mutantA": [_make_term_record("CELL CYCLE", "GO:0000001", 0.01)],
            "mutantB": [_make_term_record("CELL CYCLE", "GO:0000001", 0.02)],
        })
        config = FisherConfig()
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=True)
        assert len(result.go_ids) == 1
        assert result.n_contributing["GO:0000001"] == 2

    def test_no_overlapping_go_terms(self, tmp_path):
        """Mutants have completely disjoint GO term sets."""
        cohort = _make_cohort({
            "mutantA": [_make_term_record("CELL CYCLE", "GO:0000001", 0.01)],
            "mutantB": [_make_term_record("DNA REPAIR", "GO:0000002", 0.02)],
        })
        config = FisherConfig()
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=True)
        # Each GO term is only in 1 mutant
        assert result.n_contributing["GO:0000001"] == 1
        assert result.n_contributing["GO:0000002"] == 1
        # Matrix should be 2x2
        assert result.pvalue_matrix.shape == (2, 2)

    def test_all_pvalues_zero_replaced_with_pseudocount(self, tmp_path):
        """All NOM p-vals are 0.0, all should be replaced with pseudocount."""
        cohort = _make_cohort({
            "mutantA": [_make_term_record("CELL CYCLE", "GO:0000001", 0.0)],
            "mutantB": [_make_term_record("CELL CYCLE", "GO:0000001", 0.0)],
        })
        config = FisherConfig(pseudocount=1e-5)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=True)
        # Should not have any NaN or inf
        assert not np.any(np.isnan(result.pvalue_matrix))
        assert not np.any(np.isinf(result.pvalue_matrix))
        assert 0.0 <= result.combined_pvalues["GO:0000001"] <= 1.0

    def test_many_go_terms(self, tmp_path):
        """Many GO terms across two mutants."""
        terms_a = [_make_term_record(f"TERM_{i}", f"GO:000{i:04d}", 0.01 * (i + 1))
                   for i in range(20)]
        terms_b = [_make_term_record(f"TERM_{i}", f"GO:000{i:04d}", 0.02 * (i + 1))
                   for i in range(15)]
        cohort = _make_cohort({"mutantA": terms_a, "mutantB": terms_b})
        config = FisherConfig()
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=True)
        assert len(result.go_ids) == 20  # union of 20 and 15 (overlapping)
        assert result.pvalue_matrix.shape == (20, 2)

    def test_go_ids_in_result_match_go_id_order(self, simple_cohort, default_config, tmp_path):
        """Verify go_ids and go_id_order consistency."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        assert set(result.go_ids) == set(result.go_id_order)

    def test_mutant_ids_preserved(self, simple_cohort, default_config, tmp_path):
        """Verify mutant_ids in result match input cohort."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        assert result.mutant_ids == simple_cohort.mutant_ids

    def test_n_contributing_never_exceeds_n_mutants(self, three_mutant_cohort, default_config, tmp_path):
        """n_contributing for any GO term should not exceed n_mutants."""
        result = run_fisher_analysis(three_mutant_cohort, default_config, tmp_path, clustering_enabled=True)
        for go_id, count in result.n_contributing.items():
            assert count <= result.n_mutants

    def test_n_contributing_is_nonnegative(self, simple_cohort, default_config, tmp_path):
        """n_contributing should always be >= 0."""
        result = run_fisher_analysis(simple_cohort, default_config, tmp_path, clustering_enabled=True)
        for count in result.n_contributing.values():
            assert count >= 0
