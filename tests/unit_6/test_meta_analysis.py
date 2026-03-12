"""
Tests for Unit 6 -- Meta-Analysis Computation (Fisher's combined probability method).

Synthetic Data Assumptions
==========================
DATA ASSUMPTION: GO IDs follow the standard Gene Ontology format "GO:NNNNNNN"
    (e.g., GO:0008150, GO:0003674). These are realistic identifiers.

DATA ASSUMPTION: Nominal p-values are in [0.0, 1.0], representing statistical
    significance from GSEA preranked analyses. Typical p-values range from 0.0
    (highly significant) to 1.0 (not significant).

DATA ASSUMPTION: A cohort contains at least 2 mutant lines, each with enrichment
    profiles for various GO terms. This matches the blueprint invariant.

DATA ASSUMPTION: Pseudocount is a small positive number (default 1e-10) used to
    replace exact-zero p-values before log transformation, preventing -inf in
    Fisher's statistic computation.

DATA ASSUMPTION: NES (Normalized Enrichment Score) values can be positive or
    negative, representing up- or down-regulation. Typical range is [-3, 3].

DATA ASSUMPTION: FDR values are in [0.0, 1.0], representing false discovery
    rate-adjusted significance.

DATA ASSUMPTION: Gene set sizes are positive integers, typically in range [15, 500]
    for GO term gene sets.
"""

import inspect
import math
from dataclasses import fields as dataclass_fields
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.configuration import FisherConfig
from gsea_tool.meta_analysis import (
    FisherResult,
    build_nes_matrix,
    build_pvalue_dict_per_mutant,
    build_pvalue_matrix,
    compute_fisher_combined,
    run_fisher_analysis,
    write_pvalue_matrix_tsv,
    write_fisher_results_tsv,
)


# ---------------------------------------------------------------------------
# Helper factories for synthetic data
# ---------------------------------------------------------------------------

def _make_term_record(
    term_name: str = "SIGNAL TRANSDUCTION",
    go_id: str = "GO:0007165",
    nes: float = 1.5,
    fdr: float = 0.01,
    nom_pval: float = 0.005,
    size: int = 150,
) -> TermRecord:
    """Create a TermRecord with sensible defaults.

    DATA ASSUMPTION: Default values represent a moderately significant GO term
    from a GSEA preranked analysis.
    """
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
    """Create a MutantProfile with optional records."""
    if records is None:
        records = {}
    return MutantProfile(mutant_id=mutant_id, records=records)


def _make_cohort(
    mutant_terms: dict[str, dict[str, tuple[str, float]]] | None = None,
) -> CohortData:
    """Build a CohortData from a concise specification.

    mutant_terms: {mutant_id: {term_name: (go_id, nom_pval)}}

    DATA ASSUMPTION: Synthetic cohort with 2-4 mutants and 2-5 GO terms,
    representing a small but valid GSEA experiment.
    """
    if mutant_terms is None:
        # Default: 2 mutants, 3 GO terms
        mutant_terms = {
            "mutantA": {
                "SIGNAL TRANSDUCTION": ("GO:0007165", 0.005),
                "CELL CYCLE": ("GO:0007049", 0.02),
                "APOPTOTIC PROCESS": ("GO:0006915", 0.1),
            },
            "mutantB": {
                "SIGNAL TRANSDUCTION": ("GO:0007165", 0.01),
                "CELL CYCLE": ("GO:0007049", 0.0),   # zero p-value to test pseudocount
                "APOPTOTIC PROCESS": ("GO:0006915", 0.5),
            },
        }

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
# SECTION 1: Signature and dataclass structure tests
# ===========================================================================


class TestFisherResultDataclass:
    """Verify FisherResult dataclass has all expected fields with correct types."""

    def test_fisher_result_is_dataclass(self):
        """FisherResult should be a dataclass."""
        field_names = {f.name for f in dataclass_fields(FisherResult)}
        expected = {
            "go_ids",
            "go_id_to_name",
            "combined_pvalues",
            "n_contributing",
            "pvalue_matrix",
            "mutant_ids",
            "go_id_order",
            "n_mutants",
            "corrected_pvalues",
        }
        assert expected == field_names, (
            f"FisherResult fields mismatch. Missing: {expected - field_names}, "
            f"Extra: {field_names - expected}"
        )

    def test_fisher_result_instantiation(self):
        """FisherResult can be instantiated with all required fields."""
        # DATA ASSUMPTION: Minimal valid FisherResult with 2 GO terms, 2 mutants
        result = FisherResult(
            go_ids=["GO:0007165", "GO:0007049"],
            go_id_to_name={"GO:0007165": "SIGNAL TRANSDUCTION", "GO:0007049": "CELL CYCLE"},
            combined_pvalues={"GO:0007165": 0.001, "GO:0007049": 0.05},
            n_contributing={"GO:0007165": 2, "GO:0007049": 1},
            pvalue_matrix=np.array([[0.01, 0.02], [0.03, 1.0]]),
            mutant_ids=["mutantA", "mutantB"],
            go_id_order=["GO:0007165", "GO:0007049"],
            n_mutants=2,
            corrected_pvalues=None,
        )
        assert result.n_mutants == 2
        assert result.corrected_pvalues is None


class TestFunctionSignatures:
    """Verify function signatures match the blueprint."""

    def test_build_pvalue_dict_per_mutant_signature(self):
        sig = inspect.signature(build_pvalue_dict_per_mutant)
        params = list(sig.parameters.keys())
        assert params == ["cohort", "pseudocount"]

    def test_build_pvalue_matrix_signature(self):
        sig = inspect.signature(build_pvalue_matrix)
        params = list(sig.parameters.keys())
        assert params == ["per_mutant_pvals", "mutant_ids"]

    def test_compute_fisher_combined_signature(self):
        sig = inspect.signature(compute_fisher_combined)
        params = list(sig.parameters.keys())
        assert params == ["pvalue_matrix", "n_mutants"]

    def test_run_fisher_analysis_signature(self):
        sig = inspect.signature(run_fisher_analysis)
        params = list(sig.parameters.keys())
        assert params == ["cohort", "config", "output_dir", "clustering_enabled"]

    def test_write_pvalue_matrix_tsv_signature(self):
        sig = inspect.signature(write_pvalue_matrix_tsv)
        params = list(sig.parameters.keys())
        assert params == ["matrix", "nes_matrix", "go_id_order", "go_id_to_name", "mutant_ids", "output_dir"]

    def test_write_fisher_results_tsv_signature(self):
        sig = inspect.signature(write_fisher_results_tsv)
        params = list(sig.parameters.keys())
        assert params == ["fisher_result", "output_dir"]


# ===========================================================================
# SECTION 2: build_pvalue_dict_per_mutant tests
# ===========================================================================


class TestBuildPvalueDictPerMutant:
    """Tests for Contract 1 (per-mutant p-value dicts) and Contract 2 (p=0 replacement)."""

    def test_returns_dict_keyed_by_mutant_id(self):
        """Contract 1: Returns per-mutant {GO_ID: nom_pval} dicts."""
        cohort = _make_cohort()
        pseudocount = 1e-10
        result, _ = build_pvalue_dict_per_mutant(cohort, pseudocount)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(cohort.mutant_ids)

    def test_inner_dicts_keyed_by_go_id(self):
        """Contract 1: Inner dicts keyed by GO ID with nom_pval values."""
        cohort = _make_cohort()
        pseudocount = 1e-10
        result, _ = build_pvalue_dict_per_mutant(cohort, pseudocount)

        for mutant_id, pval_dict in result.items():
            assert isinstance(pval_dict, dict)
            profile = cohort.profiles[mutant_id]
            # Each GO ID from the mutant's records should be present
            expected_go_ids = {rec.go_id for rec in profile.records.values()}
            assert set(pval_dict.keys()) == expected_go_ids

    def test_pvalues_match_nom_pval_from_records(self):
        """Contract 1: Values are the nominal p-values from the records."""
        # DATA ASSUMPTION: mutant with known non-zero p-values
        cohort = _make_cohort({
            "mutantA": {
                "TERM1": ("GO:0000001", 0.03),
                "TERM2": ("GO:0000002", 0.15),
            },
            "mutantB": {
                "TERM1": ("GO:0000001", 0.07),
            },
        })
        pseudocount = 1e-10
        result, _ = build_pvalue_dict_per_mutant(cohort, pseudocount)

        assert result["mutantA"]["GO:0000001"] == pytest.approx(0.03)
        assert result["mutantA"]["GO:0000002"] == pytest.approx(0.15)
        assert result["mutantB"]["GO:0000001"] == pytest.approx(0.07)

    def test_zero_pvalue_replaced_with_pseudocount(self):
        """Contract 2: p=0.0 is replaced with pseudocount."""
        cohort = _make_cohort({
            "mutantA": {
                "TERM1": ("GO:0000001", 0.0),
            },
            "mutantB": {
                "TERM1": ("GO:0000001", 0.05),
            },
        })
        pseudocount = 1e-10
        result, _ = build_pvalue_dict_per_mutant(cohort, pseudocount)

        assert result["mutantA"]["GO:0000001"] == pytest.approx(pseudocount)

    def test_zero_pvalue_replaced_with_custom_pseudocount(self):
        """Contract 2: p=0.0 replacement uses the provided pseudocount value."""
        cohort = _make_cohort({
            "mutantA": {
                "TERM1": ("GO:0000001", 0.0),
            },
            "mutantB": {
                "TERM1": ("GO:0000001", 0.05),
            },
        })
        custom_pseudocount = 1e-5
        result, _ = build_pvalue_dict_per_mutant(cohort, custom_pseudocount)

        assert result["mutantA"]["GO:0000001"] == pytest.approx(custom_pseudocount)

    def test_nonzero_pvalues_unchanged(self):
        """Contract 2: Non-zero p-values are NOT replaced."""
        cohort = _make_cohort({
            "mutantA": {
                "TERM1": ("GO:0000001", 0.001),
            },
            "mutantB": {
                "TERM1": ("GO:0000001", 0.05),
            },
        })
        pseudocount = 1e-10
        result, _ = build_pvalue_dict_per_mutant(cohort, pseudocount)

        assert result["mutantA"]["GO:0000001"] == pytest.approx(0.001)
        assert result["mutantB"]["GO:0000001"] == pytest.approx(0.05)

    def test_multiple_zero_pvalues_across_mutants(self):
        """Contract 2: Multiple zero p-values across different mutants replaced."""
        cohort = _make_cohort({
            "mutantA": {
                "TERM1": ("GO:0000001", 0.0),
                "TERM2": ("GO:0000002", 0.0),
            },
            "mutantB": {
                "TERM1": ("GO:0000001", 0.0),
                "TERM2": ("GO:0000002", 0.5),
            },
        })
        pseudocount = 1e-10
        result, _ = build_pvalue_dict_per_mutant(cohort, pseudocount)

        assert result["mutantA"]["GO:0000001"] == pytest.approx(pseudocount)
        assert result["mutantA"]["GO:0000002"] == pytest.approx(pseudocount)
        assert result["mutantB"]["GO:0000001"] == pytest.approx(pseudocount)
        assert result["mutantB"]["GO:0000002"] == pytest.approx(0.5)


# ===========================================================================
# SECTION 3: build_pvalue_matrix tests
# ===========================================================================


class TestBuildPvalueMatrix:
    """Tests for Contract 3 (matrix shape and missing imputation)."""

    def test_matrix_shape(self):
        """Contract 3: Matrix shape is (n_GO_terms, n_mutants)."""
        # DATA ASSUMPTION: 3 GO terms across 2 mutants
        per_mutant = {
            "mutantA": {"GO:0000001": 0.01, "GO:0000002": 0.05, "GO:0000003": 0.1},
            "mutantB": {"GO:0000001": 0.02, "GO:0000002": 0.03, "GO:0000003": 0.2},
        }
        mutant_ids = ["mutantA", "mutantB"]
        matrix, go_id_order = build_pvalue_matrix(per_mutant, mutant_ids)

        assert matrix.shape == (3, 2)
        assert len(go_id_order) == 3

    def test_returns_ndarray_and_go_id_list(self):
        """Return type is tuple of (np.ndarray, list[str])."""
        per_mutant = {
            "mutantA": {"GO:0000001": 0.01},
            "mutantB": {"GO:0000001": 0.02},
        }
        mutant_ids = ["mutantA", "mutantB"]
        matrix, go_id_order = build_pvalue_matrix(per_mutant, mutant_ids)

        assert isinstance(matrix, np.ndarray)
        assert isinstance(go_id_order, list)
        assert all(isinstance(g, str) for g in go_id_order)

    def test_missing_term_imputed_as_one(self):
        """Contract 3: Missing GO terms imputed as 1.0."""
        # DATA ASSUMPTION: mutantA has GO:0000001 and GO:0000002,
        # mutantB only has GO:0000001. GO:0000002 should be 1.0 for mutantB.
        per_mutant = {
            "mutantA": {"GO:0000001": 0.01, "GO:0000002": 0.05},
            "mutantB": {"GO:0000001": 0.02},
        }
        mutant_ids = ["mutantA", "mutantB"]
        matrix, go_id_order = build_pvalue_matrix(per_mutant, mutant_ids)

        # Find the index of GO:0000002 in go_id_order
        idx_go2 = go_id_order.index("GO:0000002")
        # mutantB is at column index 1
        mutantB_col = mutant_ids.index("mutantB")
        assert matrix[idx_go2, mutantB_col] == pytest.approx(1.0)

    def test_present_values_correct(self):
        """Contract 3: Present p-values are correctly placed in matrix."""
        per_mutant = {
            "mutantA": {"GO:0000001": 0.01, "GO:0000002": 0.05},
            "mutantB": {"GO:0000001": 0.02, "GO:0000002": 0.03},
        }
        mutant_ids = ["mutantA", "mutantB"]
        matrix, go_id_order = build_pvalue_matrix(per_mutant, mutant_ids)

        for i, go_id in enumerate(go_id_order):
            for j, mid in enumerate(mutant_ids):
                expected = per_mutant[mid].get(go_id, 1.0)
                assert matrix[i, j] == pytest.approx(expected), (
                    f"Mismatch at ({go_id}, {mid}): expected {expected}, got {matrix[i, j]}"
                )

    def test_go_id_order_contains_all_go_ids(self):
        """go_id_order contains every unique GO ID from per_mutant_pvals."""
        per_mutant = {
            "mutantA": {"GO:0000001": 0.01, "GO:0000002": 0.05},
            "mutantB": {"GO:0000001": 0.02, "GO:0000003": 0.1},
        }
        mutant_ids = ["mutantA", "mutantB"]
        matrix, go_id_order = build_pvalue_matrix(per_mutant, mutant_ids)

        all_go_ids = {"GO:0000001", "GO:0000002", "GO:0000003"}
        assert set(go_id_order) == all_go_ids

    def test_matrix_shape_with_disjoint_terms(self):
        """Contract 3: When mutants have completely disjoint GO terms."""
        # DATA ASSUMPTION: Each mutant has unique GO terms, no overlap
        per_mutant = {
            "mutantA": {"GO:0000001": 0.01},
            "mutantB": {"GO:0000002": 0.02},
        }
        mutant_ids = ["mutantA", "mutantB"]
        matrix, go_id_order = build_pvalue_matrix(per_mutant, mutant_ids)

        assert matrix.shape == (2, 2)
        # For each GO term, the mutant that doesn't have it should get 1.0
        for i, go_id in enumerate(go_id_order):
            for j, mid in enumerate(mutant_ids):
                expected = per_mutant[mid].get(go_id, 1.0)
                assert matrix[i, j] == pytest.approx(expected)


# ===========================================================================
# SECTION 4: compute_fisher_combined tests
# ===========================================================================


class TestComputeFisherCombined:
    """Tests for Contracts 4 and 5 (Fisher statistic and chi2 p-value)."""

    def test_returns_ndarray(self):
        """compute_fisher_combined returns a numpy array."""
        # DATA ASSUMPTION: 2 GO terms x 3 mutants, moderate p-values
        matrix = np.array([
            [0.01, 0.05, 0.1],
            [0.5, 0.5, 0.5],
        ])
        result = compute_fisher_combined(matrix, n_mutants=3)
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_rows(self):
        """One combined p-value per GO term (row)."""
        matrix = np.array([
            [0.01, 0.05, 0.1],
            [0.5, 0.5, 0.5],
            [0.001, 0.001, 0.001],
        ])
        result = compute_fisher_combined(matrix, n_mutants=3)
        assert len(result) == 3

    def test_all_pvalues_in_0_1(self):
        """Post-condition: All combined p-values are in [0, 1]."""
        # DATA ASSUMPTION: Mix of significant and non-significant p-values
        matrix = np.array([
            [0.001, 0.002, 0.01],
            [0.5, 0.6, 0.9],
            [0.0001, 0.0001, 0.0001],
        ])
        result = compute_fisher_combined(matrix, n_mutants=3)
        assert all(0.0 <= p <= 1.0 for p in result)

    def test_fisher_statistic_computation(self):
        """Contract 4: X^2 = -2 * sum(ln(p_i)), df = 2k.
        Contract 5: Combined p-value from chi2 survival function.

        Manually verify Fisher's method for a known case.
        """
        from scipy.stats import chi2

        # DATA ASSUMPTION: 1 GO term with known p-values across 2 mutants
        p1, p2 = 0.05, 0.01
        matrix = np.array([[p1, p2]])
        n_mutants = 2

        # Expected Fisher statistic
        expected_chi2 = -2.0 * (math.log(p1) + math.log(p2))
        # df = 2k = 4
        expected_pvalue = chi2.sf(expected_chi2, df=2 * n_mutants)

        result = compute_fisher_combined(matrix, n_mutants=n_mutants)
        assert result[0] == pytest.approx(expected_pvalue, rel=1e-6)

    def test_uniform_pvalues_yield_nonsignificant_combined(self):
        """When all p-values are near 1.0, combined p-value should be large (non-significant)."""
        # DATA ASSUMPTION: All mutants show p=0.9, representing no enrichment
        matrix = np.array([[0.9, 0.9, 0.9]])
        result = compute_fisher_combined(matrix, n_mutants=3)
        # Combined p-value should be large (close to 1)
        assert result[0] > 0.5

    def test_highly_significant_pvalues_yield_small_combined(self):
        """When all p-values are very small, combined p-value should be very small."""
        # DATA ASSUMPTION: All mutants show p=0.001, representing strong enrichment
        matrix = np.array([[0.001, 0.001, 0.001]])
        result = compute_fisher_combined(matrix, n_mutants=3)
        assert result[0] < 0.01

    def test_single_row_matrix(self):
        """Fisher's method works with a single GO term."""
        matrix = np.array([[0.05, 0.05]])
        result = compute_fisher_combined(matrix, n_mutants=2)
        assert len(result) == 1
        assert 0.0 <= result[0] <= 1.0

    def test_two_mutants_known_result(self):
        """Verify exact Fisher combined p-value for two mutants with known values."""
        from scipy.stats import chi2

        p1, p2 = 0.1, 0.2
        matrix = np.array([[p1, p2]])
        n_mutants = 2

        chi2_stat = -2.0 * (np.log(p1) + np.log(p2))
        expected = chi2.sf(chi2_stat, df=4)

        result = compute_fisher_combined(matrix, n_mutants=n_mutants)
        assert result[0] == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# SECTION 5: FDR correction tests (Contract 6)
# ===========================================================================


class TestFDRCorrection:
    """Tests for Contract 6: BH correction applied when apply_fdr=True."""

    def test_fdr_applied_when_config_true(self, tmp_path):
        """Contract 6: When apply_fdr=True, corrected_pvalues is populated."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=True)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert result.corrected_pvalues is not None
        assert isinstance(result.corrected_pvalues, dict)
        assert len(result.corrected_pvalues) == len(result.combined_pvalues)

    def test_fdr_not_applied_when_config_false(self, tmp_path):
        """Contract 6: When apply_fdr=False, corrected_pvalues is None."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert result.corrected_pvalues is None

    def test_corrected_pvalues_keys_match_combined(self, tmp_path):
        """Contract 6: corrected_pvalues has same keys as combined_pvalues."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=True)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert set(result.corrected_pvalues.keys()) == set(result.combined_pvalues.keys())

    def test_corrected_pvalues_in_0_1(self, tmp_path):
        """Post-condition: corrected p-values are in [0, 1]."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=True)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        for go_id, pval in result.corrected_pvalues.items():
            assert 0.0 <= pval <= 1.0, f"corrected_pvalue for {go_id} is {pval}"

    def test_corrected_pvalues_geq_combined(self, tmp_path):
        """BH-corrected p-values should be >= the corresponding combined p-values."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=True)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        for go_id in result.combined_pvalues:
            assert result.corrected_pvalues[go_id] >= result.combined_pvalues[go_id] - 1e-12, (
                f"corrected ({result.corrected_pvalues[go_id]}) < combined ({result.combined_pvalues[go_id]}) "
                f"for {go_id}"
            )


# ===========================================================================
# SECTION 6: n_contributing tests (Contract 8)
# ===========================================================================


class TestNContributing:
    """Tests for Contract 8: n_contributing counts mutants with p < 1.0."""

    def test_all_mutants_contribute(self, tmp_path):
        """When all mutants have the GO term, all contribute."""
        cohort = _make_cohort({
            "mutantA": {"TERM1": ("GO:0000001", 0.01)},
            "mutantB": {"TERM1": ("GO:0000001", 0.05)},
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert result.n_contributing["GO:0000001"] == 2

    def test_missing_mutant_does_not_contribute(self, tmp_path):
        """When a mutant is missing a GO term (imputed as 1.0), it doesn't contribute."""
        # DATA ASSUMPTION: mutantB lacks GO:0000002, so imputed p=1.0,
        # meaning only mutantA contributes
        cohort = _make_cohort({
            "mutantA": {
                "TERM1": ("GO:0000001", 0.01),
                "TERM2": ("GO:0000002", 0.05),
            },
            "mutantB": {
                "TERM1": ("GO:0000001", 0.05),
            },
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert result.n_contributing["GO:0000002"] == 1

    def test_n_contributing_type(self, tmp_path):
        """n_contributing values should be integers."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        for go_id, count in result.n_contributing.items():
            assert isinstance(count, int), f"n_contributing[{go_id}] is {type(count)}"


# ===========================================================================
# SECTION 7: run_fisher_analysis integration tests
# ===========================================================================


class TestRunFisherAnalysis:
    """Tests for the top-level run_fisher_analysis function."""

    def test_returns_fisher_result(self, tmp_path):
        """run_fisher_analysis returns a FisherResult."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert isinstance(result, FisherResult)

    def test_combined_pvalues_one_per_go_id(self, tmp_path):
        """Post-condition: one combined p-value per GO ID."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert len(result.combined_pvalues) == len(result.go_ids)
        assert set(result.combined_pvalues.keys()) == set(result.go_ids)

    def test_combined_pvalues_in_0_1(self, tmp_path):
        """Post-condition: all combined p-values in [0, 1]."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        for go_id, pval in result.combined_pvalues.items():
            assert 0.0 <= pval <= 1.0, f"combined_pvalue for {go_id} is {pval}"

    def test_matrix_shape_matches(self, tmp_path):
        """Post-condition: matrix shape matches (n_go_terms, n_mutants)."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert result.pvalue_matrix.shape == (len(result.go_id_order), len(result.mutant_ids))

    def test_n_mutants_matches(self, tmp_path):
        """Post-condition: n_mutants matches actual mutant count."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert result.n_mutants == len(cohort.mutant_ids)

    def test_go_id_to_name_mapping(self, tmp_path):
        """Contract 11: go_id_to_name mapping for display."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert isinstance(result.go_id_to_name, dict)
        # Every GO ID in go_ids should have a name mapping
        for go_id in result.go_ids:
            assert go_id in result.go_id_to_name
            assert isinstance(result.go_id_to_name[go_id], str)

    def test_mutant_ids_preserved(self, tmp_path):
        """Result mutant_ids should match the cohort mutant_ids."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert set(result.mutant_ids) == set(cohort.mutant_ids)

    def test_go_id_order_matches_go_ids(self, tmp_path):
        """go_id_order and go_ids should represent the same set."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert set(result.go_id_order) == set(result.go_ids)

    def test_pvalue_matrix_tsv_written(self, tmp_path):
        """Contract 9: pvalue_matrix.tsv is always written."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        tsv_path = tmp_path / "pvalue_matrix.tsv"
        assert tsv_path.exists(), "pvalue_matrix.tsv should always be written"

    def test_fisher_results_tsv_written_when_clustering_disabled(self, tmp_path):
        """Contract 10: fisher_combined_pvalues.tsv written when clustering_enabled=False."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        tsv_path = tmp_path / "fisher_combined_pvalues.tsv"
        assert tsv_path.exists(), (
            "fisher_combined_pvalues.tsv should be written when clustering disabled"
        )

    def test_fisher_results_tsv_not_written_when_clustering_enabled(self, tmp_path):
        """Contract 10: fisher_combined_pvalues.tsv NOT written when clustering_enabled=True."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=True)

        tsv_path = tmp_path / "fisher_combined_pvalues.tsv"
        assert not tsv_path.exists(), (
            "fisher_combined_pvalues.tsv should NOT be written when clustering enabled"
        )

    def test_pvalue_matrix_tsv_written_even_when_clustering_enabled(self, tmp_path):
        """Contract 9: pvalue_matrix.tsv is always written, even with clustering enabled."""
        cohort = _make_cohort()
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=True)

        tsv_path = tmp_path / "pvalue_matrix.tsv"
        assert tsv_path.exists(), "pvalue_matrix.tsv should always be written"

    def test_with_three_mutants(self, tmp_path):
        """Integration: Handles more than 2 mutants correctly."""
        # DATA ASSUMPTION: 3 mutants, each with 2 GO terms
        cohort = _make_cohort({
            "mutantA": {
                "TERM1": ("GO:0000001", 0.01),
                "TERM2": ("GO:0000002", 0.1),
            },
            "mutantB": {
                "TERM1": ("GO:0000001", 0.05),
                "TERM2": ("GO:0000002", 0.2),
            },
            "mutantC": {
                "TERM1": ("GO:0000001", 0.001),
                "TERM2": ("GO:0000002", 0.5),
            },
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        assert result.n_mutants == 3
        assert result.pvalue_matrix.shape[1] == 3

    def test_with_zero_pvalues(self, tmp_path):
        """Integration: Zero p-values are handled (replaced with pseudocount)."""
        cohort = _make_cohort({
            "mutantA": {
                "TERM1": ("GO:0000001", 0.0),
            },
            "mutantB": {
                "TERM1": ("GO:0000001", 0.0),
            },
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        # Should not produce NaN or inf
        assert not np.isnan(result.pvalue_matrix).any()
        for pval in result.combined_pvalues.values():
            assert not math.isnan(pval)
            assert not math.isinf(pval)
            assert 0.0 <= pval <= 1.0


# ===========================================================================
# SECTION 8: write_pvalue_matrix_tsv tests
# ===========================================================================


class TestWritePvalueMatrixTsv:
    """Tests for write_pvalue_matrix_tsv file output."""

    def test_returns_path(self, tmp_path):
        """write_pvalue_matrix_tsv returns a Path."""
        matrix = np.array([[0.01, 0.02], [0.03, 0.04]])
        nes_matrix = np.full(matrix.shape, np.nan)
        go_id_order = ["GO:0000001", "GO:0000002"]
        go_id_to_name = {"GO:0000001": "TERM1", "GO:0000002": "TERM2"}
        mutant_ids = ["mutantA", "mutantB"]

        result = write_pvalue_matrix_tsv(matrix, nes_matrix, go_id_order, go_id_to_name, mutant_ids, tmp_path)
        assert isinstance(result, Path)

    def test_file_created(self, tmp_path):
        """The TSV file is created on disk."""
        matrix = np.array([[0.01, 0.02], [0.03, 0.04]])
        nes_matrix = np.full(matrix.shape, np.nan)
        go_id_order = ["GO:0000001", "GO:0000002"]
        go_id_to_name = {"GO:0000001": "TERM1", "GO:0000002": "TERM2"}
        mutant_ids = ["mutantA", "mutantB"]

        path = write_pvalue_matrix_tsv(matrix, nes_matrix, go_id_order, go_id_to_name, mutant_ids, tmp_path)
        assert path.exists()

    def test_file_is_tsv(self, tmp_path):
        """The output file contains tab-separated data."""
        matrix = np.array([[0.01, 0.02], [0.03, 0.04]])
        nes_matrix = np.full(matrix.shape, np.nan)
        go_id_order = ["GO:0000001", "GO:0000002"]
        go_id_to_name = {"GO:0000001": "TERM1", "GO:0000002": "TERM2"}
        mutant_ids = ["mutantA", "mutantB"]

        path = write_pvalue_matrix_tsv(matrix, nes_matrix, go_id_order, go_id_to_name, mutant_ids, tmp_path)
        content = path.read_text()
        lines = content.strip().split("\n")
        # Should have a header + n_go_terms data rows
        assert len(lines) >= 3  # header + 2 data rows
        # Lines should contain tabs
        for line in lines:
            assert "\t" in line

    def test_file_has_correct_row_count(self, tmp_path):
        """File has header + one row per GO term."""
        # DATA ASSUMPTION: 3 GO terms x 2 mutants
        matrix = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
        nes_matrix = np.full(matrix.shape, np.nan)
        go_id_order = ["GO:0000001", "GO:0000002", "GO:0000003"]
        go_id_to_name = {
            "GO:0000001": "TERM1",
            "GO:0000002": "TERM2",
            "GO:0000003": "TERM3",
        }
        mutant_ids = ["mutantA", "mutantB"]

        path = write_pvalue_matrix_tsv(matrix, nes_matrix, go_id_order, go_id_to_name, mutant_ids, tmp_path)
        content = path.read_text()
        lines = [l for l in content.strip().split("\n") if l.strip()]
        # At minimum: header + 3 data rows
        assert len(lines) >= 4


# ===========================================================================
# SECTION 9: write_fisher_results_tsv tests
# ===========================================================================


class TestWriteFisherResultsTsv:
    """Tests for write_fisher_results_tsv output."""

    def _make_fisher_result(self):
        """Create a minimal FisherResult for testing.

        DATA ASSUMPTION: 2 GO terms, 2 mutants, typical combined p-values.
        """
        return FisherResult(
            go_ids=["GO:0000001", "GO:0000002"],
            go_id_to_name={"GO:0000001": "TERM1", "GO:0000002": "TERM2"},
            combined_pvalues={"GO:0000001": 0.001, "GO:0000002": 0.05},
            n_contributing={"GO:0000001": 2, "GO:0000002": 1},
            pvalue_matrix=np.array([[0.01, 0.02], [0.03, 1.0]]),
            mutant_ids=["mutantA", "mutantB"],
            go_id_order=["GO:0000001", "GO:0000002"],
            n_mutants=2,
            corrected_pvalues=None,
        )

    def test_returns_path(self, tmp_path):
        """write_fisher_results_tsv returns a Path."""
        fr = self._make_fisher_result()
        result = write_fisher_results_tsv(fr, tmp_path)
        assert isinstance(result, Path)

    def test_file_created(self, tmp_path):
        """The TSV file is created on disk."""
        fr = self._make_fisher_result()
        path = write_fisher_results_tsv(fr, tmp_path)
        assert path.exists()

    def test_file_is_tsv(self, tmp_path):
        """The output file contains tab-separated data."""
        fr = self._make_fisher_result()
        path = write_fisher_results_tsv(fr, tmp_path)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) >= 3  # header + 2 data rows
        for line in lines:
            assert "\t" in line

    def test_file_has_correct_row_count(self, tmp_path):
        """File has header + one row per GO ID."""
        fr = self._make_fisher_result()
        path = write_fisher_results_tsv(fr, tmp_path)
        content = path.read_text()
        lines = [l for l in content.strip().split("\n") if l.strip()]
        assert len(lines) >= 3  # header + 2 GO terms


# ===========================================================================
# SECTION 10: Error condition tests
# ===========================================================================


class TestErrorConditions:
    """Tests for error conditions specified in the blueprint."""

    def test_write_pvalue_matrix_tsv_oserror(self):
        """Error condition: OSError on write failure for pvalue_matrix.tsv."""
        matrix = np.array([[0.01, 0.02]])
        nes_matrix = np.full(matrix.shape, np.nan)
        go_id_order = ["GO:0000001"]
        go_id_to_name = {"GO:0000001": "TERM1"}
        mutant_ids = ["mutantA", "mutantB"]

        # Use a non-existent directory that can't be written to
        bad_dir = Path("/nonexistent/path/that/does/not/exist")
        with pytest.raises(OSError):
            write_pvalue_matrix_tsv(matrix, nes_matrix, go_id_order, go_id_to_name, mutant_ids, bad_dir)

    def test_write_fisher_results_tsv_oserror(self):
        """Error condition: OSError on write failure for fisher_results.tsv."""
        fr = FisherResult(
            go_ids=["GO:0000001"],
            go_id_to_name={"GO:0000001": "TERM1"},
            combined_pvalues={"GO:0000001": 0.01},
            n_contributing={"GO:0000001": 2},
            pvalue_matrix=np.array([[0.01, 0.02]]),
            mutant_ids=["mutantA", "mutantB"],
            go_id_order=["GO:0000001"],
            n_mutants=2,
            corrected_pvalues=None,
        )

        bad_dir = Path("/nonexistent/path/that/does/not/exist")
        with pytest.raises(OSError):
            write_fisher_results_tsv(fr, bad_dir)


# ===========================================================================
# SECTION 11: End-to-end consistency tests
# ===========================================================================


class TestEndToEndConsistency:
    """End-to-end tests verifying multiple contracts together."""

    def test_full_pipeline_consistent(self, tmp_path):
        """Verify the full pipeline produces consistent results."""
        # DATA ASSUMPTION: 3 mutants, 4 GO terms with varied p-values
        cohort = _make_cohort({
            "mutantA": {
                "SIGNAL TRANSDUCTION": ("GO:0007165", 0.005),
                "CELL CYCLE": ("GO:0007049", 0.02),
                "APOPTOTIC PROCESS": ("GO:0006915", 0.1),
                "METABOLIC PROCESS": ("GO:0008152", 0.5),
            },
            "mutantB": {
                "SIGNAL TRANSDUCTION": ("GO:0007165", 0.01),
                "CELL CYCLE": ("GO:0007049", 0.0),
                "APOPTOTIC PROCESS": ("GO:0006915", 0.5),
            },
            "mutantC": {
                "SIGNAL TRANSDUCTION": ("GO:0007165", 0.001),
                "METABOLIC PROCESS": ("GO:0008152", 0.3),
            },
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=True)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        # Post-conditions
        assert len(result.combined_pvalues) == len(result.go_ids)
        assert all(0.0 <= p <= 1.0 for p in result.combined_pvalues.values())
        n_go_terms = len(result.go_id_order)
        assert result.pvalue_matrix.shape == (n_go_terms, 3)
        assert result.n_mutants == 3
        assert result.corrected_pvalues is not None
        assert len(result.corrected_pvalues) == len(result.combined_pvalues)

        # Files exist
        assert (tmp_path / "pvalue_matrix.tsv").exists()
        assert (tmp_path / "fisher_combined_pvalues.tsv").exists()

    def test_step_by_step_matches_run_fisher(self, tmp_path):
        """Verify step-by-step computation is consistent with run_fisher_analysis."""
        cohort = _make_cohort({
            "mutantA": {
                "TERM1": ("GO:0000001", 0.01),
                "TERM2": ("GO:0000002", 0.1),
            },
            "mutantB": {
                "TERM1": ("GO:0000001", 0.05),
                "TERM2": ("GO:0000002", 0.2),
            },
        })
        pseudocount = 1e-10
        config = FisherConfig(pseudocount=pseudocount, apply_fdr=False)

        # Step-by-step
        per_mutant, _ = build_pvalue_dict_per_mutant(cohort, pseudocount)
        matrix, go_id_order = build_pvalue_matrix(per_mutant, cohort.mutant_ids)
        combined = compute_fisher_combined(matrix, n_mutants=len(cohort.mutant_ids))

        # Full pipeline
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        # The combined p-values from step-by-step should match run_fisher_analysis
        for i, go_id in enumerate(go_id_order):
            if go_id in result.combined_pvalues:
                # Find the corresponding index in result
                result_idx = result.go_id_order.index(go_id)
                # The combined p-value from step-by-step should be close
                assert combined[i] == pytest.approx(
                    result.combined_pvalues[go_id], rel=1e-6
                ), f"Mismatch for {go_id}"

    def test_n_contributing_consistent_with_matrix(self, tmp_path):
        """n_contributing should count columns with p < 1.0 per row in the matrix."""
        # DATA ASSUMPTION: 2 mutants, mutantB missing TERM2
        cohort = _make_cohort({
            "mutantA": {
                "TERM1": ("GO:0000001", 0.01),
                "TERM2": ("GO:0000002", 0.05),
            },
            "mutantB": {
                "TERM1": ("GO:0000001", 0.03),
                # TERM2 missing -> imputed as 1.0
            },
        })
        config = FisherConfig(pseudocount=1e-10, apply_fdr=False)
        result = run_fisher_analysis(cohort, config, tmp_path, clustering_enabled=False)

        # Verify n_contributing matches the matrix
        for i, go_id in enumerate(result.go_id_order):
            row = result.pvalue_matrix[i, :]
            expected_count = int(np.sum(row < 1.0))
            assert result.n_contributing[go_id] == expected_count, (
                f"n_contributing mismatch for {go_id}: expected {expected_count}, "
                f"got {result.n_contributing[go_id]}"
            )
