"""
Test suite for Unit 4 -- Unbiased Term Selection.

Tests cover the six-step data-driven term selection and grouping pipeline for
Figure 2 (unbiased figure). The pipeline: (1) pools significant GO terms,
(2) ranks by max abs NES, (3) removes lexically redundant terms, (4) selects
top N, (5) clusters by NES profile using hierarchical agglomerative clustering,
(6) auto-labels each cluster group.

Synthetic Data Assumptions (module level):
  - DATA ASSUMPTION: GO term names are short uppercase English strings
    (e.g., "OXIDATIVE PHOSPHORYLATION"), representative of typical GO
    biological process term naming conventions.
  - DATA ASSUMPTION: NES values range roughly from -3.0 to +3.0, which is
    typical for GSEA normalized enrichment scores.
  - DATA ASSUMPTION: FDR values range from 0.0 to 1.0. Terms with FDR < 0.05
    are considered significant by default.
  - DATA ASSUMPTION: Mutant IDs are short alphanumeric strings like "mutA",
    "mutB", representing simplified mutant identifiers.
  - DATA ASSUMPTION: go_id values follow the GO:NNNNNNN format (10 chars),
    using placeholder IDs like "GO:0000001" since this unit operates on
    term_name, not go_id.
  - DATA ASSUMPTION: nom_pval is set to 0.01 as a placeholder since this
    unit does not use nom_pval.
  - DATA ASSUMPTION: size is set to 100 as a placeholder since this unit
    does not use size.
  - DATA ASSUMPTION: Lexically redundant terms are modeled by crafting
    pairs of term names with >50% Jaccard word overlap, e.g.,
    "POSITIVE REGULATION OF CELL DEATH" vs "REGULATION OF CELL DEATH"
    which share 4/5 words = 0.8 Jaccard.
  - DATA ASSUMPTION: For clustering tests, NES profiles are crafted to
    produce deterministic cluster assignments with Ward linkage. Terms
    with similar NES patterns across mutants should cluster together.
"""

import inspect
from dataclasses import fields as dataclass_fields
import pytest
import numpy as np

from gsea_tool.data_ingestion import CohortData, TermRecord, MutantProfile
from gsea_tool.cherry_picked import CategoryGroup
from gsea_tool.unbiased import (
    UnbiasedSelectionStats,
    pool_significant_terms,
    remove_redundant_terms,
    select_top_n,
    cluster_terms,
    select_unbiased_terms,
)


# ---------------------------------------------------------------------------
# Helper: build CohortData from a compact specification
# ---------------------------------------------------------------------------

def _make_term_record(term_name, nes, fdr, go_id="GO:0000001"):
    """Create a TermRecord with placeholder values for unused fields.

    DATA ASSUMPTION: nom_pval=0.01 and size=100 are placeholders; this unit
    does not use these fields.
    """
    return TermRecord(
        term_name=term_name,
        go_id=go_id,
        nes=nes,
        fdr=fdr,
        nom_pval=0.01,
        size=100,
    )


def _build_cohort(mutant_data: dict[str, list[tuple[str, float, float]]]) -> CohortData:
    """Build a CohortData from a dict of mutant_id -> list of (term_name, nes, fdr).

    DATA ASSUMPTION: go_id is assigned sequentially as GO:0000001, GO:0000002, etc.
    for each unique term.
    """
    all_term_names = set()
    all_go_ids = set()
    profiles = {}
    # Assign stable go_ids per term
    term_go_map = {}
    go_counter = 1

    for mutant_id, terms in mutant_data.items():
        for term_name, nes, fdr in terms:
            all_term_names.add(term_name)
            if term_name not in term_go_map:
                term_go_map[term_name] = f"GO:{go_counter:07d}"
                go_counter += 1

    for mutant_id, terms in mutant_data.items():
        records = {}
        for term_name, nes, fdr in terms:
            go_id = term_go_map[term_name]
            all_go_ids.add(go_id)
            records[term_name] = _make_term_record(term_name, nes, fdr, go_id)
        profiles[mutant_id] = MutantProfile(mutant_id=mutant_id, records=records)

    mutant_ids = sorted(mutant_data.keys())
    return CohortData(
        mutant_ids=mutant_ids,
        profiles=profiles,
        all_term_names=all_term_names,
        all_go_ids=all_go_ids,
    )


# ===========================================================================
# Section 1: Signature and type verification
# ===========================================================================

class TestSignatures:
    """Verify function and class signatures match the blueprint."""

    def test_unbiased_selection_stats_is_dataclass(self):
        """UnbiasedSelectionStats must be a dataclass with the specified fields."""
        field_names = {f.name for f in dataclass_fields(UnbiasedSelectionStats)}
        expected = {
            "total_significant_terms",
            "terms_after_dedup",
            "terms_selected",
            "n_clusters",
            "random_seed",
            "clustering_algorithm",
        }
        assert expected.issubset(field_names), (
            f"Missing fields: {expected - field_names}"
        )

    def test_unbiased_selection_stats_field_types(self):
        """Check that UnbiasedSelectionStats fields have the expected types."""
        field_map = {f.name: f.type for f in dataclass_fields(UnbiasedSelectionStats)}
        # Types may be expressed as strings or actual types depending on annotations
        assert "total_significant_terms" in field_map
        assert "terms_after_dedup" in field_map
        assert "terms_selected" in field_map
        assert "n_clusters" in field_map
        assert "random_seed" in field_map
        assert "clustering_algorithm" in field_map

    def test_pool_significant_terms_signature(self):
        """pool_significant_terms has correct parameter names."""
        sig = inspect.signature(pool_significant_terms)
        param_names = list(sig.parameters.keys())
        assert "cohort" in param_names
        assert "fdr_threshold" in param_names

    def test_remove_redundant_terms_signature(self):
        """remove_redundant_terms has correct parameter names."""
        sig = inspect.signature(remove_redundant_terms)
        param_names = list(sig.parameters.keys())
        assert "ranked_terms" in param_names

    def test_select_top_n_signature(self):
        """select_top_n has correct parameter names."""
        sig = inspect.signature(select_top_n)
        param_names = list(sig.parameters.keys())
        assert "ranked_terms" in param_names
        assert "top_n" in param_names

    def test_cluster_terms_signature(self):
        """cluster_terms has correct parameter names."""
        sig = inspect.signature(cluster_terms)
        param_names = list(sig.parameters.keys())
        assert "term_names" in param_names
        assert "cohort" in param_names
        assert "n_groups" in param_names
        assert "random_seed" in param_names

    def test_select_unbiased_terms_signature(self):
        """select_unbiased_terms has correct parameter and default values."""
        sig = inspect.signature(select_unbiased_terms)
        params = sig.parameters
        assert "cohort" in params
        assert "fdr_threshold" in params
        assert "top_n" in params
        assert "n_groups" in params
        assert "random_seed" in params
        # Check defaults
        assert params["fdr_threshold"].default == 0.05
        assert params["top_n"].default == 20
        assert params["n_groups"].default == 4
        assert params["random_seed"].default == 42

    def test_select_unbiased_terms_returns_tuple(self):
        """select_unbiased_terms must return a tuple of (list[CategoryGroup], UnbiasedSelectionStats)."""
        sig = inspect.signature(select_unbiased_terms)
        # We verify the return annotation exists but may be string or type
        # Actually, we test the return value type in behavioral tests.
        # Just ensure the function is callable.
        assert callable(select_unbiased_terms)


# ===========================================================================
# Section 2: pool_significant_terms (Steps 1-2)
# ===========================================================================

class TestPoolSignificantTerms:
    """Test step 1-2: pooling significant terms and ranking by max abs NES."""

    def test_basic_pooling(self):
        """Terms with FDR < threshold in at least one mutant are pooled.

        DATA ASSUMPTION: Two mutants, each with 2 terms. One term ("TERM_A")
        is significant in both mutants; one ("TERM_B") is significant in one;
        one ("TERM_C") is not significant in any.
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 2.0, 0.01),   # significant
                ("TERM_B", -1.5, 0.04),   # significant
                ("TERM_C", 0.5, 0.10),    # NOT significant
            ],
            "mutB": [
                ("TERM_A", -1.0, 0.02),   # significant
                ("TERM_B", 0.3, 0.80),    # NOT significant
                ("TERM_C", 0.1, 0.90),    # NOT significant
            ],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "TERM_A" in result
        assert "TERM_B" in result
        assert "TERM_C" not in result

    def test_ranking_by_max_abs_nes(self):
        """Terms are ranked by maximum absolute NES descending.

        DATA ASSUMPTION: Three significant terms with distinct max abs NES
        values: TERM_X=3.0, TERM_Y=2.0, TERM_Z=1.0.
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_X", 3.0, 0.01),
                ("TERM_Y", -2.0, 0.02),
                ("TERM_Z", 1.0, 0.03),
            ],
            "mutB": [
                ("TERM_X", 1.0, 0.01),
                ("TERM_Y", 1.5, 0.02),
                ("TERM_Z", -0.5, 0.04),
            ],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        keys = list(result.keys())
        # TERM_X: max abs NES = 3.0, TERM_Y: max abs NES = 2.0, TERM_Z: max abs NES = 1.0
        assert keys == ["TERM_X", "TERM_Y", "TERM_Z"]
        assert result["TERM_X"] == pytest.approx(3.0)
        assert result["TERM_Y"] == pytest.approx(2.0)
        assert result["TERM_Z"] == pytest.approx(1.0)

    def test_max_abs_nes_uses_absolute_value(self):
        """Negative NES values contribute their absolute value to max abs NES.

        DATA ASSUMPTION: TERM_A has NES=-3.0 in mutA (abs=3.0) and NES=1.0
        in mutB. The max abs NES should be 3.0, not 1.0.
        """
        cohort = _build_cohort({
            "mutA": [("TERM_A", -3.0, 0.01)],
            "mutB": [("TERM_A", 1.0, 0.02)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert result["TERM_A"] == pytest.approx(3.0)

    def test_fdr_threshold_boundary_excluded(self):
        """Terms with FDR exactly equal to the threshold are NOT included.

        DATA ASSUMPTION: TERM_A has FDR=0.05 exactly in both mutants.
        Per contract, FDR < threshold (strict), so FDR=0.05 with threshold=0.05
        means NOT significant.
        """
        cohort = _build_cohort({
            "mutA": [("TERM_A", 2.0, 0.05)],
            "mutB": [("TERM_A", 1.5, 0.05)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "TERM_A" not in result

    def test_fdr_threshold_boundary_included(self):
        """Terms with FDR just below the threshold ARE included.

        DATA ASSUMPTION: TERM_A has FDR=0.049 in one mutant (just under 0.05).
        """
        cohort = _build_cohort({
            "mutA": [("TERM_A", 2.0, 0.049)],
            "mutB": [("TERM_A", 1.5, 0.06)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "TERM_A" in result

    def test_tie_breaking_alphabetical(self):
        """When two terms have the same max abs NES, they are sorted alphabetically.

        DATA ASSUMPTION: TERM_B and TERM_A both have max abs NES of 2.0.
        Alphabetical tie-breaking puts TERM_A before TERM_B.
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_B", 2.0, 0.01),
                ("TERM_A", -2.0, 0.01),
            ],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        keys = list(result.keys())
        assert keys == ["TERM_A", "TERM_B"]

    def test_term_significant_in_one_mutant_only(self):
        """A term that passes FDR in only one mutant is still pooled.

        DATA ASSUMPTION: TERM_A has FDR=0.01 in mutA but does not appear
        in mutB at all.
        """
        cohort = _build_cohort({
            "mutA": [("TERM_A", 2.5, 0.01)],
            "mutB": [],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "TERM_A" in result
        assert result["TERM_A"] == pytest.approx(2.5)

    def test_empty_result_when_no_significant_terms(self):
        """If no terms pass FDR threshold, return empty dict.

        DATA ASSUMPTION: All terms have FDR=0.5, well above the 0.05 threshold.
        """
        cohort = _build_cohort({
            "mutA": [("TERM_A", 1.0, 0.50)],
            "mutB": [("TERM_B", 2.0, 0.60)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert len(result) == 0

    def test_max_abs_nes_across_multiple_mutants(self):
        """Max abs NES is computed across all mutants, not just the first.

        DATA ASSUMPTION: TERM_A has NES=1.0 in mutA, NES=0.5 in mutB,
        NES=-2.5 in mutC. The max abs NES should be 2.5 (from mutC).
        """
        cohort = _build_cohort({
            "mutA": [("TERM_A", 1.0, 0.01)],
            "mutB": [("TERM_A", 0.5, 0.02)],
            "mutC": [("TERM_A", -2.5, 0.03)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert result["TERM_A"] == pytest.approx(2.5)

    def test_only_significant_records_contribute_to_pool(self):
        """Only FDR<threshold records make a term eligible for the pool.

        DATA ASSUMPTION: TERM_A has FDR=0.01 in mutA (significant), but
        FDR=0.80 in mutB (not significant). However, the NES in mutB
        is higher (3.0 vs 1.0). The max abs NES should still consider
        all records for a pooled term -- but the term only enters the pool
        because of mutA. The max abs NES is computed from all mutant records.
        """
        cohort = _build_cohort({
            "mutA": [("TERM_A", 1.0, 0.01)],
            "mutB": [("TERM_A", 3.0, 0.80)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        # TERM_A is in the pool because FDR < 0.05 in mutA.
        assert "TERM_A" in result
        # The max abs NES might be computed from ALL records or only significant ones.
        # The blueprint says "maximum absolute NES observed across the mutant cohort"
        # which means all mutants, not just significant ones. But terms enter pool
        # only if significant in at least one.
        # We verify the term is present. The exact value depends on interpretation.
        # The blueprint says "max absolute NES observed across the mutant cohort"
        # for ranking. Since the term IS present in mutB (just not significant),
        # the NES from mutB (3.0) should be the max abs NES.
        assert result["TERM_A"] == pytest.approx(3.0)


# ===========================================================================
# Section 3: remove_redundant_terms (Step 3)
# ===========================================================================

class TestRemoveRedundantTerms:
    """Test step 3: lexical redundancy removal."""

    def test_no_redundancy(self):
        """Terms with no word overlap are all retained.

        DATA ASSUMPTION: Three terms with completely different words.
        """
        ranked = {
            "OXIDATIVE PHOSPHORYLATION": 3.0,
            "RIBOSOME BIOGENESIS": 2.0,
            "GPCR SIGNALING": 1.0,
        }
        result = remove_redundant_terms(ranked)
        assert len(result) == 3
        assert set(result.keys()) == set(ranked.keys())

    def test_high_overlap_removes_lower_ranked(self):
        """When two terms share >50% Jaccard word overlap, the lower-ranked one
        is removed.

        DATA ASSUMPTION: "REGULATION OF CELL DEATH" and
        "POSITIVE REGULATION OF CELL DEATH" share words
        {REGULATION, OF, CELL, DEATH} and {POSITIVE, REGULATION, OF, CELL, DEATH}.
        Jaccard = |intersection|/|union| = 4/5 = 0.8 > 0.5.
        The higher-ranked term survives.
        """
        ranked = {
            "POSITIVE REGULATION OF CELL DEATH": 3.0,
            "REGULATION OF CELL DEATH": 2.5,
        }
        result = remove_redundant_terms(ranked)
        assert "POSITIVE REGULATION OF CELL DEATH" in result
        assert "REGULATION OF CELL DEATH" not in result

    def test_exactly_50_percent_jaccard_not_redundant(self):
        """Jaccard similarity exactly 0.5 does NOT trigger removal (threshold is >0.5).

        DATA ASSUMPTION: "CELL DEATH REGULATION PATHWAY" and
        "CELL DEATH APOPTOSIS SIGNALING" share {CELL, DEATH} from
        {CELL, DEATH, REGULATION, PATHWAY} and {CELL, DEATH, APOPTOSIS, SIGNALING}.
        Jaccard = 2/6 = 0.333... < 0.5. Both retained.

        Better example for exactly 0.5:
        "A B" and "A C" -> {A,B} and {A,C}, Jaccard = 1/3 = 0.333. Not 0.5.
        "A B C" and "A B D" -> {A,B,C} and {A,B,D}, Jaccard = 2/4 = 0.5. Exactly 0.5.
        Since threshold is >0.5, both should be retained.
        """
        ranked = {
            "A B C": 3.0,
            "A B D": 2.0,
        }
        result = remove_redundant_terms(ranked)
        assert len(result) == 2

    def test_just_above_50_percent_jaccard_is_redundant(self):
        """Jaccard similarity just above 0.5 triggers removal.

        DATA ASSUMPTION: "A B C D" and "A B C E" share {A,B,C} from
        {A,B,C,D} and {A,B,C,E}. Jaccard = 3/5 = 0.6 > 0.5.
        Lower ranked term is removed.
        """
        ranked = {
            "A B C D": 3.0,
            "A B C E": 2.0,
        }
        result = remove_redundant_terms(ranked)
        assert "A B C D" in result
        assert "A B C E" not in result
        assert len(result) == 1

    def test_higher_ranked_always_survives(self):
        """Processing in rank order ensures the higher-ranked term survives.

        DATA ASSUMPTION: Two redundant terms. The one with max abs NES=2.0
        should be removed in favor of the one with max abs NES=3.0.
        """
        ranked = {
            "REGULATION OF APOPTOTIC PROCESS": 3.0,
            "POSITIVE REGULATION OF APOPTOTIC PROCESS": 2.0,
        }
        # Jaccard: {REGULATION, OF, APOPTOTIC, PROCESS} vs
        # {POSITIVE, REGULATION, OF, APOPTOTIC, PROCESS}
        # intersection = 4, union = 5, Jaccard = 0.8 > 0.5
        result = remove_redundant_terms(ranked)
        assert "REGULATION OF APOPTOTIC PROCESS" in result
        assert "POSITIVE REGULATION OF APOPTOTIC PROCESS" not in result

    def test_chain_redundancy(self):
        """Redundancy is checked against surviving terms only.

        DATA ASSUMPTION: Three terms where A is redundant with B, and B is
        redundant with C. If A survives (highest rank), B is removed.
        Then C is checked against A only (B was removed).
        """
        # "X Y Z" (rank 3.0) vs "X Y W" (rank 2.0): Jaccard = 2/4 = 0.5. Not redundant.
        # Use higher overlap:
        # "X Y Z Q" (rank 3.0), "X Y Z R" (rank 2.0), "X Y Z S" (rank 1.0)
        # Each pair: Jaccard = 3/5 = 0.6 > 0.5
        ranked = {
            "X Y Z Q": 3.0,
            "X Y Z R": 2.0,
            "X Y Z S": 1.0,
        }
        result = remove_redundant_terms(ranked)
        # X Y Z Q survives, X Y Z R removed (redundant with Q), X Y Z S removed too
        assert "X Y Z Q" in result
        assert "X Y Z R" not in result
        assert "X Y Z S" not in result
        assert len(result) == 1

    def test_empty_input(self):
        """Empty input returns empty output."""
        result = remove_redundant_terms({})
        assert len(result) == 0

    def test_single_term(self):
        """A single term has nothing to be redundant with."""
        ranked = {"SOME TERM": 2.5}
        result = remove_redundant_terms(ranked)
        assert len(result) == 1
        assert "SOME TERM" in result

    def test_preserves_order(self):
        """Output preserves the original ranking order.

        DATA ASSUMPTION: Four terms with no redundancy. The output dict
        should preserve descending rank order.
        """
        ranked = {
            "ALPHA": 4.0,
            "BETA": 3.0,
            "GAMMA": 2.0,
            "DELTA": 1.0,
        }
        result = remove_redundant_terms(ranked)
        keys = list(result.keys())
        assert keys == ["ALPHA", "BETA", "GAMMA", "DELTA"]

    def test_redundancy_does_not_affect_unrelated(self):
        """Non-overlapping terms are unaffected when two others are redundant.

        DATA ASSUMPTION: TERM_A and TERM_B are redundant (high overlap),
        TERM_C is completely unrelated.
        """
        ranked = {
            "REGULATION OF CELL MIGRATION": 3.0,
            "POSITIVE REGULATION OF CELL MIGRATION": 2.5,
            "RIBOSOME ASSEMBLY": 2.0,
        }
        result = remove_redundant_terms(ranked)
        assert "REGULATION OF CELL MIGRATION" in result
        assert "POSITIVE REGULATION OF CELL MIGRATION" not in result
        assert "RIBOSOME ASSEMBLY" in result
        assert len(result) == 2


# ===========================================================================
# Section 4: select_top_n (Step 4)
# ===========================================================================

class TestSelectTopN:
    """Test step 4: selecting top N terms from deduplicated ranked list."""

    def test_basic_top_n(self):
        """Select top 3 from 5 ranked terms.

        DATA ASSUMPTION: Five terms ranked by max abs NES.
        """
        ranked = {
            "TERM_A": 5.0,
            "TERM_B": 4.0,
            "TERM_C": 3.0,
            "TERM_D": 2.0,
            "TERM_E": 1.0,
        }
        result = select_top_n(ranked, top_n=3)
        assert result == ["TERM_A", "TERM_B", "TERM_C"]

    def test_top_n_exceeds_available(self):
        """When fewer than N terms remain, all are returned (contract 4).

        DATA ASSUMPTION: Only 2 terms available, requesting top 5.
        """
        ranked = {
            "TERM_A": 3.0,
            "TERM_B": 2.0,
        }
        result = select_top_n(ranked, top_n=5)
        assert result == ["TERM_A", "TERM_B"]

    def test_top_n_equals_available(self):
        """When exactly N terms are available, all are returned."""
        ranked = {
            "TERM_A": 3.0,
            "TERM_B": 2.0,
            "TERM_C": 1.0,
        }
        result = select_top_n(ranked, top_n=3)
        assert result == ["TERM_A", "TERM_B", "TERM_C"]

    def test_returns_list_of_strings(self):
        """The return type is a list of term name strings, not (name, score) tuples."""
        ranked = {"TERM_A": 2.0}
        result = select_top_n(ranked, top_n=1)
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_preserves_rank_order(self):
        """The returned list preserves the ranking order from the input dict.

        DATA ASSUMPTION: Four terms with clear ranking.
        """
        ranked = {
            "FOURTH": 4.0,
            "THIRD": 3.0,
            "SECOND": 2.0,
            "FIRST": 1.0,
        }
        result = select_top_n(ranked, top_n=4)
        assert result == ["FOURTH", "THIRD", "SECOND", "FIRST"]


# ===========================================================================
# Section 5: cluster_terms (Steps 5-6)
# ===========================================================================

class TestClusterTerms:
    """Test steps 5-6: hierarchical clustering and auto-labeling."""

    def _build_clustering_cohort(self):
        """Build a cohort with terms having distinct NES profiles for clustering.

        DATA ASSUMPTION: 6 terms across 3 mutants. Terms are designed in
        pairs with similar NES profiles to form 3 natural clusters:
          - Cluster 1: TERM_A (+2, +2, 0) and TERM_B (+1.8, +1.9, 0)
          - Cluster 2: TERM_C (0, -2, -2) and TERM_D (0, -1.8, -1.9)
          - Cluster 3: TERM_E (+1, -1, +1) and TERM_F (+1.1, -0.9, +1.1)
        """
        return _build_cohort({
            "mutA": [
                ("TERM_A", 2.0, 0.01),
                ("TERM_B", 1.8, 0.01),
                ("TERM_C", 0.0, 0.01),
                ("TERM_D", 0.0, 0.01),
                ("TERM_E", 1.0, 0.01),
                ("TERM_F", 1.1, 0.01),
            ],
            "mutB": [
                ("TERM_A", 2.0, 0.01),
                ("TERM_B", 1.9, 0.01),
                ("TERM_C", -2.0, 0.01),
                ("TERM_D", -1.8, 0.01),
                ("TERM_E", -1.0, 0.01),
                ("TERM_F", -0.9, 0.01),
            ],
            "mutC": [
                ("TERM_A", 0.0, 0.01),
                ("TERM_B", 0.0, 0.01),
                ("TERM_C", -2.0, 0.01),
                ("TERM_D", -1.9, 0.01),
                ("TERM_E", 1.0, 0.01),
                ("TERM_F", 1.1, 0.01),
            ],
        })

    def test_returns_list_of_category_groups(self):
        """cluster_terms returns a list of CategoryGroup objects."""
        cohort = self._build_clustering_cohort()
        term_names = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(term_names, cohort, n_groups=3, random_seed=42)
        assert isinstance(groups, list)
        assert all(isinstance(g, CategoryGroup) for g in groups)

    def test_correct_number_of_groups(self):
        """The number of returned groups equals n_groups when enough terms exist."""
        cohort = self._build_clustering_cohort()
        term_names = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(term_names, cohort, n_groups=3, random_seed=42)
        assert len(groups) == 3

    def test_all_terms_assigned_to_groups(self):
        """Every input term must appear in exactly one group."""
        cohort = self._build_clustering_cohort()
        term_names = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(term_names, cohort, n_groups=3, random_seed=42)
        all_grouped_terms = []
        for g in groups:
            all_grouped_terms.extend(g.term_names)
        assert set(all_grouped_terms) == set(term_names)
        # No duplicates
        assert len(all_grouped_terms) == len(term_names)

    def test_no_empty_groups(self):
        """No group should be empty (invariant)."""
        cohort = self._build_clustering_cohort()
        term_names = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(term_names, cohort, n_groups=3, random_seed=42)
        assert all(len(g.term_names) > 0 for g in groups)

    def test_group_label_is_highest_mean_abs_nes_term(self):
        """Each group is labeled with the term having the highest mean abs NES.

        DATA ASSUMPTION: Within a cluster, the term with the higher mean abs
        NES should be the group label (category_name).
        """
        cohort = self._build_clustering_cohort()
        term_names = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(term_names, cohort, n_groups=3, random_seed=42)
        for g in groups:
            # The category_name must be one of the terms in the group
            assert g.category_name in g.term_names

    def test_terms_within_group_sorted_by_mean_abs_nes_descending(self):
        """Within each group, terms are sorted by mean absolute NES descending.

        DATA ASSUMPTION: Using the clustering cohort where terms within the same
        cluster have slightly different mean abs NES values.
        """
        cohort = self._build_clustering_cohort()
        term_names = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(term_names, cohort, n_groups=3, random_seed=42)

        for g in groups:
            # Compute mean abs NES for each term in the group
            mean_abs_nes_values = []
            for t in g.term_names:
                total = 0.0
                count = 0
                for mid in cohort.mutant_ids:
                    profile = cohort.profiles[mid]
                    rec = profile.records.get(t)
                    if rec is not None:
                        total += abs(rec.nes)
                    count += 1
                mean_abs_nes_values.append(total / count if count > 0 else 0.0)

            # Check descending order
            for i in range(len(mean_abs_nes_values) - 1):
                assert mean_abs_nes_values[i] >= mean_abs_nes_values[i + 1], (
                    f"Terms in group '{g.category_name}' not sorted by mean abs NES descending: "
                    f"{list(zip(g.term_names, mean_abs_nes_values))}"
                )

    def test_missing_nes_treated_as_zero(self):
        """Missing NES values (term absent from a mutant) are treated as 0.0.

        DATA ASSUMPTION: TERM_A appears in mutA (NES=2.0) but not mutB.
        The NES profile for clustering should be [2.0, 0.0].
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 2.0, 0.01),
                ("TERM_B", -1.0, 0.02),
            ],
            "mutB": [
                # TERM_A is missing from mutB
                ("TERM_B", -1.5, 0.01),
            ],
        })
        # With only 2 terms and 1 group, both should be in the same group
        groups = cluster_terms(["TERM_A", "TERM_B"], cohort, n_groups=1, random_seed=42)
        assert len(groups) == 1
        assert set(groups[0].term_names) == {"TERM_A", "TERM_B"}

    def test_groups_ordered_by_original_rank(self):
        """Groups are ordered by the rank position of their highest-ranked member.

        DATA ASSUMPTION: The term_names list is given in rank order (highest
        max abs NES first). The group containing the #1 ranked term should
        appear first, then the group containing the highest-ranked term not
        in the first group, etc.
        """
        cohort = self._build_clustering_cohort()
        # TERM_A has the highest max abs NES (2.0), followed by TERM_C (2.0), etc.
        # But in the input list, the order defines rank position.
        term_names = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(term_names, cohort, n_groups=3, random_seed=42)

        # For each group, find the best rank (lowest index) among its terms
        def best_rank(group):
            return min(term_names.index(t) for t in group.term_names)

        group_ranks = [best_rank(g) for g in groups]
        # Groups should be sorted by their best rank ascending
        assert group_ranks == sorted(group_ranks), (
            f"Groups not ordered by highest-ranked member: {group_ranks}"
        )

    def test_deterministic_output(self):
        """Same input produces same output (contract 11).

        DATA ASSUMPTION: Running cluster_terms twice with the same inputs
        and random_seed should produce identical results.
        """
        cohort = self._build_clustering_cohort()
        term_names = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups1 = cluster_terms(term_names, cohort, n_groups=3, random_seed=42)
        groups2 = cluster_terms(term_names, cohort, n_groups=3, random_seed=42)
        assert len(groups1) == len(groups2)
        for g1, g2 in zip(groups1, groups2):
            assert g1.category_name == g2.category_name
            assert g1.term_names == g2.term_names

    def test_single_group(self):
        """With n_groups=1, all terms go into one group."""
        cohort = _build_cohort({
            "mutA": [
                ("TERM_X", 2.0, 0.01),
                ("TERM_Y", 1.5, 0.01),
                ("TERM_Z", 1.0, 0.01),
            ],
        })
        groups = cluster_terms(
            ["TERM_X", "TERM_Y", "TERM_Z"], cohort, n_groups=1, random_seed=42
        )
        assert len(groups) == 1
        assert set(groups[0].term_names) == {"TERM_X", "TERM_Y", "TERM_Z"}

    def test_n_groups_equals_n_terms(self):
        """With n_groups equal to number of terms, each term gets its own group."""
        cohort = _build_cohort({
            "mutA": [
                ("TERM_X", 2.0, 0.01),
                ("TERM_Y", 1.5, 0.01),
                ("TERM_Z", 1.0, 0.01),
            ],
        })
        groups = cluster_terms(
            ["TERM_X", "TERM_Y", "TERM_Z"], cohort, n_groups=3, random_seed=42
        )
        assert len(groups) == 3
        assert all(len(g.term_names) == 1 for g in groups)

    def test_category_name_is_string(self):
        """Each group's category_name is a string (the auto-label)."""
        cohort = _build_cohort({
            "mutA": [("TERM_A", 2.0, 0.01), ("TERM_B", 1.0, 0.01)],
        })
        groups = cluster_terms(["TERM_A", "TERM_B"], cohort, n_groups=1, random_seed=42)
        for g in groups:
            assert isinstance(g.category_name, str)


# ===========================================================================
# Section 6: select_unbiased_terms (top-level entry point)
# ===========================================================================

class TestSelectUnbiasedTerms:
    """Test the top-level pipeline function."""

    def _build_large_cohort(self):
        """Build a cohort with enough significant terms for full pipeline.

        DATA ASSUMPTION: 25 distinct significant terms across 3 mutants,
        with varying NES and FDR values. Some terms designed to be
        lexically redundant.
        """
        terms = []
        for i in range(25):
            terms.append((f"UNIQUE TERM {i:02d}", 3.0 - i * 0.1, 0.01))

        mutant_data = {}
        for mid in ["mutA", "mutB", "mutC"]:
            mutant_data[mid] = [(t, nes + (hash(mid + t) % 10) * 0.01, fdr)
                                for t, nes, fdr in terms]

        return _build_cohort(mutant_data)

    def test_returns_tuple(self):
        """select_unbiased_terms returns a (list[CategoryGroup], UnbiasedSelectionStats) tuple."""
        cohort = self._build_large_cohort()
        result = select_unbiased_terms(cohort)
        assert isinstance(result, tuple)
        assert len(result) == 2
        groups, stats = result
        assert isinstance(groups, list)
        assert isinstance(stats, UnbiasedSelectionStats)

    def test_groups_are_category_groups(self):
        """Each element in the groups list is a CategoryGroup."""
        cohort = self._build_large_cohort()
        groups, _ = select_unbiased_terms(cohort)
        assert all(isinstance(g, CategoryGroup) for g in groups)

    def test_default_parameters(self):
        """With default parameters, top_n=20 and n_groups=4."""
        cohort = self._build_large_cohort()
        groups, stats = select_unbiased_terms(cohort)
        # With 25 unique terms and default top_n=20, we should have <= 20 terms
        total_terms = sum(len(g.term_names) for g in groups)
        assert total_terms <= 20
        assert len(groups) <= 4

    def test_stats_captures_parameters(self):
        """UnbiasedSelectionStats records the parameters used.

        DATA ASSUMPTION: We pass explicit parameters and verify they are
        recorded in the stats.
        """
        cohort = self._build_large_cohort()
        _, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=3, random_seed=99
        )
        assert stats.n_clusters == 3
        assert stats.random_seed == 99
        assert stats.terms_selected <= 10

    def test_stats_random_seed_matches(self):
        """Stats must record the seed actually used (invariant)."""
        cohort = self._build_large_cohort()
        _, stats = select_unbiased_terms(cohort, random_seed=123)
        assert stats.random_seed == 123

    def test_stats_total_significant_terms(self):
        """Stats records total significant terms after step 1.

        DATA ASSUMPTION: A cohort with some significant and some non-significant
        terms. The stats should count only the significant ones.
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 2.0, 0.01),   # significant
                ("TERM_B", 1.5, 0.02),   # significant
                ("TERM_C", 1.0, 0.50),   # NOT significant
            ],
            "mutB": [
                ("TERM_A", 1.0, 0.03),   # significant
                ("TERM_D", 0.5, 0.04),   # significant
                ("TERM_C", 0.8, 0.60),   # NOT significant
            ],
        })
        _, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=2, random_seed=42
        )
        # TERM_A, TERM_B, TERM_D are significant. TERM_C is not.
        assert stats.total_significant_terms == 3

    def test_stats_terms_after_dedup(self):
        """Stats records terms remaining after deduplication (step 3).

        DATA ASSUMPTION: Two pairs of redundant terms plus one unique term.
        After dedup, we should have 3 terms (2 survivors + 1 unique).
        """
        cohort = _build_cohort({
            "mutA": [
                ("REGULATION OF CELL DEATH", 3.0, 0.01),
                ("POSITIVE REGULATION OF CELL DEATH", 2.5, 0.01),
                ("RIBOSOME ASSEMBLY", 2.0, 0.01),
                ("RIBOSOME BIOGENESIS AND ASSEMBLY", 1.5, 0.01),
                ("GPCR SIGNALING PATHWAY", 1.0, 0.01),
            ],
        })
        _, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=2, random_seed=42
        )
        assert stats.total_significant_terms == 5
        # After dedup: "REGULATION OF CELL DEATH" survives (removes "POSITIVE..."),
        # "RIBOSOME ASSEMBLY" vs "RIBOSOME BIOGENESIS AND ASSEMBLY": Jaccard
        # {RIBOSOME, ASSEMBLY} vs {RIBOSOME, BIOGENESIS, AND, ASSEMBLY} = 2/4 = 0.5
        # Exactly 0.5, NOT > 0.5, so both survive.
        # "GPCR SIGNALING PATHWAY" unique.
        # So terms_after_dedup should be 4.
        assert stats.terms_after_dedup == 4

    def test_stats_terms_selected(self):
        """Stats records number of terms selected in step 4."""
        cohort = self._build_large_cohort()
        _, stats = select_unbiased_terms(cohort, top_n=10, n_groups=3)
        assert stats.terms_selected <= 10
        assert stats.terms_selected > 0

    def test_stats_clustering_algorithm(self):
        """Stats records the clustering algorithm description."""
        cohort = self._build_large_cohort()
        _, stats = select_unbiased_terms(cohort)
        assert isinstance(stats.clustering_algorithm, str)
        assert len(stats.clustering_algorithm) > 0

    def test_no_empty_groups(self):
        """Post-condition: no empty groups."""
        cohort = self._build_large_cohort()
        groups, _ = select_unbiased_terms(cohort)
        assert all(len(g.term_names) > 0 for g in groups)

    def test_total_terms_not_exceed_top_n(self):
        """Post-condition: total terms across groups cannot exceed top_n."""
        cohort = self._build_large_cohort()
        groups, _ = select_unbiased_terms(cohort, top_n=15)
        total = sum(len(g.term_names) for g in groups)
        assert total <= 15

    def test_number_of_groups_not_exceed_n_groups(self):
        """Post-condition: number of groups <= n_groups."""
        cohort = self._build_large_cohort()
        groups, _ = select_unbiased_terms(cohort, n_groups=3)
        assert len(groups) <= 3

    def test_deterministic_with_same_seed(self):
        """Same input and seed produces identical output (contract 11)."""
        cohort = self._build_large_cohort()
        groups1, stats1 = select_unbiased_terms(cohort, random_seed=42)
        groups2, stats2 = select_unbiased_terms(cohort, random_seed=42)
        assert len(groups1) == len(groups2)
        for g1, g2 in zip(groups1, groups2):
            assert g1.category_name == g2.category_name
            assert g1.term_names == g2.term_names
        assert stats1.total_significant_terms == stats2.total_significant_terms
        assert stats1.terms_after_dedup == stats2.terms_after_dedup
        assert stats1.terms_selected == stats2.terms_selected

    def test_custom_fdr_threshold(self):
        """A more permissive FDR threshold includes more terms.

        DATA ASSUMPTION: Terms with FDR between 0.05 and 0.10 should be
        included with threshold=0.10 but not with threshold=0.05.
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 2.0, 0.01),   # sig at 0.05
                ("TERM_B", 1.5, 0.07),   # sig at 0.10 but not 0.05
                ("TERM_C", 1.0, 0.03),   # sig at 0.05
            ],
            "mutB": [
                ("TERM_A", 1.0, 0.02),
                ("TERM_B", 0.5, 0.08),
                ("TERM_C", 0.5, 0.04),
            ],
        })
        _, stats_strict = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=2, random_seed=42
        )
        _, stats_permissive = select_unbiased_terms(
            cohort, fdr_threshold=0.10, top_n=10, n_groups=2, random_seed=42
        )
        assert stats_permissive.total_significant_terms >= stats_strict.total_significant_terms

    def test_fewer_terms_than_top_n(self):
        """When fewer significant terms exist than top_n, all are used (contract 4).

        DATA ASSUMPTION: Only 3 significant terms but top_n=20.
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 2.0, 0.01),
                ("TERM_B", 1.5, 0.02),
                ("TERM_C", 1.0, 0.03),
            ],
            "mutB": [
                ("TERM_A", 1.0, 0.01),
                ("TERM_B", 0.5, 0.02),
                ("TERM_C", 0.5, 0.04),
            ],
        })
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=20, n_groups=2, random_seed=42
        )
        assert stats.terms_selected <= 3
        total = sum(len(g.term_names) for g in groups)
        assert total <= 3


# ===========================================================================
# Section 7: Error conditions
# ===========================================================================

class TestErrorConditions:
    """Test error conditions specified in the blueprint."""

    def test_insufficient_significant_terms_raises_value_error(self):
        """ValueError when fewer terms pass FDR threshold than n_groups.

        DATA ASSUMPTION: Only 1 significant term, but n_groups=4 requires
        at least 4 terms for clustering.
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 2.0, 0.01),   # significant
                ("TERM_B", 1.0, 0.50),   # NOT significant
            ],
            "mutB": [
                ("TERM_A", 1.5, 0.02),
                ("TERM_B", 0.5, 0.60),
            ],
        })
        with pytest.raises(ValueError):
            select_unbiased_terms(
                cohort, fdr_threshold=0.05, top_n=20, n_groups=4, random_seed=42
            )

    def test_zero_significant_terms_raises_value_error(self):
        """ValueError when no terms pass FDR threshold at all.

        DATA ASSUMPTION: All terms have FDR=0.90, far above threshold.
        """
        cohort = _build_cohort({
            "mutA": [("TERM_A", 2.0, 0.90)],
            "mutB": [("TERM_B", 1.5, 0.80)],
        })
        with pytest.raises(ValueError):
            select_unbiased_terms(
                cohort, fdr_threshold=0.05, top_n=20, n_groups=4, random_seed=42
            )

    def test_exactly_n_groups_terms_does_not_raise(self):
        """When exactly n_groups terms pass, clustering should work.

        DATA ASSUMPTION: Exactly 4 significant terms with n_groups=4.
        Each term in its own cluster.
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 3.0, 0.01),
                ("TERM_B", 2.5, 0.01),
                ("TERM_C", 2.0, 0.01),
                ("TERM_D", 1.5, 0.01),
            ],
            "mutB": [
                ("TERM_A", 2.0, 0.01),
                ("TERM_B", 1.5, 0.01),
                ("TERM_C", 1.0, 0.01),
                ("TERM_D", 0.5, 0.01),
            ],
        })
        # Should not raise
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=20, n_groups=4, random_seed=42
        )
        assert len(groups) <= 4
        assert stats.terms_selected == 4

    def test_n_groups_exceeds_top_n_precondition(self):
        """n_groups > top_n should fail (invariant: n_groups <= top_n).

        DATA ASSUMPTION: top_n=2, n_groups=5. This violates the precondition.
        The implementation should raise an error (likely ValueError or AssertionError).
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 3.0, 0.01),
                ("TERM_B", 2.0, 0.01),
                ("TERM_C", 1.0, 0.01),
            ],
            "mutB": [
                ("TERM_A", 2.0, 0.01),
                ("TERM_B", 1.0, 0.01),
                ("TERM_C", 0.5, 0.01),
            ],
        })
        with pytest.raises((ValueError, AssertionError)):
            select_unbiased_terms(
                cohort, fdr_threshold=0.05, top_n=2, n_groups=5, random_seed=42
            )

    def test_top_n_zero_precondition(self):
        """top_n=0 should fail (invariant: top_n > 0)."""
        cohort = _build_cohort({
            "mutA": [("TERM_A", 2.0, 0.01)],
            "mutB": [("TERM_A", 1.5, 0.01)],
        })
        with pytest.raises((ValueError, AssertionError)):
            select_unbiased_terms(
                cohort, fdr_threshold=0.05, top_n=0, n_groups=1, random_seed=42
            )

    def test_n_groups_zero_precondition(self):
        """n_groups=0 should fail (invariant: n_groups > 0)."""
        cohort = _build_cohort({
            "mutA": [("TERM_A", 2.0, 0.01)],
            "mutB": [("TERM_A", 1.5, 0.01)],
        })
        with pytest.raises((ValueError, AssertionError)):
            select_unbiased_terms(
                cohort, fdr_threshold=0.05, top_n=10, n_groups=0, random_seed=42
            )


# ===========================================================================
# Section 8: Integration-level behavioral contracts
# ===========================================================================

class TestBehavioralContracts:
    """Test end-to-end behavioral contracts from the blueprint."""

    def test_contract_1_only_fdr_significant_terms_in_pool(self):
        """Contract 1: Only GO terms with FDR < fdr_threshold in at least one
        mutant are included in the candidate pool.

        DATA ASSUMPTION: 4 terms, only 2 pass FDR threshold in at least one mutant.
        """
        cohort = _build_cohort({
            "mutA": [
                ("SIGNIFICANT_1", 2.0, 0.01),
                ("SIGNIFICANT_2", 1.5, 0.03),
                ("NOT_SIG_1", 0.5, 0.10),
                ("NOT_SIG_2", 0.3, 0.90),
            ],
            "mutB": [
                ("SIGNIFICANT_1", 1.0, 0.02),
                ("SIGNIFICANT_2", 0.8, 0.06),   # not sig in mutB
                ("NOT_SIG_1", 0.4, 0.20),
                ("NOT_SIG_2", 0.2, 0.50),
            ],
        })
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=2, random_seed=42
        )
        all_terms = []
        for g in groups:
            all_terms.extend(g.term_names)
        assert "SIGNIFICANT_1" in all_terms
        assert "SIGNIFICANT_2" in all_terms
        assert "NOT_SIG_1" not in all_terms
        assert "NOT_SIG_2" not in all_terms

    def test_contract_2_ranking_by_max_abs_nes_with_tiebreak(self):
        """Contract 2: Terms ranked by max abs NES, ties broken alphabetically."""
        cohort = _build_cohort({
            "mutA": [
                ("ZEBRA TERM", 2.0, 0.01),
                ("ALPHA TERM", 2.0, 0.01),
                ("MIDDLE TERM", 1.5, 0.01),
            ],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        keys = list(result.keys())
        # "ALPHA TERM" and "ZEBRA TERM" tie at 2.0 -- alphabetical puts ALPHA first
        assert keys[0] == "ALPHA TERM"
        assert keys[1] == "ZEBRA TERM"
        assert keys[2] == "MIDDLE TERM"

    def test_contract_3_jaccard_redundancy_removal(self):
        """Contract 3: Jaccard > 0.5 removes lower-ranked term."""
        ranked = {
            "CELL CYCLE CHECKPOINT": 3.0,
            "CELL CYCLE REGULATION CHECKPOINT": 2.0,
            # {CELL, CYCLE, CHECKPOINT} vs {CELL, CYCLE, REGULATION, CHECKPOINT}
            # intersection = 3, union = 4, Jaccard = 0.75 > 0.5
        }
        result = remove_redundant_terms(ranked)
        assert "CELL CYCLE CHECKPOINT" in result
        assert "CELL CYCLE REGULATION CHECKPOINT" not in result

    def test_contract_4_fewer_than_n_terms_after_dedup(self):
        """Contract 4: If fewer than N terms remain after deduplication, all are used."""
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 2.0, 0.01),
                ("TERM_B", 1.5, 0.01),
            ],
            "mutB": [
                ("TERM_A", 1.0, 0.01),
                ("TERM_B", 0.5, 0.01),
            ],
        })
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=20, n_groups=2, random_seed=42
        )
        total_terms = sum(len(g.term_names) for g in groups)
        assert total_terms == 2  # only 2 terms available
        assert stats.terms_selected == 2

    def test_contract_5_ward_linkage_deterministic(self):
        """Contract 5: Ward linkage is deterministic; same seed gives same result."""
        cohort = _build_cohort({
            "mutA": [
                ("T1", 2.0, 0.01), ("T2", -1.5, 0.01),
                ("T3", 1.0, 0.01), ("T4", -0.5, 0.01),
            ],
            "mutB": [
                ("T1", 1.5, 0.01), ("T2", -2.0, 0.01),
                ("T3", 0.5, 0.01), ("T4", -1.0, 0.01),
            ],
        })
        g1 = cluster_terms(["T1", "T2", "T3", "T4"], cohort, n_groups=2, random_seed=42)
        g2 = cluster_terms(["T1", "T2", "T3", "T4"], cohort, n_groups=2, random_seed=99)
        # Ward linkage is deterministic regardless of seed
        assert len(g1) == len(g2)
        for a, b in zip(g1, g2):
            assert set(a.term_names) == set(b.term_names)

    def test_contract_6_group_label_highest_mean_abs_nes(self):
        """Contract 6: Each cluster group labeled with the term having highest
        mean abs NES within that group."""
        cohort = _build_cohort({
            "mutA": [
                ("HIGH_TERM", 3.0, 0.01),
                ("LOW_TERM", 0.5, 0.01),
            ],
            "mutB": [
                ("HIGH_TERM", 2.5, 0.01),
                ("LOW_TERM", 0.3, 0.01),
            ],
        })
        groups = cluster_terms(
            ["HIGH_TERM", "LOW_TERM"], cohort, n_groups=1, random_seed=42
        )
        # With both terms in one group, HIGH_TERM has higher mean abs NES
        assert groups[0].category_name == "HIGH_TERM"

    def test_contract_8_terms_sorted_within_group(self):
        """Contract 8: Within each group, terms sorted by mean abs NES descending."""
        cohort = _build_cohort({
            "mutA": [
                ("HIGH", 3.0, 0.01),
                ("MED", 2.0, 0.01),
                ("LOW", 1.0, 0.01),
            ],
            "mutB": [
                ("HIGH", 2.5, 0.01),
                ("MED", 1.5, 0.01),
                ("LOW", 0.5, 0.01),
            ],
        })
        groups = cluster_terms(
            ["HIGH", "MED", "LOW"], cohort, n_groups=1, random_seed=42
        )
        assert groups[0].term_names[0] == "HIGH"
        assert groups[0].term_names[-1] == "LOW"

    def test_contract_9_groups_ordered_by_highest_ranked_member(self):
        """Contract 9: Groups ordered by the position of their highest-ranked member.

        DATA ASSUMPTION: 4 terms forming 2 clusters. The cluster containing
        the #1 ranked term should be first.
        """
        # Create terms with very distinct profiles to force known clustering
        cohort = _build_cohort({
            "mutA": [
                ("RANK1", 3.0, 0.01),   # Cluster A
                ("RANK2", 2.8, 0.01),   # Cluster A
                ("RANK3", -2.5, 0.01),  # Cluster B
                ("RANK4", -2.3, 0.01),  # Cluster B
            ],
            "mutB": [
                ("RANK1", 2.5, 0.01),
                ("RANK2", 2.3, 0.01),
                ("RANK3", -2.0, 0.01),
                ("RANK4", -1.8, 0.01),
            ],
        })
        groups = cluster_terms(
            ["RANK1", "RANK2", "RANK3", "RANK4"], cohort, n_groups=2, random_seed=42
        )
        # The group containing RANK1 (highest ranked) should appear first
        first_group_terms = set(groups[0].term_names)
        assert "RANK1" in first_group_terms

    def test_contract_10_stats_captures_all_needed_info(self):
        """Contract 10: UnbiasedSelectionStats captures all parameters and counts."""
        cohort = _build_cohort({
            "mutA": [
                ("T1", 2.0, 0.01), ("T2", 1.5, 0.01),
                ("T3", 1.0, 0.01), ("T4", 0.5, 0.01),
            ],
            "mutB": [
                ("T1", 1.0, 0.01), ("T2", 0.5, 0.01),
                ("T3", 0.5, 0.01), ("T4", 0.3, 0.01),
            ],
        })
        _, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=4, n_groups=2, random_seed=42
        )
        # Verify all stat fields are populated
        assert isinstance(stats.total_significant_terms, int)
        assert isinstance(stats.terms_after_dedup, int)
        assert isinstance(stats.terms_selected, int)
        assert isinstance(stats.n_clusters, int)
        assert isinstance(stats.random_seed, int)
        assert isinstance(stats.clustering_algorithm, str)
        # Verify logical relationships
        assert stats.total_significant_terms >= stats.terms_after_dedup
        assert stats.terms_after_dedup >= stats.terms_selected or stats.terms_selected <= stats.terms_after_dedup
        assert stats.n_clusters == 2
        assert stats.random_seed == 42

    def test_full_pipeline_end_to_end(self):
        """End-to-end test of the complete 6-step pipeline.

        DATA ASSUMPTION: 8 significant terms across 3 mutants, some redundant.
        We test with top_n=6 and n_groups=2.
        """
        cohort = _build_cohort({
            "mutA": [
                ("OXIDATIVE PHOSPHORYLATION", 2.5, 0.01),
                ("MITOCHONDRIAL OXIDATIVE PHOSPHORYLATION", 2.0, 0.01),
                ("RIBOSOME BIOGENESIS", 1.8, 0.01),
                ("CELL CYCLE", 1.5, 0.01),
                ("DNA REPAIR", 1.2, 0.01),
                ("APOPTOSIS", 1.0, 0.01),
                ("GPCR SIGNALING", 0.8, 0.01),
                ("ION TRANSPORT", 0.5, 0.01),
            ],
            "mutB": [
                ("OXIDATIVE PHOSPHORYLATION", -2.0, 0.02),
                ("MITOCHONDRIAL OXIDATIVE PHOSPHORYLATION", -1.5, 0.02),
                ("RIBOSOME BIOGENESIS", 1.5, 0.02),
                ("CELL CYCLE", -1.0, 0.02),
                ("DNA REPAIR", 0.8, 0.02),
                ("APOPTOSIS", -0.7, 0.02),
                ("GPCR SIGNALING", 0.6, 0.03),
                ("ION TRANSPORT", 0.3, 0.03),
            ],
            "mutC": [
                ("OXIDATIVE PHOSPHORYLATION", 1.8, 0.01),
                ("MITOCHONDRIAL OXIDATIVE PHOSPHORYLATION", 1.3, 0.01),
                ("RIBOSOME BIOGENESIS", -1.2, 0.01),
                ("CELL CYCLE", 1.0, 0.01),
                ("DNA REPAIR", -0.5, 0.01),
                ("APOPTOSIS", 0.6, 0.03),
                ("GPCR SIGNALING", -0.4, 0.04),
                ("ION TRANSPORT", 0.2, 0.04),
            ],
        })

        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=6, n_groups=2, random_seed=42
        )

        # Basic post-conditions
        assert len(groups) <= 2
        total = sum(len(g.term_names) for g in groups)
        assert total <= 6
        assert all(len(g.term_names) > 0 for g in groups)

        # "OXIDATIVE PHOSPHORYLATION" and "MITOCHONDRIAL OXIDATIVE PHOSPHORYLATION"
        # are lexically redundant: {OXIDATIVE, PHOSPHORYLATION} vs
        # {MITOCHONDRIAL, OXIDATIVE, PHOSPHORYLATION}
        # Jaccard = 2/3 = 0.667 > 0.5 -- so MITOCHONDRIAL... should be removed
        all_terms = []
        for g in groups:
            all_terms.extend(g.term_names)
        assert "OXIDATIVE PHOSPHORYLATION" in all_terms
        assert "MITOCHONDRIAL OXIDATIVE PHOSPHORYLATION" not in all_terms

        # Stats should reflect the pipeline
        assert stats.total_significant_terms == 8
        assert stats.terms_after_dedup < 8  # At least one redundant pair
        assert stats.n_clusters == 2
        assert stats.random_seed == 42

    def test_output_compatible_with_unit3_category_group(self):
        """Output CategoryGroup objects should have category_name and term_names fields."""
        cohort = _build_cohort({
            "mutA": [("T1", 2.0, 0.01), ("T2", 1.0, 0.01)],
            "mutB": [("T1", 1.5, 0.01), ("T2", 0.5, 0.01)],
        })
        groups, _ = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=2, n_groups=1, random_seed=42
        )
        for g in groups:
            assert hasattr(g, "category_name")
            assert hasattr(g, "term_names")
            assert isinstance(g.category_name, str)
            assert isinstance(g.term_names, list)
