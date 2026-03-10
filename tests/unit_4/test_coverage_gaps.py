"""
Additional coverage tests for Unit 4 -- Unbiased Term Selection.

These tests cover behavioral gaps identified during coverage review against
the blueprint contracts. Each test addresses a specific gap not covered by
the primary test suite.

Synthetic Data Assumptions (module level):
  - DATA ASSUMPTION: Same conventions as the primary test suite -- GO term
    names are short uppercase English strings, NES in [-3, 3], FDR in [0, 1],
    go_id in GO:NNNNNNN format, nom_pval=0.01 and size=100 as placeholders.
"""

import pytest

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
# Helper: build CohortData from a compact specification (same as primary suite)
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
# Gap 1: category_name equals first element of sorted term_names
# ===========================================================================

class TestGroupLabelIsFirstTerm:
    """Verify that category_name is always term_names[0] (the highest mean abs
    NES term within the group), not just any member of the group."""

    def test_label_equals_first_term_in_group(self):
        """Contract 6 + 8 combined: the group label (category_name) must be
        the first element of term_names, since terms are sorted by mean abs NES
        descending and the label is the term with highest mean abs NES.

        DATA ASSUMPTION: 4 terms with distinct mean abs NES values in one
        group. The label must be the first (highest) term.
        """
        cohort = _build_cohort({
            "mutA": [
                ("HIGH_MEAN", 3.0, 0.01),
                ("MEDIUM_MEAN", 2.0, 0.01),
                ("LOW_MEAN", 1.0, 0.01),
                ("LOWEST_MEAN", 0.5, 0.01),
            ],
            "mutB": [
                ("HIGH_MEAN", 2.5, 0.01),
                ("MEDIUM_MEAN", 1.5, 0.01),
                ("LOW_MEAN", 0.8, 0.01),
                ("LOWEST_MEAN", 0.3, 0.01),
            ],
        })
        groups = cluster_terms(
            ["HIGH_MEAN", "MEDIUM_MEAN", "LOW_MEAN", "LOWEST_MEAN"],
            cohort, n_groups=1, random_seed=42,
        )
        assert len(groups) == 1
        assert groups[0].category_name == groups[0].term_names[0]
        assert groups[0].category_name == "HIGH_MEAN"

    def test_label_equals_first_term_in_each_group_multicluster(self):
        """For multiple groups, every group's category_name must equal its
        term_names[0].

        DATA ASSUMPTION: 6 terms across 3 mutants forming 3 clusters.
        """
        cohort = _build_cohort({
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
        term_names = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(term_names, cohort, n_groups=3, random_seed=42)
        for g in groups:
            assert g.category_name == g.term_names[0], (
                f"Group label '{g.category_name}' does not match first term "
                f"'{g.term_names[0]}' in group {g.term_names}"
            )


# ===========================================================================
# Gap 2: clustering_algorithm string mentions Ward
# ===========================================================================

class TestClusteringAlgorithmString:
    """Verify that stats.clustering_algorithm describes Ward linkage."""

    def test_clustering_algorithm_mentions_ward(self):
        """Contract 5: Clustering uses Ward linkage. The algorithm description
        in stats should mention 'Ward' so users know which algorithm was used.

        DATA ASSUMPTION: Standard cohort with enough terms for clustering.
        """
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
            cohort, fdr_threshold=0.05, top_n=4, n_groups=2, random_seed=42,
        )
        assert "Ward" in stats.clustering_algorithm or "ward" in stats.clustering_algorithm, (
            f"clustering_algorithm should mention Ward linkage, got: "
            f"'{stats.clustering_algorithm}'"
        )


# ===========================================================================
# Gap 3: pool_significant_terms returns a dict
# ===========================================================================

class TestPoolReturnType:
    """Verify pool_significant_terms returns a dict mapping term_name -> max abs NES."""

    def test_pool_returns_dict(self):
        """Contract 1-2: pool_significant_terms must return a dict.

        DATA ASSUMPTION: Simple cohort with one significant term.
        """
        cohort = _build_cohort({
            "mutA": [("TERM_A", 2.0, 0.01)],
            "mutB": [("TERM_A", 1.5, 0.02)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert isinstance(result, dict)
        # Keys are strings (term names), values are floats (max abs NES)
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, float) or isinstance(val, (int, float))


# ===========================================================================
# Gap 4: cluster_terms with n_groups > n_terms
# ===========================================================================

class TestClusterMoreGroupsThanTerms:
    """Verify behavior when n_groups exceeds the number of terms.

    The blueprint invariant says 'Number of groups <= n_groups'. When there
    are fewer terms than n_groups, each term should get its own group, and
    the total number of groups will be less than n_groups.
    """

    def test_n_groups_exceeds_n_terms(self):
        """When n_groups > len(term_names), each term becomes its own group
        and the number of groups equals len(term_names) <= n_groups.

        DATA ASSUMPTION: 2 terms but requesting 5 groups.
        """
        cohort = _build_cohort({
            "mutA": [("ALPHA", 2.0, 0.01), ("BETA", 1.0, 0.01)],
            "mutB": [("ALPHA", 1.5, 0.01), ("BETA", 0.5, 0.01)],
        })
        groups = cluster_terms(
            ["ALPHA", "BETA"], cohort, n_groups=5, random_seed=42,
        )
        # Should get at most 2 groups (one per term), not 5
        assert len(groups) <= 2
        assert len(groups) > 0
        # All terms assigned
        all_terms = []
        for g in groups:
            all_terms.extend(g.term_names)
        assert set(all_terms) == {"ALPHA", "BETA"}
        # No empty groups
        assert all(len(g.term_names) > 0 for g in groups)


# ===========================================================================
# Gap 5: remove_redundant_terms preserves score values
# ===========================================================================

class TestRedundancyPreservesScores:
    """Verify that remove_redundant_terms preserves the max abs NES values
    for surviving terms."""

    def test_surviving_terms_retain_scores(self):
        """Contract 3: After redundancy removal, surviving terms must retain
        their original max abs NES values.

        DATA ASSUMPTION: Two redundant terms (Jaccard > 0.5) plus one
        non-redundant. The higher-ranked one survives with its score intact.
        """
        ranked = {
            "POSITIVE REGULATION OF APOPTOSIS": 3.0,
            "REGULATION OF APOPTOSIS": 2.5,
            "RIBOSOME BIOGENESIS": 2.0,
        }
        # Jaccard("POSITIVE REGULATION OF APOPTOSIS", "REGULATION OF APOPTOSIS"):
        # {POSITIVE, REGULATION, OF, APOPTOSIS} vs {REGULATION, OF, APOPTOSIS}
        # intersection = 3, union = 4, Jaccard = 0.75 > 0.5
        result = remove_redundant_terms(ranked)
        assert result["POSITIVE REGULATION OF APOPTOSIS"] == pytest.approx(3.0)
        assert result["RIBOSOME BIOGENESIS"] == pytest.approx(2.0)
        assert "REGULATION OF APOPTOSIS" not in result


# ===========================================================================
# Gap 6: stats.n_clusters always equals n_groups parameter
# ===========================================================================

class TestStatsNClusterMatchesParameter:
    """Verify that stats.n_clusters always equals the n_groups parameter passed."""

    def test_n_clusters_equals_n_groups_param(self):
        """Contract 10: stats.n_clusters records the n_groups parameter.

        DATA ASSUMPTION: Cohort with 10 significant terms.
        """
        terms = [(f"UNIQUE TERM {i:02d}", 3.0 - i * 0.2, 0.01) for i in range(10)]
        cohort = _build_cohort({
            "mutA": terms,
            "mutB": [(t, nes - 0.1, fdr) for t, nes, fdr in terms],
        })

        for n_groups in [2, 3, 5]:
            _, stats = select_unbiased_terms(
                cohort, fdr_threshold=0.05, top_n=10, n_groups=n_groups, random_seed=42,
            )
            assert stats.n_clusters == n_groups, (
                f"stats.n_clusters={stats.n_clusters} does not match n_groups={n_groups}"
            )


# ===========================================================================
# Gap 7: Heavy deduplication leaves fewer terms than top_n
# ===========================================================================

class TestHeavyDeduplication:
    """Test pipeline when deduplication removes most terms."""

    def test_dedup_reduces_below_top_n(self):
        """Contract 3 + 4: When deduplication heavily reduces the candidate pool,
        terms_selected should reflect the smaller count, and all surviving
        terms are used.

        DATA ASSUMPTION: 6 significant terms where 4 are lexically redundant
        with the top 2, leaving only 2 after dedup. top_n=5 but only 2 survive.
        """
        cohort = _build_cohort({
            "mutA": [
                ("CELL CYCLE REGULATION", 3.0, 0.01),
                ("POSITIVE CELL CYCLE REGULATION", 2.8, 0.01),
                ("NEGATIVE CELL CYCLE REGULATION", 2.6, 0.01),
                ("DNA REPAIR MECHANISM", 2.0, 0.01),
                ("DNA REPAIR MECHANISM PATHWAY", 1.8, 0.01),
                ("RIBOSOME ASSEMBLY", 1.5, 0.01),
            ],
            "mutB": [
                ("CELL CYCLE REGULATION", 2.5, 0.01),
                ("POSITIVE CELL CYCLE REGULATION", 2.3, 0.01),
                ("NEGATIVE CELL CYCLE REGULATION", 2.1, 0.01),
                ("DNA REPAIR MECHANISM", 1.5, 0.01),
                ("DNA REPAIR MECHANISM PATHWAY", 1.3, 0.01),
                ("RIBOSOME ASSEMBLY", 1.0, 0.01),
            ],
        })
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42,
        )
        # 6 significant terms total
        assert stats.total_significant_terms == 6
        # After dedup: "CELL CYCLE REGULATION" survives (removes POSITIVE and NEGATIVE variants)
        # "DNA REPAIR MECHANISM" survives (removes "...PATHWAY" variant -- Jaccard:
        #   {DNA, REPAIR, MECHANISM} vs {DNA, REPAIR, MECHANISM, PATHWAY} = 3/4 = 0.75 > 0.5)
        # "RIBOSOME ASSEMBLY" has no redundancy
        # So terms_after_dedup = 3
        assert stats.terms_after_dedup <= stats.total_significant_terms
        assert stats.terms_after_dedup < 6
        # terms_selected <= top_n but also <= terms_after_dedup
        assert stats.terms_selected <= 5
        assert stats.terms_selected <= stats.terms_after_dedup
        # All groups non-empty
        assert all(len(g.term_names) > 0 for g in groups)
        total = sum(len(g.term_names) for g in groups)
        assert total == stats.terms_selected


# ===========================================================================
# Gap 8: select_top_n with top_n=1
# ===========================================================================

class TestSelectTopNEdge:
    """Test edge case for select_top_n with top_n=1."""

    def test_top_n_one(self):
        """Contract 4: Selecting top 1 from multiple terms returns only the
        highest-ranked term.

        DATA ASSUMPTION: 3 ranked terms, only the first should be selected.
        """
        ranked = {
            "BEST": 5.0,
            "SECOND": 3.0,
            "THIRD": 1.0,
        }
        result = select_top_n(ranked, top_n=1)
        assert result == ["BEST"]


# ===========================================================================
# Gap 9: pool_significant_terms with term absent from some mutants
# ===========================================================================

class TestPoolWithSparseData:
    """Test pool_significant_terms when terms are not present in all mutants."""

    def test_max_abs_nes_considers_only_present_mutants(self):
        """Contract 2: Max abs NES is computed across all mutants where the
        term has a record. If a term is absent from a mutant, that mutant
        simply does not contribute to the max (it is not treated as NES=0).

        DATA ASSUMPTION: TERM_A has NES=0.5 in mutA (significant) and is
        absent from mutB. The max abs NES should be 0.5, not 0.0.
        """
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 0.5, 0.01),
                ("TERM_B", 2.0, 0.01),
            ],
            "mutB": [
                # TERM_A absent from mutB entirely
                ("TERM_B", 1.0, 0.01),
            ],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "TERM_A" in result
        # max abs NES of TERM_A = 0.5 (only mutA has it)
        assert result["TERM_A"] == pytest.approx(0.5)
        # TERM_B max abs NES = 2.0 (from mutA)
        assert result["TERM_B"] == pytest.approx(2.0)
        # TERM_B should rank higher
        keys = list(result.keys())
        assert keys.index("TERM_B") < keys.index("TERM_A")


# ===========================================================================
# Gap 10: Verify groups from select_unbiased_terms contain no duplicate terms
# ===========================================================================

class TestNoDuplicateTermsAcrossGroups:
    """Verify that no term appears in more than one group in the output."""

    def test_no_term_in_multiple_groups(self):
        """Implied invariant: Each term should appear in exactly one group.

        DATA ASSUMPTION: 8 significant terms across 3 mutants, grouped into 3.
        """
        terms = [(f"DISTINCT TERM {i}", 3.0 - i * 0.3, 0.01) for i in range(8)]
        cohort = _build_cohort({
            "mutA": terms,
            "mutB": [(t, nes - 0.2, fdr) for t, nes, fdr in terms],
            "mutC": [(t, nes + 0.1, fdr) for t, nes, fdr in terms],
        })
        groups, _ = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=8, n_groups=3, random_seed=42,
        )
        all_terms = []
        for g in groups:
            all_terms.extend(g.term_names)
        # No duplicates
        assert len(all_terms) == len(set(all_terms)), (
            f"Duplicate terms found across groups: {all_terms}"
        )


# ===========================================================================
# Gap 11: Verify end-to-end that n_groups <= top_n invariant holds at output
# ===========================================================================

class TestInvariantsAtOutput:
    """Verify post-condition invariants on the output of select_unbiased_terms."""

    def test_n_groups_lte_top_n_at_output(self):
        """Invariant: n_groups <= top_n. The number of output groups should
        not exceed the top_n parameter.

        DATA ASSUMPTION: top_n=5, n_groups=3. Output groups <= 3 <= 5.
        """
        terms = [(f"TERM {i}", 3.0 - i * 0.2, 0.01) for i in range(10)]
        cohort = _build_cohort({
            "mutA": terms,
            "mutB": [(t, nes - 0.1, fdr) for t, nes, fdr in terms],
        })
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=3, random_seed=42,
        )
        assert len(groups) <= 3
        total = sum(len(g.term_names) for g in groups)
        assert total <= 5
        assert len(groups) <= total  # at least 1 term per group


# ===========================================================================
# Gap 12: Dedup reduces terms below n_groups, causing n_clusters adjustment
# ===========================================================================

class TestDedupReducesBelowNGroups:
    """Verify pipeline behavior when dedup leaves fewer terms than n_groups.

    The implementation adjusts actual_n_groups = min(n_groups, terms_selected).
    stats.n_clusters should reflect the adjusted count, not the original n_groups.
    """

    def test_stats_n_clusters_adjusted_when_dedup_reduces_below_n_groups(self):
        """When dedup reduces terms below n_groups, stats.n_clusters should
        equal the actual number of groups used, which is min(n_groups, terms_selected).

        DATA ASSUMPTION: 4 significant terms where 3 are lexically redundant,
        leaving only 2 after dedup. n_groups=3 but only 2 terms survive.
        stats.n_clusters should be 2, not 3.
        """
        cohort = _build_cohort({
            "mutA": [
                ("CELL CYCLE REGULATION", 3.0, 0.01),
                ("POSITIVE CELL CYCLE REGULATION", 2.8, 0.01),
                ("NEGATIVE CELL CYCLE REGULATION", 2.6, 0.01),
                ("RIBOSOME ASSEMBLY", 2.0, 0.01),
            ],
            "mutB": [
                ("CELL CYCLE REGULATION", 2.5, 0.01),
                ("POSITIVE CELL CYCLE REGULATION", 2.3, 0.01),
                ("NEGATIVE CELL CYCLE REGULATION", 2.1, 0.01),
                ("RIBOSOME ASSEMBLY", 1.5, 0.01),
            ],
        })
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=3, random_seed=42,
        )
        # After dedup: "CELL CYCLE REGULATION" survives (others redundant),
        # "RIBOSOME ASSEMBLY" survives. So terms_selected = 2.
        assert stats.terms_selected == 2
        # n_clusters should be adjusted to min(3, 2) = 2
        assert stats.n_clusters == 2, (
            f"Expected stats.n_clusters=2 (adjusted), got {stats.n_clusters}"
        )
        assert len(groups) <= 2
        assert all(len(g.term_names) > 0 for g in groups)


# ===========================================================================
# Gap 13: cluster_terms with exactly one term (special code path)
# ===========================================================================

class TestClusterTermsSingleTerm:
    """Verify cluster_terms handles a single input term correctly.

    The implementation has a special code path for n_terms == 1 that bypasses
    scipy clustering entirely and returns a single group.
    """

    def test_single_term_returns_single_group(self):
        """When only one term is provided, cluster_terms should return a single
        group containing that term, with category_name set to the term.

        DATA ASSUMPTION: Single term with NES data in two mutants.
        """
        cohort = _build_cohort({
            "mutA": [("ONLY_TERM", 2.5, 0.01)],
            "mutB": [("ONLY_TERM", 1.5, 0.01)],
        })
        groups = cluster_terms(
            ["ONLY_TERM"], cohort, n_groups=1, random_seed=42,
        )
        assert len(groups) == 1
        assert groups[0].category_name == "ONLY_TERM"
        assert groups[0].term_names == ["ONLY_TERM"]

    def test_single_term_via_pipeline(self):
        """End-to-end: when only one significant term exists and n_groups=1,
        the pipeline should succeed and produce one group with one term.

        DATA ASSUMPTION: Only one significant term in the cohort.
        """
        cohort = _build_cohort({
            "mutA": [
                ("SOLE_SURVIVOR", 2.0, 0.01),
                ("NOT_SIG", 0.5, 0.50),
            ],
            "mutB": [
                ("SOLE_SURVIVOR", 1.5, 0.02),
                ("NOT_SIG", 0.3, 0.60),
            ],
        })
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=1, random_seed=42,
        )
        assert stats.total_significant_terms == 1
        assert stats.terms_selected == 1
        assert len(groups) == 1
        assert groups[0].term_names == ["SOLE_SURVIVOR"]
        assert groups[0].category_name == "SOLE_SURVIVOR"
