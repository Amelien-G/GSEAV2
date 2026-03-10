"""
Comprehensive test suite for Unit 4 -- Unbiased Term Selection.

Tests the six-step data-driven term selection and grouping pipeline for
Figure 2 (the unbiased figure):
  Step 1: Pool significant GO terms (FDR < threshold in any mutant)
  Step 2: Rank by max absolute NES across mutants
  Step 3: Remove lexically redundant terms (Jaccard > 0.5)
  Step 4: Select top N terms
  Step 5: Cluster via Ward linkage hierarchical agglomerative clustering
  Step 6: Auto-label each cluster group

Synthetic Data Assumptions:
  - GO term names: short uppercase English strings (e.g., "OXIDATIVE PHOSPHORYLATION")
  - NES values: range [-3.0, 3.0], typical GSEA normalized enrichment scores
  - FDR values: range [0.0, 1.0]; FDR < 0.05 is significant by default
  - Mutant IDs: short alphanumeric strings like "mutA", "mutB"
  - go_id: sequential GO:NNNNNNN placeholders (unit operates on term_name, not go_id)
  - nom_pval: 0.01 placeholder (unused by this unit)
  - size: 100 placeholder (unused by this unit)
  - Redundant terms: crafted with >50% Jaccard word overlap
  - Clustering data: NES profiles designed for deterministic Ward linkage clusters
"""

import pytest
from dataclasses import fields as dataclass_fields

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
# Helpers
# ---------------------------------------------------------------------------

def _make_record(term_name, nes, fdr, go_id="GO:0000001"):
    """Create a TermRecord with placeholder values for unused fields."""
    return TermRecord(
        term_name=term_name,
        go_id=go_id,
        nes=nes,
        fdr=fdr,
        nom_pval=0.01,
        size=100,
    )


def _build_cohort(mutant_data: dict[str, list[tuple[str, float, float]]]) -> CohortData:
    """Build CohortData from mutant_id -> list of (term_name, nes, fdr).

    Assigns sequential go_ids per unique term.
    """
    all_term_names: set[str] = set()
    all_go_ids: set[str] = set()
    term_go_map: dict[str, str] = {}
    go_counter = 1

    for terms in mutant_data.values():
        for term_name, _, _ in terms:
            all_term_names.add(term_name)
            if term_name not in term_go_map:
                term_go_map[term_name] = f"GO:{go_counter:07d}"
                go_counter += 1

    profiles: dict[str, MutantProfile] = {}
    for mutant_id, terms in mutant_data.items():
        records: dict[str, TermRecord] = {}
        for term_name, nes, fdr in terms:
            go_id = term_go_map[term_name]
            all_go_ids.add(go_id)
            records[term_name] = _make_record(term_name, nes, fdr, go_id)
        profiles[mutant_id] = MutantProfile(mutant_id=mutant_id, records=records)

    return CohortData(
        mutant_ids=sorted(mutant_data.keys()),
        profiles=profiles,
        all_term_names=all_term_names,
        all_go_ids=all_go_ids,
    )


def _mean_abs_nes(term_name: str, cohort: CohortData) -> float:
    """Compute mean absolute NES for a term across all mutants. Missing = 0."""
    total = 0.0
    for mid in cohort.mutant_ids:
        rec = cohort.profiles[mid].records.get(term_name)
        if rec is not None:
            total += abs(rec.nes)
    return total / len(cohort.mutant_ids)


# ===========================================================================
# UnbiasedSelectionStats dataclass
# ===========================================================================

class TestUnbiasedSelectionStats:
    """Verify the UnbiasedSelectionStats dataclass structure."""

    def test_has_all_required_fields(self):
        """The dataclass must contain every field specified in the blueprint."""
        field_names = {f.name for f in dataclass_fields(UnbiasedSelectionStats)}
        expected = {
            "total_significant_terms",
            "terms_after_dedup",
            "terms_selected",
            "n_clusters",
            "random_seed",
            "clustering_algorithm",
        }
        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_can_be_instantiated(self):
        """The dataclass can be constructed with all required fields."""
        stats = UnbiasedSelectionStats(
            total_significant_terms=50,
            terms_after_dedup=35,
            terms_selected=20,
            n_clusters=4,
            random_seed=42,
            clustering_algorithm="scipy.cluster.hierarchy (Ward linkage)",
        )
        assert stats.total_significant_terms == 50
        assert stats.random_seed == 42


# ===========================================================================
# pool_significant_terms (Steps 1-2)
# ===========================================================================

class TestPoolSignificantTerms:
    """Test Step 1 (pool FDR-significant terms) and Step 2 (rank by max abs NES)."""

    def test_only_fdr_significant_terms_are_pooled(self):
        """Contract 1: Only terms with FDR < threshold in at least one mutant are included."""
        cohort = _build_cohort({
            "mutA": [
                ("SIG_TERM", 2.0, 0.01),
                ("NOT_SIG", 0.5, 0.10),
            ],
            "mutB": [
                ("SIG_TERM", 1.0, 0.80),
                ("NOT_SIG", 0.3, 0.90),
            ],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "SIG_TERM" in result
        assert "NOT_SIG" not in result

    def test_term_significant_in_one_mutant_only_is_pooled(self):
        """Contract 1: A term passing FDR in just one mutant is included."""
        cohort = _build_cohort({
            "mutA": [("TERM_X", 1.5, 0.03)],
            "mutB": [("TERM_X", 0.5, 0.50)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "TERM_X" in result

    def test_fdr_exactly_at_threshold_is_excluded(self):
        """Contract 1: FDR < threshold is strict; FDR == threshold is NOT significant."""
        cohort = _build_cohort({
            "mutA": [("BOUNDARY", 2.0, 0.05)],
            "mutB": [("BOUNDARY", 1.5, 0.05)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "BOUNDARY" not in result

    def test_fdr_just_below_threshold_is_included(self):
        """Contract 1: FDR just below threshold qualifies."""
        cohort = _build_cohort({
            "mutA": [("ALMOST", 2.0, 0.0499)],
            "mutB": [("ALMOST", 1.0, 0.90)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "ALMOST" in result

    def test_ranking_by_max_absolute_nes_descending(self):
        """Contract 2: Terms ranked by max absolute NES descending."""
        cohort = _build_cohort({
            "mutA": [
                ("HIGH", 3.0, 0.01),
                ("MED", -2.0, 0.01),
                ("LOW", 0.5, 0.01),
            ],
            "mutB": [
                ("HIGH", 1.0, 0.01),
                ("MED", 1.0, 0.01),
                ("LOW", -1.0, 0.01),
            ],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        keys = list(result.keys())
        assert keys == ["HIGH", "MED", "LOW"]
        assert result["HIGH"] == pytest.approx(3.0)
        assert result["MED"] == pytest.approx(2.0)
        assert result["LOW"] == pytest.approx(1.0)

    def test_max_abs_nes_uses_absolute_value_of_negative_nes(self):
        """Contract 2: Negative NES contributes absolute value."""
        cohort = _build_cohort({
            "mutA": [("NEG_TERM", -3.5, 0.01)],
            "mutB": [("NEG_TERM", 1.0, 0.01)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert result["NEG_TERM"] == pytest.approx(3.5)

    def test_tie_broken_alphabetically_by_term_name(self):
        """Contract 2: Ties in max abs NES broken alphabetically."""
        cohort = _build_cohort({
            "mutA": [
                ("ZEBRA", 2.0, 0.01),
                ("ALPHA", -2.0, 0.01),
                ("MIDDLE", 2.0, 0.01),
            ],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        keys = list(result.keys())
        # All have max abs NES = 2.0, sorted alphabetically
        assert keys == ["ALPHA", "MIDDLE", "ZEBRA"]

    def test_max_abs_nes_computed_across_all_mutants(self):
        """Contract 2: Max abs NES considers all mutants where term has a record."""
        cohort = _build_cohort({
            "mutA": [("TERM", 0.5, 0.01)],
            "mutB": [("TERM", 1.0, 0.02)],
            "mutC": [("TERM", -2.5, 0.03)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert result["TERM"] == pytest.approx(2.5)

    def test_max_abs_nes_includes_non_significant_mutant_records(self):
        """Contract 2: Once a term is pooled, max abs NES uses all mutant records,
        not just the significant ones."""
        cohort = _build_cohort({
            "mutA": [("TERM", 1.0, 0.01)],  # significant -> pools term
            "mutB": [("TERM", 4.0, 0.90)],  # not significant, but has higher NES
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "TERM" in result
        assert result["TERM"] == pytest.approx(4.0)

    def test_empty_pool_when_nothing_significant(self):
        """Contract 1: If no terms pass FDR threshold, pool is empty."""
        cohort = _build_cohort({
            "mutA": [("TERM_A", 1.0, 0.50)],
            "mutB": [("TERM_B", 2.0, 0.60)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert len(result) == 0

    def test_returns_dict_mapping_name_to_float(self):
        """Signature: returns dict[str, float]."""
        cohort = _build_cohort({
            "mutA": [("TERM", 2.0, 0.01)],
            "mutB": [("TERM", 1.0, 0.01)],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, (int, float))

    def test_term_absent_from_some_mutants_still_pooled(self):
        """Contract 1: A term present in only one mutant is pooled if FDR < threshold."""
        cohort = _build_cohort({
            "mutA": [("ONLY_IN_A", 2.0, 0.01)],
            "mutB": [],
        })
        result = pool_significant_terms(cohort, fdr_threshold=0.05)
        assert "ONLY_IN_A" in result
        assert result["ONLY_IN_A"] == pytest.approx(2.0)


# ===========================================================================
# remove_redundant_terms (Step 3)
# ===========================================================================

class TestRemoveRedundantTerms:
    """Test Step 3: lexical redundancy removal via Jaccard similarity."""

    def test_no_overlap_all_retained(self):
        """Contract 3: Terms with no word overlap are all retained."""
        ranked = {
            "OXIDATIVE PHOSPHORYLATION": 3.0,
            "RIBOSOME BIOGENESIS": 2.0,
            "GPCR SIGNALING": 1.0,
        }
        result = remove_redundant_terms(ranked)
        assert len(result) == 3

    def test_high_jaccard_removes_lower_ranked(self):
        """Contract 3: Jaccard > 0.5 removes the lower-ranked term."""
        ranked = {
            "POSITIVE REGULATION OF CELL DEATH": 3.0,
            "REGULATION OF CELL DEATH": 2.5,
        }
        # Jaccard = 4/5 = 0.8 > 0.5
        result = remove_redundant_terms(ranked)
        assert "POSITIVE REGULATION OF CELL DEATH" in result
        assert "REGULATION OF CELL DEATH" not in result

    def test_jaccard_exactly_half_is_not_redundant(self):
        """Contract 3: Jaccard = 0.5 exactly does NOT trigger removal (strict >0.5)."""
        # {A, B, C} vs {A, B, D} -> intersection=2, union=4 -> Jaccard=0.5
        ranked = {"A B C": 3.0, "A B D": 2.0}
        result = remove_redundant_terms(ranked)
        assert len(result) == 2

    def test_jaccard_just_above_half_triggers_removal(self):
        """Contract 3: Jaccard > 0.5 triggers removal."""
        # {A, B, C, D} vs {A, B, C, E} -> intersection=3, union=5 -> Jaccard=0.6
        ranked = {"A B C D": 3.0, "A B C E": 2.0}
        result = remove_redundant_terms(ranked)
        assert len(result) == 1
        assert "A B C D" in result

    def test_higher_ranked_term_always_survives(self):
        """Contract 3: Terms processed in rank order; higher-ranked survives."""
        ranked = {
            "REGULATION OF APOPTOTIC PROCESS": 3.0,
            "POSITIVE REGULATION OF APOPTOTIC PROCESS": 2.0,
        }
        # Jaccard = 4/5 = 0.8
        result = remove_redundant_terms(ranked)
        assert "REGULATION OF APOPTOTIC PROCESS" in result
        assert "POSITIVE REGULATION OF APOPTOTIC PROCESS" not in result

    def test_chained_redundancy_checked_against_survivors(self):
        """Contract 3: Each term is checked against all surviving terms, not removed ones."""
        # All three have Jaccard 3/5=0.6 with each other
        ranked = {
            "X Y Z Q": 3.0,
            "X Y Z R": 2.0,
            "X Y Z S": 1.0,
        }
        result = remove_redundant_terms(ranked)
        assert result == {"X Y Z Q": 3.0}

    def test_non_redundant_term_survives_alongside_redundant_pair(self):
        """Contract 3: Unrelated terms are unaffected by redundancy removal."""
        ranked = {
            "REGULATION OF CELL MIGRATION": 3.0,
            "POSITIVE REGULATION OF CELL MIGRATION": 2.5,
            "RIBOSOME ASSEMBLY": 2.0,
        }
        result = remove_redundant_terms(ranked)
        assert "REGULATION OF CELL MIGRATION" in result
        assert "POSITIVE REGULATION OF CELL MIGRATION" not in result
        assert "RIBOSOME ASSEMBLY" in result

    def test_empty_input_returns_empty(self):
        """Edge case: empty dict input returns empty dict."""
        assert remove_redundant_terms({}) == {}

    def test_single_term_returned_unchanged(self):
        """Edge case: single term has no redundancy partner."""
        ranked = {"LONE TERM": 2.5}
        result = remove_redundant_terms(ranked)
        assert result == {"LONE TERM": 2.5}

    def test_preserves_rank_order_in_output(self):
        """Contract 3: Output dict preserves the original descending rank order."""
        ranked = {"ALPHA": 4.0, "BETA": 3.0, "GAMMA": 2.0, "DELTA": 1.0}
        result = remove_redundant_terms(ranked)
        assert list(result.keys()) == ["ALPHA", "BETA", "GAMMA", "DELTA"]

    def test_preserves_score_values(self):
        """Contract 3: Surviving terms retain their original max abs NES values."""
        ranked = {
            "POSITIVE REGULATION OF APOPTOSIS": 3.0,
            "REGULATION OF APOPTOSIS": 2.5,
            "RIBOSOME BIOGENESIS": 2.0,
        }
        result = remove_redundant_terms(ranked)
        assert result["POSITIVE REGULATION OF APOPTOSIS"] == pytest.approx(3.0)
        assert result["RIBOSOME BIOGENESIS"] == pytest.approx(2.0)


# ===========================================================================
# select_top_n (Step 4)
# ===========================================================================

class TestSelectTopN:
    """Test Step 4: selecting top N terms from deduplicated ranked list."""

    def test_selects_top_n_from_larger_list(self):
        """Contract 4: Selects first N terms from ranked dict."""
        ranked = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}
        result = select_top_n(ranked, top_n=3)
        assert result == ["A", "B", "C"]

    def test_fewer_than_n_available_returns_all(self):
        """Contract 4: If fewer than N terms remain, all are used."""
        ranked = {"A": 3.0, "B": 2.0}
        result = select_top_n(ranked, top_n=5)
        assert result == ["A", "B"]

    def test_exactly_n_available_returns_all(self):
        """Contract 4: Exactly N terms returns all N."""
        ranked = {"A": 3.0, "B": 2.0, "C": 1.0}
        result = select_top_n(ranked, top_n=3)
        assert result == ["A", "B", "C"]

    def test_top_n_one_returns_single_term(self):
        """Edge case: top_n=1 returns only the highest-ranked term."""
        ranked = {"BEST": 5.0, "SECOND": 3.0, "THIRD": 1.0}
        result = select_top_n(ranked, top_n=1)
        assert result == ["BEST"]

    def test_returns_list_of_strings(self):
        """Signature: returns list[str], not list of tuples."""
        ranked = {"TERM": 2.0}
        result = select_top_n(ranked, top_n=1)
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_preserves_rank_order(self):
        """Contract 4: Returned list preserves descending rank order."""
        ranked = {"FOURTH": 4.0, "THIRD": 3.0, "SECOND": 2.0, "FIRST": 1.0}
        result = select_top_n(ranked, top_n=4)
        assert result == ["FOURTH", "THIRD", "SECOND", "FIRST"]


# ===========================================================================
# cluster_terms (Steps 5-6)
# ===========================================================================

class TestClusterTerms:
    """Test Steps 5-6: Ward linkage clustering and auto-labeling."""

    @pytest.fixture
    def clustering_cohort(self):
        """Cohort with 6 terms designed to form 3 natural clusters.

        Cluster A: TERM_A (+2, +2, 0) and TERM_B (+1.8, +1.9, 0)
        Cluster B: TERM_C (0, -2, -2) and TERM_D (0, -1.8, -1.9)
        Cluster C: TERM_E (+1, -1, +1) and TERM_F (+1.1, -0.9, +1.1)
        """
        return _build_cohort({
            "mutA": [
                ("TERM_A", 2.0, 0.01), ("TERM_B", 1.8, 0.01),
                ("TERM_C", 0.0, 0.01), ("TERM_D", 0.0, 0.01),
                ("TERM_E", 1.0, 0.01), ("TERM_F", 1.1, 0.01),
            ],
            "mutB": [
                ("TERM_A", 2.0, 0.01), ("TERM_B", 1.9, 0.01),
                ("TERM_C", -2.0, 0.01), ("TERM_D", -1.8, 0.01),
                ("TERM_E", -1.0, 0.01), ("TERM_F", -0.9, 0.01),
            ],
            "mutC": [
                ("TERM_A", 0.0, 0.01), ("TERM_B", 0.0, 0.01),
                ("TERM_C", -2.0, 0.01), ("TERM_D", -1.9, 0.01),
                ("TERM_E", 1.0, 0.01), ("TERM_F", 1.1, 0.01),
            ],
        })

    def test_returns_list_of_category_groups(self, clustering_cohort):
        """Signature: returns list[CategoryGroup]."""
        terms = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(terms, clustering_cohort, n_groups=3, random_seed=42)
        assert isinstance(groups, list)
        assert all(isinstance(g, CategoryGroup) for g in groups)

    def test_correct_number_of_groups(self, clustering_cohort):
        """Contract 5: Number of groups equals n_groups when enough terms."""
        terms = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(terms, clustering_cohort, n_groups=3, random_seed=42)
        assert len(groups) == 3

    def test_all_terms_assigned_exactly_once(self, clustering_cohort):
        """Invariant: Every input term appears in exactly one group."""
        terms = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(terms, clustering_cohort, n_groups=3, random_seed=42)
        all_grouped = []
        for g in groups:
            all_grouped.extend(g.term_names)
        assert set(all_grouped) == set(terms)
        assert len(all_grouped) == len(terms)

    def test_no_empty_groups(self, clustering_cohort):
        """Invariant: all(len(g.term_names) > 0 for g in groups)."""
        terms = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(terms, clustering_cohort, n_groups=3, random_seed=42)
        assert all(len(g.term_names) > 0 for g in groups)

    def test_group_label_is_highest_mean_abs_nes_term(self, clustering_cohort):
        """Contract 6: Each group labeled with term having highest mean abs NES."""
        terms = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(terms, clustering_cohort, n_groups=3, random_seed=42)
        for g in groups:
            assert g.category_name in g.term_names
            # The label should be the term with highest mean abs NES in the group
            label_mean = _mean_abs_nes(g.category_name, clustering_cohort)
            for t in g.term_names:
                assert _mean_abs_nes(t, clustering_cohort) <= label_mean + 1e-9

    def test_label_equals_first_element_of_term_names(self, clustering_cohort):
        """Contracts 6+8: Since terms sorted by mean abs NES desc, label == term_names[0]."""
        terms = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(terms, clustering_cohort, n_groups=3, random_seed=42)
        for g in groups:
            assert g.category_name == g.term_names[0]

    def test_terms_within_group_sorted_by_mean_abs_nes_descending(self, clustering_cohort):
        """Contract 8: Within each group, terms sorted by mean abs NES descending."""
        terms = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(terms, clustering_cohort, n_groups=3, random_seed=42)
        for g in groups:
            means = [_mean_abs_nes(t, clustering_cohort) for t in g.term_names]
            for i in range(len(means) - 1):
                assert means[i] >= means[i + 1] - 1e-9, (
                    f"Group '{g.category_name}' not sorted: {list(zip(g.term_names, means))}"
                )

    def test_missing_nes_treated_as_zero(self):
        """Contract 5: Missing NES values (term absent from a mutant) treated as 0.0."""
        cohort = _build_cohort({
            "mutA": [("SPARSE", 2.0, 0.01), ("FULL", 1.0, 0.01)],
            "mutB": [("FULL", -1.5, 0.01)],  # SPARSE absent
        })
        # With 1 group, both terms go in one group
        groups = cluster_terms(["SPARSE", "FULL"], cohort, n_groups=1, random_seed=42)
        assert len(groups) == 1
        assert set(groups[0].term_names) == {"SPARSE", "FULL"}

    def test_groups_ordered_by_original_rank_of_highest_member(self, clustering_cohort):
        """Contract 9: Groups sorted by rank position of highest-ranked member."""
        terms = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        groups = cluster_terms(terms, clustering_cohort, n_groups=3, random_seed=42)

        def best_rank(group):
            return min(terms.index(t) for t in group.term_names)

        ranks = [best_rank(g) for g in groups]
        assert ranks == sorted(ranks), f"Groups not ordered by best rank: {ranks}"

    def test_deterministic_output_same_seed(self, clustering_cohort):
        """Contract 11: Same input and seed produces identical output."""
        terms = ["TERM_A", "TERM_B", "TERM_C", "TERM_D", "TERM_E", "TERM_F"]
        g1 = cluster_terms(terms, clustering_cohort, n_groups=3, random_seed=42)
        g2 = cluster_terms(terms, clustering_cohort, n_groups=3, random_seed=42)
        assert len(g1) == len(g2)
        for a, b in zip(g1, g2):
            assert a.category_name == b.category_name
            assert a.term_names == b.term_names

    def test_single_group_contains_all_terms(self):
        """Edge case: n_groups=1 puts all terms in one group."""
        cohort = _build_cohort({
            "mutA": [("X", 2.0, 0.01), ("Y", 1.5, 0.01), ("Z", 1.0, 0.01)],
        })
        groups = cluster_terms(["X", "Y", "Z"], cohort, n_groups=1, random_seed=42)
        assert len(groups) == 1
        assert set(groups[0].term_names) == {"X", "Y", "Z"}

    def test_n_groups_equals_n_terms_each_gets_own_group(self):
        """Edge case: n_groups == len(terms) -> each term is its own group."""
        cohort = _build_cohort({
            "mutA": [("X", 2.0, 0.01), ("Y", 1.5, 0.01), ("Z", 1.0, 0.01)],
        })
        groups = cluster_terms(["X", "Y", "Z"], cohort, n_groups=3, random_seed=42)
        assert len(groups) == 3
        assert all(len(g.term_names) == 1 for g in groups)

    def test_n_groups_exceeds_n_terms_yields_fewer_groups(self):
        """Edge case: n_groups > len(terms) -> each term gets its own group,
        resulting in fewer groups than n_groups."""
        cohort = _build_cohort({
            "mutA": [("ALPHA", 2.0, 0.01), ("BETA", 1.0, 0.01)],
            "mutB": [("ALPHA", 1.5, 0.01), ("BETA", 0.5, 0.01)],
        })
        groups = cluster_terms(["ALPHA", "BETA"], cohort, n_groups=5, random_seed=42)
        assert len(groups) <= 2
        assert len(groups) > 0
        all_terms = [t for g in groups for t in g.term_names]
        assert set(all_terms) == {"ALPHA", "BETA"}

    def test_category_name_is_string(self):
        """Signature: category_name is a string."""
        cohort = _build_cohort({
            "mutA": [("T1", 2.0, 0.01), ("T2", 1.0, 0.01)],
        })
        groups = cluster_terms(["T1", "T2"], cohort, n_groups=1, random_seed=42)
        for g in groups:
            assert isinstance(g.category_name, str)


# ===========================================================================
# select_unbiased_terms (top-level pipeline)
# ===========================================================================

class TestSelectUnbiasedTerms:
    """Test the top-level entry point that orchestrates the full pipeline."""

    @pytest.fixture
    def large_cohort(self):
        """Cohort with 25 distinct significant terms across 3 mutants."""
        terms = [(f"UNIQUE TERM {i:02d}", 3.0 - i * 0.1, 0.01) for i in range(25)]
        data = {}
        for mid in ["mutA", "mutB", "mutC"]:
            data[mid] = [(t, nes + (hash(mid + t) % 10) * 0.01, fdr)
                         for t, nes, fdr in terms]
        return _build_cohort(data)

    def test_returns_tuple_of_groups_and_stats(self, large_cohort):
        """Signature: returns (list[CategoryGroup], UnbiasedSelectionStats)."""
        result = select_unbiased_terms(large_cohort)
        assert isinstance(result, tuple)
        assert len(result) == 2
        groups, stats = result
        assert isinstance(groups, list)
        assert all(isinstance(g, CategoryGroup) for g in groups)
        assert isinstance(stats, UnbiasedSelectionStats)

    def test_default_parameters(self, large_cohort):
        """Defaults: fdr_threshold=0.05, top_n=20, n_groups=4, random_seed=42."""
        groups, stats = select_unbiased_terms(large_cohort)
        total = sum(len(g.term_names) for g in groups)
        assert total <= 20
        assert len(groups) <= 4
        assert stats.random_seed == 42

    def test_stats_random_seed_matches_parameter(self, large_cohort):
        """Invariant: stats.random_seed == random_seed."""
        _, stats = select_unbiased_terms(large_cohort, random_seed=123)
        assert stats.random_seed == 123

    def test_stats_n_clusters_matches_parameter(self, large_cohort):
        """Contract 10: stats.n_clusters records the n_groups parameter."""
        for n in [2, 3, 5]:
            _, stats = select_unbiased_terms(large_cohort, top_n=20, n_groups=n)
            assert stats.n_clusters == n

    def test_stats_total_significant_terms(self):
        """Contract 10: stats records total significant terms after step 1."""
        cohort = _build_cohort({
            "mutA": [
                ("SIG_A", 2.0, 0.01),
                ("SIG_B", 1.5, 0.02),
                ("NOT_SIG", 1.0, 0.50),
            ],
            "mutB": [
                ("SIG_A", 1.0, 0.03),
                ("SIG_C", 0.5, 0.04),
                ("NOT_SIG", 0.8, 0.60),
            ],
        })
        _, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=2, random_seed=42
        )
        assert stats.total_significant_terms == 3  # SIG_A, SIG_B, SIG_C

    def test_stats_terms_after_dedup(self):
        """Contract 10: stats records terms remaining after deduplication."""
        cohort = _build_cohort({
            "mutA": [
                ("REGULATION OF CELL DEATH", 3.0, 0.01),
                ("POSITIVE REGULATION OF CELL DEATH", 2.5, 0.01),
                ("RIBOSOME BIOGENESIS", 2.0, 0.01),
            ],
            "mutB": [
                ("REGULATION OF CELL DEATH", 2.5, 0.01),
                ("POSITIVE REGULATION OF CELL DEATH", 2.0, 0.01),
                ("RIBOSOME BIOGENESIS", 1.5, 0.01),
            ],
        })
        _, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=2, random_seed=42
        )
        assert stats.total_significant_terms == 3
        # "POSITIVE REGULATION..." removed as redundant with "REGULATION..."
        assert stats.terms_after_dedup == 2

    def test_stats_terms_selected_capped_by_top_n(self, large_cohort):
        """Contract 10: stats.terms_selected <= top_n."""
        _, stats = select_unbiased_terms(large_cohort, top_n=10)
        assert stats.terms_selected <= 10
        assert stats.terms_selected > 0

    def test_stats_clustering_algorithm_mentions_ward(self, large_cohort):
        """Contract 5: The clustering algorithm string mentions Ward linkage."""
        _, stats = select_unbiased_terms(large_cohort)
        assert isinstance(stats.clustering_algorithm, str)
        assert "ward" in stats.clustering_algorithm.lower()

    def test_post_condition_no_empty_groups(self, large_cohort):
        """Invariant: all(len(g.term_names) > 0 for g in groups)."""
        groups, _ = select_unbiased_terms(large_cohort)
        assert all(len(g.term_names) > 0 for g in groups)

    def test_post_condition_total_terms_not_exceed_top_n(self, large_cohort):
        """Invariant: sum(len(g.term_names) for g in groups) <= top_n."""
        groups, _ = select_unbiased_terms(large_cohort, top_n=15)
        assert sum(len(g.term_names) for g in groups) <= 15

    def test_post_condition_groups_not_exceed_n_groups(self, large_cohort):
        """Invariant: len(groups) <= n_groups."""
        groups, _ = select_unbiased_terms(large_cohort, n_groups=3)
        assert len(groups) <= 3

    def test_deterministic_with_same_seed(self, large_cohort):
        """Contract 11: Same input + seed = identical output."""
        g1, s1 = select_unbiased_terms(large_cohort, random_seed=42)
        g2, s2 = select_unbiased_terms(large_cohort, random_seed=42)
        assert len(g1) == len(g2)
        for a, b in zip(g1, g2):
            assert a.category_name == b.category_name
            assert a.term_names == b.term_names
        assert s1.total_significant_terms == s2.total_significant_terms
        assert s1.terms_after_dedup == s2.terms_after_dedup
        assert s1.terms_selected == s2.terms_selected

    def test_no_duplicate_terms_across_groups(self, large_cohort):
        """Implied invariant: no term appears in more than one group."""
        groups, _ = select_unbiased_terms(large_cohort, top_n=15, n_groups=3)
        all_terms = [t for g in groups for t in g.term_names]
        assert len(all_terms) == len(set(all_terms))

    def test_custom_fdr_threshold_includes_more_terms(self):
        """Contract 1: More permissive threshold includes more terms."""
        cohort = _build_cohort({
            "mutA": [
                ("STRICT_SIG", 2.0, 0.01),
                ("PERMISSIVE_SIG", 1.5, 0.07),
            ],
            "mutB": [
                ("STRICT_SIG", 1.0, 0.01),
                ("PERMISSIVE_SIG", 1.0, 0.08),
            ],
        })
        _, stats_strict = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=1, random_seed=42
        )
        _, stats_permissive = select_unbiased_terms(
            cohort, fdr_threshold=0.10, top_n=10, n_groups=1, random_seed=42
        )
        assert stats_permissive.total_significant_terms >= stats_strict.total_significant_terms

    def test_fewer_terms_after_dedup_than_top_n_uses_all_remaining(self):
        """Contract 4: If dedup reduces pool below top_n, all remaining are used."""
        cohort = _build_cohort({
            "mutA": [
                ("CELL CYCLE REGULATION", 3.0, 0.01),
                ("POSITIVE CELL CYCLE REGULATION", 2.8, 0.01),
                ("NEGATIVE CELL CYCLE REGULATION", 2.6, 0.01),
                ("DNA REPAIR MECHANISM", 2.0, 0.01),
                ("DNA REPAIR MECHANISM PATHWAY", 1.8, 0.01),
            ],
            "mutB": [
                ("CELL CYCLE REGULATION", 2.5, 0.01),
                ("POSITIVE CELL CYCLE REGULATION", 2.3, 0.01),
                ("NEGATIVE CELL CYCLE REGULATION", 2.1, 0.01),
                ("DNA REPAIR MECHANISM", 1.5, 0.01),
                ("DNA REPAIR MECHANISM PATHWAY", 1.3, 0.01),
            ],
        })
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=2, random_seed=42
        )
        # Dedup removes redundant terms; fewer than top_n=10 remain
        assert stats.terms_selected <= stats.terms_after_dedup
        assert stats.terms_selected < 10


# ===========================================================================
# Error conditions
# ===========================================================================

class TestErrorConditions:
    """Test error conditions specified in the blueprint."""

    def test_insufficient_significant_terms_raises_value_error(self):
        """Error: Fewer terms pass FDR threshold than n_groups."""
        cohort = _build_cohort({
            "mutA": [
                ("TERM_A", 2.0, 0.01),  # significant
                ("TERM_B", 1.0, 0.50),  # NOT significant
            ],
            "mutB": [
                ("TERM_A", 1.5, 0.02),
                ("TERM_B", 0.5, 0.60),
            ],
        })
        # Only 1 significant term but n_groups=4
        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            select_unbiased_terms(
                cohort, fdr_threshold=0.05, top_n=10, n_groups=4, random_seed=42
            )

    def test_zero_significant_terms_raises_value_error(self):
        """Error: No terms pass FDR threshold at all."""
        cohort = _build_cohort({
            "mutA": [("TERM_A", 1.0, 0.50)],
            "mutB": [("TERM_B", 2.0, 0.60)],
        })
        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            select_unbiased_terms(
                cohort, fdr_threshold=0.05, top_n=10, n_groups=2, random_seed=42
            )

    def test_significant_terms_equal_to_n_groups_does_not_raise(self):
        """Boundary: Exactly n_groups significant terms should NOT raise."""
        cohort = _build_cohort({
            "mutA": [
                ("T1", 2.0, 0.01),
                ("T2", 1.5, 0.01),
            ],
            "mutB": [
                ("T1", 1.0, 0.01),
                ("T2", 0.5, 0.01),
            ],
        })
        # 2 significant terms, n_groups=2 -> should work
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=2, random_seed=42
        )
        assert stats.total_significant_terms == 2
        assert len(groups) <= 2


# ===========================================================================
# Precondition invariants
# ===========================================================================

class TestPreconditionInvariants:
    """Test precondition invariants: top_n > 0, n_groups > 0, n_groups <= top_n."""

    @pytest.fixture
    def basic_cohort(self):
        """Simple cohort for invariant tests."""
        return _build_cohort({
            "mutA": [("T1", 2.0, 0.01), ("T2", 1.5, 0.01), ("T3", 1.0, 0.01)],
            "mutB": [("T1", 1.0, 0.01), ("T2", 0.5, 0.01), ("T3", 0.5, 0.01)],
        })

    def test_top_n_zero_raises(self, basic_cohort):
        """Invariant: top_n must be > 0."""
        with pytest.raises((AssertionError, ValueError)):
            select_unbiased_terms(basic_cohort, top_n=0, n_groups=1)

    def test_n_groups_zero_raises(self, basic_cohort):
        """Invariant: n_groups must be > 0."""
        with pytest.raises((AssertionError, ValueError)):
            select_unbiased_terms(basic_cohort, top_n=5, n_groups=0)

    def test_negative_top_n_raises(self, basic_cohort):
        """Invariant: top_n must be > 0."""
        with pytest.raises((AssertionError, ValueError)):
            select_unbiased_terms(basic_cohort, top_n=-1, n_groups=1)

    def test_negative_n_groups_raises(self, basic_cohort):
        """Invariant: n_groups must be > 0."""
        with pytest.raises((AssertionError, ValueError)):
            select_unbiased_terms(basic_cohort, top_n=5, n_groups=-1)


# ===========================================================================
# End-to-end integration tests
# ===========================================================================

class TestEndToEndPipeline:
    """End-to-end tests verifying the complete pipeline with known data."""

    def test_full_pipeline_small_dataset(self):
        """Full pipeline with a small, fully traceable dataset.

        5 significant terms, 2 mutants, top_n=4, n_groups=2.
        """
        cohort = _build_cohort({
            "mutA": [
                ("ELECTRON TRANSPORT CHAIN", 3.0, 0.01),
                ("OXIDATIVE PHOSPHORYLATION", 2.5, 0.01),
                ("RIBOSOME BIOGENESIS", 2.0, 0.01),
                ("GPCR SIGNALING", 1.5, 0.01),
                ("SYNAPSE ASSEMBLY", 1.0, 0.01),
            ],
            "mutB": [
                ("ELECTRON TRANSPORT CHAIN", 2.0, 0.01),
                ("OXIDATIVE PHOSPHORYLATION", 1.5, 0.01),
                ("RIBOSOME BIOGENESIS", 1.0, 0.01),
                ("GPCR SIGNALING", -2.0, 0.01),
                ("SYNAPSE ASSEMBLY", 0.5, 0.01),
            ],
        })
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=4, n_groups=2, random_seed=42
        )
        assert stats.total_significant_terms == 5
        assert stats.terms_selected <= 4
        assert len(groups) <= 2
        assert all(len(g.term_names) > 0 for g in groups)
        total = sum(len(g.term_names) for g in groups)
        assert total <= 4
        assert stats.random_seed == 42
        assert stats.n_clusters == 2

    def test_pipeline_with_redundancy_reduction(self):
        """Pipeline where redundancy removes some terms before selection."""
        cohort = _build_cohort({
            "mutA": [
                ("REGULATION OF APOPTOSIS", 3.0, 0.01),
                ("POSITIVE REGULATION OF APOPTOSIS", 2.8, 0.01),
                ("RIBOSOME BIOGENESIS", 2.5, 0.01),
                ("GPCR SIGNALING PATHWAY", 2.0, 0.01),
                ("SYNAPSE FORMATION", 1.5, 0.01),
            ],
            "mutB": [
                ("REGULATION OF APOPTOSIS", 2.5, 0.01),
                ("POSITIVE REGULATION OF APOPTOSIS", 2.3, 0.01),
                ("RIBOSOME BIOGENESIS", 2.0, 0.01),
                ("GPCR SIGNALING PATHWAY", 1.5, 0.01),
                ("SYNAPSE FORMATION", 1.0, 0.01),
            ],
        })
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        assert stats.total_significant_terms == 5
        # "POSITIVE REGULATION OF APOPTOSIS" should be removed as redundant
        assert stats.terms_after_dedup < stats.total_significant_terms
        # All output terms should not include the redundant one
        all_terms = [t for g in groups for t in g.term_names]
        assert "POSITIVE REGULATION OF APOPTOSIS" not in all_terms
        assert "REGULATION OF APOPTOSIS" in all_terms

    def test_groups_are_ordered_by_importance(self):
        """Contract 9: Groups ordered by rank position of highest-ranked member."""
        terms = [(f"UNIQUE TERM {i:02d}", 3.0 - i * 0.2, 0.01) for i in range(10)]
        cohort = _build_cohort({
            "mutA": terms,
            "mutB": [(t, nes - 0.1, fdr) for t, nes, fdr in terms],
            "mutC": [(t, nes + 0.1, fdr) for t, nes, fdr in terms],
        })
        groups, _ = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=3, random_seed=42
        )
        # Collect all terms, find original ranking
        pooled = pool_significant_terms(cohort, 0.05)
        deduped = remove_redundant_terms(pooled)
        selected = select_top_n(deduped, 10)

        def best_rank(group):
            return min(selected.index(t) for t in group.term_names)

        ranks = [best_rank(g) for g in groups]
        assert ranks == sorted(ranks)
