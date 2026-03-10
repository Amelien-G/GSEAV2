# Auto-generated stub — do not edit
from dataclasses import dataclass
import numpy as np
from gsea_tool.data_ingestion import CohortData
from gsea_tool.cherry_picked import CategoryGroup

@dataclass
class UnbiasedSelectionStats:
    """Statistics collected during unbiased selection for notes.md."""
    total_significant_terms: int
    terms_after_dedup: int
    terms_selected: int
    n_clusters: int
    random_seed: int
    clustering_algorithm: str
    ...

def pool_significant_terms(cohort: CohortData, fdr_threshold: float) -> dict[str, float]:
    """Step 1-2: Pool terms passing FDR threshold in any mutant, compute max abs NES.

    Returns dict mapping term_name -> max_absolute_nes, sorted by value descending.
    """
    ...

def remove_redundant_terms(ranked_terms: dict[str, float]) -> dict[str, float]:
    """Step 3: Remove lexically redundant terms.

    For each pair of terms sharing substantial word overlap (Jaccard similarity
    of word sets > 0.5), retain only the term with higher max abs NES.
    """
    ...

def select_top_n(ranked_terms: dict[str, float], top_n: int) -> list[str]:
    """Step 4: Select top N terms from deduplicated ranked list."""
    ...

def cluster_terms(term_names: list[str], cohort: CohortData, n_groups: int, random_seed: int) -> list[CategoryGroup]:
    """Steps 5-6: Cluster selected terms by NES profile and auto-label groups.

    Uses hierarchical agglomerative clustering (Ward linkage) on the NES profile
    matrix (terms as rows, mutants as columns). Missing NES values are treated as 0.0.
    Each group is labeled with the term having the highest mean absolute NES
    within that group. Terms within each group are sorted by mean absolute NES
    descending. Groups are sorted by the position of their highest-ranked member
    in the original top-N ranking.
    """
    ...

def select_unbiased_terms(cohort: CohortData, fdr_threshold: float=0.05, top_n: int=20, n_groups: int=4, random_seed: int=42) -> tuple[list[CategoryGroup], UnbiasedSelectionStats]:
    """Top-level entry point for unbiased term selection (Figure 2).

    Returns the grouped terms and collection statistics for notes.md.
    """
    ...
assert top_n > 0, 'top_n must be a positive integer'
assert n_groups > 0, 'n_groups must be a positive integer'
assert n_groups <= top_n, 'Cannot have more groups than selected terms'
assert sum((len(g.term_names) for g in groups)) <= top_n, 'Total terms across groups cannot exceed top_n'
assert len(groups) <= n_groups, 'Number of groups cannot exceed n_groups'
assert all((len(g.term_names) > 0 for g in groups)), 'No empty groups'
assert stats.random_seed == random_seed, 'Stats must record the seed actually used'
