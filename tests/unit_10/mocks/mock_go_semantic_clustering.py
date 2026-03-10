# Auto-generated stub — do not edit
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from gsea_tool.configuration import ClusteringConfig
from gsea_tool.meta_analysis import FisherResult

@dataclass
class ClusteringResult:
    """Results of GO semantic similarity clustering."""
    representatives: list[str]
    representative_names: list[str]
    representative_pvalues: list[float]
    representative_n_contributing: list[int]
    cluster_assignments: dict[str, int]
    n_clusters: int
    n_prefiltered: int
    similarity_metric: str
    similarity_threshold: float
    ...

def download_or_load_obo(obo_url: str, cache_dir: Path) -> Path:
    """Download the GO OBO file if not cached, or return cached path.

    Returns path to the local OBO file.
    """
    ...

def download_or_load_gaf(gaf_url: str, cache_dir: Path) -> Path:
    """Download the Drosophila GAF file if not cached, or return cached path.

    Returns path to the local GAF file.
    """
    ...

def compute_information_content(obo_path: Path, gaf_path: Path) -> dict[str, float]:
    """Compute information content for GO terms from annotation frequencies.

    Returns dict mapping GO ID -> information content value.
    """
    ...

def compute_lin_similarity(go_ids: list[str], ic_values: dict[str, float], obo_path: Path) -> np.ndarray:
    """Compute pairwise Lin similarity matrix for a list of GO IDs.

    Returns symmetric matrix of shape (n, n) with values in [0, 1].
    """
    ...

def cluster_by_similarity(similarity_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    """Hierarchical agglomerative clustering on the similarity matrix.

    Cuts the dendrogram at the given similarity threshold.

    Returns list of clusters, where each cluster is a list of row indices.
    """
    ...

def select_representatives(clusters: list[list[int]], go_ids: list[str], fisher_result: FisherResult) -> ClusteringResult:
    """Select representative GO term per cluster (lowest combined p-value).

    Returns ClusteringResult with representatives ordered by combined p-value.
    """
    ...

def run_semantic_clustering(fisher_result: FisherResult, config: ClusteringConfig, output_dir: Path, cache_dir: Path) -> ClusteringResult:
    """Top-level entry point for GO semantic clustering.

    Downloads/loads OBO and GAF, computes similarity, clusters, selects
    representatives, and writes fisher_combined_pvalues.tsv with cluster
    assignments.

    Returns ClusteringResult.
    """
    ...

def write_fisher_results_with_clusters_tsv(fisher_result: FisherResult, clustering_result: ClusteringResult, output_dir: Path) -> Path:
    """Write fisher_combined_pvalues.tsv with cluster assignment and representative columns."""
    ...
assert config.similarity_threshold > 0.0, 'Similarity threshold must be positive'
assert config.similarity_threshold <= 1.0, 'Similarity threshold must be at most 1.0'
assert len(clustering_result.representatives) == clustering_result.n_clusters, 'One representative per cluster'
assert clustering_result.n_clusters > 0, 'At least one cluster must be formed'
assert all((go_id in fisher_result.combined_pvalues for go_id in clustering_result.representatives)), 'All representatives must be present in Fisher results'
assert clustering_result.representatives == sorted(clustering_result.representatives, key=lambda gid: fisher_result.combined_pvalues[gid]), 'Representatives must be ordered by combined p-value ascending'
