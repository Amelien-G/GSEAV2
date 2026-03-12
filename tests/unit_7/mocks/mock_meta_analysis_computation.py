# Auto-generated stub — do not edit
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from gsea_tool.data_ingestion import CohortData
from gsea_tool.configuration import FisherConfig

@dataclass
class FisherResult:
    """Results of Fisher's combined probability test."""
    go_ids: list[str]
    go_id_to_name: dict[str, str]
    combined_pvalues: dict[str, float]
    n_contributing: dict[str, int]
    pvalue_matrix: np.ndarray
    mutant_ids: list[str]
    go_id_order: list[str]
    n_mutants: int
    corrected_pvalues: dict[str, float] | None
    ...

def build_pvalue_dict_per_mutant(cohort: CohortData, pseudocount: float) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Build per-mutant {GO_ID: nom_pval} and {GO_ID: NES} dictionaries from ingested data.

    Replaces NOM p-val of 0.0 with pseudocount. Skips records with missing
    or non-numeric NOM p-val (already filtered during ingestion).

    Returns tuple of (pval_dict, nes_dict) where each maps mutant_id -> {go_id: value}.
    """
    ...

def build_pvalue_matrix(per_mutant_pvals: dict[str, dict[str, float]], mutant_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build the GO term x mutant p-value matrix with imputation.

    Missing entries are imputed as p = 1.0.

    Returns:
        matrix: np.ndarray of shape (n_go_terms, n_mutants)
        go_id_order: list of GO IDs corresponding to matrix rows
    """
    ...

def compute_fisher_combined(pvalue_matrix: np.ndarray, n_mutants: int) -> np.ndarray:
    """Compute Fisher's combined p-value for each GO term (row).

    Fisher statistic: X^2 = -2 * sum(ln(p_i))
    Combined p-value from chi-squared distribution with 2k degrees of freedom.

    Returns array of combined p-values, one per row.
    """
    ...

def run_fisher_analysis(cohort: CohortData, config: FisherConfig, output_dir: Path, clustering_enabled: bool) -> FisherResult:
    """Top-level entry point for Fisher's combined probability analysis.

    Writes pvalue_matrix.tsv to output_dir.
    If clustering_enabled is False, also writes fisher_combined_pvalues.tsv.

    Returns FisherResult with all computed values.
    """
    ...

def write_pvalue_matrix_tsv(matrix: np.ndarray, nes_matrix: np.ndarray, go_id_order: list[str], go_id_to_name: dict[str, str], mutant_ids: list[str], output_dir: Path) -> Path:
    """Write the p-value matrix to pvalue_matrix.tsv."""
    ...

def write_fisher_results_tsv(fisher_result: FisherResult, output_dir: Path) -> Path:
    """Write fisher_combined_pvalues.tsv without cluster assignments.

    Used when clustering is disabled.
    """
    ...
assert len(cohort.mutant_ids) >= 2, "Fisher's method requires at least 2 mutant lines"
assert config.pseudocount > 0, 'Pseudocount must be positive'
assert output_dir.is_dir(), 'Output directory must exist'
assert len(fisher_result.combined_pvalues) == len(fisher_result.go_ids), 'One combined p-value per GO ID'
assert all((0.0 <= p <= 1.0 for p in fisher_result.combined_pvalues.values())), 'Combined p-values must be in [0, 1]'
assert fisher_result.pvalue_matrix.shape == (len(fisher_result.go_id_order), len(fisher_result.mutant_ids)), 'Matrix shape must match GO IDs x mutants'
assert fisher_result.n_mutants == len(fisher_result.mutant_ids), 'n_mutants must match mutant_ids length'
