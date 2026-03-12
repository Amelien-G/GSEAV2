"""Unit 6 -- Meta-Analysis Computation using Fisher's combined probability method."""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy import stats

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


def build_pvalue_dict_per_mutant(
    cohort: CohortData,
    pseudocount: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Build per-mutant {GO_ID: nom_pval} and {GO_ID: NES} dictionaries from ingested data.

    Replaces NOM p-val of 0.0 with pseudocount. Skips records with missing
    or non-numeric NOM p-val (already filtered during ingestion).

    Returns tuple of (pval_dict, nes_dict) each mapping mutant_id -> {go_id: value}.
    """
    pval_result: dict[str, dict[str, float]] = {}
    nes_result: dict[str, dict[str, float]] = {}
    for mutant_id in cohort.mutant_ids:
        profile = cohort.profiles[mutant_id]
        pval_dict: dict[str, float] = {}
        nes_dict: dict[str, float] = {}
        for term_name, record in profile.records.items():
            pval = record.nom_pval
            if pval == 0.0:
                pval = pseudocount
            pval_dict[record.go_id] = pval
            nes_dict[record.go_id] = record.nes
        pval_result[mutant_id] = pval_dict
        nes_result[mutant_id] = nes_dict
    return pval_result, nes_result


def build_pvalue_matrix(
    per_mutant_pvals: dict[str, dict[str, float]],
    mutant_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build the GO term x mutant p-value matrix with imputation.

    Missing entries are imputed as p = 1.0.

    Returns:
        matrix: np.ndarray of shape (n_go_terms, n_mutants)
        go_id_order: list of GO IDs corresponding to matrix rows
    """
    # Collect union of all GO IDs across all mutants
    all_go_ids: set[str] = set()
    for pvals in per_mutant_pvals.values():
        all_go_ids.update(pvals.keys())

    go_id_order = sorted(all_go_ids)
    n_go = len(go_id_order)
    n_mutants = len(mutant_ids)

    matrix = np.ones((n_go, n_mutants), dtype=np.float64)

    for j, mutant_id in enumerate(mutant_ids):
        pvals = per_mutant_pvals.get(mutant_id, {})
        for i, go_id in enumerate(go_id_order):
            if go_id in pvals:
                matrix[i, j] = pvals[go_id]

    return matrix, go_id_order


def build_nes_matrix(
    per_mutant_nes: dict[str, dict[str, float]],
    mutant_ids: list[str],
    go_id_order: list[str],
) -> np.ndarray:
    """Build the GO term x mutant NES matrix.

    Missing entries are filled with NaN.

    Returns:
        matrix: np.ndarray of shape (n_go_terms, n_mutants)
    """
    n_go = len(go_id_order)
    n_mutants = len(mutant_ids)

    matrix = np.full((n_go, n_mutants), np.nan, dtype=float)

    go_idx_map = {go_id: i for i, go_id in enumerate(go_id_order)}
    mut_idx_map = {mid: j for j, mid in enumerate(mutant_ids)}

    for mutant_id in mutant_ids:
        nes_vals = per_mutant_nes.get(mutant_id, {})
        j = mut_idx_map[mutant_id]
        for go_id, nes in nes_vals.items():
            if go_id in go_idx_map:
                i = go_idx_map[go_id]
                matrix[i, j] = nes

    return matrix


def compute_fisher_combined(
    pvalue_matrix: np.ndarray,
    n_mutants: int,
) -> np.ndarray:
    """Compute Fisher's combined p-value for each GO term (row).

    Fisher statistic: X^2 = -2 * sum(ln(p_i))
    Combined p-value from chi-squared distribution with 2k degrees of freedom.

    Returns array of combined p-values, one per row.
    """
    # X^2 = -2 * sum(ln(p_i)) for each row
    log_pvals = np.log(pvalue_matrix)
    chi2_stats = -2.0 * np.sum(log_pvals, axis=1)

    df = 2 * n_mutants
    # Combined p-value = survival function = 1 - CDF
    combined_pvalues = stats.chi2.sf(chi2_stats, df)

    return combined_pvalues


def _benjamini_hochberg(pvalues: dict[str, float]) -> dict[str, float]:
    """Apply Benjamini-Hochberg FDR correction to a dict of p-values."""
    go_ids = list(pvalues.keys())
    pvals = np.array([pvalues[gid] for gid in go_ids])
    n = len(pvals)

    if n == 0:
        return {}

    # Sort by p-value
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]

    # BH correction: adjusted_p[i] = p[i] * n / rank[i]
    # rank is 1-based
    ranks = np.arange(1, n + 1)
    adjusted = sorted_pvals * n / ranks

    # Enforce monotonicity: working backwards, ensure each adjusted p
    # is no larger than the one after it
    adjusted_monotone = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Cap at 1.0
    adjusted_monotone = np.minimum(adjusted_monotone, 1.0)

    # Map back to original order
    result_array = np.empty(n)
    result_array[sorted_indices] = adjusted_monotone

    return {go_ids[i]: float(result_array[i]) for i in range(n)}


def _build_go_id_to_name(cohort: CohortData) -> dict[str, str]:
    """Build GO ID -> term name mapping from cohort data."""
    go_id_to_name: dict[str, str] = {}
    for profile in cohort.profiles.values():
        for record in profile.records.values():
            # Last seen name wins (they should be consistent)
            go_id_to_name[record.go_id] = record.term_name
    return go_id_to_name


def run_fisher_analysis(
    cohort: CohortData,
    config: FisherConfig,
    output_dir: Path,
    clustering_enabled: bool,
) -> FisherResult:
    """Top-level entry point for Fisher's combined probability analysis.

    Writes pvalue_matrix.tsv to output_dir.
    If clustering_enabled is False, also writes fisher_combined_pvalues.tsv.

    Returns FisherResult with all computed values.
    """
    # Pre-conditions
    assert len(cohort.mutant_ids) >= 2, "Fisher's method requires at least 2 mutant lines"
    assert config.pseudocount > 0, "Pseudocount must be positive"
    assert output_dir.is_dir(), "Output directory must exist"

    mutant_ids = cohort.mutant_ids
    n_mutants = len(mutant_ids)

    # Step 1: Build per-mutant p-value and NES dicts
    per_mutant_pvals, per_mutant_nes = build_pvalue_dict_per_mutant(cohort, config.pseudocount)

    # Step 2: Build p-value matrix
    pvalue_matrix, go_id_order = build_pvalue_matrix(per_mutant_pvals, mutant_ids)

    # Step 2b: Build NES matrix
    nes_matrix = build_nes_matrix(per_mutant_nes, mutant_ids, go_id_order)

    # Step 3: Compute Fisher's combined p-values
    combined_pvals_array = compute_fisher_combined(pvalue_matrix, n_mutants)

    # Build combined_pvalues dict
    combined_pvalues: dict[str, float] = {}
    for i, go_id in enumerate(go_id_order):
        combined_pvalues[go_id] = float(combined_pvals_array[i])

    # Compute n_contributing: lines with p < 1.0 for each GO term
    n_contributing: dict[str, int] = {}
    for i, go_id in enumerate(go_id_order):
        count = int(np.sum(pvalue_matrix[i, :] < 1.0))
        n_contributing[go_id] = count

    # Build GO ID to name mapping
    go_id_to_name = _build_go_id_to_name(cohort)

    # Optional FDR correction
    corrected_pvalues: dict[str, float] | None = None
    if config.apply_fdr:
        corrected_pvalues = _benjamini_hochberg(combined_pvalues)

    go_ids = list(go_id_order)

    fisher_result = FisherResult(
        go_ids=go_ids,
        go_id_to_name=go_id_to_name,
        combined_pvalues=combined_pvalues,
        n_contributing=n_contributing,
        pvalue_matrix=pvalue_matrix,
        mutant_ids=mutant_ids,
        go_id_order=go_id_order,
        n_mutants=n_mutants,
        corrected_pvalues=corrected_pvalues,
    )

    # Write pvalue_matrix.tsv
    write_pvalue_matrix_tsv(pvalue_matrix, nes_matrix, go_id_order, go_id_to_name, mutant_ids, output_dir)

    # If clustering is disabled, also write fisher_combined_pvalues.tsv
    if not clustering_enabled:
        write_fisher_results_tsv(fisher_result, output_dir)

    return fisher_result


def write_pvalue_matrix_tsv(
    matrix: np.ndarray,
    nes_matrix: np.ndarray,
    go_id_order: list[str],
    go_id_to_name: dict[str, str],
    mutant_ids: list[str],
    output_dir: Path,
) -> Path:
    """Write the p-value matrix with NES values to pvalue_matrix.tsv.

    Each row contains: GO_ID, GO_Term, then for each mutant a pvalue column
    followed by a NES column.
    """
    output_path = output_dir / "pvalue_matrix.tsv"

    lines: list[str] = []
    # Header: GO_ID, GO_Term, then alternating pval/NES columns per mutant
    header_parts = ["GO_ID", "GO_Term"]
    for mid in mutant_ids:
        header_parts.append(f"{mid}_pval")
        header_parts.append(f"{mid}_NES")
    lines.append("\t".join(header_parts))

    for i, go_id in enumerate(go_id_order):
        term_name = go_id_to_name.get(go_id, "")
        row_parts = [go_id, term_name]
        for j in range(len(mutant_ids)):
            row_parts.append(str(matrix[i, j]))
            nes_val = nes_matrix[i, j]
            row_parts.append("" if np.isnan(nes_val) else str(nes_val))
        lines.append("\t".join(row_parts))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return output_path


def write_fisher_results_tsv(
    fisher_result: FisherResult,
    output_dir: Path,
) -> Path:
    """Write fisher_combined_pvalues.tsv without cluster assignments.

    Used when clustering is disabled.
    """
    output_path = output_dir / "fisher_combined_pvalues.tsv"

    with open(output_path, "w", newline="") as f:
        # Header
        header = "GO_ID\tTerm_Name\tCombined_PValue\tN_Contributing\n"
        f.write(header)

        for go_id in fisher_result.go_id_order:
            term_name = fisher_result.go_id_to_name.get(go_id, "")
            combined_p = fisher_result.combined_pvalues[go_id]
            n_contrib = fisher_result.n_contributing[go_id]
            line = f"{go_id}\t{term_name}\t{combined_p}\t{n_contrib}\n"
            f.write(line)

    return output_path
