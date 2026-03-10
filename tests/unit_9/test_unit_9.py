"""
Tests for Unit 9 -- Notes Generation

Comprehensive test suite validating all behavioral contracts, invariants, error
conditions, and signatures specified in the Unit 9 blueprint.

## Synthetic Data Assumptions

- CohortData uses 3 mutant lines (mutA, mutB, mutC) with 5 GO terms spanning
  FDR values from 0.001 to 0.8, representing a minimal valid GSEA cohort.
- GO term names are uppercase (e.g., "MITOCHONDRIAL TRANSLATION").
- DotPlotResult: fig1 has 15 terms/4 categories, fig2 has 20 terms/4 categories.
- BarPlotResult: n_bars=10, n_mutants=3, clustering_was_used toggled per test.
- UnbiasedSelectionStats: 100 significant, 80 after dedup, 20 selected,
  4 clusters, seed 42, Ward linkage.
- FisherResult: 5 GO terms with combined p-values [1e-10, 1e-6, 1e-4, 0.01, 0.5].
- ClusteringResult: 3 clusters, 5 prefiltered, Lin similarity at 0.7 threshold.
- ToolConfig: all default values unless specifically overridden.
- Software versions: non-empty runtime strings for Python, matplotlib, pandas,
  scipy, numpy, goatools, pyyaml.
"""

from pathlib import Path

import numpy as np
import pytest

from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.configuration import (
    ToolConfig,
    DotPlotConfig,
    FisherConfig,
    ClusteringConfig,
    PlotAppearanceConfig,
)
from gsea_tool.unbiased import UnbiasedSelectionStats
from gsea_tool.dot_plot import DotPlotResult
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.go_clustering import ClusteringResult
from gsea_tool.bar_plot import BarPlotResult

from gsea_tool.notes_generation import (
    NotesInput,
    generate_notes,
    format_figure_legends,
    format_methods_text,
    format_summary_statistics,
    format_reproducibility_note,
    format_config_guide,
    get_dependency_versions,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_cohort_data(n_mutants: int = 3) -> CohortData:
    """Build a synthetic CohortData with n_mutants mutant lines."""
    mutant_ids = [f"mut{chr(65 + i)}" for i in range(n_mutants)]
    mutant_ids.sort()

    go_terms = [
        ("MITOCHONDRIAL TRANSLATION", "GO:0070125", 2.1, 0.001, 0.0001, 50),
        ("OXIDATIVE PHOSPHORYLATION", "GO:0006119", 1.8, 0.01, 0.001, 80),
        ("RIBOSOME BIOGENESIS", "GO:0042254", -1.5, 0.03, 0.005, 120),
        ("SYNAPTIC VESICLE CYCLE", "GO:0099504", 1.2, 0.2, 0.05, 40),
        ("CELL CYCLE REGULATION", "GO:0051726", -0.8, 0.8, 0.3, 200),
    ]

    profiles = {}
    all_term_names = set()
    all_go_ids = set()

    for mid in mutant_ids:
        records = {}
        for tname, gid, nes, fdr, pval, size in go_terms:
            rec = TermRecord(
                term_name=tname, go_id=gid, nes=nes,
                fdr=fdr, nom_pval=pval, size=size,
            )
            records[tname] = rec
            all_term_names.add(tname)
            all_go_ids.add(gid)
        profiles[mid] = MutantProfile(mutant_id=mid, records=records)

    return CohortData(
        mutant_ids=mutant_ids,
        profiles=profiles,
        all_term_names=all_term_names,
        all_go_ids=all_go_ids,
    )


def _make_tool_config(**overrides) -> ToolConfig:
    """Build a ToolConfig with all defaults, accepting section-level overrides."""
    return ToolConfig(
        dot_plot=DotPlotConfig(**(overrides.get("dot_plot", {}))),
        fisher=FisherConfig(**(overrides.get("fisher", {}))),
        clustering=ClusteringConfig(**(overrides.get("clustering", {}))),
        plot_appearance=PlotAppearanceConfig(**(overrides.get("plot_appearance", {}))),
    )


def _make_dot_plot_result(
    stem: str = "figure1",
    output_dir: Path = Path("/tmp/output"),
    n_terms: int = 15,
    n_categories: int = 4,
    n_mutants: int = 3,
) -> DotPlotResult:
    return DotPlotResult(
        pdf_path=output_dir / f"{stem}.pdf",
        png_path=output_dir / f"{stem}.png",
        svg_path=output_dir / f"{stem}.svg",
        n_terms_displayed=n_terms,
        n_categories=n_categories,
        n_mutants=n_mutants,
    )


def _make_bar_plot_result(
    output_dir: Path = Path("/tmp/output"),
    n_bars: int = 10,
    n_mutants: int = 3,
    clustering_was_used: bool = True,
) -> BarPlotResult:
    return BarPlotResult(
        pdf_path=output_dir / "figure3_meta_analysis.pdf",
        png_path=output_dir / "figure3_meta_analysis.png",
        svg_path=output_dir / "figure3_meta_analysis.svg",
        n_bars=n_bars,
        n_mutants=n_mutants,
        clustering_was_used=clustering_was_used,
    )


def _make_unbiased_stats(
    total_significant_terms: int = 100,
    terms_after_dedup: int = 80,
    terms_selected: int = 20,
    n_clusters: int = 4,
    random_seed: int = 42,
    clustering_algorithm: str = "scipy.cluster.hierarchy (Ward linkage)",
) -> UnbiasedSelectionStats:
    return UnbiasedSelectionStats(
        total_significant_terms=total_significant_terms,
        terms_after_dedup=terms_after_dedup,
        terms_selected=terms_selected,
        n_clusters=n_clusters,
        random_seed=random_seed,
        clustering_algorithm=clustering_algorithm,
    )


def _make_fisher_result(n_mutants: int = 3) -> FisherResult:
    go_ids = ["GO:0070125", "GO:0006119", "GO:0042254", "GO:0099504", "GO:0051726"]
    go_id_to_name = {
        "GO:0070125": "MITOCHONDRIAL TRANSLATION",
        "GO:0006119": "OXIDATIVE PHOSPHORYLATION",
        "GO:0042254": "RIBOSOME BIOGENESIS",
        "GO:0099504": "SYNAPTIC VESICLE CYCLE",
        "GO:0051726": "CELL CYCLE REGULATION",
    }
    combined_pvalues = {
        "GO:0070125": 1e-10,
        "GO:0006119": 1e-6,
        "GO:0042254": 1e-4,
        "GO:0099504": 0.01,
        "GO:0051726": 0.5,
    }
    n_contributing = {
        "GO:0070125": 3,
        "GO:0006119": 3,
        "GO:0042254": 2,
        "GO:0099504": 2,
        "GO:0051726": 1,
    }
    mutant_ids = ["mutA", "mutB", "mutC"][:n_mutants]
    pvalue_matrix = np.array([
        [0.0001, 0.001, 0.01],
        [0.001, 0.01, 0.1],
        [0.005, 0.05, 1.0],
        [0.05, 0.1, 1.0],
        [0.3, 1.0, 1.0],
    ])[:, :n_mutants]

    return FisherResult(
        go_ids=go_ids,
        go_id_to_name=go_id_to_name,
        combined_pvalues=combined_pvalues,
        n_contributing=n_contributing,
        pvalue_matrix=pvalue_matrix,
        mutant_ids=mutant_ids,
        go_id_order=go_ids,
        n_mutants=n_mutants,
        corrected_pvalues=None,
    )


def _make_clustering_result(
    n_clusters: int = 3,
    n_prefiltered: int = 5,
    similarity_metric: str = "Lin",
    similarity_threshold: float = 0.7,
) -> ClusteringResult:
    return ClusteringResult(
        representatives=["GO:0070125", "GO:0042254", "GO:0099504"],
        representative_names=[
            "MITOCHONDRIAL TRANSLATION",
            "RIBOSOME BIOGENESIS",
            "SYNAPTIC VESICLE CYCLE",
        ],
        representative_pvalues=[1e-10, 1e-4, 0.01],
        representative_n_contributing=[3, 2, 2],
        cluster_assignments={
            "GO:0070125": 0, "GO:0006119": 0,
            "GO:0042254": 1,
            "GO:0099504": 2, "GO:0051726": 2,
        },
        n_clusters=n_clusters,
        n_prefiltered=n_prefiltered,
        similarity_metric=similarity_metric,
        similarity_threshold=similarity_threshold,
    )


def _make_notes_input(
    with_fig1: bool = True,
    with_clustering: bool = True,
    output_dir: Path = Path("/tmp/output"),
) -> NotesInput:
    """Build a complete synthetic NotesInput."""
    return NotesInput(
        cohort=_make_cohort_data(),
        config=_make_tool_config(),
        fig1_result=_make_dot_plot_result(stem="figure1") if with_fig1 else None,
        fig1_method="ontology" if with_fig1 else None,
        fig2_result=_make_dot_plot_result(stem="figure2", n_terms=20, n_categories=4),
        fig3_result=_make_bar_plot_result(
            output_dir=output_dir,
            clustering_was_used=with_clustering,
        ),
        unbiased_stats=_make_unbiased_stats(),
        fisher_result=_make_fisher_result(),
        clustering_result=_make_clustering_result() if with_clustering else None,
    )


# ---------------------------------------------------------------------------
# Contract 1: Output file is named exactly notes.md and written to output_dir
# ---------------------------------------------------------------------------

class TestGenerateNotesFileOutput:
    """Contract 1: The output file is named exactly notes.md and written to output_dir."""

    def test_generate_notes_returns_path_to_notes_md(self, tmp_path):
        """generate_notes returns the path to the written notes.md file."""
        ni = _make_notes_input()
        result = generate_notes(ni, tmp_path)
        assert result.name == "notes.md"
        assert result.parent == tmp_path

    def test_generate_notes_file_exists_after_call(self, tmp_path):
        """The notes.md file must exist on disk after generate_notes completes."""
        ni = _make_notes_input()
        result = generate_notes(ni, tmp_path)
        assert result.exists()

    def test_generate_notes_file_is_nonempty(self, tmp_path):
        """The generated notes.md must contain non-trivial content."""
        ni = _make_notes_input()
        result = generate_notes(ni, tmp_path)
        content = result.read_text(encoding="utf-8")
        assert len(content) > 100, "notes.md should contain substantial content"


# ---------------------------------------------------------------------------
# Contract 2: File contains five sections
# ---------------------------------------------------------------------------

class TestFiveSectionsPresent:
    """Contract 2: The file contains five sections: Figure Legends,
    Materials and Methods, Summary Statistics, Reproducibility Note,
    Configuration Guide."""

    def test_all_five_section_headings_present(self, tmp_path):
        """notes.md must contain all five section headings."""
        ni = _make_notes_input()
        path = generate_notes(ni, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "Figure Legends" in content
        assert "Materials and Methods" in content
        assert "Summary Statistics" in content
        assert "Reproducibility Note" in content
        assert "Configuration Guide" in content

    def test_sections_appear_in_correct_order(self, tmp_path):
        """The five sections must appear in the specified order."""
        ni = _make_notes_input()
        path = generate_notes(ni, tmp_path)
        content = path.read_text(encoding="utf-8")
        idx_legends = content.index("Figure Legends")
        idx_methods = content.index("Materials and Methods")
        idx_summary = content.index("Summary Statistics")
        idx_repro = content.index("Reproducibility Note")
        idx_config = content.index("Configuration Guide")
        assert idx_legends < idx_methods < idx_summary < idx_repro < idx_config


# ---------------------------------------------------------------------------
# Contract 3: Figure Legends section
# ---------------------------------------------------------------------------

class TestFigureLegends:
    """Contract 3: Figure Legends describe each produced figure with correct
    visual encoding details."""

    def test_legend_contains_figure_1_when_present(self):
        """When fig1_result is provided, legend must describe Figure 1."""
        ni = _make_notes_input(with_fig1=True)
        result = format_figure_legends(ni)
        assert "Figure 1" in result

    def test_legend_omits_figure_1_when_not_produced(self):
        """When fig1_result is None, Figure 1 legend must be omitted."""
        ni = _make_notes_input(with_fig1=False)
        result = format_figure_legends(ni)
        assert "Figure 1" not in result

    def test_legend_always_contains_figure_2(self):
        """Figure 2 legend must always be present."""
        ni = _make_notes_input(with_fig1=False)
        result = format_figure_legends(ni)
        assert "Figure 2" in result

    def test_legend_always_contains_figure_3(self):
        """Figure 3 legend must always be present."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        assert "Figure 3" in result

    def test_dot_plot_legend_describes_nes_color_encoding(self):
        """Dot plot legends must describe NES diverging red-blue color scale."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        lower = result.lower()
        assert "nes" in lower or "normalized enrichment" in lower
        assert "red" in lower and "blue" in lower

    def test_dot_plot_legend_describes_dot_size_as_log10_fdr(self):
        """Dot plot legends must describe dot size as -log10(FDR)."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        lower = result.lower()
        assert "log" in lower and "fdr" in lower

    def test_dot_plot_legend_describes_empty_cells(self):
        """Dot plot legends must describe what empty cells mean."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        lower = result.lower()
        assert "empty" in lower or "absent" in lower or "not" in lower

    def test_dot_plot_legend_describes_category_boxes(self):
        """Dot plot legends must describe what category boxes represent."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        lower = result.lower()
        assert "categor" in lower

    def test_dot_plot_legend_states_fdr_threshold(self):
        """Dot plot legends must state the FDR threshold used."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        assert "0.05" in result

    def test_dot_plot_legend_states_number_of_mutants(self):
        """Dot plot legends must state the number of mutants."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        assert "3" in result

    def test_fig1_legend_includes_term_and_category_counts(self):
        """Figure 1 legend must include the number of terms displayed
        and number of categories."""
        ni = _make_notes_input(with_fig1=True)
        result = format_figure_legends(ni)
        assert "15" in result  # fig1 n_terms_displayed
        assert "4" in result   # fig1 n_categories

    def test_fig2_legend_includes_term_and_category_counts(self):
        """Figure 2 legend must include the number of terms displayed
        and number of categories."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        assert "20" in result  # fig2 n_terms_displayed

    def test_fig3_legend_describes_bar_plot_encoding(self):
        """Figure 3 legend must describe bar length (-log10 combined p),
        bar color (number of contributing lines), and Fisher's method."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        lower = result.lower()
        assert "bar" in lower or "horizontal" in lower
        assert "fisher" in lower
        assert "contributing" in lower or "mutant line" in lower

    def test_fig3_legend_describes_clustering_when_used(self):
        """When clustering was used, Figure 3 legend must describe the
        GO semantic clustering step."""
        ni = _make_notes_input(with_clustering=True)
        result = format_figure_legends(ni)
        lower = result.lower()
        assert "cluster" in lower or "semantic" in lower

    def test_fig3_legend_no_clustering_states_direct_selection(self):
        """When clustering was not used, Figure 3 legend must note that
        clustering was not applied."""
        ni = _make_notes_input(with_clustering=False)
        result = format_figure_legends(ni)
        lower = result.lower()
        assert "not applied" in lower or "not" in lower and "cluster" in lower

    def test_fig3_legend_includes_top_n_bars(self):
        """Figure 3 legend must include the number of bars shown."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        assert "10" in result  # n_bars from bar_plot_result


# ---------------------------------------------------------------------------
# Contract 4: Materials and Methods section
# ---------------------------------------------------------------------------

class TestMethodsText:
    """Contract 4: Materials and Methods text describes the full analysis pipeline."""

    def test_methods_states_gsea_output_was_consumed(self):
        """Methods must state that GSEA preranked output was consumed, not generated."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        lower = result.lower()
        assert "consumed" in lower or "input" in lower
        assert "gsea" in lower

    def test_methods_describes_figure2_clustering_parameters(self):
        """Methods must name the clustering algorithm and parameters for Figure 2:
        Ward linkage, number of clusters, random seed."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        lower = result.lower()
        assert "ward" in lower
        assert "42" in result  # random seed
        assert "4" in result   # n_clusters

    def test_methods_describes_figure2_redundancy_removal(self):
        """Methods must describe redundancy removal for Figure 2:
        word-set Jaccard similarity > 0.5."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        lower = result.lower()
        assert "jaccard" in lower
        assert "0.5" in result

    def test_methods_describes_fisher_method_for_figure3(self):
        """Methods must describe Fisher's combined probability test with
        imputation details and degrees of freedom."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        lower = result.lower()
        assert "fisher" in lower
        assert "imput" in lower  # imputation / imputed
        assert "1.0" in result   # imputed p-value
        # degrees of freedom = 2 * n_mutants = 6
        assert "6" in result

    def test_methods_describes_go_semantic_clustering_when_enabled(self):
        """When clustering is enabled, methods must describe the GO semantic
        similarity clustering approach: Lin similarity, information content,
        threshold, representative selection."""
        ni = _make_notes_input(with_clustering=True)
        result = format_methods_text(ni)
        lower = result.lower()
        assert "lin" in lower
        assert "information content" in lower
        assert "0.7" in result  # similarity threshold
        assert "representative" in lower or "lowest" in lower

    def test_methods_notes_clustering_not_applied_when_disabled(self):
        """When clustering is disabled, methods must note that clustering
        was not applied and raw top-N terms were used."""
        ni = _make_notes_input(with_clustering=False)
        result = format_methods_text(ni)
        lower = result.lower()
        assert "not applied" in lower or "not" in lower

    def test_methods_lists_software_dependencies_with_versions(self):
        """Methods must list software dependencies with version numbers."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        lower = result.lower()
        assert "python" in lower or "matplotlib" in lower or "software" in lower

    def test_methods_includes_figure1_selection_criteria_when_present(self):
        """When Figure 1 was produced, methods must describe its GO term
        selection criteria."""
        ni = _make_notes_input(with_fig1=True)
        result = format_methods_text(ni)
        assert "Figure 1" in result


# ---------------------------------------------------------------------------
# Contract 5: Summary Statistics section
# ---------------------------------------------------------------------------

class TestSummaryStatistics:
    """Contract 5: Summary Statistics reports required metrics."""

    def test_summary_reports_number_of_mutants(self):
        """Summary must report number of mutants analyzed."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        lower = result.lower()
        assert "mutant" in lower
        assert "3" in result

    def test_summary_reports_total_unique_go_terms(self):
        """Summary must report total unique GO terms in input data."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        lower = result.lower()
        assert "unique" in lower or "total" in lower
        assert "5" in result  # 5 unique GO terms in our synthetic data

    def test_summary_reports_significant_terms(self):
        """Summary must report the number of GO terms passing FDR threshold."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        lower = result.lower()
        assert "significant" in lower or "passing" in lower or "fdr" in lower
        assert "100" in result  # total_significant_terms from unbiased_stats

    def test_summary_reports_terms_displayed_in_figure1_when_present(self):
        """When Figure 1 is produced, summary must report its term count."""
        ni = _make_notes_input(with_fig1=True)
        result = format_summary_statistics(ni)
        assert "15" in result  # fig1 n_terms_displayed

    def test_summary_reports_terms_displayed_in_figure2(self):
        """Summary must report number of GO terms displayed in Figure 2."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        assert "20" in result  # fig2 n_terms_displayed

    def test_summary_reports_terms_displayed_in_figure3(self):
        """Summary must report number of GO terms displayed in Figure 3."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        assert "10" in result  # fig3 n_bars

    def test_summary_reports_fisher_prefilter_count_with_clustering(self):
        """When clustering is enabled, summary must report Fisher pre-filter count."""
        ni = _make_notes_input(with_clustering=True)
        result = format_summary_statistics(ni)
        lower = result.lower()
        assert "pre-filter" in lower or "prefilter" in lower or "fisher" in lower
        assert "5" in result  # n_prefiltered from clustering_result

    def test_summary_reports_fisher_prefilter_count_without_clustering(self):
        """When clustering is disabled, summary must still report Fisher
        pre-filter count computed from combined p-values."""
        ni = _make_notes_input(with_clustering=False)
        result = format_summary_statistics(ni)
        # 4 of 5 terms have combined_pvalues < 0.05 (default prefilter_pvalue)
        assert "4" in result

    def test_summary_reports_number_of_semantic_clusters_when_used(self):
        """When clustering is used, summary must report number of semantic clusters."""
        ni = _make_notes_input(with_clustering=True)
        result = format_summary_statistics(ni)
        lower = result.lower()
        assert "cluster" in lower
        assert "3" in result  # n_clusters from clustering_result

    def test_summary_omits_figure1_count_when_not_produced(self):
        """When Figure 1 is not produced, its term count must not appear in a
        Figure 1 context."""
        ni = _make_notes_input(with_fig1=False)
        result = format_summary_statistics(ni)
        # "Figure 1" should not appear
        assert "Figure 1" not in result


# ---------------------------------------------------------------------------
# Contract 6: Reproducibility Note
# ---------------------------------------------------------------------------

class TestReproducibilityNote:
    """Contract 6: Reproducibility Note states random seed, software versions,
    and all configuration parameters."""

    def test_reproducibility_states_random_seed(self):
        """Reproducibility note must state the random seed for Figure 2 clustering."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        assert "42" in result
        lower = result.lower()
        assert "seed" in lower or "random" in lower

    def test_reproducibility_includes_software_versions(self):
        """Reproducibility note must include software version strings."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        lower = result.lower()
        assert "version" in lower or "python" in lower

    def test_reproducibility_includes_dot_plot_config_parameters(self):
        """Reproducibility note must include dot_plot configuration parameters."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        assert "0.05" in result  # fdr_threshold
        assert "20" in result    # top_n
        assert "4" in result     # n_groups
        assert "42" in result    # random_seed

    def test_reproducibility_includes_fisher_config_parameters(self):
        """Reproducibility note must include Fisher configuration parameters."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        lower = result.lower()
        assert "pseudocount" in lower or "1e-10" in result or "1e-1" in lower

    def test_reproducibility_includes_clustering_config_parameters(self):
        """Reproducibility note must include clustering configuration parameters."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        lower = result.lower()
        assert "clustering" in lower
        assert "similarity" in lower

    def test_reproducibility_includes_plot_appearance_config(self):
        """Reproducibility note must include plot appearance configuration."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        lower = result.lower()
        assert "dpi" in lower or "300" in result
        assert "arial" in lower or "font" in lower


# ---------------------------------------------------------------------------
# Contract 8: Configuration Guide
# ---------------------------------------------------------------------------

class TestConfigGuide:
    """Contract 8: Configuration Guide describes all config.yaml parameters
    with defaults, organized by section."""

    def test_config_guide_contains_dot_plot_section(self):
        """Config guide must describe the dot_plot section."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        lower = result.lower()
        assert "dot_plot" in lower or "dot plot" in lower
        assert "fdr_threshold" in lower or "fdr threshold" in lower
        assert "top_n" in lower or "top n" in lower

    def test_config_guide_contains_fisher_section(self):
        """Config guide must describe the fisher section."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        lower = result.lower()
        assert "fisher" in lower
        assert "pseudocount" in lower
        assert "prefilter" in lower or "pre-filter" in lower or "pre_filter" in lower

    def test_config_guide_contains_clustering_section(self):
        """Config guide must describe the clustering section."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        lower = result.lower()
        assert "clustering" in lower
        assert "enabled" in lower
        assert "similarity" in lower

    def test_config_guide_contains_plot_section(self):
        """Config guide must describe the plot appearance section."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        lower = result.lower()
        assert "dpi" in lower
        assert "font" in lower
        assert "colormap" in lower or "bar_colormap" in lower

    def test_config_guide_includes_default_values(self):
        """Config guide must include default values for parameters."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        # Check representative defaults
        assert "0.05" in result   # fdr_threshold default
        assert "20" in result     # top_n / top_n_bars default
        assert "42" in result     # random_seed default
        assert "300" in result    # dpi default
        assert "0.7" in result    # similarity_threshold default

    def test_config_guide_describes_how_to_modify_parameters(self):
        """Config guide must explain how to configure parameters via config.yaml."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        lower = result.lower()
        assert "config.yaml" in lower or "config" in lower


# ---------------------------------------------------------------------------
# Contract 9: All text is copy-paste-ready prose
# ---------------------------------------------------------------------------

class TestProseQuality:
    """Contract 9: All text is written as copy-paste-ready prose.
    No code blocks, no raw data dumps."""

    def test_no_code_blocks_in_output(self, tmp_path):
        """The notes.md file must not contain code block markers."""
        ni = _make_notes_input()
        path = generate_notes(ni, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "```" not in content, "notes.md must not contain code blocks"

    def test_no_raw_data_dumps(self, tmp_path):
        """The notes.md file must not contain raw data structures."""
        ni = _make_notes_input()
        path = generate_notes(ni, tmp_path)
        content = path.read_text(encoding="utf-8")
        # Should not contain Python-style dicts or lists
        assert "{'GO:" not in content
        assert "{'mut" not in content


# ---------------------------------------------------------------------------
# Contract 10: Software versions obtained at runtime
# ---------------------------------------------------------------------------

class TestGetDependencyVersions:
    """Contract 10: Software version strings are obtained at runtime,
    not hardcoded."""

    def test_returns_dict_of_strings(self):
        """get_dependency_versions must return a dict mapping str to str."""
        versions = get_dependency_versions()
        assert isinstance(versions, dict)
        for key, val in versions.items():
            assert isinstance(key, str)
            assert isinstance(val, str)

    def test_includes_python_version(self):
        """Versions must include Python."""
        versions = get_dependency_versions()
        assert "Python" in versions
        assert len(versions["Python"]) > 0

    def test_includes_matplotlib_version(self):
        """Versions must include matplotlib."""
        versions = get_dependency_versions()
        assert "matplotlib" in versions
        assert len(versions["matplotlib"]) > 0

    def test_includes_pandas_version(self):
        """Versions must include pandas."""
        versions = get_dependency_versions()
        assert "pandas" in versions
        assert len(versions["pandas"]) > 0

    def test_includes_scipy_version(self):
        """Versions must include scipy."""
        versions = get_dependency_versions()
        assert "scipy" in versions
        assert len(versions["scipy"]) > 0

    def test_includes_numpy_version(self):
        """Versions must include numpy."""
        versions = get_dependency_versions()
        assert "numpy" in versions
        assert len(versions["numpy"]) > 0

    def test_includes_goatools_version(self):
        """Versions must include goatools."""
        versions = get_dependency_versions()
        assert "goatools" in versions

    def test_includes_pyyaml_version(self):
        """Versions must include PyYAML."""
        versions = get_dependency_versions()
        assert "PyYAML" in versions

    def test_versions_are_not_empty_strings(self):
        """All version strings for installed packages must be non-empty."""
        versions = get_dependency_versions()
        for key in ["Python", "matplotlib", "pandas", "scipy", "numpy"]:
            assert versions[key] != "", f"{key} version must not be empty"


# ---------------------------------------------------------------------------
# Invariants: Pre-conditions and Post-conditions
# ---------------------------------------------------------------------------

class TestInvariants:
    """Pre-conditions and post-conditions from the blueprint invariants."""

    def test_output_dir_must_exist_precondition(self):
        """generate_notes must raise AssertionError if output_dir does not exist."""
        ni = _make_notes_input()
        nonexistent = Path("/tmp/nonexistent_dir_unit9_test_xyz")
        with pytest.raises(AssertionError, match="Output directory must exist"):
            generate_notes(ni, nonexistent)

    def test_output_file_named_notes_md(self, tmp_path):
        """Post-condition: output filename must be notes.md."""
        ni = _make_notes_input()
        result = generate_notes(ni, tmp_path)
        assert result.name == "notes.md"

    def test_output_file_exists_postcondition(self, tmp_path):
        """Post-condition: notes.md must exist after generation."""
        ni = _make_notes_input()
        result = generate_notes(ni, tmp_path)
        assert result.exists()


# ---------------------------------------------------------------------------
# Error Conditions
# ---------------------------------------------------------------------------

class TestErrorConditions:
    """Error conditions specified in the blueprint."""

    def test_oserror_on_write_failure(self, tmp_path):
        """OSError must be raised when notes.md cannot be written to output_dir."""
        ni = _make_notes_input()
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        # Create a file named notes.md that is a directory to force write failure
        blocker = readonly_dir / "notes.md"
        blocker.mkdir()
        with pytest.raises(OSError):
            generate_notes(ni, readonly_dir)


# ---------------------------------------------------------------------------
# Signature / Return Type Tests
# ---------------------------------------------------------------------------

class TestSignatures:
    """Verify function signatures and return types match the blueprint."""

    def test_generate_notes_returns_path(self, tmp_path):
        """generate_notes must return a Path object."""
        ni = _make_notes_input()
        result = generate_notes(ni, tmp_path)
        assert isinstance(result, Path)

    def test_format_figure_legends_returns_str(self):
        """format_figure_legends must return a string."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        assert isinstance(result, str)

    def test_format_methods_text_returns_str(self):
        """format_methods_text must return a string."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        assert isinstance(result, str)

    def test_format_summary_statistics_returns_str(self):
        """format_summary_statistics must return a string."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        assert isinstance(result, str)

    def test_format_reproducibility_note_returns_str(self):
        """format_reproducibility_note must return a string."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        assert isinstance(result, str)

    def test_format_config_guide_returns_str(self):
        """format_config_guide must return a string."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        assert isinstance(result, str)

    def test_get_dependency_versions_returns_dict(self):
        """get_dependency_versions must return dict[str, str]."""
        result = get_dependency_versions()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# NotesInput dataclass tests
# ---------------------------------------------------------------------------

class TestNotesInputDataclass:
    """Verify the NotesInput dataclass structure per the blueprint."""

    def test_notes_input_accepts_none_for_fig1_result(self):
        """NotesInput.fig1_result must accept None."""
        ni = _make_notes_input(with_fig1=False)
        assert ni.fig1_result is None

    def test_notes_input_accepts_none_for_clustering_result(self):
        """NotesInput.clustering_result must accept None."""
        ni = _make_notes_input(with_clustering=False)
        assert ni.clustering_result is None

    def test_notes_input_has_required_fields(self):
        """NotesInput must have all required fields from the blueprint."""
        ni = _make_notes_input()
        assert hasattr(ni, "cohort")
        assert hasattr(ni, "config")
        assert hasattr(ni, "fig1_result")
        assert hasattr(ni, "fig2_result")
        assert hasattr(ni, "fig3_result")
        assert hasattr(ni, "unbiased_stats")
        assert hasattr(ni, "fisher_result")
        assert hasattr(ni, "clustering_result")


# ---------------------------------------------------------------------------
# Integration: Full pipeline with various configurations
# ---------------------------------------------------------------------------

class TestFullPipelineVariations:
    """End-to-end tests with different input configurations."""

    def test_full_output_with_all_features(self, tmp_path):
        """Full generation with fig1, clustering, all features enabled."""
        ni = _make_notes_input(with_fig1=True, with_clustering=True)
        path = generate_notes(ni, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "Figure 1" in content
        assert "Figure 2" in content
        assert "Figure 3" in content
        assert "cluster" in content.lower()

    def test_full_output_without_figure1(self, tmp_path):
        """Full generation without Figure 1 (fig1_result=None)."""
        ni = _make_notes_input(with_fig1=False, with_clustering=True)
        path = generate_notes(ni, tmp_path)
        content = path.read_text(encoding="utf-8")
        # Figure 1 may still appear in config guide as a reference, but
        # legend and summary should not describe Figure 1 results
        assert "Figure 2" in content
        assert "Figure 3" in content

    def test_full_output_without_clustering(self, tmp_path):
        """Full generation without clustering (clustering_result=None)."""
        ni = _make_notes_input(with_fig1=True, with_clustering=False)
        path = generate_notes(ni, tmp_path)
        content = path.read_text(encoding="utf-8")
        lower = content.lower()
        # Should mention that clustering was not applied
        assert "not applied" in lower or ("not" in lower and "cluster" in lower)

    def test_full_output_without_fig1_or_clustering(self, tmp_path):
        """Full generation with minimal features: no fig1, no clustering."""
        ni = _make_notes_input(with_fig1=False, with_clustering=False)
        path = generate_notes(ni, tmp_path)
        content = path.read_text(encoding="utf-8")
        # Figure 1 may appear in config guide as a reference
        assert "Figure 2" in content
        assert "Figure 3" in content

    def test_custom_config_values_reflected_in_output(self, tmp_path):
        """Non-default config values must appear in the generated notes."""
        cohort = _make_cohort_data()
        config = _make_tool_config(
            dot_plot={"fdr_threshold": 0.1, "top_n": 30, "n_groups": 6, "random_seed": 99},
            fisher={"top_n_bars": 15},
        )
        fig2 = _make_dot_plot_result(stem="figure2", n_terms=30)
        fig3 = _make_bar_plot_result(n_bars=15, clustering_was_used=True)
        ni = NotesInput(
            cohort=cohort,
            config=config,
            fig1_result=None,
            fig1_method=None,
            fig2_result=fig2,
            fig3_result=fig3,
            unbiased_stats=_make_unbiased_stats(n_clusters=6, random_seed=99),
            fisher_result=_make_fisher_result(),
            clustering_result=_make_clustering_result(),
        )
        path = generate_notes(ni, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "0.1" in content   # fdr_threshold
        assert "99" in content    # random_seed
        assert "30" in content    # top_n
