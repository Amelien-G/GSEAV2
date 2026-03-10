"""
Tests for Unit 9 -- Notes Generation -- Coverage Gap Tests

These tests cover behavioral contracts identified as gaps in the existing
test suite (test_notes_generation.py). Each test targets a specific blueprint
contract or sub-contract that was not adequately exercised.

DATA ASSUMPTIONS (module-level):
- Reuses the same synthetic data construction helpers from the existing test
  suite. All data assumptions remain the same.
- Synthetic CohortData uses 3 mutant lines (mutA, mutB, mutC) with 5 GO terms.
- UnbiasedSelectionStats uses 100 significant terms, 80 after dedup, 20 selected,
  4 clusters, seed 42, Ward linkage.
- FisherResult uses 5 GO terms with combined p-values: 1e-10, 1e-6, 1e-4, 0.01, 0.5.
- ClusteringResult uses 3 clusters, 5 prefiltered terms, Lin similarity at 0.7.
- ToolConfig uses all default values unless specifically overridden.
- BarPlotResult uses n_bars=10, representing 10 bars in Figure 3.
- DotPlotResult for fig1 uses n_terms=15 and fig2 uses n_terms=20.
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
# Helpers for synthetic data generation (same as existing test suite)
# ---------------------------------------------------------------------------


def _make_cohort_data(n_mutants: int = 3) -> CohortData:
    """Build a synthetic CohortData with n_mutants mutant lines.

    DATA ASSUMPTION: 3 mutant lines (mutA, mutB, mutC) each with 5 GO terms.
    """
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
                term_name=tname,
                go_id=gid,
                nes=nes,
                fdr=fdr,
                nom_pval=pval,
                size=size,
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
    """Build a synthetic ToolConfig with all defaults."""
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
    """Build a synthetic DotPlotResult."""
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
    """Build a synthetic BarPlotResult."""
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
    """Build a synthetic UnbiasedSelectionStats."""
    return UnbiasedSelectionStats(
        total_significant_terms=total_significant_terms,
        terms_after_dedup=terms_after_dedup,
        terms_selected=terms_selected,
        n_clusters=n_clusters,
        random_seed=random_seed,
        clustering_algorithm=clustering_algorithm,
    )


def _make_fisher_result(
    n_mutants: int = 3,
) -> FisherResult:
    """Build a synthetic FisherResult."""
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
    """Build a synthetic ClusteringResult."""
    representatives = ["GO:0070125", "GO:0042254", "GO:0099504"]
    representative_names = [
        "MITOCHONDRIAL TRANSLATION",
        "RIBOSOME BIOGENESIS",
        "SYNAPTIC VESICLE CYCLE",
    ]
    representative_pvalues = [1e-10, 1e-4, 0.01]
    representative_n_contributing = [3, 2, 2]
    cluster_assignments = {
        "GO:0070125": 0,
        "GO:0006119": 0,
        "GO:0042254": 1,
        "GO:0099504": 2,
        "GO:0051726": 2,
    }
    return ClusteringResult(
        representatives=representatives,
        representative_names=representative_names,
        representative_pvalues=representative_pvalues,
        representative_n_contributing=representative_n_contributing,
        cluster_assignments=cluster_assignments,
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
    cohort = _make_cohort_data()
    config = _make_tool_config()
    fig1_result = _make_dot_plot_result(stem="figure1") if with_fig1 else None
    fig2_result = _make_dot_plot_result(stem="figure2", n_terms=20, n_categories=4)
    fig3_result = _make_bar_plot_result(
        output_dir=output_dir,
        clustering_was_used=with_clustering,
    )
    unbiased_stats = _make_unbiased_stats()
    fisher_result = _make_fisher_result()
    clustering_result = _make_clustering_result() if with_clustering else None

    return NotesInput(
        cohort=cohort,
        config=config,
        fig1_result=fig1_result,
        fig1_method="ontology" if with_fig1 else None,
        fig2_result=fig2_result,
        fig3_result=fig3_result,
        unbiased_stats=unbiased_stats,
        fisher_result=fisher_result,
        clustering_result=clustering_result,
    )


# ---------------------------------------------------------------------------
# Gap 1: Contract 3 -- Category boxes described in dot plot legends
# Blueprint says: "what category boxes represent"
# Existing tests check NES, colormap, dot size, empty cells, FDR, mutant
# count, but no test verifies that "category" is mentioned in the legends.
# ---------------------------------------------------------------------------


class TestFigureLegendsCategories:
    """Contract 3 gap: Figure legends must describe category boxes."""

    def test_dot_plot_legend_mentions_category_boxes(self):
        """Legend for Figures 1/2 must describe what category boxes represent.

        DATA ASSUMPTION: Default notes input with both fig1 and fig2 present.
        The blueprint states the legend must describe 'what category boxes represent'.
        """
        ni = _make_notes_input(with_fig1=True)
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "categor" in result_lower, \
            "Legend must describe category boxes for the dot plots"

    def test_fig2_legend_mentions_category_boxes_without_fig1(self):
        """Even when Figure 1 is omitted, Figure 2 legend must still describe
        category boxes.

        DATA ASSUMPTION: Notes input with fig1=None. Figure 2 legend should
        still include category box description.
        """
        ni = _make_notes_input(with_fig1=False)
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "categor" in result_lower, \
            "Figure 2 legend must describe category boxes even when Figure 1 is omitted"


# ---------------------------------------------------------------------------
# Gap 2: Contract 4 -- Information content source in methods
# Blueprint says: "information content source" for GO semantic similarity.
# Existing tests check for "lin" and threshold but not for "information content".
# ---------------------------------------------------------------------------


class TestMethodsInformationContent:
    """Contract 4 gap: Methods must describe information content source."""

    def test_methods_mentions_information_content_for_clustering(self):
        """Methods text must mention information content as the basis for
        semantic similarity computation.

        DATA ASSUMPTION: Default notes input with clustering enabled.
        The blueprint states the methods should describe 'information content source'.
        """
        ni = _make_notes_input(with_clustering=True)
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "information content" in result_lower, \
            "Methods must describe information content as the source for semantic similarity"


# ---------------------------------------------------------------------------
# Gap 3: Contract 5 -- Fisher pre-filter count reported without clustering
# Blueprint says: "number of GO terms passing the Fisher pre-filter"
# The existing test only verifies this count with clustering enabled.
# When clustering_result is None, the implementation computes the count
# differently and it must still be reported.
# ---------------------------------------------------------------------------


class TestSummaryFisherPrefilterNoClustering:
    """Contract 5 gap: Fisher pre-filter count must be reported even when
    clustering is disabled."""

    def test_summary_reports_fisher_prefilter_count_without_clustering(self):
        """When clustering is disabled, summary must still report the number
        of GO terms passing the Fisher pre-filter.

        DATA ASSUMPTION: Notes input with clustering disabled. The Fisher result
        has 5 GO terms with combined p-values [1e-10, 1e-6, 1e-4, 0.01, 0.5].
        Default prefilter_pvalue is 0.05, so 4 terms should pass (all except 0.5).
        The summary must include this count.
        """
        ni = _make_notes_input(with_clustering=False)
        result = format_summary_statistics(ni)
        result_lower = result.lower()
        assert "pre-filter" in result_lower or "prefilter" in result_lower or \
               "pre filter" in result_lower or "fisher" in result_lower, \
            "Summary must mention Fisher pre-filter even when clustering is disabled"
        # With default prefilter_pvalue=0.05, 4 of our 5 synthetic terms pass
        assert "4" in result, \
            "Summary must report the correct Fisher pre-filter count (4) without clustering"


# ---------------------------------------------------------------------------
# Gap 4: Contract 5 -- Figure 3 bar count reported in summary statistics
# Blueprint says: "number of GO terms displayed in each produced figure"
# Existing test only checks for fig1 (15) or fig2 (20) term counts.
# No test verifies that Figure 3's bar count (10) is reported.
# ---------------------------------------------------------------------------


class TestSummaryFigure3BarCount:
    """Contract 5 gap: Summary must report number of terms displayed in Figure 3."""

    def test_summary_reports_figure3_bar_count(self):
        """Summary statistics must report the number of GO terms displayed
        in Figure 3 (the bar count).

        DATA ASSUMPTION: BarPlotResult uses n_bars=10 by default.
        The summary must include '10' as the Figure 3 term count.
        """
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        # The bar plot has 10 bars; this count must appear
        assert "10" in result, \
            "Summary must report the number of GO terms displayed in Figure 3 (n_bars=10)"

    def test_summary_reports_both_fig1_and_fig2_term_counts(self):
        """Summary statistics must report term counts for both Figure 1 and
        Figure 2 when both are produced.

        DATA ASSUMPTION: fig1 n_terms=15, fig2 n_terms=20. Both values must
        appear in the summary.
        """
        ni = _make_notes_input(with_fig1=True)
        result = format_summary_statistics(ni)
        assert "15" in result, \
            "Summary must report Figure 1 term count (15)"
        assert "20" in result, \
            "Summary must report Figure 2 term count (20)"


# ---------------------------------------------------------------------------
# Gap 5: Contract 6 -- Reproducibility note includes all config sections
# Blueprint says: "all configuration parameters used in the run"
# Existing test only loosely checks for "fdr"/"threshold"/"config".
# No test verifies that all four config sections are represented.
# ---------------------------------------------------------------------------


class TestReproducibilityAllConfigSections:
    """Contract 6 gap: Reproducibility note must include parameters from all
    configuration sections."""

    def test_reproducibility_includes_dot_plot_config(self):
        """Reproducibility note must include dot_plot configuration parameters.

        DATA ASSUMPTION: Default ToolConfig with dot_plot settings:
        fdr_threshold=0.05, top_n=20, n_groups=4, random_seed=42.
        """
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        result_lower = result.lower()
        assert "fdr" in result_lower and "threshold" in result_lower, \
            "Reproducibility note must include dot_plot FDR threshold"
        assert "top" in result_lower or "top_n" in result_lower, \
            "Reproducibility note must include dot_plot top_n setting"

    def test_reproducibility_includes_fisher_config(self):
        """Reproducibility note must include Fisher configuration parameters.

        DATA ASSUMPTION: Default ToolConfig with Fisher settings including
        pseudocount, apply_fdr, fdr_threshold, prefilter_pvalue, top_n_bars.
        """
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        result_lower = result.lower()
        assert "fisher" in result_lower or "pseudocount" in result_lower, \
            "Reproducibility note must include Fisher configuration parameters"

    def test_reproducibility_includes_clustering_config(self):
        """Reproducibility note must include clustering configuration parameters.

        DATA ASSUMPTION: Default ToolConfig with clustering settings including
        enabled, similarity_metric, similarity_threshold.
        """
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        result_lower = result.lower()
        assert "clustering" in result_lower, \
            "Reproducibility note must include clustering configuration parameters"
        assert "similarity" in result_lower, \
            "Reproducibility note must include similarity metric/threshold settings"

    def test_reproducibility_includes_plot_appearance_config(self):
        """Reproducibility note must include plot appearance configuration.

        DATA ASSUMPTION: Default ToolConfig with plot_appearance settings
        including DPI, font_family, bar_colormap, etc.
        """
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        result_lower = result.lower()
        assert "dpi" in result_lower or "font" in result_lower or \
               "appearance" in result_lower or "plot" in result_lower, \
            "Reproducibility note must include plot appearance configuration"


# ---------------------------------------------------------------------------
# Gap 6: Contract 3 -- fig1_method="tsv" legend path
# Blueprint says: Figure 1 legend "additionally states whether categories
# were resolved via GO ontology ancestry ... or via a user-supplied category
# mapping file, depending on fig1_method."
# No existing test exercises the fig1_method="tsv" code path.
# ---------------------------------------------------------------------------


from gsea_tool.configuration import CherryPickCategory


def _make_notes_input_with_fig1_method(
    fig1_method: str,
    with_clustering: bool = True,
    output_dir: Path = Path("/tmp/output"),
) -> NotesInput:
    """Build a NotesInput with a specific fig1_method and populated
    cherry_pick_categories when fig1_method is 'ontology'.

    DATA ASSUMPTION: When fig1_method='ontology', cherry_pick_categories
    contains two parent GO IDs: GO:0005739 (Mitochondrion) and
    GO:0005634 (Nucleus). When fig1_method='tsv', cherry_pick_categories
    is empty (categories come from the TSV file instead).
    """
    cohort = _make_cohort_data()
    if fig1_method == "ontology":
        config = ToolConfig(
            cherry_pick_categories=[
                CherryPickCategory(go_id="GO:0005739", label="Mitochondrion"),
                CherryPickCategory(go_id="GO:0005634", label="Nucleus"),
            ],
            dot_plot=DotPlotConfig(),
            fisher=FisherConfig(),
            clustering=ClusteringConfig(),
            plot_appearance=PlotAppearanceConfig(),
        )
    else:
        config = _make_tool_config()
    fig1_result = _make_dot_plot_result(stem="figure1")
    fig2_result = _make_dot_plot_result(stem="figure2", n_terms=20, n_categories=4)
    fig3_result = _make_bar_plot_result(
        output_dir=output_dir,
        clustering_was_used=with_clustering,
    )
    unbiased_stats = _make_unbiased_stats()
    fisher_result = _make_fisher_result()
    clustering_result = _make_clustering_result() if with_clustering else None

    return NotesInput(
        cohort=cohort,
        config=config,
        fig1_result=fig1_result,
        fig1_method=fig1_method,
        fig2_result=fig2_result,
        fig3_result=fig3_result,
        unbiased_stats=unbiased_stats,
        fisher_result=fisher_result,
        clustering_result=clustering_result,
    )


class TestFig1MethodTsvLegend:
    """Contract 3 gap: Figure 1 legend must describe user-supplied category
    mapping file when fig1_method='tsv'."""

    def test_fig1_legend_tsv_mentions_user_supplied_mapping(self):
        """When fig1_method='tsv', Figure 1 legend must state that categories
        were resolved via a user-supplied category mapping file.

        DATA ASSUMPTION: NotesInput with fig1_method='tsv' and fig1_result
        present. The legend text should mention 'user-supplied' or 'mapping'.
        """
        ni = _make_notes_input_with_fig1_method(fig1_method="tsv")
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "user" in result_lower or "mapping" in result_lower or \
               "tsv" in result_lower, \
            "Figure 1 legend must mention user-supplied category mapping for tsv method"

    def test_fig1_legend_tsv_does_not_mention_ontology(self):
        """When fig1_method='tsv', Figure 1 legend must NOT describe ontology
        ancestry resolution.

        DATA ASSUMPTION: NotesInput with fig1_method='tsv'. The legend should
        not mention 'ontology ancestry' or 'parent GO IDs' in the category
        resolution context.
        """
        ni = _make_notes_input_with_fig1_method(fig1_method="tsv")
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "ontology ancestry" not in result_lower, \
            "TSV legend must not mention ontology ancestry"


# ---------------------------------------------------------------------------
# Gap 7: Contract 3 -- fig1_method="ontology" legend mentions parent GO IDs
# Blueprint says: "the legend additionally states whether categories were
# resolved via GO ontology ancestry (naming the parent GO IDs and their labels)"
# Existing tests use fig1_method="ontology" but with an empty
# cherry_pick_categories list. No test verifies that parent GO IDs and labels
# actually appear in the legend text.
# ---------------------------------------------------------------------------


class TestFig1MethodOntologyLegend:
    """Contract 3 gap: Figure 1 legend must name parent GO IDs and labels
    when fig1_method='ontology'."""

    def test_fig1_legend_ontology_mentions_parent_go_ids(self):
        """When fig1_method='ontology', the legend must mention the parent
        GO IDs used for category resolution.

        DATA ASSUMPTION: cherry_pick_categories contains GO:0005739
        (Mitochondrion) and GO:0005634 (Nucleus). Both GO IDs must appear
        in the legend text.
        """
        ni = _make_notes_input_with_fig1_method(fig1_method="ontology")
        result = format_figure_legends(ni)
        assert "GO:0005739" in result, \
            "Legend must name parent GO ID GO:0005739"
        assert "GO:0005634" in result, \
            "Legend must name parent GO ID GO:0005634"

    def test_fig1_legend_ontology_mentions_parent_labels(self):
        """When fig1_method='ontology', the legend must mention the parent
        labels used for category resolution.

        DATA ASSUMPTION: cherry_pick_categories has labels 'Mitochondrion'
        and 'Nucleus'. Both labels must appear in the legend text.
        """
        ni = _make_notes_input_with_fig1_method(fig1_method="ontology")
        result = format_figure_legends(ni)
        assert "Mitochondrion" in result, \
            "Legend must name parent label Mitochondrion"
        assert "Nucleus" in result, \
            "Legend must name parent label Nucleus"

    def test_fig1_legend_ontology_mentions_ancestry(self):
        """When fig1_method='ontology', the legend must mention ontology
        ancestry resolution.

        DATA ASSUMPTION: Default ontology-based NotesInput. The legend
        must contain 'ontology' indicating the resolution approach.
        """
        ni = _make_notes_input_with_fig1_method(fig1_method="ontology")
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "ontology" in result_lower, \
            "Legend must mention ontology for ontology-based category resolution"


# ---------------------------------------------------------------------------
# Gap 8: Contract 4 -- fig1_method="tsv" methods path
# Blueprint says: "manual category mapping when fig1_method == 'tsv'"
# No existing test exercises the TSV code path in format_methods_text.
# ---------------------------------------------------------------------------


class TestMethodsFig1Tsv:
    """Contract 4 gap: Methods text must describe manual category mapping
    when fig1_method='tsv'."""

    def test_methods_tsv_mentions_manual_mapping(self):
        """When fig1_method='tsv', methods must describe manual category
        mapping from user-supplied file.

        DATA ASSUMPTION: NotesInput with fig1_method='tsv'. Methods text
        should mention 'manual' or 'user-supplied' or 'mapping' or 'TSV'.
        """
        ni = _make_notes_input_with_fig1_method(fig1_method="tsv")
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "manual" in result_lower or "mapping" in result_lower or \
               "tsv" in result_lower or "user" in result_lower, \
            "Methods must describe manual category mapping for tsv method"

    def test_methods_tsv_includes_figure1_section(self):
        """When fig1_method='tsv', methods must include a Figure 1 section.

        DATA ASSUMPTION: NotesInput with fig1_method='tsv' and fig1_result
        present. Methods should contain 'Figure 1'.
        """
        ni = _make_notes_input_with_fig1_method(fig1_method="tsv")
        result = format_methods_text(ni)
        assert "Figure 1" in result, \
            "Methods must include Figure 1 section when fig1_method='tsv'"


# ---------------------------------------------------------------------------
# Gap 9: Contract 4 -- fig1_method="ontology" methods mentions parent GO IDs
# Blueprint says: "ontology-based resolution from parent GO IDs"
# Existing tests set fig1_method="ontology" with empty cherry_pick_categories.
# No test verifies parent GO IDs appear in the methods text.
# ---------------------------------------------------------------------------


class TestMethodsFig1Ontology:
    """Contract 4 gap: Methods text must name parent GO IDs when
    fig1_method='ontology'."""

    def test_methods_ontology_mentions_parent_go_ids(self):
        """When fig1_method='ontology', methods must mention the parent GO IDs
        used for category resolution.

        DATA ASSUMPTION: cherry_pick_categories contains GO:0005739 and
        GO:0005634. Both must appear in the methods text.
        """
        ni = _make_notes_input_with_fig1_method(fig1_method="ontology")
        result = format_methods_text(ni)
        assert "GO:0005739" in result, \
            "Methods must name parent GO ID GO:0005739"
        assert "GO:0005634" in result, \
            "Methods must name parent GO ID GO:0005634"

    def test_methods_ontology_mentions_ontology_based_resolution(self):
        """When fig1_method='ontology', methods must describe ontology-based
        resolution approach.

        DATA ASSUMPTION: NotesInput with fig1_method='ontology'. Methods
        should mention 'ontology' in the Figure 1 selection criteria.
        """
        ni = _make_notes_input_with_fig1_method(fig1_method="ontology")
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "ontology" in result_lower, \
            "Methods must describe ontology-based resolution for Figure 1"


# ---------------------------------------------------------------------------
# Gap 10: NotesInput.fig1_method field presence
# Blueprint signature shows fig1_method as a field on NotesInput. No test
# verifies this field exists and is accessible.
# ---------------------------------------------------------------------------


class TestNotesInputFig1MethodField:
    """Verify that NotesInput has a fig1_method field per the blueprint."""

    def test_notes_input_has_fig1_method_field(self):
        """NotesInput must have a fig1_method field.

        DATA ASSUMPTION: Default NotesInput with fig1_method='ontology'.
        """
        ni = _make_notes_input(with_fig1=True)
        assert hasattr(ni, "fig1_method"), \
            "NotesInput must have a fig1_method field"
        assert ni.fig1_method == "ontology"

    def test_notes_input_fig1_method_none_when_no_fig1(self):
        """NotesInput.fig1_method must be None when Figure 1 is not produced.

        DATA ASSUMPTION: NotesInput with with_fig1=False sets fig1_method=None.
        """
        ni = _make_notes_input(with_fig1=False)
        assert ni.fig1_method is None, \
            "fig1_method must be None when Figure 1 is not produced"
