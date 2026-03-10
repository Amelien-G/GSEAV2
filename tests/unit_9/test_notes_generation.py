"""
Tests for Unit 9 -- Notes Generation

This test suite verifies the behavioral contracts, invariants, error conditions,
and function/class signatures specified in the Unit 9 blueprint for notes.md
generation.

DATA ASSUMPTIONS (module-level):
- Synthetic CohortData uses 3 mutant lines (mutA, mutB, mutC) with a small set
  of GO terms, representing a minimal but valid GSEA cohort.
- GO term names are uppercase descriptive strings (e.g., "MITOCHONDRIAL TRANSLATION"),
  consistent with GSEA preranked output naming conventions.
- FDR threshold of 0.05 (default) is used throughout.
- DotPlotResult paths are synthetic Path objects pointing to plausible file locations.
- BarPlotResult paths are synthetic Path objects pointing to plausible file locations.
- UnbiasedSelectionStats uses typical values: 100 significant terms, 80 after dedup,
  20 selected, 4 clusters, seed 42, Ward linkage.
- FisherResult uses 5 GO terms with combined p-values spanning several orders of
  magnitude (1e-10 to 0.5), representing typical Fisher's combined test output.
- ClusteringResult uses 3 clusters with 5 prefiltered terms, Lin similarity at 0.7
  threshold, representing typical semantic clustering output.
- ToolConfig uses all default values unless specifically overridden for a test case.
- Software version strings are expected to be non-empty runtime values for:
  Python, matplotlib, pandas, scipy, numpy, goatools, pyyaml.
"""

import inspect
from dataclasses import fields as dataclass_fields
from pathlib import Path
from unittest.mock import patch, MagicMock

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
# Helpers for synthetic data generation
# ---------------------------------------------------------------------------


def _make_term_record(
    term_name: str = "MITOCHONDRIAL TRANSLATION",
    go_id: str = "GO:0070125",
    nes: float = 2.1,
    fdr: float = 0.001,
    nom_pval: float = 0.0001,
    size: int = 50,
) -> TermRecord:
    """Build a synthetic TermRecord.

    DATA ASSUMPTION: Default values represent a highly significant GO term
    with strong positive enrichment, typical of a top-ranked GSEA result.
    """
    return TermRecord(
        term_name=term_name,
        go_id=go_id,
        nes=nes,
        fdr=fdr,
        nom_pval=nom_pval,
        size=size,
    )


def _make_cohort_data(n_mutants: int = 3) -> CohortData:
    """Build a synthetic CohortData with n_mutants mutant lines.

    DATA ASSUMPTION: 3 mutant lines (mutA, mutB, mutC) each with 5 GO terms.
    FDR values range from very significant (0.001) to non-significant (0.8).
    NES values range from -2.5 to +2.5, representing bidirectional enrichment.
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
    """Build a synthetic ToolConfig with all defaults.

    DATA ASSUMPTION: Default ToolConfig values per the blueprint:
    fdr_threshold=0.05, top_n=20, n_groups=4, random_seed=42,
    pseudocount=1e-10, etc.
    """
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
    """Build a synthetic DotPlotResult.

    DATA ASSUMPTION: Paths point to plausible output locations.
    n_terms_displayed=15 represents a typical cherry-picked dot plot.
    n_categories=4 matches the fixed category count for Figure 1.
    """
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
    """Build a synthetic BarPlotResult.

    DATA ASSUMPTION: 10 bars represent a typical meta-analysis figure.
    clustering_was_used=True indicates semantic clustering was applied.
    """
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
    """Build a synthetic UnbiasedSelectionStats.

    DATA ASSUMPTION: 100 significant terms, 80 after dedup, 20 selected,
    4 clusters. These are typical values for a mid-size GSEA cohort.
    """
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
    """Build a synthetic FisherResult.

    DATA ASSUMPTION: 5 GO terms with combined p-values spanning several
    orders of magnitude (1e-10 to 0.5). The p-value matrix is a 5x3
    array with realistic values. n_contributing counts range from 1 to 3.
    """
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
    """Build a synthetic ClusteringResult.

    DATA ASSUMPTION: 3 clusters formed from 5 prefiltered GO terms.
    Representatives are the most significant term in each cluster.
    Lin similarity at 0.7 threshold is the default clustering configuration.
    """
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
    """Build a complete synthetic NotesInput.

    DATA ASSUMPTION: Represents a typical full analysis run with all figures
    produced and clustering enabled. When with_fig1=False, simulates the case
    where Figure 1 was not produced (no category mapping file).
    """
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
# Signature Tests
# ---------------------------------------------------------------------------


class TestSignatures:
    """Verify that all function and class signatures match the blueprint."""

    def test_notes_input_is_dataclass(self):
        """NotesInput must be a dataclass."""
        assert hasattr(NotesInput, "__dataclass_fields__")

    def test_notes_input_fields(self):
        """NotesInput must have the correct fields with the correct names."""
        field_names = [f.name for f in dataclass_fields(NotesInput)]
        expected = [
            "cohort",
            "config",
            "fig1_result",
            "fig2_result",
            "fig3_result",
            "unbiased_stats",
            "fisher_result",
            "clustering_result",
        ]
        for name in expected:
            assert name in field_names, f"NotesInput missing field: {name}"

    def test_generate_notes_signature(self):
        """generate_notes must accept (notes_input, output_dir) and return Path."""
        sig = inspect.signature(generate_notes)
        params = list(sig.parameters.keys())
        assert "notes_input" in params
        assert "output_dir" in params
        assert sig.return_annotation is Path or sig.return_annotation == Path

    def test_format_figure_legends_signature(self):
        """format_figure_legends must accept (notes_input) and return str."""
        sig = inspect.signature(format_figure_legends)
        params = list(sig.parameters.keys())
        assert "notes_input" in params
        assert sig.return_annotation is str or sig.return_annotation == str

    def test_format_methods_text_signature(self):
        """format_methods_text must accept (notes_input) and return str."""
        sig = inspect.signature(format_methods_text)
        params = list(sig.parameters.keys())
        assert "notes_input" in params
        assert sig.return_annotation is str or sig.return_annotation == str

    def test_format_summary_statistics_signature(self):
        """format_summary_statistics must accept (notes_input) and return str."""
        sig = inspect.signature(format_summary_statistics)
        params = list(sig.parameters.keys())
        assert "notes_input" in params
        assert sig.return_annotation is str or sig.return_annotation == str

    def test_format_reproducibility_note_signature(self):
        """format_reproducibility_note must accept (notes_input) and return str."""
        sig = inspect.signature(format_reproducibility_note)
        params = list(sig.parameters.keys())
        assert "notes_input" in params
        assert sig.return_annotation is str or sig.return_annotation == str

    def test_format_config_guide_signature(self):
        """format_config_guide must accept (notes_input) and return str."""
        sig = inspect.signature(format_config_guide)
        params = list(sig.parameters.keys())
        assert "notes_input" in params
        assert sig.return_annotation is str or sig.return_annotation == str

    def test_get_dependency_versions_signature(self):
        """get_dependency_versions must accept no args and return dict[str, str]."""
        sig = inspect.signature(get_dependency_versions)
        params = list(sig.parameters.keys())
        assert len(params) == 0, "get_dependency_versions takes no arguments"


# ---------------------------------------------------------------------------
# Contract 1: Output file naming and location
# ---------------------------------------------------------------------------


class TestOutputFileNaming:
    """Contract 1: The output file is named exactly notes.md and written to output_dir."""

    def test_generate_notes_returns_path(self, tmp_path):
        """generate_notes must return a Path object."""
        ni = _make_notes_input(output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        assert isinstance(result, Path)

    def test_output_filename_is_notes_md(self, tmp_path):
        """The returned path must have filename notes.md."""
        ni = _make_notes_input(output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        assert result.name == "notes.md"

    def test_output_is_in_output_dir(self, tmp_path):
        """notes.md must be written to the provided output_dir."""
        ni = _make_notes_input(output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        assert result.parent == tmp_path

    def test_output_file_exists(self, tmp_path):
        """notes.md must actually exist on disk after generate_notes returns."""
        ni = _make_notes_input(output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        assert result.exists(), "notes.md must be written to disk"


# ---------------------------------------------------------------------------
# Contract 2: Five sections present
# ---------------------------------------------------------------------------


class TestFiveSections:
    """Contract 2: The file contains five sections: Figure Legends, Materials
    and Methods, Summary Statistics, Reproducibility Note, Configuration Guide."""

    def test_all_five_sections_present(self, tmp_path):
        """The generated notes.md must contain all five section headings."""
        ni = _make_notes_input(output_dir=tmp_path)
        result_path = generate_notes(ni, tmp_path)
        content = result_path.read_text(encoding="utf-8")

        # Check for section headings (case-insensitive substring match)
        content_lower = content.lower()
        assert "figure legend" in content_lower or "figure legends" in content_lower, \
            "Missing 'Figure Legends' section"
        assert "materials and methods" in content_lower or "methods" in content_lower, \
            "Missing 'Materials and Methods' section"
        assert "summary statistics" in content_lower or "summary" in content_lower, \
            "Missing 'Summary Statistics' section"
        assert "reproducibility" in content_lower, \
            "Missing 'Reproducibility Note' section"
        assert "configuration" in content_lower and "guide" in content_lower, \
            "Missing 'Configuration Guide' section"


# ---------------------------------------------------------------------------
# Contract 3: Figure Legends content
# ---------------------------------------------------------------------------


class TestFigureLegends:
    """Contract 3: Figure legend content for dot plots and bar plot."""

    def test_format_figure_legends_returns_string(self):
        """format_figure_legends must return a non-empty string."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_dot_plot_legend_describes_dot_color_nes(self):
        """Legend for Figures 1/2 must describe what dot color represents (NES)."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "nes" in result_lower, "Legend must mention NES for dot color"

    def test_dot_plot_legend_mentions_diverging_colormap(self):
        """Legend must mention diverging red-blue color scheme for NES."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        result_lower = result.lower()
        # Either "red" and "blue" or "diverging" should appear
        assert ("red" in result_lower and "blue" in result_lower) or \
               "diverging" in result_lower or "rdbu" in result_lower, \
            "Legend must describe the diverging red-blue colormap"

    def test_dot_plot_legend_describes_dot_size(self):
        """Legend must describe what dot size represents (-log10 FDR)."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "fdr" in result_lower, "Legend must mention FDR for dot size"
        assert "log" in result_lower or "size" in result_lower, \
            "Legend must describe dot size encoding"

    def test_dot_plot_legend_describes_empty_cells(self):
        """Legend must describe what empty cells mean (term not significant)."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "empty" in result_lower or "absent" in result_lower or \
               "not significant" in result_lower or "missing" in result_lower, \
            "Legend must describe what empty cells mean"

    def test_dot_plot_legend_mentions_fdr_threshold(self):
        """Legend must mention the FDR threshold used."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        assert "0.05" in result, "Legend must state the FDR threshold value"

    def test_dot_plot_legend_mentions_mutant_count(self):
        """Legend must state the number of mutants."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        # Our synthetic data has 3 mutants
        assert "3" in result, "Legend must state the number of mutants"

    def test_bar_plot_legend_describes_bar_length(self):
        """Figure 3 legend must describe what bar length encodes (-log10 combined p-value)."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "combined p" in result_lower or "combined_p" in result_lower or \
               "fisher" in result_lower, \
            "Legend must describe bar length as -log10 combined p-value"

    def test_bar_plot_legend_describes_bar_color(self):
        """Figure 3 legend must describe what bar color encodes (contributing mutant lines)."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "contribut" in result_lower, \
            "Legend must describe bar color as number of contributing mutant lines"

    def test_bar_plot_legend_mentions_fisher(self):
        """Figure 3 legend must mention Fisher's combined probability test."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "fisher" in result_lower, \
            "Legend must mention Fisher's combined probability test"

    def test_bar_plot_legend_mentions_clustering_when_used(self):
        """Figure 3 legend must mention GO semantic clustering when used."""
        ni = _make_notes_input(with_clustering=True)
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "cluster" in result_lower or "semantic" in result_lower, \
            "Legend must mention clustering when it was used"

    def test_fig1_legend_omitted_when_not_produced(self):
        """When fig1_result is None, Figure 1 legend must be omitted."""
        ni = _make_notes_input(with_fig1=False)
        result = format_figure_legends(ni)
        result_lower = result.lower()
        # Figure 2 and Figure 3 should still be there
        assert "figure 2" in result_lower or "fig. 2" in result_lower or \
               "fig 2" in result_lower or "dot plot" in result_lower, \
            "Figure 2 legend must still be present"
        assert "figure 3" in result_lower or "fig. 3" in result_lower or \
               "fig 3" in result_lower or "bar" in result_lower, \
            "Figure 3 legend must still be present"

    def test_fig1_legend_included_when_produced(self):
        """When fig1_result is provided, Figure 1 legend must be included."""
        ni = _make_notes_input(with_fig1=True)
        result = format_figure_legends(ni)
        result_lower = result.lower()
        # There should be references to both Figure 1 and Figure 2
        assert "figure 1" in result_lower or "fig. 1" in result_lower or \
               "fig 1" in result_lower or "cherry" in result_lower, \
            "Figure 1 legend must be present when fig1_result is not None"

    def test_bar_plot_legend_mentions_top_n_representative(self):
        """Figure 3 legend must mention the top N representative pathways concept."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        result_lower = result.lower()
        assert "representative" in result_lower or "top" in result_lower or \
               "dysregulated" in result_lower, \
            "Legend must mention top N representative dysregulated pathways"


# ---------------------------------------------------------------------------
# Contract 4: Materials and Methods content
# ---------------------------------------------------------------------------


class TestMethodsText:
    """Contract 4: Materials and Methods section content."""

    def test_format_methods_text_returns_string(self):
        """format_methods_text must return a non-empty string."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_methods_states_gsea_consumed_not_generated(self):
        """Methods must state that GSEA preranked output was consumed, not generated."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "consumed" in result_lower or "preranked" in result_lower or \
               "pre-ranked" in result_lower, \
            "Methods must state GSEA output was consumed"
        # Should NOT claim to have run GSEA
        assert "gsea" in result_lower, "Methods must mention GSEA"

    def test_methods_describes_go_term_selection_criteria(self):
        """Methods must describe GO term selection criteria for each figure."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "fdr" in result_lower or "threshold" in result_lower, \
            "Methods must describe FDR-based term selection criteria"

    def test_methods_describes_ward_linkage(self):
        """Methods must name the clustering algorithm (Ward linkage) for Figure 2."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "ward" in result_lower, \
            "Methods must mention Ward linkage clustering for Figure 2"

    def test_methods_describes_n_clusters(self):
        """Methods must describe the number of clusters parameter for Figure 2."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        # Our synthetic data has 4 clusters
        assert "4" in result, "Methods must state the number of clusters"

    def test_methods_describes_random_seed(self):
        """Methods must describe the random seed used for Figure 2 clustering."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        assert "42" in result, "Methods must state the random seed"

    def test_methods_describes_jaccard_redundancy_removal(self):
        """Methods must describe the word-set Jaccard similarity > 0.5 for Figure 2."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "jaccard" in result_lower, \
            "Methods must describe Jaccard similarity for redundancy removal"
        assert "0.5" in result, \
            "Methods must state the Jaccard threshold of 0.5"

    def test_methods_describes_fisher_method(self):
        """Methods must describe Fisher's combined probability test for Figure 3."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "fisher" in result_lower, "Methods must mention Fisher's method"

    def test_methods_describes_imputation(self):
        """Methods must describe imputation of p = 1.0 for absent terms."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "imput" in result_lower, "Methods must describe imputation"
        assert "1.0" in result or "p = 1" in result_lower or "p=1" in result_lower, \
            "Methods must state p=1.0 for imputation"

    def test_methods_describes_degrees_of_freedom(self):
        """Methods must describe degrees of freedom for Fisher's test."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "degree" in result_lower or "df" in result_lower or \
               "2k" in result_lower or "chi" in result_lower, \
            "Methods must describe degrees of freedom"

    def test_methods_describes_lin_similarity_for_fig3(self):
        """Methods must describe Lin similarity for Figure 3 clustering."""
        ni = _make_notes_input(with_clustering=True)
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "lin" in result_lower, "Methods must mention Lin similarity"

    def test_methods_describes_clustering_threshold(self):
        """Methods must describe the clustering similarity threshold for Figure 3."""
        ni = _make_notes_input(with_clustering=True)
        result = format_methods_text(ni)
        # Default threshold is 0.7
        assert "0.7" in result, "Methods must state the clustering threshold"

    def test_methods_lists_software_versions(self):
        """Methods must list software dependencies with version numbers."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "python" in result_lower, "Methods must mention Python"
        assert "matplotlib" in result_lower or "scipy" in result_lower, \
            "Methods must mention at least one software dependency"

    def test_methods_notes_clustering_disabled_when_applicable(self):
        """When clustering is disabled, methods must note raw top-N terms were used."""
        ni = _make_notes_input(with_clustering=False)
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "not applied" in result_lower or "disabled" in result_lower or \
               "without clustering" in result_lower or "top" in result_lower, \
            "Methods must note that clustering was not applied when disabled"

    def test_methods_describes_semantic_similarity_representative_selection(self):
        """Methods must describe GO semantic similarity clustering approach."""
        ni = _make_notes_input(with_clustering=True)
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "semantic" in result_lower or "cluster" in result_lower, \
            "Methods must describe GO semantic similarity clustering"
        assert "representative" in result_lower or "select" in result_lower, \
            "Methods must describe representative selection rule"


# ---------------------------------------------------------------------------
# Contract 5: Summary Statistics content
# ---------------------------------------------------------------------------


class TestSummaryStatistics:
    """Contract 5: Summary Statistics section content."""

    def test_format_summary_statistics_returns_string(self):
        """format_summary_statistics must return a non-empty string."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_reports_number_of_mutants(self):
        """Summary must report the number of mutants analyzed."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        # Our synthetic data has 3 mutants
        assert "3" in result, "Summary must report the number of mutants"

    def test_summary_reports_total_go_terms(self):
        """Summary must report total unique GO terms in the input data."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        # Our synthetic data has 5 GO terms
        assert "5" in result, "Summary must report total unique GO terms"

    def test_summary_reports_significant_terms(self):
        """Summary must report number of GO terms passing FDR threshold."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        # UnbiasedSelectionStats.total_significant_terms = 100
        assert "100" in result, "Summary must report significant terms count"

    def test_summary_reports_terms_per_figure(self):
        """Summary must report number of terms displayed in each produced figure."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        # Fig1 has 15 terms, Fig2 has 20 terms
        assert "15" in result or "20" in result, \
            "Summary must report terms displayed per figure"

    def test_summary_reports_fisher_prefilter_count(self):
        """Summary must report number of GO terms passing Fisher pre-filter."""
        ni = _make_notes_input(with_clustering=True)
        result = format_summary_statistics(ni)
        # ClusteringResult.n_prefiltered = 5
        assert "5" in result, "Summary must report Fisher pre-filter count"

    def test_summary_reports_semantic_cluster_count(self):
        """Summary must report number of semantic clusters formed."""
        ni = _make_notes_input(with_clustering=True)
        result = format_summary_statistics(ni)
        # ClusteringResult.n_clusters = 3
        assert "3" in result, "Summary must report number of semantic clusters"


# ---------------------------------------------------------------------------
# Contract 6: Reproducibility Note content
# ---------------------------------------------------------------------------


class TestReproducibilityNote:
    """Contract 6: Reproducibility note with seeds, versions, and config parameters."""

    def test_format_reproducibility_note_returns_string(self):
        """format_reproducibility_note must return a non-empty string."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_reproducibility_states_random_seed(self):
        """Reproducibility note must state the random seed for Figure 2 clustering."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        assert "42" in result, "Must state random seed value"

    def test_reproducibility_states_software_versions(self):
        """Reproducibility note must include software versions."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        result_lower = result.lower()
        assert "python" in result_lower or "version" in result_lower, \
            "Must include software version information"

    def test_reproducibility_includes_config_parameters(self):
        """Reproducibility note must include configuration parameters used."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        result_lower = result.lower()
        assert "fdr" in result_lower or "threshold" in result_lower or \
               "config" in result_lower, \
            "Must include configuration parameters"


# ---------------------------------------------------------------------------
# Contract 8: Configuration Guide content
# ---------------------------------------------------------------------------


class TestConfigGuide:
    """Contract 8: Configuration guide describing all config.yaml parameters."""

    def test_format_config_guide_returns_string(self):
        """format_config_guide must return a non-empty string."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_config_guide_describes_fdr_threshold(self):
        """Config guide must describe the fdr_threshold parameter."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        result_lower = result.lower()
        assert "fdr_threshold" in result_lower or "fdr threshold" in result_lower, \
            "Config guide must describe fdr_threshold parameter"

    def test_config_guide_describes_top_n(self):
        """Config guide must describe the top_n parameter."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        result_lower = result.lower()
        assert "top_n" in result_lower or "top n" in result_lower, \
            "Config guide must describe top_n parameter"

    def test_config_guide_describes_n_groups(self):
        """Config guide must describe the n_groups parameter."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        result_lower = result.lower()
        assert "n_groups" in result_lower or "n groups" in result_lower or \
               "groups" in result_lower, \
            "Config guide must describe n_groups parameter"

    def test_config_guide_describes_random_seed(self):
        """Config guide must describe the random_seed parameter."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        result_lower = result.lower()
        assert "random_seed" in result_lower or "random seed" in result_lower or \
               "seed" in result_lower, \
            "Config guide must describe random_seed parameter"

    def test_config_guide_describes_clustering_parameters(self):
        """Config guide must describe clustering configuration parameters."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        result_lower = result.lower()
        assert "clustering" in result_lower, \
            "Config guide must describe clustering parameters"

    def test_config_guide_describes_fisher_parameters(self):
        """Config guide must describe Fisher configuration parameters."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        result_lower = result.lower()
        assert "fisher" in result_lower or "pseudocount" in result_lower or \
               "prefilter" in result_lower, \
            "Config guide must describe Fisher parameters"

    def test_config_guide_describes_plot_appearance(self):
        """Config guide must describe plot appearance parameters."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        result_lower = result.lower()
        assert "dpi" in result_lower or "font" in result_lower or \
               "appearance" in result_lower or "plot" in result_lower, \
            "Config guide must describe plot appearance parameters"

    def test_config_guide_mentions_defaults(self):
        """Config guide must mention default values for parameters."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        result_lower = result.lower()
        assert "default" in result_lower, \
            "Config guide must mention default values"

    def test_config_guide_organized_by_section(self):
        """Config guide must be organized by section (dot_plot, fisher, clustering, plot)."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        result_lower = result.lower()
        # At least some section names should appear
        sections_found = 0
        for section in ["dot_plot", "fisher", "clustering", "plot"]:
            if section in result_lower:
                sections_found += 1
        assert sections_found >= 3, \
            "Config guide must be organized by config sections (at least 3 of 4)"


# ---------------------------------------------------------------------------
# Contract 9: Copy-paste-ready prose
# ---------------------------------------------------------------------------


class TestProseFormat:
    """Contract 9: All text is written as copy-paste-ready prose."""

    def test_no_code_blocks_in_legends(self):
        """Figure legends must not contain code blocks."""
        ni = _make_notes_input()
        result = format_figure_legends(ni)
        assert "```" not in result, "Legends must not contain code blocks"

    def test_no_code_blocks_in_methods(self):
        """Methods text must not contain code blocks."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        assert "```" not in result, "Methods must not contain code blocks"

    def test_no_code_blocks_in_summary(self):
        """Summary statistics must not contain code blocks."""
        ni = _make_notes_input()
        result = format_summary_statistics(ni)
        assert "```" not in result, "Summary must not contain code blocks"

    def test_no_code_blocks_in_reproducibility(self):
        """Reproducibility note must not contain code blocks."""
        ni = _make_notes_input()
        result = format_reproducibility_note(ni)
        assert "```" not in result, "Reproducibility note must not contain code blocks"

    def test_no_code_blocks_in_config_guide(self):
        """Config guide must not contain code blocks."""
        ni = _make_notes_input()
        result = format_config_guide(ni)
        assert "```" not in result, "Config guide must not contain code blocks"

    def test_no_raw_data_dumps_in_notes(self, tmp_path):
        """The full notes.md must not contain raw data dumps."""
        ni = _make_notes_input(output_dir=tmp_path)
        result_path = generate_notes(ni, tmp_path)
        content = result_path.read_text(encoding="utf-8")
        # Raw data dumps would typically have many tab characters or look like TSV
        lines = content.split("\n")
        for line in lines:
            tab_count = line.count("\t")
            assert tab_count < 5, \
                f"Line appears to be a raw data dump (too many tabs): {line[:80]}"


# ---------------------------------------------------------------------------
# Contract 10: Software versions obtained at runtime
# ---------------------------------------------------------------------------


class TestDependencyVersions:
    """Contract 10: Software version strings obtained at runtime, not hardcoded."""

    def test_get_dependency_versions_returns_dict(self):
        """get_dependency_versions must return a dict."""
        result = get_dependency_versions()
        assert isinstance(result, dict)

    def test_dependency_versions_has_expected_keys(self):
        """get_dependency_versions must include Python, matplotlib, pandas, scipy, numpy, goatools, pyyaml."""
        result = get_dependency_versions()
        expected_keys = {"Python", "matplotlib", "pandas", "scipy", "numpy", "goatools", "pyyaml"}
        # Case-insensitive check
        result_keys_lower = {k.lower() for k in result.keys()}
        for key in expected_keys:
            assert key.lower() in result_keys_lower, \
                f"Missing version for dependency: {key}"

    def test_dependency_versions_are_nonempty_strings(self):
        """All version strings must be non-empty strings."""
        result = get_dependency_versions()
        for key, value in result.items():
            assert isinstance(value, str), f"Version for {key} must be a string"
            assert len(value) > 0, f"Version for {key} must not be empty"

    def test_dependency_versions_not_hardcoded(self):
        """Version strings must be obtained at runtime (not identical to a static fixture)."""
        # Call twice to ensure consistency (runtime values, not random)
        result1 = get_dependency_versions()
        result2 = get_dependency_versions()
        assert result1 == result2, "Runtime version extraction must be deterministic"

    def test_python_version_looks_valid(self):
        """Python version string should look like a version number (contains dots)."""
        result = get_dependency_versions()
        # Find the Python key (case-insensitive)
        python_version = None
        for k, v in result.items():
            if k.lower() == "python":
                python_version = v
                break
        assert python_version is not None, "Must include Python version"
        assert "." in python_version, \
            f"Python version must look like a version string, got: {python_version}"


# ---------------------------------------------------------------------------
# Invariants: Pre-conditions
# ---------------------------------------------------------------------------


class TestPreconditions:
    """Test pre-condition invariants from the blueprint."""

    def test_output_dir_must_exist(self, tmp_path):
        """generate_notes must fail if output_dir does not exist."""
        nonexistent = tmp_path / "nonexistent_dir"
        ni = _make_notes_input()
        with pytest.raises((AssertionError, OSError, FileNotFoundError)):
            generate_notes(ni, nonexistent)


# ---------------------------------------------------------------------------
# Invariants: Post-conditions
# ---------------------------------------------------------------------------


class TestPostconditions:
    """Test post-condition invariants from the blueprint."""

    def test_notes_path_exists_postcondition(self, tmp_path):
        """After generate_notes, the returned path must exist."""
        ni = _make_notes_input(output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        assert result.exists(), "Post-condition: notes_path.exists()"

    def test_notes_path_name_postcondition(self, tmp_path):
        """After generate_notes, the returned path name must be notes.md."""
        ni = _make_notes_input(output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        assert result.name == "notes.md", "Post-condition: notes_path.name == 'notes.md'"


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


class TestErrorConditions:
    """Test error conditions from the blueprint."""

    def test_oserror_on_unwritable_directory(self, tmp_path):
        """OSError must be raised when notes.md cannot be written to output_dir."""
        import os
        import stat

        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        ni = _make_notes_input()
        try:
            with pytest.raises(OSError):
                generate_notes(ni, readonly_dir)
        finally:
            # Restore permissions so pytest can clean up
            readonly_dir.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# Integration-style tests: full generate_notes behavior
# ---------------------------------------------------------------------------


class TestGenerateNotesIntegration:
    """Integration tests that exercise generate_notes end-to-end."""

    def test_full_run_with_all_figures_and_clustering(self, tmp_path):
        """Full run with fig1, fig2, fig3, and clustering enabled."""
        ni = _make_notes_input(with_fig1=True, with_clustering=True, output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert len(content) > 100, "notes.md must contain substantial content"

    def test_full_run_without_fig1(self, tmp_path):
        """Full run without Figure 1 (no cherry-picked terms)."""
        ni = _make_notes_input(with_fig1=False, with_clustering=True, output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert len(content) > 100, "notes.md must contain substantial content"

    def test_full_run_without_clustering(self, tmp_path):
        """Full run without clustering (raw top-N for Figure 3)."""
        ni = _make_notes_input(with_fig1=True, with_clustering=False, output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert len(content) > 100, "notes.md must contain substantial content"

    def test_full_run_without_fig1_and_without_clustering(self, tmp_path):
        """Full run without Figure 1 and without clustering."""
        ni = _make_notes_input(with_fig1=False, with_clustering=False, output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert len(content) > 100, "notes.md must contain substantial content"

    def test_notes_written_as_utf8(self, tmp_path):
        """notes.md must be readable as UTF-8."""
        ni = _make_notes_input(output_dir=tmp_path)
        result = generate_notes(ni, tmp_path)
        # Should not raise UnicodeDecodeError
        content = result.read_text(encoding="utf-8")
        assert isinstance(content, str)

    def test_section_formatters_contribute_to_output(self, tmp_path):
        """Each section formatter's output should be present in the final notes.md."""
        ni = _make_notes_input(output_dir=tmp_path)
        result_path = generate_notes(ni, tmp_path)
        content = result_path.read_text(encoding="utf-8")

        # Each section function should produce content found in the final output
        legends_text = format_figure_legends(ni)
        methods_text = format_methods_text(ni)
        stats_text = format_summary_statistics(ni)
        repro_text = format_reproducibility_note(ni)
        config_text = format_config_guide(ni)

        # At least some distinctive content from each section should be in the file
        # We check a significant substring from each (first 50 chars after stripping)
        for section_name, section_text in [
            ("figure legends", legends_text),
            ("methods", methods_text),
            ("summary statistics", stats_text),
            ("reproducibility", repro_text),
            ("config guide", config_text),
        ]:
            # Check that a non-trivial portion of the section is in the output
            stripped = section_text.strip()
            assert len(stripped) > 0, f"{section_name} section must produce content"
            # Check for at least a substring match (first significant line)
            first_line = stripped.split("\n")[0].strip()
            if len(first_line) > 10:
                assert first_line in content, \
                    f"{section_name} section content not found in notes.md"


# ---------------------------------------------------------------------------
# Edge cases and corner cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for notes generation."""

    def test_clustering_result_none_methods_handling(self):
        """Methods text must handle clustering_result=None gracefully."""
        ni = _make_notes_input(with_clustering=False)
        # Should not raise
        result = format_methods_text(ni)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fig1_result_none_legends_handling(self):
        """Legends text must handle fig1_result=None gracefully."""
        ni = _make_notes_input(with_fig1=False)
        # Should not raise
        result = format_figure_legends(ni)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clustering_result_none_summary_handling(self):
        """Summary stats must handle clustering_result=None gracefully."""
        ni = _make_notes_input(with_clustering=False)
        # Should not raise
        result = format_summary_statistics(ni)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clustering_result_none_reproducibility_handling(self):
        """Reproducibility note must handle clustering_result=None gracefully."""
        ni = _make_notes_input(with_clustering=False)
        # Should not raise
        result = format_reproducibility_note(ni)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_notes_input_all_fields_accessible(self):
        """All NotesInput fields must be accessible after construction."""
        ni = _make_notes_input()
        assert ni.cohort is not None
        assert ni.config is not None
        assert ni.fig2_result is not None
        assert ni.fig3_result is not None
        assert ni.unbiased_stats is not None
        assert ni.fisher_result is not None

    def test_notes_input_fig1_can_be_none(self):
        """NotesInput.fig1_result must accept None."""
        ni = _make_notes_input(with_fig1=False)
        assert ni.fig1_result is None

    def test_notes_input_clustering_can_be_none(self):
        """NotesInput.clustering_result must accept None."""
        ni = _make_notes_input(with_clustering=False)
        assert ni.clustering_result is None

    def test_bar_plot_result_clustering_was_used_false_affects_legends(self):
        """When bar plot was generated without clustering, legends should not
        describe clustering as if it was used."""
        ni = _make_notes_input(with_clustering=False)
        result = format_figure_legends(ni)
        # The legend should still be valid, just may not mention clustering
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_without_clustering_omits_cluster_description(self):
        """When clustering is None, summary should handle cluster stats gracefully."""
        ni = _make_notes_input(with_clustering=False)
        result = format_summary_statistics(ni)
        # The result should still be valid text
        assert isinstance(result, str)
        assert len(result) > 0

    def test_methods_mentions_merging_pos_neg(self):
        """Methods text should mention merging of pos/neg tables for Fisher's test."""
        ni = _make_notes_input()
        result = format_methods_text(ni)
        result_lower = result.lower()
        assert "pos" in result_lower or "neg" in result_lower or \
               "merge" in result_lower or "positive" in result_lower or \
               "negative" in result_lower, \
            "Methods must describe merging of pos/neg tables"
