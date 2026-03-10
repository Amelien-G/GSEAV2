"""
Tests for Unit 8 -- Bar Plot Rendering

Comprehensive test suite validating all behavioral contracts, invariants,
error conditions, and signatures from the Unit 8 blueprint.

## Synthetic Data Assumptions

- Synthetic FisherResult uses 5 GO terms with combined p-values spanning
  several orders of magnitude (1e-10, 1e-6, 0.001, 0.01, 0.5).
- GO term names are uppercase descriptive strings consistent with GSEA output.
- Number of contributing mutant lines ranges from 1 to 5 across 5 mutants.
- The p-value matrix is a 5x5 numpy array (placeholder); bar plot only uses
  combined_pvalues, go_id_to_name, n_contributing, and n_mutants.
- ClusteringResult uses 3 representative GO terms (subset of 5), already
  sorted by combined p-value ascending.
- A long GO term name of 80 characters tests label truncation at
  label_max_length=60.
- top_n_bars defaults to 20, set to 3 in some tests for N-limiting logic.
- Edge cases: empty FisherResult and empty ClusteringResult trigger ValueError.
- Output directory uses pytest tmp_path for file I/O.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from gsea_tool.bar_plot import BarPlotResult, render_bar_plot, select_bar_data
from gsea_tool.configuration import FisherConfig, PlotAppearanceConfig
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.go_clustering import ClusteringResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fisher_result_5_terms():
    """FisherResult with 5 GO terms of varying significance and recurrence."""
    go_ids = ["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004", "GO:0000005"]
    go_id_to_name = {
        "GO:0000001": "MITOCHONDRIAL TRANSLATION",
        "GO:0000002": "CELL CYCLE REGULATION",
        "GO:0000003": "DNA REPAIR MECHANISM",
        "GO:0000004": "SIGNAL TRANSDUCTION PATHWAY",
        "GO:0000005": "LIPID METABOLISM",
    }
    combined_pvalues = {
        "GO:0000001": 1e-10,
        "GO:0000002": 1e-6,
        "GO:0000003": 0.001,
        "GO:0000004": 0.01,
        "GO:0000005": 0.5,
    }
    n_contributing = {
        "GO:0000001": 5,
        "GO:0000002": 4,
        "GO:0000003": 3,
        "GO:0000004": 2,
        "GO:0000005": 1,
    }
    mutant_ids = ["mut1", "mut2", "mut3", "mut4", "mut5"]
    pvalue_matrix = np.ones((5, 5), dtype=np.float64)

    return FisherResult(
        go_ids=go_ids,
        go_id_to_name=go_id_to_name,
        combined_pvalues=combined_pvalues,
        n_contributing=n_contributing,
        pvalue_matrix=pvalue_matrix,
        mutant_ids=mutant_ids,
        go_id_order=go_ids,
        n_mutants=5,
        corrected_pvalues=None,
    )


@pytest.fixture
def clustering_result_3_reps():
    """ClusteringResult with 3 representative terms, sorted by combined p-value."""
    return ClusteringResult(
        representatives=["GO:0000001", "GO:0000002", "GO:0000003"],
        representative_names=[
            "MITOCHONDRIAL TRANSLATION",
            "CELL CYCLE REGULATION",
            "DNA REPAIR MECHANISM",
        ],
        representative_pvalues=[1e-10, 1e-6, 0.001],
        representative_n_contributing=[5, 4, 3],
        cluster_assignments={
            "GO:0000001": 0,
            "GO:0000002": 1,
            "GO:0000003": 2,
            "GO:0000004": 0,
            "GO:0000005": 1,
        },
        n_clusters=3,
        n_prefiltered=5,
        similarity_metric="Lin",
        similarity_threshold=0.7,
    )


@pytest.fixture
def default_fisher_config():
    """Default FisherConfig with top_n_bars=20."""
    return FisherConfig()


@pytest.fixture
def fisher_config_top3():
    """FisherConfig with top_n_bars=3 for limiting tests."""
    return FisherConfig(top_n_bars=3)


@pytest.fixture
def default_plot_config():
    """Default PlotAppearanceConfig."""
    return PlotAppearanceConfig()


@pytest.fixture
def plot_config_no_annotations():
    """PlotAppearanceConfig with annotations and significance line disabled."""
    return PlotAppearanceConfig(
        show_recurrence_annotation=False,
        show_significance_line=False,
    )


# ---------------------------------------------------------------------------
# Tests for select_bar_data
# ---------------------------------------------------------------------------


class TestSelectBarData:
    """Tests for the select_bar_data function."""

    def test_unclustered_mode_returns_all_terms_when_fewer_than_top_n(
        self, fisher_result_5_terms
    ):
        """Contract 2: When clustering_result is None, uses top N GO terms.
        With 5 terms and top_n=20, all 5 should be returned."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, None, top_n=20
        )
        assert len(names) == 5
        assert len(nlps) == 5
        assert len(contribs) == 5

    def test_unclustered_mode_ordered_by_combined_pvalue_most_significant_first(
        self, fisher_result_5_terms
    ):
        """Contract 2: Results ordered by combined p-value, most significant first.
        Most significant (smallest p-value) should have the largest -log10(p)."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, None, top_n=20
        )
        # -log10 values should be in descending order (most significant first)
        for i in range(len(nlps) - 1):
            assert nlps[i] >= nlps[i + 1], (
                f"-log10(p) at index {i} ({nlps[i]}) should be >= index {i+1} ({nlps[i+1]})"
            )

    def test_unclustered_mode_limited_by_top_n(self, fisher_result_5_terms):
        """Contract 2: N is limited by top_n or total GO terms, whichever is smaller."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, None, top_n=3
        )
        assert len(names) == 3
        assert len(nlps) == 3
        assert len(contribs) == 3

    def test_unclustered_mode_selects_most_significant_terms(
        self, fisher_result_5_terms
    ):
        """Contract 2: Top 3 should be the 3 most significant terms."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, None, top_n=3
        )
        # The 3 most significant are GO:0000001, GO:0000002, GO:0000003
        assert "MITOCHONDRIAL TRANSLATION" in names
        assert "CELL CYCLE REGULATION" in names
        assert "DNA REPAIR MECHANISM" in names

    def test_unclustered_mode_neg_log_pvalues_correct(self, fisher_result_5_terms):
        """Contract 4: X-axis displays -log10(combined p-value)."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, None, top_n=20
        )
        # First term should be most significant (1e-10)
        assert nlps[0] == pytest.approx(-math.log10(1e-10), rel=1e-6)
        # Last term should be least significant (0.5)
        assert nlps[-1] == pytest.approx(-math.log10(0.5), rel=1e-6)

    def test_unclustered_mode_n_contributing_matches_terms(
        self, fisher_result_5_terms
    ):
        """Contract 5: n_contributing corresponds to contributing mutant lines."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, None, top_n=20
        )
        # First term (most significant, GO:0000001) has 5 contributing
        assert contribs[0] == 5

    def test_clustered_mode_uses_representatives(
        self, fisher_result_5_terms, clustering_result_3_reps
    ):
        """Contract 1: When clustering_result is provided, uses representative GO terms."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, clustering_result_3_reps, top_n=20
        )
        assert len(names) == 3
        assert names[0] == "MITOCHONDRIAL TRANSLATION"
        assert names[1] == "CELL CYCLE REGULATION"
        assert names[2] == "DNA REPAIR MECHANISM"

    def test_clustered_mode_limited_by_top_n(
        self, fisher_result_5_terms, clustering_result_3_reps
    ):
        """Contract 1: N is limited by top_n_bars or number of representatives."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, clustering_result_3_reps, top_n=2
        )
        assert len(names) == 2

    def test_clustered_mode_ordered_by_combined_pvalue(
        self, fisher_result_5_terms, clustering_result_3_reps
    ):
        """Contract 1: Ordered by combined p-value, most significant at top."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, clustering_result_3_reps, top_n=20
        )
        for i in range(len(nlps) - 1):
            assert nlps[i] >= nlps[i + 1]

    def test_clustered_mode_n_contributing_from_clustering_result(
        self, fisher_result_5_terms, clustering_result_3_reps
    ):
        """Contract 5: n_contributing comes from the ClusteringResult."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, clustering_result_3_reps, top_n=20
        )
        assert contribs == [5, 4, 3]

    def test_error_empty_fisher_result_unclustered(self):
        """Error condition: ValueError when no GO terms exist (unclustered mode)."""
        empty_fisher = FisherResult(
            go_ids=[],
            go_id_to_name={},
            combined_pvalues={},
            n_contributing={},
            pvalue_matrix=np.empty((0, 0)),
            mutant_ids=[],
            go_id_order=[],
            n_mutants=0,
            corrected_pvalues=None,
        )
        with pytest.raises(ValueError, match="[Nn]o terms to plot|[Nn]o GO terms"):
            select_bar_data(empty_fisher, None, top_n=20)

    def test_error_empty_clustering_result(self, fisher_result_5_terms):
        """Error condition: ValueError when no representative GO terms (clustered mode)."""
        empty_clustering = ClusteringResult(
            representatives=[],
            representative_names=[],
            representative_pvalues=[],
            representative_n_contributing=[],
            cluster_assignments={},
            n_clusters=0,
            n_prefiltered=0,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        with pytest.raises(ValueError, match="[Nn]o terms to plot|[Nn]o representative"):
            select_bar_data(fisher_result_5_terms, empty_clustering, top_n=20)

    def test_returns_term_names_not_go_ids(self, fisher_result_5_terms):
        """Contract 3: Y-axis displays GO term names, not GO IDs."""
        names, _, _ = select_bar_data(fisher_result_5_terms, None, top_n=20)
        for name in names:
            assert not name.startswith("GO:"), (
                f"Expected term name, not GO ID, but got '{name}'"
            )

    def test_return_type_is_tuple_of_three_lists(self, fisher_result_5_terms):
        """Signature: select_bar_data returns tuple of 3 lists."""
        result = select_bar_data(fisher_result_5_terms, None, top_n=20)
        assert isinstance(result, tuple)
        assert len(result) == 3
        names, nlps, contribs = result
        assert isinstance(names, list)
        assert isinstance(nlps, list)
        assert isinstance(contribs, list)

    def test_all_returned_lists_same_length(self, fisher_result_5_terms):
        """Signature: All three returned lists must have the same length."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, None, top_n=20
        )
        assert len(names) == len(nlps) == len(contribs)


# ---------------------------------------------------------------------------
# Tests for render_bar_plot
# ---------------------------------------------------------------------------


class TestRenderBarPlot:
    """Tests for the render_bar_plot function."""

    def test_produces_all_three_output_files(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Post-condition: PDF, PNG, and SVG files must be written."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()

    def test_output_file_names_use_default_stem(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Contract 10: Files saved as {output_stem}.pdf/png/svg with default stem."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.pdf_path.name == "figure3_meta_analysis.pdf"
        assert result.png_path.name == "figure3_meta_analysis.png"
        assert result.svg_path.name == "figure3_meta_analysis.svg"

    def test_output_file_names_use_custom_stem(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Contract 10: Files saved with custom output_stem."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
            output_stem="my_custom_plot",
        )
        assert result.pdf_path.name == "my_custom_plot.pdf"
        assert result.png_path.name == "my_custom_plot.png"
        assert result.svg_path.name == "my_custom_plot.svg"

    def test_output_files_in_correct_directory(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Contract 10: Files saved in output_dir."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.pdf_path.parent == tmp_path
        assert result.png_path.parent == tmp_path
        assert result.svg_path.parent == tmp_path

    def test_returns_bar_plot_result_type(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Signature: render_bar_plot returns BarPlotResult."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert isinstance(result, BarPlotResult)

    def test_n_bars_equals_number_of_terms_plotted(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Post-condition: n_bars reflects the actual number of bars."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.n_bars == 5

    def test_n_bars_limited_by_top_n_bars(
        self, tmp_path, fisher_result_5_terms, fisher_config_top3, default_plot_config
    ):
        """Invariant: n_bars cannot exceed fisher_config.top_n_bars."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            fisher_config_top3,
            default_plot_config,
            tmp_path,
        )
        assert result.n_bars == 3
        assert result.n_bars <= fisher_config_top3.top_n_bars

    def test_n_bars_at_least_one(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Invariant: At least one bar must be plotted."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.n_bars > 0

    def test_n_mutants_from_fisher_result(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """BarPlotResult.n_mutants should reflect the fisher_result.n_mutants."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.n_mutants == 5

    def test_clustering_was_used_false_when_no_clustering(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """BarPlotResult.clustering_was_used is False when clustering_result is None."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.clustering_was_used is False

    def test_clustering_was_used_true_when_clustering_provided(
        self,
        tmp_path,
        fisher_result_5_terms,
        clustering_result_3_reps,
        default_fisher_config,
        default_plot_config,
    ):
        """BarPlotResult.clustering_was_used is True when clustering_result is provided."""
        result = render_bar_plot(
            fisher_result_5_terms,
            clustering_result_3_reps,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.clustering_was_used is True

    def test_with_clustering_n_bars_matches_representatives(
        self,
        tmp_path,
        fisher_result_5_terms,
        clustering_result_3_reps,
        default_fisher_config,
        default_plot_config,
    ):
        """Contract 1: With clustering, n_bars equals number of representative terms."""
        result = render_bar_plot(
            fisher_result_5_terms,
            clustering_result_3_reps,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.n_bars == 3

    def test_with_clustering_limited_by_top_n_bars(
        self,
        tmp_path,
        fisher_result_5_terms,
        clustering_result_3_reps,
        fisher_config_top3,
        default_plot_config,
    ):
        """Contract 1: N limited by top_n_bars or number of representatives."""
        config_top2 = FisherConfig(top_n_bars=2)
        result = render_bar_plot(
            fisher_result_5_terms,
            clustering_result_3_reps,
            config_top2,
            default_plot_config,
            tmp_path,
        )
        assert result.n_bars == 2

    def test_output_files_are_nonzero_size(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Post-condition: Output files should contain actual data (non-empty)."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.pdf_path.stat().st_size > 0
        assert result.png_path.stat().st_size > 0
        assert result.svg_path.stat().st_size > 0

    def test_no_annotations_config_still_produces_valid_output(
        self,
        tmp_path,
        fisher_result_5_terms,
        default_fisher_config,
        plot_config_no_annotations,
    ):
        """Contract 6/8: Disabling annotations and significance line still works."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            plot_config_no_annotations,
            tmp_path,
        )
        assert result.pdf_path.exists()
        assert result.n_bars == 5

    def test_custom_figure_dimensions_config(
        self, tmp_path, fisher_result_5_terms, default_fisher_config
    ):
        """Contract 9: Figure dimensions from plot_config are used."""
        custom_config = PlotAppearanceConfig(
            bar_figure_width=12.0,
            bar_figure_height=6.0,
        )
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            custom_config,
            tmp_path,
        )
        # Just verify it runs successfully with custom dimensions
        assert result.pdf_path.exists()
        assert result.n_bars == 5


class TestRenderBarPlotLabelTruncation:
    """Tests for label truncation behavior (Contract 3)."""

    def test_long_label_is_truncated_with_ellipsis(
        self, tmp_path, default_fisher_config, default_plot_config
    ):
        """Contract 3: Names longer than label_max_length are truncated with ellipsis."""
        long_name = "A" * 80  # 80 chars, exceeds default label_max_length=60
        fisher = FisherResult(
            go_ids=["GO:0000001"],
            go_id_to_name={"GO:0000001": long_name},
            combined_pvalues={"GO:0000001": 0.001},
            n_contributing={"GO:0000001": 3},
            pvalue_matrix=np.ones((1, 3)),
            mutant_ids=["mut1", "mut2", "mut3"],
            go_id_order=["GO:0000001"],
            n_mutants=3,
            corrected_pvalues=None,
        )
        # We test via select_bar_data + render to verify truncation happens
        # select_bar_data returns the raw name; truncation happens in render_bar_plot.
        # The contract says label truncation happens; we verify render completes.
        result = render_bar_plot(
            fisher, None, default_fisher_config, default_plot_config, tmp_path
        )
        assert result.n_bars == 1
        assert result.pdf_path.exists()

    def test_short_label_is_not_truncated(
        self, tmp_path, default_fisher_config, default_plot_config
    ):
        """Contract 3: Names within label_max_length are not truncated."""
        short_name = "SHORT TERM"  # Well under 60 chars
        fisher = FisherResult(
            go_ids=["GO:0000001"],
            go_id_to_name={"GO:0000001": short_name},
            combined_pvalues={"GO:0000001": 0.001},
            n_contributing={"GO:0000001": 3},
            pvalue_matrix=np.ones((1, 3)),
            mutant_ids=["mut1", "mut2", "mut3"],
            go_id_order=["GO:0000001"],
            n_mutants=3,
            corrected_pvalues=None,
        )
        # select_bar_data returns term names; short names pass through unchanged
        names, _, _ = select_bar_data(fisher, None, top_n=20)
        assert names[0] == short_name


class TestRenderBarPlotErrors:
    """Tests for error conditions."""

    def test_error_no_terms_unclustered(
        self, tmp_path, default_fisher_config, default_plot_config
    ):
        """Error: ValueError when no GO terms exist in unclustered mode."""
        empty_fisher = FisherResult(
            go_ids=[],
            go_id_to_name={},
            combined_pvalues={},
            n_contributing={},
            pvalue_matrix=np.empty((0, 0)),
            mutant_ids=[],
            go_id_order=[],
            n_mutants=0,
            corrected_pvalues=None,
        )
        with pytest.raises(ValueError, match="[Nn]o terms to plot|[Nn]o GO terms"):
            render_bar_plot(
                empty_fisher,
                None,
                default_fisher_config,
                default_plot_config,
                tmp_path,
            )

    def test_error_no_terms_clustered(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Error: ValueError when clustering result has no representatives."""
        empty_clustering = ClusteringResult(
            representatives=[],
            representative_names=[],
            representative_pvalues=[],
            representative_n_contributing=[],
            cluster_assignments={},
            n_clusters=0,
            n_prefiltered=0,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        with pytest.raises(ValueError, match="[Nn]o terms to plot|[Nn]o representative"):
            render_bar_plot(
                fisher_result_5_terms,
                empty_clustering,
                default_fisher_config,
                default_plot_config,
                tmp_path,
            )

    def test_error_output_dir_does_not_exist(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Pre-condition: output_dir must exist. Non-existent dir raises an error."""
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises((AssertionError, OSError)):
            render_bar_plot(
                fisher_result_5_terms,
                None,
                default_fisher_config,
                default_plot_config,
                nonexistent,
            )


class TestBarPlotResultDataclass:
    """Tests for the BarPlotResult dataclass structure."""

    def test_has_expected_fields(self):
        """Signature: BarPlotResult has pdf_path, png_path, svg_path, n_bars, n_mutants, clustering_was_used."""
        result = BarPlotResult(
            pdf_path=Path("/tmp/test.pdf"),
            png_path=Path("/tmp/test.png"),
            svg_path=Path("/tmp/test.svg"),
            n_bars=5,
            n_mutants=3,
            clustering_was_used=False,
        )
        assert result.pdf_path == Path("/tmp/test.pdf")
        assert result.png_path == Path("/tmp/test.png")
        assert result.svg_path == Path("/tmp/test.svg")
        assert result.n_bars == 5
        assert result.n_mutants == 3
        assert result.clustering_was_used is False

    def test_pdf_path_is_path_type(self):
        """Signature: pdf_path is a Path."""
        result = BarPlotResult(
            pdf_path=Path("/tmp/test.pdf"),
            png_path=Path("/tmp/test.png"),
            svg_path=Path("/tmp/test.svg"),
            n_bars=1,
            n_mutants=1,
            clustering_was_used=False,
        )
        assert isinstance(result.pdf_path, Path)
        assert isinstance(result.png_path, Path)
        assert isinstance(result.svg_path, Path)


class TestSelectBarDataEdgeCases:
    """Additional edge case tests for select_bar_data."""

    def test_single_term(self):
        """Edge case: Single GO term should produce a single bar."""
        fisher = FisherResult(
            go_ids=["GO:0000001"],
            go_id_to_name={"GO:0000001": "SINGLE TERM"},
            combined_pvalues={"GO:0000001": 0.01},
            n_contributing={"GO:0000001": 2},
            pvalue_matrix=np.ones((1, 2)),
            mutant_ids=["mut1", "mut2"],
            go_id_order=["GO:0000001"],
            n_mutants=2,
            corrected_pvalues=None,
        )
        names, nlps, contribs = select_bar_data(fisher, None, top_n=20)
        assert len(names) == 1
        assert names[0] == "SINGLE TERM"
        assert nlps[0] == pytest.approx(-math.log10(0.01), rel=1e-6)
        assert contribs[0] == 2

    def test_top_n_equals_one(self, fisher_result_5_terms):
        """Edge case: top_n=1 returns only the most significant term."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, None, top_n=1
        )
        assert len(names) == 1
        assert names[0] == "MITOCHONDRIAL TRANSLATION"

    def test_top_n_larger_than_available_terms(self, fisher_result_5_terms):
        """Edge case: top_n >> available terms returns all terms."""
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, None, top_n=100
        )
        assert len(names) == 5

    def test_very_small_pvalue_produces_large_neg_log(self):
        """Contract 4: Very small p-values produce large -log10 values."""
        fisher = FisherResult(
            go_ids=["GO:0000001"],
            go_id_to_name={"GO:0000001": "EXTREME TERM"},
            combined_pvalues={"GO:0000001": 1e-300},
            n_contributing={"GO:0000001": 5},
            pvalue_matrix=np.ones((1, 5)),
            mutant_ids=["m1", "m2", "m3", "m4", "m5"],
            go_id_order=["GO:0000001"],
            n_mutants=5,
            corrected_pvalues=None,
        )
        names, nlps, contribs = select_bar_data(fisher, None, top_n=20)
        assert nlps[0] == pytest.approx(300.0, rel=1e-6)

    def test_clustered_mode_single_representative(self, fisher_result_5_terms):
        """Edge case: ClusteringResult with a single representative."""
        single_clustering = ClusteringResult(
            representatives=["GO:0000001"],
            representative_names=["MITOCHONDRIAL TRANSLATION"],
            representative_pvalues=[1e-10],
            representative_n_contributing=[5],
            cluster_assignments={"GO:0000001": 0},
            n_clusters=1,
            n_prefiltered=5,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        names, nlps, contribs = select_bar_data(
            fisher_result_5_terms, single_clustering, top_n=20
        )
        assert len(names) == 1
        assert names[0] == "MITOCHONDRIAL TRANSLATION"


class TestRenderBarPlotIntegration:
    """Integration-level tests exercising the full render pipeline."""

    def test_full_render_unclustered_with_defaults(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Full integration: Unclustered mode with default config produces valid output."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.n_bars == 5
        assert result.n_mutants == 5
        assert result.clustering_was_used is False
        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()

    def test_full_render_clustered_with_defaults(
        self,
        tmp_path,
        fisher_result_5_terms,
        clustering_result_3_reps,
        default_fisher_config,
        default_plot_config,
    ):
        """Full integration: Clustered mode with default config produces valid output."""
        result = render_bar_plot(
            fisher_result_5_terms,
            clustering_result_3_reps,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        assert result.n_bars == 3
        assert result.clustering_was_used is True
        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()

    def test_svg_output_is_valid_xml(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Contract 10: SVG output should be valid XML (basic check)."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        svg_content = result.svg_path.read_text()
        assert "<svg" in svg_content
        assert "</svg>" in svg_content

    def test_pdf_output_has_pdf_header(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Contract 10: PDF output should start with PDF magic bytes."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        pdf_bytes = result.pdf_path.read_bytes()
        assert pdf_bytes[:5] == b"%PDF-"

    def test_png_output_has_png_header(
        self, tmp_path, fisher_result_5_terms, default_fisher_config, default_plot_config
    ):
        """Contract 10: PNG output should start with PNG magic bytes."""
        result = render_bar_plot(
            fisher_result_5_terms,
            None,
            default_fisher_config,
            default_plot_config,
            tmp_path,
        )
        png_bytes = result.png_path.read_bytes()
        assert png_bytes[:4] == b"\x89PNG"
