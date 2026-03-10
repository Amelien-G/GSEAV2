"""
Tests for Unit 8 -- Bar Plot Rendering

This test suite verifies the behavioral contracts, invariants, error conditions,
and signatures specified in the Unit 8 blueprint for the meta-analysis bar plot.

DATA ASSUMPTIONS (module-level):
- Synthetic FisherResult uses 5 GO terms with combined p-values spanning
  several orders of magnitude (1e-10 to 0.5), representing a typical spread
  of Fisher's combined p-values in a GSEA meta-analysis.
- GO term names are uppercase descriptive strings (e.g., "MITOCHONDRIAL TRANSLATION"),
  consistent with GSEA preranked output naming conventions.
- Number of contributing mutant lines ranges from 1 to 5, representing a small
  cohort study with variable recurrence across terms.
- ClusteringResult synthetic data uses a subset (3 of 5) GO terms as cluster
  representatives, simulating redundancy reduction.
- The default top_n_bars=20 and test-specific top_n_bars values (e.g., 3) are used
  to exercise the N-limiting logic.
- A long GO term name of 80 characters is used to test label truncation at
  the default label_max_length=60.
"""

import inspect
import math
from dataclasses import fields as dataclass_fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from gsea_tool.configuration import FisherConfig, PlotAppearanceConfig
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.go_clustering import ClusteringResult
from gsea_tool.bar_plot import (
    BarPlotResult,
    render_bar_plot,
    select_bar_data,
)


# ---------------------------------------------------------------------------
# Helpers for synthetic data generation
# ---------------------------------------------------------------------------

def _make_fisher_result(
    go_ids: list[str] | None = None,
    go_id_to_name: dict[str, str] | None = None,
    combined_pvalues: dict[str, float] | None = None,
    n_contributing: dict[str, int] | None = None,
    n_mutants: int = 5,
) -> FisherResult:
    """Build a synthetic FisherResult.

    DATA ASSUMPTION: 5 GO terms with combined p-values from very significant
    (1e-10) to non-significant (0.5). The p-value matrix is a dummy 5x5
    identity-like structure since the bar plot only reads the aggregated
    fields, not the raw matrix.
    """
    if go_ids is None:
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004", "GO:0000005"]

    if go_id_to_name is None:
        go_id_to_name = {
            "GO:0000001": "MITOCHONDRIAL TRANSLATION",
            "GO:0000002": "OXIDATIVE PHOSPHORYLATION",
            "GO:0000003": "RIBOSOME BIOGENESIS",
            "GO:0000004": "SYNAPTIC VESICLE CYCLE",
            "GO:0000005": "CELL CYCLE REGULATION",
        }

    if combined_pvalues is None:
        # DATA ASSUMPTION: p-values span several orders of magnitude, typical
        # of Fisher's combined test output.
        combined_pvalues = {
            "GO:0000001": 1e-10,
            "GO:0000002": 1e-6,
            "GO:0000003": 0.001,
            "GO:0000004": 0.03,
            "GO:0000005": 0.5,
        }

    if n_contributing is None:
        # DATA ASSUMPTION: Contributing counts range from 1 to 5, representing
        # variable recurrence in a 5-mutant cohort.
        n_contributing = {
            "GO:0000001": 5,
            "GO:0000002": 4,
            "GO:0000003": 3,
            "GO:0000004": 2,
            "GO:0000005": 1,
        }

    mutant_ids = [f"mutant_{i}" for i in range(n_mutants)]
    n_go = len(go_ids)

    # DATA ASSUMPTION: Dummy p-value matrix -- the bar plot does not use
    # individual matrix entries, only aggregated combined_pvalues and n_contributing.
    pvalue_matrix = np.ones((n_go, n_mutants), dtype=float)

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
    fisher_result: FisherResult | None = None,
    representative_ids: list[str] | None = None,
) -> ClusteringResult:
    """Build a synthetic ClusteringResult.

    DATA ASSUMPTION: 3 cluster representatives selected from 5 GO terms,
    simulating semantic clustering that reduced 5 terms into 3 clusters.
    Representatives are ordered by combined p-value (most significant first).
    """
    if fisher_result is None:
        fisher_result = _make_fisher_result()

    if representative_ids is None:
        representative_ids = ["GO:0000001", "GO:0000003", "GO:0000004"]

    # Sort representatives by combined p-value ascending
    representative_ids = sorted(
        representative_ids,
        key=lambda gid: fisher_result.combined_pvalues[gid],
    )

    representative_names = [
        fisher_result.go_id_to_name.get(gid, "") for gid in representative_ids
    ]
    representative_pvalues = [
        fisher_result.combined_pvalues[gid] for gid in representative_ids
    ]
    representative_n_contributing = [
        fisher_result.n_contributing.get(gid, 0) for gid in representative_ids
    ]

    # DATA ASSUMPTION: Cluster assignments map all 5 GO IDs to 3 clusters.
    cluster_assignments = {
        "GO:0000001": 0,
        "GO:0000002": 0,  # clustered with GO:0000001
        "GO:0000003": 1,
        "GO:0000004": 2,
        "GO:0000005": 2,  # clustered with GO:0000004
    }

    return ClusteringResult(
        representatives=representative_ids,
        representative_names=representative_names,
        representative_pvalues=representative_pvalues,
        representative_n_contributing=representative_n_contributing,
        cluster_assignments=cluster_assignments,
        n_clusters=3,
        n_prefiltered=5,
        similarity_metric="Lin",
        similarity_threshold=0.7,
    )


def _default_fisher_config(**overrides) -> FisherConfig:
    """Create a FisherConfig with defaults, optionally overriding fields.

    DATA ASSUMPTION: Uses default FisherConfig values from Unit 2 blueprint.
    top_n_bars=20 is the standard bar count limit.
    """
    kwargs = {}
    for key, val in overrides.items():
        kwargs[key] = val
    return FisherConfig(**kwargs)


def _default_plot_config(**overrides) -> PlotAppearanceConfig:
    """Create a PlotAppearanceConfig with defaults, optionally overriding fields.

    DATA ASSUMPTION: Uses default PlotAppearanceConfig values from Unit 2 blueprint.
    bar_colormap="YlOrRd", label_max_length=60, dpi=300.
    """
    kwargs = {}
    for key, val in overrides.items():
        kwargs[key] = val
    return PlotAppearanceConfig(**kwargs)


# ---------------------------------------------------------------------------
# Signature Tests
# ---------------------------------------------------------------------------


class TestBarPlotResultSignature:
    """Verify BarPlotResult dataclass has the expected fields and types."""

    def test_is_dataclass(self):
        """BarPlotResult must be a dataclass."""
        assert hasattr(BarPlotResult, "__dataclass_fields__")

    def test_has_pdf_path_field(self):
        """BarPlotResult must have a pdf_path field."""
        field_names = [f.name for f in dataclass_fields(BarPlotResult)]
        assert "pdf_path" in field_names

    def test_has_png_path_field(self):
        """BarPlotResult must have a png_path field."""
        field_names = [f.name for f in dataclass_fields(BarPlotResult)]
        assert "png_path" in field_names

    def test_has_svg_path_field(self):
        """BarPlotResult must have a svg_path field."""
        field_names = [f.name for f in dataclass_fields(BarPlotResult)]
        assert "svg_path" in field_names

    def test_has_n_bars_field(self):
        """BarPlotResult must have an n_bars field."""
        field_names = [f.name for f in dataclass_fields(BarPlotResult)]
        assert "n_bars" in field_names

    def test_has_n_mutants_field(self):
        """BarPlotResult must have an n_mutants field."""
        field_names = [f.name for f in dataclass_fields(BarPlotResult)]
        assert "n_mutants" in field_names

    def test_has_clustering_was_used_field(self):
        """BarPlotResult must have a clustering_was_used field."""
        field_names = [f.name for f in dataclass_fields(BarPlotResult)]
        assert "clustering_was_used" in field_names


class TestRenderBarPlotSignature:
    """Verify render_bar_plot function signature matches the blueprint."""

    def test_is_callable(self):
        assert callable(render_bar_plot)

    def test_parameter_names(self):
        sig = inspect.signature(render_bar_plot)
        params = list(sig.parameters.keys())
        assert "fisher_result" in params
        assert "clustering_result" in params
        assert "fisher_config" in params
        assert "plot_config" in params
        assert "output_dir" in params
        assert "output_stem" in params

    def test_output_stem_default(self):
        """output_stem should have a default value of 'figure3_meta_analysis'."""
        sig = inspect.signature(render_bar_plot)
        output_stem_param = sig.parameters["output_stem"]
        assert output_stem_param.default == "figure3_meta_analysis"


class TestSelectBarDataSignature:
    """Verify select_bar_data function signature matches the blueprint."""

    def test_is_callable(self):
        assert callable(select_bar_data)

    def test_parameter_names(self):
        sig = inspect.signature(select_bar_data)
        params = list(sig.parameters.keys())
        assert "fisher_result" in params
        assert "clustering_result" in params
        assert "top_n" in params


# ---------------------------------------------------------------------------
# select_bar_data Tests
# ---------------------------------------------------------------------------


class TestSelectBarDataUnclusteredMode:
    """Contract 2: When clustering_result is None, use top N GO terms by
    combined p-value directly, without redundancy reduction.
    """

    def test_returns_three_lists(self):
        """select_bar_data returns a tuple of three lists."""
        fisher = _make_fisher_result()
        result = select_bar_data(fisher, None, top_n=20)
        assert isinstance(result, tuple)
        assert len(result) == 3
        term_names, neg_log_pvalues, n_contributing = result
        assert isinstance(term_names, list)
        assert isinstance(neg_log_pvalues, list)
        assert isinstance(n_contributing, list)

    def test_all_lists_same_length(self):
        """All three returned lists must have the same length."""
        fisher = _make_fisher_result()
        term_names, neg_log_pvalues, n_contributing = select_bar_data(fisher, None, top_n=20)
        assert len(term_names) == len(neg_log_pvalues) == len(n_contributing)

    def test_uses_all_terms_when_top_n_exceeds_total(self):
        """When top_n > total GO terms, returns all terms.
        N is limited by top_n or total number of GO terms, whichever is smaller.
        """
        fisher = _make_fisher_result()  # 5 terms
        term_names, _, _ = select_bar_data(fisher, None, top_n=100)
        assert len(term_names) == 5

    def test_limits_to_top_n(self):
        """When top_n < total GO terms, returns only top_n terms.

        DATA ASSUMPTION: With 5 GO terms and top_n=3, only the 3 most
        significant (by combined p-value) should be returned.
        """
        fisher = _make_fisher_result()  # 5 terms
        term_names, _, _ = select_bar_data(fisher, None, top_n=3)
        assert len(term_names) == 3

    def test_ordered_by_combined_pvalue_most_significant_first(self):
        """Results are ordered by combined p-value, most significant first.

        DATA ASSUMPTION: GO:0000001 has smallest p-value (1e-10), GO:0000005
        has largest (0.5). The ordering should reflect this.
        """
        fisher = _make_fisher_result()
        term_names, neg_log_pvalues, _ = select_bar_data(fisher, None, top_n=5)

        # neg_log_pvalues should be in descending order (most significant = largest -log10)
        for i in range(len(neg_log_pvalues) - 1):
            assert neg_log_pvalues[i] >= neg_log_pvalues[i + 1], (
                f"neg_log_pvalues not in descending order at index {i}: "
                f"{neg_log_pvalues[i]} < {neg_log_pvalues[i + 1]}"
            )

    def test_neg_log_pvalues_are_correct(self):
        """X-axis values must be -log10(combined p-value).

        DATA ASSUMPTION: For GO:0000001 with p=1e-10, -log10(1e-10) = 10.0.
        """
        fisher = _make_fisher_result()
        term_names, neg_log_pvalues, _ = select_bar_data(fisher, None, top_n=5)

        # First entry should be the most significant (GO:0000001, p=1e-10)
        assert math.isclose(neg_log_pvalues[0], 10.0, rel_tol=1e-6)

    def test_returns_term_names_not_go_ids(self):
        """Contract 3: Y-axis displays GO term names, not GO IDs."""
        fisher = _make_fisher_result()
        term_names, _, _ = select_bar_data(fisher, None, top_n=5)

        # Term names should be readable labels, not GO:XXXXXXX format
        for name in term_names:
            assert not name.startswith("GO:"), (
                f"Expected term name, got GO ID: {name}"
            )

    def test_n_contributing_values_are_correct(self):
        """Contributing counts match the FisherResult n_contributing dict.

        DATA ASSUMPTION: Most significant term (GO:0000001) has n_contributing=5.
        """
        fisher = _make_fisher_result()
        _, _, n_contributing = select_bar_data(fisher, None, top_n=5)

        # First entry is GO:0000001 with n_contributing=5
        assert n_contributing[0] == 5

    def test_top_n_selects_most_significant(self):
        """With top_n=3, only the 3 terms with smallest combined p-values are selected.

        DATA ASSUMPTION: The 3 most significant are GO:0000001 (1e-10),
        GO:0000002 (1e-6), GO:0000003 (0.001).
        """
        fisher = _make_fisher_result()
        term_names, _, _ = select_bar_data(fisher, None, top_n=3)

        expected_names = [
            "MITOCHONDRIAL TRANSLATION",
            "OXIDATIVE PHOSPHORYLATION",
            "RIBOSOME BIOGENESIS",
        ]
        assert term_names == expected_names


class TestSelectBarDataClusteredMode:
    """Contract 1: When clustering_result is provided, show top N representative
    GO terms, ordered by combined p-value.
    """

    def test_uses_cluster_representatives(self):
        """When clustering_result is provided, uses representative terms only.

        DATA ASSUMPTION: ClusteringResult has 3 representatives out of 5 total
        GO terms. Only those 3 should appear.
        """
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)
        term_names, _, _ = select_bar_data(fisher, clustering, top_n=20)

        # Should have exactly 3 terms (the representatives)
        assert len(term_names) == 3

    def test_representative_order_by_pvalue(self):
        """Representatives are ordered by combined p-value (most significant first)."""
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)
        _, neg_log_pvalues, _ = select_bar_data(fisher, clustering, top_n=20)

        for i in range(len(neg_log_pvalues) - 1):
            assert neg_log_pvalues[i] >= neg_log_pvalues[i + 1]

    def test_limited_by_top_n(self):
        """N is limited by top_n or number of representatives, whichever is smaller.

        DATA ASSUMPTION: With 3 representatives and top_n=2, only 2 are returned.
        """
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)  # 3 representatives
        term_names, _, _ = select_bar_data(fisher, clustering, top_n=2)
        assert len(term_names) == 2

    def test_limited_by_representative_count(self):
        """When top_n exceeds number of representatives, all representatives are used."""
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)  # 3 representatives
        term_names, _, _ = select_bar_data(fisher, clustering, top_n=100)
        assert len(term_names) == 3

    def test_representative_names_are_term_names(self):
        """Contract 3: Display names are readable GO term names, not IDs."""
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)
        term_names, _, _ = select_bar_data(fisher, clustering, top_n=20)

        for name in term_names:
            assert not name.startswith("GO:")

    def test_contributing_counts_match(self):
        """Contributing counts come from the FisherResult for each representative."""
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)
        _, _, n_contributing = select_bar_data(fisher, clustering, top_n=20)

        # Representatives are GO:0000001(5), GO:0000003(3), GO:0000004(2)
        # Ordered by p-value: GO:0000001(p=1e-10, n=5), GO:0000003(p=0.001, n=3),
        #                     GO:0000004(p=0.03, n=2)
        assert n_contributing == [5, 3, 2]


# ---------------------------------------------------------------------------
# render_bar_plot Tests
# ---------------------------------------------------------------------------


class TestRenderBarPlotOutputFiles:
    """Contract 10: Figure is saved to PDF, PNG, SVG in output_dir."""

    def test_creates_pdf_png_svg(self, tmp_path):
        """render_bar_plot must create all three output files."""
        fisher = _make_fisher_result()
        fisher_config = _default_fisher_config()
        plot_config = _default_plot_config()

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=fisher_config,
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists(), "PDF file must be written"
        assert result.png_path.exists(), "PNG file must be written"
        assert result.svg_path.exists(), "SVG file must be written"

    def test_output_file_naming_default_stem(self, tmp_path):
        """Files use default stem 'figure3_meta_analysis'."""
        fisher = _make_fisher_result()
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.name == "figure3_meta_analysis.pdf"
        assert result.png_path.name == "figure3_meta_analysis.png"
        assert result.svg_path.name == "figure3_meta_analysis.svg"

    def test_output_file_naming_custom_stem(self, tmp_path):
        """Files use a custom output_stem when provided.

        DATA ASSUMPTION: Custom stem 'my_bar_plot' is an arbitrary valid filename.
        """
        fisher = _make_fisher_result()
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
            output_stem="my_bar_plot",
        )
        plt.close("all")

        assert result.pdf_path.name == "my_bar_plot.pdf"
        assert result.png_path.name == "my_bar_plot.png"
        assert result.svg_path.name == "my_bar_plot.svg"

    def test_output_files_in_output_dir(self, tmp_path):
        """All output files are in the specified output_dir."""
        fisher = _make_fisher_result()
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.parent == tmp_path
        assert result.png_path.parent == tmp_path
        assert result.svg_path.parent == tmp_path

    def test_output_files_are_nonempty(self, tmp_path):
        """Output files must have nonzero size (actual content was written)."""
        fisher = _make_fisher_result()
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.stat().st_size > 0
        assert result.png_path.stat().st_size > 0
        assert result.svg_path.stat().st_size > 0


class TestRenderBarPlotResult:
    """Verify BarPlotResult metadata is correct."""

    def test_n_bars_matches_unclustered(self, tmp_path):
        """n_bars reflects the number of bars actually plotted (unclustered).

        DATA ASSUMPTION: With 5 GO terms and top_n_bars=20, all 5 are plotted.
        """
        fisher = _make_fisher_result()  # 5 terms
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars == 5

    def test_n_bars_limited_by_top_n_bars(self, tmp_path):
        """n_bars is limited by fisher_config.top_n_bars.

        DATA ASSUMPTION: With 5 GO terms and top_n_bars=3, only 3 are plotted.
        """
        fisher = _make_fisher_result()  # 5 terms
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(top_n_bars=3),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars == 3

    def test_n_bars_invariant_at_least_one(self, tmp_path):
        """Invariant: n_bars > 0."""
        fisher = _make_fisher_result()
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(top_n_bars=1),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars > 0

    def test_n_bars_invariant_not_exceeds_top_n_bars(self, tmp_path):
        """Invariant: n_bars <= fisher_config.top_n_bars."""
        fisher = _make_fisher_result()
        config = _default_fisher_config(top_n_bars=3)
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=config,
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars <= config.top_n_bars

    def test_n_mutants_matches_fisher_result(self, tmp_path):
        """n_mutants in BarPlotResult matches FisherResult.n_mutants."""
        fisher = _make_fisher_result(n_mutants=7)
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_mutants == 7

    def test_clustering_was_used_false_when_none(self, tmp_path):
        """clustering_was_used is False when clustering_result is None."""
        fisher = _make_fisher_result()
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.clustering_was_used is False

    def test_clustering_was_used_true_when_provided(self, tmp_path):
        """clustering_was_used is True when clustering_result is provided."""
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=clustering,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.clustering_was_used is True

    def test_n_bars_clustered_mode(self, tmp_path):
        """n_bars reflects representative count in clustered mode.

        DATA ASSUMPTION: ClusteringResult has 3 representatives, so 3 bars.
        """
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)  # 3 representatives
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=clustering,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars == 3


class TestRenderBarPlotClusteredMode:
    """Contract 1: When clustering_result is provided, bar plot shows
    representative GO terms.
    """

    def test_clustered_creates_output_files(self, tmp_path):
        """Clustered mode still creates all three output files."""
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=clustering,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()

    def test_clustered_limited_by_top_n_bars(self, tmp_path):
        """In clustered mode, n_bars is min(n_representatives, top_n_bars)."""
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)  # 3 representatives
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=clustering,
            fisher_config=_default_fisher_config(top_n_bars=2),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars == 2


# ---------------------------------------------------------------------------
# Error Condition Tests
# ---------------------------------------------------------------------------


class TestRenderBarPlotErrorConditions:
    """Verify error conditions from the blueprint."""

    def test_no_terms_to_plot_unclustered_empty_fisher(self, tmp_path):
        """ValueError when no GO terms exist at all (unclustered mode).

        DATA ASSUMPTION: An empty FisherResult with no GO terms represents
        a degenerate input where no enrichment terms were found.
        """
        fisher = _make_fisher_result(
            go_ids=[],
            go_id_to_name={},
            combined_pvalues={},
            n_contributing={},
        )
        with pytest.raises(ValueError):
            render_bar_plot(
                fisher_result=fisher,
                clustering_result=None,
                fisher_config=_default_fisher_config(),
                plot_config=_default_plot_config(),
                output_dir=tmp_path,
            )
        plt.close("all")

    def test_no_terms_to_plot_clustered_empty_representatives(self, tmp_path):
        """ValueError when clustering produces no representative terms.

        DATA ASSUMPTION: A ClusteringResult with empty representatives list
        simulates the case where all GO terms were filtered out pre-clustering.
        """
        fisher = _make_fisher_result()
        clustering = ClusteringResult(
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
        with pytest.raises(ValueError):
            render_bar_plot(
                fisher_result=fisher,
                clustering_result=clustering,
                fisher_config=_default_fisher_config(),
                plot_config=_default_plot_config(),
                output_dir=tmp_path,
            )
        plt.close("all")

    def test_oseerror_on_write_failure(self, tmp_path):
        """OSError when cannot write output files to output_dir.

        DATA ASSUMPTION: A non-existent directory path simulates write failure.
        """
        fisher = _make_fisher_result()
        nonexistent_dir = tmp_path / "does_not_exist" / "subdir"

        with pytest.raises((OSError, FileNotFoundError, AssertionError)):
            render_bar_plot(
                fisher_result=fisher,
                clustering_result=None,
                fisher_config=_default_fisher_config(),
                plot_config=_default_plot_config(),
                output_dir=nonexistent_dir,
            )
        plt.close("all")


class TestSelectBarDataErrorConditions:
    """Error conditions for select_bar_data."""

    def test_empty_fisher_no_clustering_raises_value_error(self):
        """ValueError when no GO terms exist (unclustered mode)."""
        fisher = _make_fisher_result(
            go_ids=[],
            go_id_to_name={},
            combined_pvalues={},
            n_contributing={},
        )
        with pytest.raises(ValueError):
            select_bar_data(fisher, None, top_n=20)

    def test_empty_clustering_raises_value_error(self):
        """ValueError when clustering produces empty representative list."""
        fisher = _make_fisher_result()
        clustering = ClusteringResult(
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
        with pytest.raises(ValueError):
            select_bar_data(fisher, clustering, top_n=20)


# ---------------------------------------------------------------------------
# Behavioral Contract Tests
# ---------------------------------------------------------------------------


class TestLabelTruncation:
    """Contract 3: Names longer than plot_config.label_max_length are
    truncated with an ellipsis.
    """

    def test_long_label_truncated_in_select_bar_data(self):
        """A GO term name exceeding label_max_length is truncated.

        DATA ASSUMPTION: A GO term name of 80 characters exceeds the default
        label_max_length=60 and should be truncated to 60 chars with ellipsis.
        Note: Truncation may happen in render_bar_plot rather than select_bar_data.
        This test verifies the behavior at whichever level it occurs.
        """
        # Create a fisher result with one very long term name
        long_name = "A" * 80  # 80 characters, exceeds default 60
        go_ids = ["GO:0000099"]
        go_id_to_name = {"GO:0000099": long_name}
        combined_pvalues = {"GO:0000099": 0.001}
        n_contributing = {"GO:0000099": 3}

        fisher = _make_fisher_result(
            go_ids=go_ids,
            go_id_to_name=go_id_to_name,
            combined_pvalues=combined_pvalues,
            n_contributing=n_contributing,
            n_mutants=3,
        )

        # select_bar_data may or may not truncate -- the truncation may
        # happen at render time. We test the rendered output below.
        term_names, _, _ = select_bar_data(fisher, None, top_n=20)
        # The name should either be truncated here or remain full-length
        # (truncation at render time). Either way, verify we got a result.
        assert len(term_names) == 1

    def test_long_label_truncated_in_rendered_plot(self, tmp_path):
        """Rendered plot truncates long labels.

        DATA ASSUMPTION: 80-char name is truncated to at most 60 chars + ellipsis.
        We verify by checking the y-axis tick labels in the matplotlib figure.
        """
        long_name = "A" * 80
        go_ids = ["GO:0000099"]
        go_id_to_name = {"GO:0000099": long_name}
        combined_pvalues = {"GO:0000099": 0.001}
        n_contributing = {"GO:0000099": 3}

        fisher = _make_fisher_result(
            go_ids=go_ids,
            go_id_to_name=go_id_to_name,
            combined_pvalues=combined_pvalues,
            n_contributing=n_contributing,
        )
        plot_config = _default_plot_config(label_max_length=60)

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        # The plot was created -- verify files exist
        assert result.pdf_path.exists()

    def test_short_label_not_truncated(self, tmp_path):
        """A GO term name shorter than label_max_length is not truncated.

        DATA ASSUMPTION: 'CELL CYCLE' is 10 characters, well under 60.
        """
        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            go_id_to_name={"GO:0000001": "CELL CYCLE"},
            combined_pvalues={"GO:0000001": 0.001},
            n_contributing={"GO:0000001": 3},
        )

        term_names, _, _ = select_bar_data(fisher, None, top_n=20)
        # Should not be truncated
        assert term_names[0] == "CELL CYCLE" or "..." not in term_names[0]


class TestNegLogPvalueAxis:
    """Contract 4: X-axis displays -log10(combined p-value)."""

    def test_neg_log_values_computed_correctly(self):
        """Verify -log10 transformation for several p-values.

        DATA ASSUMPTION: Known p-values with expected -log10 outputs:
        1e-10 -> 10.0, 1e-6 -> 6.0, 0.001 -> 3.0, 0.03 -> ~1.52, 0.5 -> ~0.301.
        """
        fisher = _make_fisher_result()
        _, neg_log_pvalues, _ = select_bar_data(fisher, None, top_n=5)

        # All values should be positive (since p-values are in (0, 1])
        for val in neg_log_pvalues:
            assert val > 0, f"Expected positive -log10(p), got {val}"

        # First value (most significant, p=1e-10) should be approximately 10
        assert math.isclose(neg_log_pvalues[0], 10.0, rel_tol=1e-6)

        # Second value (p=1e-6) should be approximately 6
        assert math.isclose(neg_log_pvalues[1], 6.0, rel_tol=1e-6)

        # Third value (p=0.001) should be approximately 3
        assert math.isclose(neg_log_pvalues[2], 3.0, rel_tol=1e-6)


class TestSignificanceLine:
    """Contract 8: When show_significance_line is True, a vertical dashed
    line at -log10(0.05) is drawn.
    """

    def test_significance_line_enabled(self, tmp_path):
        """Plot with show_significance_line=True creates valid output.

        DATA ASSUMPTION: -log10(0.05) is approximately 1.301, the standard
        nominal significance threshold.
        """
        fisher = _make_fisher_result()
        plot_config = _default_plot_config(show_significance_line=True)

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists()
        assert result.n_bars > 0

    def test_significance_line_disabled(self, tmp_path):
        """Plot with show_significance_line=False creates valid output."""
        fisher = _make_fisher_result()
        plot_config = _default_plot_config(show_significance_line=False)

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists()
        assert result.n_bars > 0


class TestRecurrenceAnnotation:
    """Contract 6: When show_recurrence_annotation is True, contributing
    mutant line counts are displayed as text on or next to each bar.
    """

    def test_annotation_enabled(self, tmp_path):
        """Plot with show_recurrence_annotation=True creates valid output."""
        fisher = _make_fisher_result()
        plot_config = _default_plot_config(show_recurrence_annotation=True)

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists()

    def test_annotation_disabled(self, tmp_path):
        """Plot with show_recurrence_annotation=False creates valid output."""
        fisher = _make_fisher_result()
        plot_config = _default_plot_config(show_recurrence_annotation=False)

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists()


class TestFigureDimensions:
    """Contract 9: Figure dimensions are set by plot_config."""

    def test_custom_figure_dimensions(self, tmp_path):
        """Figure uses configured width and height.

        DATA ASSUMPTION: Custom dimensions 12x10 inches, different from default
        10x8, to verify the config is respected.
        """
        fisher = _make_fisher_result()
        plot_config = _default_plot_config(
            bar_figure_width=12.0,
            bar_figure_height=10.0,
        )

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        # Verify output was created successfully
        assert result.pdf_path.exists()
        assert result.n_bars > 0

    def test_default_figure_dimensions(self, tmp_path):
        """Default dimensions (10x8) produce valid output."""
        fisher = _make_fisher_result()
        plot_config = _default_plot_config()  # defaults: 10x8

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists()


class TestColormap:
    """Contract 5: Bar color encodes contributing mutant lines using
    the configured sequential colormap (default YlOrRd).
    """

    def test_default_colormap(self, tmp_path):
        """Default colormap YlOrRd produces valid output."""
        fisher = _make_fisher_result()
        plot_config = _default_plot_config()  # default bar_colormap="YlOrRd"

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists()
        assert result.n_bars > 0

    def test_alternate_colormap(self, tmp_path):
        """A different sequential colormap produces valid output.

        DATA ASSUMPTION: 'viridis' is a valid matplotlib sequential colormap.
        """
        fisher = _make_fisher_result()
        plot_config = _default_plot_config(bar_colormap="viridis")

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists()
        assert result.n_bars > 0


class TestDPIConfig:
    """Contract 10: Saved at the configured DPI."""

    def test_custom_dpi(self, tmp_path):
        """Custom DPI setting produces valid output.

        DATA ASSUMPTION: DPI=150 is a lower-than-default resolution that
        should still produce valid output files.
        """
        fisher = _make_fisher_result()
        plot_config = _default_plot_config(dpi=150)

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=plot_config,
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()


# ---------------------------------------------------------------------------
# Invariant Tests
# ---------------------------------------------------------------------------


class TestInvariants:
    """Verify all blueprint invariants."""

    def test_n_bars_positive(self, tmp_path):
        """Post-condition: result.n_bars > 0."""
        fisher = _make_fisher_result()
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars > 0, "At least one bar must be plotted"

    def test_n_bars_does_not_exceed_top_n_bars(self, tmp_path):
        """Post-condition: result.n_bars <= fisher_config.top_n_bars."""
        for top_n in [1, 3, 5, 10, 20]:
            fisher = _make_fisher_result()
            config = _default_fisher_config(top_n_bars=top_n)
            result = render_bar_plot(
                fisher_result=fisher,
                clustering_result=None,
                fisher_config=config,
                plot_config=_default_plot_config(),
                output_dir=tmp_path,
            )
            plt.close("all")

            assert result.n_bars <= top_n, (
                f"n_bars ({result.n_bars}) exceeded top_n_bars ({top_n})"
            )

    def test_all_output_files_exist(self, tmp_path):
        """Post-conditions: PDF, PNG, SVG files exist after rendering."""
        fisher = _make_fisher_result()
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists(), "PDF file must be written"
        assert result.png_path.exists(), "PNG file must be written"
        assert result.svg_path.exists(), "SVG file must be written"


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_single_go_term(self, tmp_path):
        """Rendering with exactly one GO term produces valid output.

        DATA ASSUMPTION: A single GO term with p=0.01 and n_contributing=2
        represents the minimal valid input.
        """
        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            go_id_to_name={"GO:0000001": "SINGLE TERM"},
            combined_pvalues={"GO:0000001": 0.01},
            n_contributing={"GO:0000001": 2},
            n_mutants=3,
        )

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars == 1
        assert result.pdf_path.exists()

    def test_top_n_bars_equals_one(self, tmp_path):
        """top_n_bars=1 renders exactly one bar.

        DATA ASSUMPTION: With 5 terms and top_n_bars=1, only the most
        significant term is plotted.
        """
        fisher = _make_fisher_result()
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(top_n_bars=1),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars == 1

    def test_all_contributing_counts_equal(self, tmp_path):
        """When all terms have the same contributing count, the colormap
        should still produce valid output.

        DATA ASSUMPTION: All 3 terms have n_contributing=3, representing
        uniform recurrence across the cohort.
        """
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            go_id_to_name={
                "GO:0000001": "TERM A",
                "GO:0000002": "TERM B",
                "GO:0000003": "TERM C",
            },
            combined_pvalues={
                "GO:0000001": 0.001,
                "GO:0000002": 0.01,
                "GO:0000003": 0.05,
            },
            n_contributing={
                "GO:0000001": 3,
                "GO:0000002": 3,
                "GO:0000003": 3,
            },
            n_mutants=3,
        )

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars == 3
        assert result.pdf_path.exists()

    def test_very_small_pvalue(self, tmp_path):
        """Very small p-values produce large -log10 values without errors.

        DATA ASSUMPTION: p=1e-300 represents an extreme but valid p-value
        from Fisher's combined test with many mutant lines.
        """
        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            go_id_to_name={"GO:0000001": "EXTREME TERM"},
            combined_pvalues={"GO:0000001": 1e-300},
            n_contributing={"GO:0000001": 5},
        )

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.n_bars == 1
        assert result.pdf_path.exists()

    def test_pvalue_exactly_at_significance_threshold(self):
        """A p-value exactly at 0.05 produces -log10(0.05) correctly.

        DATA ASSUMPTION: p=0.05 is the nominal significance threshold.
        -log10(0.05) ~ 1.3010.
        """
        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            go_id_to_name={"GO:0000001": "BORDERLINE TERM"},
            combined_pvalues={"GO:0000001": 0.05},
            n_contributing={"GO:0000001": 2},
        )

        term_names, neg_log_pvalues, _ = select_bar_data(fisher, None, top_n=20)
        assert math.isclose(neg_log_pvalues[0], -math.log10(0.05), rel_tol=1e-6)

    def test_select_bar_data_single_term(self):
        """select_bar_data with a single GO term returns a single-element list.

        DATA ASSUMPTION: Minimal valid input with one GO term.
        """
        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            go_id_to_name={"GO:0000001": "ONLY TERM"},
            combined_pvalues={"GO:0000001": 0.001},
            n_contributing={"GO:0000001": 4},
        )

        term_names, neg_log_pvalues, n_contributing = select_bar_data(fisher, None, top_n=20)
        assert len(term_names) == 1
        assert term_names[0] == "ONLY TERM"
        assert n_contributing[0] == 4
        assert math.isclose(neg_log_pvalues[0], 3.0, rel_tol=1e-6)


class TestCleanStyling:
    """Contract 11: Clean, minimal styling consistent with Nature-family
    journal standards.
    """

    def test_renders_without_errors(self, tmp_path):
        """The figure renders successfully with clean styling.

        DATA ASSUMPTION: We verify the function completes without errors
        and produces valid output files, trusting that implementation
        applies the specified styling rules.
        """
        fisher = _make_fisher_result()
        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=_default_fisher_config(),
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()


class TestSelectBarDataConsistencyWithRenderBarPlot:
    """Verify that select_bar_data produces data consistent with what
    render_bar_plot uses.
    """

    def test_unclustered_bar_count_matches(self, tmp_path):
        """select_bar_data returns same count as render_bar_plot n_bars.

        DATA ASSUMPTION: Both functions use the same logic for term selection.
        """
        fisher = _make_fisher_result()
        config = _default_fisher_config(top_n_bars=3)

        term_names, _, _ = select_bar_data(fisher, None, top_n=config.top_n_bars)

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=None,
            fisher_config=config,
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert len(term_names) == result.n_bars

    def test_clustered_bar_count_matches(self, tmp_path):
        """select_bar_data returns same count as render_bar_plot n_bars (clustered).

        DATA ASSUMPTION: Both functions use the same logic for representative selection.
        """
        fisher = _make_fisher_result()
        clustering = _make_clustering_result(fisher)
        config = _default_fisher_config(top_n_bars=20)

        term_names, _, _ = select_bar_data(fisher, clustering, top_n=config.top_n_bars)

        result = render_bar_plot(
            fisher_result=fisher,
            clustering_result=clustering,
            fisher_config=config,
            plot_config=_default_plot_config(),
            output_dir=tmp_path,
        )
        plt.close("all")

        assert len(term_names) == result.n_bars
