"""
Tests for Unit 5 -- Dot Plot Rendering

This test suite verifies the behavioral contracts, invariants, error conditions,
and signatures specified in the Unit 5 blueprint for dot plot rendering.

DATA ASSUMPTIONS (module-level):
- Synthetic cohort data uses 3 mutants (alpha, beta, gamma) with alphabetical IDs,
  representing a minimal multi-mutant GSEA cohort.
- GO term names are uppercase strings without GO ID prefix (e.g., "MITOCHONDRIAL TRANSLATION"),
  consistent with Unit 1 output convention.
- NES values range from -3.0 to +3.0, representing typical GSEA normalized enrichment scores.
- FDR values range from 0.001 to 0.5, representing typical GSEA false discovery rates.
- A default FDR threshold of 0.05 is used, which is the standard significance cutoff.
- Category groups contain 2-3 terms each, representing a minimal but realistic grouping.
"""

import inspect
from dataclasses import fields as dataclass_fields
from pathlib import Path
from unittest.mock import MagicMock, patch
import math

import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes

from gsea_tool.data_ingestion import CohortData, TermRecord, MutantProfile
from gsea_tool.cherry_picked import CategoryGroup
from gsea_tool.dot_plot import (
    DotPlotResult,
    render_dot_plot,
    build_dot_grid,
    draw_category_boxes,
)


# ---------------------------------------------------------------------------
# Helpers for synthetic data generation
# ---------------------------------------------------------------------------

def _make_term_record(term_name: str, go_id: str, nes: float, fdr: float,
                      nom_pval: float = 0.01, size: int = 50) -> TermRecord:
    """Create a TermRecord with sensible defaults.

    DATA ASSUMPTION: nom_pval=0.01 and size=50 are typical values that don't
    affect dot plot rendering (only NES and FDR matter for the plot).
    """
    return TermRecord(
        term_name=term_name,
        go_id=go_id,
        nes=nes,
        fdr=fdr,
        nom_pval=nom_pval,
        size=size,
    )


def _make_cohort(
    mutant_ids: list[str] | None = None,
    term_data: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> CohortData:
    """Build a CohortData object from simplified term_data.

    DATA ASSUMPTION: 3 mutants (alpha, beta, gamma) with 4 GO terms spanning
    two categories. NES values are chosen to have clear positive/negative
    separation for colormap testing. FDR values straddle the 0.05 threshold
    to test dot presence/absence logic.

    Args:
        mutant_ids: List of mutant identifiers (will be sorted).
        term_data: dict mapping mutant_id -> dict mapping term_name -> (nes, fdr).
    """
    if mutant_ids is None:
        mutant_ids = ["alpha", "beta", "gamma"]

    if term_data is None:
        # Default: 4 terms across 3 mutants with varying significance
        term_data = {
            "alpha": {
                "MITOCHONDRIAL TRANSLATION": (2.5, 0.001),
                "OXIDATIVE PHOSPHORYLATION": (1.8, 0.01),
                "RIBOSOME BIOGENESIS": (-1.5, 0.03),
                "SYNAPTIC VESICLE CYCLE": (0.5, 0.2),  # not significant at 0.05
            },
            "beta": {
                "MITOCHONDRIAL TRANSLATION": (-1.2, 0.04),
                "OXIDATIVE PHOSPHORYLATION": (0.3, 0.6),  # not significant
                "RIBOSOME BIOGENESIS": (2.1, 0.005),
                "SYNAPTIC VESICLE CYCLE": (-2.0, 0.002),
            },
            "gamma": {
                "MITOCHONDRIAL TRANSLATION": (1.0, 0.08),  # not significant at 0.05
                "OXIDATIVE PHOSPHORYLATION": (-2.3, 0.001),
                "RIBOSOME BIOGENESIS": (0.8, 0.07),  # not significant at 0.05
                "SYNAPTIC VESICLE CYCLE": (1.9, 0.01),
            },
        }

    sorted_ids = sorted(mutant_ids)
    profiles = {}
    all_term_names = set()
    all_go_ids = set()

    # DATA ASSUMPTION: GO IDs are synthetic (GO:0000001 etc.) and serve only
    # as unique identifiers. Real GO IDs are 7-digit codes like GO:0006412.
    go_id_counter = 1

    # Build a consistent GO ID mapping
    all_terms_across_mutants = set()
    for m_id in sorted_ids:
        if m_id in term_data:
            all_terms_across_mutants.update(term_data[m_id].keys())
    go_id_map = {}
    for term in sorted(all_terms_across_mutants):
        go_id_map[term] = f"GO:{go_id_counter:07d}"
        go_id_counter += 1

    for m_id in sorted_ids:
        records = {}
        if m_id in term_data:
            for term_name, (nes, fdr) in term_data[m_id].items():
                go_id = go_id_map[term_name]
                rec = _make_term_record(term_name, go_id, nes, fdr)
                records[term_name] = rec
                all_term_names.add(term_name)
                all_go_ids.add(go_id)
        profiles[m_id] = MutantProfile(mutant_id=m_id, records=records)

    return CohortData(
        mutant_ids=sorted_ids,
        profiles=profiles,
        all_term_names=all_term_names,
        all_go_ids=all_go_ids,
    )


def _make_groups() -> list[CategoryGroup]:
    """Build default category groups for testing.

    DATA ASSUMPTION: Two categories ("Mitochondria" and "Signaling") with 2 terms
    each, representing a minimal grouped dot plot layout. Term names match
    the default cohort's term names.
    """
    return [
        CategoryGroup(
            category_name="Mitochondria",
            term_names=["MITOCHONDRIAL TRANSLATION", "OXIDATIVE PHOSPHORYLATION"],
        ),
        CategoryGroup(
            category_name="Signaling",
            term_names=["RIBOSOME BIOGENESIS", "SYNAPTIC VESICLE CYCLE"],
        ),
    ]


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------

class TestSignatures:
    """Verify that function and class signatures match the blueprint."""

    def test_dot_plot_result_is_dataclass(self):
        """DotPlotResult must be a dataclass."""
        assert hasattr(DotPlotResult, "__dataclass_fields__"), (
            "DotPlotResult must be a dataclass"
        )

    def test_dot_plot_result_fields(self):
        """DotPlotResult must have the specified fields with correct types."""
        field_map = {f.name: f.type for f in dataclass_fields(DotPlotResult)}
        assert "pdf_path" in field_map, "DotPlotResult must have pdf_path field"
        assert "png_path" in field_map, "DotPlotResult must have png_path field"
        assert "svg_path" in field_map, "DotPlotResult must have svg_path field"
        assert "n_terms_displayed" in field_map, "DotPlotResult must have n_terms_displayed field"
        assert "n_categories" in field_map, "DotPlotResult must have n_categories field"
        assert "n_mutants" in field_map, "DotPlotResult must have n_mutants field"

    def test_render_dot_plot_signature(self):
        """render_dot_plot must accept the blueprint parameters."""
        sig = inspect.signature(render_dot_plot)
        param_names = list(sig.parameters.keys())
        assert "cohort" in param_names
        assert "groups" in param_names
        assert "fdr_threshold" in param_names
        assert "output_stem" in param_names
        assert "output_dir" in param_names

    def test_render_dot_plot_optional_params(self):
        """render_dot_plot must have optional dpi, font_family, and title parameters."""
        sig = inspect.signature(render_dot_plot)
        params = sig.parameters

        # dpi should have a default
        if "dpi" in params:
            assert params["dpi"].default != inspect.Parameter.empty, (
                "dpi should have a default value"
            )

        # title should have a default
        if "title" in params:
            assert params["title"].default != inspect.Parameter.empty, (
                "title should have a default value"
            )

    def test_render_dot_plot_return_type(self):
        """render_dot_plot must return DotPlotResult."""
        sig = inspect.signature(render_dot_plot)
        # Check the return annotation if present
        if sig.return_annotation != inspect.Signature.empty:
            assert sig.return_annotation is DotPlotResult or (
                hasattr(sig.return_annotation, "__name__")
                and sig.return_annotation.__name__ == "DotPlotResult"
            ), f"render_dot_plot should return DotPlotResult, got {sig.return_annotation}"

    def test_build_dot_grid_signature(self):
        """build_dot_grid must accept cohort, groups, fdr_threshold."""
        sig = inspect.signature(build_dot_grid)
        param_names = list(sig.parameters.keys())
        assert "cohort" in param_names
        assert "groups" in param_names
        assert "fdr_threshold" in param_names

    def test_draw_category_boxes_signature(self):
        """draw_category_boxes must accept ax, groups, y_start."""
        sig = inspect.signature(draw_category_boxes)
        param_names = list(sig.parameters.keys())
        assert "ax" in param_names
        assert "groups" in param_names
        assert "y_start" in param_names


# ---------------------------------------------------------------------------
# build_dot_grid tests
# ---------------------------------------------------------------------------

class TestBuildDotGrid:
    """Tests for the build_dot_grid function."""

    def test_returns_four_element_tuple(self):
        """build_dot_grid must return a 4-tuple: nes_matrix, sig_matrix, term_labels, mutant_labels."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = build_dot_grid(cohort, groups, fdr_threshold=0.05)
        assert isinstance(result, tuple), "build_dot_grid must return a tuple"
        assert len(result) == 4, "build_dot_grid must return a 4-element tuple"

    def test_mutant_labels_alphabetical(self):
        """Contract 1: X-axis mutant labels must be in alphabetical order."""
        cohort = _make_cohort()
        groups = _make_groups()
        _, _, _, mutant_labels = build_dot_grid(cohort, groups, fdr_threshold=0.05)
        assert mutant_labels == sorted(mutant_labels), (
            "Mutant labels must be in alphabetical order"
        )
        assert mutant_labels == sorted(cohort.mutant_ids), (
            "Mutant labels must match cohort.mutant_ids alphabetically"
        )

    def test_term_labels_grouped_order(self):
        """Contract 2: Y-axis term labels must follow groups order with terms
        within each category in the order given by group.term_names."""
        cohort = _make_cohort()
        groups = _make_groups()
        _, _, term_labels, _ = build_dot_grid(cohort, groups, fdr_threshold=0.05)

        expected_terms = []
        for g in groups:
            expected_terms.extend(g.term_names)

        assert term_labels == expected_terms, (
            f"Term labels must follow category group order. "
            f"Expected {expected_terms}, got {term_labels}"
        )

    def test_matrix_dimensions(self):
        """Matrices must have dimensions [n_terms][n_mutants]."""
        cohort = _make_cohort()
        groups = _make_groups()
        nes_matrix, sig_matrix, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )
        n_terms = len(term_labels)
        n_mutants = len(mutant_labels)

        assert len(nes_matrix) == n_terms
        assert len(sig_matrix) == n_terms
        for row in nes_matrix:
            assert len(row) == n_mutants
        for row in sig_matrix:
            assert len(row) == n_mutants

    def test_significant_cells_have_nes_values(self):
        """Contract 3: Cells where FDR < threshold must have NES values (not None)."""
        # DATA ASSUMPTION: alpha has MITOCHONDRIAL TRANSLATION with FDR=0.001 < 0.05
        cohort = _make_cohort()
        groups = _make_groups()
        nes_matrix, sig_matrix, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )

        # Find the index for "MITOCHONDRIAL TRANSLATION" and "alpha"
        term_idx = term_labels.index("MITOCHONDRIAL TRANSLATION")
        mutant_idx = mutant_labels.index("alpha")

        assert nes_matrix[term_idx][mutant_idx] is not None, (
            "Significant cell must have NES value"
        )
        assert sig_matrix[term_idx][mutant_idx] is not None, (
            "Significant cell must have significance value"
        )

    def test_nonsignificant_cells_are_none(self):
        """Contract 3: Cells where FDR >= threshold must be None (empty)."""
        # DATA ASSUMPTION: alpha has SYNAPTIC VESICLE CYCLE with FDR=0.2 >= 0.05
        cohort = _make_cohort()
        groups = _make_groups()
        nes_matrix, sig_matrix, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )

        term_idx = term_labels.index("SYNAPTIC VESICLE CYCLE")
        mutant_idx = mutant_labels.index("alpha")

        assert nes_matrix[term_idx][mutant_idx] is None, (
            "Non-significant cell (FDR >= threshold) must be None"
        )
        assert sig_matrix[term_idx][mutant_idx] is None, (
            "Non-significant cell (FDR >= threshold) must have None significance"
        )

    def test_nes_values_match_cohort_data(self):
        """NES matrix values must match the actual NES from cohort data for significant cells."""
        cohort = _make_cohort()
        groups = _make_groups()
        nes_matrix, _, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )

        # Check alpha / MITOCHONDRIAL TRANSLATION: NES = 2.5, FDR = 0.001
        term_idx = term_labels.index("MITOCHONDRIAL TRANSLATION")
        mutant_idx = mutant_labels.index("alpha")
        assert nes_matrix[term_idx][mutant_idx] == pytest.approx(2.5, abs=1e-6)

    def test_sig_values_are_neg_log10_fdr(self):
        """Contract 5: Significance values must be -log10(FDR) for significant cells."""
        cohort = _make_cohort()
        groups = _make_groups()
        _, sig_matrix, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )

        # alpha / MITOCHONDRIAL TRANSLATION: FDR = 0.001
        term_idx = term_labels.index("MITOCHONDRIAL TRANSLATION")
        mutant_idx = mutant_labels.index("alpha")
        expected_sig = -math.log10(0.001)  # = 3.0
        assert sig_matrix[term_idx][mutant_idx] == pytest.approx(expected_sig, abs=1e-4)

    def test_fdr_at_exact_threshold_is_empty(self):
        """A cell with FDR exactly equal to threshold should be empty (FDR < threshold, not <=)."""
        # DATA ASSUMPTION: Custom cohort with FDR exactly at 0.05 boundary
        term_data = {
            "alpha": {"TERM_A": (1.5, 0.05)},  # FDR == threshold => not significant
            "beta": {"TERM_A": (1.0, 0.049)},   # FDR < threshold => significant
        }
        cohort = _make_cohort(mutant_ids=["alpha", "beta"], term_data=term_data)
        groups = [CategoryGroup(category_name="Cat1", term_names=["TERM_A"])]

        nes_matrix, sig_matrix, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )

        alpha_idx = mutant_labels.index("alpha")
        beta_idx = mutant_labels.index("beta")

        # alpha: FDR == 0.05, should be None
        assert nes_matrix[0][alpha_idx] is None, (
            "FDR equal to threshold should result in empty cell"
        )
        # beta: FDR < 0.05, should have value
        assert nes_matrix[0][beta_idx] is not None, (
            "FDR below threshold should have NES value"
        )

    def test_missing_term_in_mutant_is_none(self):
        """If a term is not present in a mutant's profile, cell should be None."""
        # DATA ASSUMPTION: Custom cohort where beta has no record for TERM_X
        term_data = {
            "alpha": {"TERM_X": (1.5, 0.01)},
            "beta": {},  # no terms at all
        }
        cohort = _make_cohort(mutant_ids=["alpha", "beta"], term_data=term_data)
        groups = [CategoryGroup(category_name="Cat1", term_names=["TERM_X"])]

        nes_matrix, _, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )

        beta_idx = mutant_labels.index("beta")
        assert nes_matrix[0][beta_idx] is None, (
            "Missing term in mutant profile should yield None"
        )


# ---------------------------------------------------------------------------
# render_dot_plot tests
# ---------------------------------------------------------------------------

class TestRenderDotPlot:
    """Tests for the render_dot_plot function."""

    def test_creates_all_three_output_files(self, tmp_path):
        """Contract 10: Must create PDF, PNG, and SVG files."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="test_figure",
            output_dir=tmp_path,
        )
        assert result.pdf_path.exists(), "PDF file must be written"
        assert result.png_path.exists(), "PNG file must be written"
        assert result.svg_path.exists(), "SVG file must be written"

    def test_output_filenames_use_stem(self, tmp_path):
        """Contract 10: Files must be named {output_stem}.pdf/.png/.svg."""
        cohort = _make_cohort()
        groups = _make_groups()
        stem = "my_dot_plot"
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem=stem,
            output_dir=tmp_path,
        )
        assert result.pdf_path == tmp_path / f"{stem}.pdf"
        assert result.png_path == tmp_path / f"{stem}.png"
        assert result.svg_path == tmp_path / f"{stem}.svg"

    def test_output_files_in_output_dir(self, tmp_path):
        """Files must be written in the specified output_dir."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="fig",
            output_dir=tmp_path,
        )
        assert result.pdf_path.parent == tmp_path
        assert result.png_path.parent == tmp_path
        assert result.svg_path.parent == tmp_path

    def test_n_terms_displayed_matches_groups(self, tmp_path):
        """Invariant: n_terms_displayed == sum(len(g.term_names) for g in groups)."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="fig",
            output_dir=tmp_path,
        )
        expected = sum(len(g.term_names) for g in groups)
        assert result.n_terms_displayed == expected, (
            f"n_terms_displayed should be {expected}, got {result.n_terms_displayed}"
        )

    def test_n_categories_matches_groups(self, tmp_path):
        """n_categories must equal the number of groups provided."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="fig",
            output_dir=tmp_path,
        )
        assert result.n_categories == len(groups), (
            f"n_categories should be {len(groups)}, got {result.n_categories}"
        )

    def test_n_mutants_matches_cohort(self, tmp_path):
        """Invariant: n_mutants == len(cohort.mutant_ids)."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="fig",
            output_dir=tmp_path,
        )
        assert result.n_mutants == len(cohort.mutant_ids), (
            f"n_mutants should be {len(cohort.mutant_ids)}, got {result.n_mutants}"
        )

    def test_returns_dot_plot_result(self, tmp_path):
        """render_dot_plot must return a DotPlotResult instance."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="fig",
            output_dir=tmp_path,
        )
        assert isinstance(result, DotPlotResult)

    def test_pdf_file_is_nonempty(self, tmp_path):
        """PDF output must contain data (non-zero file size)."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="fig",
            output_dir=tmp_path,
        )
        assert result.pdf_path.stat().st_size > 0, "PDF file must be non-empty"

    def test_png_file_is_nonempty(self, tmp_path):
        """PNG output must contain data (non-zero file size)."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="fig",
            output_dir=tmp_path,
        )
        assert result.png_path.stat().st_size > 0, "PNG file must be non-empty"

    def test_svg_file_is_nonempty(self, tmp_path):
        """SVG output must contain data (non-zero file size)."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="fig",
            output_dir=tmp_path,
        )
        assert result.svg_path.stat().st_size > 0, "SVG file must be non-empty"

    def test_single_category_group(self, tmp_path):
        """Should work correctly with a single category group."""
        cohort = _make_cohort()
        groups = [
            CategoryGroup(
                category_name="Mito",
                term_names=["MITOCHONDRIAL TRANSLATION", "OXIDATIVE PHOSPHORYLATION"],
            )
        ]
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="single_cat",
            output_dir=tmp_path,
        )
        assert result.n_categories == 1
        assert result.n_terms_displayed == 2

    def test_many_categories(self, tmp_path):
        """Should work with multiple category groups."""
        # DATA ASSUMPTION: 4 single-term categories to test multi-group layout
        term_data = {
            "alpha": {
                "TERM_A": (1.5, 0.01),
                "TERM_B": (-2.0, 0.005),
                "TERM_C": (0.8, 0.03),
                "TERM_D": (-1.2, 0.02),
            },
            "beta": {
                "TERM_A": (1.0, 0.02),
                "TERM_B": (-1.5, 0.01),
                "TERM_C": (1.2, 0.04),
                "TERM_D": (-0.9, 0.03),
            },
        }
        cohort = _make_cohort(mutant_ids=["alpha", "beta"], term_data=term_data)
        groups = [
            CategoryGroup(category_name="Cat1", term_names=["TERM_A"]),
            CategoryGroup(category_name="Cat2", term_names=["TERM_B"]),
            CategoryGroup(category_name="Cat3", term_names=["TERM_C"]),
            CategoryGroup(category_name="Cat4", term_names=["TERM_D"]),
        ]
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="multi_cat",
            output_dir=tmp_path,
        )
        assert result.n_categories == 4
        assert result.n_terms_displayed == 4
        assert result.n_mutants == 2

    def test_with_title(self, tmp_path):
        """Should accept and use a title string."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="titled",
            output_dir=tmp_path,
            title="Figure 2: GSEA Dot Plot",
        )
        # Should still produce valid output
        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()

    def test_custom_dpi(self, tmp_path):
        """Should accept a custom DPI parameter."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="hires",
            output_dir=tmp_path,
            dpi=150,
        )
        assert result.pdf_path.exists()
        assert result.png_path.exists()

    def test_all_cells_significant(self, tmp_path):
        """When all cells are significant, all dots should be rendered."""
        # DATA ASSUMPTION: All FDR values well below threshold
        term_data = {
            "alpha": {"TERM_A": (2.0, 0.001), "TERM_B": (-1.5, 0.002)},
            "beta": {"TERM_A": (1.0, 0.01), "TERM_B": (-2.0, 0.005)},
        }
        cohort = _make_cohort(mutant_ids=["alpha", "beta"], term_data=term_data)
        groups = [CategoryGroup(category_name="All", term_names=["TERM_A", "TERM_B"])]
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="all_sig",
            output_dir=tmp_path,
        )
        assert result.pdf_path.exists()
        assert result.n_terms_displayed == 2

    def test_all_cells_nonsignificant(self, tmp_path):
        """When no cells are significant, figure should still render (all empty cells)."""
        # DATA ASSUMPTION: All FDR values above threshold
        term_data = {
            "alpha": {"TERM_A": (2.0, 0.5), "TERM_B": (-1.5, 0.3)},
            "beta": {"TERM_A": (1.0, 0.8), "TERM_B": (-2.0, 0.6)},
        }
        cohort = _make_cohort(mutant_ids=["alpha", "beta"], term_data=term_data)
        groups = [CategoryGroup(category_name="NoSig", term_names=["TERM_A", "TERM_B"])]
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="no_sig",
            output_dir=tmp_path,
        )
        assert result.pdf_path.exists()
        assert result.n_terms_displayed == 2


# ---------------------------------------------------------------------------
# Error condition tests
# ---------------------------------------------------------------------------

class TestErrorConditions:
    """Test all error conditions from the blueprint."""

    def test_empty_groups_raises_value_error(self, tmp_path):
        """ValueError must be raised when groups is an empty list."""
        cohort = _make_cohort()
        with pytest.raises(ValueError):
            render_dot_plot(
                cohort=cohort,
                groups=[],
                fdr_threshold=0.05,
                output_stem="fig",
                output_dir=tmp_path,
            )

    def test_output_dir_not_exist_raises_error(self, tmp_path):
        """Error when output_dir does not exist.

        The blueprint specifies OSError for output write failure and an invariant
        that output_dir must be a directory. Either AssertionError or OSError
        is acceptable depending on implementation (invariant vs explicit check).
        """
        cohort = _make_cohort()
        groups = _make_groups()
        nonexistent = tmp_path / "truly_nonexistent_subdir"
        assert not nonexistent.exists()
        with pytest.raises((OSError, AssertionError, ValueError)):
            render_dot_plot(
                cohort=cohort,
                groups=groups,
                fdr_threshold=0.05,
                output_stem="fig",
                output_dir=nonexistent,
            )

    def test_empty_groups_returns_empty_matrices_with_build_dot_grid(self):
        """build_dot_grid with empty groups returns empty term labels and matrices."""
        cohort = _make_cohort()
        nes_matrix, sig_matrix, term_labels, mutant_labels = build_dot_grid(
            cohort, [], fdr_threshold=0.05
        )
        assert term_labels == []
        assert len(nes_matrix) == 0
        assert len(sig_matrix) == 0


# ---------------------------------------------------------------------------
# draw_category_boxes tests
# ---------------------------------------------------------------------------

class TestDrawCategoryBoxes:
    """Tests for the draw_category_boxes function."""

    def test_does_not_raise_on_valid_input(self):
        """draw_category_boxes must not raise with valid inputs."""
        fig, ax = plt.subplots()
        groups = _make_groups()
        try:
            draw_category_boxes(ax, groups, y_start=0.0)
        finally:
            plt.close(fig)

    def test_adds_visual_elements_to_axes(self):
        """draw_category_boxes must add patches/texts to the axes.

        Contract 6: Category boxes as visible rectangles with bold text labels.
        """
        fig, ax = plt.subplots()
        groups = _make_groups()

        # Count children before
        children_before = len(ax.get_children())

        draw_category_boxes(ax, groups, y_start=0.0)

        children_after = len(ax.get_children())
        # Must have added something (rectangles, text, etc.)
        assert children_after > children_before, (
            "draw_category_boxes must add visual elements to axes"
        )
        plt.close(fig)

    def test_handles_single_group(self):
        """draw_category_boxes should work with a single category group."""
        fig, ax = plt.subplots()
        groups = [CategoryGroup(category_name="Single", term_names=["TERM_A"])]
        try:
            draw_category_boxes(ax, groups, y_start=0.0)
        finally:
            plt.close(fig)

    def test_handles_multiple_groups(self):
        """draw_category_boxes should work with multiple category groups."""
        fig, ax = plt.subplots()
        groups = [
            CategoryGroup(category_name="Cat1", term_names=["T1", "T2"]),
            CategoryGroup(category_name="Cat2", term_names=["T3"]),
            CategoryGroup(category_name="Cat3", term_names=["T4", "T5", "T6"]),
        ]
        try:
            draw_category_boxes(ax, groups, y_start=0.0)
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Behavioral contract tests (visual/rendering properties)
# ---------------------------------------------------------------------------

class TestBehavioralContracts:
    """Test behavioral contracts that can be verified programmatically."""

    def test_mutants_alphabetical_on_xaxis(self, tmp_path):
        """Contract 1: X-axis displays mutant identifiers in alphabetical order.

        DATA ASSUMPTION: Mutant IDs 'gamma', 'alpha', 'beta' given in non-alphabetical
        order to verify sorting. The cohort auto-sorts, but we verify the grid output.
        """
        cohort = _make_cohort()
        groups = _make_groups()
        _, _, _, mutant_labels = build_dot_grid(cohort, groups, fdr_threshold=0.05)
        assert mutant_labels == ["alpha", "beta", "gamma"]

    def test_terms_follow_group_order(self, tmp_path):
        """Contract 2: Y-axis follows category group order."""
        cohort = _make_cohort()
        groups = _make_groups()
        _, _, term_labels, _ = build_dot_grid(cohort, groups, fdr_threshold=0.05)

        # First two terms from group 1, next two from group 2
        assert term_labels[0] == "MITOCHONDRIAL TRANSLATION"
        assert term_labels[1] == "OXIDATIVE PHOSPHORYLATION"
        assert term_labels[2] == "RIBOSOME BIOGENESIS"
        assert term_labels[3] == "SYNAPTIC VESICLE CYCLE"

    def test_dot_presence_matches_fdr_threshold(self, tmp_path):
        """Contract 3: Dot present only if FDR < threshold; empty otherwise."""
        cohort = _make_cohort()
        groups = _make_groups()
        nes_matrix, _, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )

        for i, term_name in enumerate(term_labels):
            for j, mutant_id in enumerate(mutant_labels):
                profile = cohort.profiles.get(mutant_id)
                if profile is None:
                    continue
                record = profile.records.get(term_name)
                if record is not None and record.fdr < 0.05:
                    assert nes_matrix[i][j] is not None, (
                        f"Dot should exist at ({term_name}, {mutant_id}), "
                        f"FDR={record.fdr} < 0.05"
                    )
                else:
                    assert nes_matrix[i][j] is None, (
                        f"Cell should be empty at ({term_name}, {mutant_id}), "
                        f"FDR={'N/A' if record is None else record.fdr} >= 0.05 or missing"
                    )

    def test_nes_values_preserve_sign(self, tmp_path):
        """Contract 4: NES values must preserve sign (positive/negative) for colormap."""
        cohort = _make_cohort()
        groups = _make_groups()
        nes_matrix, _, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )

        # alpha / MITOCHONDRIAL TRANSLATION: NES = 2.5 (positive, should be red)
        t_idx = term_labels.index("MITOCHONDRIAL TRANSLATION")
        m_idx = mutant_labels.index("alpha")
        assert nes_matrix[t_idx][m_idx] > 0, "Positive NES should remain positive"

        # beta / SYNAPTIC VESICLE CYCLE: NES = -2.0 (negative, should be blue)
        t_idx = term_labels.index("SYNAPTIC VESICLE CYCLE")
        m_idx = mutant_labels.index("beta")
        assert nes_matrix[t_idx][m_idx] < 0, "Negative NES should remain negative"

    def test_sig_encodes_neg_log10_fdr(self, tmp_path):
        """Contract 5: Dot size encodes -log10(FDR)."""
        cohort = _make_cohort()
        groups = _make_groups()
        _, sig_matrix, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )

        # beta / RIBOSOME BIOGENESIS: FDR = 0.005
        t_idx = term_labels.index("RIBOSOME BIOGENESIS")
        m_idx = mutant_labels.index("beta")
        expected = -math.log10(0.005)
        assert sig_matrix[t_idx][m_idx] == pytest.approx(expected, abs=0.01)

        # beta / SYNAPTIC VESICLE CYCLE: FDR = 0.002
        t_idx = term_labels.index("SYNAPTIC VESICLE CYCLE")
        m_idx = mutant_labels.index("beta")
        expected = -math.log10(0.002)
        assert sig_matrix[t_idx][m_idx] == pytest.approx(expected, abs=0.01)

    def test_high_fdr_threshold_includes_more_dots(self, tmp_path):
        """Higher FDR threshold should include more dots (fewer None cells)."""
        cohort = _make_cohort()
        groups = _make_groups()

        nes_strict, _, _, _ = build_dot_grid(cohort, groups, fdr_threshold=0.05)
        nes_relaxed, _, _, _ = build_dot_grid(cohort, groups, fdr_threshold=0.5)

        def count_nones(matrix):
            return sum(1 for row in matrix for cell in row if cell is None)

        nones_strict = count_nones(nes_strict)
        nones_relaxed = count_nones(nes_relaxed)
        assert nones_relaxed <= nones_strict, (
            "Relaxing FDR threshold should not increase the number of empty cells"
        )

    def test_low_fdr_threshold_excludes_more_dots(self, tmp_path):
        """Very low FDR threshold should exclude many dots."""
        cohort = _make_cohort()
        groups = _make_groups()

        nes_very_strict, _, _, _ = build_dot_grid(cohort, groups, fdr_threshold=0.001)

        # DATA ASSUMPTION: Only alpha/MITOCHONDRIAL TRANSLATION has FDR=0.001 which
        # is NOT < 0.001, so at threshold 0.001 almost nothing should be present.
        # Actually FDR=0.001 is NOT < 0.001, so this cell should also be None.
        def count_non_none(matrix):
            return sum(1 for row in matrix for cell in row if cell is not None)

        non_none = count_non_none(nes_very_strict)
        # With threshold 0.001, only cells with FDR < 0.001 should be present.
        # In our data, no cell has FDR strictly < 0.001, so all should be None.
        assert non_none == 0, (
            f"At FDR threshold 0.001, no cells should have dots (got {non_none})"
        )


# ---------------------------------------------------------------------------
# Integration-level render tests (verify full pipeline output)
# ---------------------------------------------------------------------------

class TestRenderIntegration:
    """Integration tests that verify the full render pipeline."""

    def test_render_with_default_cohort(self, tmp_path):
        """Full render with default cohort produces valid DotPlotResult."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="integration_test",
            output_dir=tmp_path,
        )
        assert isinstance(result, DotPlotResult)
        assert result.n_terms_displayed == 4
        assert result.n_categories == 2
        assert result.n_mutants == 3
        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()

    def test_render_two_mutants(self, tmp_path):
        """Render works correctly with just 2 mutants.

        DATA ASSUMPTION: Minimal 2-mutant cohort to verify edge case near minimum size.
        """
        term_data = {
            "delta": {"CELL CYCLE": (1.8, 0.01), "APOPTOSIS": (-2.5, 0.003)},
            "epsilon": {"CELL CYCLE": (-1.0, 0.04), "APOPTOSIS": (0.5, 0.15)},
        }
        cohort = _make_cohort(mutant_ids=["delta", "epsilon"], term_data=term_data)
        groups = [
            CategoryGroup(category_name="Proliferation", term_names=["CELL CYCLE", "APOPTOSIS"]),
        ]
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="two_mutants",
            output_dir=tmp_path,
        )
        assert result.n_mutants == 2
        assert result.n_terms_displayed == 2

    def test_render_large_cohort(self, tmp_path):
        """Render works with a larger cohort (10 mutants, 8 terms).

        DATA ASSUMPTION: 10 mutants with sequential IDs and 8 terms in 3 categories.
        NES values generated deterministically using simple formula, FDR values
        all below 0.05 for simplicity.
        """
        import random
        rng = random.Random(42)  # deterministic seed

        mutant_ids = [f"mutant_{i:02d}" for i in range(10)]
        term_names = [
            "TERM_A", "TERM_B", "TERM_C",
            "TERM_D", "TERM_E",
            "TERM_F", "TERM_G", "TERM_H",
        ]

        term_data = {}
        for m_id in mutant_ids:
            term_data[m_id] = {}
            for t in term_names:
                nes = rng.uniform(-3.0, 3.0)
                fdr = rng.uniform(0.001, 0.04)
                term_data[m_id][t] = (round(nes, 3), round(fdr, 4))

        cohort = _make_cohort(mutant_ids=mutant_ids, term_data=term_data)
        groups = [
            CategoryGroup(category_name="Group1", term_names=["TERM_A", "TERM_B", "TERM_C"]),
            CategoryGroup(category_name="Group2", term_names=["TERM_D", "TERM_E"]),
            CategoryGroup(category_name="Group3", term_names=["TERM_F", "TERM_G", "TERM_H"]),
        ]

        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="large",
            output_dir=tmp_path,
        )
        assert result.n_mutants == 10
        assert result.n_terms_displayed == 8
        assert result.n_categories == 3
        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()

    def test_render_with_empty_title(self, tmp_path):
        """Render with empty string title should work."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="no_title",
            output_dir=tmp_path,
            title="",
        )
        assert result.pdf_path.exists()

    def test_different_output_stems_produce_different_files(self, tmp_path):
        """Two calls with different stems should produce different file sets."""
        cohort = _make_cohort()
        groups = _make_groups()

        result1 = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="stem_a",
            output_dir=tmp_path,
        )
        result2 = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="stem_b",
            output_dir=tmp_path,
        )

        assert result1.pdf_path != result2.pdf_path
        assert result1.png_path != result2.png_path
        assert result1.svg_path != result2.svg_path
        assert result1.pdf_path.exists()
        assert result2.pdf_path.exists()


# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------

class TestInvariants:
    """Test pre-conditions and post-conditions from the blueprint."""

    def test_precondition_groups_nonempty(self, tmp_path):
        """Pre-condition: len(groups) > 0 must be enforced."""
        cohort = _make_cohort()
        with pytest.raises((ValueError, AssertionError)):
            render_dot_plot(
                cohort=cohort,
                groups=[],
                fdr_threshold=0.05,
                output_stem="fig",
                output_dir=tmp_path,
            )

    def test_postcondition_files_exist(self, tmp_path):
        """Post-condition: All three output files must exist after render."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="post",
            output_dir=tmp_path,
        )
        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()

    def test_postcondition_term_count(self, tmp_path):
        """Post-condition: n_terms_displayed == sum(len(g.term_names) for g in groups)."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="count",
            output_dir=tmp_path,
        )
        expected = sum(len(g.term_names) for g in groups)
        assert result.n_terms_displayed == expected

    def test_postcondition_mutant_count(self, tmp_path):
        """Post-condition: n_mutants == len(cohort.mutant_ids)."""
        cohort = _make_cohort()
        groups = _make_groups()
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="mcnt",
            output_dir=tmp_path,
        )
        assert result.n_mutants == len(cohort.mutant_ids)

    def test_precondition_no_empty_groups(self, tmp_path):
        """Pre-condition: all groups must have at least one term (no empty groups).

        The invariant says all(len(g.term_names) > 0 for g in groups).
        """
        cohort = _make_cohort()
        groups = [
            CategoryGroup(category_name="Empty", term_names=[]),
            CategoryGroup(
                category_name="NonEmpty",
                term_names=["MITOCHONDRIAL TRANSLATION"],
            ),
        ]
        with pytest.raises((ValueError, AssertionError)):
            render_dot_plot(
                cohort=cohort,
                groups=groups,
                fdr_threshold=0.05,
                output_stem="empty_group",
                output_dir=tmp_path,
            )

    def test_precondition_output_dir_exists(self, tmp_path):
        """Pre-condition: output_dir must be an existing directory."""
        cohort = _make_cohort()
        groups = _make_groups()
        nonexistent = tmp_path / "truly_nonexistent_subdir"
        assert not nonexistent.exists()
        with pytest.raises((AssertionError, OSError, ValueError)):
            render_dot_plot(
                cohort=cohort,
                groups=groups,
                fdr_threshold=0.05,
                output_stem="fig",
                output_dir=nonexistent,
            )
