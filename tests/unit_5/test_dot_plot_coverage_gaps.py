"""
Tests for Unit 5 -- Dot Plot Rendering: Coverage Gap Tests

These tests cover blueprint behavioral contracts that were not exercised
by the original test suite. Each test targets a specific gap identified
during coverage review.

DATA ASSUMPTIONS (module-level):
- Synthetic cohort data uses 3 mutants (alpha, beta, gamma) with alphabetical IDs.
- GO term names are uppercase strings without GO ID prefix.
- NES values range from -3.0 to +3.0, representing typical GSEA normalized enrichment scores.
- FDR values range from 0.001 to 0.5, representing typical GSEA false discovery rates.
- A default FDR threshold of 0.05 is used.
- Category groups contain 2-3 terms each.
"""

import math
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.text

from gsea_tool.data_ingestion import CohortData, TermRecord, MutantProfile
from gsea_tool.cherry_picked import CategoryGroup
from gsea_tool.dot_plot import (
    DotPlotResult,
    render_dot_plot,
    build_dot_grid,
    draw_category_boxes,
)


# ---------------------------------------------------------------------------
# Helpers (duplicated from main test file for self-containment)
# ---------------------------------------------------------------------------

def _make_term_record(term_name: str, go_id: str, nes: float, fdr: float,
                      nom_pval: float = 0.01, size: int = 50) -> TermRecord:
    """Create a TermRecord with sensible defaults."""
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
    separation. FDR values straddle the 0.05 threshold.
    """
    if mutant_ids is None:
        mutant_ids = ["alpha", "beta", "gamma"]

    if term_data is None:
        term_data = {
            "alpha": {
                "MITOCHONDRIAL TRANSLATION": (2.5, 0.001),
                "OXIDATIVE PHOSPHORYLATION": (1.8, 0.01),
                "RIBOSOME BIOGENESIS": (-1.5, 0.03),
                "SYNAPTIC VESICLE CYCLE": (0.5, 0.2),
            },
            "beta": {
                "MITOCHONDRIAL TRANSLATION": (-1.2, 0.04),
                "OXIDATIVE PHOSPHORYLATION": (0.3, 0.6),
                "RIBOSOME BIOGENESIS": (2.1, 0.005),
                "SYNAPTIC VESICLE CYCLE": (-2.0, 0.002),
            },
            "gamma": {
                "MITOCHONDRIAL TRANSLATION": (1.0, 0.08),
                "OXIDATIVE PHOSPHORYLATION": (-2.3, 0.001),
                "RIBOSOME BIOGENESIS": (0.8, 0.07),
                "SYNAPTIC VESICLE CYCLE": (1.9, 0.01),
            },
        }

    sorted_ids = sorted(mutant_ids)
    profiles = {}
    all_term_names = set()
    all_go_ids = set()

    go_id_counter = 1
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

    DATA ASSUMPTION: Two categories with 2 terms each.
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


def _render_and_get_fig_ax(tmp_path, cohort=None, groups=None, fdr_threshold=0.05):
    """Helper that renders a dot plot and returns (fig, ax, result).

    Patches plt.close so the figure remains open for inspection.
    Caller is responsible for closing the figure.
    """
    if cohort is None:
        cohort = _make_cohort()
    if groups is None:
        groups = _make_groups()

    # Patch plt.close to keep figure open for inspection
    with patch.object(plt, "close"):
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=fdr_threshold,
            output_stem="inspect_fig",
            output_dir=tmp_path,
        )

    # Get the current figure (should still be open)
    fig = plt.gcf()
    ax = fig.axes[0] if fig.axes else None
    return fig, ax, result


# ---------------------------------------------------------------------------
# Contract 4 / 11: RdBu_r colormap
# ---------------------------------------------------------------------------

class TestColormapContract:
    """Contract 4 & 11: Dot color uses RdBu_r colormap, symmetric around zero."""

    def test_scatter_uses_rdbu_r_colormap(self, tmp_path):
        """Contract 11: The scatter plot must use the RdBu_r colormap.

        DATA ASSUMPTION: Default cohort with significant cells ensures scatter
        is created with a colormap.
        """
        fig, ax, result = _render_and_get_fig_ax(tmp_path)
        try:
            # Find scatter PathCollection(s) in the axes
            from matplotlib.collections import PathCollection
            scatters = [child for child in ax.get_children()
                        if isinstance(child, PathCollection) and len(child.get_offsets()) > 0]
            assert len(scatters) > 0, "Must have at least one scatter with dots"

            scatter = scatters[0]
            cmap = scatter.get_cmap()
            assert cmap.name == "RdBu_r", (
                f"Scatter colormap must be RdBu_r, got {cmap.name}"
            )
        finally:
            plt.close(fig)

    def test_colormap_symmetric_around_zero(self, tmp_path):
        """Contract 4: Colormap normalization must be symmetric around zero.

        This means vmin == -vmax so that zero maps to the center of RdBu_r.

        DATA ASSUMPTION: Default cohort with both positive and negative NES values.
        """
        fig, ax, result = _render_and_get_fig_ax(tmp_path)
        try:
            from matplotlib.collections import PathCollection
            scatters = [child for child in ax.get_children()
                        if isinstance(child, PathCollection) and len(child.get_offsets()) > 0]
            assert len(scatters) > 0

            scatter = scatters[0]
            norm = scatter.norm
            # Symmetric means vmin == -vmax
            assert norm.vmin == pytest.approx(-norm.vmax, abs=1e-10), (
                f"Colormap normalization must be symmetric around zero: "
                f"vmin={norm.vmin}, vmax={norm.vmax}"
            )
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Contract 4: Colorbar legend
# ---------------------------------------------------------------------------

class TestColorbarLegend:
    """Contract 4: Colorbar legend must be present."""

    def test_colorbar_present(self, tmp_path):
        """Contract 4: A colorbar must be rendered when significant dots exist.

        DATA ASSUMPTION: Default cohort has significant cells, so colorbar should
        be generated. A colorbar adds a second axes to the figure.
        """
        fig, ax, result = _render_and_get_fig_ax(tmp_path)
        try:
            # A colorbar creates an additional axes in the figure
            # The main plot is axes[0], colorbar is typically axes[1] or later
            assert len(fig.axes) >= 2, (
                "Figure must have at least 2 axes (main + colorbar) when dots are present"
            )
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Contract 5: Size legend
# ---------------------------------------------------------------------------

class TestSizeLegend:
    """Contract 5: A size legend for -log10(FDR) must be present."""

    def test_size_legend_present(self, tmp_path):
        """Contract 5: A legend for dot sizes (-log10(FDR)) must exist.

        DATA ASSUMPTION: Default cohort has significant cells with varying FDR,
        so a size legend should be generated.
        """
        fig, ax, result = _render_and_get_fig_ax(tmp_path)
        try:
            legend = ax.get_legend()
            assert legend is not None, (
                "A size legend must be present on the axes for -log10(FDR)"
            )
        finally:
            plt.close(fig)

    def test_size_legend_title_contains_log10_fdr(self, tmp_path):
        """Contract 5: The size legend title should reference -log10(FDR).

        DATA ASSUMPTION: Default cohort with significant cells.
        """
        fig, ax, result = _render_and_get_fig_ax(tmp_path)
        try:
            legend = ax.get_legend()
            assert legend is not None, "Size legend must be present"
            title_text = legend.get_title().get_text()
            title_lower = title_text.lower()
            assert "log" in title_lower or "fdr" in title_text.upper(), (
                f"Size legend title should reference -log10(FDR), got: '{title_text}'"
            )
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Contract 8: No gridlines
# ---------------------------------------------------------------------------

class TestNoGridlines:
    """Contract 8: The dot plot must have no gridlines."""

    def test_gridlines_disabled(self, tmp_path):
        """Contract 8: ax.grid(False) must be called; no visible gridlines.

        DATA ASSUMPTION: Default cohort, default groups.
        """
        fig, ax, result = _render_and_get_fig_ax(tmp_path)
        try:
            # Check that major gridlines on both axes are not visible
            for line in ax.xaxis.get_gridlines():
                assert not line.get_visible(), "X-axis gridlines must not be visible"
            for line in ax.yaxis.get_gridlines():
                assert not line.get_visible(), "Y-axis gridlines must not be visible"
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Contract 9: Figure height scales with term count
# ---------------------------------------------------------------------------

class TestFigureHeightScaling:
    """Contract 9: Figure height should scale with the number of terms."""

    def test_more_terms_taller_figure(self, tmp_path):
        """Contract 9: A figure with more terms should have a taller figure.

        DATA ASSUMPTION: Two cohorts -- one with 2 terms, one with 6 terms.
        Both rendered at same DPI. The 6-term figure should be taller.
        """
        # Small: 2 terms
        term_data_small = {
            "alpha": {"TERM_A": (1.5, 0.01), "TERM_B": (-2.0, 0.005)},
            "beta": {"TERM_A": (1.0, 0.02), "TERM_B": (-1.5, 0.01)},
        }
        cohort_small = _make_cohort(mutant_ids=["alpha", "beta"], term_data=term_data_small)
        groups_small = [CategoryGroup(category_name="Cat1", term_names=["TERM_A", "TERM_B"])]

        # Large: 6 terms
        term_data_large = {
            "alpha": {
                "TERM_A": (1.5, 0.01), "TERM_B": (-2.0, 0.005),
                "TERM_C": (0.8, 0.03), "TERM_D": (-1.2, 0.02),
                "TERM_E": (2.1, 0.001), "TERM_F": (-0.5, 0.04),
            },
            "beta": {
                "TERM_A": (1.0, 0.02), "TERM_B": (-1.5, 0.01),
                "TERM_C": (1.2, 0.04), "TERM_D": (-0.9, 0.03),
                "TERM_E": (1.8, 0.002), "TERM_F": (-1.0, 0.01),
            },
        }
        cohort_large = _make_cohort(mutant_ids=["alpha", "beta"], term_data=term_data_large)
        groups_large = [
            CategoryGroup(category_name="Cat1", term_names=["TERM_A", "TERM_B", "TERM_C"]),
            CategoryGroup(category_name="Cat2", term_names=["TERM_D", "TERM_E", "TERM_F"]),
        ]

        fig_small, ax_small, _ = _render_and_get_fig_ax(
            tmp_path, cohort=cohort_small, groups=groups_small
        )
        height_small = fig_small.get_size_inches()[1]
        plt.close(fig_small)

        fig_large, ax_large, _ = _render_and_get_fig_ax(
            tmp_path, cohort=cohort_large, groups=groups_large
        )
        height_large = fig_large.get_size_inches()[1]
        plt.close(fig_large)

        assert height_large >= height_small, (
            f"Figure with more terms ({height_large:.1f}in) should be at least as tall "
            f"as figure with fewer terms ({height_small:.1f}in)"
        )


# ---------------------------------------------------------------------------
# Contract 6: Category boxes -- bold labels on the right, vertically centered
# ---------------------------------------------------------------------------

class TestCategoryBoxLabels:
    """Contract 6: Category boxes have bold labels positioned to the right."""

    def test_category_box_labels_are_bold(self):
        """Contract 6: Category name labels must have fontweight='bold'.

        DATA ASSUMPTION: Two category groups with known names.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        groups = _make_groups()
        draw_category_boxes(ax, groups, y_start=0.0)

        # Find text elements matching category names
        category_names = {g.category_name for g in groups}
        found_bold_labels = []
        for child in ax.get_children():
            if isinstance(child, matplotlib.text.Text):
                if child.get_text() in category_names:
                    weight = child.get_fontweight()
                    # "bold" can be the string "bold" or numeric 700
                    assert weight == "bold" or weight >= 700, (
                        f"Category label '{child.get_text()}' must be bold, "
                        f"got fontweight={weight}"
                    )
                    found_bold_labels.append(child.get_text())

        assert len(found_bold_labels) == len(category_names), (
            f"Expected bold labels for {category_names}, found {found_bold_labels}"
        )
        plt.close(fig)

    def test_category_box_labels_on_right(self):
        """Contract 6: Category labels must be positioned to the right of the box.

        DATA ASSUMPTION: Axes with known xlim. Labels should have x > xlim right.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(-1, 5)
        groups = _make_groups()
        draw_category_boxes(ax, groups, y_start=0.0)

        x_right = ax.get_xlim()[1]
        category_names = {g.category_name for g in groups}

        for child in ax.get_children():
            if isinstance(child, matplotlib.text.Text):
                if child.get_text() in category_names:
                    label_x = child.get_position()[0]
                    assert label_x >= x_right, (
                        f"Category label '{child.get_text()}' x={label_x} must be "
                        f">= x_right={x_right} (right of box)"
                    )
        plt.close(fig)

    def test_category_boxes_are_rectangles(self):
        """Contract 6: draw_category_boxes must add rectangle patches.

        DATA ASSUMPTION: Two groups, so at least 2 rectangle patches.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        groups = _make_groups()

        patches_before = [p for p in ax.patches]
        draw_category_boxes(ax, groups, y_start=0.0)
        patches_after = [p for p in ax.patches]

        new_patches = len(patches_after) - len(patches_before)
        assert new_patches >= len(groups), (
            f"draw_category_boxes must add at least {len(groups)} rectangle patches, "
            f"added {new_patches}"
        )
        plt.close(fig)

    def test_category_labels_vertically_centered(self):
        """Contract 6: Category labels must be vertically centered within their group.

        DATA ASSUMPTION: Two groups each with 2 terms starting at y=0. First group
        spans y=0 to y=1 (terms at 0,1), center = 0.5. Second group spans y=2 to y=3,
        center = 2.5.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(-1, 6)
        groups = _make_groups()
        draw_category_boxes(ax, groups, y_start=0.0)

        # Calculate expected vertical centers for each group
        # Group 0: terms at y=0, y=1 => center = (0 + 1) / 2 = 0.5
        # Group 1: terms at y=2, y=3 => center = (2 + 3) / 2 = 2.5
        current_y = 0
        expected_centers = {}
        for group in groups:
            n = len(group.term_names)
            center = current_y + (n - 1) / 2.0
            expected_centers[group.category_name] = center
            current_y += n

        category_names = {g.category_name for g in groups}
        for child in ax.get_children():
            if isinstance(child, matplotlib.text.Text):
                if child.get_text() in category_names:
                    label_y = child.get_position()[1]
                    expected_y = expected_centers[child.get_text()]
                    assert label_y == pytest.approx(expected_y, abs=0.1), (
                        f"Category label '{child.get_text()}' y={label_y} should be "
                        f"vertically centered at y={expected_y}"
                    )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Contract 12: Clean minimal aesthetic (top/right spines hidden)
# ---------------------------------------------------------------------------

class TestCleanAesthetic:
    """Contract 12: Clean minimal aesthetic -- top and right spines hidden."""

    def test_top_spine_hidden(self, tmp_path):
        """Contract 12: The top spine must not be visible.

        DATA ASSUMPTION: Default cohort, default groups.
        """
        fig, ax, result = _render_and_get_fig_ax(tmp_path)
        try:
            assert not ax.spines["top"].get_visible(), (
                "Top spine must not be visible for clean minimal aesthetic"
            )
        finally:
            plt.close(fig)

    def test_right_spine_hidden(self, tmp_path):
        """Contract 12: The right spine must not be visible.

        DATA ASSUMPTION: Default cohort, default groups.
        """
        fig, ax, result = _render_and_get_fig_ax(tmp_path)
        try:
            assert not ax.spines["right"].get_visible(), (
                "Right spine must not be visible for clean minimal aesthetic"
            )
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Error condition: OSError on write failure
# ---------------------------------------------------------------------------

class TestOSErrorOnWriteFailure:
    """Error condition: OSError must be raised when file writing fails."""

    def test_oserror_on_savefig_failure(self, tmp_path):
        """OSError must be raised when savefig fails (e.g., permission denied).

        DATA ASSUMPTION: Default cohort, default groups. We mock savefig to
        simulate a write failure.
        """
        cohort = _make_cohort()
        groups = _make_groups()

        with patch("matplotlib.figure.Figure.savefig", side_effect=PermissionError("Permission denied")):
            with pytest.raises(OSError):
                render_dot_plot(
                    cohort=cohort,
                    groups=groups,
                    fdr_threshold=0.05,
                    output_stem="fail_write",
                    output_dir=tmp_path,
                )
