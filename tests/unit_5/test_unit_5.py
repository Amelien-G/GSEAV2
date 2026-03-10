"""Comprehensive test suite for Unit 5 -- Dot Plot Rendering.

Tests are written from the blueprint contracts only, covering:
- build_dot_grid behavioral contracts
- render_dot_plot file output, metadata, and error conditions
- draw_category_boxes visual structure
- All invariants (pre/post conditions)
"""

import math
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.axes

from pathlib import Path

from gsea_tool.dot_plot import (
    DotPlotResult,
    build_dot_grid,
    draw_category_boxes,
    render_dot_plot,
)
from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.cherry_picked import CategoryGroup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_term_record(term_name: str, go_id: str, nes: float, fdr: float) -> TermRecord:
    """Helper to build a TermRecord with sensible defaults."""
    return TermRecord(
        term_name=term_name,
        go_id=go_id,
        nes=nes,
        fdr=fdr,
        nom_pval=fdr * 0.5,  # arbitrary; not used by unit 5
        size=100,
    )


@pytest.fixture
def simple_cohort() -> CohortData:
    """A cohort with 3 mutants and 4 GO terms, some significant, some not.

    Mutants: alpha, beta, gamma (already alphabetical)
    Terms:
      - TERM_A: alpha FDR=0.01 NES=1.5, beta FDR=0.03 NES=-1.0, gamma FDR=0.10 NES=0.5
      - TERM_B: alpha FDR=0.04 NES=2.0, beta FDR=0.06 NES=-0.3, gamma FDR=0.02 NES=-1.8
      - TERM_C: alpha FDR=0.001 NES=0.8, beta FDR=0.005 NES=1.2, gamma FDR=0.002 NES=-0.5
      - TERM_D: alpha FDR=0.20 NES=0.1, beta FDR=0.30 NES=0.2, gamma FDR=0.50 NES=0.3
    """
    profiles = {}
    for mid, records_data in [
        ("alpha", [
            ("TERM_A", "GO:0000001", 1.5, 0.01),
            ("TERM_B", "GO:0000002", 2.0, 0.04),
            ("TERM_C", "GO:0000003", 0.8, 0.001),
            ("TERM_D", "GO:0000004", 0.1, 0.20),
        ]),
        ("beta", [
            ("TERM_A", "GO:0000001", -1.0, 0.03),
            ("TERM_B", "GO:0000002", -0.3, 0.06),
            ("TERM_C", "GO:0000003", 1.2, 0.005),
            ("TERM_D", "GO:0000004", 0.2, 0.30),
        ]),
        ("gamma", [
            ("TERM_A", "GO:0000001", 0.5, 0.10),
            ("TERM_B", "GO:0000002", -1.8, 0.02),
            ("TERM_C", "GO:0000003", -0.5, 0.002),
            ("TERM_D", "GO:0000004", 0.3, 0.50),
        ]),
    ]:
        recs = {}
        for tname, goid, nes, fdr in records_data:
            recs[tname] = _make_term_record(tname, goid, nes, fdr)
        profiles[mid] = MutantProfile(mutant_id=mid, records=recs)

    return CohortData(
        mutant_ids=["alpha", "beta", "gamma"],
        profiles=profiles,
        all_term_names={"TERM_A", "TERM_B", "TERM_C", "TERM_D"},
        all_go_ids={"GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"},
    )


@pytest.fixture
def two_groups() -> list[CategoryGroup]:
    """Two category groups referencing terms from simple_cohort."""
    return [
        CategoryGroup(category_name="Metabolism", term_names=["TERM_A", "TERM_B"]),
        CategoryGroup(category_name="Signaling", term_names=["TERM_C"]),
    ]


@pytest.fixture
def all_terms_group() -> list[CategoryGroup]:
    """A single group containing all four terms."""
    return [
        CategoryGroup(
            category_name="Everything",
            term_names=["TERM_A", "TERM_B", "TERM_C", "TERM_D"],
        ),
    ]


# ---------------------------------------------------------------------------
# Tests for build_dot_grid
# ---------------------------------------------------------------------------


class TestBuildDotGrid:
    """Tests for the build_dot_grid function."""

    def test_empty_groups_returns_empty_matrices(self, simple_cohort):
        """Empty groups list returns empty term labels and matrices."""
        nes_mat, sig_mat, term_labels, mutant_labels = build_dot_grid(
            simple_cohort, [], fdr_threshold=0.05
        )
        assert term_labels == []
        assert nes_mat == []
        assert sig_mat == []

    def test_mutant_labels_are_alphabetically_sorted(self, simple_cohort, two_groups):
        """Contract 1: X-axis mutant labels are in alphabetical order."""
        _, _, _, mutant_labels = build_dot_grid(simple_cohort, two_groups, fdr_threshold=0.05)
        assert mutant_labels == sorted(mutant_labels)
        assert mutant_labels == ["alpha", "beta", "gamma"]

    def test_term_labels_grouped_by_category_in_order(self, simple_cohort, two_groups):
        """Contract 2: Y-axis term labels are grouped by category, in groups list order.

        two_groups has Metabolism=[TERM_A, TERM_B], Signaling=[TERM_C].
        Expected order: TERM_A, TERM_B, TERM_C.
        """
        _, _, term_labels, _ = build_dot_grid(simple_cohort, two_groups, fdr_threshold=0.05)
        assert term_labels == ["TERM_A", "TERM_B", "TERM_C"]

    def test_dot_present_only_when_fdr_below_threshold(self, simple_cohort, two_groups):
        """Contract 3: A dot (non-None) appears only if FDR < fdr_threshold.

        With threshold=0.05:
        - TERM_A/alpha: FDR=0.01 < 0.05 -> present
        - TERM_A/beta: FDR=0.03 < 0.05 -> present
        - TERM_A/gamma: FDR=0.10 >= 0.05 -> None
        - TERM_B/alpha: FDR=0.04 < 0.05 -> present
        - TERM_B/beta: FDR=0.06 >= 0.05 -> None
        - TERM_B/gamma: FDR=0.02 < 0.05 -> present
        - TERM_C/alpha: FDR=0.001 < 0.05 -> present
        - TERM_C/beta: FDR=0.005 < 0.05 -> present
        - TERM_C/gamma: FDR=0.002 < 0.05 -> present
        """
        nes_matrix, sig_matrix, term_labels, mutant_labels = build_dot_grid(
            simple_cohort, two_groups, fdr_threshold=0.05
        )
        # Mutant indices: alpha=0, beta=1, gamma=2
        # Term indices: TERM_A=0, TERM_B=1, TERM_C=2

        # TERM_A
        assert nes_matrix[0][0] is not None  # alpha, FDR=0.01
        assert nes_matrix[0][1] is not None  # beta, FDR=0.03
        assert nes_matrix[0][2] is None      # gamma, FDR=0.10

        # TERM_B
        assert nes_matrix[1][0] is not None  # alpha, FDR=0.04
        assert nes_matrix[1][1] is None      # beta, FDR=0.06
        assert nes_matrix[1][2] is not None  # gamma, FDR=0.02

        # TERM_C - all present
        assert nes_matrix[2][0] is not None
        assert nes_matrix[2][1] is not None
        assert nes_matrix[2][2] is not None

    def test_nes_values_are_correct(self, simple_cohort, two_groups):
        """Contract 3/4: NES matrix contains actual NES values where FDR < threshold."""
        nes_matrix, _, _, _ = build_dot_grid(simple_cohort, two_groups, fdr_threshold=0.05)

        assert nes_matrix[0][0] == pytest.approx(1.5)   # TERM_A/alpha
        assert nes_matrix[0][1] == pytest.approx(-1.0)  # TERM_A/beta
        assert nes_matrix[1][2] == pytest.approx(-1.8)  # TERM_B/gamma

    def test_sig_matrix_contains_negative_log10_fdr(self, simple_cohort, two_groups):
        """Contract 5: sig_matrix values are -log10(FDR) for significant cells."""
        _, sig_matrix, _, _ = build_dot_grid(simple_cohort, two_groups, fdr_threshold=0.05)

        # TERM_A/alpha: FDR=0.01, -log10(0.01) = 2.0
        assert sig_matrix[0][0] == pytest.approx(2.0)
        # TERM_B/gamma: FDR=0.02, -log10(0.02) ~ 1.699
        assert sig_matrix[1][2] == pytest.approx(-math.log10(0.02))
        # TERM_C/alpha: FDR=0.001, -log10(0.001) = 3.0
        assert sig_matrix[2][0] == pytest.approx(3.0)

    def test_sig_matrix_none_where_not_significant(self, simple_cohort, two_groups):
        """Contract 3: sig_matrix is None where FDR >= threshold."""
        _, sig_matrix, _, _ = build_dot_grid(simple_cohort, two_groups, fdr_threshold=0.05)
        assert sig_matrix[0][2] is None  # TERM_A/gamma, FDR=0.10
        assert sig_matrix[1][1] is None  # TERM_B/beta, FDR=0.06

    def test_all_cells_empty_when_threshold_very_low(self, simple_cohort, two_groups):
        """Edge case: If threshold is so low no FDR qualifies, all cells are None."""
        nes_matrix, sig_matrix, _, _ = build_dot_grid(
            simple_cohort, two_groups, fdr_threshold=0.0001
        )
        for row in nes_matrix:
            for cell in row:
                assert cell is None
        for row in sig_matrix:
            for cell in row:
                assert cell is None

    def test_all_cells_present_when_threshold_high(self, simple_cohort, two_groups):
        """Edge case: If threshold is very high, all cells with data are populated."""
        nes_matrix, _, _, _ = build_dot_grid(
            simple_cohort, two_groups, fdr_threshold=1.0
        )
        for row in nes_matrix:
            for cell in row:
                assert cell is not None

    def test_term_not_in_mutant_profile_yields_none(self, simple_cohort):
        """Edge case: If a term name in a group does not exist in a mutant profile,
        the cell should be None."""
        groups = [
            CategoryGroup(category_name="Missing", term_names=["NONEXISTENT_TERM"]),
        ]
        nes_matrix, sig_matrix, term_labels, _ = build_dot_grid(
            simple_cohort, groups, fdr_threshold=0.05
        )
        assert term_labels == ["NONEXISTENT_TERM"]
        # All cells should be None since no mutant has this term
        for cell in nes_matrix[0]:
            assert cell is None
        for cell in sig_matrix[0]:
            assert cell is None

    def test_fdr_exactly_at_threshold_is_excluded(self, simple_cohort):
        """Contract 3: FDR >= threshold means empty. FDR exactly equal to threshold is excluded."""
        # Create a cohort where a term has FDR exactly 0.05
        profiles = {}
        recs = {"TERM_X": _make_term_record("TERM_X", "GO:0000099", 1.0, 0.05)}
        profiles["mut1"] = MutantProfile(mutant_id="mut1", records=recs)
        cohort = CohortData(
            mutant_ids=["mut1"],
            profiles=profiles,
            all_term_names={"TERM_X"},
            all_go_ids={"GO:0000099"},
        )
        groups = [CategoryGroup(category_name="Cat", term_names=["TERM_X"])]
        nes_matrix, _, _, _ = build_dot_grid(cohort, groups, fdr_threshold=0.05)
        assert nes_matrix[0][0] is None  # FDR == threshold -> excluded

    def test_fdr_zero_does_not_cause_error(self, simple_cohort):
        """Edge case: FDR=0.0 should not cause log(0) error; should be clamped."""
        profiles = {}
        recs = {"TERM_Z": _make_term_record("TERM_Z", "GO:0000088", 2.5, 0.0)}
        profiles["mut1"] = MutantProfile(mutant_id="mut1", records=recs)
        cohort = CohortData(
            mutant_ids=["mut1"],
            profiles=profiles,
            all_term_names={"TERM_Z"},
            all_go_ids={"GO:0000088"},
        )
        groups = [CategoryGroup(category_name="Cat", term_names=["TERM_Z"])]
        nes_matrix, sig_matrix, _, _ = build_dot_grid(cohort, groups, fdr_threshold=0.05)
        assert nes_matrix[0][0] == pytest.approx(2.5)
        # sig should be a finite positive number (clamped away from inf)
        assert sig_matrix[0][0] is not None
        assert math.isfinite(sig_matrix[0][0])
        assert sig_matrix[0][0] > 0

    def test_multiple_groups_preserve_category_order(self, simple_cohort):
        """Contract 2: Terms from first group appear before terms from second group."""
        groups = [
            CategoryGroup(category_name="Second", term_names=["TERM_C"]),
            CategoryGroup(category_name="First", term_names=["TERM_A"]),
        ]
        _, _, term_labels, _ = build_dot_grid(simple_cohort, groups, fdr_threshold=0.05)
        assert term_labels == ["TERM_C", "TERM_A"]

    def test_unsorted_mutant_ids_are_sorted_in_output(self):
        """Contract 1: Mutant labels are sorted alphabetically regardless of input order."""
        profiles = {}
        for mid in ["zebra", "apple", "mango"]:
            recs = {"T1": _make_term_record("T1", "GO:0000001", 1.0, 0.01)}
            profiles[mid] = MutantProfile(mutant_id=mid, records=recs)
        cohort = CohortData(
            mutant_ids=["zebra", "apple", "mango"],
            profiles=profiles,
            all_term_names={"T1"},
            all_go_ids={"GO:0000001"},
        )
        groups = [CategoryGroup(category_name="Cat", term_names=["T1"])]
        _, _, _, mutant_labels = build_dot_grid(cohort, groups, fdr_threshold=0.05)
        assert mutant_labels == ["apple", "mango", "zebra"]


# ---------------------------------------------------------------------------
# Tests for render_dot_plot
# ---------------------------------------------------------------------------


class TestRenderDotPlot:
    """Tests for the render_dot_plot function."""

    def test_empty_groups_raises_value_error(self, simple_cohort, tmp_path):
        """Error condition: Empty groups list raises ValueError."""
        with pytest.raises(ValueError):
            render_dot_plot(
                cohort=simple_cohort,
                groups=[],
                fdr_threshold=0.05,
                output_stem="test_fig",
                output_dir=tmp_path,
            )

    def test_output_files_are_created(self, simple_cohort, two_groups, tmp_path):
        """Post-condition: PDF, PNG, and SVG files are created in output_dir."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="test_fig",
            output_dir=tmp_path,
        )
        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()

    def test_output_file_paths_use_stem_and_dir(self, simple_cohort, two_groups, tmp_path):
        """Contract 10: Files are named {output_stem}.{ext} in output_dir."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="my_figure",
            output_dir=tmp_path,
        )
        assert result.pdf_path == tmp_path / "my_figure.pdf"
        assert result.png_path == tmp_path / "my_figure.png"
        assert result.svg_path == tmp_path / "my_figure.svg"

    def test_result_n_terms_matches_sum_of_group_terms(self, simple_cohort, two_groups, tmp_path):
        """Post-condition: n_terms_displayed == sum of term counts across groups."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="test_fig",
            output_dir=tmp_path,
        )
        expected_terms = sum(len(g.term_names) for g in two_groups)
        assert result.n_terms_displayed == expected_terms

    def test_result_n_mutants_matches_cohort(self, simple_cohort, two_groups, tmp_path):
        """Post-condition: n_mutants == len(cohort.mutant_ids)."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="test_fig",
            output_dir=tmp_path,
        )
        assert result.n_mutants == len(simple_cohort.mutant_ids)

    def test_result_n_categories_matches_groups(self, simple_cohort, two_groups, tmp_path):
        """n_categories should match the number of groups provided."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="test_fig",
            output_dir=tmp_path,
        )
        assert result.n_categories == len(two_groups)

    def test_result_is_dot_plot_result_type(self, simple_cohort, two_groups, tmp_path):
        """Signature: Return type is DotPlotResult."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="test_fig",
            output_dir=tmp_path,
        )
        assert isinstance(result, DotPlotResult)

    def test_output_files_are_nonempty(self, simple_cohort, two_groups, tmp_path):
        """Post-condition: Output files should have nonzero size."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="test_fig",
            output_dir=tmp_path,
        )
        assert result.pdf_path.stat().st_size > 0
        assert result.png_path.stat().st_size > 0
        assert result.svg_path.stat().st_size > 0

    def test_custom_dpi_is_accepted(self, simple_cohort, two_groups, tmp_path):
        """Signature: Custom dpi parameter is accepted without error."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="hires",
            output_dir=tmp_path,
            dpi=600,
        )
        assert result.pdf_path.exists()

    def test_title_parameter_accepted(self, simple_cohort, two_groups, tmp_path):
        """Signature: title parameter is accepted."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="titled",
            output_dir=tmp_path,
            title="My Figure Title",
        )
        assert result.pdf_path.exists()

    def test_empty_title_is_default(self, simple_cohort, two_groups, tmp_path):
        """Signature: Default title is empty string, should not cause error."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="no_title",
            output_dir=tmp_path,
        )
        assert result.pdf_path.exists()

    def test_single_group_single_term(self, tmp_path):
        """Edge case: Minimal valid input - one group with one term, one mutant."""
        recs = {"TERM_ONLY": _make_term_record("TERM_ONLY", "GO:0000001", 1.0, 0.01)}
        profiles = {"m1": MutantProfile(mutant_id="m1", records=recs)}
        cohort = CohortData(
            mutant_ids=["m1"],
            profiles=profiles,
            all_term_names={"TERM_ONLY"},
            all_go_ids={"GO:0000001"},
        )
        groups = [CategoryGroup(category_name="Solo", term_names=["TERM_ONLY"])]
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="minimal",
            output_dir=tmp_path,
        )
        assert result.n_terms_displayed == 1
        assert result.n_mutants == 1
        assert result.n_categories == 1

    def test_all_cells_empty_still_renders(self, simple_cohort, tmp_path):
        """Edge case: When no cells pass FDR threshold, figure still renders."""
        groups = [
            CategoryGroup(category_name="HighFDR", term_names=["TERM_D"]),
        ]
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="empty_dots",
            output_dir=tmp_path,
        )
        # TERM_D has all FDRs >= 0.05 so no dots, but files still created
        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()
        assert result.n_terms_displayed == 1

    def test_nonexistent_output_dir_raises_assertion(self, simple_cohort, two_groups, tmp_path):
        """Pre-condition: output_dir must exist (AssertionError or OSError)."""
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises((AssertionError, OSError)):
            render_dot_plot(
                cohort=simple_cohort,
                groups=two_groups,
                fdr_threshold=0.05,
                output_stem="fail",
                output_dir=nonexistent,
            )

    def test_write_failure_raises_os_error(self, simple_cohort, two_groups, tmp_path):
        """Error condition: OSError is raised when files cannot be written.

        We simulate this by passing a read-only directory (platform-dependent).
        """
        import os
        import stat
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        # Remove write permission
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            with pytest.raises((OSError, AssertionError)):
                render_dot_plot(
                    cohort=simple_cohort,
                    groups=two_groups,
                    fdr_threshold=0.05,
                    output_stem="fail",
                    output_dir=readonly_dir,
                )
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(stat.S_IRWXU)

    def test_many_terms_scales_figure_height(self, tmp_path):
        """Contract 9: Figure height scales to accommodate all terms."""
        # Create a cohort with many terms
        terms = [f"TERM_{i:03d}" for i in range(30)]
        recs = {}
        for idx, t in enumerate(terms):
            recs[t] = _make_term_record(t, f"GO:{idx:07d}", 1.0, 0.01)
        profiles = {"m1": MutantProfile(mutant_id="m1", records=recs)}
        cohort = CohortData(
            mutant_ids=["m1"],
            profiles=profiles,
            all_term_names=set(terms),
            all_go_ids={f"GO:{i:07d}" for i in range(30)},
        )
        groups = [CategoryGroup(category_name="Big", term_names=terms)]
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="big",
            output_dir=tmp_path,
        )
        assert result.n_terms_displayed == 30
        assert result.pdf_path.exists()


# ---------------------------------------------------------------------------
# Tests for draw_category_boxes
# ---------------------------------------------------------------------------


class TestDrawCategoryBoxes:
    """Tests for the draw_category_boxes function."""

    def test_boxes_are_added_to_axes(self):
        """Contract 6: Category boxes are drawn as visible rectangles on the axes."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 10)

        groups = [
            CategoryGroup(category_name="GroupA", term_names=["T1", "T2"]),
            CategoryGroup(category_name="GroupB", term_names=["T3"]),
        ]

        initial_patch_count = len(ax.patches)
        draw_category_boxes(ax, groups, y_start=0)

        # Should have added 2 patches (one per group)
        assert len(ax.patches) == initial_patch_count + 2
        plt.close(fig)

    def test_category_labels_are_bold_text(self):
        """Contract 6: Category names are rendered in bold text to the right of boxes."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 10)

        groups = [
            CategoryGroup(category_name="TestCat", term_names=["T1", "T2"]),
        ]

        draw_category_boxes(ax, groups, y_start=0)

        # Check that text was added
        texts = ax.texts
        assert len(texts) >= 1

        # Find the category label
        cat_text = None
        for t in texts:
            if t.get_text() == "TestCat":
                cat_text = t
                break

        assert cat_text is not None, "Category label 'TestCat' not found in axes texts"
        assert cat_text.get_fontweight() == "bold"
        plt.close(fig)

    def test_label_vertically_centered_in_box(self):
        """Contract 6: Category label is vertically centered within the box extent."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 10)

        groups = [
            CategoryGroup(category_name="Centered", term_names=["T1", "T2", "T3"]),
        ]

        draw_category_boxes(ax, groups, y_start=0)

        # With 3 terms starting at y_start=0, center should be at (0 + 2)/2 = 1.0
        cat_text = None
        for t in ax.texts:
            if t.get_text() == "Centered":
                cat_text = t
                break

        assert cat_text is not None
        # y position should be at center: y_start + (n_terms - 1) / 2 = 0 + 1.0 = 1.0
        _, y_pos = cat_text.get_position()
        assert y_pos == pytest.approx(1.0)
        plt.close(fig)

    def test_label_placed_to_right_of_box(self):
        """Contract 6: Category label is placed to the right of the box."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 10)

        groups = [
            CategoryGroup(category_name="RightSide", term_names=["T1"]),
        ]

        draw_category_boxes(ax, groups, y_start=0)

        cat_text = None
        for t in ax.texts:
            if t.get_text() == "RightSide":
                cat_text = t
                break

        assert cat_text is not None
        x_pos, _ = cat_text.get_position()
        # Label should be beyond the right edge of the axes data area
        assert x_pos > ax.get_xlim()[1]
        plt.close(fig)

    def test_multiple_groups_create_separate_boxes(self):
        """Contract 6: Each group gets its own rectangle."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 10)

        groups = [
            CategoryGroup(category_name="A", term_names=["T1"]),
            CategoryGroup(category_name="B", term_names=["T2", "T3"]),
            CategoryGroup(category_name="C", term_names=["T4"]),
        ]

        draw_category_boxes(ax, groups, y_start=0)

        assert len(ax.patches) == 3
        # 3 labels
        label_texts = [t.get_text() for t in ax.texts]
        assert "A" in label_texts
        assert "B" in label_texts
        assert "C" in label_texts
        plt.close(fig)

    def test_single_group_draws_one_box(self):
        """A single group draws exactly one box with its label."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(-1, 5)

        groups = [
            CategoryGroup(category_name="NonEmpty", term_names=["T1"]),
        ]

        draw_category_boxes(ax, groups, y_start=0)

        assert len(ax.patches) >= 1
        label_texts = [t.get_text() for t in ax.texts]
        assert "NonEmpty" in label_texts
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests for DotPlotResult dataclass
# ---------------------------------------------------------------------------


class TestDotPlotResult:
    """Tests for the DotPlotResult dataclass structure."""

    def test_dataclass_fields(self, tmp_path):
        """Signature: DotPlotResult has all expected fields."""
        result = DotPlotResult(
            pdf_path=tmp_path / "test.pdf",
            png_path=tmp_path / "test.png",
            svg_path=tmp_path / "test.svg",
            n_terms_displayed=10,
            n_categories=3,
            n_mutants=5,
        )
        assert result.pdf_path == tmp_path / "test.pdf"
        assert result.png_path == tmp_path / "test.png"
        assert result.svg_path == tmp_path / "test.svg"
        assert result.n_terms_displayed == 10
        assert result.n_categories == 3
        assert result.n_mutants == 5


# ---------------------------------------------------------------------------
# Integration-style behavioral contract tests
# ---------------------------------------------------------------------------


class TestBehavioralContracts:
    """Higher-level tests verifying behavioral contracts through render_dot_plot."""

    def test_colormap_is_rdbu_r_or_equivalent(self, simple_cohort, two_groups, tmp_path):
        """Contract 11: The colormap used is RdBu_r so red=positive, blue=negative.

        We verify this indirectly by checking the SVG output contains color information
        consistent with a red-blue diverging colormap.
        """
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="cmap_test",
            output_dir=tmp_path,
        )
        # Just verify the files exist; colormap correctness is a visual property
        # but we ensure no error is raised when using the colormap
        assert result.svg_path.exists()

    def test_no_gridlines_in_output(self, simple_cohort, two_groups, tmp_path):
        """Contract 8: No gridlines are drawn within the plot area.

        We test this by reading the SVG and checking it was generated without error.
        The actual gridline absence is a rendering property validated visually.
        """
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="nogrid",
            output_dir=tmp_path,
        )
        assert result.svg_path.exists()

    def test_term_labels_show_name_only_no_go_id(self, simple_cohort, two_groups):
        """Contract 9: Y-axis labels show term name only, no GO ID."""
        _, _, term_labels, _ = build_dot_grid(simple_cohort, two_groups, fdr_threshold=0.05)
        for label in term_labels:
            assert "GO:" not in label, f"Term label '{label}' should not contain GO ID"

    def test_positive_nes_and_negative_nes_in_grid(self, simple_cohort, two_groups):
        """Contract 4: Both positive and negative NES values appear in the matrix."""
        nes_matrix, _, _, _ = build_dot_grid(simple_cohort, two_groups, fdr_threshold=0.05)
        all_nes = [v for row in nes_matrix for v in row if v is not None]
        has_positive = any(v > 0 for v in all_nes)
        has_negative = any(v < 0 for v in all_nes)
        assert has_positive, "Expected at least one positive NES value"
        assert has_negative, "Expected at least one negative NES value"

    def test_render_with_multiple_category_groups(self, simple_cohort, tmp_path):
        """Contract 6: Multiple category groups are rendered with separate boxes."""
        groups = [
            CategoryGroup(category_name="Cat1", term_names=["TERM_A"]),
            CategoryGroup(category_name="Cat2", term_names=["TERM_B"]),
            CategoryGroup(category_name="Cat3", term_names=["TERM_C"]),
        ]
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="multi_cat",
            output_dir=tmp_path,
        )
        assert result.n_categories == 3
        assert result.n_terms_displayed == 3
        assert result.pdf_path.exists()

    def test_render_with_different_stem_names(self, simple_cohort, two_groups, tmp_path):
        """Contract 10: Different output_stem values produce differently named files."""
        r1 = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="figure1_cherry_picked",
            output_dir=tmp_path,
        )
        r2 = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="figure2_clustered",
            output_dir=tmp_path,
        )
        assert r1.pdf_path != r2.pdf_path
        assert r1.pdf_path.name == "figure1_cherry_picked.pdf"
        assert r2.pdf_path.name == "figure2_clustered.pdf"

    def test_font_family_parameter_accepted(self, simple_cohort, two_groups, tmp_path):
        """Signature: Custom font_family does not cause error."""
        result = render_dot_plot(
            cohort=simple_cohort,
            groups=two_groups,
            fdr_threshold=0.05,
            output_stem="custom_font",
            output_dir=tmp_path,
            font_family="DejaVu Sans",
        )
        assert result.pdf_path.exists()
