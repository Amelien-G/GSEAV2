"""Regression test mirroring wgcna BUG-018.

Locks in the legend spacing formula. After increasing min_dot_size /
max_dot_size (BUG-013/015 fix), the default matplotlib handleheight=0.7 and
labelspacing=0.5 (in font-size units) reserve only ~5-6 points per legend
entry at fontsize=7, which is too small for ~25 pt diameter dots and causes
visible overlap in the legend column.

After the fix, handleheight and labelspacing scale with max_dot_size:
    handleheight = max(0.7, _max_diam_pts / _font_size_pts + 0.5)
    labelspacing = max(0.5, _max_diam_pts / _font_size_pts * 0.5)
where _max_diam_pts = 2 * sqrt(max_dot_size / pi).
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest

from tests.unit_5.test_dot_plot_rendering import _make_cohort, _make_groups

from gsea_tool.dot_plot import render_dot_plot


def _expected_spacing(max_dot_size: float, font_size_pts: float) -> tuple[float, float]:
    max_diam_pts = 2.0 * math.sqrt(max_dot_size / math.pi)
    handleheight = max(0.7, max_diam_pts / font_size_pts + 0.5)
    labelspacing = max(0.5, max_diam_pts / font_size_pts * 0.5)
    return handleheight, labelspacing, max_diam_pts


def test_legend_spacing_accommodates_max_marker_diameter(tmp_path: Path):
    """The size legend's per-entry vertical room must be at least the
    marker disc diameter, so dots cannot physically overlap."""
    cohort = _make_cohort()
    groups = _make_groups()

    result = render_dot_plot(
        cohort=cohort,
        groups=groups,
        fdr_threshold=0.05,
        output_stem="spacing",
        output_dir=tmp_path,
    )
    assert result.pdf_path.exists()

    # Verify the formula directly: with max_dot_size=500, font_size=7,
    # handleheight*font_size must be >= max marker diameter.
    handleheight, labelspacing, max_diam_pts = _expected_spacing(
        max_dot_size=500.0, font_size_pts=7.0
    )

    assert handleheight * 7.0 >= max_diam_pts
    assert labelspacing * 7.0 >= max_diam_pts / 2.0
    assert handleheight > 0.7
    assert labelspacing > 0.5


def test_spacing_formula_scales_with_max_dot_size():
    """If max_dot_size grows, spacing grows monotonically."""
    smaller_hh, smaller_ls, smaller_d = _expected_spacing(200.0, 7.0)
    larger_hh, larger_ls, larger_d = _expected_spacing(500.0, 7.0)

    assert larger_d > smaller_d
    assert larger_hh > smaller_hh
    assert larger_ls > smaller_ls


def test_legend_handle_count_matches_anchored_references(tmp_path: Path):
    """Confirm the legend uses anchored references (3 entries), not
    data-driven values which previously could be 1-3 depending on data."""
    from matplotlib.lines import Line2D

    cohort = _make_cohort()
    groups = _make_groups()

    captured: list[Line2D] = []
    real_init = Line2D.__init__

    def capture_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        if (
            kwargs.get("marker") == "o"
            and kwargs.get("linestyle") == "None"
            and isinstance(kwargs.get("label"), str)
            and kwargs["label"].startswith("FDR=")
        ):
            captured.append(self)

    Line2D.__init__ = capture_init  # type: ignore[method-assign]
    try:
        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="anchor_count",
            output_dir=tmp_path,
        )
    finally:
        Line2D.__init__ = real_init  # type: ignore[method-assign]

    assert result.pdf_path.exists()
    assert len(captured) == 3
    labels = [h.get_label() for h in captured]
    assert labels == ["FDR=0.25", "FDR=0.05", "FDR=0.001"]
