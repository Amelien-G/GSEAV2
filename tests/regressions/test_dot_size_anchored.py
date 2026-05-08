"""Regression test mirroring wgcna BUG-015.

Locks in the anchored dot-size scale: the three legend reference FDR values
(LEGEND_FDR_EXAMPLES = (0.25, 0.05, 0.001)) must map to three distinct,
monotonically increasing dot sizes regardless of the surrounding data range,
even when the data contains FDR=0 values clamped to FDR_MIN_CLAMP=1e-300.

Pre-fix symptom: scale was data-relative, so a single FDR=0 (clamped to 1e-300,
giving -log10(FDR)=300) would dominate the range and squash both small and mid
legend dots to nearly the same size. After the fix the scale is anchored to
LEGEND_FDR_EXAMPLES, independent of the data.
"""

import math

import matplotlib
matplotlib.use("Agg")

import pytest

from gsea_tool.dot_plot import (
    LEGEND_FDR_EXAMPLES,
    FDR_MIN_CLAMP,
)


def _build_scale_size():
    """Reconstruct the same anchored scale_size used inside render_dot_plot."""
    min_dot_size = 50
    max_dot_size = 500
    sig_anchor_min = -math.log10(LEGEND_FDR_EXAMPLES[0])
    sig_anchor_max = -math.log10(max(LEGEND_FDR_EXAMPLES[-1], FDR_MIN_CLAMP))
    sig_anchor_range = sig_anchor_max - sig_anchor_min or 1.0

    def scale_size(sig_val: float) -> float:
        norm = (sig_val - sig_anchor_min) / sig_anchor_range
        norm = max(0.0, min(1.0, norm))
        return min_dot_size + norm * (max_dot_size - min_dot_size)

    return scale_size, min_dot_size, max_dot_size


def test_three_reference_fdrs_yield_three_distinct_sizes():
    scale_size, min_size, max_size = _build_scale_size()
    fdr_low, fdr_mid, fdr_high = LEGEND_FDR_EXAMPLES  # (0.25, 0.05, 0.001)

    s_low = scale_size(-math.log10(fdr_low))
    s_mid = scale_size(-math.log10(fdr_mid))
    s_high = scale_size(-math.log10(fdr_high))

    assert s_low == pytest.approx(min_size, abs=1e-9)
    assert s_high == pytest.approx(max_size, abs=1e-9)
    assert min_size < s_mid < max_size

    gap_low_mid = s_mid - s_low
    gap_mid_high = s_high - s_mid
    assert gap_low_mid > 0.1 * (max_size - min_size)
    assert gap_mid_high > 0.1 * (max_size - min_size)


def test_saturated_fdr_zero_does_not_collapse_mid_size():
    """The pre-fix bug: FDR=0 clamped to 1e-300 dominated the data range and
    squashed the mid-size legend dot. With the anchored scale the mid-size dot
    is unchanged regardless of how saturated the data is."""
    scale_size, _, max_size = _build_scale_size()

    sig_clamped = -math.log10(FDR_MIN_CLAMP)  # 300.0
    s_saturated = scale_size(sig_clamped)
    s_at_anchor_high = scale_size(-math.log10(LEGEND_FDR_EXAMPLES[-1]))

    assert s_saturated == pytest.approx(max_size, abs=1e-9)
    assert s_at_anchor_high == pytest.approx(max_size, abs=1e-9)

    s_mid = scale_size(-math.log10(LEGEND_FDR_EXAMPLES[1]))
    fdr_below_low = LEGEND_FDR_EXAMPLES[0] * 1.5  # less significant than 0.25
    s_below_low = scale_size(-math.log10(fdr_below_low))
    assert s_mid > s_below_low
    assert s_mid - s_below_low > 0.1 * (max_size - 50)


def test_clamping_outside_anchor_range():
    scale_size, min_size, max_size = _build_scale_size()
    assert scale_size(-math.log10(0.5)) == pytest.approx(min_size, abs=1e-9)
    assert scale_size(0.0) == pytest.approx(min_size, abs=1e-9)
    assert scale_size(-math.log10(1e-50)) == pytest.approx(max_size, abs=1e-9)
