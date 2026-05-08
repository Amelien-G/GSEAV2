"""Unit 5 -- Dot Plot Rendering."""

from pathlib import Path
from dataclasses import dataclass
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from gsea_tool.data_ingestion import CohortData
from gsea_tool.cherry_picked import CategoryGroup


# Anchored size-scale references for dot-size legend.
# See bug log entries cross-referencing wgcna BUG-013 / BUG-015 / BUG-018.
LEGEND_FDR_EXAMPLES = (0.25, 0.05, 0.001)   # smallest -> largest reference dot
FDR_MIN_CLAMP = 1e-300


@dataclass
class DotPlotResult:
    """Metadata about a rendered dot plot figure, for notes.md consumption."""
    pdf_path: Path
    png_path: Path
    svg_path: Path
    n_terms_displayed: int
    n_categories: int
    n_mutants: int


def build_dot_grid(
    cohort: CohortData,
    groups: list[CategoryGroup],
    fdr_threshold: float,
) -> tuple[list[list[float | None]], list[list[float | None]], list[str], list[str]]:
    """Build the NES and significance matrices for the dot grid.

    Returns:
        nes_matrix: 2D list [term_index][mutant_index], None for empty cells.
        sig_matrix: 2D list [term_index][mutant_index] of -log10(FDR), None for empty cells.
        term_labels: Ordered Y-axis labels (term names, grouped by category).
        mutant_labels: Ordered X-axis labels (mutant IDs, alphabetical).
    """
    mutant_labels = sorted(cohort.mutant_ids)
    term_labels: list[str] = []
    for group in groups:
        for term_name in group.term_names:
            term_labels.append(term_name)

    nes_matrix: list[list[float | None]] = []
    sig_matrix: list[list[float | None]] = []

    for term_name in term_labels:
        nes_row: list[float | None] = []
        sig_row: list[float | None] = []
        for mutant_id in mutant_labels:
            profile = cohort.profiles.get(mutant_id)
            if profile is None:
                nes_row.append(None)
                sig_row.append(None)
                continue
            record = profile.records.get(term_name)
            if record is None or record.fdr >= fdr_threshold:
                nes_row.append(None)
                sig_row.append(None)
            else:
                nes_row.append(record.nes)
                # -log10(FDR), handle FDR=0 by clamping
                fdr_val = record.fdr
                if fdr_val <= 0:
                    fdr_val = 1e-300
                sig_row.append(-math.log10(fdr_val))
        nes_matrix.append(nes_row)
        sig_matrix.append(sig_row)

    return nes_matrix, sig_matrix, term_labels, mutant_labels


def draw_category_boxes(
    ax: matplotlib.axes.Axes,
    groups: list[CategoryGroup],
    y_start: float,
) -> None:
    """Draw category grouping rectangles and bold right-side labels on the axes.

    Each box encloses the rows belonging to one category group. The category name
    is rendered in bold, vertically centered to the right of the box.
    """
    current_y = y_start
    x_left = ax.get_xlim()[0]
    x_right = ax.get_xlim()[1]
    box_width = x_right - x_left

    for group in groups:
        n_terms = len(group.term_names)
        # Box from current_y - 0.5 to current_y + n_terms - 0.5
        box_bottom = current_y - 0.5
        box_height = n_terms

        rect = Rectangle(
            (x_left, box_bottom),
            box_width,
            box_height,
            linewidth=1.0,
            edgecolor="black",
            facecolor="none",
            clip_on=False,
            zorder=3,
        )
        ax.add_patch(rect)

        # Category label to the right of the box, vertically centered
        label_x = x_right + 0.3
        label_y = current_y + (n_terms - 1) / 2.0
        ax.text(
            label_x,
            label_y,
            group.category_name,
            fontweight="bold",
            fontsize=9,
            va="center",
            ha="left",
            clip_on=False,
        )

        current_y += n_terms


def render_dot_plot(
    cohort: CohortData,
    groups: list[CategoryGroup],
    fdr_threshold: float,
    output_stem: str,
    output_dir: Path,
    dpi: int = 300,
    font_family: str = "Arial",
    title: str = "",
) -> DotPlotResult:
    """Render a grouped dot plot figure and save to PDF, PNG, and SVG."""
    # Validate inputs
    if len(groups) == 0:
        raise ValueError("Empty groups list")
    if not all(len(g.term_names) > 0 for g in groups):
        raise ValueError("No empty groups passed to renderer")
    if not output_dir.is_dir():
        raise OSError(f"Output directory does not exist: {output_dir}")

    # Build data grid
    nes_matrix, sig_matrix, term_labels, mutant_labels = build_dot_grid(
        cohort, groups, fdr_threshold
    )

    n_terms = len(term_labels)
    n_mutants = len(mutant_labels)

    # Set font
    plt.rcParams["font.family"] = font_family

    # Figure sizing: scale height by number of terms
    fig_width = max(6, n_mutants * 0.7 + 4)
    fig_height = max(4, n_terms * 0.35 + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Collect valid data points for scatter
    xs = []
    ys = []
    colors = []
    sizes = []

    for i in range(n_terms):
        for j in range(n_mutants):
            nes_val = nes_matrix[i][j]
            sig_val = sig_matrix[i][j]
            if nes_val is not None and sig_val is not None:
                xs.append(j)
                ys.append(i)
                colors.append(nes_val)
                sizes.append(sig_val)

    # Determine color normalization: symmetric around zero
    if colors:
        max_abs_nes = max(abs(c) for c in colors)
        if max_abs_nes == 0:
            max_abs_nes = 1.0
    else:
        max_abs_nes = 1.0

    norm = mcolors.Normalize(vmin=-max_abs_nes, vmax=max_abs_nes)
    cmap = plt.get_cmap("RdBu_r")

    # Anchored size scaling: dot size is mapped to fixed FDR reference values
    # (LEGEND_FDR_EXAMPLES), not to the per-figure data range. This keeps the
    # three legend reference dots visually distinct even when the data contains
    # FDR=0 values clamped to FDR_MIN_CLAMP=1e-300.
    min_dot_size = 50
    max_dot_size = 500

    sig_anchor_min = -math.log10(LEGEND_FDR_EXAMPLES[0])
    sig_anchor_max = -math.log10(max(LEGEND_FDR_EXAMPLES[-1], FDR_MIN_CLAMP))
    sig_anchor_range = sig_anchor_max - sig_anchor_min or 1.0

    def scale_size(sig_val: float) -> float:
        """Anchored: -log10(FDR) -> dot AREA (points^2)."""
        norm = (sig_val - sig_anchor_min) / sig_anchor_range
        norm = max(0.0, min(1.0, norm))
        return min_dot_size + norm * (max_dot_size - min_dot_size)

    scaled_sizes = [scale_size(s) for s in sizes]

    # Scatter plot
    if xs:
        sc = ax.scatter(
            xs, ys,
            c=colors,
            s=scaled_sizes,
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            zorder=5,
        )
    else:
        # Create invisible scatter for colorbar
        sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm)

    # Axis configuration
    ax.set_xticks(range(n_mutants))
    ax.set_xticklabels(mutant_labels, rotation=90, fontsize=8, ha="center")
    ax.set_yticks(range(n_terms))
    ax.set_yticklabels(term_labels, fontsize=7)

    ax.set_xlim(-0.5, n_mutants - 0.5)
    ax.set_ylim(-0.5, n_terms - 0.5)

    # Invert y-axis so first term is at top
    ax.invert_yaxis()

    # Remove gridlines
    ax.grid(False)

    # Clean aesthetic: remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    # Title
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Draw category boxes
    draw_category_boxes(ax, groups, y_start=0)

    # Colorbar for NES
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=15, pad=0.15)
    cbar.set_label("NES", fontsize=9)

    # Size legend for -log10(FDR): three reference dots anchored to
    # LEGEND_FDR_EXAMPLES. Line2D.markersize is a diameter in points;
    # scatter `s` is an area in points^2. Convert via 2*sqrt(s/pi).
    if sizes:
        legend_handles = []
        for fdr in LEGEND_FDR_EXAMPLES:
            sig_val = -math.log10(max(fdr, FDR_MIN_CLAMP))
            s = scale_size(sig_val)
            label_text = f"FDR={fdr}"
            handle = Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=2.0 * math.sqrt(s / math.pi),
                label=label_text,
                linestyle="None",
            )
            legend_handles.append(handle)

        # Scale legend spacing to the largest configured marker so dots do not
        # visually overlap when min_dot_size / max_dot_size are increased.
        _max_diam_pts = 2.0 * math.sqrt(max_dot_size / math.pi)
        _font_size_pts = 7.0
        size_legend = ax.legend(
            handles=legend_handles,
            title="-log$_{10}$(FDR)",
            loc="lower right",
            bbox_to_anchor=(1.35, 0),
            frameon=False,
            fontsize=7,
            title_fontsize=8,
            handleheight=max(0.7, _max_diam_pts / _font_size_pts + 0.5),
            labelspacing=max(0.5, _max_diam_pts / _font_size_pts * 0.5),
            borderpad=1.0,
        )
        ax.add_artist(size_legend)

    # No background color
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plt.tight_layout()

    # Save files
    pdf_path = output_dir / f"{output_stem}.pdf"
    png_path = output_dir / f"{output_stem}.png"
    svg_path = output_dir / f"{output_stem}.svg"

    try:
        fig.savefig(str(pdf_path), format="pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(str(png_path), format="png", dpi=dpi, bbox_inches="tight")
        fig.savefig(str(svg_path), format="svg", dpi=dpi, bbox_inches="tight")
    except Exception as e:
        raise OSError(f"Failed to write output files: {e}") from e
    finally:
        plt.close(fig)

    n_terms_displayed = sum(len(g.term_names) for g in groups)
    n_categories = len(groups)

    result = DotPlotResult(
        pdf_path=pdf_path,
        png_path=png_path,
        svg_path=svg_path,
        n_terms_displayed=n_terms_displayed,
        n_categories=n_categories,
        n_mutants=n_mutants,
    )

    return result
