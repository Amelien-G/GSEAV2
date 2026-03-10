# Auto-generated stub — do not edit
from pathlib import Path
from dataclasses import dataclass
import matplotlib.figure
import matplotlib.axes
from gsea_tool.data_ingestion import CohortData
from gsea_tool.cherry_picked import CategoryGroup

@dataclass
class DotPlotResult:
    """Metadata about a rendered dot plot figure, for notes.md consumption."""
    pdf_path: Path
    png_path: Path
    svg_path: Path
    n_terms_displayed: int
    n_categories: int
    n_mutants: int
    ...

def render_dot_plot(cohort: CohortData, groups: list[CategoryGroup], fdr_threshold: float, output_stem: str, output_dir: Path, dpi: int=300, font_family: str='Arial', title: str='') -> DotPlotResult:
    """Render a grouped dot plot figure and save to PDF, PNG, and SVG.

    Args:
        cohort: The full cohort enrichment data.
        groups: Ordered list of category groups defining Y-axis layout.
        fdr_threshold: FDR threshold for dot presence (cells with FDR >= threshold are empty).
        output_stem: Base filename without extension (e.g., "figure1_cherry_picked").
        output_dir: Directory to write output files.
        dpi: Resolution for PNG output.
        font_family: Font family for all text.
        title: Optional figure title.

    Returns:
        DotPlotResult with paths and summary counts.
    """
    ...

def build_dot_grid(cohort: CohortData, groups: list[CategoryGroup], fdr_threshold: float) -> tuple[list[list[float | None]], list[list[float | None]], list[str], list[str]]:
    """Build the NES and significance matrices for the dot grid.

    Returns:
        nes_matrix: 2D list [term_index][mutant_index], None for empty cells.
        sig_matrix: 2D list [term_index][mutant_index] of -log10(FDR), None for empty cells.
        term_labels: Ordered Y-axis labels (term names, grouped by category).
        mutant_labels: Ordered X-axis labels (mutant IDs, alphabetical).
    """
    ...

def draw_category_boxes(ax: matplotlib.axes.Axes, groups: list[CategoryGroup], y_start: float) -> None:
    """Draw category grouping rectangles and bold right-side labels on the axes.

    Each box encloses the rows belonging to one category group. The category name
    is rendered in bold, vertically centered to the right of the box.
    """
    ...
assert len(groups) > 0, 'At least one category group is required'
assert all((len(g.term_names) > 0 for g in groups)), 'No empty groups passed to renderer'
assert output_dir.is_dir(), 'Output directory must exist'
assert result.pdf_path.exists(), 'PDF file must be written'
assert result.png_path.exists(), 'PNG file must be written'
assert result.svg_path.exists(), 'SVG file must be written'
assert result.n_terms_displayed == sum((len(g.term_names) for g in groups)), 'Term count must match input'
assert result.n_mutants == len(cohort.mutant_ids), 'Mutant count must match cohort'
