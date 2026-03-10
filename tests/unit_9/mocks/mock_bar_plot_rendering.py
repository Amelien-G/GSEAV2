# Auto-generated stub — do not edit
from pathlib import Path
from dataclasses import dataclass
from gsea_tool.configuration import FisherConfig, PlotAppearanceConfig
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.go_clustering import ClusteringResult

@dataclass
class BarPlotResult:
    """Metadata about the rendered bar plot figure, for notes.md consumption."""
    pdf_path: Path
    png_path: Path
    svg_path: Path
    n_bars: int
    n_mutants: int
    clustering_was_used: bool
    ...

def render_bar_plot(fisher_result: FisherResult, clustering_result: ClusteringResult | None, fisher_config: FisherConfig, plot_config: PlotAppearanceConfig, output_dir: Path, output_stem: str='figure3_meta_analysis') -> BarPlotResult:
    """Render the meta-analysis bar plot and save to PDF, PNG, and SVG.

    If clustering_result is provided, uses representative terms.
    If clustering_result is None, uses top N terms by combined p-value.

    Returns BarPlotResult with paths and summary counts.
    """
    ...

def select_bar_data(fisher_result: FisherResult, clustering_result: ClusteringResult | None, top_n: int) -> tuple[list[str], list[float], list[int]]:
    """Select GO terms, p-values, and contributing counts for the bar plot.

    Returns:
        term_names: Display names for Y-axis labels.
        neg_log_pvalues: -log10(combined p-value) for X-axis.
        n_contributing: Number of contributing mutant lines for color encoding.
    All lists are ordered by combined p-value (most significant first).
    """
    ...
assert output_dir.is_dir(), 'Output directory must exist'
assert fisher_config.top_n_bars > 0, 'top_n_bars must be positive'
assert result.pdf_path.exists(), 'PDF file must be written'
assert result.png_path.exists(), 'PNG file must be written'
assert result.svg_path.exists(), 'SVG file must be written'
assert result.n_bars > 0, 'At least one bar must be plotted'
assert result.n_bars <= fisher_config.top_n_bars, 'Number of bars cannot exceed top_n_bars'
