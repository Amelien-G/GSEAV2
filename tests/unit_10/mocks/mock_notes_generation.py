# Auto-generated stub — do not edit
from pathlib import Path
from dataclasses import dataclass
from gsea_tool.data_ingestion import CohortData
from gsea_tool.configuration import ToolConfig
from gsea_tool.unbiased import UnbiasedSelectionStats
from gsea_tool.dot_plot import DotPlotResult
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.go_clustering import ClusteringResult
from gsea_tool.bar_plot import BarPlotResult

@dataclass
class NotesInput:
    """All inputs needed to generate notes.md, gathered by orchestration."""
    cohort: CohortData
    config: ToolConfig
    fig1_result: DotPlotResult | None
    fig1_method: str | None
    fig2_result: DotPlotResult
    fig3_result: BarPlotResult
    unbiased_stats: UnbiasedSelectionStats
    fisher_result: FisherResult
    clustering_result: ClusteringResult | None
    ...

def generate_notes(notes_input: NotesInput, output_dir: Path) -> Path:
    """Generate notes.md and write it to output_dir.

    Returns the path to the written file.
    """
    ...

def format_figure_legends(notes_input: NotesInput) -> str:
    """Generate the figure legend text section for all produced figures."""
    ...

def format_methods_text(notes_input: NotesInput) -> str:
    """Generate the unified materials and methods text section."""
    ...

def format_summary_statistics(notes_input: NotesInput) -> str:
    """Generate the summary statistics section."""
    ...

def format_reproducibility_note(notes_input: NotesInput) -> str:
    """Generate the reproducibility note with seeds and versions."""
    ...

def format_config_guide(notes_input: NotesInput) -> str:
    """Generate the configuration guide section describing all config.yaml parameters."""
    ...

def get_dependency_versions() -> dict[str, str]:
    """Collect version strings for all key dependencies (Python, matplotlib, pandas, scipy, numpy, goatools, pyyaml)."""
    ...
assert output_dir.is_dir(), 'Output directory must exist'
assert notes_path.exists(), 'notes.md must be written'
assert notes_path.name == 'notes.md', 'Output filename must be notes.md'
