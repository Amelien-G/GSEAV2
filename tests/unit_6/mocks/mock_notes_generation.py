# Auto-generated stub — do not edit
from pathlib import Path
from dataclasses import dataclass
from gsea_tool.data_ingestion import CohortData
from gsea_tool.unbiased import UnbiasedSelectionStats
from gsea_tool.dot_plot import FigureResult

@dataclass
class NotesInput:
    """All inputs needed to generate notes.md, gathered by orchestration."""
    cohort: CohortData
    fig1_result: FigureResult
    fig2_result: FigureResult
    unbiased_stats: UnbiasedSelectionStats
    fdr_threshold: float
    ...

def generate_notes(notes_input: NotesInput, output_dir: Path) -> Path:
    """Generate notes.md and write it to output_dir.

    Returns the path to the written file.
    """
    ...

def format_figure_legends(notes_input: NotesInput) -> str:
    """Generate the figure legend text section for both figures."""
    ...

def format_methods_text(notes_input: NotesInput) -> str:
    """Generate the materials and methods text section."""
    ...

def format_summary_statistics(notes_input: NotesInput) -> str:
    """Generate the summary statistics section."""
    ...

def get_dependency_versions() -> dict[str, str]:
    """Collect version strings for all key dependencies (Python, matplotlib, pandas, scipy/sklearn, numpy)."""
    ...
assert output_dir.is_dir(), 'Output directory must exist'
assert notes_path.exists(), 'notes.md must be written'
assert notes_path.name == 'notes.md', 'Output filename must be notes.md'
