# Auto-generated stub — do not edit
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class CherryPickCategory:
    """A single cherry-pick category entry from config."""
    go_id: str
    label: str
    ...

@dataclass
class DotPlotConfig:
    """Configuration for dot plot figures (Figures 1 and 2)."""
    fdr_threshold: float = 0.05
    top_n: int = 20
    n_groups: int = 4
    random_seed: int = 42
    ...

@dataclass
class FisherConfig:
    """Configuration for Fisher's combined probability test."""
    pseudocount: float = 1e-10
    apply_fdr: bool = False
    fdr_threshold: float = 0.25
    prefilter_pvalue: float = 0.05
    top_n_bars: int = 20
    ...

@dataclass
class ClusteringConfig:
    """Configuration for GO semantic similarity clustering."""
    enabled: bool = True
    similarity_metric: str = 'Lin'
    similarity_threshold: float = 0.7
    go_obo_url: str = 'https://current.geneontology.org/ontology/go-basic.obo'
    gaf_url: str = ''
    ...

@dataclass
class PlotAppearanceConfig:
    """Configuration for plot appearance across all figures."""
    dpi: int = 300
    font_family: str = 'Arial'
    bar_colormap: str = 'YlOrRd'
    bar_figure_width: float = 10.0
    bar_figure_height: float = 8.0
    label_max_length: int = 60
    show_significance_line: bool = True
    show_recurrence_annotation: bool = True
    ...

@dataclass
class ToolConfig:
    """Complete tool configuration assembled from config.yaml or defaults."""
    cherry_pick_categories: list[CherryPickCategory] = field(default_factory=list)
    dot_plot: DotPlotConfig = field(default_factory=DotPlotConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    plot_appearance: PlotAppearanceConfig = field(default_factory=PlotAppearanceConfig)
    ...

class ConfigError(Exception):
    """Raised when config.yaml exists but cannot be parsed or validated."""
    ...

def load_config(project_dir: Path) -> ToolConfig:
    """Load configuration from config.yaml in project_dir, or return defaults.

    If config.yaml exists, parse and validate it. If it does not exist, return
    a ToolConfig with all default values. Raises ConfigError on invalid syntax
    or type errors.
    """
    ...

def validate_config(raw: dict) -> ToolConfig:
    """Validate a parsed YAML dictionary against the expected schema.

    Applies defaults for missing keys. Raises ConfigError on type errors
    or invalid values (e.g., negative thresholds).
    """
    ...
assert project_dir.is_dir(), 'project_dir must be an existing directory'
assert 0.0 < config.dot_plot.fdr_threshold <= 1.0, 'FDR threshold must be in (0, 1]'
assert config.dot_plot.top_n > 0, 'top_n must be positive'
assert config.dot_plot.n_groups > 0, 'n_groups must be positive'
assert config.fisher.pseudocount > 0, 'pseudocount must be positive'
assert 0.0 < config.fisher.prefilter_pvalue <= 1.0, 'prefilter_pvalue must be in (0, 1]'
assert config.fisher.top_n_bars > 0, 'top_n_bars must be positive'
assert 0.0 < config.clustering.similarity_threshold <= 1.0, 'similarity_threshold must be in (0, 1]'
assert all((re.match('GO:\\d{7}$', c.go_id) for c in config.cherry_pick_categories)), 'Each cherry_pick go_id must match GO:\\d{7}'
assert all((len(c.label.strip()) > 0 for c in config.cherry_pick_categories)), 'Each cherry_pick label must be non-empty'
assert config.plot_appearance.dpi > 0, 'DPI must be positive'
assert config.plot_appearance.label_max_length > 0, 'label_max_length must be positive'
