"""Unit 2 -- Configuration loading and validation."""

from pathlib import Path
from dataclasses import dataclass, field, replace
import re
import yaml


_DEFAULT_GAF_URL: str = (
    "https://current.geneontology.org/annotations/fb.gaf.gz"
)


@dataclass(frozen=True)
class CherryPickCategory:
    """A single cherry-pick category entry from config."""
    go_id: str
    label: str


@dataclass(frozen=True)
class DotPlotConfig:
    """Configuration for dot plot figures (Figures 1 and 2)."""
    fdr_threshold: float = 0.05
    top_n: int = 20
    n_groups: int = 4
    random_seed: int = 42


@dataclass(frozen=True)
class FisherConfig:
    """Configuration for Fisher's combined probability test."""
    pseudocount: float = 1e-10
    apply_fdr: bool = False
    fdr_threshold: float = 0.25
    prefilter_pvalue: float = 0.05
    top_n_bars: int = 20


@dataclass(frozen=True)
class ClusteringConfig:
    """Configuration for GO semantic similarity clustering."""
    enabled: bool = True
    similarity_metric: str = "Lin"
    similarity_threshold: float = 0.7
    go_obo_url: str = "https://current.geneontology.org/ontology/go-basic.obo"
    gaf_url: str = ""
    # Optional local file overrides. When set, the file is used directly and
    # no network access occurs -- the supported offline workflow (BUG-006).
    go_obo_path: str = ""
    gaf_path: str = ""


@dataclass(frozen=True)
class PlotAppearanceConfig:
    """Configuration for plot appearance across all figures."""
    dpi: int = 300
    font_family: str = "Arial"
    bar_colormap: str = "YlOrRd"
    bar_figure_width: float = 10.0
    bar_figure_height: float = 8.0
    label_max_length: int = 60
    show_significance_line: bool = True
    show_recurrence_annotation: bool = True


@dataclass(frozen=True)
class GoTreeConfig:
    """Configuration for the GO-hierarchy figures (Figure 1B / Figure 2B).

    Defaults reproduce the hard-coded rendering behaviour that preceded this
    config section, so an existing config.yaml renders byte-identical figures.
    """
    # Wrap width for node labels, in characters (was Unit 11 _LABEL_WRAP_WIDTH).
    label_max_chars: int = 28
    # Render the namespace root node (e.g. "biological_process") as a node.
    # When False, its children become the roots of the drawn tree.
    show_namespace_root: bool = True


@dataclass(frozen=True)
class ToolConfig:
    """Complete tool configuration assembled from config.yaml or defaults."""
    cherry_pick_categories: list = field(default_factory=list)
    dot_plot: DotPlotConfig = field(default_factory=DotPlotConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    plot_appearance: PlotAppearanceConfig = field(default_factory=PlotAppearanceConfig)
    go_tree: GoTreeConfig = field(default_factory=GoTreeConfig)


class ConfigError(Exception):
    """Raised when config.yaml exists but cannot be parsed or validated."""
    pass


def load_config(project_dir: Path) -> ToolConfig:
    """Load configuration from config.yaml in project_dir, or return defaults.

    If config.yaml exists, parse and validate it. If it does not exist, return
    a ToolConfig with all default values. Raises ConfigError on invalid syntax
    or type errors.
    """
    assert project_dir.is_dir(), "project_dir must be an existing directory"

    config_path = project_dir / "config.yaml"
    if not config_path.exists():
        return _apply_gaf_default(ToolConfig())

    try:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML syntax in config.yaml: {e}") from e

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ConfigError(
            f"config.yaml must contain a YAML mapping, got {type(raw).__name__}"
        )

    return validate_config(raw)


def validate_config(raw: dict) -> ToolConfig:
    """Validate a parsed YAML dictionary against the expected schema.

    Applies defaults for missing keys. Raises ConfigError on type errors
    or invalid values (e.g., negative thresholds).
    """
    # --- cherry_pick ---
    cherry_picks = []
    if "cherry_pick" in raw:
        cp_raw = raw["cherry_pick"]
        if not isinstance(cp_raw, list):
            raise ConfigError(
                "cherry_pick must be a list of dicts with 'go_id' and 'label'"
            )
        for i, entry in enumerate(cp_raw):
            if not isinstance(entry, dict):
                raise ConfigError(
                    f"cherry_pick[{i}] must be a dict with 'go_id' and 'label'"
                )
            go_id = entry.get("go_id")
            label = entry.get("label")
            if go_id is None or label is None:
                raise ConfigError(
                    f"cherry_pick[{i}] must have both 'go_id' and 'label'"
                )
            _check_type("cherry_pick[{}].go_id".format(i), go_id, str)
            _check_type("cherry_pick[{}].label".format(i), label, str)
            if not re.match(r"GO:\d{7}$", go_id):
                raise ConfigError(
                    f"cherry_pick[{i}].go_id must match GO:\\d{{7}}, got '{go_id}'"
                )
            if len(label.strip()) == 0:
                raise ConfigError(
                    f"cherry_pick[{i}].label must be non-empty"
                )
            cherry_picks.append(CherryPickCategory(go_id=go_id, label=label))

    # --- dot_plot ---
    dp_defaults = DotPlotConfig()
    dp_raw = raw.get("dot_plot", {})
    if not isinstance(dp_raw, dict):
        raise ConfigError("dot_plot must be a mapping")
    dp_kwargs = {}
    _extract_field(dp_raw, dp_kwargs, "fdr_threshold", float, dp_defaults.fdr_threshold)
    _extract_field(dp_raw, dp_kwargs, "top_n", int, dp_defaults.top_n)
    _extract_field(dp_raw, dp_kwargs, "n_groups", int, dp_defaults.n_groups)
    _extract_field(dp_raw, dp_kwargs, "random_seed", int, dp_defaults.random_seed)
    dot_plot = DotPlotConfig(**dp_kwargs)

    # --- fisher ---
    fi_defaults = FisherConfig()
    fi_raw = raw.get("fisher", {})
    if not isinstance(fi_raw, dict):
        raise ConfigError("fisher must be a mapping")
    fi_kwargs = {}
    _extract_field(fi_raw, fi_kwargs, "pseudocount", float, fi_defaults.pseudocount)
    _extract_field(fi_raw, fi_kwargs, "apply_fdr", bool, fi_defaults.apply_fdr)
    _extract_field(fi_raw, fi_kwargs, "fdr_threshold", float, fi_defaults.fdr_threshold)
    _extract_field(fi_raw, fi_kwargs, "prefilter_pvalue", float, fi_defaults.prefilter_pvalue)
    _extract_field(fi_raw, fi_kwargs, "top_n_bars", int, fi_defaults.top_n_bars)
    fisher = FisherConfig(**fi_kwargs)

    # --- clustering ---
    cl_defaults = ClusteringConfig()
    cl_raw = raw.get("clustering", {})
    if not isinstance(cl_raw, dict):
        raise ConfigError("clustering must be a mapping")
    cl_kwargs = {}
    _extract_field(cl_raw, cl_kwargs, "enabled", bool, cl_defaults.enabled)
    _extract_field(cl_raw, cl_kwargs, "similarity_metric", str, cl_defaults.similarity_metric)
    _extract_field(cl_raw, cl_kwargs, "similarity_threshold", float, cl_defaults.similarity_threshold)
    _extract_field(cl_raw, cl_kwargs, "go_obo_url", str, cl_defaults.go_obo_url)
    _extract_field(cl_raw, cl_kwargs, "gaf_url", str, cl_defaults.gaf_url)
    _extract_field(cl_raw, cl_kwargs, "go_obo_path", str, cl_defaults.go_obo_path)
    _extract_field(cl_raw, cl_kwargs, "gaf_path", str, cl_defaults.gaf_path)
    clustering = ClusteringConfig(**cl_kwargs)

    # --- plot appearance ---
    pa_defaults = PlotAppearanceConfig()
    pa_raw = raw.get("plot", {})
    if not isinstance(pa_raw, dict):
        raise ConfigError("plot must be a mapping")
    pa_kwargs = {}
    _extract_field(pa_raw, pa_kwargs, "dpi", int, pa_defaults.dpi)
    _extract_field(pa_raw, pa_kwargs, "font_family", str, pa_defaults.font_family)
    _extract_field(pa_raw, pa_kwargs, "bar_colormap", str, pa_defaults.bar_colormap)
    _extract_field(pa_raw, pa_kwargs, "bar_figure_width", float, pa_defaults.bar_figure_width)
    _extract_field(pa_raw, pa_kwargs, "bar_figure_height", float, pa_defaults.bar_figure_height)
    _extract_field(pa_raw, pa_kwargs, "label_max_length", int, pa_defaults.label_max_length)
    _extract_field(pa_raw, pa_kwargs, "show_significance_line", bool, pa_defaults.show_significance_line)
    _extract_field(pa_raw, pa_kwargs, "show_recurrence_annotation", bool, pa_defaults.show_recurrence_annotation)
    plot_appearance = PlotAppearanceConfig(**pa_kwargs)

    # --- go_tree ---
    gt_defaults = GoTreeConfig()
    gt_raw = raw.get("go_tree", {})
    if not isinstance(gt_raw, dict):
        raise ConfigError("go_tree must be a mapping")
    gt_kwargs = {}
    _extract_field(gt_raw, gt_kwargs, "label_max_chars", int, gt_defaults.label_max_chars)
    _extract_field(gt_raw, gt_kwargs, "show_namespace_root", bool, gt_defaults.show_namespace_root)
    go_tree = GoTreeConfig(**gt_kwargs)

    config = ToolConfig(
        cherry_pick_categories=list(cherry_picks),
        dot_plot=dot_plot,
        fisher=fisher,
        clustering=clustering,
        plot_appearance=plot_appearance,
        go_tree=go_tree,
    )

    # Apply GAF URL default
    config = _apply_gaf_default(config)

    # Post-condition validation
    _validate_ranges(config)

    return config


def _apply_gaf_default(config: ToolConfig) -> ToolConfig:
    """Set the GAF URL to the default if it was not provided.

    Uses dataclasses.replace rather than re-listing every field: the previous
    hand-written reconstruction silently dropped any newly added field that the
    author forgot to copy across.
    """
    if config.clustering.gaf_url == "":
        new_clustering = replace(config.clustering, gaf_url=_DEFAULT_GAF_URL)
        config = replace(config, clustering=new_clustering)
    return config


def _validate_ranges(config: ToolConfig) -> None:
    """Validate that all numeric parameters are in their expected ranges."""
    if not (0.0 < config.dot_plot.fdr_threshold <= 1.0):
        raise ConfigError(
            f"dot_plot.fdr_threshold must be in (0, 1], got {config.dot_plot.fdr_threshold}"
        )
    if config.dot_plot.top_n <= 0:
        raise ConfigError(
            f"dot_plot.top_n must be positive, got {config.dot_plot.top_n}"
        )
    if config.dot_plot.n_groups <= 0:
        raise ConfigError(
            f"dot_plot.n_groups must be positive, got {config.dot_plot.n_groups}"
        )
    if config.fisher.pseudocount <= 0:
        raise ConfigError(
            f"fisher.pseudocount must be positive, got {config.fisher.pseudocount}"
        )
    if not (0.0 < config.fisher.prefilter_pvalue <= 1.0):
        raise ConfigError(
            f"fisher.prefilter_pvalue must be in (0, 1], got {config.fisher.prefilter_pvalue}"
        )
    if config.fisher.top_n_bars <= 0:
        raise ConfigError(
            f"fisher.top_n_bars must be positive, got {config.fisher.top_n_bars}"
        )
    if not (0.0 < config.clustering.similarity_threshold <= 1.0):
        raise ConfigError(
            f"clustering.similarity_threshold must be in (0, 1], got {config.clustering.similarity_threshold}"
        )
    if config.plot_appearance.dpi <= 0:
        raise ConfigError(
            f"plot.dpi must be positive, got {config.plot_appearance.dpi}"
        )
    if config.plot_appearance.label_max_length <= 0:
        raise ConfigError(
            f"plot.label_max_length must be positive, got {config.plot_appearance.label_max_length}"
        )
    if config.go_tree.label_max_chars <= 0:
        raise ConfigError(
            f"go_tree.label_max_chars must be positive, got {config.go_tree.label_max_chars}"
        )
    if not (0.0 < config.fisher.fdr_threshold <= 1.0):
        raise ConfigError(
            f"fisher.fdr_threshold must be in (0, 1], got {config.fisher.fdr_threshold}"
        )


def _extract_field(
    raw_section: dict,
    kwargs: dict,
    key: str,
    expected_type: type,
    default,
) -> None:
    """Extract a field from raw_section into kwargs, validating its type."""
    if key not in raw_section:
        kwargs[key] = default
        return

    value = raw_section[key]

    # No type coercion: strict type checking
    if expected_type is float:
        # Accept int as float (YAML often parses 1 as int)
        if isinstance(value, bool):
            raise ConfigError(
                f"{key}: expected {expected_type.__name__}, got {type(value).__name__}"
            )
        if not isinstance(value, (int, float)):
            raise ConfigError(
                f"{key}: expected {expected_type.__name__}, got {type(value).__name__}"
            )
        kwargs[key] = float(value)
    elif expected_type is int:
        # bool is subclass of int in Python, reject it
        if isinstance(value, bool):
            raise ConfigError(
                f"{key}: expected {expected_type.__name__}, got {type(value).__name__}"
            )
        if not isinstance(value, int):
            raise ConfigError(
                f"{key}: expected {expected_type.__name__}, got {type(value).__name__}"
            )
        kwargs[key] = value
    elif expected_type is bool:
        if not isinstance(value, bool):
            raise ConfigError(
                f"{key}: expected {expected_type.__name__}, got {type(value).__name__}"
            )
        kwargs[key] = value
    elif expected_type is str:
        if not isinstance(value, str):
            raise ConfigError(
                f"{key}: expected {expected_type.__name__}, got {type(value).__name__}"
            )
        kwargs[key] = value
    else:
        if not isinstance(value, expected_type):
            raise ConfigError(
                f"{key}: expected {expected_type.__name__}, got {type(value).__name__}"
            )
        kwargs[key] = value


def _check_type(path: str, value, expected_type: type) -> None:
    """Check that value is of expected_type, raising ConfigError if not."""
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"{path}: expected {expected_type.__name__}, got {type(value).__name__}"
        )
