"""Tests for Unit 2 -- Configuration loading and validation.

Tests are derived from the blueprint contracts, invariants, and error conditions.

Synthetic Data Assumptions:
  - YAML config files use the nested key hierarchy: dot_plot, fisher, clustering,
    plot (top-level keys) as defined in the blueprint.
  - Default values match the blueprint signatures (e.g., fdr_threshold=0.05,
    top_n=20, dpi=300, etc.).
  - Invalid YAML is represented by syntactically broken YAML text.
  - Type errors are triggered by providing a string where a numeric type is
    expected, per contract 5 (no type coercion).
  - Value range errors use boundary values: 0 for positive-required fields,
    negative for (0, 1] ranges, >1 for thresholds capped at 1.
  - The default GAF URL is the FlyBase Drosophila melanogaster GAF download URL.
  - All config dataclasses are frozen (immutable after construction).
"""

import textwrap
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from gsea_tool.configuration import (
    CherryPickCategory,
    ClusteringConfig,
    ConfigError,
    DotPlotConfig,
    FisherConfig,
    PlotAppearanceConfig,
    ToolConfig,
    load_config,
    validate_config,
    _DEFAULT_GAF_URL,
)

# Handle FrozenInstanceError availability (Python 3.11+)
try:
    from dataclasses import FrozenInstanceError
except ImportError:
    FrozenInstanceError = AttributeError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_project_dir(tmp_path):
    """A project directory with no config.yaml."""
    return tmp_path


@pytest.fixture
def project_dir_with_config(tmp_path):
    """Factory fixture: write config.yaml content and return the directory."""
    def _write(yaml_content: str) -> Path:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(textwrap.dedent(yaml_content))
        return tmp_path
    return _write


# ===========================================================================
# Dataclass structure and defaults
# ===========================================================================


class TestDataclassStructures:
    """Verify configuration dataclasses have expected fields and defaults."""

    def test_dot_plot_config_fields(self):
        """DotPlotConfig has all blueprint-specified fields."""
        field_names = {f.name for f in dataclass_fields(DotPlotConfig)}
        assert {"fdr_threshold", "top_n", "n_groups", "random_seed"} <= field_names

    def test_dot_plot_config_defaults(self):
        """DotPlotConfig defaults match the blueprint."""
        cfg = DotPlotConfig()
        assert cfg.fdr_threshold == 0.05
        assert cfg.top_n == 20
        assert cfg.n_groups == 4
        assert cfg.random_seed == 42

    def test_fisher_config_fields(self):
        """FisherConfig has all blueprint-specified fields."""
        field_names = {f.name for f in dataclass_fields(FisherConfig)}
        assert {"pseudocount", "apply_fdr", "fdr_threshold", "prefilter_pvalue", "top_n_bars"} <= field_names

    def test_fisher_config_defaults(self):
        """FisherConfig defaults match the blueprint."""
        cfg = FisherConfig()
        assert cfg.pseudocount == 1e-10
        assert cfg.apply_fdr is False
        assert cfg.fdr_threshold == 0.25
        assert cfg.prefilter_pvalue == 0.05
        assert cfg.top_n_bars == 20

    def test_clustering_config_fields(self):
        """ClusteringConfig has all blueprint-specified fields."""
        field_names = {f.name for f in dataclass_fields(ClusteringConfig)}
        assert {"enabled", "similarity_metric", "similarity_threshold", "go_obo_url", "gaf_url"} <= field_names

    def test_clustering_config_defaults(self):
        """ClusteringConfig defaults match the blueprint."""
        cfg = ClusteringConfig()
        assert cfg.enabled is True
        assert cfg.similarity_metric == "Lin"
        assert cfg.similarity_threshold == 0.7
        assert cfg.go_obo_url == "https://current.geneontology.org/ontology/go-basic.obo"

    def test_plot_appearance_config_fields(self):
        """PlotAppearanceConfig has all blueprint-specified fields."""
        field_names = {f.name for f in dataclass_fields(PlotAppearanceConfig)}
        expected = {"dpi", "font_family", "bar_colormap", "bar_figure_width",
                    "bar_figure_height", "label_max_length", "show_significance_line",
                    "show_recurrence_annotation"}
        assert expected <= field_names

    def test_plot_appearance_config_defaults(self):
        """PlotAppearanceConfig defaults match the blueprint."""
        cfg = PlotAppearanceConfig()
        assert cfg.dpi == 300
        assert cfg.font_family == "Arial"
        assert cfg.bar_colormap == "YlOrRd"
        assert cfg.bar_figure_width == 10.0
        assert cfg.bar_figure_height == 8.0
        assert cfg.label_max_length == 60
        assert cfg.show_significance_line is True
        assert cfg.show_recurrence_annotation is True

    def test_tool_config_fields(self):
        """ToolConfig has all section fields."""
        field_names = {f.name for f in dataclass_fields(ToolConfig)}
        assert {"dot_plot", "fisher", "clustering", "plot_appearance"} <= field_names

    def test_tool_config_defaults_compose_sub_configs(self):
        """ToolConfig default creates sub-config objects with their defaults."""
        cfg = ToolConfig()
        assert isinstance(cfg.dot_plot, DotPlotConfig)
        assert isinstance(cfg.fisher, FisherConfig)
        assert isinstance(cfg.clustering, ClusteringConfig)
        assert isinstance(cfg.plot_appearance, PlotAppearanceConfig)

    def test_config_error_is_exception(self):
        """ConfigError is a subclass of Exception."""
        assert issubclass(ConfigError, Exception)


# ===========================================================================
# Contract 1: No config.yaml returns all defaults
# ===========================================================================


class TestNoConfigFileDefaults:
    """Contract 1: If config.yaml does not exist, return ToolConfig with defaults."""

    def test_no_config_file_returns_tool_config(self, empty_project_dir):
        """load_config returns a ToolConfig when no config.yaml exists."""
        cfg = load_config(empty_project_dir)
        assert isinstance(cfg, ToolConfig)

    def test_all_dot_plot_defaults_when_no_config(self, empty_project_dir):
        """All dot_plot values are at defaults when no config file exists."""
        cfg = load_config(empty_project_dir)
        assert cfg.dot_plot.fdr_threshold == 0.05
        assert cfg.dot_plot.top_n == 20
        assert cfg.dot_plot.n_groups == 4
        assert cfg.dot_plot.random_seed == 42

    def test_all_fisher_defaults_when_no_config(self, empty_project_dir):
        """All fisher values are at defaults when no config file exists."""
        cfg = load_config(empty_project_dir)
        assert cfg.fisher.pseudocount == 1e-10
        assert cfg.fisher.apply_fdr is False
        assert cfg.fisher.fdr_threshold == 0.25
        assert cfg.fisher.prefilter_pvalue == 0.05
        assert cfg.fisher.top_n_bars == 20

    def test_all_clustering_defaults_when_no_config(self, empty_project_dir):
        """All clustering values are at defaults when no config file exists."""
        cfg = load_config(empty_project_dir)
        assert cfg.clustering.enabled is True
        assert cfg.clustering.similarity_metric == "Lin"
        assert cfg.clustering.similarity_threshold == 0.7
        assert cfg.clustering.go_obo_url == "https://current.geneontology.org/ontology/go-basic.obo"

    def test_all_plot_defaults_when_no_config(self, empty_project_dir):
        """All plot appearance values are at defaults when no config file exists."""
        cfg = load_config(empty_project_dir)
        assert cfg.plot_appearance.dpi == 300
        assert cfg.plot_appearance.font_family == "Arial"
        assert cfg.plot_appearance.bar_colormap == "YlOrRd"
        assert cfg.plot_appearance.bar_figure_width == 10.0
        assert cfg.plot_appearance.bar_figure_height == 8.0
        assert cfg.plot_appearance.label_max_length == 60
        assert cfg.plot_appearance.show_significance_line is True
        assert cfg.plot_appearance.show_recurrence_annotation is True

    def test_default_gaf_url_is_flybase_drosophila(self, empty_project_dir):
        """Contract 8: Default GAF URL is the GO Consortium Drosophila melanogaster URL."""
        cfg = load_config(empty_project_dir)
        assert cfg.clustering.gaf_url == _DEFAULT_GAF_URL
        assert "fb.gaf" in cfg.clustering.gaf_url


# ===========================================================================
# Contract 2: Valid config overrides defaults; unspecified keys keep defaults
# ===========================================================================


class TestValidConfigOverrides:
    """Contract 2: Specified values override defaults; unspecified keys retain defaults."""

    def test_override_dot_plot_fdr_threshold(self, project_dir_with_config):
        """Specifying dot_plot.fdr_threshold overrides the default."""
        d = project_dir_with_config("""\
            dot_plot:
              fdr_threshold: 0.1
        """)
        cfg = load_config(d)
        assert cfg.dot_plot.fdr_threshold == 0.1
        # Unspecified keys retain defaults
        assert cfg.dot_plot.top_n == 20
        assert cfg.dot_plot.n_groups == 4

    def test_override_fisher_section(self, project_dir_with_config):
        """Specifying fisher keys overrides those defaults."""
        d = project_dir_with_config("""\
            fisher:
              pseudocount: 0.001
              apply_fdr: true
              top_n_bars: 30
        """)
        cfg = load_config(d)
        assert cfg.fisher.pseudocount == 0.001
        assert cfg.fisher.apply_fdr is True
        assert cfg.fisher.top_n_bars == 30
        # Unspecified fisher keys retain defaults
        assert cfg.fisher.fdr_threshold == 0.25
        assert cfg.fisher.prefilter_pvalue == 0.05

    def test_override_clustering_section(self, project_dir_with_config):
        """Specifying clustering keys overrides those defaults."""
        d = project_dir_with_config("""\
            clustering:
              enabled: false
              similarity_metric: "Resnik"
              similarity_threshold: 0.5
        """)
        cfg = load_config(d)
        assert cfg.clustering.enabled is False
        assert cfg.clustering.similarity_metric == "Resnik"
        assert cfg.clustering.similarity_threshold == 0.5

    def test_override_plot_section(self, project_dir_with_config):
        """Specifying plot keys overrides those defaults."""
        d = project_dir_with_config("""\
            plot:
              dpi: 150
              font_family: "Helvetica"
              bar_colormap: "viridis"
              bar_figure_width: 12.0
              bar_figure_height: 6.0
              label_max_length: 40
              show_significance_line: false
              show_recurrence_annotation: false
        """)
        cfg = load_config(d)
        assert cfg.plot_appearance.dpi == 150
        assert cfg.plot_appearance.font_family == "Helvetica"
        assert cfg.plot_appearance.bar_colormap == "viridis"
        assert cfg.plot_appearance.bar_figure_width == 12.0
        assert cfg.plot_appearance.bar_figure_height == 6.0
        assert cfg.plot_appearance.label_max_length == 40
        assert cfg.plot_appearance.show_significance_line is False
        assert cfg.plot_appearance.show_recurrence_annotation is False

    def test_override_gaf_url(self, project_dir_with_config):
        """Specifying clustering.gaf_url overrides the default FlyBase URL."""
        d = project_dir_with_config("""\
            clustering:
              gaf_url: "https://example.com/custom.gaf.gz"
        """)
        cfg = load_config(d)
        assert cfg.clustering.gaf_url == "https://example.com/custom.gaf.gz"

    def test_partial_config_retains_other_section_defaults(self, project_dir_with_config):
        """Specifying only one section leaves other sections at defaults."""
        d = project_dir_with_config("""\
            dot_plot:
              top_n: 10
        """)
        cfg = load_config(d)
        assert cfg.dot_plot.top_n == 10
        # Other sections unchanged
        assert cfg.fisher.pseudocount == 1e-10
        assert cfg.clustering.enabled is True
        assert cfg.plot_appearance.dpi == 300

    def test_integer_value_accepted_for_float_field(self, project_dir_with_config):
        """An integer value (e.g. 1) is accepted for float fields like fdr_threshold."""
        d = project_dir_with_config("""\
            dot_plot:
              fdr_threshold: 1
        """)
        cfg = load_config(d)
        assert cfg.dot_plot.fdr_threshold == 1.0
        assert isinstance(cfg.dot_plot.fdr_threshold, float)

    def test_override_random_seed(self, project_dir_with_config):
        """Specifying dot_plot.random_seed overrides the default."""
        d = project_dir_with_config("""\
            dot_plot:
              random_seed: 99
        """)
        cfg = load_config(d)
        assert cfg.dot_plot.random_seed == 99

    def test_override_go_obo_url(self, project_dir_with_config):
        """Specifying clustering.go_obo_url overrides the default."""
        d = project_dir_with_config("""\
            clustering:
              go_obo_url: "https://example.com/go.obo"
        """)
        cfg = load_config(d)
        assert cfg.clustering.go_obo_url == "https://example.com/go.obo"


# ===========================================================================
# Contract 3: Unknown keys are silently ignored
# ===========================================================================


class TestUnknownKeysIgnored:
    """Contract 3: Unknown keys in the YAML file are silently ignored."""

    def test_unknown_top_level_key_ignored(self, project_dir_with_config):
        """Unknown top-level keys do not cause errors."""
        d = project_dir_with_config("""\
            unknown_section:
              foo: bar
            dot_plot:
              top_n: 15
        """)
        cfg = load_config(d)
        assert cfg.dot_plot.top_n == 15

    def test_unknown_nested_key_ignored(self, project_dir_with_config):
        """Unknown keys within a known section do not cause errors."""
        d = project_dir_with_config("""\
            dot_plot:
              top_n: 15
              unknown_key: 999
        """)
        cfg = load_config(d)
        assert cfg.dot_plot.top_n == 15

    def test_multiple_unknown_top_level_keys(self, project_dir_with_config):
        """Multiple unknown top-level keys are all silently ignored."""
        d = project_dir_with_config("""\
            future_feature_1: true
            future_feature_2:
              nested: value
            fisher:
              apply_fdr: true
        """)
        cfg = load_config(d)
        assert cfg.fisher.apply_fdr is True


# ===========================================================================
# Contract 5: No type coercion
# ===========================================================================


class TestNoTypeCoercion:
    """Contract 5: Type coercion is not performed; wrong types cause ConfigError."""

    def test_string_where_float_expected_raises_config_error(self, project_dir_with_config):
        """A string '0.05' where a float is expected raises ConfigError."""
        d = project_dir_with_config("""\
            dot_plot:
              fdr_threshold: "0.05"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_string_where_int_expected_raises_config_error(self, project_dir_with_config):
        """A string '20' where an int is expected raises ConfigError."""
        d = project_dir_with_config("""\
            dot_plot:
              top_n: "20"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_float_where_int_expected_raises_config_error(self, project_dir_with_config):
        """A float 20.5 where an int is expected raises ConfigError."""
        d = project_dir_with_config("""\
            dot_plot:
              top_n: 20.5
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_string_where_bool_expected_raises_config_error(self, project_dir_with_config):
        """A string 'true' where a bool is expected raises ConfigError."""
        d = project_dir_with_config("""\
            fisher:
              apply_fdr: "true"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_int_where_bool_expected_raises_config_error(self, project_dir_with_config):
        """An integer 1 where a bool is expected raises ConfigError."""
        d = project_dir_with_config("""\
            clustering:
              enabled: 1
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_bool_where_int_expected_raises_config_error(self, project_dir_with_config):
        """A boolean where an int is expected raises ConfigError (bool is subclass of int)."""
        d = project_dir_with_config("""\
            dot_plot:
              top_n: true
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_bool_where_float_expected_raises_config_error(self, project_dir_with_config):
        """A boolean where a float is expected raises ConfigError."""
        d = project_dir_with_config("""\
            dot_plot:
              fdr_threshold: false
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_int_where_string_expected_raises_config_error(self, project_dir_with_config):
        """An integer where a string is expected raises ConfigError."""
        d = project_dir_with_config("""\
            plot:
              font_family: 42
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_list_where_float_expected_raises_config_error(self, project_dir_with_config):
        """A list where a float is expected raises ConfigError."""
        d = project_dir_with_config("""\
            fisher:
              pseudocount: [1, 2, 3]
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_dict_where_int_expected_raises_config_error(self, project_dir_with_config):
        """A dict where an int is expected raises ConfigError."""
        d = project_dir_with_config("""\
            plot:
              dpi:
                value: 300
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_string_where_float_in_clustering_raises_config_error(self, project_dir_with_config):
        """A string where float expected in clustering section raises ConfigError."""
        d = project_dir_with_config("""\
            clustering:
              similarity_threshold: "high"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_bool_where_float_in_fisher_raises_config_error(self, project_dir_with_config):
        """A boolean where float expected in fisher section raises ConfigError."""
        d = project_dir_with_config("""\
            fisher:
              fdr_threshold: true
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_bool_where_int_in_plot_raises_config_error(self, project_dir_with_config):
        """A boolean where int expected in plot section raises ConfigError."""
        d = project_dir_with_config("""\
            plot:
              dpi: true
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_bool_where_int_label_max_length_raises_config_error(self, project_dir_with_config):
        """A boolean where int expected for label_max_length raises ConfigError."""
        d = project_dir_with_config("""\
            plot:
              label_max_length: false
        """)
        with pytest.raises(ConfigError):
            load_config(d)


# ===========================================================================
# Contract 6: ToolConfig is immutable after construction
# ===========================================================================


class TestImmutability:
    """Contract 6: ToolConfig is immutable (frozen dataclass)."""

    def test_tool_config_is_frozen(self, empty_project_dir):
        """Cannot assign to ToolConfig attributes after construction."""
        cfg = load_config(empty_project_dir)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.dot_plot = DotPlotConfig(fdr_threshold=0.99)

    def test_dot_plot_config_is_frozen(self, empty_project_dir):
        """Cannot assign to DotPlotConfig attributes after construction."""
        cfg = load_config(empty_project_dir)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.dot_plot.fdr_threshold = 0.99

    def test_fisher_config_is_frozen(self, empty_project_dir):
        """Cannot assign to FisherConfig attributes after construction."""
        cfg = load_config(empty_project_dir)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.fisher.pseudocount = 0.5

    def test_clustering_config_is_frozen(self, empty_project_dir):
        """Cannot assign to ClusteringConfig attributes after construction."""
        cfg = load_config(empty_project_dir)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.clustering.enabled = False

    def test_plot_config_is_frozen(self, empty_project_dir):
        """Cannot assign to PlotAppearanceConfig attributes after construction."""
        cfg = load_config(empty_project_dir)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.plot_appearance.dpi = 72


# ===========================================================================
# Error Condition: Invalid YAML syntax
# ===========================================================================


class TestInvalidYAMLSyntax:
    """ConfigError is raised when config.yaml has invalid YAML syntax."""

    def test_malformed_yaml_raises_config_error(self, project_dir_with_config):
        """Invalid YAML syntax triggers ConfigError."""
        d = project_dir_with_config("""\
            key: [invalid
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_non_mapping_yaml_raises_config_error(self, project_dir_with_config):
        """A YAML file containing a list instead of a mapping raises ConfigError."""
        d = project_dir_with_config("""\
            - item1
            - item2
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_non_mapping_scalar_yaml_raises_config_error(self, project_dir_with_config):
        """A YAML file containing just a scalar string raises ConfigError."""
        d = project_dir_with_config("""\
            just a plain string
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_section_not_a_mapping_raises_config_error(self, project_dir_with_config):
        """A section value that is not a dict raises ConfigError."""
        d = project_dir_with_config("""\
            dot_plot: "not_a_dict"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_fisher_section_not_a_mapping_raises_config_error(self, project_dir_with_config):
        """Fisher section as a list raises ConfigError."""
        d = project_dir_with_config("""\
            fisher:
              - item1
              - item2
        """)
        with pytest.raises(ConfigError):
            load_config(d)


# ===========================================================================
# Error Condition: Invalid parameter values (invariant violations)
# ===========================================================================


class TestInvariantViolations:
    """ConfigError raised when values violate post-condition invariants."""

    # --- dot_plot.fdr_threshold: must be in (0, 1] ---

    def test_fdr_threshold_zero_raises_config_error(self, project_dir_with_config):
        """dot_plot.fdr_threshold of 0 violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            dot_plot:
              fdr_threshold: 0.0
        """)
        with pytest.raises(ConfigError, match="fdr_threshold"):
            load_config(d)

    def test_fdr_threshold_negative_raises_config_error(self, project_dir_with_config):
        """dot_plot.fdr_threshold negative violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            dot_plot:
              fdr_threshold: -0.1
        """)
        with pytest.raises(ConfigError, match="fdr_threshold"):
            load_config(d)

    def test_fdr_threshold_above_one_raises_config_error(self, project_dir_with_config):
        """dot_plot.fdr_threshold above 1 violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            dot_plot:
              fdr_threshold: 1.5
        """)
        with pytest.raises(ConfigError, match="fdr_threshold"):
            load_config(d)

    def test_fdr_threshold_exactly_one_is_valid(self, project_dir_with_config):
        """dot_plot.fdr_threshold of exactly 1.0 is within (0, 1]."""
        d = project_dir_with_config("""\
            dot_plot:
              fdr_threshold: 1.0
        """)
        cfg = load_config(d)
        assert cfg.dot_plot.fdr_threshold == 1.0

    def test_fdr_threshold_small_positive_is_valid(self, project_dir_with_config):
        """dot_plot.fdr_threshold of a very small positive value is valid."""
        d = project_dir_with_config("""\
            dot_plot:
              fdr_threshold: 0.001
        """)
        cfg = load_config(d)
        assert cfg.dot_plot.fdr_threshold == 0.001

    # --- dot_plot.top_n: must be positive ---

    def test_top_n_zero_raises_config_error(self, project_dir_with_config):
        """dot_plot.top_n of 0 violates positive invariant."""
        d = project_dir_with_config("""\
            dot_plot:
              top_n: 0
        """)
        with pytest.raises(ConfigError, match="top_n"):
            load_config(d)

    def test_top_n_negative_raises_config_error(self, project_dir_with_config):
        """dot_plot.top_n negative violates positive invariant."""
        d = project_dir_with_config("""\
            dot_plot:
              top_n: -5
        """)
        with pytest.raises(ConfigError, match="top_n"):
            load_config(d)

    # --- dot_plot.n_groups: must be positive ---

    def test_n_groups_zero_raises_config_error(self, project_dir_with_config):
        """dot_plot.n_groups of 0 violates positive invariant."""
        d = project_dir_with_config("""\
            dot_plot:
              n_groups: 0
        """)
        with pytest.raises(ConfigError, match="n_groups"):
            load_config(d)

    def test_n_groups_negative_raises_config_error(self, project_dir_with_config):
        """dot_plot.n_groups negative violates positive invariant."""
        d = project_dir_with_config("""\
            dot_plot:
              n_groups: -1
        """)
        with pytest.raises(ConfigError, match="n_groups"):
            load_config(d)

    # --- fisher.pseudocount: must be positive ---

    def test_pseudocount_zero_raises_config_error(self, project_dir_with_config):
        """fisher.pseudocount of 0 violates positive invariant."""
        d = project_dir_with_config("""\
            fisher:
              pseudocount: 0.0
        """)
        with pytest.raises(ConfigError, match="pseudocount"):
            load_config(d)

    def test_pseudocount_negative_raises_config_error(self, project_dir_with_config):
        """fisher.pseudocount negative violates positive invariant."""
        d = project_dir_with_config("""\
            fisher:
              pseudocount: -1.0
        """)
        with pytest.raises(ConfigError, match="pseudocount"):
            load_config(d)

    # --- fisher.prefilter_pvalue: must be in (0, 1] ---

    def test_prefilter_pvalue_zero_raises_config_error(self, project_dir_with_config):
        """fisher.prefilter_pvalue of 0 violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            fisher:
              prefilter_pvalue: 0.0
        """)
        with pytest.raises(ConfigError, match="prefilter_pvalue"):
            load_config(d)

    def test_prefilter_pvalue_above_one_raises_config_error(self, project_dir_with_config):
        """fisher.prefilter_pvalue above 1 violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            fisher:
              prefilter_pvalue: 1.5
        """)
        with pytest.raises(ConfigError, match="prefilter_pvalue"):
            load_config(d)

    def test_prefilter_pvalue_negative_raises_config_error(self, project_dir_with_config):
        """fisher.prefilter_pvalue negative violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            fisher:
              prefilter_pvalue: -0.01
        """)
        with pytest.raises(ConfigError, match="prefilter_pvalue"):
            load_config(d)

    def test_prefilter_pvalue_exactly_one_is_valid(self, project_dir_with_config):
        """fisher.prefilter_pvalue of exactly 1.0 is within (0, 1]."""
        d = project_dir_with_config("""\
            fisher:
              prefilter_pvalue: 1.0
        """)
        cfg = load_config(d)
        assert cfg.fisher.prefilter_pvalue == 1.0

    # --- fisher.top_n_bars: must be positive ---

    def test_top_n_bars_zero_raises_config_error(self, project_dir_with_config):
        """fisher.top_n_bars of 0 violates positive invariant."""
        d = project_dir_with_config("""\
            fisher:
              top_n_bars: 0
        """)
        with pytest.raises(ConfigError, match="top_n_bars"):
            load_config(d)

    def test_top_n_bars_negative_raises_config_error(self, project_dir_with_config):
        """fisher.top_n_bars negative violates positive invariant."""
        d = project_dir_with_config("""\
            fisher:
              top_n_bars: -10
        """)
        with pytest.raises(ConfigError, match="top_n_bars"):
            load_config(d)

    # --- clustering.similarity_threshold: must be in (0, 1] ---

    def test_similarity_threshold_zero_raises_config_error(self, project_dir_with_config):
        """clustering.similarity_threshold of 0 violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            clustering:
              similarity_threshold: 0.0
        """)
        with pytest.raises(ConfigError, match="similarity_threshold"):
            load_config(d)

    def test_similarity_threshold_above_one_raises_config_error(self, project_dir_with_config):
        """clustering.similarity_threshold above 1 violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            clustering:
              similarity_threshold: 1.5
        """)
        with pytest.raises(ConfigError, match="similarity_threshold"):
            load_config(d)

    def test_similarity_threshold_negative_raises_config_error(self, project_dir_with_config):
        """clustering.similarity_threshold negative violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            clustering:
              similarity_threshold: -0.3
        """)
        with pytest.raises(ConfigError, match="similarity_threshold"):
            load_config(d)

    def test_similarity_threshold_exactly_one_is_valid(self, project_dir_with_config):
        """clustering.similarity_threshold of exactly 1.0 is within (0, 1]."""
        d = project_dir_with_config("""\
            clustering:
              similarity_threshold: 1.0
        """)
        cfg = load_config(d)
        assert cfg.clustering.similarity_threshold == 1.0

    # --- plot.dpi: must be positive ---

    def test_dpi_zero_raises_config_error(self, project_dir_with_config):
        """plot.dpi of 0 violates positive invariant."""
        d = project_dir_with_config("""\
            plot:
              dpi: 0
        """)
        with pytest.raises(ConfigError, match="dpi"):
            load_config(d)

    def test_dpi_negative_raises_config_error(self, project_dir_with_config):
        """plot.dpi negative violates positive invariant."""
        d = project_dir_with_config("""\
            plot:
              dpi: -100
        """)
        with pytest.raises(ConfigError, match="dpi"):
            load_config(d)

    # --- plot.label_max_length: must be positive ---

    def test_label_max_length_zero_raises_config_error(self, project_dir_with_config):
        """plot.label_max_length of 0 violates positive invariant."""
        d = project_dir_with_config("""\
            plot:
              label_max_length: 0
        """)
        with pytest.raises(ConfigError, match="label_max_length"):
            load_config(d)

    def test_label_max_length_negative_raises_config_error(self, project_dir_with_config):
        """plot.label_max_length negative violates positive invariant."""
        d = project_dir_with_config("""\
            plot:
              label_max_length: -5
        """)
        with pytest.raises(ConfigError, match="label_max_length"):
            load_config(d)


# ===========================================================================
# Empty / comment-only YAML
# ===========================================================================


class TestEmptyYAML:
    """An empty or comment-only config.yaml returns defaults."""

    def test_empty_file_returns_defaults(self, tmp_path):
        """An empty config.yaml returns ToolConfig with all defaults."""
        (tmp_path / "config.yaml").write_text("")
        cfg = load_config(tmp_path)
        assert isinstance(cfg, ToolConfig)
        assert cfg.dot_plot.fdr_threshold == 0.05

    def test_comment_only_file_returns_defaults(self, tmp_path):
        """A config.yaml with only comments returns ToolConfig with all defaults."""
        (tmp_path / "config.yaml").write_text("# This is a comment\n# Another comment\n")
        cfg = load_config(tmp_path)
        assert isinstance(cfg, ToolConfig)
        assert cfg.dot_plot.top_n == 20


# ===========================================================================
# Pre-condition: project_dir must be an existing directory
# ===========================================================================


class TestPreConditions:
    """Pre-condition: project_dir must be an existing directory."""

    def test_nonexistent_directory_raises_assertion(self, tmp_path):
        """Passing a non-existent directory raises AssertionError."""
        fake_dir = tmp_path / "nonexistent"
        with pytest.raises(AssertionError, match="project_dir"):
            load_config(fake_dir)

    def test_file_path_instead_of_directory_raises_assertion(self, tmp_path):
        """Passing a file path instead of directory raises AssertionError."""
        file_path = tmp_path / "somefile.txt"
        file_path.write_text("content")
        with pytest.raises(AssertionError, match="project_dir"):
            load_config(file_path)


# ===========================================================================
# validate_config direct tests
# ===========================================================================


class TestValidateConfig:
    """Direct tests of validate_config with raw dictionaries."""

    def test_empty_dict_returns_defaults(self):
        """An empty dict produces ToolConfig with all defaults."""
        cfg = validate_config({})
        assert isinstance(cfg, ToolConfig)
        assert cfg.dot_plot.fdr_threshold == 0.05
        assert cfg.fisher.pseudocount == 1e-10
        assert cfg.clustering.enabled is True
        assert cfg.plot_appearance.dpi == 300

    def test_section_not_a_mapping_raises_config_error(self):
        """A section value that is not a dict raises ConfigError."""
        with pytest.raises(ConfigError, match="mapping"):
            validate_config({"dot_plot": "not_a_dict"})

    def test_valid_overrides_applied(self):
        """Valid overrides in the raw dict are applied."""
        raw = {
            "dot_plot": {"top_n": 10, "fdr_threshold": 0.01},
            "fisher": {"apply_fdr": True},
        }
        cfg = validate_config(raw)
        assert cfg.dot_plot.top_n == 10
        assert cfg.dot_plot.fdr_threshold == 0.01
        assert cfg.fisher.apply_fdr is True

    def test_type_error_in_validate_config(self):
        """Type error in raw dict raises ConfigError."""
        with pytest.raises(ConfigError):
            validate_config({"dot_plot": {"fdr_threshold": "bad"}})

    def test_unknown_keys_ignored_in_validate_config(self):
        """Unknown keys at top level and in sections are ignored."""
        raw = {
            "unknown_top": {"x": 1},
            "dot_plot": {"top_n": 5, "unknown_nested": "ignored"},
        }
        cfg = validate_config(raw)
        assert cfg.dot_plot.top_n == 5

    def test_clustering_section_as_list_raises_config_error(self):
        """Clustering section as a list raises ConfigError."""
        with pytest.raises(ConfigError, match="mapping"):
            validate_config({"clustering": [1, 2, 3]})

    def test_plot_section_as_int_raises_config_error(self):
        """Plot section as an integer raises ConfigError."""
        with pytest.raises(ConfigError, match="mapping"):
            validate_config({"plot": 42})


# ===========================================================================
# Full integration-style test with all sections
# ===========================================================================


class TestFullConfig:
    """A comprehensive config.yaml with all sections specified."""

    def test_full_config_all_overrides(self, project_dir_with_config):
        """All config values can be overridden simultaneously."""
        d = project_dir_with_config("""\
            dot_plot:
              fdr_threshold: 0.01
              top_n: 10
              n_groups: 3
              random_seed: 123
            fisher:
              pseudocount: 0.001
              apply_fdr: true
              fdr_threshold: 0.1
              prefilter_pvalue: 0.01
              top_n_bars: 50
            clustering:
              enabled: false
              similarity_metric: "Resnik"
              similarity_threshold: 0.8
              go_obo_url: "https://example.com/go.obo"
              gaf_url: "https://example.com/custom.gaf.gz"
            plot:
              dpi: 600
              font_family: "Times New Roman"
              bar_colormap: "coolwarm"
              bar_figure_width: 14.0
              bar_figure_height: 10.0
              label_max_length: 80
              show_significance_line: false
              show_recurrence_annotation: false
        """)
        cfg = load_config(d)

        assert cfg.dot_plot.fdr_threshold == 0.01
        assert cfg.dot_plot.top_n == 10
        assert cfg.dot_plot.n_groups == 3
        assert cfg.dot_plot.random_seed == 123

        assert cfg.fisher.pseudocount == 0.001
        assert cfg.fisher.apply_fdr is True
        assert cfg.fisher.fdr_threshold == 0.1
        assert cfg.fisher.prefilter_pvalue == 0.01
        assert cfg.fisher.top_n_bars == 50

        assert cfg.clustering.enabled is False
        assert cfg.clustering.similarity_metric == "Resnik"
        assert cfg.clustering.similarity_threshold == 0.8
        assert cfg.clustering.go_obo_url == "https://example.com/go.obo"
        assert cfg.clustering.gaf_url == "https://example.com/custom.gaf.gz"

        assert cfg.plot_appearance.dpi == 600
        assert cfg.plot_appearance.font_family == "Times New Roman"
        assert cfg.plot_appearance.bar_colormap == "coolwarm"
        assert cfg.plot_appearance.bar_figure_width == 14.0
        assert cfg.plot_appearance.bar_figure_height == 10.0
        assert cfg.plot_appearance.label_max_length == 80
        assert cfg.plot_appearance.show_significance_line is False
        assert cfg.plot_appearance.show_recurrence_annotation is False


# ===========================================================================
# Return type verification
# ===========================================================================


class TestReturnTypes:
    """Verify return types of load_config and validate_config."""

    def test_load_config_returns_tool_config(self, empty_project_dir):
        """load_config returns a ToolConfig instance."""
        result = load_config(empty_project_dir)
        assert isinstance(result, ToolConfig)

    def test_validate_config_returns_tool_config(self):
        """validate_config returns a ToolConfig instance."""
        result = validate_config({})
        assert isinstance(result, ToolConfig)

    def test_load_config_with_valid_yaml_returns_tool_config(self, project_dir_with_config):
        """load_config with valid YAML returns a ToolConfig instance."""
        d = project_dir_with_config("""\
            dot_plot:
              top_n: 5
        """)
        result = load_config(d)
        assert isinstance(result, ToolConfig)


# ===========================================================================
# CherryPickCategory dataclass and cherry_pick config section
# ===========================================================================


class TestCherryPickCategoryDataclass:
    """Verify CherryPickCategory dataclass structure."""

    def test_cherry_pick_category_has_go_id_and_label(self):
        """CherryPickCategory has go_id and label fields."""
        field_names = {f.name for f in dataclass_fields(CherryPickCategory)}
        assert {"go_id", "label"} <= field_names

    def test_cherry_pick_category_construction(self):
        """CherryPickCategory can be constructed with go_id and label."""
        cat = CherryPickCategory(go_id="GO:0005739", label="Mitochondria")
        assert cat.go_id == "GO:0005739"
        assert cat.label == "Mitochondria"

    def test_cherry_pick_category_is_frozen(self):
        """CherryPickCategory is immutable (frozen)."""
        cat = CherryPickCategory(go_id="GO:0005739", label="Mitochondria")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cat.go_id = "GO:0000000"

    def test_tool_config_has_cherry_pick_categories_field(self):
        """ToolConfig has cherry_pick_categories field."""
        field_names = {f.name for f in dataclass_fields(ToolConfig)}
        assert "cherry_pick_categories" in field_names


class TestCherryPickConfigLoading:
    """Contract 4: cherry_pick section loading and validation."""

    def test_default_cherry_pick_categories_is_empty_list(self, empty_project_dir):
        """Default cherry_pick_categories is an empty list."""
        cfg = load_config(empty_project_dir)
        assert cfg.cherry_pick_categories == []

    def test_valid_cherry_pick_loaded(self, project_dir_with_config):
        """Valid cherry_pick entries are loaded as CherryPickCategory objects."""
        d = project_dir_with_config("""\
            cherry_pick:
              - go_id: "GO:0005739"
                label: "Mitochondria"
              - go_id: "GO:0005634"
                label: "Nucleus"
        """)
        cfg = load_config(d)
        assert len(cfg.cherry_pick_categories) == 2
        assert isinstance(cfg.cherry_pick_categories[0], CherryPickCategory)
        assert cfg.cherry_pick_categories[0].go_id == "GO:0005739"
        assert cfg.cherry_pick_categories[0].label == "Mitochondria"
        assert cfg.cherry_pick_categories[1].go_id == "GO:0005634"
        assert cfg.cherry_pick_categories[1].label == "Nucleus"

    def test_cherry_pick_not_a_list_raises_config_error(self, project_dir_with_config):
        """cherry_pick as a string instead of a list raises ConfigError."""
        d = project_dir_with_config("""\
            cherry_pick: "not_a_list"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_cherry_pick_entry_not_a_dict_raises_config_error(self, project_dir_with_config):
        """cherry_pick entry that is not a dict raises ConfigError."""
        d = project_dir_with_config("""\
            cherry_pick:
              - "just_a_string"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_cherry_pick_missing_go_id_raises_config_error(self, project_dir_with_config):
        """cherry_pick entry missing go_id raises ConfigError."""
        d = project_dir_with_config("""\
            cherry_pick:
              - label: "Mitochondria"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_cherry_pick_missing_label_raises_config_error(self, project_dir_with_config):
        """cherry_pick entry missing label raises ConfigError."""
        d = project_dir_with_config("""\
            cherry_pick:
              - go_id: "GO:0005739"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_cherry_pick_invalid_go_id_format_raises_config_error(self, project_dir_with_config):
        """cherry_pick go_id not matching GO:\\d{7} raises ConfigError."""
        d = project_dir_with_config("""\
            cherry_pick:
              - go_id: "INVALID"
                label: "Mitochondria"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_cherry_pick_go_id_too_few_digits_raises_config_error(self, project_dir_with_config):
        """cherry_pick go_id with too few digits raises ConfigError."""
        d = project_dir_with_config("""\
            cherry_pick:
              - go_id: "GO:123"
                label: "Mitochondria"
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_cherry_pick_empty_label_raises_config_error(self, project_dir_with_config):
        """cherry_pick entry with empty label raises ConfigError."""
        d = project_dir_with_config("""\
            cherry_pick:
              - go_id: "GO:0005739"
                label: ""
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_cherry_pick_whitespace_only_label_raises_config_error(self, project_dir_with_config):
        """cherry_pick entry with whitespace-only label raises ConfigError."""
        d = project_dir_with_config("""\
            cherry_pick:
              - go_id: "GO:0005739"
                label: "   "
        """)
        with pytest.raises(ConfigError):
            load_config(d)

    def test_cherry_pick_empty_list_is_valid(self, project_dir_with_config):
        """An empty cherry_pick list is valid and results in empty list."""
        d = project_dir_with_config("""\
            cherry_pick: []
        """)
        cfg = load_config(d)
        assert cfg.cherry_pick_categories == []


# ===========================================================================
# Invariant: fisher.fdr_threshold must be in (0, 1]
# ===========================================================================


class TestFisherFdrThresholdInvariant:
    """fisher.fdr_threshold range validation (0, 1]."""

    def test_fisher_fdr_threshold_zero_raises_config_error(self, project_dir_with_config):
        """fisher.fdr_threshold of 0 violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            fisher:
              fdr_threshold: 0.0
        """)
        with pytest.raises(ConfigError, match="fdr_threshold"):
            load_config(d)

    def test_fisher_fdr_threshold_negative_raises_config_error(self, project_dir_with_config):
        """fisher.fdr_threshold negative violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            fisher:
              fdr_threshold: -0.1
        """)
        with pytest.raises(ConfigError, match="fdr_threshold"):
            load_config(d)

    def test_fisher_fdr_threshold_above_one_raises_config_error(self, project_dir_with_config):
        """fisher.fdr_threshold above 1 violates (0, 1] invariant."""
        d = project_dir_with_config("""\
            fisher:
              fdr_threshold: 1.5
        """)
        with pytest.raises(ConfigError, match="fdr_threshold"):
            load_config(d)

    def test_fisher_fdr_threshold_exactly_one_is_valid(self, project_dir_with_config):
        """fisher.fdr_threshold of exactly 1.0 is within (0, 1]."""
        d = project_dir_with_config("""\
            fisher:
              fdr_threshold: 1.0
        """)
        cfg = load_config(d)
        assert cfg.fisher.fdr_threshold == 1.0

    def test_fisher_fdr_threshold_valid_value(self, project_dir_with_config):
        """fisher.fdr_threshold of 0.1 is valid."""
        d = project_dir_with_config("""\
            fisher:
              fdr_threshold: 0.1
        """)
        cfg = load_config(d)
        assert cfg.fisher.fdr_threshold == 0.1
