"""
Comprehensive test suite for Unit 10 -- Orchestration.

Tests verify all behavioral contracts, invariants, error conditions,
and signatures specified in the Unit 10 blueprint.

Synthetic Data Assumptions
--------------------------
DATA ASSUMPTION: Project directory structure follows the convention:
    <project_dir>/data/ -- must exist, contains GSEA data
    <project_dir>/output/ -- created automatically if absent
    <project_dir>/cache/ -- created automatically if absent

DATA ASSUMPTION: Category mapping file is a plain text file whose
    existence is the only thing that matters for orchestration tests.

DATA ASSUMPTION: All upstream units (1-9) are mocked. Only the
    orchestration wiring logic is tested.

DATA ASSUMPTION: CohortData mock uses synthetic mutant_ids like
    'mutant_1', 'mutant_2', etc. with n=3 as default count.

DATA ASSUMPTION: ToolConfig mock uses default-like values matching
    the ToolConfig dataclass structure.

DATA ASSUMPTION: The stub's main() resolves project_dir as
    Path(__file__).resolve().parent.parent.parent, so __file__
    is patched to <tmp_path>/a/b/stub.py to get tmp_path as
    the resolved project directory.

DATA ASSUMPTION: Ontology path uses cherry_pick_categories from
    config with CherryPickCategory(go_id, label) entries.

DATA ASSUMPTION: When both config cherry_pick_categories and CLI
    mapping file are provided, config takes precedence with a
    warning to stderr.
"""

import argparse
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from gsea_tool.scripts.svp_launcher import (
    build_argument_parser,
    resolve_paths,
    main,
)


# ---------------------------------------------------------------------------
# Helpers: mock object factories
# ---------------------------------------------------------------------------

def _make_mock_cohort(n_mutants=3):
    """Create a mock CohortData with the given number of mutants."""
    cohort = MagicMock()
    cohort.mutant_ids = [f"mutant_{i}" for i in range(1, n_mutants + 1)]
    return cohort


def _make_mock_config(clustering_enabled=True, cherry_pick_categories=None):
    """Create a mock ToolConfig with default-like values.

    Args:
        clustering_enabled: Whether GO clustering is enabled.
        cherry_pick_categories: List of mock CherryPickCategory objects,
            or None/empty list for no ontology-based cherry-picking.
    """
    config = MagicMock()
    config.dot_plot.fdr_threshold = 0.05
    config.dot_plot.top_n = 20
    config.dot_plot.n_groups = 4
    config.dot_plot.random_seed = 42
    config.plot_appearance.dpi = 300
    config.plot_appearance.font_family = "Arial"
    config.clustering.enabled = clustering_enabled
    config.clustering.go_obo_url = "http://example.com/go.obo"
    if cherry_pick_categories is None:
        cherry_pick_categories = []
    config.cherry_pick_categories = cherry_pick_categories
    return config


def _make_mock_dot_plot_result(stem="figure2_unbiased"):
    """Create a mock DotPlotResult."""
    result = MagicMock()
    result.n_terms_displayed = 10
    result.n_categories = 3
    result.n_mutants = 3
    result.pdf_path = Path(f"/tmp/output/{stem}.pdf")
    result.png_path = Path(f"/tmp/output/{stem}.png")
    result.svg_path = Path(f"/tmp/output/{stem}.svg")
    return result


def _make_mock_fisher_result():
    """Create a mock FisherResult."""
    result = MagicMock()
    result.n_mutants = 3
    return result


def _make_mock_clustering_result():
    """Create a mock ClusteringResult."""
    result = MagicMock()
    result.n_clusters = 5
    return result


def _make_mock_bar_plot_result():
    """Create a mock BarPlotResult."""
    result = MagicMock()
    result.n_bars = 15
    result.pdf_path = Path("/tmp/output/figure3_meta_analysis.pdf")
    result.png_path = Path("/tmp/output/figure3_meta_analysis.png")
    result.svg_path = Path("/tmp/output/figure3_meta_analysis.svg")
    return result


def _make_mock_unbiased_stats():
    """Create a mock UnbiasedSelectionStats."""
    stats = MagicMock()
    stats.total_significant_terms = 50
    stats.terms_selected = 20
    return stats


def _setup_main_env(tmp_path, monkeypatch, mapping_file_arg=None):
    """Set up monkeypatching for main() tests.

    Creates data/ dir, patches __file__ and sys.argv.
    If mapping_file_arg is not None, creates a mapping file and adds it to argv.

    Returns the mapping file Path (or None).
    """
    import gsea_tool.scripts.svp_launcher as stub_module

    (tmp_path / "data").mkdir(exist_ok=True)

    # __file__ -> tmp_path/a/b/stub.py so project_dir = tmp_path
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))

    argv = ["stub.py"]
    mapping_path = None
    if mapping_file_arg is not None:
        mapping_path = tmp_path / "mapping.txt"
        mapping_path.write_text("category\tterm\n")
        argv.append(str(mapping_path))

    monkeypatch.setattr(sys, "argv", argv)
    return mapping_path


def _build_patches(mocks, mapping_provided=False):
    """Build a dict of patch context managers for all upstream units."""
    patches = {}

    patches["load_config"] = patch(
        "gsea_tool.configuration.load_config", return_value=mocks["config"]
    )
    patches["ingest_data"] = patch(
        "gsea_tool.data_ingestion.ingest_data", return_value=mocks["cohort"]
    )
    patches["parse_category_mapping"] = patch(
        "gsea_tool.cherry_picked.parse_category_mapping",
        return_value=mocks["term_to_category"],
    )
    patches["select_cherry_picked_terms"] = patch(
        "gsea_tool.cherry_picked.select_cherry_picked_terms",
        return_value=mocks["cherry_groups"],
    )
    patches["resolve_categories_from_ontology"] = patch(
        "gsea_tool.cherry_picked.resolve_categories_from_ontology",
        return_value=mocks["term_to_category"],
    )
    patches["select_unbiased_terms"] = patch(
        "gsea_tool.unbiased.select_unbiased_terms",
        return_value=(mocks["unbiased_groups"], mocks["unbiased_stats"]),
    )

    if mapping_provided:
        patches["render_dot_plot"] = patch(
            "gsea_tool.dot_plot.render_dot_plot",
            side_effect=[mocks["fig1_result"], mocks["fig2_result"]],
        )
    else:
        patches["render_dot_plot"] = patch(
            "gsea_tool.dot_plot.render_dot_plot",
            return_value=mocks["fig2_result"],
        )

    patches["run_fisher_analysis"] = patch(
        "gsea_tool.meta_analysis.run_fisher_analysis",
        return_value=mocks["fisher_result"],
    )
    patches["run_semantic_clustering"] = patch(
        "gsea_tool.go_clustering.run_semantic_clustering",
        return_value=mocks["clustering_result"],
    )
    patches["download_or_load_obo"] = patch(
        "gsea_tool.go_clustering.download_or_load_obo",
        return_value=Path("/fake/go.obo"),
    )
    patches["render_bar_plot"] = patch(
        "gsea_tool.bar_plot.render_bar_plot",
        return_value=mocks["bar_result"],
    )
    patches["generate_notes"] = patch(
        "gsea_tool.notes_generation.generate_notes",
        return_value=Path("/tmp/output/notes.md"),
    )
    patches["NotesInput"] = patch(
        "gsea_tool.notes_generation.NotesInput",
        return_value=MagicMock(),
    )

    return patches


def _make_all_mocks(clustering_enabled=True, cherry_pick_categories=None):
    """Create a complete set of mock objects for main() testing."""
    return {
        "cohort": _make_mock_cohort(),
        "config": _make_mock_config(
            clustering_enabled=clustering_enabled,
            cherry_pick_categories=cherry_pick_categories,
        ),
        "fig1_result": _make_mock_dot_plot_result(stem="figure1_cherry_picked"),
        "fig2_result": _make_mock_dot_plot_result(stem="figure2_unbiased"),
        "fisher_result": _make_mock_fisher_result(),
        "clustering_result": _make_mock_clustering_result(),
        "bar_result": _make_mock_bar_plot_result(),
        "unbiased_stats": _make_mock_unbiased_stats(),
        "unbiased_groups": [MagicMock()],
        "cherry_groups": [MagicMock()],
        "term_to_category": {"TERM_A": "Category1"},
    }


def _run_main_with_patches(patches):
    """Enter all patch context managers, run main(), return entered mocks."""
    entered = {}
    for name, p in patches.items():
        entered[name] = p.start()
    try:
        main()
    finally:
        for p in patches.values():
            p.stop()
    return entered


# ===========================================================================
# Signature tests
# ===========================================================================


class TestBuildArgumentParserSignature:
    """Verify build_argument_parser signature matches the blueprint."""

    def test_build_argument_parser_takes_no_parameters(self):
        """build_argument_parser accepts no arguments."""
        sig = inspect.signature(build_argument_parser)
        assert len(sig.parameters) == 0

    def test_build_argument_parser_returns_argument_parser(self):
        """build_argument_parser return annotation is argparse.ArgumentParser."""
        sig = inspect.signature(build_argument_parser)
        assert sig.return_annotation is argparse.ArgumentParser


class TestResolvePathsSignature:
    """Verify resolve_paths signature matches the blueprint."""

    def test_resolve_paths_parameter_names(self):
        """resolve_paths has parameters (project_dir, mapping_file)."""
        sig = inspect.signature(resolve_paths)
        assert list(sig.parameters.keys()) == ["project_dir", "mapping_file"]

    def test_resolve_paths_project_dir_annotated_as_path(self):
        """resolve_paths project_dir parameter is annotated as Path."""
        sig = inspect.signature(resolve_paths)
        assert sig.parameters["project_dir"].annotation is Path

    def test_resolve_paths_has_return_annotation(self):
        """resolve_paths has a return type annotation."""
        sig = inspect.signature(resolve_paths)
        assert sig.return_annotation is not inspect.Parameter.empty


class TestMainSignature:
    """Verify main() signature matches the blueprint."""

    def test_main_takes_no_parameters(self):
        """main accepts no arguments."""
        sig = inspect.signature(main)
        assert len(sig.parameters) == 0

    def test_main_returns_none(self):
        """main return annotation is None."""
        sig = inspect.signature(main)
        assert sig.return_annotation is None


# ===========================================================================
# build_argument_parser behavioral tests
# ===========================================================================


class TestBuildArgumentParserBehavior:
    """Contract 1: No required CLI arguments.
    Contract 2: No parameter override CLI flags."""

    def test_returns_argparse_instance(self):
        """build_argument_parser returns an ArgumentParser instance."""
        parser = build_argument_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_no_arguments_parses_successfully(self):
        """Contract 1: Tool has no required CLI arguments."""
        parser = build_argument_parser()
        args = parser.parse_args([])
        assert args is not None

    def test_no_arguments_gives_mapping_file_none(self):
        """Contract 1: Without mapping file, mapping_file attribute is None."""
        parser = build_argument_parser()
        args = parser.parse_args([])
        assert args.mapping_file is None

    def test_single_positional_argument_accepted(self):
        """Contract 1: One optional positional argument for mapping file."""
        parser = build_argument_parser()
        args = parser.parse_args(["/path/to/mapping.tsv"])
        assert args.mapping_file == "/path/to/mapping.tsv"

    def test_two_positional_arguments_rejected(self):
        """Only zero or one positional arguments are accepted."""
        parser = build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["file1.txt", "file2.txt"])

    def test_no_optional_flags_beyond_help(self):
        """Contract 2: No parameter override CLI flags (only --help)."""
        parser = build_argument_parser()
        non_help_flags = [
            opt
            for action in parser._actions
            for opt in action.option_strings
            if opt not in ("-h", "--help")
        ]
        assert non_help_flags == [], (
            f"Expected no flags beyond --help, found: {non_help_flags}"
        )

    def test_unknown_flag_rejected(self):
        """Unknown flags are rejected by the parser."""
        parser = build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--threshold", "0.01"])

    def test_mapping_file_with_spaces_in_path(self):
        """Parser handles file paths containing spaces."""
        parser = build_argument_parser()
        args = parser.parse_args(["/path/with spaces/file.txt"])
        assert args.mapping_file == "/path/with spaces/file.txt"

    def test_mapping_file_relative_path(self):
        """Parser accepts relative paths."""
        parser = build_argument_parser()
        args = parser.parse_args(["./data/mapping.tsv"])
        assert args.mapping_file == "./data/mapping.tsv"


# ===========================================================================
# resolve_paths behavioral tests
# ===========================================================================


class TestResolvePathsReturnStructure:
    """Verify resolve_paths return value structure."""

    def test_returns_four_element_tuple(self, tmp_path):
        """resolve_paths returns a 4-tuple."""
        (tmp_path / "data").mkdir()
        result = resolve_paths(tmp_path, None)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_first_three_elements_are_paths(self, tmp_path):
        """First three elements (data_dir, output_dir, cache_dir) are Path."""
        (tmp_path / "data").mkdir()
        data_dir, output_dir, cache_dir, _ = resolve_paths(tmp_path, None)
        assert isinstance(data_dir, Path)
        assert isinstance(output_dir, Path)
        assert isinstance(cache_dir, Path)

    def test_fourth_element_none_without_mapping(self, tmp_path):
        """Fourth element is None when no mapping file specified."""
        (tmp_path / "data").mkdir()
        _, _, _, mapping = resolve_paths(tmp_path, None)
        assert mapping is None

    def test_fourth_element_is_path_with_mapping(self, tmp_path):
        """Fourth element is a Path when mapping file exists."""
        (tmp_path / "data").mkdir()
        mf = tmp_path / "mapping.txt"
        mf.write_text("data\n")
        _, _, _, mapping = resolve_paths(tmp_path, str(mf))
        assert isinstance(mapping, Path)


class TestResolvePathsDataDir:
    """Contract 4: data_dir = <project_dir>/data/, must exist."""

    def test_data_dir_equals_project_dir_slash_data(self, tmp_path):
        """Contract 4: data_dir is <project_dir>/data/."""
        (tmp_path / "data").mkdir()
        data_dir, _, _, _ = resolve_paths(tmp_path, None)
        assert data_dir == tmp_path / "data"

    def test_missing_data_dir_raises_file_not_found_error(self, tmp_path):
        """Error: FileNotFoundError when data/ does not exist."""
        with pytest.raises(FileNotFoundError):
            resolve_paths(tmp_path, None)

    def test_data_as_file_not_directory_raises_error(self, tmp_path):
        """If 'data' is a file (not directory), raises FileNotFoundError."""
        (tmp_path / "data").write_text("not a directory")
        with pytest.raises(FileNotFoundError):
            resolve_paths(tmp_path, None)


class TestResolvePathsOutputDir:
    """Contract 5: output_dir = <project_dir>/output/, created if absent."""

    def test_output_dir_equals_project_dir_slash_output(self, tmp_path):
        """Contract 5: output_dir is <project_dir>/output/."""
        (tmp_path / "data").mkdir()
        _, output_dir, _, _ = resolve_paths(tmp_path, None)
        assert output_dir == tmp_path / "output"

    def test_output_dir_created_if_absent(self, tmp_path):
        """Contract 5: output/ is created automatically if it does not exist."""
        (tmp_path / "data").mkdir()
        assert not (tmp_path / "output").exists()
        resolve_paths(tmp_path, None)
        assert (tmp_path / "output").is_dir()

    def test_output_dir_already_exists_no_error(self, tmp_path):
        """If output/ already exists, no error is raised."""
        (tmp_path / "data").mkdir()
        (tmp_path / "output").mkdir()
        _, output_dir, _, _ = resolve_paths(tmp_path, None)
        assert output_dir == tmp_path / "output"


class TestResolvePathsCacheDir:
    """Contract 6: cache_dir = <project_dir>/cache/, created if absent."""

    def test_cache_dir_equals_project_dir_slash_cache(self, tmp_path):
        """Contract 6: cache_dir is <project_dir>/cache/."""
        (tmp_path / "data").mkdir()
        _, _, cache_dir, _ = resolve_paths(tmp_path, None)
        assert cache_dir == tmp_path / "cache"

    def test_cache_dir_created_if_absent(self, tmp_path):
        """Contract 6: cache/ is created automatically if it does not exist."""
        (tmp_path / "data").mkdir()
        assert not (tmp_path / "cache").exists()
        resolve_paths(tmp_path, None)
        assert (tmp_path / "cache").is_dir()

    def test_cache_dir_already_exists_no_error(self, tmp_path):
        """If cache/ already exists, no error is raised."""
        (tmp_path / "data").mkdir()
        (tmp_path / "cache").mkdir()
        _, _, cache_dir, _ = resolve_paths(tmp_path, None)
        assert cache_dir == tmp_path / "cache"


class TestResolvePathsMappingFile:
    """Error condition: FileNotFoundError when mapping file does not exist."""

    def test_nonexistent_mapping_file_raises_file_not_found_error(self, tmp_path):
        """Error: FileNotFoundError when specified mapping file does not exist."""
        (tmp_path / "data").mkdir()
        with pytest.raises(FileNotFoundError):
            resolve_paths(tmp_path, str(tmp_path / "nonexistent.txt"))

    def test_mapping_file_resolved_to_correct_path(self, tmp_path):
        """Mapping file path is resolved correctly to a Path object."""
        (tmp_path / "data").mkdir()
        mf = tmp_path / "categories.tsv"
        mf.write_text("data\n")
        _, _, _, mapping = resolve_paths(tmp_path, str(mf))
        assert mapping == mf or str(mapping) == str(mf)

    def test_mapping_file_outside_project_dir(self, tmp_path):
        """Mapping file may be an absolute path outside the project directory."""
        (tmp_path / "data").mkdir()
        external = tmp_path / "elsewhere"
        external.mkdir()
        mf = external / "mapping.tsv"
        mf.write_text("data\n")
        _, _, _, mapping = resolve_paths(tmp_path, str(mf))
        assert mapping is not None

    def test_missing_data_dir_error_precedes_mapping_check(self, tmp_path):
        """FileNotFoundError for missing data/ takes precedence over mapping file check."""
        mf = tmp_path / "mapping.txt"
        mf.write_text("content")
        # data/ does not exist
        with pytest.raises(FileNotFoundError):
            resolve_paths(tmp_path, str(mf))

    def test_both_output_and_cache_created_in_single_call(self, tmp_path):
        """Both output/ and cache/ are created in a single resolve_paths call."""
        (tmp_path / "data").mkdir()
        assert not (tmp_path / "output").exists()
        assert not (tmp_path / "cache").exists()
        resolve_paths(tmp_path, None)
        assert (tmp_path / "output").is_dir()
        assert (tmp_path / "cache").is_dir()


# ===========================================================================
# main() error handling and exit behavior
# ===========================================================================


class TestMainExitOnError:
    """Contract 12: Any unit raising -> exit code 1 + stderr message."""

    def test_missing_data_dir_exits_code_1(self, tmp_path, monkeypatch):
        """main exits with code 1 when data/ directory is missing."""
        import gsea_tool.scripts.svp_launcher as stub_module
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py"])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_missing_data_dir_prints_to_stderr(self, tmp_path, monkeypatch, capsys):
        """main prints a descriptive error to stderr when data/ is missing."""
        import gsea_tool.scripts.svp_launcher as stub_module
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py"])

        with pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert len(captured.err) > 0

    def test_config_error_exits_code_1(self, tmp_path, monkeypatch):
        """Contract 7: When load_config raises, main exits with code 1."""
        _setup_main_env(tmp_path, monkeypatch)

        with patch("gsea_tool.configuration.load_config", side_effect=Exception("invalid config")):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_config_error_halts_before_data_ingestion(self, tmp_path, monkeypatch):
        """Contract 7: Config error prevents data ingestion from running."""
        _setup_main_env(tmp_path, monkeypatch)

        with patch("gsea_tool.configuration.load_config", side_effect=Exception("bad")):
            with patch("gsea_tool.data_ingestion.ingest_data") as mock_ingest:
                with pytest.raises(SystemExit):
                    main()
                mock_ingest.assert_not_called()

    def test_ingestion_error_exits_code_1(self, tmp_path, monkeypatch):
        """Contract 8: When ingest_data raises, main exits with code 1."""
        _setup_main_env(tmp_path, monkeypatch)
        mock_config = _make_mock_config()

        with patch("gsea_tool.configuration.load_config", return_value=mock_config):
            with patch("gsea_tool.data_ingestion.ingest_data", side_effect=Exception("fewer than 2 mutants")):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_ingestion_error_prints_to_stderr(self, tmp_path, monkeypatch, capsys):
        """Contract 8: Ingestion error message appears on stderr."""
        _setup_main_env(tmp_path, monkeypatch)
        mock_config = _make_mock_config()

        with patch("gsea_tool.configuration.load_config", return_value=mock_config):
            with patch("gsea_tool.data_ingestion.ingest_data", side_effect=Exception("fewer than 2 mutants")):
                with pytest.raises(SystemExit):
                    main()

        captured = capsys.readouterr()
        assert "fewer than 2 mutants" in captured.err

    def test_fisher_error_exits_code_1(self, tmp_path, monkeypatch):
        """Contract 12: Fisher analysis failure exits with code 1."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        patches["run_fisher_analysis"] = patch(
            "gsea_tool.meta_analysis.run_fisher_analysis",
            side_effect=RuntimeError("Fisher failed"),
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_main_with_patches(patches)
        assert exc_info.value.code == 1

    def test_render_error_exits_code_1(self, tmp_path, monkeypatch):
        """Contract 12: Render failure exits with code 1."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        patches["render_dot_plot"] = patch(
            "gsea_tool.dot_plot.render_dot_plot",
            side_effect=ValueError("render error"),
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_main_with_patches(patches)
        assert exc_info.value.code == 1

    def test_notes_error_exits_code_1(self, tmp_path, monkeypatch):
        """Contract 12: Notes generation failure exits with code 1."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        patches["generate_notes"] = patch(
            "gsea_tool.notes_generation.generate_notes",
            side_effect=OSError("cannot write notes"),
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_main_with_patches(patches)
        assert exc_info.value.code == 1

    def test_error_message_appears_on_stderr(self, tmp_path, monkeypatch, capsys):
        """Contract 12: Error message from a failed unit appears on stderr."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        patches["run_fisher_analysis"] = patch(
            "gsea_tool.meta_analysis.run_fisher_analysis",
            side_effect=RuntimeError("chi-squared computation error"),
        )

        with pytest.raises(SystemExit):
            _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "chi-squared computation error" in captured.err

    def test_nonexistent_mapping_file_cli_arg_exits_code_1(self, tmp_path, monkeypatch):
        """Contract 12: main exits 1 when CLI mapping file does not exist."""
        import gsea_tool.scripts.svp_launcher as stub_module
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        (tmp_path / "data").mkdir()
        monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py", str(tmp_path / "nonexistent.txt")])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


# ===========================================================================
# Invocation order (Contract 9)
# ===========================================================================


class TestInvocationOrder:
    """Contract 9: The invocation order is:
    load_config -> ingest_data -> [cherry-pick] -> unbiased -> [fig1] -> fig2
    -> fisher -> [clustering] -> fig3 -> notes.
    """

    def test_invocation_order_without_mapping(self, tmp_path, monkeypatch):
        """Without mapping: config -> ingest -> unbiased -> fig2 -> fisher -> clustering -> fig3 -> notes."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)

        call_order = []

        def tracker(name, return_value=None):
            def fn(*args, **kwargs):
                call_order.append(name)
                return return_value
            return fn

        patches["load_config"] = patch(
            "gsea_tool.configuration.load_config",
            side_effect=tracker("load_config", mocks["config"]),
        )
        patches["ingest_data"] = patch(
            "gsea_tool.data_ingestion.ingest_data",
            side_effect=tracker("ingest_data", mocks["cohort"]),
        )
        patches["select_unbiased_terms"] = patch(
            "gsea_tool.unbiased.select_unbiased_terms",
            side_effect=tracker("select_unbiased_terms",
                                (mocks["unbiased_groups"], mocks["unbiased_stats"])),
        )
        patches["render_dot_plot"] = patch(
            "gsea_tool.dot_plot.render_dot_plot",
            side_effect=tracker("render_dot_plot", mocks["fig2_result"]),
        )
        patches["run_fisher_analysis"] = patch(
            "gsea_tool.meta_analysis.run_fisher_analysis",
            side_effect=tracker("run_fisher_analysis", mocks["fisher_result"]),
        )
        patches["run_semantic_clustering"] = patch(
            "gsea_tool.go_clustering.run_semantic_clustering",
            side_effect=tracker("run_semantic_clustering", mocks["clustering_result"]),
        )
        patches["render_bar_plot"] = patch(
            "gsea_tool.bar_plot.render_bar_plot",
            side_effect=tracker("render_bar_plot", mocks["bar_result"]),
        )
        patches["generate_notes"] = patch(
            "gsea_tool.notes_generation.generate_notes",
            side_effect=tracker("generate_notes", None),
        )

        _run_main_with_patches(patches)

        expected = [
            "load_config",
            "ingest_data",
            "select_unbiased_terms",
            "render_dot_plot",          # fig2
            "run_fisher_analysis",
            "run_semantic_clustering",
            "render_bar_plot",          # fig3
            "generate_notes",
        ]
        assert call_order == expected

    def test_invocation_order_with_mapping(self, tmp_path, monkeypatch):
        """With mapping: includes cherry-pick steps and fig1."""
        _setup_main_env(tmp_path, monkeypatch, mapping_file_arg="mapping.txt")
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=True)

        call_order = []

        def tracker(name, return_value=None):
            def fn(*args, **kwargs):
                call_order.append(name)
                return return_value
            return fn

        patches["load_config"] = patch(
            "gsea_tool.configuration.load_config",
            side_effect=tracker("load_config", mocks["config"]),
        )
        patches["ingest_data"] = patch(
            "gsea_tool.data_ingestion.ingest_data",
            side_effect=tracker("ingest_data", mocks["cohort"]),
        )
        patches["parse_category_mapping"] = patch(
            "gsea_tool.cherry_picked.parse_category_mapping",
            side_effect=tracker("parse_category_mapping", mocks["term_to_category"]),
        )
        patches["select_cherry_picked_terms"] = patch(
            "gsea_tool.cherry_picked.select_cherry_picked_terms",
            side_effect=tracker("select_cherry_picked_terms", mocks["cherry_groups"]),
        )
        patches["select_unbiased_terms"] = patch(
            "gsea_tool.unbiased.select_unbiased_terms",
            side_effect=tracker("select_unbiased_terms",
                                (mocks["unbiased_groups"], mocks["unbiased_stats"])),
        )

        fig_count = [0]
        def track_render(*args, **kwargs):
            call_order.append("render_dot_plot")
            fig_count[0] += 1
            return mocks["fig1_result"] if fig_count[0] == 1 else mocks["fig2_result"]

        patches["render_dot_plot"] = patch(
            "gsea_tool.dot_plot.render_dot_plot",
            side_effect=track_render,
        )
        patches["run_fisher_analysis"] = patch(
            "gsea_tool.meta_analysis.run_fisher_analysis",
            side_effect=tracker("run_fisher_analysis", mocks["fisher_result"]),
        )
        patches["run_semantic_clustering"] = patch(
            "gsea_tool.go_clustering.run_semantic_clustering",
            side_effect=tracker("run_semantic_clustering", mocks["clustering_result"]),
        )
        patches["render_bar_plot"] = patch(
            "gsea_tool.bar_plot.render_bar_plot",
            side_effect=tracker("render_bar_plot", mocks["bar_result"]),
        )
        patches["generate_notes"] = patch(
            "gsea_tool.notes_generation.generate_notes",
            side_effect=tracker("generate_notes", None),
        )

        _run_main_with_patches(patches)

        expected = [
            "load_config",
            "ingest_data",
            "parse_category_mapping",
            "select_cherry_picked_terms",
            "render_dot_plot",          # fig1
            "select_unbiased_terms",
            "render_dot_plot",          # fig2
            "run_fisher_analysis",
            "run_semantic_clustering",
            "render_bar_plot",          # fig3
            "generate_notes",
        ]
        assert call_order == expected


# ===========================================================================
# Contract 10: Figure 1 conditional behavior
# ===========================================================================


class TestFigure1ConditionalBehavior:
    """Contract 10: Figure 1 produced when (a) config.cherry_pick_categories
    is non-empty (ontology path) or (b) mapping file provided (TSV path).
    Config takes precedence over CLI mapping file with warning to stderr.
    If neither, Figure 1 is skipped."""

    def test_no_mapping_skips_cherry_pick_and_fig1(self, tmp_path, monkeypatch):
        """Without mapping file and no config categories, no Figure 1."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        entered["parse_category_mapping"].assert_not_called()
        entered["select_cherry_picked_terms"].assert_not_called()
        assert entered["render_dot_plot"].call_count == 1  # only fig2

    def test_with_mapping_calls_cherry_pick_and_fig1(self, tmp_path, monkeypatch):
        """With mapping file, cherry-pick and fig1 render are called."""
        _setup_main_env(tmp_path, monkeypatch, mapping_file_arg="mapping.txt")
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        entered["parse_category_mapping"].assert_called_once()
        entered["select_cherry_picked_terms"].assert_called_once()
        assert entered["render_dot_plot"].call_count == 2  # fig1 + fig2


# ===========================================================================
# Contract 11: Figure 3 always produced; clustering conditional
# ===========================================================================


class TestFigure3AlwaysProduced:
    """Contract 11: Figure 3 is always produced. When clustering
    disabled, bar plot shows unclustered top-N terms."""

    def test_fig3_rendered_with_clustering_enabled(self, tmp_path, monkeypatch):
        """Figure 3 rendered when clustering is enabled."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks(clustering_enabled=True)
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        entered["render_bar_plot"].assert_called_once()
        entered["run_semantic_clustering"].assert_called_once()

    def test_fig3_rendered_with_clustering_disabled(self, tmp_path, monkeypatch):
        """Figure 3 still rendered when clustering is disabled."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks(clustering_enabled=False)
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        entered["render_bar_plot"].assert_called_once()
        entered["run_semantic_clustering"].assert_not_called()

    def test_clustering_disabled_passes_none_clustering_result(self, tmp_path, monkeypatch):
        """When clustering disabled, render_bar_plot receives clustering_result=None."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks(clustering_enabled=False)
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        bar_call = entered["render_bar_plot"].call_args
        # Check clustering_result is None (could be positional or keyword)
        if "clustering_result" in bar_call.kwargs:
            assert bar_call.kwargs["clustering_result"] is None
        else:
            # Second positional argument
            assert bar_call.args[1] is None


# ===========================================================================
# Contract 13: Success summary to stdout
# ===========================================================================


class TestSuccessSummary:
    """Contract 13: Brief summary to stdout on success includes
    number of mutants, figures produced, output file paths."""

    def test_success_prints_mutant_count(self, tmp_path, monkeypatch, capsys):
        """Success summary includes number of mutants processed."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "3 mutants" in captured.out

    def test_success_without_mapping_lists_fig2_and_fig3(self, tmp_path, monkeypatch, capsys):
        """Without mapping, summary lists figure2 and figure3 but not figure1."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure2_unbiased" in captured.out
        assert "figure3_meta_analysis" in captured.out
        assert "figure1_cherry_picked" not in captured.out

    def test_success_with_mapping_lists_all_three_figures(self, tmp_path, monkeypatch, capsys):
        """With mapping, summary lists figure1, figure2, and figure3."""
        _setup_main_env(tmp_path, monkeypatch, mapping_file_arg="mapping.txt")
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=True)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure1_cherry_picked" in captured.out
        assert "figure2_unbiased" in captured.out
        assert "figure3_meta_analysis" in captured.out

    def test_success_prints_output_directory(self, tmp_path, monkeypatch, capsys):
        """Success summary includes the output directory path."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "output" in captured.out.lower()

    def test_success_no_stderr(self, tmp_path, monkeypatch, capsys):
        """On success, nothing is printed to stderr."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert captured.err == ""


# ===========================================================================
# Contract 14: Figure output file stem names
# ===========================================================================


class TestFigureStemNames:
    """Contract 14: Figure stems are figure1_cherry_picked,
    figure2_unbiased, figure3_meta_analysis."""

    def test_fig2_stem_passed_correctly(self, tmp_path, monkeypatch):
        """render_dot_plot for fig2 receives output_stem='figure2_unbiased'."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["render_dot_plot"].call_args[1]
        assert call_kwargs["output_stem"] == "figure2_unbiased"

    def test_fig1_stem_passed_correctly(self, tmp_path, monkeypatch):
        """render_dot_plot for fig1 receives output_stem='figure1_cherry_picked'."""
        _setup_main_env(tmp_path, monkeypatch, mapping_file_arg="mapping.txt")
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        first_call_kwargs = entered["render_dot_plot"].call_args_list[0][1]
        assert first_call_kwargs["output_stem"] == "figure1_cherry_picked"

    def test_fig3_stem_passed_correctly(self, tmp_path, monkeypatch):
        """render_bar_plot receives output_stem='figure3_meta_analysis'."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["render_bar_plot"].call_args[1]
        assert call_kwargs["output_stem"] == "figure3_meta_analysis"


# ===========================================================================
# Contract 15: Always-produced output files
# ===========================================================================


class TestAlwaysProducedFiles:
    """Contract 15: Always produced in output/:
    figure2.{pdf,png,svg}, figure3.{pdf,png,svg},
    pvalue_matrix.tsv, fisher_combined_pvalues.tsv, notes.md."""

    def test_stdout_mentions_pvalue_matrix(self, tmp_path, monkeypatch, capsys):
        """Success summary mentions pvalue_matrix.tsv."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "pvalue_matrix.tsv" in captured.out

    def test_stdout_mentions_fisher_combined(self, tmp_path, monkeypatch, capsys):
        """Success summary mentions fisher_combined_pvalues.tsv."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "fisher_combined_pvalues.tsv" in captured.out

    def test_stdout_mentions_notes_md(self, tmp_path, monkeypatch, capsys):
        """Success summary mentions notes.md."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "notes.md" in captured.out

    def test_stdout_mentions_figure2_all_formats(self, tmp_path, monkeypatch, capsys):
        """Success summary mentions figure2 in pdf/png/svg."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure2_unbiased.pdf" in captured.out
        assert "figure2_unbiased.png" in captured.out
        assert "figure2_unbiased.svg" in captured.out

    def test_stdout_mentions_figure3_all_formats(self, tmp_path, monkeypatch, capsys):
        """Success summary mentions figure3 in pdf/png/svg."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure3_meta_analysis.pdf" in captured.out
        assert "figure3_meta_analysis.png" in captured.out
        assert "figure3_meta_analysis.svg" in captured.out


# ===========================================================================
# Contract 16: Mapping file adds figure1 files
# ===========================================================================


class TestMappingAddsF1Files:
    """Contract 16: When mapping file provided,
    figure1_cherry_picked.{pdf,png,svg} additionally produced."""

    def test_with_mapping_stdout_includes_figure1_formats(self, tmp_path, monkeypatch, capsys):
        """With mapping file, stdout lists figure1 in all formats."""
        _setup_main_env(tmp_path, monkeypatch, mapping_file_arg="mapping.txt")
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=True)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure1_cherry_picked.pdf" in captured.out
        assert "figure1_cherry_picked.png" in captured.out
        assert "figure1_cherry_picked.svg" in captured.out

    def test_without_mapping_stdout_omits_figure1(self, tmp_path, monkeypatch, capsys):
        """Without mapping file, stdout does not list figure1."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure1_cherry_picked" not in captured.out


# ===========================================================================
# Contract 3: Project directory resolved from script location
# ===========================================================================


class TestProjectDirResolution:
    """Contract 3: Project directory resolved as directory containing the script."""

    def test_project_dir_derived_from_file_passed_to_load_config(self, tmp_path, monkeypatch):
        """main() passes the correct project_dir to load_config."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        called_project_dir = entered["load_config"].call_args[0][0]
        assert called_project_dir == tmp_path


# ===========================================================================
# Contract 6: cache_dir passed to Unit 7
# ===========================================================================


class TestCacheDirPassedToUnit7:
    """Contract 6: cache_dir passed to Unit 7 for OBO/GAF file caching."""

    def test_cache_dir_passed_to_clustering(self, tmp_path, monkeypatch):
        """run_semantic_clustering receives cache_dir as an argument."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks(clustering_enabled=True)
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["run_semantic_clustering"].call_args[1]
        assert "cache_dir" in call_kwargs
        assert str(call_kwargs["cache_dir"]).endswith("cache")


# ===========================================================================
# Config parameter forwarding
# ===========================================================================


class TestConfigParameterForwarding:
    """Verify that config parameters are correctly forwarded to downstream units."""

    def test_fdr_threshold_forwarded_to_render_dot_plot(self, tmp_path, monkeypatch):
        """dot_plot.fdr_threshold is forwarded to render_dot_plot."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["render_dot_plot"].call_args[1]
        assert call_kwargs["fdr_threshold"] == 0.05

    def test_top_n_forwarded_to_select_unbiased_terms(self, tmp_path, monkeypatch):
        """dot_plot.top_n is forwarded to select_unbiased_terms."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["select_unbiased_terms"].call_args[1]
        assert call_kwargs["top_n"] == 20

    def test_dpi_forwarded_to_render_dot_plot(self, tmp_path, monkeypatch):
        """plot_appearance.dpi is forwarded to render_dot_plot."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["render_dot_plot"].call_args[1]
        assert call_kwargs["dpi"] == 300

    def test_font_family_forwarded_to_render_dot_plot(self, tmp_path, monkeypatch):
        """plot_appearance.font_family is forwarded to render_dot_plot."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["render_dot_plot"].call_args[1]
        assert call_kwargs["font_family"] == "Arial"

    def test_fisher_config_forwarded_to_run_fisher_analysis(self, tmp_path, monkeypatch):
        """config.fisher is forwarded to run_fisher_analysis."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["run_fisher_analysis"].call_args[1]
        assert call_kwargs["config"] is mocks["config"].fisher

    def test_clustering_config_forwarded_to_run_semantic_clustering(self, tmp_path, monkeypatch):
        """config.clustering is forwarded to run_semantic_clustering."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks(clustering_enabled=True)
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["run_semantic_clustering"].call_args[1]
        assert call_kwargs["config"] is mocks["config"].clustering

    def test_output_dir_forwarded_to_render_dot_plot(self, tmp_path, monkeypatch):
        """output_dir is forwarded to render_dot_plot."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["render_dot_plot"].call_args[1]
        assert str(call_kwargs["output_dir"]).endswith("output")

    def test_output_dir_forwarded_to_run_fisher_analysis(self, tmp_path, monkeypatch):
        """output_dir is forwarded to run_fisher_analysis."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["run_fisher_analysis"].call_args[1]
        assert str(call_kwargs["output_dir"]).endswith("output")

    def test_output_dir_forwarded_to_render_bar_plot(self, tmp_path, monkeypatch):
        """output_dir is forwarded to render_bar_plot."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["render_bar_plot"].call_args[1]
        assert str(call_kwargs["output_dir"]).endswith("output")

    def test_clustering_enabled_forwarded_to_fisher(self, tmp_path, monkeypatch):
        """config.clustering.enabled is forwarded to run_fisher_analysis."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks(clustering_enabled=True)
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["run_fisher_analysis"].call_args[1]
        assert call_kwargs["clustering_enabled"] is True


# ===========================================================================
# NotesInput construction and generate_notes call
# ===========================================================================


class TestNotesGeneration:
    """Verify generate_notes is called with proper NotesInput and output_dir."""

    def test_generate_notes_called_once(self, tmp_path, monkeypatch):
        """generate_notes is called exactly once on success."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        entered["generate_notes"].assert_called_once()

    def test_notes_input_constructed(self, tmp_path, monkeypatch):
        """NotesInput is constructed before generate_notes is called."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        entered["NotesInput"].assert_called_once()

    def test_generate_notes_receives_output_dir(self, tmp_path, monkeypatch):
        """generate_notes receives output_dir as second argument."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_args = entered["generate_notes"].call_args
        # Second positional argument should be the output_dir
        output_dir_arg = call_args[0][1]
        assert str(output_dir_arg).endswith("output")


# ===========================================================================
# Contract 10: Ontology path (config.cherry_pick_categories non-empty)
# ===========================================================================


class TestFigure1OntologyPath:
    """Contract 10: When config.cherry_pick_categories is non-empty,
    Figure 1 is produced via the ontology path using download_or_load_obo
    and resolve_categories_from_ontology (not parse_category_mapping)."""

    def test_ontology_path_calls_download_or_load_obo(self, tmp_path, monkeypatch):
        """When config has cherry_pick_categories, download_or_load_obo is called."""
        _setup_main_env(tmp_path, monkeypatch)
        mock_category = MagicMock()
        mock_category.go_id = "GO:0008150"
        mock_category.label = "biological_process"
        mocks = _make_all_mocks(cherry_pick_categories=[mock_category])
        patches = _build_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        entered["download_or_load_obo"].assert_called_once()

    def test_ontology_path_calls_resolve_categories_from_ontology(self, tmp_path, monkeypatch):
        """When config has cherry_pick_categories, resolve_categories_from_ontology is called."""
        _setup_main_env(tmp_path, monkeypatch)
        mock_category = MagicMock()
        mock_category.go_id = "GO:0008150"
        mock_category.label = "biological_process"
        mocks = _make_all_mocks(cherry_pick_categories=[mock_category])
        patches = _build_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        entered["resolve_categories_from_ontology"].assert_called_once()

    def test_ontology_path_does_not_call_parse_category_mapping(self, tmp_path, monkeypatch):
        """When config has cherry_pick_categories, parse_category_mapping is NOT called."""
        _setup_main_env(tmp_path, monkeypatch)
        mock_category = MagicMock()
        mock_category.go_id = "GO:0008150"
        mock_category.label = "biological_process"
        mocks = _make_all_mocks(cherry_pick_categories=[mock_category])
        patches = _build_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        entered["parse_category_mapping"].assert_not_called()

    def test_ontology_path_produces_fig1(self, tmp_path, monkeypatch):
        """When config has cherry_pick_categories, fig1 render_dot_plot is called."""
        _setup_main_env(tmp_path, monkeypatch)
        mock_category = MagicMock()
        mock_category.go_id = "GO:0008150"
        mock_category.label = "biological_process"
        mocks = _make_all_mocks(cherry_pick_categories=[mock_category])
        patches = _build_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        # render_dot_plot called twice: fig1 + fig2
        assert entered["render_dot_plot"].call_count == 2


# ===========================================================================
# Contract 10: Both config categories and mapping file present
# ===========================================================================


class TestFigure1BothPathsPrecedence:
    """Contract 10: When both config.cherry_pick_categories is non-empty
    AND a mapping file is provided, config takes precedence with a
    warning printed to stderr."""

    def test_both_present_uses_ontology_path(self, tmp_path, monkeypatch):
        """Config categories take precedence: resolve_categories_from_ontology called,
        parse_category_mapping NOT called."""
        _setup_main_env(tmp_path, monkeypatch, mapping_file_arg="mapping.txt")
        mock_category = MagicMock()
        mock_category.go_id = "GO:0008150"
        mock_category.label = "biological_process"
        mocks = _make_all_mocks(cherry_pick_categories=[mock_category])
        patches = _build_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        entered["resolve_categories_from_ontology"].assert_called_once()
        entered["parse_category_mapping"].assert_not_called()

    def test_both_present_prints_warning_to_stderr(self, tmp_path, monkeypatch, capsys):
        """When both are present, a warning is printed to stderr."""
        _setup_main_env(tmp_path, monkeypatch, mapping_file_arg="mapping.txt")
        mock_category = MagicMock()
        mock_category.go_id = "GO:0008150"
        mock_category.label = "biological_process"
        mocks = _make_all_mocks(cherry_pick_categories=[mock_category])
        patches = _build_patches(mocks, mapping_provided=True)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "warning" in captured.err.lower() or "Warning" in captured.err


# ===========================================================================
# Contract 10: fig1_method forwarded to NotesInput
# ===========================================================================


class TestFig1MethodInNotesInput:
    """Contract 10/9: The fig1_method passed to NotesInput should be
    'ontology', 'tsv', or None depending on the path taken."""

    def test_fig1_method_none_when_no_fig1(self, tmp_path, monkeypatch):
        """When neither config categories nor mapping file, fig1_method is None."""
        _setup_main_env(tmp_path, monkeypatch)
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        notes_input_call = entered["NotesInput"].call_args
        assert notes_input_call[1].get("fig1_method") is None

    def test_fig1_method_tsv_with_mapping_file(self, tmp_path, monkeypatch):
        """When mapping file provided (no config categories), fig1_method is 'tsv'."""
        _setup_main_env(tmp_path, monkeypatch, mapping_file_arg="mapping.txt")
        mocks = _make_all_mocks()
        patches = _build_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        notes_input_call = entered["NotesInput"].call_args
        assert notes_input_call[1].get("fig1_method") == "tsv"

    def test_fig1_method_ontology_with_config_categories(self, tmp_path, monkeypatch):
        """When config.cherry_pick_categories is non-empty, fig1_method is 'ontology'."""
        _setup_main_env(tmp_path, monkeypatch)
        mock_category = MagicMock()
        mock_category.go_id = "GO:0008150"
        mock_category.label = "biological_process"
        mocks = _make_all_mocks(cherry_pick_categories=[mock_category])
        patches = _build_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        notes_input_call = entered["NotesInput"].call_args
        assert notes_input_call[1].get("fig1_method") == "ontology"
