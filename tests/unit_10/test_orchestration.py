"""
Test suite for Unit 10 -- Orchestration.

Tests verify all behavioral contracts, invariants, error conditions,
and signatures specified in the Unit 10 blueprint.

Synthetic Data Assumptions
--------------------------
DATA ASSUMPTION: Project directory structure follows the convention:
    <project_dir>/data/ -- must exist, contains GSEA data
    <project_dir>/output/ -- created automatically if absent
    <project_dir>/cache/ -- created automatically if absent
    <project_dir>/config.yaml -- optional configuration file

DATA ASSUMPTION: Category mapping file is a plain text file whose path
    is provided as a CLI argument. Its contents are irrelevant to
    orchestration path-resolution tests; only its existence matters.

DATA ASSUMPTION: Mapping file paths may contain spaces or special
    characters, which the CLI parser must handle correctly.

DATA ASSUMPTION: The CLI accepts zero or one positional arguments.
    Zero means no mapping file (no Figure 1). One means the mapping
    file path is provided (Figure 1 will be produced).
"""

import argparse
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from gsea_tool.scripts.svp_launcher import (
    build_argument_parser,
    resolve_paths,
    main,
)


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------


class TestSignatures:
    """Verify that function signatures match the blueprint exactly."""

    def test_build_argument_parser_no_parameters(self):
        """build_argument_parser takes no arguments."""
        sig = inspect.signature(build_argument_parser)
        params = list(sig.parameters.values())
        assert len(params) == 0, (
            f"build_argument_parser should take no parameters, got {len(params)}"
        )

    def test_build_argument_parser_return_type(self):
        """build_argument_parser returns argparse.ArgumentParser."""
        sig = inspect.signature(build_argument_parser)
        assert sig.return_annotation is argparse.ArgumentParser

    def test_resolve_paths_parameter_names(self):
        """resolve_paths has parameters (project_dir, mapping_file)."""
        sig = inspect.signature(resolve_paths)
        param_names = list(sig.parameters.keys())
        assert param_names == ["project_dir", "mapping_file"]

    def test_resolve_paths_project_dir_type(self):
        """resolve_paths project_dir parameter is annotated as Path."""
        sig = inspect.signature(resolve_paths)
        assert sig.parameters["project_dir"].annotation is Path

    def test_resolve_paths_return_annotation_present(self):
        """resolve_paths has a return type annotation (not empty)."""
        sig = inspect.signature(resolve_paths)
        assert sig.return_annotation is not inspect.Parameter.empty

    def test_main_no_parameters(self):
        """main takes no arguments."""
        sig = inspect.signature(main)
        params = list(sig.parameters.values())
        assert len(params) == 0

    def test_main_return_type_none(self):
        """main returns None."""
        sig = inspect.signature(main)
        assert sig.return_annotation is None


# ---------------------------------------------------------------------------
# build_argument_parser tests
# ---------------------------------------------------------------------------


class TestBuildArgumentParser:
    """Tests for the CLI argument parser builder.

    Contract 1: No required CLI arguments.
    Contract 2: No parameter override flags -- all tunable params via config.yaml.
    """

    def test_returns_argument_parser_instance(self):
        """build_argument_parser returns an argparse.ArgumentParser."""
        parser = build_argument_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_no_arguments_parses_successfully(self):
        """Contract 1: The tool has no required CLI arguments."""
        parser = build_argument_parser()
        # Should not raise
        args = parser.parse_args([])
        assert args is not None

    def test_no_arguments_mapping_file_is_none(self):
        """Contract 1: When no mapping file provided, its value is None."""
        parser = build_argument_parser()
        args = parser.parse_args([])
        assert args.mapping_file is None

    def test_one_positional_argument_accepted(self):
        """Contract 1: One optional positional argument (mapping file path) is accepted."""
        parser = build_argument_parser()
        args = parser.parse_args(["/path/to/mapping.txt"])
        assert args.mapping_file == "/path/to/mapping.txt"

    def test_mapping_file_attribute_name(self):
        """The parser stores the mapping file path in the 'mapping_file' attribute."""
        parser = build_argument_parser()
        args = parser.parse_args(["/some/file.txt"])
        assert hasattr(args, "mapping_file")
        assert args.mapping_file == "/some/file.txt"

    def test_two_positional_arguments_rejected(self):
        """Only zero or one positional arguments are accepted."""
        parser = build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["file1.txt", "file2.txt"])

    def test_no_optional_flags_besides_help(self):
        """Contract 2: No parameter override CLI flags.

        All tunable parameters are controlled via config.yaml.
        Only --help / -h should exist as optional flags.
        """
        parser = build_argument_parser()
        non_help_flags = []
        for action in parser._actions:
            if action.option_strings:
                for opt in action.option_strings:
                    if opt not in ("-h", "--help"):
                        non_help_flags.append(opt)
        assert non_help_flags == [], (
            f"Parser should have no flags besides --help, found: {non_help_flags}"
        )

    def test_mapping_file_path_with_spaces(self):
        """Parser handles file paths containing spaces."""
        # DATA ASSUMPTION: File paths with spaces are valid on the target OS
        parser = build_argument_parser()
        args = parser.parse_args(["/path/with spaces/mapping file.txt"])
        assert args.mapping_file == "/path/with spaces/mapping file.txt"

    def test_mapping_file_path_with_special_chars(self):
        """Parser handles file paths with hyphens, underscores, dots."""
        # DATA ASSUMPTION: Common special characters in filenames
        parser = build_argument_parser()
        args = parser.parse_args(["/data/my-mapping_v2.0.txt"])
        assert args.mapping_file == "/data/my-mapping_v2.0.txt"

    def test_mapping_file_relative_path(self):
        """Parser accepts relative paths."""
        parser = build_argument_parser()
        args = parser.parse_args(["./mapping.txt"])
        assert args.mapping_file == "./mapping.txt"

    def test_unknown_flag_rejected(self):
        """Unknown flags like --foo are rejected."""
        parser = build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--foo", "bar"])


# ---------------------------------------------------------------------------
# resolve_paths tests
# ---------------------------------------------------------------------------


class TestResolvePaths:
    """Tests for path resolution and validation.

    Contract 4: data_dir = <project_dir>/data/  (must exist)
    Contract 5: output_dir = <project_dir>/output/  (created if absent)
    Contract 6: cache_dir = <project_dir>/cache/   (created if absent)
    """

    # --- Return structure ---

    def test_returns_four_element_tuple(self, tmp_path):
        """resolve_paths returns a 4-tuple."""
        # DATA ASSUMPTION: Minimal project directory with data/ subfolder
        (tmp_path / "data").mkdir()
        result = resolve_paths(tmp_path, None)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_all_directory_returns_are_path_objects(self, tmp_path):
        """First three elements are Path objects."""
        (tmp_path / "data").mkdir()
        result = resolve_paths(tmp_path, None)
        assert isinstance(result[0], Path), "data_dir should be Path"
        assert isinstance(result[1], Path), "output_dir should be Path"
        assert isinstance(result[2], Path), "cache_dir should be Path"

    def test_fourth_element_none_when_no_mapping(self, tmp_path):
        """Fourth element is None when mapping_file is None."""
        (tmp_path / "data").mkdir()
        result = resolve_paths(tmp_path, None)
        assert result[3] is None

    def test_fourth_element_path_when_mapping_provided(self, tmp_path):
        """Fourth element is a Path when mapping_file exists."""
        (tmp_path / "data").mkdir()
        # DATA ASSUMPTION: Mapping file is a simple text file
        mapping = tmp_path / "mapping.txt"
        mapping.write_text("category1\tcategory2\n")
        result = resolve_paths(tmp_path, str(mapping))
        assert isinstance(result[3], Path)

    # --- data_dir ---

    def test_data_dir_is_project_dir_data(self, tmp_path):
        """Contract 4: data_dir is always <project_dir>/data/."""
        (tmp_path / "data").mkdir()
        result = resolve_paths(tmp_path, None)
        assert result[0] == tmp_path / "data"

    # --- output_dir ---

    def test_output_dir_is_project_dir_output(self, tmp_path):
        """Contract 5: output_dir is always <project_dir>/output/."""
        (tmp_path / "data").mkdir()
        result = resolve_paths(tmp_path, None)
        assert result[1] == tmp_path / "output"

    def test_output_dir_created_if_absent(self, tmp_path):
        """Contract 5: output/ is created automatically if it does not exist."""
        (tmp_path / "data").mkdir()
        output_dir = tmp_path / "output"
        assert not output_dir.exists()

        resolve_paths(tmp_path, None)
        assert output_dir.is_dir(), "output/ should be created automatically"

    def test_output_dir_already_exists_no_error(self, tmp_path):
        """If output/ already exists, no error is raised."""
        (tmp_path / "data").mkdir()
        (tmp_path / "output").mkdir()
        result = resolve_paths(tmp_path, None)
        assert result[1] == tmp_path / "output"

    # --- cache_dir ---

    def test_cache_dir_is_project_dir_cache(self, tmp_path):
        """Contract 6: cache_dir is always <project_dir>/cache/."""
        (tmp_path / "data").mkdir()
        result = resolve_paths(tmp_path, None)
        assert result[2] == tmp_path / "cache"

    def test_cache_dir_created_if_absent(self, tmp_path):
        """Contract 6: cache/ is created automatically if it does not exist."""
        (tmp_path / "data").mkdir()
        cache_dir = tmp_path / "cache"
        assert not cache_dir.exists()

        resolve_paths(tmp_path, None)
        assert cache_dir.is_dir(), "cache/ should be created automatically"

    def test_cache_dir_already_exists_no_error(self, tmp_path):
        """If cache/ already exists, no error is raised."""
        (tmp_path / "data").mkdir()
        (tmp_path / "cache").mkdir()
        result = resolve_paths(tmp_path, None)
        assert result[2] == tmp_path / "cache"

    # --- mapping file resolution ---

    def test_mapping_file_resolved_to_correct_path(self, tmp_path):
        """Mapping file path is resolved correctly."""
        (tmp_path / "data").mkdir()
        mapping = tmp_path / "categories.txt"
        mapping.write_text("data")
        result = resolve_paths(tmp_path, str(mapping))
        # The returned path should point to the same file
        assert result[3] == mapping or str(result[3]) == str(mapping)

    def test_mapping_file_absolute_path_outside_project(self, tmp_path):
        """Mapping file specified as absolute path outside project dir works."""
        (tmp_path / "data").mkdir()
        subdir = tmp_path / "elsewhere"
        subdir.mkdir()
        mapping = subdir / "categories.txt"
        mapping.write_text("data")

        result = resolve_paths(tmp_path, str(mapping))
        assert result[3] is not None

    # --- Error conditions ---

    def test_missing_data_dir_raises_file_not_found_error(self, tmp_path):
        """Error: FileNotFoundError when data/ directory does not exist.

        Invariant: data/ directory must exist in the project directory.
        """
        # Do NOT create tmp_path / "data"
        with pytest.raises(FileNotFoundError):
            resolve_paths(tmp_path, None)

    def test_missing_mapping_file_raises_file_not_found_error(self, tmp_path):
        """Error: FileNotFoundError when specified mapping file does not exist."""
        (tmp_path / "data").mkdir()
        nonexistent = str(tmp_path / "nonexistent_mapping.txt")

        with pytest.raises(FileNotFoundError):
            resolve_paths(tmp_path, nonexistent)

    def test_data_as_file_not_dir_raises_file_not_found(self, tmp_path):
        """If 'data' is a file (not a directory), should raise FileNotFoundError."""
        # DATA ASSUMPTION: data must be a directory, not a file
        (tmp_path / "data").write_text("not a directory")

        with pytest.raises(FileNotFoundError):
            resolve_paths(tmp_path, None)

    def test_both_output_and_cache_created_in_single_call(self, tmp_path):
        """Both output/ and cache/ are created if both are absent."""
        (tmp_path / "data").mkdir()
        assert not (tmp_path / "output").exists()
        assert not (tmp_path / "cache").exists()

        resolve_paths(tmp_path, None)
        assert (tmp_path / "output").is_dir()
        assert (tmp_path / "cache").is_dir()

    def test_all_dirs_exist_beforehand_no_error(self, tmp_path):
        """When all directories already exist, resolve_paths succeeds cleanly."""
        (tmp_path / "data").mkdir()
        (tmp_path / "output").mkdir()
        (tmp_path / "cache").mkdir()

        result = resolve_paths(tmp_path, None)
        assert result[0] == tmp_path / "data"
        assert result[1] == tmp_path / "output"
        assert result[2] == tmp_path / "cache"
        assert result[3] is None

    def test_missing_data_dir_error_even_with_mapping_file(self, tmp_path):
        """FileNotFoundError for missing data/ takes precedence."""
        mapping = tmp_path / "mapping.txt"
        mapping.write_text("content")
        # data/ does not exist
        with pytest.raises(FileNotFoundError):
            resolve_paths(tmp_path, str(mapping))


# ---------------------------------------------------------------------------
# main() tests -- exit behavior
# ---------------------------------------------------------------------------


class TestMainExitBehavior:
    """Tests for main() error handling and exit code behavior.

    Contract 12: If any unit raises, exit with code 1 (stderr message).
    Contract 13: Brief summary to stdout on success.

    DATA ASSUMPTION: main() resolves the project directory from the
    location of the script file (__file__). We control this by
    patching the module's __file__ attribute or by providing a
    controlled sys.argv.
    """

    def test_main_exits_with_code_1_on_missing_data(self, tmp_path, monkeypatch):
        """Contract 12: main exits with code 1 when data/ is missing.

        We set up an environment where main() will find a project directory
        without a data/ subdirectory, triggering a FileNotFoundError that
        should be caught and converted to sys.exit(1).
        """
        # Patch the module-level __file__ so project_dir resolves to tmp_path
        import gsea_tool.scripts.svp_launcher as stub_module
        monkeypatch.setattr(stub_module, "__file__", str(tmp_path / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_main_prints_to_stderr_on_error(self, tmp_path, monkeypatch, capsys):
        """Contract 12: main prints a descriptive error message to stderr.

        When an error occurs, the message should be printed to stderr.
        """
        import gsea_tool.scripts.svp_launcher as stub_module
        monkeypatch.setattr(stub_module, "__file__", str(tmp_path / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py"])
        # No data/ directory -> should fail

        with pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert len(captured.err) > 0, (
            "main should print a descriptive error message to stderr"
        )


# ---------------------------------------------------------------------------
# Contract-specific tests for conditional behavior
# ---------------------------------------------------------------------------


class TestConditionalBehavior:
    """Tests verifying the two independent binary conditionals.

    Contract 10: Figure 1 iff mapping file provided
    Contract 11: Figure 3 always produced

    These are tested through resolve_paths since the mapping file
    presence/absence is surfaced through its return value.
    """

    def test_no_mapping_file_fourth_element_none(self, tmp_path):
        """Contract 10: No mapping file -> no Figure 1 trigger (None)."""
        (tmp_path / "data").mkdir()
        result = resolve_paths(tmp_path, None)
        assert result[3] is None, (
            "Without mapping file, fourth element must be None"
        )

    def test_mapping_file_present_fourth_element_not_none(self, tmp_path):
        """Contract 10: Mapping file provided -> Figure 1 trigger (Path)."""
        (tmp_path / "data").mkdir()
        mapping = tmp_path / "mapping.txt"
        mapping.write_text("categories")
        result = resolve_paths(tmp_path, str(mapping))
        assert result[3] is not None, (
            "With mapping file, fourth element must be a Path"
        )

    def test_mapping_file_controls_figure1_only(self, tmp_path):
        """Contract 10: Mapping file presence only affects the fourth return element.

        The first three elements (data_dir, output_dir, cache_dir) should be
        identical regardless of mapping file presence.
        """
        (tmp_path / "data").mkdir()
        mapping = tmp_path / "mapping.txt"
        mapping.write_text("content")

        result_without = resolve_paths(tmp_path, None)
        result_with = resolve_paths(tmp_path, str(mapping))

        assert result_without[0] == result_with[0], "data_dir should be same"
        assert result_without[1] == result_with[1], "output_dir should be same"
        assert result_without[2] == result_with[2], "cache_dir should be same"
        assert result_without[3] is None
        assert result_with[3] is not None
