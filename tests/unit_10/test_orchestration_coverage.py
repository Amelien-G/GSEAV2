"""
Additional coverage tests for Unit 10 -- Orchestration.

These tests fill coverage gaps identified by comparing the blueprint's
behavioral contracts against the existing test suite.

Gaps covered:
- Contract 7: Config loaded first; invalid config halts before data processing.
- Contract 8: Data ingestion failure halts the tool.
- Contract 9: Invocation order of all units.
- Contract 10: Figure 1 produced iff mapping file provided (via main, end-to-end).
- Contract 11: Figure 3 is always produced; clustering disabled -> unclustered.
- Contract 12: Any unit raising -> exit 1 + stderr (beyond just missing data).
- Contract 13: Brief success summary to stdout.
- Contract 14: Figure output file stem names.
- Contract 15: Always-produced output files listed in summary.
- Contract 16: Mapping file -> figure1 files additionally produced.

Synthetic Data Assumptions
--------------------------
DATA ASSUMPTION: All upstream units (1-9) are mocked. Only the
    orchestration wiring logic is tested.

DATA ASSUMPTION: CohortData is constructed with synthetic mutant_ids
    and minimal profiles. The actual data values are irrelevant to
    orchestration tests.

DATA ASSUMPTION: ToolConfig uses default values unless a specific
    config override is being tested.

DATA ASSUMPTION: Mock return values from upstream units are simple
    sentinel objects with the minimal attributes needed by the
    orchestration code.
"""

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
# Helper: build mock objects for upstream unit returns
# ---------------------------------------------------------------------------

def _make_mock_cohort(n_mutants=3):
    """Create a mock CohortData with the given number of mutants.

    DATA ASSUMPTION: Mutant IDs are synthetic strings like 'mutant_1', 'mutant_2', etc.
    """
    cohort = MagicMock()
    cohort.mutant_ids = [f"mutant_{i}" for i in range(1, n_mutants + 1)]
    return cohort


def _make_mock_config(clustering_enabled=True):
    """Create a mock ToolConfig with default-like values.

    DATA ASSUMPTION: Config attributes match the ToolConfig dataclass structure.
    """
    config = MagicMock()
    config.dot_plot.fdr_threshold = 0.05
    config.dot_plot.top_n = 20
    config.dot_plot.n_groups = 4
    config.dot_plot.random_seed = 42
    config.plot_appearance.dpi = 300
    config.plot_appearance.font_family = "Arial"
    config.clustering.enabled = clustering_enabled
    return config


def _make_mock_dot_plot_result(stem="figure2_unbiased", output_dir=None):
    """Create a mock DotPlotResult.

    DATA ASSUMPTION: Result has pdf/png/svg paths and numeric metadata.
    """
    result = MagicMock()
    base = output_dir or Path("/tmp/output")
    result.pdf_path = base / f"{stem}.pdf"
    result.png_path = base / f"{stem}.png"
    result.svg_path = base / f"{stem}.svg"
    result.n_terms_displayed = 10
    result.n_categories = 3
    result.n_mutants = 3
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
    result.n_prefiltered = 15
    result.similarity_metric = "Lin"
    result.similarity_threshold = 0.7
    return result


def _make_mock_bar_plot_result(output_dir=None):
    """Create a mock BarPlotResult."""
    result = MagicMock()
    base = output_dir or Path("/tmp/output")
    result.pdf_path = base / "figure3_meta_analysis.pdf"
    result.png_path = base / "figure3_meta_analysis.png"
    result.svg_path = base / "figure3_meta_analysis.svg"
    result.n_bars = 15
    result.n_mutants = 3
    result.clustering_was_used = True
    return result


def _make_mock_unbiased_stats():
    """Create a mock UnbiasedSelectionStats."""
    stats = MagicMock()
    stats.total_significant_terms = 50
    stats.terms_after_dedup = 30
    stats.terms_selected = 20
    stats.n_clusters = 4
    stats.random_seed = 42
    stats.clustering_algorithm = "scipy.cluster.hierarchy (Ward linkage)"
    return stats


def _setup_main_patches(tmp_path, monkeypatch, mapping_file_arg=None, clustering_enabled=True):
    """Set up monkeypatching and mocks for main() tests.

    Returns dict of mock objects for inspection.
    """
    import gsea_tool.scripts.svp_launcher as stub_module

    # Set up project directory: needs data/
    (tmp_path / "data").mkdir(exist_ok=True)

    # Patch __file__ so project_dir resolves to tmp_path
    # The implementation does: Path(__file__).resolve().parent.parent.parent
    # So we need __file__ to be at tmp_path/a/b/stub.py to get tmp_path
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))

    argv = ["stub.py"]
    if mapping_file_arg is not None:
        # Create the mapping file so resolve_paths won't raise
        mapping_path = tmp_path / "mapping.txt"
        mapping_path.write_text("category\tterm\n")
        argv.append(str(mapping_path))

    monkeypatch.setattr(sys, "argv", argv)

    # Build mock returns
    mock_cohort = _make_mock_cohort()
    mock_config = _make_mock_config(clustering_enabled=clustering_enabled)
    mock_fig1_result = _make_mock_dot_plot_result(stem="figure1_cherry_picked")
    mock_fig2_result = _make_mock_dot_plot_result(stem="figure2_unbiased")
    mock_fisher_result = _make_mock_fisher_result()
    mock_clustering_result = _make_mock_clustering_result()
    mock_bar_result = _make_mock_bar_plot_result()
    mock_unbiased_stats = _make_mock_unbiased_stats()
    mock_unbiased_groups = [MagicMock()]
    mock_cherry_groups = [MagicMock()]
    mock_term_to_category = {"TERM_A": "Category1"}
    mock_notes_path = tmp_path / "output" / "notes.md"

    mocks = {
        "cohort": mock_cohort,
        "config": mock_config,
        "fig1_result": mock_fig1_result,
        "fig2_result": mock_fig2_result,
        "fisher_result": mock_fisher_result,
        "clustering_result": mock_clustering_result,
        "bar_result": mock_bar_result,
        "unbiased_stats": mock_unbiased_stats,
        "unbiased_groups": mock_unbiased_groups,
        "cherry_groups": mock_cherry_groups,
        "term_to_category": mock_term_to_category,
        "notes_path": mock_notes_path,
    }

    return mocks


def _apply_all_patches(mocks, clustering_enabled=True, mapping_provided=False):
    """Return a list of patch context managers for all upstream units.

    Returns a dict of patchers keyed by unit name.
    """
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
    patches["select_unbiased_terms"] = patch(
        "gsea_tool.unbiased.select_unbiased_terms",
        return_value=(mocks["unbiased_groups"], mocks["unbiased_stats"]),
    )

    # render_dot_plot needs to return different results for fig1 vs fig2
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
    patches["render_bar_plot"] = patch(
        "gsea_tool.bar_plot.render_bar_plot",
        return_value=mocks["bar_result"],
    )
    patches["generate_notes"] = patch(
        "gsea_tool.notes_generation.generate_notes",
        return_value=mocks["notes_path"],
    )
    patches["NotesInput"] = patch(
        "gsea_tool.notes_generation.NotesInput",
        return_value=MagicMock(),
    )

    return patches


def _run_main_with_patches(patches):
    """Enter all patch context managers and run main(). Returns mock dict."""
    entered = {}
    managers = {}
    for name, p in patches.items():
        managers[name] = p
        entered[name] = p.start()
    try:
        main()
    finally:
        for p in managers.values():
            p.stop()
    return entered


# ---------------------------------------------------------------------------
# Contract 7: Config loaded first; invalid config halts before processing
# ---------------------------------------------------------------------------


class TestContract7ConfigLoadedFirst:
    """Contract 7: Configuration is loaded first. If config.yaml exists
    and is invalid, the tool halts before any data processing."""

    def test_config_error_exits_with_code_1(self, tmp_path, monkeypatch):
        """When load_config raises, main exits with code 1."""
        import gsea_tool.scripts.svp_launcher as stub_module
        (tmp_path / "data").mkdir()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py"])

        from gsea_tool.configuration import ConfigError

        with patch("gsea_tool.configuration.load_config", side_effect=ConfigError("bad config")):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_config_error_prints_stderr(self, tmp_path, monkeypatch, capsys):
        """When load_config raises, error message goes to stderr."""
        import gsea_tool.scripts.svp_launcher as stub_module
        (tmp_path / "data").mkdir()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py"])

        from gsea_tool.configuration import ConfigError

        with patch("gsea_tool.configuration.load_config", side_effect=ConfigError("bad config")):
            with pytest.raises(SystemExit):
                main()

        captured = capsys.readouterr()
        assert "bad config" in captured.err

    def test_config_error_prevents_data_ingestion(self, tmp_path, monkeypatch):
        """When load_config raises, ingest_data is never called (halts before processing)."""
        import gsea_tool.scripts.svp_launcher as stub_module
        (tmp_path / "data").mkdir()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py"])

        from gsea_tool.configuration import ConfigError

        with patch("gsea_tool.configuration.load_config", side_effect=ConfigError("bad config")):
            with patch("gsea_tool.data_ingestion.ingest_data") as mock_ingest:
                with pytest.raises(SystemExit):
                    main()
                mock_ingest.assert_not_called()


# ---------------------------------------------------------------------------
# Contract 8: Data ingestion failure halts the tool
# ---------------------------------------------------------------------------


class TestContract8DataIngestionFailure:
    """Contract 8: Data ingestion runs after config. If it fails
    (e.g., fewer than 2 mutants), the tool halts."""

    def test_ingestion_error_exits_with_code_1(self, tmp_path, monkeypatch):
        """When ingest_data raises, main exits with code 1."""
        import gsea_tool.scripts.svp_launcher as stub_module
        (tmp_path / "data").mkdir()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py"])

        mock_config = _make_mock_config()
        from gsea_tool.data_ingestion import DataIngestionError

        with patch("gsea_tool.configuration.load_config", return_value=mock_config):
            with patch(
                "gsea_tool.data_ingestion.ingest_data",
                side_effect=DataIngestionError("fewer than 2 mutants"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_ingestion_error_prints_stderr(self, tmp_path, monkeypatch, capsys):
        """When ingest_data raises, error message goes to stderr."""
        import gsea_tool.scripts.svp_launcher as stub_module
        (tmp_path / "data").mkdir()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py"])

        mock_config = _make_mock_config()
        from gsea_tool.data_ingestion import DataIngestionError

        with patch("gsea_tool.configuration.load_config", return_value=mock_config):
            with patch(
                "gsea_tool.data_ingestion.ingest_data",
                side_effect=DataIngestionError("fewer than 2 mutants"),
            ):
                with pytest.raises(SystemExit):
                    main()

        captured = capsys.readouterr()
        assert "fewer than 2 mutants" in captured.err


# ---------------------------------------------------------------------------
# Contract 9: Invocation order of all units
# ---------------------------------------------------------------------------


class TestContract9InvocationOrder:
    """Contract 9: The invocation order is:
    load_config -> ingest_data -> [cherry-pick] -> unbiased -> [fig1] -> fig2
    -> fisher -> [clustering] -> fig3 -> notes.
    """

    def test_unit_call_order_without_mapping(self, tmp_path, monkeypatch):
        """Without mapping file, config -> ingest -> unbiased -> fig2 -> fisher -> clustering -> fig3 -> notes."""
        mocks = _setup_main_patches(tmp_path, monkeypatch, mapping_file_arg=None)
        patches = _apply_all_patches(mocks, mapping_provided=False)

        call_order = []

        def track(name, original_side_effect=None, original_return=None):
            def wrapper(*args, **kwargs):
                call_order.append(name)
                if original_side_effect:
                    return original_side_effect(*args, **kwargs)
                return original_return
            return wrapper

        # Override patches with tracking versions
        patches["load_config"] = patch(
            "gsea_tool.configuration.load_config",
            side_effect=track("load_config", original_return=mocks["config"]),
        )
        patches["ingest_data"] = patch(
            "gsea_tool.data_ingestion.ingest_data",
            side_effect=track("ingest_data", original_return=mocks["cohort"]),
        )
        patches["select_unbiased_terms"] = patch(
            "gsea_tool.unbiased.select_unbiased_terms",
            side_effect=track(
                "select_unbiased_terms",
                original_return=(mocks["unbiased_groups"], mocks["unbiased_stats"]),
            ),
        )
        patches["render_dot_plot"] = patch(
            "gsea_tool.dot_plot.render_dot_plot",
            side_effect=track("render_dot_plot", original_return=mocks["fig2_result"]),
        )
        patches["run_fisher_analysis"] = patch(
            "gsea_tool.meta_analysis.run_fisher_analysis",
            side_effect=track("run_fisher_analysis", original_return=mocks["fisher_result"]),
        )
        patches["run_semantic_clustering"] = patch(
            "gsea_tool.go_clustering.run_semantic_clustering",
            side_effect=track(
                "run_semantic_clustering", original_return=mocks["clustering_result"]
            ),
        )
        patches["render_bar_plot"] = patch(
            "gsea_tool.bar_plot.render_bar_plot",
            side_effect=track("render_bar_plot", original_return=mocks["bar_result"]),
        )
        patches["generate_notes"] = patch(
            "gsea_tool.notes_generation.generate_notes",
            side_effect=track("generate_notes", original_return=mocks["notes_path"]),
        )

        _run_main_with_patches(patches)

        expected_order = [
            "load_config",
            "ingest_data",
            "select_unbiased_terms",
            "render_dot_plot",       # fig2
            "run_fisher_analysis",
            "run_semantic_clustering",
            "render_bar_plot",       # fig3
            "generate_notes",
        ]
        assert call_order == expected_order, (
            f"Expected invocation order {expected_order}, got {call_order}"
        )

    def test_unit_call_order_with_mapping(self, tmp_path, monkeypatch):
        """With mapping file, includes cherry-pick and fig1 steps."""
        mocks = _setup_main_patches(
            tmp_path, monkeypatch, mapping_file_arg="mapping.txt"
        )
        patches = _apply_all_patches(mocks, mapping_provided=True)

        call_order = []

        def track(name, original_return=None):
            def wrapper(*args, **kwargs):
                call_order.append(name)
                return original_return
            return wrapper

        patches["load_config"] = patch(
            "gsea_tool.configuration.load_config",
            side_effect=track("load_config", original_return=mocks["config"]),
        )
        patches["ingest_data"] = patch(
            "gsea_tool.data_ingestion.ingest_data",
            side_effect=track("ingest_data", original_return=mocks["cohort"]),
        )
        patches["parse_category_mapping"] = patch(
            "gsea_tool.cherry_picked.parse_category_mapping",
            side_effect=track(
                "parse_category_mapping", original_return=mocks["term_to_category"]
            ),
        )
        patches["select_cherry_picked_terms"] = patch(
            "gsea_tool.cherry_picked.select_cherry_picked_terms",
            side_effect=track(
                "select_cherry_picked_terms", original_return=mocks["cherry_groups"]
            ),
        )
        patches["select_unbiased_terms"] = patch(
            "gsea_tool.unbiased.select_unbiased_terms",
            side_effect=track(
                "select_unbiased_terms",
                original_return=(mocks["unbiased_groups"], mocks["unbiased_stats"]),
            ),
        )

        fig_call_count = [0]
        def track_render_dot_plot(*args, **kwargs):
            call_order.append("render_dot_plot")
            fig_call_count[0] += 1
            if fig_call_count[0] == 1:
                return mocks["fig1_result"]
            return mocks["fig2_result"]

        patches["render_dot_plot"] = patch(
            "gsea_tool.dot_plot.render_dot_plot",
            side_effect=track_render_dot_plot,
        )
        patches["run_fisher_analysis"] = patch(
            "gsea_tool.meta_analysis.run_fisher_analysis",
            side_effect=track("run_fisher_analysis", original_return=mocks["fisher_result"]),
        )
        patches["run_semantic_clustering"] = patch(
            "gsea_tool.go_clustering.run_semantic_clustering",
            side_effect=track(
                "run_semantic_clustering", original_return=mocks["clustering_result"]
            ),
        )
        patches["render_bar_plot"] = patch(
            "gsea_tool.bar_plot.render_bar_plot",
            side_effect=track("render_bar_plot", original_return=mocks["bar_result"]),
        )
        patches["generate_notes"] = patch(
            "gsea_tool.notes_generation.generate_notes",
            side_effect=track("generate_notes", original_return=mocks["notes_path"]),
        )

        _run_main_with_patches(patches)

        expected_order = [
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
        assert call_order == expected_order, (
            f"Expected invocation order {expected_order}, got {call_order}"
        )


# ---------------------------------------------------------------------------
# Contract 10: Figure 1 iff mapping file provided (end-to-end via main)
# ---------------------------------------------------------------------------


class TestContract10Figure1ViaMain:
    """Contract 10: Figure 1 is produced if and only if a category mapping
    file was provided as a CLI argument."""

    def test_no_mapping_skips_cherry_pick_and_fig1(self, tmp_path, monkeypatch):
        """Without mapping file, cherry-pick and fig1 render are not called."""
        mocks = _setup_main_patches(tmp_path, monkeypatch, mapping_file_arg=None)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        entered["parse_category_mapping"].assert_not_called()
        entered["select_cherry_picked_terms"].assert_not_called()
        # render_dot_plot called once (fig2 only)
        assert entered["render_dot_plot"].call_count == 1

    def test_with_mapping_calls_cherry_pick_and_fig1(self, tmp_path, monkeypatch):
        """With mapping file, cherry-pick and fig1 render are called."""
        mocks = _setup_main_patches(
            tmp_path, monkeypatch, mapping_file_arg="mapping.txt"
        )
        patches = _apply_all_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        entered["parse_category_mapping"].assert_called_once()
        entered["select_cherry_picked_terms"].assert_called_once()
        # render_dot_plot called twice (fig1 + fig2)
        assert entered["render_dot_plot"].call_count == 2


# ---------------------------------------------------------------------------
# Contract 11: Figure 3 always produced; clustering conditional
# ---------------------------------------------------------------------------


class TestContract11Figure3Always:
    """Contract 11: Figure 3 is always produced. When clustering is
    disabled, the bar plot shows unclustered top-N terms."""

    def test_fig3_rendered_with_clustering_enabled(self, tmp_path, monkeypatch):
        """Figure 3 (render_bar_plot) is called when clustering is enabled."""
        mocks = _setup_main_patches(
            tmp_path, monkeypatch, clustering_enabled=True
        )
        patches = _apply_all_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        entered["render_bar_plot"].assert_called_once()
        entered["run_semantic_clustering"].assert_called_once()

    def test_fig3_rendered_with_clustering_disabled(self, tmp_path, monkeypatch):
        """Figure 3 is still rendered when clustering is disabled."""
        mocks = _setup_main_patches(
            tmp_path, monkeypatch, clustering_enabled=False
        )
        patches = _apply_all_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        entered["render_bar_plot"].assert_called_once()
        # clustering should NOT be called
        entered["run_semantic_clustering"].assert_not_called()

    def test_clustering_disabled_passes_none_to_bar_plot(self, tmp_path, monkeypatch):
        """When clustering disabled, render_bar_plot gets clustering_result=None."""
        mocks = _setup_main_patches(
            tmp_path, monkeypatch, clustering_enabled=False
        )
        patches = _apply_all_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        bar_call_kwargs = entered["render_bar_plot"].call_args
        # clustering_result should be None
        assert bar_call_kwargs[1].get("clustering_result") is None or \
            (len(bar_call_kwargs[0]) > 1 and bar_call_kwargs[0][1] is None)


# ---------------------------------------------------------------------------
# Contract 12: Any unit raising -> exit 1 + stderr (multiple units)
# ---------------------------------------------------------------------------


class TestContract12AnyUnitError:
    """Contract 12: If any unit raises an exception, main exits with
    code 1 and prints to stderr."""

    def test_fisher_failure_exits_code_1(self, tmp_path, monkeypatch):
        """When run_fisher_analysis raises, main exits with code 1."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        patches["run_fisher_analysis"] = patch(
            "gsea_tool.meta_analysis.run_fisher_analysis",
            side_effect=RuntimeError("Fisher computation failed"),
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_main_with_patches(patches)
        assert exc_info.value.code == 1

    def test_render_dot_plot_failure_exits_code_1(self, tmp_path, monkeypatch):
        """When render_dot_plot raises, main exits with code 1."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        patches["render_dot_plot"] = patch(
            "gsea_tool.dot_plot.render_dot_plot",
            side_effect=ValueError("rendering failed"),
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_main_with_patches(patches)
        assert exc_info.value.code == 1

    def test_generate_notes_failure_exits_code_1(self, tmp_path, monkeypatch):
        """When generate_notes raises, main exits with code 1."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        patches["generate_notes"] = patch(
            "gsea_tool.notes_generation.generate_notes",
            side_effect=OSError("cannot write notes"),
        )

        with pytest.raises(SystemExit) as exc_info:
            _run_main_with_patches(patches)
        assert exc_info.value.code == 1

    def test_unit_error_prints_message_to_stderr(self, tmp_path, monkeypatch, capsys):
        """Error message from a failed unit appears on stderr."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        patches["run_fisher_analysis"] = patch(
            "gsea_tool.meta_analysis.run_fisher_analysis",
            side_effect=RuntimeError("chi-squared computation error"),
        )

        with pytest.raises(SystemExit):
            _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "chi-squared computation error" in captured.err

    def test_mapping_file_cli_arg_nonexistent_exits_code_1(self, tmp_path, monkeypatch):
        """Contract 12: main exits 1 when CLI mapping file does not exist."""
        import gsea_tool.scripts.svp_launcher as stub_module
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        (tmp_path / "data").mkdir()
        monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))
        monkeypatch.setattr(
            sys, "argv", ["stub.py", str(tmp_path / "nonexistent_mapping.txt")]
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Contract 13: Brief success summary to stdout
# ---------------------------------------------------------------------------


class TestContract13SuccessSummary:
    """Contract 13: The tool prints a brief summary to stdout on success:
    number of mutants processed, figures produced, and output file paths."""

    def test_success_prints_mutant_count(self, tmp_path, monkeypatch, capsys):
        """Success summary includes number of mutants processed."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "3 mutants" in captured.out

    def test_success_prints_figures_produced_no_mapping(self, tmp_path, monkeypatch, capsys):
        """Without mapping, summary lists figure2 and figure3 but not figure1."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure2_unbiased" in captured.out
        assert "figure3_meta_analysis" in captured.out
        assert "figure1_cherry_picked" not in captured.out

    def test_success_prints_figures_produced_with_mapping(self, tmp_path, monkeypatch, capsys):
        """With mapping, summary lists figure1, figure2, and figure3."""
        mocks = _setup_main_patches(
            tmp_path, monkeypatch, mapping_file_arg="mapping.txt"
        )
        patches = _apply_all_patches(mocks, mapping_provided=True)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure1_cherry_picked" in captured.out
        assert "figure2_unbiased" in captured.out
        assert "figure3_meta_analysis" in captured.out

    def test_success_prints_output_directory(self, tmp_path, monkeypatch, capsys):
        """Success summary includes the output directory path."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "output" in captured.out.lower()

    def test_success_no_stderr(self, tmp_path, monkeypatch, capsys):
        """On success, nothing is printed to stderr."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert captured.err == ""


# ---------------------------------------------------------------------------
# Contract 14: Figure output file stem names
# ---------------------------------------------------------------------------


class TestContract14FigureStemNames:
    """Contract 14: Figure 1 uses stem 'figure1_cherry_picked'.
    Figure 2 uses stem 'figure2_unbiased'.
    Figure 3 uses stem 'figure3_meta_analysis'."""

    def test_fig2_stem_passed_to_render_dot_plot(self, tmp_path, monkeypatch):
        """render_dot_plot for fig2 receives output_stem='figure2_unbiased'."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["render_dot_plot"].call_args[1]
        assert call_kwargs["output_stem"] == "figure2_unbiased"

    def test_fig1_stem_passed_to_render_dot_plot(self, tmp_path, monkeypatch):
        """render_dot_plot for fig1 receives output_stem='figure1_cherry_picked'."""
        mocks = _setup_main_patches(
            tmp_path, monkeypatch, mapping_file_arg="mapping.txt"
        )
        patches = _apply_all_patches(mocks, mapping_provided=True)
        entered = _run_main_with_patches(patches)

        # First call is fig1
        first_call_kwargs = entered["render_dot_plot"].call_args_list[0][1]
        assert first_call_kwargs["output_stem"] == "figure1_cherry_picked"

    def test_fig3_stem_passed_to_render_bar_plot(self, tmp_path, monkeypatch):
        """render_bar_plot receives output_stem='figure3_meta_analysis'."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["render_bar_plot"].call_args[1]
        assert call_kwargs["output_stem"] == "figure3_meta_analysis"


# ---------------------------------------------------------------------------
# Contract 15: Always-produced output files listed in summary
# ---------------------------------------------------------------------------


class TestContract15AlwaysProducedFiles:
    """Contract 15: The following files are always produced in output/:
    figure2_unbiased.{pdf,png,svg}, figure3_meta_analysis.{pdf,png,svg},
    pvalue_matrix.tsv, fisher_combined_pvalues.tsv, notes.md."""

    def test_stdout_mentions_pvalue_matrix(self, tmp_path, monkeypatch, capsys):
        """Success summary mentions pvalue_matrix.tsv."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "pvalue_matrix.tsv" in captured.out

    def test_stdout_mentions_fisher_combined(self, tmp_path, monkeypatch, capsys):
        """Success summary mentions fisher_combined_pvalues.tsv."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "fisher_combined_pvalues.tsv" in captured.out

    def test_stdout_mentions_notes_md(self, tmp_path, monkeypatch, capsys):
        """Success summary mentions notes.md."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "notes.md" in captured.out

    def test_stdout_mentions_figure2_formats(self, tmp_path, monkeypatch, capsys):
        """Success summary mentions figure2 in pdf/png/svg formats."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure2_unbiased.pdf" in captured.out
        assert "figure2_unbiased.png" in captured.out
        assert "figure2_unbiased.svg" in captured.out

    def test_stdout_mentions_figure3_formats(self, tmp_path, monkeypatch, capsys):
        """Success summary mentions figure3 in pdf/png/svg formats."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure3_meta_analysis.pdf" in captured.out
        assert "figure3_meta_analysis.png" in captured.out
        assert "figure3_meta_analysis.svg" in captured.out


# ---------------------------------------------------------------------------
# Contract 16: Mapping file -> figure1 files additionally
# ---------------------------------------------------------------------------


class TestContract16MappingAddsF1:
    """Contract 16: When a mapping file is provided,
    figure1_cherry_picked.{pdf,png,svg} is additionally produced in output/."""

    def test_with_mapping_stdout_includes_figure1_formats(self, tmp_path, monkeypatch, capsys):
        """With mapping file, stdout lists figure1_cherry_picked.{pdf,png,svg}."""
        mocks = _setup_main_patches(
            tmp_path, monkeypatch, mapping_file_arg="mapping.txt"
        )
        patches = _apply_all_patches(mocks, mapping_provided=True)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure1_cherry_picked.pdf" in captured.out
        assert "figure1_cherry_picked.png" in captured.out
        assert "figure1_cherry_picked.svg" in captured.out

    def test_without_mapping_stdout_omits_figure1_formats(self, tmp_path, monkeypatch, capsys):
        """Without mapping file, stdout does not list figure1 files."""
        mocks = _setup_main_patches(tmp_path, monkeypatch)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        _run_main_with_patches(patches)

        captured = capsys.readouterr()
        assert "figure1_cherry_picked" not in captured.out


# ---------------------------------------------------------------------------
# Contract 3: Project directory resolved from script location
# ---------------------------------------------------------------------------


class TestContract3ProjectDirResolution:
    """Contract 3: The project directory is resolved as the directory
    containing the script (via __file__)."""

    def test_project_dir_used_for_resolve_paths(self, tmp_path, monkeypatch):
        """main() passes the project_dir derived from __file__ to resolve_paths."""
        import gsea_tool.scripts.svp_launcher as stub_module
        (tmp_path / "data").mkdir()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(stub_module, "__file__", str(nested / "stub.py"))
        monkeypatch.setattr(sys, "argv", ["stub.py"])

        mock_config = _make_mock_config()
        mock_cohort = _make_mock_cohort()

        with patch("gsea_tool.configuration.load_config", return_value=mock_config) as mock_lc:
            with patch("gsea_tool.data_ingestion.ingest_data", return_value=mock_cohort):
                with patch("gsea_tool.unbiased.select_unbiased_terms",
                           return_value=([MagicMock()], _make_mock_unbiased_stats())):
                    with patch("gsea_tool.dot_plot.render_dot_plot",
                               return_value=_make_mock_dot_plot_result()):
                        with patch("gsea_tool.meta_analysis.run_fisher_analysis",
                                   return_value=_make_mock_fisher_result()):
                            with patch("gsea_tool.go_clustering.run_semantic_clustering",
                                       return_value=_make_mock_clustering_result()):
                                with patch("gsea_tool.bar_plot.render_bar_plot",
                                           return_value=_make_mock_bar_plot_result()):
                                    with patch("gsea_tool.notes_generation.generate_notes"):
                                        with patch("gsea_tool.notes_generation.NotesInput"):
                                            main()

        # load_config should have been called with tmp_path (the resolved project_dir)
        called_project_dir = mock_lc.call_args[0][0]
        assert called_project_dir == tmp_path


# ---------------------------------------------------------------------------
# Contract 6: cache_dir passed to Unit 7 for OBO/GAF file caching
# ---------------------------------------------------------------------------


class TestContract6CacheDirPassedToUnit7:
    """Contract 6: The cache directory is passed to Unit 7 for
    OBO/GAF file caching."""

    def test_cache_dir_passed_to_clustering(self, tmp_path, monkeypatch):
        """run_semantic_clustering receives cache_dir as an argument."""
        mocks = _setup_main_patches(tmp_path, monkeypatch, clustering_enabled=True)
        patches = _apply_all_patches(mocks, mapping_provided=False)
        entered = _run_main_with_patches(patches)

        call_kwargs = entered["run_semantic_clustering"].call_args[1]
        assert "cache_dir" in call_kwargs
        # Should be a Path ending with /cache
        cache_path = call_kwargs["cache_dir"]
        assert str(cache_path).endswith("cache")
