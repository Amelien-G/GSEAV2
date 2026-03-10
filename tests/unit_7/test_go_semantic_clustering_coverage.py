"""
Additional coverage tests for Unit 7 -- GO Semantic Clustering.

These tests cover gaps identified in the blueprint behavioral contracts
that are not exercised by the main test suite.

Synthetic Data Assumptions
==========================
DATA ASSUMPTION: GO IDs follow the standard Gene Ontology format "GO:NNNNNNN".
DATA ASSUMPTION: Combined p-values from Fisher's method are in [0.0, 1.0].
DATA ASSUMPTION: IC values are non-negative floats. IC = -log(freq).
DATA ASSUMPTION: Lin similarity values are in [0.0, 1.0].
DATA ASSUMPTION: OBO and GAF URLs are strings representing HTTP endpoints.
    In tests, these are mocked and never actually fetched.
"""

import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from gsea_tool.configuration import ClusteringConfig
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.go_clustering import (
    ClusteringResult,
    download_or_load_obo,
    download_or_load_gaf,
    compute_lin_similarity,
    cluster_by_similarity,
    select_representatives,
    run_semantic_clustering,
    write_fisher_results_with_clusters_tsv,
)


# ---------------------------------------------------------------------------
# Helper factories (same conventions as main test file)
# ---------------------------------------------------------------------------


def _make_fisher_result(
    go_ids: list[str] | None = None,
    combined_pvalues: dict[str, float] | None = None,
    go_id_to_name: dict[str, str] | None = None,
    n_contributing: dict[str, int] | None = None,
) -> FisherResult:
    """Create a FisherResult with sensible defaults for clustering tests."""
    if go_ids is None:
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
    if combined_pvalues is None:
        combined_pvalues = {gid: 0.01 * (i + 1) for i, gid in enumerate(go_ids)}
    if go_id_to_name is None:
        go_id_to_name = {gid: f"term_{i}" for i, gid in enumerate(go_ids)}
    if n_contributing is None:
        n_contributing = {gid: 3 for gid in go_ids}

    n_mutants = 3
    pvalue_matrix = np.ones((len(go_ids), n_mutants))

    return FisherResult(
        go_ids=go_ids,
        go_id_to_name=go_id_to_name,
        combined_pvalues=combined_pvalues,
        n_contributing=n_contributing,
        pvalue_matrix=pvalue_matrix,
        mutant_ids=["m1", "m2", "m3"],
        go_id_order=go_ids,
        n_mutants=n_mutants,
        corrected_pvalues=None,
    )


def _make_clustering_config(
    similarity_threshold: float = 0.7,
    **kwargs,
) -> ClusteringConfig:
    """Create a ClusteringConfig with test-friendly defaults."""
    return ClusteringConfig(
        enabled=kwargs.get("enabled", True),
        similarity_metric=kwargs.get("similarity_metric", "Lin"),
        similarity_threshold=similarity_threshold,
        go_obo_url=kwargs.get("go_obo_url", "http://example.com/go-basic.obo"),
        gaf_url=kwargs.get("gaf_url", "http://example.com/fb.gaf.gz"),
    )


# ===========================================================================
# Gap 1: Download retry behavior (Contract 3)
# ===========================================================================


class TestDownloadRetryBehavior:
    """Contract 3: Download failures are retried once. If the retry also fails,
    ConnectionError is raised with a descriptive message."""

    def test_obo_download_retries_once_before_raising(self, tmp_path):
        """OBO download should be attempted twice (initial + 1 retry) before
        raising ConnectionError.

        Contract 3: Download failures are retried once.
        """
        cache_dir = tmp_path / "cache"
        # cache_dir does not need to pre-exist; download_or_load_obo creates it

        with patch("gsea_tool.go_clustering.urllib.request.urlretrieve") as mock_retrieve:
            mock_retrieve.side_effect = urllib.error.URLError("network error")

            with pytest.raises(ConnectionError):
                download_or_load_obo("http://example.com/go-basic.obo", cache_dir)

            # Should have been called exactly 2 times (initial attempt + 1 retry)
            assert mock_retrieve.call_count == 2

    def test_gaf_download_retries_once_before_raising(self, tmp_path):
        """GAF download should be attempted twice (initial + 1 retry) before
        raising ConnectionError.

        Contract 3: Download failures are retried once.
        """
        cache_dir = tmp_path / "cache"

        with patch("gsea_tool.go_clustering.urllib.request.urlretrieve") as mock_retrieve:
            mock_retrieve.side_effect = urllib.error.URLError("network error")

            with pytest.raises(ConnectionError):
                download_or_load_gaf("http://example.com/fb.gaf.gz", cache_dir)

            assert mock_retrieve.call_count == 2

    def test_obo_download_succeeds_on_retry(self, tmp_path):
        """If the first download attempt fails but the retry succeeds,
        the function should return successfully without raising.

        Contract 3: Download failures are retried once.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target_path = cache_dir / "go-basic.obo"

        def side_effect_succeed_on_retry(url, path):
            if not hasattr(side_effect_succeed_on_retry, "_called"):
                side_effect_succeed_on_retry._called = True
                raise urllib.error.URLError("first attempt fails")
            # Second attempt succeeds: write a file
            Path(path).write_text("format-version: 1.2\n")

        with patch("gsea_tool.go_clustering.urllib.request.urlretrieve") as mock_retrieve:
            mock_retrieve.side_effect = side_effect_succeed_on_retry
            result = download_or_load_obo("http://example.com/go-basic.obo", cache_dir)
            assert isinstance(result, Path)

    def test_gaf_download_succeeds_on_retry(self, tmp_path):
        """If the first GAF download attempt fails but the retry succeeds,
        the function should return successfully without raising.

        Contract 3: Download failures are retried once.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        def side_effect_succeed_on_retry(url, path):
            if not hasattr(side_effect_succeed_on_retry, "_called"):
                side_effect_succeed_on_retry._called = True
                raise urllib.error.URLError("first attempt fails")
            Path(path).write_bytes(b"gaf content")

        with patch("gsea_tool.go_clustering.urllib.request.urlretrieve") as mock_retrieve:
            mock_retrieve.side_effect = side_effect_succeed_on_retry
            result = download_or_load_gaf("http://example.com/fb.gaf.gz", cache_dir)
            assert isinstance(result, Path)

    def test_obo_connection_error_has_descriptive_message(self, tmp_path):
        """ConnectionError raised on OBO download failure should have a
        descriptive message mentioning the URL.

        Contract 3: raises ConnectionError with a descriptive message.
        """
        cache_dir = tmp_path / "cache"

        with patch("gsea_tool.go_clustering.urllib.request.urlretrieve") as mock_retrieve:
            mock_retrieve.side_effect = urllib.error.URLError("network error")

            with pytest.raises(ConnectionError, match="OBO"):
                download_or_load_obo("http://example.com/go-basic.obo", cache_dir)

    def test_gaf_connection_error_has_descriptive_message(self, tmp_path):
        """ConnectionError raised on GAF download failure should have a
        descriptive message mentioning GAF.

        Contract 3: raises ConnectionError with a descriptive message.
        """
        cache_dir = tmp_path / "cache"

        with patch("gsea_tool.go_clustering.urllib.request.urlretrieve") as mock_retrieve:
            mock_retrieve.side_effect = urllib.error.URLError("network error")

            with pytest.raises(ConnectionError, match="GAF"):
                download_or_load_gaf("http://example.com/fb.gaf.gz", cache_dir)


# ===========================================================================
# Gap 2: Similarity threshold invariants
# ===========================================================================


class TestSimilarityThresholdInvariants:
    """Invariants: similarity_threshold must be > 0.0 and <= 1.0."""

    def test_threshold_zero_raises_assertion_error(self, tmp_path):
        """Pre-condition invariant: similarity_threshold > 0.0.
        run_semantic_clustering should raise AssertionError when threshold is 0.0."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            combined_pvalues={"GO:0000001": 0.01},
            go_id_to_name={"GO:0000001": "term_a"},
            n_contributing={"GO:0000001": 3},
        )
        config = _make_clustering_config(similarity_threshold=0.0)

        with pytest.raises(AssertionError, match="positive"):
            run_semantic_clustering(fisher, config, output_dir, cache_dir)

    def test_threshold_negative_raises_assertion_error(self, tmp_path):
        """Pre-condition invariant: similarity_threshold > 0.0.
        A negative threshold should also be rejected."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            combined_pvalues={"GO:0000001": 0.01},
            go_id_to_name={"GO:0000001": "term_a"},
            n_contributing={"GO:0000001": 3},
        )
        config = _make_clustering_config(similarity_threshold=-0.5)

        with pytest.raises(AssertionError, match="positive"):
            run_semantic_clustering(fisher, config, output_dir, cache_dir)

    def test_threshold_above_one_raises_assertion_error(self, tmp_path):
        """Pre-condition invariant: similarity_threshold <= 1.0.
        A threshold above 1.0 should be rejected."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            combined_pvalues={"GO:0000001": 0.01},
            go_id_to_name={"GO:0000001": "term_a"},
            n_contributing={"GO:0000001": 3},
        )
        config = _make_clustering_config(similarity_threshold=1.5)

        with pytest.raises(AssertionError, match="at most 1"):
            run_semantic_clustering(fisher, config, output_dir, cache_dir)


# ===========================================================================
# Gap 3: Lin similarity with MICA (common ancestor)
# ===========================================================================


class TestLinSimilarityWithCommonAncestor:
    """Contract 5: Lin similarity = 2 * IC(MICA) / (IC(t1) + IC(t2))
    where MICA is the most informative common ancestor."""

    def test_lin_similarity_with_shared_ancestor(self, tmp_path):
        """Two sibling terms sharing a common ancestor should have non-zero
        similarity proportional to the ancestor's IC.

        Contract 5: sim(t1, t2) = 2 * IC(MICA) / (IC(t1) + IC(t2))
        """
        # DATA ASSUMPTION: OBO hierarchy:
        #   GO:0000003 (root, IC=2.0)
        #     is_a parent of GO:0000001 (IC=5.0)
        #     is_a parent of GO:0000002 (IC=3.0)
        # MICA of GO:0000001 and GO:0000002 is GO:0000003 (IC=2.0)
        # Expected: sim = 2 * 2.0 / (5.0 + 3.0) = 4.0 / 8.0 = 0.5
        go_ids = ["GO:0000001", "GO:0000002"]
        ic_values = {
            "GO:0000001": 5.0,
            "GO:0000002": 3.0,
            "GO:0000003": 2.0,
        }
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000003\nname: root_term\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000001\nname: child1\nnamespace: biological_process\n"
            "is_a: GO:0000003 ! root_term\n\n"
            "[Term]\nid: GO:0000002\nname: child2\nnamespace: biological_process\n"
            "is_a: GO:0000003 ! root_term\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        # sim(child1, child2) = 2 * IC(root) / (IC(child1) + IC(child2))
        #                     = 2 * 2.0 / (5.0 + 3.0) = 0.5
        assert abs(result[0, 1] - 0.5) < 1e-10
        assert abs(result[1, 0] - 0.5) < 1e-10

    def test_lin_similarity_mica_selects_most_informative(self, tmp_path):
        """When two terms share multiple common ancestors, the MICA (most
        informative) should be used.

        Contract 5: MICA = most informative common ancestor (highest IC).
        """
        # DATA ASSUMPTION: OBO hierarchy:
        #   GO:0000010 (top root, IC=1.0)
        #     is_a parent of GO:0000020 (intermediate, IC=3.0)
        #       is_a parent of GO:0000001 (leaf1, IC=6.0)
        #       is_a parent of GO:0000002 (leaf2, IC=4.0)
        # Common ancestors of leaf1 and leaf2: GO:0000020 (IC=3.0), GO:0000010 (IC=1.0)
        # MICA = GO:0000020 (IC=3.0)
        # Expected: sim = 2 * 3.0 / (6.0 + 4.0) = 6.0 / 10.0 = 0.6
        go_ids = ["GO:0000001", "GO:0000002"]
        ic_values = {
            "GO:0000001": 6.0,
            "GO:0000002": 4.0,
            "GO:0000010": 1.0,
            "GO:0000020": 3.0,
        }
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000010\nname: top_root\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000020\nname: intermediate\nnamespace: biological_process\n"
            "is_a: GO:0000010 ! top_root\n\n"
            "[Term]\nid: GO:0000001\nname: leaf1\nnamespace: biological_process\n"
            "is_a: GO:0000020 ! intermediate\n\n"
            "[Term]\nid: GO:0000002\nname: leaf2\nnamespace: biological_process\n"
            "is_a: GO:0000020 ! intermediate\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        expected = 2.0 * 3.0 / (6.0 + 4.0)
        assert abs(result[0, 1] - expected) < 1e-10

    def test_lin_similarity_no_common_ancestor_is_zero(self, tmp_path):
        """Terms with no common ancestor should have similarity 0.0.

        Contract 5: If there is no common ancestor, similarity should be 0.
        """
        go_ids = ["GO:0000001", "GO:0000002"]
        ic_values = {"GO:0000001": 5.0, "GO:0000002": 3.0}
        obo_path = tmp_path / "go.obo"
        # Two terms with no is_a relationship (no shared hierarchy)
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000001\nname: term1\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000002\nname: term2\nnamespace: biological_process\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0


# ===========================================================================
# Gap 4: cluster_by_similarity with empty input
# ===========================================================================


class TestClusterBySimilarityEmpty:
    """Edge case: empty similarity matrix."""

    def test_empty_similarity_matrix_returns_empty_list(self):
        """An empty similarity matrix (0x0) should return an empty list of clusters."""
        sim = np.array([]).reshape(0, 0)
        result = cluster_by_similarity(sim, threshold=0.7)
        assert result == []


# ===========================================================================
# Gap 5: similarity_threshold stored from config in result
# ===========================================================================


class TestSimilarityThresholdStoredInResult:
    """The ClusteringResult should store the similarity_threshold from config."""

    def test_threshold_from_config_stored_in_result(self, tmp_path):
        """run_semantic_clustering should set similarity_threshold in
        ClusteringResult from the config value."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002"],
            combined_pvalues={"GO:0000001": 0.01, "GO:0000002": 0.02},
            go_id_to_name={"GO:0000001": "a", "GO:0000002": "b"},
            n_contributing={"GO:0000001": 3, "GO:0000002": 2},
        )
        config = _make_clustering_config(similarity_threshold=0.8)

        with patch("gsea_tool.go_clustering.download_or_load_obo") as mock_obo, \
             patch("gsea_tool.go_clustering.download_or_load_gaf") as mock_gaf, \
             patch("gsea_tool.go_clustering.compute_information_content") as mock_ic, \
             patch("gsea_tool.go_clustering.compute_lin_similarity") as mock_sim, \
             patch("gsea_tool.go_clustering.cluster_by_similarity") as mock_cluster, \
             patch("gsea_tool.go_clustering.select_representatives") as mock_select, \
             patch("gsea_tool.go_clustering.write_fisher_results_with_clusters_tsv") as mock_write:

            mock_obo.return_value = cache_dir / "go.obo"
            mock_gaf.return_value = cache_dir / "fb.gaf"
            mock_ic.return_value = {"GO:0000001": 3.0, "GO:0000002": 4.0}
            mock_sim.return_value = np.eye(2)
            mock_cluster.return_value = [[0], [1]]
            cr = ClusteringResult(
                representatives=["GO:0000001", "GO:0000002"],
                representative_names=["a", "b"],
                representative_pvalues=[0.01, 0.02],
                representative_n_contributing=[3, 2],
                cluster_assignments={"GO:0000001": 0, "GO:0000002": 1},
                n_clusters=2,
                n_prefiltered=2,
                similarity_metric="Lin",
                similarity_threshold=0.0,  # select_representatives sets a default
            )
            mock_select.return_value = cr
            mock_write.return_value = output_dir / "fisher_combined_pvalues.tsv"

            result = run_semantic_clustering(fisher, config, output_dir, cache_dir)
            # run_semantic_clustering should override the threshold from config
            assert result.similarity_threshold == 0.8


# ===========================================================================
# Gap 6: TSV representative column values
# ===========================================================================


class TestTsvRepresentativeMarking:
    """Contract 10: TSV includes whether each term is a representative."""

    def test_representative_terms_marked_true_in_tsv(self, tmp_path):
        """Representative terms should be marked as True in the representative
        column of the TSV output.

        Contract 10: TSV includes whether the term is a representative.
        """
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.005,
                "GO:0000002": 0.01,
                "GO:0000003": 0.03,
            },
            go_id_to_name={
                "GO:0000001": "term_a",
                "GO:0000002": "term_b",
                "GO:0000003": "term_c",
            },
            n_contributing={
                "GO:0000001": 3,
                "GO:0000002": 2,
                "GO:0000003": 3,
            },
        )
        cr = ClusteringResult(
            representatives=["GO:0000001", "GO:0000002"],
            representative_names=["term_a", "term_b"],
            representative_pvalues=[0.005, 0.01],
            representative_n_contributing=[3, 2],
            cluster_assignments={
                "GO:0000001": 0,
                "GO:0000002": 1,
                "GO:0000003": 0,  # in same cluster as GO:0000001 but not representative
            },
            n_clusters=2,
            n_prefiltered=3,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        result_path = write_fisher_results_with_clusters_tsv(fisher, cr, tmp_path)
        content = result_path.read_text()
        lines = content.strip().split("\n")

        # Parse data rows (skip header)
        data_rows = {}
        for line in lines[1:]:
            fields = line.split("\t")
            go_id = fields[0]
            representative_flag = fields[-1]  # last column is representative
            data_rows[go_id] = representative_flag

        # Representatives should be marked True
        assert data_rows["GO:0000001"] == "True"
        assert data_rows["GO:0000002"] == "True"
        # Non-representative should be marked False
        assert data_rows["GO:0000003"] == "False"

    def test_tsv_returns_correct_path(self, tmp_path):
        """write_fisher_results_with_clusters_tsv should return a Path pointing
        to the written file in output_dir."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            combined_pvalues={"GO:0000001": 0.01},
            go_id_to_name={"GO:0000001": "term_a"},
            n_contributing={"GO:0000001": 3},
        )
        cr = ClusteringResult(
            representatives=["GO:0000001"],
            representative_names=["term_a"],
            representative_pvalues=[0.01],
            representative_n_contributing=[3],
            cluster_assignments={"GO:0000001": 0},
            n_clusters=1,
            n_prefiltered=1,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        result_path = write_fisher_results_with_clusters_tsv(fisher, cr, tmp_path)
        assert result_path == tmp_path / "fisher_combined_pvalues.tsv"
        assert result_path.exists()


# ===========================================================================
# Gap 7: TSV data row content verification
# ===========================================================================


class TestTsvDataRowContent:
    """Contract 10: TSV rows contain correct data for each prefiltered term."""

    def test_tsv_row_contains_correct_pvalue_and_cluster(self, tmp_path):
        """Each data row in the TSV should contain the correct combined p-value,
        n_contributing, and cluster index for that GO term.

        Contract 10: TSV columns include GO ID, GO term name, combined p-value,
        number of contributing lines, cluster index, and representative flag.
        """
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002"],
            combined_pvalues={"GO:0000001": 0.005, "GO:0000002": 0.03},
            go_id_to_name={"GO:0000001": "term_a", "GO:0000002": "term_b"},
            n_contributing={"GO:0000001": 5, "GO:0000002": 2},
        )
        cr = ClusteringResult(
            representatives=["GO:0000001"],
            representative_names=["term_a"],
            representative_pvalues=[0.005],
            representative_n_contributing=[5],
            cluster_assignments={"GO:0000001": 0, "GO:0000002": 0},
            n_clusters=1,
            n_prefiltered=2,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        result_path = write_fisher_results_with_clusters_tsv(fisher, cr, tmp_path)
        content = result_path.read_text()
        lines = content.strip().split("\n")

        # Parse data rows
        for line in lines[1:]:
            fields = line.split("\t")
            go_id = fields[0]
            term_name = fields[1]
            combined_p = float(fields[2])
            n_contrib = int(fields[3])
            cluster_idx = int(fields[4])

            assert term_name == fisher.go_id_to_name[go_id]
            assert abs(combined_p - fisher.combined_pvalues[go_id]) < 1e-10
            assert n_contrib == fisher.n_contributing[go_id]
            assert cluster_idx == cr.cluster_assignments[go_id]


# ===========================================================================
# Gap 8: select_representatives similarity_metric field
# ===========================================================================


class TestSelectRepresentativesSimilarityMetric:
    """select_representatives should set similarity_metric to 'Lin'."""

    def test_similarity_metric_is_lin(self):
        """select_representatives should set similarity_metric field."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            combined_pvalues={"GO:0000001": 0.01},
            go_id_to_name={"GO:0000001": "term_a"},
            n_contributing={"GO:0000001": 3},
        )
        clusters = [[0]]
        go_ids = ["GO:0000001"]
        result = select_representatives(clusters, go_ids, fisher)
        assert result.similarity_metric == "Lin"
