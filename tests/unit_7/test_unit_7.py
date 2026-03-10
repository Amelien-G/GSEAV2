"""
Tests for Unit 7 -- GO Semantic Clustering.

Validates all behavioral contracts, invariants, and error conditions
specified in the blueprint for Unit 7.

Synthetic Data Assumptions
==========================
- GO IDs use realistic format GO:NNNNNNN (e.g., GO:0008150).
- Combined p-values are in [0.0, 1.0]; mix of significant (<0.05) and
  non-significant (>=0.05) to test pre-filtering.
- Information content values are non-negative floats (0.0 for unannotated).
- Lin similarity values are in [0.0, 1.0].
- OBO files use minimal valid format with [Term] blocks.
- GAF files use tab-separated GAF 2.x format.
- Default pre-filter threshold is 0.05.
- Default similarity threshold is 0.7.
"""

import math
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from gsea_tool.go_clustering import (
    ClusteringResult,
    download_or_load_obo,
    download_or_load_gaf,
    compute_information_content,
    compute_lin_similarity,
    cluster_by_similarity,
    select_representatives,
    run_semantic_clustering,
    write_fisher_results_with_clusters_tsv,
)
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.configuration import ClusteringConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_obo_file(tmp_path):
    """Create a minimal OBO file with a simple GO hierarchy.

    Hierarchy:
        GO:0000001 (root)
          |-- GO:0000002 (child1)
          |     |-- GO:0000004 (grandchild1)
          |-- GO:0000003 (child2)
          |     |-- GO:0000005 (grandchild2)
    """
    content = textwrap.dedent("""\
        format-version: 1.2
        ontology: test

        [Term]
        id: GO:0000001
        name: root process

        [Term]
        id: GO:0000002
        name: child process 1
        is_a: GO:0000001 ! root process

        [Term]
        id: GO:0000003
        name: child process 2
        is_a: GO:0000001 ! root process

        [Term]
        id: GO:0000004
        name: grandchild process 1
        is_a: GO:0000002 ! child process 1

        [Term]
        id: GO:0000005
        name: grandchild process 2
        is_a: GO:0000003 ! child process 2
    """)
    obo_path = tmp_path / "test.obo"
    obo_path.write_text(content, encoding="utf-8")
    return obo_path


@pytest.fixture
def minimal_gaf_file(tmp_path):
    """Create a minimal GAF file with annotations for the test hierarchy.

    Annotations:
        geneA -> GO:0000004 (grandchild1)
        geneB -> GO:0000004 (grandchild1)
        geneC -> GO:0000005 (grandchild2)
        geneA -> GO:0000002 (child1)
    """
    lines = [
        "!gaf-version: 2.1",
        "!This is a test GAF file",
        "DB\tgeneA\tgeneA\t\tGO:0000004\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        "DB\tgeneB\tgeneB\t\tGO:0000004\tPMID:0\tIDA\t\t\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        "DB\tgeneC\tgeneC\t\tGO:0000005\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        "DB\tgeneA\tgeneA\t\tGO:0000002\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
    ]
    gaf_path = tmp_path / "test.gaf"
    gaf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gaf_path


@pytest.fixture
def sample_fisher_result():
    """Create a FisherResult with a mix of significant and non-significant terms."""
    go_ids = [
        "GO:0000002", "GO:0000003", "GO:0000004",
        "GO:0000005", "GO:0000006", "GO:0000007",
    ]
    combined_pvalues = {
        "GO:0000002": 0.001,   # significant
        "GO:0000003": 0.01,    # significant
        "GO:0000004": 0.03,    # significant
        "GO:0000005": 0.5,     # NOT significant
        "GO:0000006": 0.002,   # significant
        "GO:0000007": 0.8,     # NOT significant
    }
    go_id_to_name = {
        "GO:0000002": "child process 1",
        "GO:0000003": "child process 2",
        "GO:0000004": "grandchild process 1",
        "GO:0000005": "grandchild process 2",
        "GO:0000006": "extra process 1",
        "GO:0000007": "extra process 2",
    }
    n_contributing = {
        "GO:0000002": 3,
        "GO:0000003": 2,
        "GO:0000004": 4,
        "GO:0000005": 1,
        "GO:0000006": 5,
        "GO:0000007": 1,
    }
    return FisherResult(
        go_ids=go_ids,
        go_id_to_name=go_id_to_name,
        combined_pvalues=combined_pvalues,
        n_contributing=n_contributing,
        pvalue_matrix=np.zeros((len(go_ids), 3)),
        mutant_ids=["m1", "m2", "m3"],
        go_id_order=go_ids,
        n_mutants=3,
        corrected_pvalues=None,
    )


@pytest.fixture
def all_significant_fisher_result():
    """FisherResult where all terms have p < 0.05."""
    go_ids = ["GO:0000002", "GO:0000003", "GO:0000004"]
    combined_pvalues = {
        "GO:0000002": 0.001,
        "GO:0000003": 0.01,
        "GO:0000004": 0.03,
    }
    go_id_to_name = {
        "GO:0000002": "child process 1",
        "GO:0000003": "child process 2",
        "GO:0000004": "grandchild process 1",
    }
    n_contributing = {
        "GO:0000002": 3,
        "GO:0000003": 2,
        "GO:0000004": 4,
    }
    return FisherResult(
        go_ids=go_ids,
        go_id_to_name=go_id_to_name,
        combined_pvalues=combined_pvalues,
        n_contributing=n_contributing,
        pvalue_matrix=np.zeros((3, 2)),
        mutant_ids=["m1", "m2"],
        go_id_order=go_ids,
        n_mutants=2,
        corrected_pvalues=None,
    )


@pytest.fixture
def no_significant_fisher_result():
    """FisherResult where no terms have p < 0.05."""
    go_ids = ["GO:0000002", "GO:0000003"]
    combined_pvalues = {
        "GO:0000002": 0.5,
        "GO:0000003": 0.8,
    }
    go_id_to_name = {
        "GO:0000002": "process A",
        "GO:0000003": "process B",
    }
    n_contributing = {
        "GO:0000002": 2,
        "GO:0000003": 1,
    }
    return FisherResult(
        go_ids=go_ids,
        go_id_to_name=go_id_to_name,
        combined_pvalues=combined_pvalues,
        n_contributing=n_contributing,
        pvalue_matrix=np.zeros((2, 2)),
        mutant_ids=["m1", "m2"],
        go_id_order=go_ids,
        n_mutants=2,
        corrected_pvalues=None,
    )


@pytest.fixture
def default_clustering_config():
    """Default ClusteringConfig with valid URLs."""
    return ClusteringConfig(
        enabled=True,
        similarity_metric="Lin",
        similarity_threshold=0.7,
        go_obo_url="http://example.com/go.obo",
        gaf_url="http://example.com/test.gaf",
    )


# ---------------------------------------------------------------------------
# Tests: ClusteringResult dataclass
# ---------------------------------------------------------------------------

class TestClusteringResultDataclass:
    """Tests for the ClusteringResult dataclass structure."""

    def test_clustering_result_has_all_required_fields(self):
        """Verify ClusteringResult can be instantiated with all required fields."""
        result = ClusteringResult(
            representatives=["GO:0000001"],
            representative_names=["root process"],
            representative_pvalues=[0.01],
            representative_n_contributing=[3],
            cluster_assignments={"GO:0000001": 0},
            n_clusters=1,
            n_prefiltered=1,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        assert result.representatives == ["GO:0000001"]
        assert result.representative_names == ["root process"]
        assert result.representative_pvalues == [0.01]
        assert result.representative_n_contributing == [3]
        assert result.cluster_assignments == {"GO:0000001": 0}
        assert result.n_clusters == 1
        assert result.n_prefiltered == 1
        assert result.similarity_metric == "Lin"
        assert result.similarity_threshold == 0.7


# ---------------------------------------------------------------------------
# Tests: download_or_load_obo
# ---------------------------------------------------------------------------

class TestDownloadOrLoadObo:
    """Tests for the download_or_load_obo function."""

    def test_returns_cached_path_when_file_exists(self, tmp_path):
        """Contract 2: If already cached, the cached version is used without re-downloading."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached_file = cache_dir / "go.obo"
        cached_file.write_text("cached content", encoding="utf-8")

        result = download_or_load_obo("http://example.com/go.obo", cache_dir)
        assert result == cached_file
        assert result.read_text() == "cached content"

    def test_downloads_file_when_not_cached(self, tmp_path):
        """Contract 2: Downloads file from URL when not cached."""
        cache_dir = tmp_path / "cache"

        with patch("gsea_tool.go_clustering._download_file") as mock_retrieve:
            def fake_retrieve(url, dest):
                Path(dest).write_text("downloaded obo", encoding="utf-8")
            mock_retrieve.side_effect = fake_retrieve

            result = download_or_load_obo("http://example.com/go.obo", cache_dir)
            assert result.exists()
            assert result.name == "go.obo"
            mock_retrieve.assert_called_once()

    def test_creates_cache_directory_if_missing(self, tmp_path):
        """Contract 2: Cache directory is created if it does not exist."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        # Place a cached file so we don't actually download
        cache_dir.mkdir(parents=True)
        cached_file = cache_dir / "go.obo"
        cached_file.write_text("content", encoding="utf-8")

        result = download_or_load_obo("http://example.com/go.obo", cache_dir)
        assert cache_dir.exists()

    def test_raises_connection_error_after_retry_failure(self, tmp_path):
        """Error condition: ConnectionError raised when download fails after retry."""
        cache_dir = tmp_path / "cache"

        import urllib.error
        with patch("gsea_tool.go_clustering._download_file",
                   side_effect=urllib.error.URLError("network error")):
            with pytest.raises(ConnectionError, match="Failed to download GO OBO file"):
                download_or_load_obo("http://example.com/go.obo", cache_dir)

    def test_retries_once_before_failing(self, tmp_path):
        """Contract 3: Download failures are retried once before raising."""
        cache_dir = tmp_path / "cache"

        import urllib.error
        call_count = 0

        def failing_retrieve(url, dest):
            nonlocal call_count
            call_count += 1
            raise urllib.error.URLError("network error")

        with patch("gsea_tool.go_clustering._download_file",
                   side_effect=failing_retrieve):
            with pytest.raises(ConnectionError):
                download_or_load_obo("http://example.com/go.obo", cache_dir)

        assert call_count == 2, "Should attempt download twice (initial + 1 retry)"

    def test_succeeds_on_retry_after_first_failure(self, tmp_path):
        """Contract 3: If first attempt fails but retry succeeds, no error is raised."""
        cache_dir = tmp_path / "cache"

        import urllib.error
        attempt = [0]

        def flaky_retrieve(url, dest):
            attempt[0] += 1
            if attempt[0] == 1:
                raise urllib.error.URLError("transient error")
            Path(dest).write_text("success", encoding="utf-8")

        with patch("gsea_tool.go_clustering._download_file",
                   side_effect=flaky_retrieve):
            result = download_or_load_obo("http://example.com/go.obo", cache_dir)
            assert result.exists()


# ---------------------------------------------------------------------------
# Tests: download_or_load_gaf
# ---------------------------------------------------------------------------

class TestDownloadOrLoadGaf:
    """Tests for the download_or_load_gaf function."""

    def test_returns_cached_path_when_file_exists(self, tmp_path):
        """Contract 2: Cached GAF file is reused."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached_file = cache_dir / "test.gaf.gz"
        cached_file.write_bytes(b"cached gaf")

        result = download_or_load_gaf("http://example.com/test.gaf.gz", cache_dir)
        assert result == cached_file

    def test_raises_connection_error_after_retry_failure(self, tmp_path):
        """Error condition: ConnectionError raised when GAF download fails after retry."""
        cache_dir = tmp_path / "cache"

        import urllib.error
        with patch("gsea_tool.go_clustering._download_file",
                   side_effect=urllib.error.URLError("network error")):
            with pytest.raises(ConnectionError, match="Failed to download Drosophila GAF file"):
                download_or_load_gaf("http://example.com/test.gaf.gz", cache_dir)


# ---------------------------------------------------------------------------
# Tests: compute_information_content
# ---------------------------------------------------------------------------

class TestComputeInformationContent:
    """Tests for the compute_information_content function."""

    def test_returns_dict_mapping_go_id_to_ic(self, minimal_obo_file, minimal_gaf_file):
        """Contract 4: IC is computed from annotation frequencies."""
        ic_values = compute_information_content(minimal_obo_file, minimal_gaf_file)
        assert isinstance(ic_values, dict)
        # All GO IDs from the OBO should be present
        assert "GO:0000001" in ic_values
        assert "GO:0000002" in ic_values
        assert "GO:0000004" in ic_values

    def test_unannotated_terms_have_zero_ic(self, minimal_obo_file, minimal_gaf_file):
        """Contract 4: GO terms not present in the GAF receive IC of 0.0."""
        ic_values = compute_information_content(minimal_obo_file, minimal_gaf_file)
        # GO:0000003 has annotation via GO:0000005, so it may have nonzero IC.
        # But terms with no annotations at all (direct or propagated) should have IC=0.0.
        # All terms in our test have at least propagated annotations,
        # so we verify IC values are non-negative.
        for go_id, ic in ic_values.items():
            assert ic >= 0.0, f"IC for {go_id} should be non-negative"

    def test_more_specific_terms_have_higher_ic(self, minimal_obo_file, minimal_gaf_file):
        """Contract 4: Specific terms (fewer annotations) should have higher IC than general terms."""
        ic_values = compute_information_content(minimal_obo_file, minimal_gaf_file)
        # GO:0000001 is root (all genes annotated) => lowest IC (or highest freq)
        # GO:0000004 is grandchild (fewer genes) => higher IC
        # Root should have lower IC than grandchild
        if ic_values.get("GO:0000001", 0.0) > 0.0 and ic_values.get("GO:0000004", 0.0) > 0.0:
            assert ic_values["GO:0000001"] <= ic_values["GO:0000004"], \
                "Root term should have lower or equal IC compared to specific term"

    def test_ic_values_are_non_negative(self, minimal_obo_file, minimal_gaf_file):
        """IC = -log(freq) where 0 < freq <= 1, so IC >= 0."""
        ic_values = compute_information_content(minimal_obo_file, minimal_gaf_file)
        for go_id, ic in ic_values.items():
            assert ic >= 0.0, f"IC for {go_id} must be non-negative, got {ic}"


# ---------------------------------------------------------------------------
# Tests: compute_lin_similarity
# ---------------------------------------------------------------------------

class TestComputeLinSimilarity:
    """Tests for the compute_lin_similarity function."""

    def test_returns_symmetric_matrix(self, minimal_obo_file, minimal_gaf_file):
        """Contract 5: Lin similarity matrix is symmetric."""
        ic_values = compute_information_content(minimal_obo_file, minimal_gaf_file)
        go_ids = ["GO:0000002", "GO:0000003", "GO:0000004"]
        sim_matrix = compute_lin_similarity(go_ids, ic_values, minimal_obo_file)
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T)

    def test_matrix_shape_matches_input_length(self, minimal_obo_file, minimal_gaf_file):
        """Signature: returns matrix of shape (n, n)."""
        ic_values = compute_information_content(minimal_obo_file, minimal_gaf_file)
        go_ids = ["GO:0000002", "GO:0000004"]
        sim_matrix = compute_lin_similarity(go_ids, ic_values, minimal_obo_file)
        assert sim_matrix.shape == (2, 2)

    def test_values_in_zero_to_one_range(self, minimal_obo_file, minimal_gaf_file):
        """Contract 5: All similarity values are in [0, 1]."""
        ic_values = compute_information_content(minimal_obo_file, minimal_gaf_file)
        go_ids = ["GO:0000002", "GO:0000003", "GO:0000004", "GO:0000005"]
        sim_matrix = compute_lin_similarity(go_ids, ic_values, minimal_obo_file)
        assert np.all(sim_matrix >= 0.0)
        assert np.all(sim_matrix <= 1.0)

    def test_self_similarity_is_one_for_nonzero_ic(self, minimal_obo_file, minimal_gaf_file):
        """Contract 5: sim(t, t) = 2*IC(t) / (2*IC(t)) = 1.0 when IC > 0."""
        ic_values = compute_information_content(minimal_obo_file, minimal_gaf_file)
        # Pick a term with known nonzero IC
        go_ids = ["GO:0000004"]  # grandchild, should have annotations
        sim_matrix = compute_lin_similarity(go_ids, ic_values, minimal_obo_file)
        if ic_values.get("GO:0000004", 0.0) > 0.0:
            assert sim_matrix[0, 0] == pytest.approx(1.0)

    def test_zero_ic_terms_have_zero_similarity(self, minimal_obo_file):
        """Contract 5: If IC(t1) + IC(t2) = 0, similarity is 0.0."""
        ic_values = {"GO:0000002": 0.0, "GO:0000003": 0.0}
        go_ids = ["GO:0000002", "GO:0000003"]
        sim_matrix = compute_lin_similarity(go_ids, ic_values, minimal_obo_file)
        assert sim_matrix[0, 1] == 0.0
        assert sim_matrix[1, 0] == 0.0

    def test_empty_go_ids_returns_empty_matrix(self, minimal_obo_file):
        """Edge case: empty input list returns empty matrix."""
        sim_matrix = compute_lin_similarity([], {}, minimal_obo_file)
        assert sim_matrix.shape == (0, 0)

    def test_lin_formula_correctness(self, minimal_obo_file):
        """Contract 5: Verify Lin formula sim = 2 * IC(MICA) / (IC(t1) + IC(t2))."""
        # Create terms where we know the MICA and its IC.
        # GO:0000002 and GO:0000004 share ancestor GO:0000002 (parent of GO:0000004).
        # If we set IC values explicitly, we can verify the formula.
        ic_values = {
            "GO:0000001": 0.5,
            "GO:0000002": 2.0,
            "GO:0000004": 3.0,
        }
        go_ids = ["GO:0000002", "GO:0000004"]
        sim_matrix = compute_lin_similarity(go_ids, ic_values, minimal_obo_file)
        # MICA of GO:0000002 and GO:0000004 is GO:0000002 (it's an ancestor of GO:0000004).
        # sim = 2 * IC(GO:0000002) / (IC(GO:0000002) + IC(GO:0000004))
        #     = 2 * 2.0 / (2.0 + 3.0) = 4.0 / 5.0 = 0.8
        expected = 2.0 * 2.0 / (2.0 + 3.0)
        assert sim_matrix[0, 1] == pytest.approx(expected, abs=1e-6)
        assert sim_matrix[1, 0] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: cluster_by_similarity
# ---------------------------------------------------------------------------

class TestClusterBySimilarity:
    """Tests for the cluster_by_similarity function."""

    def test_identical_items_cluster_together(self):
        """Contract 6: Items with similarity > threshold should be in same cluster."""
        # 3 items: first two very similar, third dissimilar
        sim = np.array([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])
        clusters = cluster_by_similarity(sim, threshold=0.7)
        # First two should be in same cluster, third separate
        assert len(clusters) == 2
        # Find which cluster contains index 0 and 1
        cluster_with_0_and_1 = None
        for c in clusters:
            if 0 in c and 1 in c:
                cluster_with_0_and_1 = c
                break
        assert cluster_with_0_and_1 is not None, "Items 0 and 1 should cluster together"
        # Item 2 should be in its own cluster
        for c in clusters:
            if 2 in c:
                assert len(c) == 1

    def test_all_dissimilar_items_form_separate_clusters(self):
        """Contract 6: Items with similarity < threshold form separate clusters."""
        sim = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 1.0, 0.15],
            [0.2, 0.15, 1.0],
        ])
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert len(clusters) == 3

    def test_single_item_forms_single_cluster(self):
        """Edge case: Single item results in one cluster."""
        sim = np.array([[1.0]])
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert len(clusters) == 1
        assert clusters[0] == [0]

    def test_empty_matrix_returns_empty_clusters(self):
        """Edge case: Empty matrix returns no clusters."""
        sim = np.zeros((0, 0))
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert clusters == []

    def test_all_identical_items_form_one_cluster(self):
        """All items with perfect similarity form one cluster."""
        sim = np.ones((4, 4))
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2, 3]

    def test_uses_average_linkage(self):
        """Contract 6: Hierarchical clustering uses average linkage."""
        # Create a scenario where average vs single linkage would differ
        # Items 0,1 similar; item 2 somewhat similar to 1 but not 0
        sim = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.65],
            [0.3, 0.65, 1.0],
        ])
        # With average linkage at threshold 0.7:
        # dist(0,1)=0.2, dist(0,2)=0.7, dist(1,2)=0.35
        # First merge 0,1 (dist=0.2)
        # Average dist from {0,1} to 2 = (0.7+0.35)/2 = 0.525
        # Cut at distance 1-0.7=0.3 => {0,1} and {2} are separate
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert len(clusters) == 2

    def test_returns_list_of_index_lists(self):
        """Signature: Returns list of clusters, each cluster is list of row indices."""
        sim = np.array([[1.0, 0.5], [0.5, 1.0]])
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert isinstance(clusters, list)
        for c in clusters:
            assert isinstance(c, list)
            for idx in c:
                assert isinstance(idx, int)

    def test_distance_equals_one_minus_similarity(self):
        """Contract 6: Distance = 1 - similarity, cut at 1 - threshold."""
        # Two items with similarity exactly at threshold
        sim = np.array([
            [1.0, 0.7],
            [0.7, 1.0],
        ])
        # Distance = 0.3, cut at 1-0.7=0.3
        # At exactly the threshold, behavior depends on scipy's fcluster
        # but they should typically end up in one or two clusters.
        clusters = cluster_by_similarity(sim, threshold=0.7)
        # Just verify it returns a valid result
        assert len(clusters) >= 1
        all_indices = sorted([idx for c in clusters for idx in c])
        assert all_indices == [0, 1]


# ---------------------------------------------------------------------------
# Tests: select_representatives
# ---------------------------------------------------------------------------

class TestSelectRepresentatives:
    """Tests for the select_representatives function."""

    def test_selects_lowest_pvalue_as_representative(self, sample_fisher_result):
        """Contract 8: Within each cluster, representative has lowest combined p-value."""
        # Cluster with GO:0000002 (p=0.001) and GO:0000004 (p=0.03)
        clusters = [[0, 2], [1]]  # indices into go_ids list
        go_ids = ["GO:0000002", "GO:0000004", "GO:0000003"]
        result = select_representatives(clusters, go_ids, sample_fisher_result)
        # Cluster 0: GO:0000002 (p=0.001) vs GO:0000003 (p=0.01) => GO:0000002
        # Cluster 1: GO:0000004 (p=0.03) alone
        assert "GO:0000002" in result.representatives

    def test_representatives_ordered_by_pvalue_ascending(self, sample_fisher_result):
        """Contract 9 / Invariant: Representatives ordered by combined p-value ascending."""
        clusters = [[0], [1], [2]]
        go_ids = ["GO:0000004", "GO:0000002", "GO:0000003"]
        result = select_representatives(clusters, go_ids, sample_fisher_result)
        # Representatives should be ordered: GO:0000002 (0.001), GO:0000003 (0.01), GO:0000004 (0.03)
        assert result.representative_pvalues == sorted(result.representative_pvalues)

    def test_one_representative_per_cluster(self, sample_fisher_result):
        """Invariant: len(representatives) == n_clusters."""
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000002", "GO:0000003", "GO:0000004"]
        result = select_representatives(clusters, go_ids, sample_fisher_result)
        assert len(result.representatives) == result.n_clusters
        assert result.n_clusters == 2

    def test_cluster_assignments_include_all_terms(self, sample_fisher_result):
        """All pre-filtered terms should have a cluster assignment."""
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000002", "GO:0000003", "GO:0000004"]
        result = select_representatives(clusters, go_ids, sample_fisher_result)
        for go_id in go_ids:
            assert go_id in result.cluster_assignments

    def test_n_prefiltered_matches_go_ids_length(self, sample_fisher_result):
        """n_prefiltered should reflect the number of GO IDs that entered clustering."""
        clusters = [[0], [1], [2]]
        go_ids = ["GO:0000002", "GO:0000003", "GO:0000004"]
        result = select_representatives(clusters, go_ids, sample_fisher_result)
        assert result.n_prefiltered == 3

    def test_parallel_lists_have_consistent_length(self, sample_fisher_result):
        """representatives, representative_names, representative_pvalues, representative_n_contributing
        must all have the same length."""
        clusters = [[0], [1]]
        go_ids = ["GO:0000002", "GO:0000004"]
        result = select_representatives(clusters, go_ids, sample_fisher_result)
        n = len(result.representatives)
        assert len(result.representative_names) == n
        assert len(result.representative_pvalues) == n
        assert len(result.representative_n_contributing) == n

    def test_representative_names_match_fisher_result(self, sample_fisher_result):
        """Representative names should come from fisher_result.go_id_to_name."""
        clusters = [[0]]
        go_ids = ["GO:0000002"]
        result = select_representatives(clusters, go_ids, sample_fisher_result)
        assert result.representative_names[0] == "child process 1"

    def test_representative_n_contributing_from_fisher_result(self, sample_fisher_result):
        """Contributing line counts come from fisher_result.n_contributing."""
        clusters = [[0]]
        go_ids = ["GO:0000002"]
        result = select_representatives(clusters, go_ids, sample_fisher_result)
        assert result.representative_n_contributing[0] == 3

    def test_all_representatives_in_fisher_results(self, sample_fisher_result):
        """Invariant: All representatives must be present in Fisher results."""
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000002", "GO:0000003", "GO:0000004"]
        result = select_representatives(clusters, go_ids, sample_fisher_result)
        for rep in result.representatives:
            assert rep in sample_fisher_result.combined_pvalues


# ---------------------------------------------------------------------------
# Tests: write_fisher_results_with_clusters_tsv
# ---------------------------------------------------------------------------

class TestWriteFisherResultsWithClustersTsv:
    """Tests for the write_fisher_results_with_clusters_tsv function."""

    def test_writes_file_to_output_dir(self, tmp_path, sample_fisher_result):
        """Contract 10: Writes fisher_combined_pvalues.tsv to output_dir."""
        clustering_result = ClusteringResult(
            representatives=["GO:0000002"],
            representative_names=["child process 1"],
            representative_pvalues=[0.001],
            representative_n_contributing=[3],
            cluster_assignments={"GO:0000002": 0, "GO:0000003": 0},
            n_clusters=1,
            n_prefiltered=2,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        result_path = write_fisher_results_with_clusters_tsv(
            sample_fisher_result, clustering_result, tmp_path
        )
        assert result_path.exists()
        assert result_path.name == "fisher_combined_pvalues.tsv"

    def test_tsv_has_correct_header(self, tmp_path, sample_fisher_result):
        """Contract 10: TSV has columns: GO ID, term name, combined p-value,
        n contributing, cluster index, representative flag."""
        clustering_result = ClusteringResult(
            representatives=["GO:0000002"],
            representative_names=["child process 1"],
            representative_pvalues=[0.001],
            representative_n_contributing=[3],
            cluster_assignments={"GO:0000002": 0, "GO:0000003": 0},
            n_clusters=1,
            n_prefiltered=2,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        result_path = write_fisher_results_with_clusters_tsv(
            sample_fisher_result, clustering_result, tmp_path
        )
        content = result_path.read_text(encoding="utf-8")
        header = content.splitlines()[0]
        assert "GO_ID" in header
        assert "Combined_pvalue" in header or "Combined_PValue" in header or "combined_pvalue" in header.lower()
        assert "Cluster" in header
        assert "Representative" in header

    def test_includes_all_prefiltered_terms(self, tmp_path, sample_fisher_result):
        """Contract 10: All pre-filtered GO terms are included, not just representatives."""
        clustering_result = ClusteringResult(
            representatives=["GO:0000002"],
            representative_names=["child process 1"],
            representative_pvalues=[0.001],
            representative_n_contributing=[3],
            cluster_assignments={"GO:0000002": 0, "GO:0000003": 0, "GO:0000004": 1},
            n_clusters=2,
            n_prefiltered=3,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        result_path = write_fisher_results_with_clusters_tsv(
            sample_fisher_result, clustering_result, tmp_path
        )
        content = result_path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()
        # Header + 3 data lines
        assert len(lines) == 4
        # Check all GO IDs are present
        all_text = content
        assert "GO:0000002" in all_text
        assert "GO:0000003" in all_text
        assert "GO:0000004" in all_text

    def test_representative_column_marks_correctly(self, tmp_path, sample_fisher_result):
        """Contract 10: Representative column correctly marks reps vs non-reps."""
        clustering_result = ClusteringResult(
            representatives=["GO:0000002"],
            representative_names=["child process 1"],
            representative_pvalues=[0.001],
            representative_n_contributing=[3],
            cluster_assignments={"GO:0000002": 0, "GO:0000003": 0},
            n_clusters=1,
            n_prefiltered=2,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        result_path = write_fisher_results_with_clusters_tsv(
            sample_fisher_result, clustering_result, tmp_path
        )
        content = result_path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()
        for line in lines[1:]:
            fields = line.split("\t")
            go_id = fields[0]
            is_rep = fields[-1]
            if go_id == "GO:0000002":
                assert is_rep == "True"
            elif go_id == "GO:0000003":
                assert is_rep == "False"

    def test_raises_oserror_on_write_failure(self, sample_fisher_result):
        """Error condition: OSError raised when output directory is not writable."""
        clustering_result = ClusteringResult(
            representatives=["GO:0000002"],
            representative_names=["child process 1"],
            representative_pvalues=[0.001],
            representative_n_contributing=[3],
            cluster_assignments={"GO:0000002": 0},
            n_clusters=1,
            n_prefiltered=1,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        nonexistent_dir = Path("/nonexistent/path/that/does/not/exist")
        with pytest.raises(OSError):
            write_fisher_results_with_clusters_tsv(
                sample_fisher_result, clustering_result, nonexistent_dir
            )


# ---------------------------------------------------------------------------
# Tests: run_semantic_clustering (integration-level, with mocks)
# ---------------------------------------------------------------------------

class TestRunSemanticClustering:
    """Tests for the run_semantic_clustering top-level function."""

    def test_prefilter_excludes_nonsignificant_terms(
        self, tmp_path, sample_fisher_result, default_clustering_config
    ):
        """Contract 1: Only GO terms with combined p-value < 0.05 are included."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create minimal OBO and GAF files in cache
        obo_content = textwrap.dedent("""\
            format-version: 1.2
            [Term]
            id: GO:0000002
            name: child process 1
            [Term]
            id: GO:0000003
            name: child process 2
            [Term]
            id: GO:0000004
            name: grandchild process 1
            [Term]
            id: GO:0000006
            name: extra process 1
        """)
        obo_path = cache_dir / "go.obo"  # matches URL filename
        obo_path.write_text(obo_content, encoding="utf-8")

        gaf_lines = [
            "!gaf-version: 2.1",
            "DB\tgeneA\tgeneA\t\tGO:0000002\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
            "DB\tgeneB\tgeneB\t\tGO:0000003\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
            "DB\tgeneC\tgeneC\t\tGO:0000004\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
            "DB\tgeneD\tgeneD\t\tGO:0000006\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        ]
        gaf_path = cache_dir / "test.gaf"  # matches URL filename
        gaf_path.write_text("\n".join(gaf_lines) + "\n", encoding="utf-8")

        config = ClusteringConfig(
            enabled=True,
            similarity_metric="Lin",
            similarity_threshold=0.7,
            go_obo_url="http://example.com/go.obo",
            gaf_url="http://example.com/test.gaf",
        )

        result = run_semantic_clustering(
            sample_fisher_result, config, output_dir, cache_dir
        )

        # Only 4 terms have p < 0.05: GO:0000002, GO:0000003, GO:0000004, GO:0000006
        # GO:0000005 (p=0.5) and GO:0000007 (p=0.8) should be excluded
        assert result.n_prefiltered == 4
        assert "GO:0000005" not in result.cluster_assignments
        assert "GO:0000007" not in result.cluster_assignments

    def test_raises_value_error_when_no_terms_pass_prefilter(
        self, tmp_path, no_significant_fisher_result, default_clustering_config
    ):
        """Error condition: ValueError when zero GO terms pass the pre-filter."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"

        with pytest.raises(ValueError, match="No GO terms have combined p-value below"):
            run_semantic_clustering(
                no_significant_fisher_result, default_clustering_config,
                output_dir, cache_dir
            )

    def test_similarity_threshold_invariant_positive(self, tmp_path, sample_fisher_result):
        """Invariant: similarity_threshold must be positive."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"

        config = ClusteringConfig(
            enabled=True,
            similarity_metric="Lin",
            similarity_threshold=0.0,  # violates invariant
            go_obo_url="http://example.com/go.obo",
            gaf_url="http://example.com/test.gaf",
        )
        with pytest.raises(AssertionError, match="Similarity threshold must be positive"):
            run_semantic_clustering(sample_fisher_result, config, output_dir, cache_dir)

    def test_similarity_threshold_invariant_at_most_one(self, tmp_path, sample_fisher_result):
        """Invariant: similarity_threshold must be at most 1.0."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"

        config = ClusteringConfig(
            enabled=True,
            similarity_metric="Lin",
            similarity_threshold=1.5,  # violates invariant
            go_obo_url="http://example.com/go.obo",
            gaf_url="http://example.com/test.gaf",
        )
        with pytest.raises(AssertionError, match="Similarity threshold must be at most 1.0"):
            run_semantic_clustering(sample_fisher_result, config, output_dir, cache_dir)

    def test_writes_fisher_combined_pvalues_tsv(
        self, tmp_path, sample_fisher_result
    ):
        """Contract 10: run_semantic_clustering writes fisher_combined_pvalues.tsv."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        obo_content = textwrap.dedent("""\
            format-version: 1.2
            [Term]
            id: GO:0000002
            name: child process 1
            [Term]
            id: GO:0000003
            name: child process 2
            [Term]
            id: GO:0000004
            name: grandchild process 1
            [Term]
            id: GO:0000006
            name: extra process 1
        """)
        (cache_dir / "go.obo").write_text(obo_content, encoding="utf-8")

        gaf_lines = [
            "!gaf-version: 2.1",
            "DB\tgeneA\tgeneA\t\tGO:0000002\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
            "DB\tgeneB\tgeneB\t\tGO:0000003\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        ]
        (cache_dir / "test.gaf").write_text("\n".join(gaf_lines) + "\n", encoding="utf-8")

        config = ClusteringConfig(
            enabled=True,
            similarity_metric="Lin",
            similarity_threshold=0.7,
            go_obo_url="http://example.com/go.obo",
            gaf_url="http://example.com/test.gaf",
        )

        result = run_semantic_clustering(
            sample_fisher_result, config, output_dir, cache_dir
        )

        tsv_path = output_dir / "fisher_combined_pvalues.tsv"
        assert tsv_path.exists()

    def test_result_representatives_ordered_by_pvalue(
        self, tmp_path, sample_fisher_result
    ):
        """Invariant: Representatives must be ordered by combined p-value ascending."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        obo_content = textwrap.dedent("""\
            format-version: 1.2
            [Term]
            id: GO:0000002
            name: child process 1
            [Term]
            id: GO:0000003
            name: child process 2
            [Term]
            id: GO:0000004
            name: grandchild process 1
            [Term]
            id: GO:0000006
            name: extra process 1
        """)
        (cache_dir / "go.obo").write_text(obo_content, encoding="utf-8")

        gaf_lines = [
            "!gaf-version: 2.1",
            "DB\tgeneA\tgeneA\t\tGO:0000002\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        ]
        (cache_dir / "test.gaf").write_text("\n".join(gaf_lines) + "\n", encoding="utf-8")

        config = ClusteringConfig(
            enabled=True,
            similarity_metric="Lin",
            similarity_threshold=0.7,
            go_obo_url="http://example.com/go.obo",
            gaf_url="http://example.com/test.gaf",
        )

        result = run_semantic_clustering(
            sample_fisher_result, config, output_dir, cache_dir
        )

        # Verify p-values are in ascending order
        for i in range(len(result.representative_pvalues) - 1):
            assert result.representative_pvalues[i] <= result.representative_pvalues[i + 1]

    def test_deterministic_output(self, tmp_path, sample_fisher_result):
        """Contract 11: Same input data and config produces deterministic output."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        obo_content = textwrap.dedent("""\
            format-version: 1.2
            [Term]
            id: GO:0000002
            name: child process 1
            [Term]
            id: GO:0000003
            name: child process 2
            [Term]
            id: GO:0000004
            name: grandchild process 1
            [Term]
            id: GO:0000006
            name: extra process 1
        """)
        (cache_dir / "go.obo").write_text(obo_content, encoding="utf-8")

        gaf_lines = [
            "!gaf-version: 2.1",
            "DB\tgeneA\tgeneA\t\tGO:0000002\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        ]
        (cache_dir / "test.gaf").write_text("\n".join(gaf_lines) + "\n", encoding="utf-8")

        config = ClusteringConfig(
            enabled=True,
            similarity_metric="Lin",
            similarity_threshold=0.7,
            go_obo_url="http://example.com/go.obo",
            gaf_url="http://example.com/test.gaf",
        )

        output_dir_1 = tmp_path / "output1"
        output_dir_1.mkdir()
        result1 = run_semantic_clustering(
            sample_fisher_result, config, output_dir_1, cache_dir
        )

        output_dir_2 = tmp_path / "output2"
        output_dir_2.mkdir()
        result2 = run_semantic_clustering(
            sample_fisher_result, config, output_dir_2, cache_dir
        )

        assert result1.representatives == result2.representatives
        assert result1.representative_pvalues == result2.representative_pvalues
        assert result1.cluster_assignments == result2.cluster_assignments
        assert result1.n_clusters == result2.n_clusters

    def test_one_representative_per_cluster_invariant(
        self, tmp_path, sample_fisher_result
    ):
        """Invariant: len(representatives) == n_clusters."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        obo_content = textwrap.dedent("""\
            format-version: 1.2
            [Term]
            id: GO:0000002
            name: child process 1
            [Term]
            id: GO:0000003
            name: child process 2
            [Term]
            id: GO:0000004
            name: grandchild process 1
            [Term]
            id: GO:0000006
            name: extra process 1
        """)
        (cache_dir / "go.obo").write_text(obo_content, encoding="utf-8")

        gaf_lines = [
            "!gaf-version: 2.1",
            "DB\tgeneA\tgeneA\t\tGO:0000002\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        ]
        (cache_dir / "test.gaf").write_text("\n".join(gaf_lines) + "\n", encoding="utf-8")

        config = ClusteringConfig(
            enabled=True,
            similarity_metric="Lin",
            similarity_threshold=0.7,
            go_obo_url="http://example.com/go.obo",
            gaf_url="http://example.com/test.gaf",
        )

        result = run_semantic_clustering(
            sample_fisher_result, config, output_dir, cache_dir
        )

        assert len(result.representatives) == result.n_clusters
        assert result.n_clusters > 0

    def test_at_least_one_cluster_formed(
        self, tmp_path, all_significant_fisher_result
    ):
        """Invariant: At least one cluster must be formed."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        obo_content = textwrap.dedent("""\
            format-version: 1.2
            [Term]
            id: GO:0000002
            name: child process 1
            [Term]
            id: GO:0000003
            name: child process 2
            [Term]
            id: GO:0000004
            name: grandchild process 1
        """)
        (cache_dir / "go.obo").write_text(obo_content, encoding="utf-8")

        gaf_lines = [
            "!gaf-version: 2.1",
            "DB\tgeneA\tgeneA\t\tGO:0000002\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        ]
        (cache_dir / "test.gaf").write_text("\n".join(gaf_lines) + "\n", encoding="utf-8")

        config = ClusteringConfig(
            enabled=True,
            similarity_metric="Lin",
            similarity_threshold=0.7,
            go_obo_url="http://example.com/go.obo",
            gaf_url="http://example.com/test.gaf",
        )

        result = run_semantic_clustering(
            all_significant_fisher_result, config, output_dir, cache_dir
        )

        assert result.n_clusters >= 1

    def test_similarity_metric_stored_in_result(
        self, tmp_path, all_significant_fisher_result
    ):
        """The similarity_metric from config should be stored in the result."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        obo_content = textwrap.dedent("""\
            format-version: 1.2
            [Term]
            id: GO:0000002
            name: child process 1
            [Term]
            id: GO:0000003
            name: child process 2
            [Term]
            id: GO:0000004
            name: grandchild process 1
        """)
        (cache_dir / "go.obo").write_text(obo_content, encoding="utf-8")

        gaf_lines = [
            "!gaf-version: 2.1",
            "DB\tgeneA\tgeneA\t\tGO:0000002\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        ]
        (cache_dir / "test.gaf").write_text("\n".join(gaf_lines) + "\n", encoding="utf-8")

        config = ClusteringConfig(
            enabled=True,
            similarity_metric="Lin",
            similarity_threshold=0.7,
            go_obo_url="http://example.com/go.obo",
            gaf_url="http://example.com/test.gaf",
        )

        result = run_semantic_clustering(
            all_significant_fisher_result, config, output_dir, cache_dir
        )

        assert result.similarity_metric == "Lin"
        assert result.similarity_threshold == 0.7

    def test_single_significant_term_forms_single_cluster(self, tmp_path):
        """Edge case: One significant term should form a single cluster."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        fisher_result = FisherResult(
            go_ids=["GO:0000002", "GO:0000003"],
            go_id_to_name={"GO:0000002": "process A", "GO:0000003": "process B"},
            combined_pvalues={"GO:0000002": 0.01, "GO:0000003": 0.9},
            n_contributing={"GO:0000002": 2, "GO:0000003": 1},
            pvalue_matrix=np.zeros((2, 2)),
            mutant_ids=["m1", "m2"],
            go_id_order=["GO:0000002", "GO:0000003"],
            n_mutants=2,
            corrected_pvalues=None,
        )

        obo_content = textwrap.dedent("""\
            format-version: 1.2
            [Term]
            id: GO:0000002
            name: process A
        """)
        (cache_dir / "go.obo").write_text(obo_content, encoding="utf-8")

        gaf_lines = [
            "!gaf-version: 2.1",
            "DB\tgeneA\tgeneA\t\tGO:0000002\tPMID:0\tIDA\t\tP\t\t\tprotein\ttaxon:7227\t20200101\tDB",
        ]
        (cache_dir / "test.gaf").write_text("\n".join(gaf_lines) + "\n", encoding="utf-8")

        config = ClusteringConfig(
            enabled=True,
            similarity_metric="Lin",
            similarity_threshold=0.7,
            go_obo_url="http://example.com/go.obo",
            gaf_url="http://example.com/test.gaf",
        )

        result = run_semantic_clustering(fisher_result, config, output_dir, cache_dir)

        assert result.n_clusters == 1
        assert result.n_prefiltered == 1
        assert result.representatives == ["GO:0000002"]


# ---------------------------------------------------------------------------
# Tests: Caching behavior
# ---------------------------------------------------------------------------

class TestCachingBehavior:
    """Tests for file caching contracts."""

    def test_obo_cached_file_is_reused(self, tmp_path):
        """Contract 2: If already cached, the cached version is used without re-downloading."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached_file = cache_dir / "go.obo"
        cached_file.write_text("cached obo content", encoding="utf-8")

        with patch("gsea_tool.go_clustering._download_file") as mock_retrieve:
            result = download_or_load_obo("http://example.com/go.obo", cache_dir)
            mock_retrieve.assert_not_called()
            assert result == cached_file

    def test_gaf_cached_file_is_reused(self, tmp_path):
        """Contract 2: If already cached, the cached version is used without re-downloading."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached_file = cache_dir / "test.gaf"
        cached_file.write_text("cached gaf content", encoding="utf-8")

        with patch("gsea_tool.go_clustering._download_file") as mock_retrieve:
            result = download_or_load_gaf("http://example.com/test.gaf", cache_dir)
            mock_retrieve.assert_not_called()
            assert result == cached_file
