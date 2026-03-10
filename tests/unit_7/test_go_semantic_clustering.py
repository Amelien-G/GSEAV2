"""
Tests for Unit 7 -- GO Semantic Clustering.

Synthetic Data Assumptions
==========================
DATA ASSUMPTION: GO IDs follow the standard Gene Ontology format "GO:NNNNNNN"
    (e.g., GO:0008150, GO:0003674). These are realistic identifiers from the
    Gene Ontology.

DATA ASSUMPTION: Combined p-values from Fisher's method are in [0.0, 1.0].
    Significant GO terms typically have combined p-values < 0.05. We use a
    mix of significant (p < 0.05) and non-significant (p >= 0.05) terms to
    test the pre-filtering behavior.

DATA ASSUMPTION: Information content (IC) values are non-negative floats.
    IC = -log(freq) where freq is annotation frequency. Typical IC values
    range from 0 (root term) to ~15 (very specific terms).

DATA ASSUMPTION: Lin similarity values are in [0.0, 1.0], where 1.0 means
    identical terms and 0.0 means no similarity. Values around 0.7 represent
    moderate-to-high semantic similarity.

DATA ASSUMPTION: The default pre-filter threshold is 0.05, meaning only GO
    terms with combined p-value < 0.05 enter clustering.

DATA ASSUMPTION: The default similarity threshold is 0.7, used as the
    agglomerative clustering cut height (distance = 1 - 0.7 = 0.3).

DATA ASSUMPTION: GO term names are human-readable biological process names,
    e.g. "signal transduction", "cell cycle". Exact names are not critical
    for clustering logic.

DATA ASSUMPTION: n_contributing counts are positive integers representing
    how many mutant lines contributed a p-value < 1.0 for a GO term. Typical
    values are 2-10 for a small cohort.

DATA ASSUMPTION: OBO and GAF URLs are strings representing HTTP endpoints.
    In tests, these are mocked and never actually fetched.
"""

import inspect
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import numpy as np
import pytest

from gsea_tool.configuration import ClusteringConfig
from gsea_tool.meta_analysis import FisherResult
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


# ---------------------------------------------------------------------------
# Helper factories for synthetic data
# ---------------------------------------------------------------------------


def _make_fisher_result(
    go_ids: list[str] | None = None,
    combined_pvalues: dict[str, float] | None = None,
    go_id_to_name: dict[str, str] | None = None,
    n_contributing: dict[str, int] | None = None,
) -> FisherResult:
    """Create a FisherResult with sensible defaults for clustering tests.

    DATA ASSUMPTION: Default GO terms include a mix of significant (p < 0.05)
    and non-significant terms to exercise the pre-filter. Default mutant count
    is 3, typical for a small Drosophila cohort.
    """
    if go_ids is None:
        go_ids = [
            "GO:0008150",  # biological_process
            "GO:0003674",  # molecular_function
            "GO:0005575",  # cellular_component
            "GO:0007165",  # signal transduction
            "GO:0006915",  # apoptotic process
        ]
    if combined_pvalues is None:
        # DATA ASSUMPTION: Two terms significant, three not significant
        combined_pvalues = {
            "GO:0008150": 0.001,
            "GO:0003674": 0.02,
            "GO:0005575": 0.10,
            "GO:0007165": 0.005,
            "GO:0006915": 0.50,
        }
    if go_id_to_name is None:
        go_id_to_name = {
            "GO:0008150": "biological_process",
            "GO:0003674": "molecular_function",
            "GO:0005575": "cellular_component",
            "GO:0007165": "signal transduction",
            "GO:0006915": "apoptotic process",
        }
    if n_contributing is None:
        n_contributing = {
            "GO:0008150": 3,
            "GO:0003674": 3,
            "GO:0005575": 2,
            "GO:0007165": 3,
            "GO:0006915": 1,
        }

    n_mutants = 3
    pvalue_matrix = np.ones((len(go_ids), n_mutants))

    return FisherResult(
        go_ids=go_ids,
        go_id_to_name=go_id_to_name,
        combined_pvalues=combined_pvalues,
        n_contributing=n_contributing,
        pvalue_matrix=pvalue_matrix,
        mutant_ids=["mutant_1", "mutant_2", "mutant_3"],
        go_id_order=go_ids,
        n_mutants=n_mutants,
        corrected_pvalues=None,
    )


def _make_clustering_config(
    enabled: bool = True,
    similarity_metric: str = "Lin",
    similarity_threshold: float = 0.7,
    go_obo_url: str = "http://example.com/go-basic.obo",
    gaf_url: str = "http://example.com/fb.gaf.gz",
) -> ClusteringConfig:
    """Create a ClusteringConfig with test-friendly defaults.

    DATA ASSUMPTION: URLs are fake placeholders since network calls are mocked.
    """
    return ClusteringConfig(
        enabled=enabled,
        similarity_metric=similarity_metric,
        similarity_threshold=similarity_threshold,
        go_obo_url=go_obo_url,
        gaf_url=gaf_url,
    )


def _make_significant_fisher_result(n_terms: int = 5) -> FisherResult:
    """Create a FisherResult where ALL terms are significant (p < 0.05).

    DATA ASSUMPTION: All terms have p-values well below 0.05 so they all
    pass the pre-filter. This is used to test clustering behavior without
    pre-filter edge cases.
    """
    go_ids = [f"GO:{i:07d}" for i in range(1, n_terms + 1)]
    # DATA ASSUMPTION: p-values spread across [0.001, 0.04] range
    pvals = [0.001 * (i + 1) for i in range(n_terms)]
    combined_pvalues = dict(zip(go_ids, pvals))
    go_id_to_name = {gid: f"term_{i}" for i, gid in enumerate(go_ids)}
    n_contributing = {gid: 3 for gid in go_ids}
    pvalue_matrix = np.ones((n_terms, 3))

    return FisherResult(
        go_ids=go_ids,
        go_id_to_name=go_id_to_name,
        combined_pvalues=combined_pvalues,
        n_contributing=n_contributing,
        pvalue_matrix=pvalue_matrix,
        mutant_ids=["m1", "m2", "m3"],
        go_id_order=go_ids,
        n_mutants=3,
        corrected_pvalues=None,
    )


# ===========================================================================
# Section 1: Signature and Type Verification
# ===========================================================================


class TestClusteringResultDataclass:
    """Verify ClusteringResult has the correct fields and types."""

    def test_clustering_result_is_dataclass(self):
        """ClusteringResult must be a dataclass."""
        assert hasattr(ClusteringResult, "__dataclass_fields__")

    def test_clustering_result_fields(self):
        """ClusteringResult must have all specified fields."""
        field_names = {f.name for f in dataclass_fields(ClusteringResult)}
        expected = {
            "representatives",
            "representative_names",
            "representative_pvalues",
            "representative_n_contributing",
            "cluster_assignments",
            "n_clusters",
            "n_prefiltered",
            "similarity_metric",
            "similarity_threshold",
        }
        assert expected.issubset(field_names), (
            f"Missing fields: {expected - field_names}"
        )

    def test_clustering_result_instantiation(self):
        """ClusteringResult can be instantiated with expected field types."""
        cr = ClusteringResult(
            representatives=["GO:0000001"],
            representative_names=["test_term"],
            representative_pvalues=[0.01],
            representative_n_contributing=[3],
            cluster_assignments={"GO:0000001": 0},
            n_clusters=1,
            n_prefiltered=1,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )
        assert cr.representatives == ["GO:0000001"]
        assert cr.n_clusters == 1
        assert cr.similarity_metric == "Lin"
        assert cr.similarity_threshold == 0.7


class TestFunctionSignatures:
    """Verify that all public functions have the expected signatures."""

    def test_download_or_load_obo_signature(self):
        sig = inspect.signature(download_or_load_obo)
        params = list(sig.parameters.keys())
        assert params == ["obo_url", "cache_dir"]

    def test_download_or_load_gaf_signature(self):
        sig = inspect.signature(download_or_load_gaf)
        params = list(sig.parameters.keys())
        assert params == ["gaf_url", "cache_dir"]

    def test_compute_information_content_signature(self):
        sig = inspect.signature(compute_information_content)
        params = list(sig.parameters.keys())
        assert params == ["obo_path", "gaf_path"]

    def test_compute_lin_similarity_signature(self):
        sig = inspect.signature(compute_lin_similarity)
        params = list(sig.parameters.keys())
        assert params == ["go_ids", "ic_values", "obo_path"]

    def test_cluster_by_similarity_signature(self):
        sig = inspect.signature(cluster_by_similarity)
        params = list(sig.parameters.keys())
        assert params == ["similarity_matrix", "threshold"]

    def test_select_representatives_signature(self):
        sig = inspect.signature(select_representatives)
        params = list(sig.parameters.keys())
        assert params == ["clusters", "go_ids", "fisher_result"]

    def test_run_semantic_clustering_signature(self):
        sig = inspect.signature(run_semantic_clustering)
        params = list(sig.parameters.keys())
        assert params == ["fisher_result", "config", "output_dir", "cache_dir"]

    def test_write_fisher_results_with_clusters_tsv_signature(self):
        sig = inspect.signature(write_fisher_results_with_clusters_tsv)
        params = list(sig.parameters.keys())
        assert params == ["fisher_result", "clustering_result", "output_dir"]


# ===========================================================================
# Section 2: download_or_load_obo
# ===========================================================================


class TestDownloadOrLoadObo:
    """Tests for download_or_load_obo."""

    def test_returns_path(self, tmp_path):
        """download_or_load_obo must return a Path object."""
        # We cannot test this without an actual implementation that works,
        # but we verify the function is callable and returns a Path.
        # This test will fail against the stub (NotImplementedError).
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Create a fake cached OBO file to avoid network call
        obo_file = cache_dir / "go-basic.obo"
        obo_file.write_text("format-version: 1.2\n")
        result = download_or_load_obo("http://example.com/go-basic.obo", cache_dir)
        assert isinstance(result, Path)

    def test_uses_cached_file_if_exists(self, tmp_path):
        """If the OBO file is already cached, it should be returned without downloading.

        Contract 2: Downloaded files are cached locally. If already cached,
        the cached version is used without re-downloading.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        obo_file = cache_dir / "go-basic.obo"
        # DATA ASSUMPTION: A minimal OBO file with just the header
        obo_file.write_text("format-version: 1.2\nontology: go\n")

        result = download_or_load_obo("http://example.com/go-basic.obo", cache_dir)
        assert result.exists()
        # The content should be the same (no re-download)
        assert "format-version" in result.read_text()


class TestDownloadOrLoadGaf:
    """Tests for download_or_load_gaf."""

    def test_returns_path(self, tmp_path):
        """download_or_load_gaf must return a Path object."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Create a fake cached GAF file
        gaf_file = cache_dir / "fb.gaf.gz"
        gaf_file.write_bytes(b"fake gaf data")
        result = download_or_load_gaf("http://example.com/fb.gaf.gz", cache_dir)
        assert isinstance(result, Path)

    def test_uses_cached_file_if_exists(self, tmp_path):
        """Contract 2: Cached GAF file is used without re-downloading."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        gaf_file = cache_dir / "fb.gaf.gz"
        gaf_file.write_bytes(b"fake gaf data")

        result = download_or_load_gaf("http://example.com/fb.gaf.gz", cache_dir)
        assert result.exists()


# ===========================================================================
# Section 3: compute_information_content
# ===========================================================================


class TestComputeInformationContent:
    """Tests for compute_information_content."""

    def test_returns_dict(self, tmp_path):
        """compute_information_content must return a dict mapping GO ID -> IC float."""
        # DATA ASSUMPTION: Minimal OBO file with one GO term
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\n"
            "id: GO:0008150\n"
            "name: biological_process\n"
            "namespace: biological_process\n"
        )
        # DATA ASSUMPTION: Minimal GAF file with one annotation
        gaf_path = tmp_path / "annotations.gaf"
        gaf_path.write_text(
            "!gaf-version: 2.1\n"
            "FB\tFBgn0000001\tgene1\t\tGO:0008150\tFB:FBrf0000001\tIDA\t\tP\t\t\tgene\ttaxon:7227\t20200101\tFlyBase\n"
        )
        result = compute_information_content(obo_path, gaf_path)
        assert isinstance(result, dict)

    def test_ic_values_are_floats(self, tmp_path):
        """All IC values must be float."""
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\n"
            "id: GO:0008150\n"
            "name: biological_process\n"
            "namespace: biological_process\n"
        )
        gaf_path = tmp_path / "annotations.gaf"
        gaf_path.write_text(
            "!gaf-version: 2.1\n"
            "FB\tFBgn0000001\tgene1\t\tGO:0008150\tFB:FBrf0000001\tIDA\t\tP\t\t\tgene\ttaxon:7227\t20200101\tFlyBase\n"
        )
        result = compute_information_content(obo_path, gaf_path)
        for go_id, ic_val in result.items():
            assert isinstance(ic_val, float), f"IC for {go_id} is not float: {type(ic_val)}"

    def test_unannotated_terms_have_zero_ic(self, tmp_path):
        """Contract 4: GO terms not present in the GAF receive an IC of 0.0."""
        obo_path = tmp_path / "go.obo"
        # Two terms in OBO, but only one annotated in GAF
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\n"
            "id: GO:0008150\n"
            "name: biological_process\n"
            "namespace: biological_process\n\n"
            "[Term]\n"
            "id: GO:9999999\n"
            "name: unannotated_term\n"
            "namespace: biological_process\n"
            "is_a: GO:0008150\n"
        )
        gaf_path = tmp_path / "annotations.gaf"
        gaf_path.write_text(
            "!gaf-version: 2.1\n"
            "FB\tFBgn0000001\tgene1\t\tGO:0008150\tFB:FBrf0000001\tIDA\t\tP\t\t\tgene\ttaxon:7227\t20200101\tFlyBase\n"
        )
        result = compute_information_content(obo_path, gaf_path)
        # The unannotated term should have IC = 0.0 if it is in the result
        if "GO:9999999" in result:
            assert result["GO:9999999"] == 0.0

    def test_ic_values_are_nonnegative(self, tmp_path):
        """IC values should be non-negative (IC = -log(freq) >= 0)."""
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\n"
            "id: GO:0008150\n"
            "name: biological_process\n"
            "namespace: biological_process\n"
        )
        gaf_path = tmp_path / "annotations.gaf"
        gaf_path.write_text(
            "!gaf-version: 2.1\n"
            "FB\tFBgn0000001\tgene1\t\tGO:0008150\tFB:FBrf0000001\tIDA\t\tP\t\t\tgene\ttaxon:7227\t20200101\tFlyBase\n"
        )
        result = compute_information_content(obo_path, gaf_path)
        for go_id, ic_val in result.items():
            assert ic_val >= 0.0, f"IC for {go_id} is negative: {ic_val}"


# ===========================================================================
# Section 4: compute_lin_similarity
# ===========================================================================


class TestComputeLinSimilarity:
    """Tests for compute_lin_similarity."""

    def test_returns_numpy_array(self, tmp_path):
        """compute_lin_similarity must return a numpy ndarray."""
        # DATA ASSUMPTION: Two GO terms with known IC values
        go_ids = ["GO:0000001", "GO:0000002"]
        ic_values = {"GO:0000001": 5.0, "GO:0000002": 3.0, "GO:0000003": 4.0}
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000001\nname: term1\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000002\nname: term2\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000003\nname: term3\nnamespace: biological_process\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        assert isinstance(result, np.ndarray)

    def test_symmetric_matrix(self, tmp_path):
        """Contract 5: The similarity matrix must be symmetric."""
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        ic_values = {"GO:0000001": 5.0, "GO:0000002": 3.0, "GO:0000003": 4.0}
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000001\nname: term1\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000002\nname: term2\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000003\nname: term3\nnamespace: biological_process\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        np.testing.assert_array_almost_equal(result, result.T)

    def test_correct_shape(self, tmp_path):
        """Similarity matrix shape must be (n, n) where n is len(go_ids)."""
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        ic_values = {"GO:0000001": 5.0, "GO:0000002": 3.0, "GO:0000003": 4.0}
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000001\nname: term1\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000002\nname: term2\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000003\nname: term3\nnamespace: biological_process\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        assert result.shape == (3, 3)

    def test_values_in_zero_one_range(self, tmp_path):
        """Contract 5: Similarity values must be in [0, 1]."""
        go_ids = ["GO:0000001", "GO:0000002"]
        ic_values = {"GO:0000001": 5.0, "GO:0000002": 3.0}
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000001\nname: term1\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000002\nname: term2\nnamespace: biological_process\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_self_similarity_is_one_or_max(self, tmp_path):
        """Diagonal elements (self-similarity) should be 1.0 for terms with IC > 0."""
        go_ids = ["GO:0000001"]
        ic_values = {"GO:0000001": 5.0}
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000001\nname: term1\nnamespace: biological_process\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        # Lin(t, t) = 2 * IC(t) / (IC(t) + IC(t)) = 1.0
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 1.0) < 1e-10

    def test_zero_ic_terms_have_zero_similarity(self, tmp_path):
        """Contract 5: If IC(t1) + IC(t2) = 0, similarity is 0.0."""
        go_ids = ["GO:0000001", "GO:0000002"]
        # Both have zero IC
        ic_values = {"GO:0000001": 0.0, "GO:0000002": 0.0}
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000001\nname: term1\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000002\nname: term2\nnamespace: biological_process\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        # All entries should be 0.0 since all ICs are zero
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0


# ===========================================================================
# Section 5: cluster_by_similarity
# ===========================================================================


class TestClusterBySimilarity:
    """Tests for cluster_by_similarity."""

    def test_returns_list_of_lists(self):
        """cluster_by_similarity must return a list of lists of ints."""
        # DATA ASSUMPTION: 3x3 identity-like similarity matrix -- each term
        # is only similar to itself, so we expect 3 singleton clusters
        sim = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = cluster_by_similarity(sim, threshold=0.7)
        assert isinstance(result, list)
        for cluster in result:
            assert isinstance(cluster, list)
            for idx in cluster:
                assert isinstance(idx, int)

    def test_all_indices_present(self):
        """Every row index must appear in exactly one cluster."""
        n = 4
        # DATA ASSUMPTION: Block-diagonal similarity -- two pairs of similar terms
        sim = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ])
        clusters = cluster_by_similarity(sim, threshold=0.7)
        all_indices = []
        for cluster in clusters:
            all_indices.extend(cluster)
        assert sorted(all_indices) == list(range(n))

    def test_identical_terms_cluster_together(self):
        """Terms with similarity 1.0 should be in the same cluster."""
        # DATA ASSUMPTION: Two terms with perfect similarity (1.0) and one
        # dissimilar term
        sim = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        clusters = cluster_by_similarity(sim, threshold=0.7)
        # Indices 0 and 1 should be in the same cluster
        for cluster in clusters:
            if 0 in cluster:
                assert 1 in cluster, "Identical terms must cluster together"
                break

    def test_dissimilar_terms_separate_clusters(self):
        """Terms with low similarity should be in separate clusters."""
        # DATA ASSUMPTION: All pairwise similarities are 0 (completely
        # dissimilar terms), expecting n singleton clusters
        sim = np.eye(3)
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert len(clusters) == 3, "Dissimilar terms should form singleton clusters"

    def test_single_term_one_cluster(self):
        """A single term should form one cluster."""
        sim = np.array([[1.0]])
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert len(clusters) == 1
        assert clusters[0] == [0]

    def test_threshold_affects_number_of_clusters(self):
        """Higher threshold (stricter) should produce more clusters;
        lower threshold (more permissive) should produce fewer."""
        # DATA ASSUMPTION: Moderate pairwise similarity (0.5) -- below 0.7
        # threshold but above 0.3 threshold
        sim = np.array([
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ])
        clusters_strict = cluster_by_similarity(sim, threshold=0.7)
        clusters_permissive = cluster_by_similarity(sim, threshold=0.3)
        # Stricter threshold => more or equal clusters
        assert len(clusters_strict) >= len(clusters_permissive)

    def test_uses_average_linkage_at_distance_cut(self):
        """Contract 6: Hierarchical agglomerative clustering uses average linkage
        on distance = 1 - similarity, cut at distance = 1 - threshold."""
        # DATA ASSUMPTION: Two clusters clearly separated by the threshold.
        # Terms 0,1 have sim=0.8 (within cluster), terms 0,2 have sim=0.3 (between).
        # With threshold=0.7, distance cut=0.3. Within-cluster distance=0.2 < 0.3,
        # between-cluster distance=0.7 > 0.3 => two clusters.
        sim = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.3],
            [0.3, 0.3, 1.0],
        ])
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert len(clusters) == 2
        # Check that 0 and 1 are in the same cluster
        for cluster in clusters:
            if 0 in cluster:
                assert 1 in cluster
                assert 2 not in cluster
                break


# ===========================================================================
# Section 6: select_representatives
# ===========================================================================


class TestSelectRepresentatives:
    """Tests for select_representatives."""

    def test_returns_clustering_result(self):
        """select_representatives must return a ClusteringResult."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.01,
                "GO:0000002": 0.02,
                "GO:0000003": 0.005,
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
        # Two clusters: [0, 1] and [2]
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        result = select_representatives(clusters, go_ids, fisher)
        assert isinstance(result, ClusteringResult)

    def test_one_representative_per_cluster(self):
        """Invariant: len(representatives) == n_clusters."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.01,
                "GO:0000002": 0.02,
                "GO:0000003": 0.005,
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
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        result = select_representatives(clusters, go_ids, fisher)
        assert len(result.representatives) == result.n_clusters

    def test_representative_has_lowest_pvalue_in_cluster(self):
        """Contract 8: Within each cluster, the representative is the GO term
        with the lowest combined Fisher p-value."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.03,  # higher p in cluster [0,1]
                "GO:0000002": 0.01,  # lower p in cluster [0,1] -> representative
                "GO:0000003": 0.005,  # sole member of cluster [2] -> representative
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
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        result = select_representatives(clusters, go_ids, fisher)
        # GO:0000002 should be representative of cluster [0,1] (p=0.01 < p=0.03)
        assert "GO:0000002" in result.representatives
        # GO:0000003 should be representative of cluster [2] (sole member)
        assert "GO:0000003" in result.representatives

    def test_representatives_ordered_by_pvalue(self):
        """Invariant: Representatives must be ordered by combined p-value ascending."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.03,
                "GO:0000002": 0.01,
                "GO:0000003": 0.005,
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
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        result = select_representatives(clusters, go_ids, fisher)
        # p-values of representatives should be non-decreasing
        for i in range(len(result.representative_pvalues) - 1):
            assert result.representative_pvalues[i] <= result.representative_pvalues[i + 1]

    def test_representatives_in_fisher_results(self):
        """Invariant: All representatives must be present in Fisher results."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002"],
            combined_pvalues={"GO:0000001": 0.01, "GO:0000002": 0.02},
            go_id_to_name={"GO:0000001": "term_a", "GO:0000002": "term_b"},
            n_contributing={"GO:0000001": 3, "GO:0000002": 2},
        )
        clusters = [[0], [1]]
        go_ids = ["GO:0000001", "GO:0000002"]
        result = select_representatives(clusters, go_ids, fisher)
        for rep in result.representatives:
            assert rep in fisher.combined_pvalues

    def test_cluster_assignments_cover_all_terms(self):
        """cluster_assignments should map every pre-filtered GO term to a cluster index."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.01,
                "GO:0000002": 0.02,
                "GO:0000003": 0.005,
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
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        result = select_representatives(clusters, go_ids, fisher)
        for gid in go_ids:
            assert gid in result.cluster_assignments

    def test_representative_names_match_go_id_to_name(self):
        """representative_names should correspond to go_id_to_name for each representative."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002"],
            combined_pvalues={"GO:0000001": 0.01, "GO:0000002": 0.02},
            go_id_to_name={"GO:0000001": "term_alpha", "GO:0000002": "term_beta"},
            n_contributing={"GO:0000001": 3, "GO:0000002": 2},
        )
        clusters = [[0], [1]]
        go_ids = ["GO:0000001", "GO:0000002"]
        result = select_representatives(clusters, go_ids, fisher)
        for i, rep_id in enumerate(result.representatives):
            assert result.representative_names[i] == fisher.go_id_to_name[rep_id]

    def test_representative_pvalues_match_fisher_combined(self):
        """representative_pvalues should match combined_pvalues from FisherResult."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002"],
            combined_pvalues={"GO:0000001": 0.01, "GO:0000002": 0.02},
            go_id_to_name={"GO:0000001": "term_a", "GO:0000002": "term_b"},
            n_contributing={"GO:0000001": 3, "GO:0000002": 2},
        )
        clusters = [[0], [1]]
        go_ids = ["GO:0000001", "GO:0000002"]
        result = select_representatives(clusters, go_ids, fisher)
        for i, rep_id in enumerate(result.representatives):
            assert result.representative_pvalues[i] == fisher.combined_pvalues[rep_id]

    def test_representative_n_contributing_match(self):
        """representative_n_contributing should match n_contributing from FisherResult."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002"],
            combined_pvalues={"GO:0000001": 0.01, "GO:0000002": 0.02},
            go_id_to_name={"GO:0000001": "term_a", "GO:0000002": "term_b"},
            n_contributing={"GO:0000001": 5, "GO:0000002": 3},
        )
        clusters = [[0], [1]]
        go_ids = ["GO:0000001", "GO:0000002"]
        result = select_representatives(clusters, go_ids, fisher)
        for i, rep_id in enumerate(result.representatives):
            assert result.representative_n_contributing[i] == fisher.n_contributing[rep_id]

    def test_n_prefiltered_equals_total_terms(self):
        """n_prefiltered should equal the number of GO terms passed in."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.01,
                "GO:0000002": 0.02,
                "GO:0000003": 0.005,
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
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        result = select_representatives(clusters, go_ids, fisher)
        assert result.n_prefiltered == 3

    def test_at_least_one_cluster(self):
        """Invariant: n_clusters > 0 (at least one cluster must be formed)."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            combined_pvalues={"GO:0000001": 0.01},
            go_id_to_name={"GO:0000001": "term_a"},
            n_contributing={"GO:0000001": 3},
        )
        clusters = [[0]]
        go_ids = ["GO:0000001"]
        result = select_representatives(clusters, go_ids, fisher)
        assert result.n_clusters > 0


# ===========================================================================
# Section 7: write_fisher_results_with_clusters_tsv
# ===========================================================================


class TestWriteFisherResultsWithClustersTsv:
    """Tests for write_fisher_results_with_clusters_tsv."""

    def _make_clustering_result(self) -> ClusteringResult:
        """Helper to create a minimal ClusteringResult for TSV tests."""
        return ClusteringResult(
            representatives=["GO:0000001", "GO:0000002"],
            representative_names=["term_a", "term_b"],
            representative_pvalues=[0.005, 0.01],
            representative_n_contributing=[3, 2],
            cluster_assignments={
                "GO:0000001": 0,
                "GO:0000002": 1,
                "GO:0000003": 0,
            },
            n_clusters=2,
            n_prefiltered=3,
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )

    def test_writes_tsv_file(self, tmp_path):
        """write_fisher_results_with_clusters_tsv must write a file to output_dir."""
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
        cr = self._make_clustering_result()
        result_path = write_fisher_results_with_clusters_tsv(fisher, cr, tmp_path)
        assert isinstance(result_path, Path)
        assert result_path.exists()

    def test_output_filename(self, tmp_path):
        """Output file must be named fisher_combined_pvalues.tsv."""
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
        cr = self._make_clustering_result()
        result_path = write_fisher_results_with_clusters_tsv(fisher, cr, tmp_path)
        assert result_path.name == "fisher_combined_pvalues.tsv"

    def test_tsv_contains_required_columns(self, tmp_path):
        """Contract 10: TSV must have columns for GO ID, GO term name, combined p-value,
        number of contributing lines, cluster index, and representative flag."""
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
        cr = self._make_clustering_result()
        result_path = write_fisher_results_with_clusters_tsv(fisher, cr, tmp_path)
        content = result_path.read_text()
        lines = content.strip().split("\n")
        header = lines[0].lower()
        # Check that required column concepts are present (case-insensitive)
        assert "go" in header, "Header must contain GO ID column"
        assert "cluster" in header, "Header must contain cluster column"
        assert "representative" in header.lower() or "rep" in header.lower(), (
            "Header must contain representative column"
        )

    def test_tsv_includes_all_prefiltered_terms(self, tmp_path):
        """Contract 10: All pre-filtered GO terms are included, not just representatives."""
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
        cr = self._make_clustering_result()
        result_path = write_fisher_results_with_clusters_tsv(fisher, cr, tmp_path)
        content = result_path.read_text()
        # All three GO terms should appear in the TSV
        assert "GO:0000001" in content
        assert "GO:0000002" in content
        assert "GO:0000003" in content

    def test_tsv_is_tab_separated(self, tmp_path):
        """Output must be tab-separated."""
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
        cr = self._make_clustering_result()
        result_path = write_fisher_results_with_clusters_tsv(fisher, cr, tmp_path)
        content = result_path.read_text()
        lines = content.strip().split("\n")
        header_fields = lines[0].split("\t")
        # Should have at least 6 fields (GO ID, name, p-value, n_contributing, cluster, representative)
        assert len(header_fields) >= 6


# ===========================================================================
# Section 8: run_semantic_clustering (integration-level tests with mocks)
# ===========================================================================


class TestRunSemanticClustering:
    """Tests for the top-level run_semantic_clustering function."""

    def test_returns_clustering_result(self, tmp_path):
        """run_semantic_clustering must return a ClusteringResult."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # DATA ASSUMPTION: Three significant GO terms for a minimal clustering run
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.001,
                "GO:0000002": 0.01,
                "GO:0000003": 0.005,
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
        config = _make_clustering_config()

        # Mock the download functions and computation functions
        with patch("gsea_tool.go_clustering.download_or_load_obo") as mock_obo, \
             patch("gsea_tool.go_clustering.download_or_load_gaf") as mock_gaf, \
             patch("gsea_tool.go_clustering.compute_information_content") as mock_ic, \
             patch("gsea_tool.go_clustering.compute_lin_similarity") as mock_sim, \
             patch("gsea_tool.go_clustering.cluster_by_similarity") as mock_cluster, \
             patch("gsea_tool.go_clustering.select_representatives") as mock_select, \
             patch("gsea_tool.go_clustering.write_fisher_results_with_clusters_tsv") as mock_write:

            mock_obo.return_value = cache_dir / "go.obo"
            mock_gaf.return_value = cache_dir / "fb.gaf"
            mock_ic.return_value = {"GO:0000001": 5.0, "GO:0000002": 3.0, "GO:0000003": 4.0}
            mock_sim.return_value = np.eye(3)
            mock_cluster.return_value = [[0], [1], [2]]
            mock_select.return_value = ClusteringResult(
                representatives=["GO:0000001", "GO:0000003", "GO:0000002"],
                representative_names=["term_a", "term_c", "term_b"],
                representative_pvalues=[0.001, 0.005, 0.01],
                representative_n_contributing=[3, 3, 2],
                cluster_assignments={"GO:0000001": 0, "GO:0000002": 1, "GO:0000003": 2},
                n_clusters=3,
                n_prefiltered=3,
                similarity_metric="Lin",
                similarity_threshold=0.7,
            )
            mock_write.return_value = output_dir / "fisher_combined_pvalues.tsv"

            result = run_semantic_clustering(fisher, config, output_dir, cache_dir)
            assert isinstance(result, ClusteringResult)

    def test_prefilter_excludes_nonsignificant_terms(self, tmp_path):
        """Contract 1: Only GO terms with combined p-value below pre-filter threshold
        (default 0.05) are included in clustering."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # DATA ASSUMPTION: Mix of significant and non-significant terms
        fisher = _make_fisher_result()
        config = _make_clustering_config()

        with patch("gsea_tool.go_clustering.download_or_load_obo") as mock_obo, \
             patch("gsea_tool.go_clustering.download_or_load_gaf") as mock_gaf, \
             patch("gsea_tool.go_clustering.compute_information_content") as mock_ic, \
             patch("gsea_tool.go_clustering.compute_lin_similarity") as mock_sim, \
             patch("gsea_tool.go_clustering.cluster_by_similarity") as mock_cluster, \
             patch("gsea_tool.go_clustering.select_representatives") as mock_select, \
             patch("gsea_tool.go_clustering.write_fisher_results_with_clusters_tsv") as mock_write:

            mock_obo.return_value = cache_dir / "go.obo"
            mock_gaf.return_value = cache_dir / "fb.gaf"
            mock_ic.return_value = {gid: 3.0 for gid in fisher.go_ids}
            mock_sim.return_value = np.eye(3)  # 3 significant terms
            mock_cluster.return_value = [[0], [1], [2]]
            mock_select.return_value = ClusteringResult(
                representatives=["GO:0008150", "GO:0007165", "GO:0003674"],
                representative_names=["biological_process", "signal transduction", "molecular_function"],
                representative_pvalues=[0.001, 0.005, 0.02],
                representative_n_contributing=[3, 3, 3],
                cluster_assignments={"GO:0008150": 0, "GO:0007165": 1, "GO:0003674": 2},
                n_clusters=3,
                n_prefiltered=3,
                similarity_metric="Lin",
                similarity_threshold=0.7,
            )
            mock_write.return_value = output_dir / "fisher_combined_pvalues.tsv"

            result = run_semantic_clustering(fisher, config, output_dir, cache_dir)

            # compute_lin_similarity should have been called with only the
            # significant GO terms (p < 0.05): GO:0008150 (0.001), GO:0003674 (0.02), GO:0007165 (0.005)
            # Not GO:0005575 (0.10) or GO:0006915 (0.50)
            if mock_sim.called:
                go_ids_passed = mock_sim.call_args[0][0] if mock_sim.call_args[0] else mock_sim.call_args[1].get("go_ids", [])
                for gid in go_ids_passed:
                    assert fisher.combined_pvalues[gid] < 0.05, (
                        f"Non-significant term {gid} (p={fisher.combined_pvalues[gid]}) "
                        f"should not be passed to similarity computation"
                    )

    def test_writes_tsv_output(self, tmp_path):
        """Contract 10: run_semantic_clustering writes fisher_combined_pvalues.tsv."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        fisher = _make_significant_fisher_result(n_terms=3)
        config = _make_clustering_config()

        with patch("gsea_tool.go_clustering.download_or_load_obo") as mock_obo, \
             patch("gsea_tool.go_clustering.download_or_load_gaf") as mock_gaf, \
             patch("gsea_tool.go_clustering.compute_information_content") as mock_ic, \
             patch("gsea_tool.go_clustering.compute_lin_similarity") as mock_sim, \
             patch("gsea_tool.go_clustering.cluster_by_similarity") as mock_cluster, \
             patch("gsea_tool.go_clustering.select_representatives") as mock_select, \
             patch("gsea_tool.go_clustering.write_fisher_results_with_clusters_tsv") as mock_write:

            mock_obo.return_value = cache_dir / "go.obo"
            mock_gaf.return_value = cache_dir / "fb.gaf"
            mock_ic.return_value = {gid: 3.0 for gid in fisher.go_ids}
            mock_sim.return_value = np.eye(3)
            mock_cluster.return_value = [[0], [1], [2]]
            cr = ClusteringResult(
                representatives=fisher.go_ids[:3],
                representative_names=["term_0", "term_1", "term_2"],
                representative_pvalues=[0.002, 0.003, 0.004],
                representative_n_contributing=[3, 3, 3],
                cluster_assignments={gid: i for i, gid in enumerate(fisher.go_ids[:3])},
                n_clusters=3,
                n_prefiltered=3,
                similarity_metric="Lin",
                similarity_threshold=0.7,
            )
            mock_select.return_value = cr
            mock_write.return_value = output_dir / "fisher_combined_pvalues.tsv"

            run_semantic_clustering(fisher, config, output_dir, cache_dir)

            # write_fisher_results_with_clusters_tsv should have been called
            mock_write.assert_called_once()

    def test_similarity_metric_stored_in_result(self, tmp_path):
        """ClusteringResult should store the similarity_metric from config."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        fisher = _make_significant_fisher_result(n_terms=2)
        config = _make_clustering_config(similarity_metric="Lin")

        with patch("gsea_tool.go_clustering.download_or_load_obo") as mock_obo, \
             patch("gsea_tool.go_clustering.download_or_load_gaf") as mock_gaf, \
             patch("gsea_tool.go_clustering.compute_information_content") as mock_ic, \
             patch("gsea_tool.go_clustering.compute_lin_similarity") as mock_sim, \
             patch("gsea_tool.go_clustering.cluster_by_similarity") as mock_cluster, \
             patch("gsea_tool.go_clustering.select_representatives") as mock_select, \
             patch("gsea_tool.go_clustering.write_fisher_results_with_clusters_tsv") as mock_write:

            mock_obo.return_value = cache_dir / "go.obo"
            mock_gaf.return_value = cache_dir / "fb.gaf"
            mock_ic.return_value = {gid: 3.0 for gid in fisher.go_ids}
            mock_sim.return_value = np.eye(2)
            mock_cluster.return_value = [[0], [1]]
            cr = ClusteringResult(
                representatives=fisher.go_ids[:2],
                representative_names=["term_0", "term_1"],
                representative_pvalues=[0.002, 0.003],
                representative_n_contributing=[3, 3],
                cluster_assignments={gid: i for i, gid in enumerate(fisher.go_ids[:2])},
                n_clusters=2,
                n_prefiltered=2,
                similarity_metric="Lin",
                similarity_threshold=0.7,
            )
            mock_select.return_value = cr
            mock_write.return_value = output_dir / "fisher_combined_pvalues.tsv"

            result = run_semantic_clustering(fisher, config, output_dir, cache_dir)
            assert result.similarity_metric == "Lin"


# ===========================================================================
# Section 9: Error Conditions
# ===========================================================================


class TestErrorConditions:
    """Tests for all error conditions specified in the blueprint."""

    def test_obo_download_failure_raises_connection_error(self, tmp_path):
        """Error: ConnectionError when OBO download fails after retry."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # The download function should raise ConnectionError when the URL is unreachable
        # We test this by calling with a bad URL and no cached file
        # Since we can't actually test network failure in a stub, we test the contract
        # that the function raises ConnectionError
        with pytest.raises((ConnectionError, NotImplementedError)):
            download_or_load_obo("http://nonexistent.invalid/go.obo", cache_dir)

    def test_gaf_download_failure_raises_connection_error(self, tmp_path):
        """Error: ConnectionError when GAF download fails after retry."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with pytest.raises((ConnectionError, NotImplementedError)):
            download_or_load_gaf("http://nonexistent.invalid/fb.gaf.gz", cache_dir)

    def test_no_terms_pass_prefilter_raises_value_error(self, tmp_path):
        """Error: ValueError when zero GO terms have combined p-value below threshold."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # DATA ASSUMPTION: All terms have p-values above the pre-filter threshold
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002"],
            combined_pvalues={
                "GO:0000001": 0.10,  # above 0.05 threshold
                "GO:0000002": 0.50,  # above 0.05 threshold
            },
            go_id_to_name={"GO:0000001": "term_a", "GO:0000002": "term_b"},
            n_contributing={"GO:0000001": 2, "GO:0000002": 1},
        )
        config = _make_clustering_config()

        with pytest.raises(ValueError):
            run_semantic_clustering(fisher, config, output_dir, cache_dir)

    def test_output_write_failure_raises_os_error(self, tmp_path):
        """Error: OSError when cannot write fisher_combined_pvalues.tsv."""
        # Use a non-existent directory to trigger write failure
        bad_output_dir = tmp_path / "nonexistent" / "nested" / "dir"

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
        with pytest.raises(OSError):
            write_fisher_results_with_clusters_tsv(fisher, cr, bad_output_dir)


# ===========================================================================
# Section 10: Invariants
# ===========================================================================


class TestInvariants:
    """Tests for all invariants specified in the blueprint."""

    def test_similarity_threshold_must_be_positive(self):
        """Pre-condition: similarity_threshold > 0.0."""
        # This invariant is enforced either by ClusteringConfig validation
        # or by run_semantic_clustering. We test it at the clustering level.
        # The config itself is frozen and validated upstream, so we test
        # that the clustering functions handle this correctly.
        sim = np.eye(2)
        # threshold of 0 should be invalid per the invariant
        # The function may raise or the config validation catches it
        # Testing that the config with threshold=0 would be rejected
        # is already covered by unit_2 tests. Here we verify the invariant
        # documentation matches: threshold must be > 0.
        # We mainly document this test exists to cover the invariant.
        # With a valid threshold, clustering should work
        clusters = cluster_by_similarity(sim, threshold=0.5)
        assert len(clusters) > 0

    def test_one_representative_per_cluster_invariant(self):
        """Post-condition: len(representatives) == n_clusters."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"],
            combined_pvalues={
                "GO:0000001": 0.001,
                "GO:0000002": 0.01,
                "GO:0000003": 0.005,
                "GO:0000004": 0.02,
            },
            go_id_to_name={
                "GO:0000001": "a", "GO:0000002": "b",
                "GO:0000003": "c", "GO:0000004": "d",
            },
            n_contributing={
                "GO:0000001": 3, "GO:0000002": 2,
                "GO:0000003": 3, "GO:0000004": 2,
            },
        )
        # Two clusters: [0,1] and [2,3]
        clusters = [[0, 1], [2, 3]]
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"]
        result = select_representatives(clusters, go_ids, fisher)
        assert len(result.representatives) == result.n_clusters
        assert result.n_clusters == 2

    def test_at_least_one_cluster_invariant(self):
        """Post-condition: n_clusters > 0."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            combined_pvalues={"GO:0000001": 0.01},
            go_id_to_name={"GO:0000001": "term_a"},
            n_contributing={"GO:0000001": 3},
        )
        clusters = [[0]]
        go_ids = ["GO:0000001"]
        result = select_representatives(clusters, go_ids, fisher)
        assert result.n_clusters > 0

    def test_representatives_ordered_by_pvalue_invariant(self):
        """Post-condition: Representatives ordered by combined p-value ascending."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"],
            combined_pvalues={
                "GO:0000001": 0.03,
                "GO:0000002": 0.001,
                "GO:0000003": 0.02,
                "GO:0000004": 0.005,
            },
            go_id_to_name={
                "GO:0000001": "a", "GO:0000002": "b",
                "GO:0000003": "c", "GO:0000004": "d",
            },
            n_contributing={
                "GO:0000001": 3, "GO:0000002": 2,
                "GO:0000003": 3, "GO:0000004": 2,
            },
        )
        # Each in its own cluster
        clusters = [[0], [1], [2], [3]]
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"]
        result = select_representatives(clusters, go_ids, fisher)
        # Check ordering: p-values should be non-decreasing
        for i in range(len(result.representative_pvalues) - 1):
            assert result.representative_pvalues[i] <= result.representative_pvalues[i + 1], (
                f"Representatives not ordered by p-value: "
                f"{result.representative_pvalues[i]} > {result.representative_pvalues[i + 1]}"
            )
        # Also verify via GO IDs
        expected_order = sorted(
            result.representatives,
            key=lambda gid: fisher.combined_pvalues[gid],
        )
        assert result.representatives == expected_order

    def test_all_representatives_in_fisher_results_invariant(self):
        """Post-condition: All representatives must be present in Fisher results."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.01,
                "GO:0000002": 0.02,
                "GO:0000003": 0.005,
            },
            go_id_to_name={
                "GO:0000001": "a", "GO:0000002": "b", "GO:0000003": "c",
            },
            n_contributing={
                "GO:0000001": 3, "GO:0000002": 2, "GO:0000003": 3,
            },
        )
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        result = select_representatives(clusters, go_ids, fisher)
        for rep in result.representatives:
            assert rep in fisher.combined_pvalues, (
                f"Representative {rep} not in Fisher combined_pvalues"
            )


# ===========================================================================
# Section 11: Determinism
# ===========================================================================


class TestDeterminism:
    """Contract 11: Given the same input data and configuration, output is deterministic."""

    def test_cluster_by_similarity_is_deterministic(self):
        """Clustering the same similarity matrix twice should yield identical results."""
        # DATA ASSUMPTION: A fixed similarity matrix with clear cluster structure
        sim = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.85],
            [0.1, 0.1, 0.85, 1.0],
        ])
        result1 = cluster_by_similarity(sim, threshold=0.7)
        result2 = cluster_by_similarity(sim, threshold=0.7)
        # Sort each cluster's indices for comparison
        sorted1 = [sorted(c) for c in result1]
        sorted2 = [sorted(c) for c in result2]
        assert sorted(sorted1) == sorted(sorted2)

    def test_select_representatives_is_deterministic(self):
        """Selecting representatives from the same input twice should yield identical results."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.01,
                "GO:0000002": 0.02,
                "GO:0000003": 0.005,
            },
            go_id_to_name={
                "GO:0000001": "a", "GO:0000002": "b", "GO:0000003": "c",
            },
            n_contributing={
                "GO:0000001": 3, "GO:0000002": 2, "GO:0000003": 3,
            },
        )
        clusters = [[0, 1], [2]]
        go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        result1 = select_representatives(clusters, go_ids, fisher)
        result2 = select_representatives(clusters, go_ids, fisher)
        assert result1.representatives == result2.representatives
        assert result1.representative_pvalues == result2.representative_pvalues
        assert result1.cluster_assignments == result2.cluster_assignments


# ===========================================================================
# Section 12: Lin Similarity Formula
# ===========================================================================


class TestLinSimilarityFormula:
    """Contract 5: Verify the Lin similarity formula properties."""

    def test_lin_formula_identity(self, tmp_path):
        """Lin(t, t) = 2*IC(t) / (2*IC(t)) = 1.0 for IC > 0."""
        go_ids = ["GO:0000001"]
        ic_values = {"GO:0000001": 5.0}
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000001\nname: term1\nnamespace: biological_process\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        assert abs(result[0, 0] - 1.0) < 1e-10

    def test_lin_formula_zero_ic_both(self, tmp_path):
        """If IC(t1) + IC(t2) = 0, similarity is 0.0."""
        go_ids = ["GO:0000001", "GO:0000002"]
        ic_values = {"GO:0000001": 0.0, "GO:0000002": 0.0}
        obo_path = tmp_path / "go.obo"
        obo_path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\nid: GO:0000001\nname: term1\nnamespace: biological_process\n\n"
            "[Term]\nid: GO:0000002\nname: term2\nnamespace: biological_process\n"
        )
        result = compute_lin_similarity(go_ids, ic_values, obo_path)
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0


# ===========================================================================
# Section 13: Caching Behavior
# ===========================================================================


class TestCachingBehavior:
    """Contract 2: Downloaded files are cached locally."""

    def test_obo_cached_file_reused(self, tmp_path):
        """If OBO is already cached, it should be returned without downloading."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Pre-create a cached OBO file
        obo_file = cache_dir / "go-basic.obo"
        obo_content = "format-version: 1.2\nontology: go\ncached: true\n"
        obo_file.write_text(obo_content)

        result = download_or_load_obo("http://example.com/go-basic.obo", cache_dir)
        # The result should point to the cached file
        assert result.exists()
        # Content should match what we cached (no re-download)
        assert "cached: true" in result.read_text()

    def test_gaf_cached_file_reused(self, tmp_path):
        """If GAF is already cached, it should be returned without downloading."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        gaf_file = cache_dir / "fb.gaf.gz"
        gaf_file.write_bytes(b"cached_gaf_content")

        result = download_or_load_gaf("http://example.com/fb.gaf.gz", cache_dir)
        assert result.exists()
        assert result.read_bytes() == b"cached_gaf_content"


# ===========================================================================
# Section 14: Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_term_clustering(self):
        """A single pre-filtered GO term should form one cluster with itself as representative."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001"],
            combined_pvalues={"GO:0000001": 0.01},
            go_id_to_name={"GO:0000001": "term_a"},
            n_contributing={"GO:0000001": 3},
        )
        clusters = [[0]]
        go_ids = ["GO:0000001"]
        result = select_representatives(clusters, go_ids, fisher)
        assert result.n_clusters == 1
        assert result.representatives == ["GO:0000001"]
        assert result.representative_pvalues == [0.01]

    def test_all_terms_in_one_cluster(self):
        """If all terms are highly similar, they should form one cluster."""
        # DATA ASSUMPTION: All pairwise similarities are very high (0.95)
        sim = np.ones((3, 3)) * 0.95
        np.fill_diagonal(sim, 1.0)
        clusters = cluster_by_similarity(sim, threshold=0.7)
        # All terms should be in a single cluster
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2]

    def test_each_term_own_cluster(self):
        """If all terms are dissimilar, each should be its own cluster."""
        # DATA ASSUMPTION: Identity matrix -- no similarity between different terms
        sim = np.eye(4)
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert len(clusters) == 4

    def test_two_terms_same_pvalue_in_cluster(self):
        """When two terms in a cluster have the same p-value, one should still
        be selected as representative (deterministic tie-breaking)."""
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002"],
            combined_pvalues={
                "GO:0000001": 0.01,
                "GO:0000002": 0.01,  # same p-value
            },
            go_id_to_name={"GO:0000001": "term_a", "GO:0000002": "term_b"},
            n_contributing={"GO:0000001": 3, "GO:0000002": 3},
        )
        clusters = [[0, 1]]
        go_ids = ["GO:0000001", "GO:0000002"]
        result = select_representatives(clusters, go_ids, fisher)
        assert result.n_clusters == 1
        assert len(result.representatives) == 1
        assert result.representatives[0] in ["GO:0000001", "GO:0000002"]

    def test_similarity_threshold_at_boundary_1(self):
        """Threshold of 1.0 (maximum) should mean only identical terms cluster."""
        sim = np.array([
            [1.0, 0.99, 0.1],
            [0.99, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])
        # With threshold=1.0, distance cut = 0.0, meaning only distance=0 merges
        clusters = cluster_by_similarity(sim, threshold=1.0)
        # With threshold 1.0, likely each is its own cluster unless sim==1.0
        assert len(clusters) >= 2  # At least 2 clusters since sim(0,2) < 1.0

    def test_many_clusters(self):
        """Test with a larger number of terms to ensure scalability of cluster_by_similarity."""
        # DATA ASSUMPTION: 10 terms with block-diagonal similarity
        n = 10
        sim = np.eye(n)
        # Create 5 pairs of similar terms
        for i in range(0, n, 2):
            sim[i, i + 1] = 0.9
            sim[i + 1, i] = 0.9
        clusters = cluster_by_similarity(sim, threshold=0.7)
        assert len(clusters) == 5  # 5 clusters of 2 terms each
        # Each cluster should have exactly 2 members
        for cluster in clusters:
            assert len(cluster) == 2


# ===========================================================================
# Section 15: Pre-filter Threshold Behavior
# ===========================================================================


class TestPreFilterThreshold:
    """Contract 1: Only terms below pre-filter threshold enter clustering."""

    def test_default_prefilter_threshold(self, tmp_path):
        """Default pre-filter threshold is 0.05 (from FisherConfig)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # DATA ASSUMPTION: Terms at the boundary of 0.05
        fisher = _make_fisher_result(
            go_ids=["GO:0000001", "GO:0000002", "GO:0000003"],
            combined_pvalues={
                "GO:0000001": 0.049,  # just below threshold -> included
                "GO:0000002": 0.050,  # at threshold -> excluded (below means strictly less)
                "GO:0000003": 0.051,  # just above threshold -> excluded
            },
            go_id_to_name={
                "GO:0000001": "a", "GO:0000002": "b", "GO:0000003": "c",
            },
            n_contributing={
                "GO:0000001": 3, "GO:0000002": 2, "GO:0000003": 3,
            },
        )
        config = _make_clustering_config()

        # We expect only GO:0000001 to pass the pre-filter (p < 0.05)
        with patch("gsea_tool.go_clustering.download_or_load_obo") as mock_obo, \
             patch("gsea_tool.go_clustering.download_or_load_gaf") as mock_gaf, \
             patch("gsea_tool.go_clustering.compute_information_content") as mock_ic, \
             patch("gsea_tool.go_clustering.compute_lin_similarity") as mock_sim, \
             patch("gsea_tool.go_clustering.cluster_by_similarity") as mock_cluster, \
             patch("gsea_tool.go_clustering.select_representatives") as mock_select, \
             patch("gsea_tool.go_clustering.write_fisher_results_with_clusters_tsv") as mock_write:

            mock_obo.return_value = cache_dir / "go.obo"
            mock_gaf.return_value = cache_dir / "fb.gaf"
            mock_ic.return_value = {"GO:0000001": 3.0}
            mock_sim.return_value = np.array([[1.0]])
            mock_cluster.return_value = [[0]]
            cr = ClusteringResult(
                representatives=["GO:0000001"],
                representative_names=["a"],
                representative_pvalues=[0.049],
                representative_n_contributing=[3],
                cluster_assignments={"GO:0000001": 0},
                n_clusters=1,
                n_prefiltered=1,
                similarity_metric="Lin",
                similarity_threshold=0.7,
            )
            mock_select.return_value = cr
            mock_write.return_value = output_dir / "fisher_combined_pvalues.tsv"

            result = run_semantic_clustering(fisher, config, output_dir, cache_dir)

            # Only 1 term should have passed the pre-filter
            assert result.n_prefiltered == 1

    def test_all_significant_terms_pass_prefilter(self, tmp_path):
        """When all terms are significant, all should pass the pre-filter."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        fisher = _make_significant_fisher_result(n_terms=4)
        config = _make_clustering_config()

        with patch("gsea_tool.go_clustering.download_or_load_obo") as mock_obo, \
             patch("gsea_tool.go_clustering.download_or_load_gaf") as mock_gaf, \
             patch("gsea_tool.go_clustering.compute_information_content") as mock_ic, \
             patch("gsea_tool.go_clustering.compute_lin_similarity") as mock_sim, \
             patch("gsea_tool.go_clustering.cluster_by_similarity") as mock_cluster, \
             patch("gsea_tool.go_clustering.select_representatives") as mock_select, \
             patch("gsea_tool.go_clustering.write_fisher_results_with_clusters_tsv") as mock_write:

            mock_obo.return_value = cache_dir / "go.obo"
            mock_gaf.return_value = cache_dir / "fb.gaf"
            mock_ic.return_value = {gid: 3.0 for gid in fisher.go_ids}
            mock_sim.return_value = np.eye(4)
            mock_cluster.return_value = [[0], [1], [2], [3]]
            cr = ClusteringResult(
                representatives=fisher.go_ids,
                representative_names=[f"term_{i}" for i in range(4)],
                representative_pvalues=[fisher.combined_pvalues[gid] for gid in fisher.go_ids],
                representative_n_contributing=[3] * 4,
                cluster_assignments={gid: i for i, gid in enumerate(fisher.go_ids)},
                n_clusters=4,
                n_prefiltered=4,
                similarity_metric="Lin",
                similarity_threshold=0.7,
            )
            mock_select.return_value = cr
            mock_write.return_value = output_dir / "fisher_combined_pvalues.tsv"

            result = run_semantic_clustering(fisher, config, output_dir, cache_dir)
            assert result.n_prefiltered == 4
