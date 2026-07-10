"""Unit 7 -- GO Semantic Clustering."""

from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import gzip
import http.client
import math
import os
import urllib.request
import urllib.error

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from gsea_tool.configuration import ClusteringConfig
from gsea_tool.meta_analysis import FisherResult


@dataclass
class ClusteringResult:
    """Results of GO semantic similarity clustering."""
    representatives: list[str]
    representative_names: list[str]
    representative_pvalues: list[float]
    representative_n_contributing: list[int]
    cluster_assignments: dict[str, int]
    n_clusters: int
    n_prefiltered: int
    similarity_metric: str
    similarity_threshold: float


class CorruptDownloadError(Exception):
    """Raised when a downloaded or cached ontology/annotation file is unusable.

    Signals an empty file, a truncated transfer, or a payload that does not
    look like the expected format (e.g. an HTML error page served with 200).
    """


# Only the head of a file is inspected during validation: enough to identify
# the format, cheap enough to run on every cache hit (the OBO is ~150 MB).
_VALIDATION_READ_BYTES = 65536

# Streaming chunk size for downloads.
_DOWNLOAD_CHUNK_BYTES = 65536

# Retry-worthy transfer failures. http.client.HTTPException is included
# because IncompleteRead (a truncated response body) subclasses it rather
# than URLError or OSError, and would otherwise escape the retry loop.
_TRANSIENT_DOWNLOAD_ERRORS = (
    urllib.error.URLError,
    OSError,
    http.client.HTTPException,
    CorruptDownloadError,
)


def _validate_download(path: Path, kind: str, gzipped: "bool | None" = None) -> None:
    """Validate a freshly downloaded OBO/GAF file, raising CorruptDownloadError.

    Applied to the temporary file before it is promoted into the cache, so a
    malformed payload never becomes a cache entry.

    `gzipped` must be supplied when validating a temporary file, whose name
    ends in ".part" and therefore cannot be sniffed for a ".gz" suffix; it
    defaults to inspecting `path` itself.
    """
    if gzipped is None:
        gzipped = path.name.endswith(".gz")

    if not path.exists() or path.stat().st_size == 0:
        raise CorruptDownloadError(f"Downloaded {kind} file is empty: {path}")

    with open(path, "rb") as fh:
        head = fh.read(_VALIDATION_READ_BYTES)

    if kind == "obo":
        text = head.decode("utf-8", errors="replace")
        if "format-version:" not in text and "[Term]" not in text:
            raise CorruptDownloadError(
                f"Downloaded file does not look like a GO OBO file: {path}"
            )
    elif kind == "gaf":
        if gzipped:
            if not head.startswith(b"\x1f\x8b"):
                raise CorruptDownloadError(
                    f"Downloaded file is not gzip-compressed as expected: {path}"
                )
        else:
            text = head.decode("utf-8", errors="replace")
            if not any(
                line.startswith("!") or "\t" in line for line in text.splitlines()
            ):
                raise CorruptDownloadError(
                    f"Downloaded file does not look like a GAF file: {path}"
                )


def _cached_file_is_usable(path: Path) -> bool:
    """Report whether an existing cache entry may be returned without re-downloading.

    Deliberately weaker than _validate_download: a cache entry written by an
    earlier version of this module (or by a user placing a file by hand) is
    trusted on content, but never when it is empty. A zero-byte entry is the
    signature of an interrupted download under the pre-atomic implementation.
    """
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _download_file(url: str, dest: Path, kind: str = "obo") -> None:
    """Atomically download a file from url to dest, with a browser-like User-Agent.

    The Gene Ontology server rejects requests with the default Python
    user-agent, so we set a standard browser User-Agent header.

    The body is streamed to a sibling ``.part`` file, validated, and only then
    moved into place with os.replace(). Consequently `dest` either does not
    exist or is a complete, well-formed file -- an interrupted transfer can
    never leave a truncated or empty entry behind for a later run to consume.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; GSEA-Tool/2.1)"},
    )
    # NOTE: with_name, not with_suffix -- with_suffix would turn
    # "fb.gaf.gz" into "fb.gaf.part", clobbering the .gz extension that
    # _parse_gaf and _validate_download both key off.
    tmp = dest.with_name(dest.name + ".part")
    gzipped = dest.name.endswith(".gz")
    try:
        # Stream in chunks: response.read() would buffer the entire ~150 MB
        # OBO in memory. Chunked reads have a sharp edge, though -- unlike
        # response.read(), response.read(n) does NOT raise IncompleteRead when
        # the peer hangs up early; it simply returns b"" as though the body had
        # ended. A truncated transfer is therefore indistinguishable from a
        # complete one unless we check the length ourselves.
        with urllib.request.urlopen(req) as response, open(tmp, "wb") as out:
            declared = response.headers.get("Content-Length")
            written = 0
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)

        if declared is not None:
            try:
                expected = int(declared)
            except ValueError:
                expected = None  # malformed header; fall back to format sniff
            if expected is not None and written != expected:
                raise CorruptDownloadError(
                    f"Truncated download from {url}: received {written} bytes, "
                    f"expected {expected}"
                )

        _validate_download(tmp, kind, gzipped=gzipped)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _resolve_local_override(local_path: "str | Path", kind: str, config_key: str) -> Path:
    """Return a validated user-supplied local ontology/annotation file.

    Never touches the network. A bad override is reported immediately rather
    than silently falling back to a download the user asked us not to make.
    """
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configured {config_key} does not exist: {path}"
        )
    _validate_download(path, kind)
    return path


def download_or_load_obo(
    obo_url: str,
    cache_dir: Path,
    local_path: "str | Path | None" = None,
) -> Path:
    """Download the GO OBO file if not cached, or return cached path.

    If `local_path` is set (from clustering.go_obo_path), that file is used
    directly and no network access occurs -- the supported offline workflow.

    A cache entry is reused only if it is non-empty; an empty entry left by an
    interrupted download under an older version is discarded and re-fetched.

    Returns path to the local OBO file.
    """
    if local_path:
        return _resolve_local_override(local_path, "obo", "clustering.go_obo_path")

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Derive filename from URL
    filename = obo_url.rsplit("/", 1)[-1]
    cached_path = cache_dir / filename

    if cached_path.exists():
        if _cached_file_is_usable(cached_path):
            return cached_path
        # Poisoned cache entry (zero bytes): drop it and re-download.
        cached_path.unlink()

    # Try downloading with one retry
    for attempt in range(2):
        try:
            _download_file(obo_url, cached_path, kind="obo")
            return cached_path
        except _TRANSIENT_DOWNLOAD_ERRORS:
            if attempt == 1:
                raise ConnectionError(
                    f"Failed to download GO OBO file from {obo_url} after retry. "
                    f"To run offline, set clustering.go_obo_path in config.yaml to a "
                    f"local copy of the OBO file."
                )

    # Should not reach here, but just in case
    raise ConnectionError(f"Failed to download GO OBO file from {obo_url} after retry")


def download_or_load_gaf(
    gaf_url: str,
    cache_dir: Path,
    local_path: "str | Path | None" = None,
) -> Path:
    """Download the Drosophila GAF file if not cached, or return cached path.

    If `local_path` is set (from clustering.gaf_path), that file is used
    directly and no network access occurs.

    Returns path to the local GAF file.
    """
    if local_path:
        return _resolve_local_override(local_path, "gaf", "clustering.gaf_path")

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = gaf_url.rsplit("/", 1)[-1]
    cached_path = cache_dir / filename

    if cached_path.exists():
        if _cached_file_is_usable(cached_path):
            return cached_path
        cached_path.unlink()

    for attempt in range(2):
        try:
            _download_file(gaf_url, cached_path, kind="gaf")
            return cached_path
        except _TRANSIENT_DOWNLOAD_ERRORS:
            if attempt == 1:
                raise ConnectionError(
                    f"Failed to download Drosophila GAF file from {gaf_url} after retry. "
                    f"To run offline, set clustering.gaf_path in config.yaml to a "
                    f"local copy of the GAF file."
                )

    raise ConnectionError(f"Failed to download Drosophila GAF file from {gaf_url} after retry")


def _parse_obo(obo_path: Path) -> dict[str, dict]:
    """Parse an OBO file, returning a dict of GO term ID -> {name, is_a, namespace, is_obsolete}.

    Each entry has:
      - name: str
      - is_a: list[str]  (parent GO IDs)
      - namespace: str
      - is_obsolete: bool
    """
    terms: dict[str, dict] = {}
    current_term: dict | None = None
    in_term = False

    with open(obo_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                in_term = True
                current_term = {"name": "", "is_a": [], "namespace": "", "is_obsolete": False}
            elif line.startswith("[") and line.endswith("]"):
                # Another stanza type, end current term
                if in_term and current_term and "id" in current_term:
                    terms[current_term["id"]] = current_term
                in_term = False
                current_term = None
            elif in_term and current_term is not None:
                if line == "":
                    # End of term block
                    if "id" in current_term:
                        terms[current_term["id"]] = current_term
                    in_term = False
                    current_term = None
                elif line.startswith("id: "):
                    current_term["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    current_term["name"] = line[6:].strip()
                elif line.startswith("namespace: "):
                    current_term["namespace"] = line[11:].strip()
                elif line.startswith("is_a: "):
                    # "is_a: GO:0008150 ! biological_process"
                    parent_id = line[6:].split("!")[0].strip()
                    current_term["is_a"].append(parent_id)
                elif line.startswith("is_obsolete: true"):
                    current_term["is_obsolete"] = True

    # Handle last term if file doesn't end with blank line
    if in_term and current_term is not None and "id" in current_term:
        terms[current_term["id"]] = current_term

    return terms


def _get_ancestors(go_id: str, terms: dict[str, dict], cache: dict[str, set[str]]) -> set[str]:
    """Get all ancestors of a GO term (including itself), using the is_a hierarchy."""
    if go_id in cache:
        return cache[go_id]

    ancestors = {go_id}
    if go_id in terms:
        for parent in terms[go_id]["is_a"]:
            ancestors |= _get_ancestors(parent, terms, cache)

    cache[go_id] = ancestors
    return ancestors


def _parse_gaf(gaf_path: Path) -> dict[str, set[str]]:
    """Parse a GAF file (possibly gzipped), returning GO ID -> set of gene products annotated.

    Returns dict mapping GO ID -> set of gene/protein identifiers.
    """
    annotations: dict[str, set[str]] = defaultdict(set)

    open_func = gzip.open if str(gaf_path).endswith(".gz") else open

    with open_func(gaf_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            gene_product = parts[1]  # DB Object ID
            go_id = parts[4]  # GO ID
            # Skip NOT annotations (qualifier column is index 3)
            qualifier = parts[3]
            if "NOT" in qualifier.upper():
                continue
            annotations[go_id].add(gene_product)

    return dict(annotations)


def compute_information_content(obo_path: Path, gaf_path: Path) -> dict[str, float]:
    """Compute information content for GO terms from annotation frequencies.

    Returns dict mapping GO ID -> information content value.
    """
    terms = _parse_obo(obo_path)
    direct_annotations = _parse_gaf(gaf_path)

    # Build ancestor cache
    ancestor_cache: dict[str, set[str]] = {}

    # Propagate annotations up the hierarchy
    # For each gene annotated to a term, that gene is also annotated to all ancestors
    term_gene_counts: dict[str, set[str]] = defaultdict(set)

    for go_id, genes in direct_annotations.items():
        ancestors = _get_ancestors(go_id, terms, ancestor_cache)
        for ancestor in ancestors:
            term_gene_counts[ancestor] |= genes

    # Total number of annotated gene products
    all_genes: set[str] = set()
    for genes in term_gene_counts.values():
        all_genes |= genes
    total = len(all_genes)

    if total == 0:
        return {}

    # IC = -log(freq)  where freq = count / total
    ic_values: dict[str, float] = {}
    for go_id in terms:
        if go_id in term_gene_counts and len(term_gene_counts[go_id]) > 0:
            freq = len(term_gene_counts[go_id]) / total
            ic_values[go_id] = -math.log(freq)
        else:
            ic_values[go_id] = 0.0

    return ic_values


def compute_lin_similarity(
    go_ids: list[str],
    ic_values: dict[str, float],
    obo_path: Path,
) -> np.ndarray:
    """Compute pairwise Lin similarity matrix for a list of GO IDs.

    Returns symmetric matrix of shape (n, n) with values in [0, 1].
    """
    terms = _parse_obo(obo_path)
    ancestor_cache: dict[str, set[str]] = {}
    n = len(go_ids)
    sim_matrix = np.zeros((n, n), dtype=np.float64)

    # Pre-compute ancestors for all GO IDs
    go_id_ancestors: dict[str, set[str]] = {}
    for go_id in go_ids:
        go_id_ancestors[go_id] = _get_ancestors(go_id, terms, ancestor_cache)

    for i in range(n):
        ic_i = ic_values.get(go_ids[i], 0.0)
        sim_matrix[i, i] = 1.0 if ic_i > 0.0 else 0.0
        for j in range(i + 1, n):
            id_i = go_ids[i]
            id_j = go_ids[j]

            ic_i = ic_values.get(id_i, 0.0)
            ic_j = ic_values.get(id_j, 0.0)

            if ic_i + ic_j == 0.0:
                sim_matrix[i, j] = 0.0
                sim_matrix[j, i] = 0.0
                continue

            # Find common ancestors
            ancestors_i = go_id_ancestors[id_i]
            ancestors_j = go_id_ancestors[id_j]
            common = ancestors_i & ancestors_j

            if not common:
                sim_matrix[i, j] = 0.0
                sim_matrix[j, i] = 0.0
                continue

            # MICA = most informative common ancestor (highest IC)
            mica_ic = max(ic_values.get(a, 0.0) for a in common)

            # Lin similarity
            sim = (2.0 * mica_ic) / (ic_i + ic_j)
            # Clamp to [0, 1]
            sim = min(max(sim, 0.0), 1.0)

            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix


def cluster_by_similarity(
    similarity_matrix: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Hierarchical agglomerative clustering on the similarity matrix.

    Cuts the dendrogram at the given similarity threshold.

    Returns list of clusters, where each cluster is a list of row indices.
    """
    n = similarity_matrix.shape[0]

    if n == 0:
        return []

    if n == 1:
        return [[0]]

    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix

    # Ensure diagonal is 0 and matrix is symmetric
    np.fill_diagonal(distance_matrix, 0.0)

    # Clamp to non-negative (numerical issues)
    distance_matrix = np.maximum(distance_matrix, 0.0)

    # Convert to condensed form for scipy
    condensed = squareform(distance_matrix, checks=False)

    # Average linkage clustering
    Z = linkage(condensed, method="average")

    # Cut at distance = 1 - similarity_threshold
    cut_distance = 1.0 - threshold
    labels = fcluster(Z, t=cut_distance, criterion="distance")

    # Group indices by cluster label
    clusters_dict: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters_dict[int(label)].append(idx)

    # Return as sorted list of clusters (sorted by first index in each cluster)
    clusters = sorted(clusters_dict.values(), key=lambda c: c[0])
    return clusters


def select_representatives(
    clusters: list[list[int]],
    go_ids: list[str],
    fisher_result: FisherResult,
) -> ClusteringResult:
    """Select representative GO term per cluster (lowest combined p-value).

    Returns ClusteringResult with representatives ordered by combined p-value.
    """
    representatives_info: list[tuple[str, str, float, int, int]] = []
    # (go_id, name, pvalue, n_contributing, cluster_index)

    cluster_assignments: dict[str, int] = {}

    for cluster_idx, cluster_indices in enumerate(clusters):
        # Find the term with the lowest combined p-value in this cluster
        best_go_id = None
        best_pval = float("inf")

        for idx in cluster_indices:
            go_id = go_ids[idx]
            cluster_assignments[go_id] = cluster_idx
            pval = fisher_result.combined_pvalues.get(go_id, 1.0)
            if pval < best_pval:
                best_pval = pval
                best_go_id = go_id

        if best_go_id is not None:
            name = fisher_result.go_id_to_name.get(best_go_id, "")
            n_contrib = fisher_result.n_contributing.get(best_go_id, 0)
            representatives_info.append(
                (best_go_id, name, best_pval, n_contrib, cluster_idx)
            )

    # Sort representatives by combined p-value (ascending)
    representatives_info.sort(key=lambda x: x[2])

    representatives = [r[0] for r in representatives_info]
    representative_names = [r[1] for r in representatives_info]
    representative_pvalues = [r[2] for r in representatives_info]
    representative_n_contributing = [r[3] for r in representatives_info]

    n_clusters = len(clusters)

    return ClusteringResult(
        representatives=representatives,
        representative_names=representative_names,
        representative_pvalues=representative_pvalues,
        representative_n_contributing=representative_n_contributing,
        cluster_assignments=cluster_assignments,
        n_clusters=n_clusters,
        n_prefiltered=len(go_ids),
        similarity_metric="Lin",
        similarity_threshold=0.0,  # Will be set by caller
    )


def run_semantic_clustering(
    fisher_result: FisherResult,
    config: ClusteringConfig,
    output_dir: Path,
    cache_dir: Path,
) -> ClusteringResult:
    """Top-level entry point for GO semantic clustering.

    Downloads/loads OBO and GAF, computes similarity, clusters, selects
    representatives, and writes fisher_combined_pvalues.tsv with cluster
    assignments.

    Returns ClusteringResult.
    """
    # Pre-conditions
    assert config.similarity_threshold > 0.0, "Similarity threshold must be positive"
    assert config.similarity_threshold <= 1.0, "Similarity threshold must be at most 1.0"

    # Step 1: Pre-filter GO terms by combined p-value
    prefilter_threshold = 0.05  # default
    # Use fisher prefilter_pvalue if available from config context -- but the
    # blueprint says "configurable threshold (default 0.05)". The ClusteringConfig
    # doesn't have a prefilter field, so we use 0.05 as default.
    # Actually, looking at FisherConfig, it has prefilter_pvalue. But we receive
    # ClusteringConfig here, not FisherConfig. The blueprint says "pre-filter GO
    # terms to those with combined p-value below a configurable threshold (default
    # 0.05)". We'll use 0.05.

    filtered_go_ids = []
    for go_id in sorted(fisher_result.combined_pvalues.keys()):
        if fisher_result.combined_pvalues[go_id] < prefilter_threshold:
            filtered_go_ids.append(go_id)

    if len(filtered_go_ids) == 0:
        raise ValueError(
            "No GO terms have combined p-value below the pre-filter threshold "
            f"({prefilter_threshold})"
        )

    # Step 2: Download/load OBO and GAF. The local_path overrides make the
    # whole clustering path runnable offline (see clustering.go_obo_path).
    obo_path = download_or_load_obo(
        config.go_obo_url, cache_dir, local_path=config.go_obo_path
    )
    gaf_path = download_or_load_gaf(
        config.gaf_url, cache_dir, local_path=config.gaf_path
    )

    # Step 3: Compute information content
    ic_values = compute_information_content(obo_path, gaf_path)

    # Step 4: Compute pairwise Lin similarity
    sim_matrix = compute_lin_similarity(filtered_go_ids, ic_values, obo_path)

    # Step 5: Hierarchical agglomerative clustering
    clusters = cluster_by_similarity(sim_matrix, config.similarity_threshold)

    # Step 6: Select representatives
    clustering_result = select_representatives(clusters, filtered_go_ids, fisher_result)
    # Set the actual threshold used
    clustering_result.similarity_threshold = config.similarity_threshold
    clustering_result.similarity_metric = config.similarity_metric

    # Post-conditions
    assert len(clustering_result.representatives) == clustering_result.n_clusters, \
        "One representative per cluster"
    assert clustering_result.n_clusters > 0, "At least one cluster must be formed"
    assert all(
        go_id in fisher_result.combined_pvalues
        for go_id in clustering_result.representatives
    ), "All representatives must be present in Fisher results"
    assert clustering_result.representatives == sorted(
        clustering_result.representatives,
        key=lambda gid: fisher_result.combined_pvalues[gid]
    ), "Representatives must be ordered by combined p-value ascending"

    # Write output TSV
    write_fisher_results_with_clusters_tsv(fisher_result, clustering_result, output_dir)

    return clustering_result


def write_fisher_results_with_clusters_tsv(
    fisher_result: FisherResult,
    clustering_result: ClusteringResult,
    output_dir: Path,
) -> Path:
    """Write fisher_combined_pvalues.tsv with cluster assignment and representative columns."""
    output_path = output_dir / "fisher_combined_pvalues.tsv"

    # Build set of representatives for quick lookup
    representative_set = set(clustering_result.representatives)

    try:
        with open(output_path, "w", newline="") as f:
            # Header
            header = "GO_ID\tTerm_Name\tCombined_PValue\tN_Contributing\tCluster\tIs_Representative\n"
            f.write(header)

            # Write all pre-filtered GO terms (those with cluster assignments)
            # Sort by GO ID for deterministic output
            prefiltered_ids = sorted(clustering_result.cluster_assignments.keys())

            for go_id in prefiltered_ids:
                term_name = fisher_result.go_id_to_name.get(go_id, "")
                combined_p = fisher_result.combined_pvalues.get(go_id, 1.0)
                n_contrib = fisher_result.n_contributing.get(go_id, 0)
                cluster_idx = clustering_result.cluster_assignments[go_id]
                is_rep = go_id in representative_set

                line = f"{go_id}\t{term_name}\t{combined_p}\t{n_contrib}\t{cluster_idx}\t{is_rep}\n"
                f.write(line)
    except OSError as e:
        raise OSError(f"Cannot write fisher_combined_pvalues.tsv to {output_dir}: {e}") from e

    return output_path
