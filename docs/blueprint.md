# Blueprint: GSEA Proteomics Visualizer

**Version:** 2.1
**Stakeholder Spec Version:** 2.1
**Units:** 10
**Dependency order:** Strictly backward (topological)

---

## Unit 1 -- Data Ingestion

### Tier 1 -- Description

This unit is responsible for discovering, validating, parsing, and merging all GSEA preranked output files from the `data/` directory inside the project directory. It traverses exactly one level of subdirectories inside `data/`, where each subdirectory represents one fly mutant experiment. For each subdirectory, it extracts the mutant identifier from the folder name (the portion before `.GseaPreranked`), locates exactly one positive and one negative TSV report file by glob pattern, and halts with a descriptive error if the expected file count is violated. It then parses both TSV files, extracting the NAME, NES, FDR q-val, NOM p-val, and SIZE columns. The NAME column value has the format `GO:NNNNNNN TERM_NAME`; this unit extracts both the GO ID (the first token matching the regex `GO:\d{7}`) and the term name (everything following the GO ID, stripped of leading/trailing whitespace). Rows without a valid GO ID are skipped with a warning. The TSV parser must handle the HTML artifact in the second column header (`GS<br> follow link to MSigDB`) and trailing tab characters on each row. The positive and negative files for each mutant are merged into a single enrichment profile. The final output is a structured dataset indexed by (GO term name, mutant identifier) containing NES, FDR, NOM p-val, SIZE, and GO ID values, plus a sorted list of mutant identifiers (alphabetical order). The unit enforces a minimum cohort size of 2 mutant lines; if fewer than 2 valid mutant subfolders are discovered, it halts with a descriptive error.

### Tier 2 — Signatures

```python
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TermRecord:
    """Enrichment data for one GO term in one mutant."""
    term_name: str
    go_id: str
    nes: float
    fdr: float
    nom_pval: float
    size: int
    ...


@dataclass
class MutantProfile:
    """Complete enrichment profile for one mutant (merged pos + neg)."""
    mutant_id: str
    records: dict[str, TermRecord]  # keyed by term_name
    ...


@dataclass
class CohortData:
    """All enrichment data for the entire mutant cohort."""
    mutant_ids: list[str]  # sorted alphabetically
    profiles: dict[str, MutantProfile]  # keyed by mutant_id
    all_term_names: set[str]  # union of all GO term names across all mutants
    all_go_ids: set[str]  # union of all GO IDs across all mutants
    ...


class DataIngestionError(Exception):
    """Raised when input data violates structural expectations."""
    ...


def discover_mutant_folders(data_dir: Path) -> list[tuple[str, Path]]:
    """Discover level-1 mutant subfolders and extract mutant identifiers.

    Returns list of (mutant_id, folder_path) sorted alphabetically by mutant_id.
    """
    ...


def locate_report_files(mutant_folder: Path, mutant_id: str) -> tuple[Path, Path]:
    """Locate exactly one pos and one neg TSV file in a mutant subfolder.

    Returns (pos_file_path, neg_file_path).
    Raises DataIngestionError if zero or more than one match for either pattern.
    """
    ...


def parse_gsea_report(tsv_path: Path) -> list[TermRecord]:
    """Parse a single GSEA preranked TSV report file.

    Extracts GO ID and term name from NAME column. Handles HTML artifact in
    column headers and trailing tabs. Skips rows without valid GO ID with warning.
    """
    ...


def merge_pos_neg(pos_records: list[TermRecord], neg_records: list[TermRecord]) -> dict[str, TermRecord]:
    """Merge positive and negative report records into a single profile dict keyed by term_name.

    If a term appears in both pos and neg records, the entry with the smaller
    nominal p-value is retained (conflict resolution per spec Section 6.2 Step 1).
    """
    ...


def ingest_data(data_dir: Path) -> CohortData:
    """Top-level ingestion entry point. Discovers folders, validates, parses, and merges.

    Raises DataIngestionError on structural violations including fewer than 2 mutant lines.
    """
    ...


# --- Invariants ---
# Pre-conditions
assert data_dir.is_dir(), "data_dir must be an existing directory"

# Post-conditions
assert len(cohort.mutant_ids) >= 2, "At least 2 mutant lines are required"
assert len(cohort.mutant_ids) == len(cohort.profiles), "Every mutant_id must have a corresponding profile"
assert cohort.mutant_ids == sorted(cohort.mutant_ids), "mutant_ids must be in alphabetical order"
assert all(
    rec.term_name == rec.term_name.upper() and not rec.term_name.startswith("GO:")
    for profile in cohort.profiles.values()
    for rec in profile.records.values()
), "All term names must be uppercase with GO ID prefix stripped"
assert all(
    rec.go_id.startswith("GO:") and len(rec.go_id) == 10
    for profile in cohort.profiles.values()
    for rec in profile.records.values()
), "All GO IDs must match GO:NNNNNNN format"
```

### Tier 3 -- Behavioral Contracts

**Error conditions:**

| Exception | Description | Trigger |
|---|---|---|
| `DataIngestionError` | Missing negative report file | Zero files matching `gsea_report_for_na_neg_*.tsv` in a mutant subfolder |
| `DataIngestionError` | Missing positive report file | Zero files matching `gsea_report_for_na_pos_*.tsv` in a mutant subfolder |
| `DataIngestionError` | Ambiguous negative report file | More than one file matching `gsea_report_for_na_neg_*.tsv` in a mutant subfolder |
| `DataIngestionError` | Ambiguous positive report file | More than one file matching `gsea_report_for_na_pos_*.tsv` in a mutant subfolder |
| `DataIngestionError` | Insufficient mutant lines | Fewer than 2 valid mutant subfolders discovered in `data/` |

**Behavioral contracts:**

1. The unit traverses exactly one level of subdirectories inside `data_dir`. Nested subdirectories are ignored.
2. Only subdirectories whose names contain `.GseaPreranked` are processed. Other entries are silently ignored.
3. The mutant identifier is extracted as the portion of the folder name before the first `.GseaPreranked` substring.
4. The GO ID is extracted from the NAME column as the first token matching the regex `GO:\d{7}`. The term name is everything following the GO ID, stripped of leading/trailing whitespace, and normalized to uppercase during parsing. This normalization ensures consistent cross-unit lookups regardless of case variations in the input files. Both GO ID and normalized term name are stored in the `TermRecord`.
5. Rows without a valid GO ID in the NAME column are skipped with a warning to stderr. They do not cause a halt.
6. Rows with missing or non-numeric values in NES, FDR q-val, NOM p-val, or SIZE are skipped with a warning to stderr. They do not cause a halt.
8. The second column header containing the HTML artifact `GS<br> follow link to MSigDB` does not cause a parse failure. That column is not consumed.
9. Trailing tab characters on data rows do not produce spurious empty fields in the parsed output.
10. Each mutant profile contains the union of terms from its pos and neg files. If a term appears in both files for the same mutant, the entry with the smaller nominal p-value is retained (conflict resolution).
11. `CohortData.all_term_names` is the union of all term names across all mutant profiles.
12. `CohortData.all_go_ids` is the union of all GO IDs across all mutant profiles.
13. `CohortData.mutant_ids` is sorted alphabetically.
14. Files in mutant subfolders that do not match either the pos or neg glob pattern are silently ignored.
15. If fewer than 2 valid mutant subfolders are discovered, the unit raises `DataIngestionError` with a descriptive message. No partial output is produced.

**Dependencies:** None (this is the first unit).

---

## Unit 2 -- Configuration

### Tier 1 -- Description

This unit loads and validates the tool configuration. It looks for a `config.yaml` file in the project directory. If the file exists, it is parsed and validated against the expected schema. If the file does not exist, all parameters use built-in defaults. If the file exists but contains invalid YAML syntax, the unit halts with a descriptive error. The unit produces a single typed configuration object containing all tunable parameters organized by domain: cherry-pick categories (list of parent GO IDs with display labels for Figure 1), dot plot parameters (FDR threshold, top-N, n-groups), meta-analysis parameters (pseudocount, FDR correction toggle, FDR threshold, pre-filter p-value, top-N bars), clustering parameters (enabled flag, similarity metric, similarity threshold, GO OBO URL, GAF URL), and plot appearance parameters (DPI, font family, bar plot colormap, figure dimensions, label max length, significance line toggle, recurrence annotation toggle). Unknown keys in the YAML file are silently ignored to allow forward compatibility. Missing keys use built-in defaults. Type errors (e.g., a string where a float is expected) cause a halt with a descriptive error.

### Tier 2 — Signatures

```python
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class CherryPickCategory:
    """A single cherry-pick category entry from config."""
    go_id: str  # parent GO term ID, e.g. "GO:0005739"
    label: str  # display name, e.g. "Mitochondria"
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
    similarity_metric: str = "Lin"
    similarity_threshold: float = 0.7
    go_obo_url: str = "https://current.geneontology.org/ontology/go-basic.obo"
    gaf_url: str = ""  # default set at runtime to GO Consortium/FlyBase URL
    ...


@dataclass
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


# --- Invariants ---
# Pre-conditions
assert project_dir.is_dir(), "project_dir must be an existing directory"

# Post-conditions
assert 0.0 < config.dot_plot.fdr_threshold <= 1.0, "FDR threshold must be in (0, 1]"
assert config.dot_plot.top_n > 0, "top_n must be positive"
assert config.dot_plot.n_groups > 0, "n_groups must be positive"
assert config.fisher.pseudocount > 0, "pseudocount must be positive"
assert 0.0 < config.fisher.prefilter_pvalue <= 1.0, "prefilter_pvalue must be in (0, 1]"
assert config.fisher.top_n_bars > 0, "top_n_bars must be positive"
assert 0.0 < config.clustering.similarity_threshold <= 1.0, "similarity_threshold must be in (0, 1]"
assert all(re.match(r"GO:\d{7}$", c.go_id) for c in config.cherry_pick_categories), "Each cherry_pick go_id must match GO:\\d{7}"
assert all(len(c.label.strip()) > 0 for c in config.cherry_pick_categories), "Each cherry_pick label must be non-empty"
assert config.plot_appearance.dpi > 0, "DPI must be positive"
assert config.plot_appearance.label_max_length > 0, "label_max_length must be positive"
```

### Tier 3 -- Behavioral Contracts

**Error conditions:**

| Exception | Description | Trigger |
|---|---|---|
| `ConfigError` | Invalid YAML syntax | `config.yaml` exists but cannot be parsed as valid YAML |
| `ConfigError` | Type validation failure | A config value has the wrong type (e.g., string where float expected) |
| `ConfigError` | Invalid parameter value | A config value is out of its valid range (e.g., negative threshold) |

**Behavioral contracts:**

1. If `config.yaml` does not exist in the project directory, the unit returns a `ToolConfig` with all built-in defaults. No warning is emitted.
2. If `config.yaml` exists and is valid, all specified values override the defaults. Unspecified keys retain their defaults.
3. Unknown keys in the YAML file are silently ignored to allow forward compatibility.
4. The YAML file uses the following nested key hierarchy:
   - `cherry_pick` (list of dicts, each with `go_id` (str matching `GO:\d{7}`) and `label` (non-empty str)). Default: empty list.
   - `dot_plot.fdr_threshold` (float), `dot_plot.top_n` (int), `dot_plot.n_groups` (int), `dot_plot.random_seed` (int)
   - `fisher.pseudocount` (float), `fisher.apply_fdr` (bool), `fisher.fdr_threshold` (float), `fisher.prefilter_pvalue` (float), `fisher.top_n_bars` (int)
   - `clustering.enabled` (bool), `clustering.similarity_metric` (str), `clustering.similarity_threshold` (float), `clustering.go_obo_url` (str), `clustering.gaf_url` (str)
   - `plot.dpi` (int), `plot.font_family` (str), `plot.bar_colormap` (str), `plot.bar_figure_width` (float), `plot.bar_figure_height` (float), `plot.label_max_length` (int), `plot.show_significance_line` (bool), `plot.show_recurrence_annotation` (bool)
5. Type coercion is not performed -- a string `"0.05"` where a float is expected causes a `ConfigError`, not silent conversion.
6. The `ToolConfig` object is immutable after construction. Downstream units read from it but do not modify it.
8. The GAF URL default is set to the GO Consortium Drosophila melanogaster GAF download URL if not specified in config.

**Dependencies:** None (this is an independent unit alongside Unit 1).

---

## Unit 3 -- Cherry-Picked Term Selection

### Tier 1 -- Description

This unit implements the term selection and grouping logic for Figure 1 (the hypothesis-driven figure). It supports two approaches: (1) **config-based ontology resolution** (preferred), where categories are specified as parent GO term IDs in `config.yaml` and the unit uses the GO OBO ontology to resolve all descendant terms, then intersects with GSEA results; (2) **TSV-based mapping** (fallback), where a user-supplied two-column TSV maps GO term names to category names. The config-based approach requires access to the GO OBO file (already downloaded/cached by Unit 7's infrastructure). In both approaches, categories with zero matching terms are silently omitted. Within each category, terms are sorted by mean absolute NES across all mutants (descending). A GO term whose GO ID descends from multiple configured parent GO IDs appears in all matching categories. The output is an ordered list of category groups, each containing a category name and an ordered list of GO term names, ready for the renderer.

### Tier 2 — Signatures

```python
from pathlib import Path
from dataclasses import dataclass

# Upstream types
from gsea_tool.data_ingestion import CohortData
from gsea_tool.configuration import CherryPickCategory


@dataclass
class CategoryGroup:
    """A named group of GO terms for dot plot rendering."""
    category_name: str
    term_names: list[str]  # ordered by sort criterion (e.g., mean abs NES descending)
    ...


class MappingFileError(Exception):
    """Raised when the category mapping file cannot be parsed."""
    ...


def parse_category_mapping(mapping_path: Path) -> dict[str, str]:
    """Parse the user-supplied category mapping file (TSV fallback path).

    Returns dict mapping term_name (uppercase) -> category_name.
    Raises MappingFileError if the file cannot be parsed.
    """
    ...


def select_cherry_picked_terms(
    cohort: CohortData,
    term_to_category: dict[str, str],
) -> list[CategoryGroup]:
    """Select and group GO terms for Figure 1 based on user-supplied category mapping (TSV fallback path).

    Terms are included if they appear in both the mapping and the GSEA results.
    Within each category, terms are sorted by mean absolute NES descending.
    Categories are returned in the order they first appear in the mapping dict.
    """
    ...


def get_all_descendants(parent_go_id: str, obo_path: Path) -> set[str]:
    """Resolve all descendant GO IDs of a parent GO term using the OBO ontology.

    Parses the OBO file, builds a children map (inverting the is_a parent relationships),
    and performs a breadth-first traversal from parent_go_id to collect all descendants.
    The parent GO ID itself is included in the result set.

    Returns set of GO IDs (including the parent).
    """
    ...


def resolve_categories_from_ontology(
    cohort: CohortData,
    categories: list[CherryPickCategory],
    obo_path: Path,
) -> list[CategoryGroup]:
    """Select and group GO terms for Figure 1 using ontology-based category resolution.

    For each configured category:
    1. Resolve all descendant GO IDs of the parent GO ID via the OBO hierarchy.
    2. Intersect descendants with GO IDs present in the GSEA results (cohort.all_go_ids).
    3. Map matching GO IDs back to term names via the cohort data.
    4. Sort terms within each category by mean absolute NES across all mutants, descending.

    A GO term matching multiple categories appears in all of them.
    Categories with zero matching terms are silently omitted.
    Categories are returned in the order specified in the config list.
    """
    ...


# --- Invariants ---
# Pre-conditions (for TSV path)
assert mapping_path.is_file(), "Mapping file path must point to an existing file"
# Pre-conditions (for ontology path)
assert obo_path.is_file(), "OBO file path must point to an existing file"
assert all(re.match(r"GO:\d{7}$", c.go_id) for c in categories), "Each category go_id must be valid"

# Post-conditions (both paths)
assert all(len(group.term_names) > 0 for group in groups), "Empty categories are omitted"
```

### Tier 3 -- Behavioral Contracts

**Error conditions:**

| Exception | Description | Trigger |
|---|---|---|
| `MappingFileError` | Unparseable mapping file | File does not conform to the expected two-column TSV format (TSV path only) |
| `FileNotFoundError` | Missing OBO file | The OBO file path does not exist (ontology path only) |
| `ValueError` | Invalid parent GO ID | A configured parent GO ID is not found in the OBO ontology |

**Behavioral contracts:**

1. **TSV fallback path:** The mapping file is parsed as a two-column TSV: first column is the GO term name (matched case-insensitively against GSEA data after uppercasing both sides), second column is the category name. Categories are returned in the order they first appear in the file.
2. **Ontology path:** For each configured `CherryPickCategory`, the unit resolves all descendant GO IDs of the parent `go_id` using the GO OBO hierarchy (is_a relationships, transitive closure). The parent GO ID itself is included. Descendant GO IDs are intersected with `cohort.all_go_ids` to find terms present in the data. GO IDs are mapped back to term names via the cohort's `TermRecord.go_id` field.
3. GO terms present in the mapping/ontology but absent from all mutant profiles in the cohort are silently dropped.
4. Within each category, terms are sorted by mean absolute NES across all mutants, descending. The mean is computed over all mutants (using NES=0 for mutants where the term is absent).
5. Categories are returned in the order specified by the config list (ontology path) or the order they first appear in the file (TSV path). Categories with zero matching terms are omitted.
6. A GO term whose GO ID descends from multiple configured parent GO IDs appears in all matching `CategoryGroup` objects (ontology path only).
7. The `CategoryGroup` data structure is the same type consumed by the dot plot renderer (Unit 5), ensuring interface compatibility.
8. Lines in the mapping file that are empty or start with `#` are treated as comments and skipped (TSV path only).

**Dependencies:** Unit 1 (CohortData), Unit 2 (CherryPickCategory).

---

## Unit 4 -- Unbiased Term Selection

### Tier 1 -- Description

This unit implements the six-step data-driven term selection and grouping pipeline for Figure 2 (the unbiased figure). Step 1: pool all GO terms that pass the FDR significance threshold in at least one mutant. Step 2: rank the pooled terms by their maximum absolute NES observed across the mutant cohort. Step 3: remove lexically redundant terms -- when two GO term names share substantial word overlap, retain only the one with the higher maximum absolute NES. Step 4: from the deduplicated ranked list, select the top N terms (configurable, default 20). Step 5: cluster the selected terms into groups using hierarchical agglomerative clustering on their NES profiles across mutants, with a configurable number of groups (default 4) and a fixed random seed for reproducibility. Step 6: label each cluster group with the name of its most representative term (highest mean absolute NES within the group). The output is the same `CategoryGroup` list structure used by Unit 3, ensuring the renderer has a single interface.

### Tier 2 — Signatures

```python
from dataclasses import dataclass
import numpy as np

# Upstream types
from gsea_tool.data_ingestion import CohortData
from gsea_tool.cherry_picked import CategoryGroup


@dataclass
class UnbiasedSelectionStats:
    """Statistics collected during unbiased selection for notes.md."""
    total_significant_terms: int  # after step 1
    terms_after_dedup: int  # after step 3
    terms_selected: int  # after step 4 (top N)
    n_clusters: int  # step 5 parameter
    random_seed: int  # step 5 parameter
    clustering_algorithm: str  # e.g. "scipy.cluster.hierarchy (Ward linkage)"
    ...


def pool_significant_terms(
    cohort: CohortData,
    fdr_threshold: float,
) -> dict[str, float]:
    """Step 1-2: Pool terms passing FDR threshold in any mutant, compute max abs NES.

    Returns dict mapping term_name -> max_absolute_nes, sorted by value descending.
    """
    ...


def remove_redundant_terms(
    ranked_terms: dict[str, float],
) -> dict[str, float]:
    """Step 3: Remove lexically redundant terms.

    For each pair of terms sharing substantial word overlap (Jaccard similarity
    of word sets > 0.5), retain only the term with higher max abs NES.
    """
    ...


def select_top_n(
    ranked_terms: dict[str, float],
    top_n: int,
) -> list[str]:
    """Step 4: Select top N terms from deduplicated ranked list."""
    ...


def cluster_terms(
    term_names: list[str],
    cohort: CohortData,
    n_groups: int,
    random_seed: int,
) -> list[CategoryGroup]:
    """Steps 5-6: Cluster selected terms by NES profile and auto-label groups.

    Uses hierarchical agglomerative clustering (Ward linkage) on the NES profile
    matrix (terms as rows, mutants as columns). Missing NES values are treated as 0.0.
    Each group is labeled with the term having the highest mean absolute NES
    within that group. Terms within each group are sorted by mean absolute NES
    descending. Groups are sorted by the position of their highest-ranked member
    in the original top-N ranking.
    """
    ...


def select_unbiased_terms(
    cohort: CohortData,
    fdr_threshold: float = 0.05,
    top_n: int = 20,
    n_groups: int = 4,
    random_seed: int = 42,
) -> tuple[list[CategoryGroup], UnbiasedSelectionStats]:
    """Top-level entry point for unbiased term selection (Figure 2).

    Returns the grouped terms and collection statistics for notes.md.
    """
    ...


# --- Invariants ---
# Pre-conditions
assert top_n > 0, "top_n must be a positive integer"
assert n_groups > 0, "n_groups must be a positive integer"
assert n_groups <= top_n, "Cannot have more groups than selected terms"

# Post-conditions
assert sum(len(g.term_names) for g in groups) <= top_n, "Total terms across groups cannot exceed top_n"
assert len(groups) <= n_groups, "Number of groups cannot exceed n_groups"
assert all(len(g.term_names) > 0 for g in groups), "No empty groups"
assert stats.random_seed == random_seed, "Stats must record the seed actually used"
```

### Tier 3 -- Behavioral Contracts

**Error conditions:**

| Exception | Description | Trigger |
|---|---|---|
| `ValueError` | Insufficient significant terms | Fewer terms pass the FDR threshold than `n_groups`, making clustering impossible |

**Behavioral contracts:**

1. Only GO terms with FDR < `fdr_threshold` in at least one mutant are included in the candidate pool (step 1).
2. Terms are ranked by maximum absolute NES across all mutants (step 2). Ties are broken alphabetically by term name for determinism.
3. Lexical redundancy is determined by Jaccard similarity of the word sets (split on whitespace) of two term names. If Jaccard similarity exceeds 0.5, the term with lower max absolute NES is removed (step 3). Terms are processed in rank order so the higher-ranked term always survives.
4. The top N terms are selected from the deduplicated list (step 4). If fewer than N terms remain after deduplication, all remaining terms are used.
5. Clustering uses hierarchical agglomerative clustering (Ward linkage) on the NES profile matrix (terms as rows, mutants as columns). Missing NES values (term absent from a mutant) are treated as 0.0 in the profile matrix. Ward linkage is deterministic and does not consume random state; the random seed is set as a defensive measure and recorded in `UnbiasedSelectionStats` for reproducibility documentation, but does not affect the clustering output.
6. Each cluster group is labeled with the term name having the highest mean absolute NES within that group (step 6).
8. Within each group, terms are sorted by mean absolute NES descending.
9. Groups are ordered by the rank position of their highest-ranked member (the term with the highest max absolute NES in the group), preserving the original importance ranking at the group level.
10. The `UnbiasedSelectionStats` dataclass captures all parameters and intermediate counts needed by Unit 9 (Notes Generation).
11. Given the same input data and the same random seed, the output is deterministic.

**Dependencies:** Unit 1 (CohortData), Unit 3 (CategoryGroup type).

---

## Unit 5 -- Dot Plot Rendering

### Tier 1 -- Description

This unit renders a single publication-quality dot plot figure given a list of category groups and the cohort enrichment data. It is called once or twice at runtime: once for Figure 2 (always), and optionally once for Figure 1 (when a mapping file is provided). The renderer does not know or care whether the groups came from user-defined categories or from clustering -- it receives the same `CategoryGroup` structure either way. The figure layout follows Gordon et al. 2024 Figure 3a: mutants on the X-axis (alphabetical order), GO terms on the Y-axis grouped into labeled categories. Each cell in the grid either contains a dot (if FDR < threshold for that term in that mutant) or is empty. Dot color encodes NES on a diverging red-blue colormap symmetric around zero. Dot size encodes -log10(FDR). Category boxes are drawn as visible rectangles enclosing each group of terms, with the category name rendered in bold to the right of the box, vertically centered. The figure includes a colorbar legend for NES and a size legend for significance. Styling is clean and minimal: no gridlines, no chartjunk, Nature-family journal aesthetic. Output is saved to PDF, PNG, and SVG in the output directory at 300+ DPI.

### Tier 2 — Signatures

```python
from pathlib import Path
from dataclasses import dataclass
import matplotlib.figure
import matplotlib.axes

# Upstream types
from gsea_tool.data_ingestion import CohortData
from gsea_tool.cherry_picked import CategoryGroup


@dataclass
class DotPlotResult:
    """Metadata about a rendered dot plot figure, for notes.md consumption."""
    pdf_path: Path
    png_path: Path
    svg_path: Path
    n_terms_displayed: int
    n_categories: int
    n_mutants: int
    ...


def render_dot_plot(
    cohort: CohortData,
    groups: list[CategoryGroup],
    fdr_threshold: float,
    output_stem: str,
    output_dir: Path,
    dpi: int = 300,
    font_family: str = "Arial",
    title: str = "",
) -> DotPlotResult:
    """Render a grouped dot plot figure and save to PDF, PNG, and SVG.

    Args:
        cohort: The full cohort enrichment data.
        groups: Ordered list of category groups defining Y-axis layout.
        fdr_threshold: FDR threshold for dot presence (cells with FDR >= threshold are empty).
        output_stem: Base filename without extension (e.g., "figure1_cherry_picked").
        output_dir: Directory to write output files.
        dpi: Resolution for PNG output.
        font_family: Font family for all text.
        title: Optional figure title.

    Returns:
        DotPlotResult with paths and summary counts.
    """
    ...


def build_dot_grid(
    cohort: CohortData,
    groups: list[CategoryGroup],
    fdr_threshold: float,
) -> tuple[list[list[float | None]], list[list[float | None]], list[str], list[str]]:
    """Build the NES and significance matrices for the dot grid.

    Returns:
        nes_matrix: 2D list [term_index][mutant_index], None for empty cells.
        sig_matrix: 2D list [term_index][mutant_index] of -log10(FDR), None for empty cells.
        term_labels: Ordered Y-axis labels (term names, grouped by category).
        mutant_labels: Ordered X-axis labels (mutant IDs, alphabetical).
    """
    ...


def draw_category_boxes(
    ax: matplotlib.axes.Axes,
    groups: list[CategoryGroup],
    y_start: float,
) -> None:
    """Draw category grouping rectangles and bold right-side labels on the axes.

    Each box encloses the rows belonging to one category group. The category name
    is rendered in bold, vertically centered to the right of the box.
    """
    ...


# --- Invariants ---
# Pre-conditions
assert len(groups) > 0, "At least one category group is required"
assert all(len(g.term_names) > 0 for g in groups), "No empty groups passed to renderer"
assert output_dir.is_dir(), "Output directory must exist"

# Post-conditions
assert result.pdf_path.exists(), "PDF file must be written"
assert result.png_path.exists(), "PNG file must be written"
assert result.svg_path.exists(), "SVG file must be written"
assert result.n_terms_displayed == sum(len(g.term_names) for g in groups), "Term count must match input"
assert result.n_mutants == len(cohort.mutant_ids), "Mutant count must match cohort"
```

### Tier 3 -- Behavioral Contracts

**Error conditions:**

| Exception | Description | Trigger |
|---|---|---|
| `ValueError` | Empty groups list | `groups` is an empty list |
| `OSError` | Output write failure | Cannot write PDF, PNG, or SVG to `output_dir` |

**Behavioral contracts:**

1. The X-axis displays mutant identifiers in alphabetical order, matching `cohort.mutant_ids`.
2. The Y-axis displays GO term names grouped by category. Categories appear in the order given by the `groups` list. Within each category, terms appear in the order given by `group.term_names`.
3. A dot is drawn at position (mutant, term) only if FDR < `fdr_threshold` for that term in that mutant. Otherwise the cell is empty (no glyph, no placeholder).
4. Dot color encodes NES on a diverging colormap: red for positive NES (upregulation), blue for negative NES (downregulation). The colormap is symmetric around zero (vmin = -max_abs_nes, vmax = +max_abs_nes). A colorbar legend is included.
5. Dot size encodes -log10(FDR q-val). A size legend with representative values is included.
6. Category boxes are drawn as visible rectangles enclosing each group of terms on the Y-axis. The category name is rendered in bold text to the right of the box, vertically centered within the box extent.
8. No gridlines are drawn within the plot area. Category boxes are the sole visual grouping structure.
9. GO term labels on the Y-axis show the term name only (no GO ID). Labels are legible and do not overlap. If necessary, the figure height is scaled to accommodate all terms.
10. The figure is saved to `{output_stem}.pdf`, `{output_stem}.png`, and `{output_stem}.svg` in `output_dir` at the configured DPI minimum.
11. The colormap uses `RdBu_r` (or equivalent) so that red is positive and blue is negative, matching the Gordon et al. convention.
12. The figure uses a clean, minimal aesthetic consistent with Nature-family journal standards: no background color, no unnecessary borders, tight layout.

**Dependencies:** Unit 1 (CohortData), Unit 3 (CategoryGroup type).

---

## Unit 6 -- Meta-Analysis Computation

### Tier 1 -- Description

This unit implements Fisher's combined probability method to aggregate GSEA evidence across all mutant lines. It operates on nominal p-values keyed by GO ID from the ingested cohort data. Step 1: for each mutant, build a per-mutant p-value dictionary keyed by GO ID from the already-ingested TermRecords. If a GO term has `NOM p-val` of exactly 0.0, replace it with a configurable pseudocount (default 1e-10). Step 2: collect the union of all GO IDs across all mutants and build a p-value matrix of shape (n_GO_terms x n_mutants), imputing p = 1.0 for missing entries. Step 3: for each GO term, compute the Fisher statistic X^2 = -2 * sum(ln(p_i)) and the combined p-value from a chi-squared distribution with 2k degrees of freedom, where k is the number of mutant lines. Optionally apply Benjamini-Hochberg FDR correction on the combined p-values. The unit writes `pvalue_matrix.tsv` to the output directory. When clustering is disabled (per config), this unit also writes `fisher_combined_pvalues.tsv` (without cluster assignment columns). The unit also computes and stores the number of contributing mutant lines per GO term (lines with p < 1.0).

### Tier 2 — Signatures

```python
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# Upstream types
from gsea_tool.data_ingestion import CohortData
from gsea_tool.configuration import FisherConfig


@dataclass
class FisherResult:
    """Results of Fisher's combined probability test."""
    go_ids: list[str]  # all GO IDs tested
    go_id_to_name: dict[str, str]  # GO ID -> term name for display
    combined_pvalues: dict[str, float]  # GO ID -> combined p-value
    n_contributing: dict[str, int]  # GO ID -> number of mutant lines with p < 1.0
    pvalue_matrix: np.ndarray  # shape (n_go_terms, n_mutants)
    mutant_ids: list[str]  # column labels for the matrix
    go_id_order: list[str]  # row labels for the matrix
    n_mutants: int  # total mutant lines (k in the Fisher formula)
    corrected_pvalues: dict[str, float] | None  # BH-FDR corrected, or None if not applied
    ...


def build_pvalue_dict_per_mutant(
    cohort: CohortData,
    pseudocount: float,
) -> dict[str, dict[str, float]]:
    """Build per-mutant {GO_ID: nom_pval} dictionaries from ingested data.

    Replaces NOM p-val of 0.0 with pseudocount. Skips records with missing
    or non-numeric NOM p-val (already filtered during ingestion).

    Returns dict mapping mutant_id -> {go_id: nom_pval}.
    """
    ...


def build_pvalue_matrix(
    per_mutant_pvals: dict[str, dict[str, float]],
    mutant_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build the GO term x mutant p-value matrix with imputation.

    Missing entries are imputed as p = 1.0.

    Returns:
        matrix: np.ndarray of shape (n_go_terms, n_mutants)
        go_id_order: list of GO IDs corresponding to matrix rows
    """
    ...


def compute_fisher_combined(
    pvalue_matrix: np.ndarray,
    n_mutants: int,
) -> np.ndarray:
    """Compute Fisher's combined p-value for each GO term (row).

    Fisher statistic: X^2 = -2 * sum(ln(p_i))
    Combined p-value from chi-squared distribution with 2k degrees of freedom.

    Returns array of combined p-values, one per row.
    """
    ...


def run_fisher_analysis(
    cohort: CohortData,
    config: FisherConfig,
    output_dir: Path,
    clustering_enabled: bool,
) -> FisherResult:
    """Top-level entry point for Fisher's combined probability analysis.

    Writes pvalue_matrix.tsv to output_dir.
    If clustering_enabled is False, also writes fisher_combined_pvalues.tsv.

    Returns FisherResult with all computed values.
    """
    ...


def write_pvalue_matrix_tsv(
    matrix: np.ndarray,
    go_id_order: list[str],
    go_id_to_name: dict[str, str],
    mutant_ids: list[str],
    output_dir: Path,
) -> Path:
    """Write the p-value matrix to pvalue_matrix.tsv."""
    ...


def write_fisher_results_tsv(
    fisher_result: FisherResult,
    output_dir: Path,
) -> Path:
    """Write fisher_combined_pvalues.tsv without cluster assignments.

    Used when clustering is disabled.
    """
    ...


# --- Invariants ---
# Pre-conditions
assert len(cohort.mutant_ids) >= 2, "Fisher's method requires at least 2 mutant lines"
assert config.pseudocount > 0, "Pseudocount must be positive"
assert output_dir.is_dir(), "Output directory must exist"

# Post-conditions
assert len(fisher_result.combined_pvalues) == len(fisher_result.go_ids), "One combined p-value per GO ID"
assert all(0.0 <= p <= 1.0 for p in fisher_result.combined_pvalues.values()), "Combined p-values must be in [0, 1]"
assert fisher_result.pvalue_matrix.shape == (len(fisher_result.go_id_order), len(fisher_result.mutant_ids)), "Matrix shape must match GO IDs x mutants"
assert fisher_result.n_mutants == len(fisher_result.mutant_ids), "n_mutants must match mutant_ids length"
```

### Tier 3 -- Behavioral Contracts

**Error conditions:**

| Exception | Description | Trigger |
|---|---|---|
| `OSError` | Output write failure | Cannot write TSV files to `output_dir` |

**Behavioral contracts:**

1. Per-mutant p-value dictionaries are keyed by GO ID (not term name), extracted from the `go_id` field of `TermRecord`.
2. If `NOM p-val` is exactly 0.0, it is replaced with the configured pseudocount (default 1e-10) before inclusion in the matrix. This avoids negative infinity in the log transform.
3. The p-value matrix has shape (n_GO_terms, n_mutants). Missing entries (GO term absent from a mutant) are imputed as p = 1.0, contributing 0 to the Fisher statistic since ln(1.0) = 0.
4. The Fisher statistic for each GO term is X^2 = -2 * sum(ln(p_i)) with k = total number of mutant lines. The degrees of freedom are 2k, constant across all GO terms due to imputation.
5. Combined p-values are computed from the chi-squared survival function (1 - CDF) with 2k degrees of freedom.
6. If `apply_fdr` is True in config, Benjamini-Hochberg FDR correction is applied to the combined p-values. The corrected p-values are stored in `FisherResult.corrected_pvalues`. If False, `corrected_pvalues` is None.
8. The number of contributing mutant lines per GO term counts how many mutant lines have p < 1.0 (i.e., the term was actually observed, not imputed) for that GO term.
9. `pvalue_matrix.tsv` is always written to `output_dir`. Columns are mutant IDs, rows are GO IDs with an additional column for GO term name.
10. When `clustering_enabled` is False, `fisher_combined_pvalues.tsv` is also written by this unit, containing GO ID, GO term name, combined p-value, and number of contributing lines. No cluster assignment column is included.
11. The `go_id_to_name` mapping allows downstream units and TSV outputs to display human-readable term names alongside GO IDs.

**Dependencies:** Unit 1 (CohortData), Unit 2 (FisherConfig).

---

## Unit 7 -- GO Semantic Clustering

### Tier 1 -- Description

This unit performs GO semantic similarity clustering to reduce redundancy among significantly dysregulated GO terms identified by Fisher's method. It is executed only when clustering is enabled in the configuration. Step 1: pre-filter GO terms to those with combined p-value below a configurable threshold (default 0.05). Step 2: download or load the GO OBO ontology file and the Drosophila melanogaster Gene Annotation File (GAF) from configurable URLs. Both files are cached locally after download. Step 3: compute information content for all GO terms from annotation frequencies in the GAF. Step 4: compute pairwise Lin semantic similarity between all pre-filtered GO terms. Step 5: perform hierarchical agglomerative clustering on the similarity matrix, cutting at a configurable similarity threshold (default 0.7). Step 6: within each cluster, select the representative GO term as the one with the lowest combined Fisher p-value. The unit writes `fisher_combined_pvalues.tsv` to the output directory with cluster assignments included.

### Tier 2 — Signatures

```python
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# Upstream types
from gsea_tool.configuration import ClusteringConfig
from gsea_tool.meta_analysis import FisherResult


@dataclass
class ClusteringResult:
    """Results of GO semantic similarity clustering."""
    representatives: list[str]  # GO IDs of cluster representatives, ordered by combined p-value
    representative_names: list[str]  # term names of representatives, same order
    representative_pvalues: list[float]  # combined p-values of representatives, same order
    representative_n_contributing: list[int]  # contributing line counts of representatives, same order
    cluster_assignments: dict[str, int]  # GO ID -> cluster index for all pre-filtered terms
    n_clusters: int  # total number of clusters formed
    n_prefiltered: int  # number of GO terms that passed the pre-filter
    similarity_metric: str  # e.g. "Lin"
    similarity_threshold: float  # clustering cut height
    ...


def download_or_load_obo(obo_url: str, cache_dir: Path) -> Path:
    """Download the GO OBO file if not cached, or return cached path.

    Returns path to the local OBO file.
    """
    ...


def download_or_load_gaf(gaf_url: str, cache_dir: Path) -> Path:
    """Download the Drosophila GAF file if not cached, or return cached path.

    Returns path to the local GAF file.
    """
    ...


def compute_information_content(obo_path: Path, gaf_path: Path) -> dict[str, float]:
    """Compute information content for GO terms from annotation frequencies.

    Returns dict mapping GO ID -> information content value.
    """
    ...


def compute_lin_similarity(
    go_ids: list[str],
    ic_values: dict[str, float],
    obo_path: Path,
) -> np.ndarray:
    """Compute pairwise Lin similarity matrix for a list of GO IDs.

    Returns symmetric matrix of shape (n, n) with values in [0, 1].
    """
    ...


def cluster_by_similarity(
    similarity_matrix: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Hierarchical agglomerative clustering on the similarity matrix.

    Cuts the dendrogram at the given similarity threshold.

    Returns list of clusters, where each cluster is a list of row indices.
    """
    ...


def select_representatives(
    clusters: list[list[int]],
    go_ids: list[str],
    fisher_result: FisherResult,
) -> ClusteringResult:
    """Select representative GO term per cluster (lowest combined p-value).

    Returns ClusteringResult with representatives ordered by combined p-value.
    """
    ...


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
    ...


def write_fisher_results_with_clusters_tsv(
    fisher_result: FisherResult,
    clustering_result: ClusteringResult,
    output_dir: Path,
) -> Path:
    """Write fisher_combined_pvalues.tsv with cluster assignment and representative columns."""
    ...


# --- Invariants ---
# Pre-conditions
assert config.similarity_threshold > 0.0, "Similarity threshold must be positive"
assert config.similarity_threshold <= 1.0, "Similarity threshold must be at most 1.0"

# Post-conditions
assert len(clustering_result.representatives) == clustering_result.n_clusters, "One representative per cluster"
assert clustering_result.n_clusters > 0, "At least one cluster must be formed"
assert all(
    go_id in fisher_result.combined_pvalues
    for go_id in clustering_result.representatives
), "All representatives must be present in Fisher results"
assert clustering_result.representatives == sorted(
    clustering_result.representatives,
    key=lambda gid: fisher_result.combined_pvalues[gid]
), "Representatives must be ordered by combined p-value ascending"
```

### Tier 3 -- Behavioral Contracts

**Error conditions:**

| Exception | Description | Trigger |
|---|---|---|
| `ConnectionError` | OBO download failure | Cannot download the GO OBO file from the configured URL after retry |
| `ConnectionError` | GAF download failure | Cannot download the Drosophila GAF file from the configured URL after retry |
| `OSError` | Output write failure | Cannot write fisher_combined_pvalues.tsv to `output_dir` |
| `ValueError` | No terms pass pre-filter | Zero GO terms have combined p-value below the pre-filter threshold |

**Behavioral contracts:**

1. Only GO terms with combined p-value below the configured pre-filter threshold (default 0.05) are included in the clustering. GO terms above this threshold are excluded from similarity computation.
2. The GO OBO file and Drosophila GAF file are downloaded from configurable URLs. Downloaded files are cached in a `cache/` subdirectory of the project directory. If already cached, the cached version is used without re-downloading.
3. Download failures are retried once. If the retry also fails, the unit raises `ConnectionError` with a descriptive message.
4. Information content is computed from annotation frequencies in the Drosophila GAF, using the GO hierarchy from the OBO file. GO terms not present in the GAF receive an IC of 0.0.
5. Lin similarity is computed as: sim(t1, t2) = 2 * IC(MICA) / (IC(t1) + IC(t2)), where MICA is the most informative common ancestor. If IC(t1) + IC(t2) = 0, similarity is 0.0.
6. Hierarchical agglomerative clustering uses average linkage on a distance matrix derived from the similarity matrix (distance = 1 - similarity). The dendrogram is cut at distance = 1 - similarity_threshold.
8. Within each cluster, the representative is the GO term with the lowest combined Fisher p-value (most significant).
9. Representatives are ordered by their combined p-value (most significant first).
10. `fisher_combined_pvalues.tsv` is written with columns: GO ID, GO term name, combined p-value, number of contributing lines, cluster index, and whether the term is a representative. All pre-filtered GO terms are included, not just representatives.
11. Given the same input data and the same configuration, the output is deterministic.

**Dependencies:** Unit 2 (ClusteringConfig), Unit 6 (FisherResult).

---

## Unit 8 -- Bar Plot Rendering

### Tier 1 -- Description

This unit renders the horizontal bar plot for Figure 3 (the meta-analysis figure). It receives either a `ClusteringResult` (when clustering is enabled, providing representative GO terms) or the raw `FisherResult` (when clustering is disabled, using the top N GO terms by combined p-value directly). The Y-axis shows representative GO term names ordered by combined p-value (most significant at top). The X-axis shows -log10(combined p-value). Bar color encodes the number of contributing mutant lines using a sequential colormap. The number of contributing lines is annotated on or next to each bar. A vertical dashed significance line is drawn at -log10(0.05). The figure uses clean, minimal styling consistent with the dot plot figures and suitable for publication. Output is saved to PDF, PNG, and SVG in the output directory.

### Tier 2 — Signatures

```python
from pathlib import Path
from dataclasses import dataclass

# Upstream types
from gsea_tool.configuration import FisherConfig, PlotAppearanceConfig
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.go_clustering import ClusteringResult


@dataclass
class BarPlotResult:
    """Metadata about the rendered bar plot figure, for notes.md consumption."""
    pdf_path: Path
    png_path: Path
    svg_path: Path
    n_bars: int
    n_mutants: int
    clustering_was_used: bool
    ...


def render_bar_plot(
    fisher_result: FisherResult,
    clustering_result: ClusteringResult | None,
    fisher_config: FisherConfig,
    plot_config: PlotAppearanceConfig,
    output_dir: Path,
    output_stem: str = "figure3_meta_analysis",
) -> BarPlotResult:
    """Render the meta-analysis bar plot and save to PDF, PNG, and SVG.

    If clustering_result is provided, uses representative terms.
    If clustering_result is None, uses top N terms by combined p-value.

    Returns BarPlotResult with paths and summary counts.
    """
    ...


def select_bar_data(
    fisher_result: FisherResult,
    clustering_result: ClusteringResult | None,
    top_n: int,
) -> tuple[list[str], list[float], list[int]]:
    """Select GO terms, p-values, and contributing counts for the bar plot.

    Returns:
        term_names: Display names for Y-axis labels.
        neg_log_pvalues: -log10(combined p-value) for X-axis.
        n_contributing: Number of contributing mutant lines for color encoding.
    All lists are ordered by combined p-value (most significant first).
    """
    ...


# --- Invariants ---
# Pre-conditions
assert output_dir.is_dir(), "Output directory must exist"
assert fisher_config.top_n_bars > 0, "top_n_bars must be positive"

# Post-conditions
assert result.pdf_path.exists(), "PDF file must be written"
assert result.png_path.exists(), "PNG file must be written"
assert result.svg_path.exists(), "SVG file must be written"
assert result.n_bars > 0, "At least one bar must be plotted"
assert result.n_bars <= fisher_config.top_n_bars, "Number of bars cannot exceed top_n_bars"
```

### Tier 3 -- Behavioral Contracts

**Error conditions:**

| Exception | Description | Trigger |
|---|---|---|
| `ValueError` | No terms to plot | No GO terms have combined p-value below the pre-filter (clustered mode) or no GO terms exist at all (unclustered mode) |
| `OSError` | Output write failure | Cannot write PDF, PNG, or SVG to `output_dir` |

**Behavioral contracts:**

1. When `clustering_result` is provided (clustering enabled), the bar plot shows the top N representative GO terms, ordered by combined p-value (most significant at top). N is limited by `fisher_config.top_n_bars` or the number of representatives, whichever is smaller.
2. When `clustering_result` is None (clustering disabled), the bar plot shows the top N GO terms by combined p-value directly, without redundancy reduction. N is limited by `fisher_config.top_n_bars` or the total number of GO terms, whichever is smaller.
3. The Y-axis displays GO term names (not GO IDs), formatted as readable labels. Names longer than `plot_config.label_max_length` characters are truncated with an ellipsis.
4. The X-axis displays -log10(combined p-value).
5. Bar color encodes the number of contributing mutant lines (lines with p < 1.0 for that term) using the sequential colormap specified in `plot_config.bar_colormap` (default YlOrRd). A colorbar legend is included.
6. When `plot_config.show_recurrence_annotation` is True, the number of contributing mutant lines is displayed as text on or next to each bar.
8. When `plot_config.show_significance_line` is True, a vertical dashed line is drawn at -log10(0.05) to indicate nominal significance.
9. The figure dimensions are set by `plot_config.bar_figure_width` and `plot_config.bar_figure_height`.
10. The figure is saved to `{output_stem}.pdf`, `{output_stem}.png`, and `{output_stem}.svg` in `output_dir` at the configured DPI.
11. The figure uses clean, minimal styling consistent with the dot plot figures and Nature-family journal standards: no background color, no unnecessary borders, tight layout.

**Dependencies:** Unit 2 (FisherConfig, PlotAppearanceConfig), Unit 6 (FisherResult), Unit 7 (ClusteringResult).

---

## Unit 9 -- Notes Generation

### Tier 1 -- Description

This unit generates the `notes.md` Markdown file containing all manuscript-support text. It receives summary information from all upstream units and assembles five sections: (1) figure legend text for each produced figure -- describing the visual encoding, thresholds, and cohort size for the dot plots, and the statistical method, bar encoding, and clustering approach for the bar plot; (2) unified materials and methods text describing the full analysis pipeline, including that GSEA output was consumed not generated, GO term selection criteria for each figure, Figure 2 clustering parameters, Figure 2 redundancy removal method, Figure 3 Fisher's method with imputation details, Figure 3 GO semantic similarity clustering approach, and software dependencies with versions; (3) summary statistics including number of mutants, total GO terms, significant terms, terms displayed per figure, Fisher pre-filter counts, and number of semantic clusters; (4) a reproducibility note with random seeds, software versions, and configuration parameters used; (5) a configuration guide describing all `config.yaml` parameters with their defaults. The file is written to the output directory. All text is written as copy-paste-ready prose suitable for a manuscript draft. The Figure 1 legend is included only when Figure 1 was produced.

### Tier 2 — Signatures

```python
from pathlib import Path
from dataclasses import dataclass

# Upstream types
from gsea_tool.data_ingestion import CohortData
from gsea_tool.configuration import ToolConfig
from gsea_tool.unbiased import UnbiasedSelectionStats
from gsea_tool.dot_plot import DotPlotResult
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.go_clustering import ClusteringResult
from gsea_tool.bar_plot import BarPlotResult


@dataclass
class NotesInput:
    """All inputs needed to generate notes.md, gathered by orchestration."""
    cohort: CohortData
    config: ToolConfig
    fig1_result: DotPlotResult | None  # None if Figure 1 was not produced
    fig1_method: str | None  # "ontology" or "tsv" or None if Figure 1 was not produced
    fig2_result: DotPlotResult
    fig3_result: BarPlotResult
    unbiased_stats: UnbiasedSelectionStats
    fisher_result: FisherResult
    clustering_result: ClusteringResult | None  # None if clustering was disabled
    ...


def generate_notes(
    notes_input: NotesInput,
    output_dir: Path,
) -> Path:
    """Generate notes.md and write it to output_dir.

    Returns the path to the written file.
    """
    ...


def format_figure_legends(
    notes_input: NotesInput,
) -> str:
    """Generate the figure legend text section for all produced figures."""
    ...


def format_methods_text(
    notes_input: NotesInput,
) -> str:
    """Generate the unified materials and methods text section."""
    ...


def format_summary_statistics(
    notes_input: NotesInput,
) -> str:
    """Generate the summary statistics section."""
    ...


def format_reproducibility_note(
    notes_input: NotesInput,
) -> str:
    """Generate the reproducibility note with seeds and versions."""
    ...


def format_config_guide(
    notes_input: NotesInput,
) -> str:
    """Generate the configuration guide section describing all config.yaml parameters."""
    ...


def get_dependency_versions() -> dict[str, str]:
    """Collect version strings for all key dependencies (Python, matplotlib, pandas, scipy, numpy, goatools, pyyaml)."""
    ...


# --- Invariants ---
# Pre-conditions
assert output_dir.is_dir(), "Output directory must exist"

# Post-conditions
assert notes_path.exists(), "notes.md must be written"
assert notes_path.name == "notes.md", "Output filename must be notes.md"
```

### Tier 3 -- Behavioral Contracts

**Error conditions:**

| Exception | Description | Trigger |
|---|---|---|
| `OSError` | Output write failure | Cannot write notes.md to `output_dir` |

**Behavioral contracts:**

1. The output file is named exactly `notes.md` and is written to `output_dir`.
2. The file contains five sections: Figure Legends, Materials and Methods, Summary Statistics, Reproducibility Note, and Configuration Guide.
3. The Figure Legends section describes each produced figure. For Figures 1 and 2: what dot color represents (NES, diverging red-blue), what dot size represents (-log10 FDR), what empty cells mean (term not significant), what category boxes represent, the FDR threshold used, and the number of mutants. For Figure 1, the legend additionally states whether categories were resolved via GO ontology ancestry (naming the parent GO IDs and their labels) or via a user-supplied category mapping file, depending on `fig1_method`. For Figure 3: what the bar plot shows (top N representative dysregulated pathways), what bar length encodes (-log10 combined p-value), what bar color encodes (number of contributing mutant lines), the statistical method (Fisher's combined probability test), and the GO semantic clustering step (if used). The Figure 1 legend is omitted if Figure 1 was not produced.
4. The Materials and Methods section states that GSEA preranked output was consumed (not generated), describes the GO term selection criteria for each figure (for Figure 1: ontology-based resolution from parent GO IDs when `fig1_method == "ontology"`, or manual category mapping when `fig1_method == "tsv"`), names the clustering algorithm and its parameters for Figure 2 (Ward linkage, number of clusters, random seed), describes the redundancy removal method for Figure 2 (word-set Jaccard similarity > 0.5), describes the Fisher's method for Figure 3 (merging of pos/neg tables, imputation of p = 1.0 for absent terms, degrees of freedom), describes the GO semantic similarity clustering approach for Figure 3 (Lin similarity, information content source, clustering threshold, representative selection rule), and lists software dependencies with version numbers. If clustering was disabled for Figure 3, the methods text notes that clustering was not applied and raw top-N terms were used.
5. The Summary Statistics section reports: number of mutants analyzed, total unique GO terms in the input data, number of GO terms passing the FDR threshold in at least one mutant, number of GO terms displayed in each produced figure, number of GO terms passing the Fisher pre-filter, and number of semantic clusters formed (if clustering was used).
6. The Reproducibility Note states the random seed used for Figure 2 clustering, software versions, and all configuration parameters used in the run.
8. The Configuration Guide describes all `config.yaml` parameters with their defaults, organized by section, explaining how to modify graphical elements and analysis parameters.
9. All text is written as copy-paste-ready prose. No code blocks, no raw data dumps.
10. Software version strings are obtained at runtime, not hardcoded.

**Dependencies:** Unit 1 (CohortData), Unit 2 (ToolConfig), Unit 4 (UnbiasedSelectionStats), Unit 5 (DotPlotResult), Unit 6 (FisherResult), Unit 7 (ClusteringResult), Unit 8 (BarPlotResult).

---

## Unit 10 -- Orchestration

### Tier 1 -- Description

This unit is the top-level entry point for the tool. Figure 1 is produced when either (a) `cherry_pick_categories` is non-empty in the loaded config, or (b) a category mapping TSV file is provided as a CLI argument. If both are present, the config-based ontology approach takes precedence and a warning is printed to stderr. If neither is present, only Figures 2 and 3 are produced. All other parameters are controlled via `config.yaml`. The unit resolves the project directory (the directory containing the script), the data directory (`data/` inside the project directory), and the output directory (`output/` inside the project directory, created automatically if it does not exist). It loads the configuration (Unit 2), ingests data (Unit 1), then executes the dot plot path and meta-analysis path. The dot plot path: optionally select cherry-picked terms (Unit 3, via ontology resolution or TSV mapping), select unbiased terms (Unit 4), render Figure 1 (Unit 5, conditional), render Figure 2 (Unit 5). When using the ontology path, the OBO file is obtained via Unit 7's `download_or_load_obo` function using the URL from `config.clustering.go_obo_url`. The meta-analysis path: run Fisher's analysis (Unit 6), optionally run GO clustering (Unit 7, if enabled in config), render Figure 3 (Unit 8). Finally, generate notes.md (Unit 9) with `fig1_method` set to `"ontology"`, `"tsv"`, or `None` accordingly. This unit contains no domain logic -- it is pure wiring and CLI interface.

### Tier 2 — Signatures

```python
from pathlib import Path
import argparse


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Optional argument:
        mapping_file: Path to the GO term category mapping file.
                      If provided, Figure 1 is produced.
                      If omitted, only Figures 2 and 3 are produced.
    """
    ...


def resolve_paths(project_dir: Path, mapping_file: str | None) -> tuple[Path, Path, Path, Path | None]:
    """Resolve and validate the data directory, output directory, cache directory, and optional mapping file path.

    Returns (data_dir, output_dir, cache_dir, mapping_path_or_none).
    data_dir is always <project_dir>/data/.
    output_dir is always <project_dir>/output/ (created if absent).
    cache_dir is always <project_dir>/cache/ (created if absent).

    Raises FileNotFoundError if data_dir does not exist.
    Raises FileNotFoundError if mapping_file is specified but does not exist.
    """
    ...


def main() -> None:
    """Top-level entry point. Parses arguments and orchestrates all units.

    Exit code 0 on success, 1 on any error (with message printed to stderr).
    """
    ...


# --- Invariants ---
# Pre-conditions
assert Path(project_dir / "data").is_dir(), "data/ directory must exist in the project directory"

# Post-conditions
# On success: figure2 PDF/PNG/SVG, figure3 PDF/PNG/SVG, notes.md, pvalue_matrix.tsv,
# and fisher_combined_pvalues.tsv all exist in output_dir.
# If mapping file was provided: figure1 PDF/PNG/SVG also exist in output_dir.
```

### Tier 3 -- Behavioral Contracts

**Error conditions:**

| Exception | Description | Trigger |
|---|---|---|
| `FileNotFoundError` | Missing data directory | The `data/` directory does not exist in the project directory |
| `FileNotFoundError` | Missing mapping file | The specified category mapping file does not exist |

**Behavioral contracts:**

1. The tool has no required CLI arguments. The only optional positional argument is the path to the category mapping file for Figure 1 (retained for backward compatibility; the preferred approach is to configure cherry-pick categories in `config.yaml`).
2. All tunable parameters are controlled via `config.yaml`. The tool does not accept parameter overrides via CLI flags.
3. The project directory is resolved as the directory containing the script.
4. The data directory is resolved as `data/` inside the project directory. It must exist.
5. The output directory is resolved as `output/` inside the project directory. It is created automatically if it does not exist.
6. The cache directory is resolved as `cache/` inside the project directory. It is created automatically if it does not exist and passed to Unit 7 for OBO/GAF file caching.
8. Configuration is loaded first (Unit 2). If `config.yaml` exists and is invalid, the tool halts before any data processing.
8. Data ingestion (Unit 1) runs next. If it fails (e.g., fewer than 2 mutants), the tool halts.
9. The invocation order is: load config (Unit 2), ingest data (Unit 1), optionally select cherry-picked terms (Unit 3, via ontology or TSV), select unbiased terms (Unit 4), optionally render Figure 1 (Unit 5), render Figure 2 (Unit 5), run Fisher analysis (Unit 6), optionally run GO clustering (Unit 7), render Figure 3 (Unit 8), generate notes (Unit 9).
10. Figure 1 is produced when: (a) `config.cherry_pick_categories` is non-empty (ontology path), or (b) a category mapping file was provided as a CLI argument (TSV path). If both are present, config takes precedence with a warning to stderr. If neither, Figure 1 is skipped.
11. Figure 3 is always produced. When clustering is disabled in config, the bar plot shows unclustered top-N terms.
12. If any unit raises an exception, the tool prints a descriptive error message to stderr and exits with code 1. No partial output is guaranteed.
13. The tool prints a brief summary to stdout on success: number of mutants processed, figures produced, and output file paths.
14. Figure 1 output files use the stem `figure1_cherry_picked`. Figure 2 output files use the stem `figure2_unbiased`. Figure 3 output files use the stem `figure3_meta_analysis`.
15. The following files are always produced in `output/`: `figure2_unbiased.{pdf,png,svg}`, `figure3_meta_analysis.{pdf,png,svg}`, `pvalue_matrix.tsv`, `fisher_combined_pvalues.tsv`, `notes.md`.
16. When a mapping file is provided, `figure1_cherry_picked.{pdf,png,svg}` is additionally produced in `output/`.

**Dependencies:** Unit 1 (ingest_data), Unit 2 (load_config, CherryPickCategory), Unit 3 (parse_category_mapping, select_cherry_picked_terms, resolve_categories_from_ontology), Unit 4 (select_unbiased_terms), Unit 5 (render_dot_plot), Unit 6 (run_fisher_analysis), Unit 7 (run_semantic_clustering, download_or_load_obo), Unit 8 (render_bar_plot), Unit 9 (generate_notes).

---

## Dependency Summary

```
Unit 1:  Data Ingestion              -> (no dependencies)
Unit 2:  Configuration               -> (no dependencies)
Unit 3:  Cherry-Picked Selection     -> Unit 1, Unit 2 (CherryPickCategory type)
Unit 4:  Unbiased Selection          -> Unit 1, Unit 3 (CategoryGroup type only)
Unit 5:  Dot Plot Rendering          -> Unit 1, Unit 3 (CategoryGroup type only)
Unit 6:  Meta-Analysis Computation   -> Unit 1, Unit 2
Unit 7:  GO Semantic Clustering      -> Unit 2, Unit 6
Unit 8:  Bar Plot Rendering          -> Unit 2, Unit 6, Unit 7
Unit 9:  Notes Generation            -> Unit 1, Unit 2, Unit 4, Unit 5, Unit 6, Unit 7, Unit 8
Unit 10: Orchestration               -> Unit 1, Unit 2, Unit 3, Unit 4, Unit 5, Unit 6, Unit 7, Unit 8, Unit 9
```

All dependencies point backward. No circular dependencies exist.
