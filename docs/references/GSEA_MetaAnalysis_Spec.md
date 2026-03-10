# Specification Document: GSEA Meta-Analysis Pipeline

## 1. Overview

This document specifies a Python command-line tool that combines independent Gene Set Enrichment Analysis (GSEA) results from multiple Drosophila mutant lines into a unified summary visualization. Each mutant line has been compared individually against a shared control, yielding per-line enrichment tables for positively and negatively enriched GO terms. The tool applies Fisher's combined probability method to aggregate evidence of pathway dysregulation across all mutant lines, clusters redundant GO terms by semantic similarity, and produces a publication-ready bar plot of the top representative dysregulated pathways.

---

## 2. Input Data

### 2.1 Directory Structure

```
project/
├── data/
│   ├── mutant_A/
│   │   ├── gsea_report_for_na_pos_<timestamp>.tsv
│   │   ├── gsea_report_for_na_neg_<timestamp>.tsv
│   │   └── (other files — ignored)
│   ├── mutant_B/
│   │   ├── gsea_report_for_na_pos_<timestamp>.tsv
│   │   ├── gsea_report_for_na_neg_<timestamp>.tsv
│   │   └── (other files — ignored)
│   └── ...
├── config.yaml          (optional, see Section 7)
├── gsea_meta_analysis.py
└── output/              (created by the tool)
```

### 2.2 File Discovery Rules

1. The tool receives the path to the `data/` directory as a required argument.
2. It iterates over all **immediate subdirectories** of `data/`. Each subdirectory is treated as one mutant line. The subdirectory name is used as the mutant line identifier.
3. Within each subdirectory, the tool searches for files matching these patterns (case-insensitive):
   - **Positive enrichment table**: filename contains both `gsea_report` and `pos` and ends with `.tsv`
   - **Negative enrichment table**: filename contains both `gsea_report` and `neg` and ends with `.tsv`
4. **Nested subdirectories** within each mutant folder are **ignored**. Only files at the first level are scanned.
5. If a mutant folder contains no matching GSEA files, it is skipped with a warning to stderr.
6. If a mutant folder contains multiple matching files for the same direction (e.g., two `pos` files), the tool exits with an error identifying the ambiguous folder.

### 2.3 GSEA Table Format

Each `.tsv` file is a tab-separated table with the following relevant columns:

| Column | Description | Used |
|--------|-------------|------|
| `NAME` | GO term ID and name, space-separated (e.g., `GO:0007602 PHOTOTRANSDUCTION`) | **Yes** — parse GO ID and term name |
| `SIZE` | Number of genes in the gene set | No |
| `ES` | Enrichment score (raw) | No |
| `NES` | Normalized enrichment score (signed) | No |
| `NOM p-val` | Nominal p-value | **Yes** — input to Fisher's method |
| `FDR q-val` | FDR-adjusted p-value | No |
| `FWER p-val` | Family-wise error rate p-value | No |
| `RANK AT MAX` | Rank at maximum enrichment | No |
| `LEADING EDGE` | Leading edge statistics | No |

#### Parsing rules for the `NAME` column

- Extract the GO ID as the first token matching the regex `GO:\d{7}`.
- Extract the term name as everything following the GO ID, stripped of leading/trailing whitespace.
- If a row does not contain a valid GO ID, skip it with a warning.

#### Handling of p-value edge cases

- If `NOM p-val` is exactly `0.0`, replace with a small pseudocount: `1e-10`. This avoids `−∞` in the log transform. The pseudocount value should be configurable (see Section 7).
- If `NOM p-val` is missing or non-numeric, skip the row with a warning.

---

## 3. Processing Pipeline

### Step 1: Per-Mutant Table Merging

For each mutant line:

1. Load the positive and negative enrichment tables.
2. Merge them into a single table keyed by GO ID.
3. **Conflict resolution**: If a GO term appears in **both** tables for the same mutant, retain the entry with the **smaller nominal p-value** (strongest evidence of dysregulation in either direction).
4. The output of this step is a dictionary: `{GO_ID: nominal_p_value}` per mutant line.

### Step 2: Cross-Mutant P-Value Matrix

1. Collect the union of all GO IDs observed across all mutant lines.
2. Build a matrix of shape `(n_GO_terms × n_mutants)`.
3. Fill each cell with the nominal p-value from Step 1.
4. For missing entries (GO term not found in a mutant line), **impute p = 1.0**. Rationale: absence of enrichment is treated as absence of evidence; `ln(1.0) = 0` contributes nothing to the Fisher statistic.

### Step 3: Fisher's Combined Probability Test

For each GO term (row of the matrix):

1. Compute the Fisher statistic:
   $$X^2 = -2 \sum_{i=1}^{k} \ln(p_i)$$
   where $k$ is the total number of mutant lines (constant across all GO terms due to imputation).
2. Compute the combined p-value from a χ² distribution with $2k$ degrees of freedom.
3. Store the combined p-value for each GO term.

**Multiple testing correction** (optional, configurable): Apply Benjamini-Hochberg FDR correction on the combined p-values. Default: **no correction** (exploratory mode). The user can enable it and set a threshold via the configuration file.

### Step 4: GO Semantic Similarity Clustering

1. Compute pairwise semantic similarity between all GO terms that pass a pre-filter (combined p-value < configurable threshold, default: 0.05). This avoids computing similarity on thousands of irrelevant terms.
2. **Similarity metric**: Lin similarity using the GO graph structure (via the `goatools` or `semantic-similarity` library — see Section 6).
3. **Clustering method**: Hierarchical agglomerative clustering on the similarity matrix, cut at a configurable similarity threshold (default: 0.7). Alternative: affinity propagation.
4. **Representative selection**: Within each cluster, select the GO term with the **best (lowest) combined Fisher p-value** as the representative.
5. Store the cluster assignments and representative terms.

### Step 5: Visualization

Generate a horizontal bar plot:

- **Y-axis**: Representative GO term names (formatted as readable labels, not GO IDs), ordered by combined Fisher p-value (most significant at top).
- **X-axis**: $-\log_{10}(\text{combined p-value})$.
- **Number of bars**: Top N representative terms (default: 20, configurable).
- **Color**: Single color or a gradient mapped to the number of mutant lines contributing (i.e., how many lines had p < 1.0 for that term). This adds a recurrence dimension without requiring a full heatmap.
- **Optional annotation**: Display the number of contributing mutant lines as text on or next to each bar.
- **Optional threshold line**: A vertical dashed line at $-\log_{10}(0.05)$ to indicate nominal significance.

---

## 4. Outputs

All outputs are written to an `output/` directory (created if absent) inside the project root (or a user-specified output path).

### 4.1 Figure

- `gsea_meta_barplot.svg` — Scalable vector graphic (publication quality).
- `gsea_meta_barplot.pdf` — PDF version.

### 4.2 Supplementary Markdown File (`gsea_meta_analysis_report.md`)

This file contains three sections:

#### Section 1: Figure Legend

A ready-to-use figure legend for publication, written in the style of a scientific journal. It should describe:

- What the figure shows (top N representative dysregulated pathways).
- The statistical method (Fisher's combined probability test on nominal GSEA p-values across N mutant lines).
- The GO semantic clustering step and how representatives were selected.
- What the bar length and color encode.
- The number of mutant lines analyzed.
- Any thresholds used.

#### Section 2: Materials and Methods

A detailed paragraph suitable for the Methods section of a manuscript, describing:

- The proteomic comparison design (individual mutant vs. shared control).
- The GSEA procedure (referencing the tool used, if known, or left as a placeholder).
- The meta-analysis strategy: merging of positive/negative tables, Fisher's method with imputation of p = 1.0 for absent terms, degrees of freedom.
- The semantic similarity clustering approach (metric, threshold, representative selection).
- Software and library versions used.

#### Section 3: Configuration Guide

Either:

**(Option A)** A detailed guide explaining how to modify graphical elements of the figure by editing the configuration file (see Section 7), including all available parameters with descriptions and defaults.

**(Option B)** If no configuration file is used, a guide explaining which code variables to modify and where to find them.

**Preferred: Option A** — the tool should ship with a `config.yaml` that controls all user-facing parameters.

### 4.3 Intermediate Data (optional, configurable)

If enabled via configuration:

- `fisher_combined_pvalues.tsv` — Full table of GO terms with combined p-values, number of contributing lines, and cluster assignments.
- `pvalue_matrix.tsv` — The raw GO term × mutant line p-value matrix before combination.

---

## 5. Command-Line Interface

```
python gsea_meta_analysis.py --data <path_to_data_dir> [--config <config.yaml>] [--output <output_dir>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data` | Yes | — | Path to the `data/` directory containing mutant subdirectories |
| `--config` | No | `config.yaml` in working directory (if present) | Path to YAML configuration file |
| `--output` | No | `./output/` | Output directory |

If no config file is found, the tool uses built-in defaults (see Section 7).

---

## 6. Dependencies

| Package | Purpose | Notes |
|---------|---------|-------|
| `pandas` | Data manipulation | |
| `numpy` | Numerical operations | |
| `scipy` | Fisher's method (`scipy.stats.chi2`, `scipy.stats.combine_pvalues`) | |
| `matplotlib` | Plotting | |
| `goatools` | GO graph parsing, semantic similarity | Requires GO OBO file (auto-downloaded) |
| `pyyaml` | Configuration parsing | |
| `statsmodels` | FDR correction (`multipletests`) | Optional, only if FDR enabled |

**Note on GO semantic similarity in Python**: The Python ecosystem for GO semantic similarity is less mature than R's. The recommended approach is:

1. Use `goatools` to load the GO DAG (OBO file).
2. Implement Lin similarity using information content derived from annotation frequencies, or use a precomputed IC file.
3. Alternatively, if this proves cumbersome, use `rrvgo` via `rpy2` as a bridge to R, or precompute the similarity matrix with a lightweight R script called from Python. The spec should support both approaches; the configuration file should allow the user to select the backend.

**Fallback if semantic clustering is too complex**: As a simpler alternative, skip the clustering step entirely and just show the top N GO terms by combined p-value, with a note in the legend that redundant terms may be present. This should be a configurable option (`clustering: enabled: false`).

---

## 7. Configuration File (`config.yaml`)

```yaml
# ============================================================
# GSEA Meta-Analysis Pipeline Configuration
# ============================================================

# --- Input parsing ---
input:
  pos_pattern: "gsea_report.*pos.*\\.tsv$"   # Regex for positive enrichment files
  neg_pattern: "gsea_report.*neg.*\\.tsv$"   # Regex for negative enrichment files
  pvalue_column: "NOM p-val"                  # Column name for nominal p-values
  name_column: "NAME"                         # Column name for GO term identifiers
  zero_pvalue_pseudocount: 1.0e-10            # Replacement for p-values of exactly 0.0

# --- Fisher's method ---
fisher:
  apply_fdr: false                            # Apply BH-FDR correction on combined p-values
  fdr_threshold: 0.25                         # FDR threshold (only used if apply_fdr is true)
  prefilter_pvalue: 0.05                      # Only cluster GO terms with combined p below this

# --- Semantic clustering ---
clustering:
  enabled: true                               # Set to false to skip clustering
  similarity_metric: "Lin"                    # Similarity metric: Lin, Resnik, Wang
  similarity_threshold: 0.7                   # Clustering cut height (0-1, higher = fewer clusters)
  go_obo_url: "https://current.geneontology.org/ontology/go-basic.obo"
  backend: "goatools"                         # "goatools" (pure Python) or "rrvgo" (requires R)

# --- Visualization ---
plot:
  top_n: 20                                   # Number of bars to show
  figure_width: 10                            # Figure width in inches
  figure_height: 8                            # Figure height in inches
  dpi: 300                                    # Resolution for rasterized elements
  bar_color: "#4C72B0"                        # Bar color (hex or matplotlib named color)
  color_by_recurrence: true                   # Color bars by number of contributing mutant lines
  colormap: "YlOrRd"                          # Matplotlib colormap (used if color_by_recurrence is true)
  show_recurrence_annotation: true            # Show n/N contributing lines as text on bars
  show_significance_line: true                # Vertical dashed line at -log10(0.05)
  significance_threshold: 0.05               # Threshold for the significance line
  xlabel: "$-\\log_{10}$(combined p-value)"
  ylabel: ""
  title: "Top Dysregulated Pathways (Fisher's Combined GSEA)"
  font_family: "Arial"
  font_size_labels: 10                        # Y-axis label font size
  font_size_title: 14
  font_size_axes: 12
  label_max_length: 60                        # Truncate GO term names longer than this (chars)

# --- Output ---
output:
  save_intermediate: true                     # Save Fisher p-value table and p-value matrix
  figure_basename: "gsea_meta_barplot"        # Base name for SVG/PDF outputs
  report_filename: "gsea_meta_analysis_report.md"
```

---

## 8. Error Handling and Logging

- **Logging**: Use Python's `logging` module. Default level: `INFO`. Configurable to `DEBUG` for troubleshooting.
- **Warnings** (non-fatal):
  - Mutant folder with no GSEA files found.
  - Rows with missing/invalid p-values skipped.
  - GO terms without valid GO IDs skipped.
  - GO OBO file download issues (retry once, then fail).
- **Errors** (fatal):
  - No valid mutant folders found in `data/`.
  - Ambiguous file matching (multiple pos or neg files in one folder).
  - Fewer than 2 mutant lines with valid data.
  - Invalid configuration file syntax.

---

## 9. Testing Checklist

Before release, verify the following:

- [ ] Tool correctly discovers files across multiple mutant folders.
- [ ] Duplicate GO terms (appearing in both pos and neg tables) are resolved by minimum p-value.
- [ ] P-values of 0.0 are replaced with the configured pseudocount.
- [ ] Imputed p = 1.0 entries contribute 0 to the Fisher statistic.
- [ ] Fisher's combined p-values are correct against a known reference (e.g., manual calculation for 2-3 terms).
- [ ] Clustering reduces redundancy visibly (compare with and without clustering).
- [ ] SVG and PDF outputs render correctly and are publication quality.
- [ ] The Markdown report contains accurate method descriptions with correct parameter values matching the actual run.
- [ ] The tool runs cleanly with default configuration (no `config.yaml` provided).
- [ ] The tool exits gracefully with informative errors for malformed inputs.

---

## 10. Future Extensions (Out of Scope)

The following are explicitly **not** included in this version but may be added later:

- Directional meta-analysis (combining NES with sign awareness).
- Heatmap or dot plot visualization showing per-line contributions.
- Interactive HTML output (e.g., Plotly).
- Support for non-GO gene set databases (KEGG, Reactome).
- Weighted Fisher's method (weighting lines by data quality or sample size).
