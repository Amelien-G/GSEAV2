# Stakeholder Specification: GSEA Proteomics Visualizer

**Version:** 2.1
**Status:** Draft (pending approval)
**Authors:** Stakeholder dialogue

---

## 1. Purpose

This document specifies the requirements for a command-line tool that ingests GSEA proteomics output from multiple fly mutant experiments and produces up to three publication-quality figures:

1. **Figure 1** -- A hypothesis-driven dot plot with user-curated GO term categories (produced only when a category mapping file is provided).
2. **Figure 2** -- An unbiased dot plot with data-driven GO term selection and unsupervised clustering.
3. **Figure 3** -- A meta-analysis bar plot showing top dysregulated pathways identified by Fisher's combined probability method across the mutant cohort, with GO semantic similarity clustering to reduce redundancy.

The visual target for Figures 1 and 2 is **Figure 3a of Gordon et al. 2024** (bioRxiv doi: 10.1101/2024.04.01.587492). The figures must match that reference as closely as possible in terms of layout, visual encoding, and aesthetic quality. Figure 3 is a horizontal bar plot summarizing cohort-level pathway dysregulation.

All figures are produced in a single invocation of the tool.

---

## 2. Context

The experiment consists of a full shotgun proteomics analysis of a cohort of *Drosophila melanogaster* mutants, where each mutant carries a gene anticipated to be associated with Autism Spectrum Disorder (ASD). For each mutant, a GSEA preranked analysis has been run, producing enrichment scores for Gene Ontology terms in both the upregulated and downregulated directions. The goals are:

- To visualize which biological processes are enriched or depleted across the mutant cohort in a compact, grouped, and publication-ready format (Figures 1 and 2).
- To aggregate evidence across all mutant lines using Fisher's combined probability method and identify the pathways most consistently dysregulated at the cohort level (Figure 3).

---

## 3. Input Data

### 3.1 Folder Structure

The input data lives in a `data/` folder inside the project directory. The tool always reads from this fixed location -- the user does not pass an input path. Inside `data/`, there is one subfolder per mutant experiment. The tool must traverse exactly one level deep -- no deeper.

Each subfolder follows the naming convention:

```
<mutant_id>.GseaPreranked.<timestamp>
```

The mutant identifier is the portion of the folder name that precedes the `.GseaPreranked` suffix. This identifier is used as the column label in the dot plot figures and as the mutant line identifier in the meta-analysis.

Inside each subfolder, the tool must locate exactly **two** files -- both are required and both are consumed for every mutant:

- One file matching the pattern: `gsea_report_for_na_pos_*.tsv` -- the **upregulated** enrichment results (positive NES values)
- One file matching the pattern: `gsea_report_for_na_neg_*.tsv` -- the **downregulated** enrichment results (negative NES values)

The tool reads **both** files for each mutant and merges them into a single combined enrichment profile for that mutant. A GO term typically appears in either the pos file (upregulated) or the neg file (downregulated), but not both. If a GO term does appear in both files for the same mutant, the entry with the smaller nominal p-value is retained (see Section 6.2, Step 1 for details). Together, the two files provide the complete GSEA result for one mutant.

All other files in the subfolder are ignored. Any subfolders nested deeper than level 1 are ignored.

### 3.2 File Format

Each TSV file is a standard GSEA preranked report containing at minimum the following columns:

- `NAME`: GO term identifier and label concatenated (e.g. `GO:0007602 PHOTOTRANSDUCTION`). The GO ID is extracted as the first token matching the regex `GO:\d{7}`. The term name is everything following the GO ID, stripped of leading/trailing whitespace. Rows without a valid GO ID are skipped with a warning.
- `NES`: Normalized Enrichment Score (negative for downregulated file, positive for upregulated file)
- `FDR q-val`: False Discovery Rate adjusted p-value
- `NOM p-val`: Nominal p-value (used by the meta-analysis pipeline)
- `SIZE`: Number of genes in the gene set

### 3.3 Minimum Cohort Size

The tool requires a minimum of **2 mutant lines** with valid data. If fewer than 2 valid mutant subfolders are found in `data/`, the tool halts with a descriptive error message. No partial output is produced.

### 3.4 Error Handling

The tool must halt with a descriptive error message under the following conditions:

- A level-1 subfolder contains zero files matching the `neg` pattern
- A level-1 subfolder contains zero files matching the `pos` pattern
- A level-1 subfolder contains more than one file matching the `neg` pattern
- A level-1 subfolder contains more than one file matching the `pos` pattern
- Fewer than 2 valid mutant subfolders are found (see Section 3.3)
- Invalid configuration file syntax (if a `config.yaml` is provided)

No other condition should cause the tool to halt. Non-conforming content within otherwise valid subfolders (e.g., rows with missing or invalid values) is handled with warnings, not fatal errors.

---

## 4. Output

All outputs are written to an `output/` subdirectory relative to the project directory. The directory is created automatically if it does not exist.

### 4.1 Figures Produced

Depending on invocation:

| Figure | Produced when | Description |
|---|---|---|
| Figure 1 (hypothesis-driven dot plot) | Category mapping file is provided | Curated GO terms in user-defined categories |
| Figure 2 (unbiased dot plot) | Always | Data-driven GO term selection with unsupervised clustering |
| Figure 3 (meta-analysis bar plot) | Always | Fisher's combined probability bar plot with GO semantic clustering |

### 4.2 Output Formats

All figures are produced in three formats:

- **PDF** -- vector format for journal submission
- **PNG** -- raster format at 300 DPI for quick viewing and presentations
- **SVG** -- vector format for post-editing in tools like Inkscape or Illustrator

### 4.3 Intermediate Data Files

The following intermediate TSV files are always produced in the `output/` directory:

- `fisher_combined_pvalues.tsv` -- Full table of GO terms with combined Fisher p-values, number of contributing mutant lines, and cluster assignments (if clustering is enabled).
- `pvalue_matrix.tsv` -- The raw GO term by mutant line nominal p-value matrix before combination, with interleaved NES (Normalized Enrichment Score) columns for each mutant. Missing p-value entries are filled with the imputed value of 1.0; missing NES entries are left blank.

### 4.4 Notes File (notes.md)

A single `notes.md` file is written to the `output/` directory. This file contains all the information a researcher needs to write figure legends and the materials-and-methods section of a manuscript. It must include:

- **Figure legend text** for each produced figure:
  - Figures 1 and 2: a detailed description of the visual encoding (what dot color means, what dot size means, what empty cells mean, what the category boxes represent), the FDR threshold used, and the number of mutants included.
  - Figure 3: a description of what the bar plot shows (top N representative dysregulated pathways), what bar length encodes (-log10 combined p-value), what bar color encodes (number of contributing mutant lines), the statistical method (Fisher's combined probability test), and the GO semantic clustering step.
- **Unified materials and methods text**: a description of the full analysis pipeline suitable for a methods section, including:
  - That GSEA preranked output was consumed (not generated).
  - The GO term selection criteria for each figure.
  - The clustering method and parameters used for Figure 2 (algorithm name, number of clusters, random seed).
  - The redundancy removal method for Figure 2.
  - The meta-analysis strategy for Figure 3: merging of positive/negative tables, Fisher's method with imputation of p = 1.0 for absent terms, degrees of freedom.
  - The GO semantic similarity clustering approach for Figure 3 (Lin similarity, information content source, clustering threshold, representative selection rule).
  - Software dependencies with versions (Python, matplotlib, scipy, scikit-learn, pandas, goatools, etc.).
- **Summary statistics**: number of mutants analyzed, total GO terms in the input data, number of GO terms passing FDR threshold, number of GO terms displayed in each figure, number of GO terms passing the Fisher pre-filter, number of clusters formed.
- **Reproducibility note**: random seeds used for clustering, software versions, configuration parameters used.
- **Configuration guide**: a description of all `config.yaml` parameters with their defaults, explaining how to modify graphical elements and analysis parameters.

This file is a plain Markdown file. It is not a rendered figure -- it is text for the researcher to copy-paste into their manuscript draft.

---

## 5. Dot Plot Figures (Figures 1 and 2)

### 5.1 Shared Visual Encoding

Both dot plot figures share the same visual language, derived from Figure 3a of Gordon et al. 2024:

- **X-axis:** One column per mutant, labeled with the mutant identifier extracted from the subfolder name. Column order should reflect the alphabetical order of mutant identifiers, or be configurable.
- **Y-axis:** GO terms, grouped into labeled categories. Each category is visually delimited by a surrounding box or bracket, with the category name displayed as a label to the right of the box -- matching the layout of Figure 3a.
- **Dot color:** Encodes the NES value on a diverging colormap. Red indicates positive NES (upregulation); blue indicates negative NES (downregulation). The color scale is symmetric around zero. A colorbar legend is included.
- **Dot size:** Encodes statistical significance as -log10(FDR q-val). Larger dots indicate higher significance. A size legend is included.
- **Dot presence:** A dot is shown only if the GO term reached a significance threshold of FDR < 0.05 in that mutant. If a GO term is not significant in a given mutant, the cell is empty (no dot).
- **GO term labels:** Displayed on the Y-axis. The GO term name only is shown (not the GO ID). Labels should be legible and not overlap.
- **Category boxes:** Drawn as visible rectangles or brackets enclosing the GO terms belonging to each category, with the category name rendered in bold to the right of the box, vertically centered. This grouping structure is the primary organizational feature of the figure and must be faithfully reproduced.
- **No gridlines** within the plot area; only the category boxes provide visual structure.
- **Clean, minimal axis styling** consistent with a Nature-family journal figure.

### 5.2 Figure 1 -- Cherry-Picked Categories

This figure is produced when either (a) cherry-pick categories are configured in `config.yaml`, or (b) a category mapping TSV file is provided as a CLI argument. If both are present, the config-based approach takes precedence with a warning.

It displays a curated set of GO terms organized into user-defined biological categories. The number and names of categories are not fixed -- they are driven entirely by configuration.

**Config-based approach (preferred):** Categories are specified in `config.yaml` as a list of entries, each with a `go_id` (a parent GO term ID, e.g. `GO:0005739` for mitochondrion) and a `label` (display name, e.g. "Mitochondria"). The tool uses the GO OBO ontology to resolve all descendant GO terms of each parent GO ID, then intersects with the GO terms present in the GSEA results. The order of categories in the config file determines the order of category boxes in the figure. A GO term whose GO ID descends from multiple configured parent GO IDs appears in all matching categories.

**TSV fallback approach:** A category mapping file (two-column TSV: term name, category name) can be provided as a CLI argument, as in version 2.0. This is retained for backward compatibility and for cases where automatic ontology resolution is not desired.

In both approaches: categories with zero matching terms after intersection with GSEA results are silently omitted. Within each category, terms are sorted by mean absolute NES across all mutants, descending -- so the most consistently enriched terms appear at the top of each box.

### 5.3 Figure 2 -- Unbiased Top Categories

This figure is always produced. It displays GO terms selected automatically from the data, with no biological pre-filtering or prior knowledge applied. The selection is driven purely by statistical criteria.

**Selection logic:**

1. Pool all GO terms from all mutants (both upregulated and downregulated files) that pass the FDR < 0.05 threshold in at least one mutant.
2. Rank all terms by their maximum absolute NES observed across the mutant cohort.
3. Remove redundant terms: if two GO term names share substantial lexical overlap, retain only the one with the higher maximum absolute NES.
4. From the deduplicated ranked list, select the top N terms, where N is a configurable parameter with a default of 20.
5. Automatically assign selected terms to 4 or 5 groups based on unsupervised clustering of their NES profiles across mutants. The number of groups is configurable with a default of 4.
6. Each group is labeled automatically with the name of its most representative term (highest mean absolute NES within the group).

This figure requires no user-provided mapping file.

---

## 6. Meta-Analysis Bar Plot (Figure 3)

### 6.1 Purpose

Figure 3 aggregates evidence across all mutant lines using Fisher's combined probability method to identify GO terms that are consistently dysregulated at the cohort level. It complements the per-mutant dot plots by providing a single unified ranking of pathway dysregulation.

### 6.2 Processing Pipeline

#### Step 1: Per-Mutant Table Merging

For each mutant line:

1. Load the positive and negative enrichment tables (same files used for the dot plots).
2. Merge them into a single table keyed by GO ID.
3. **Conflict resolution**: If a GO term appears in both tables for the same mutant, retain the entry with the smaller nominal p-value (strongest evidence of dysregulation in either direction).
4. The output of this step is a dictionary: `{GO_ID: nominal_p_value}` per mutant line.

**P-value edge cases:**
- If `NOM p-val` is exactly 0.0, replace with a configurable pseudocount (default: 1e-10). This avoids negative infinity in the log transform.
- If `NOM p-val` is missing or non-numeric, skip the row with a warning.

#### Step 2: Cross-Mutant P-Value Matrix

1. Collect the union of all GO IDs observed across all mutant lines.
2. Build a matrix of shape (n_GO_terms x n_mutants).
3. Fill each cell with the nominal p-value from Step 1.
4. For missing entries (GO term not found in a mutant line), impute p = 1.0. Rationale: absence of enrichment is treated as absence of evidence; ln(1.0) = 0 contributes nothing to the Fisher statistic.

#### Step 3: Fisher's Combined Probability Test

For each GO term (row of the matrix):

1. Compute the Fisher statistic: X^2 = -2 * sum(ln(p_i)) where k is the total number of mutant lines (constant across all GO terms due to imputation).
2. Compute the combined p-value from a chi-squared distribution with 2k degrees of freedom.
3. Store the combined p-value for each GO term.

**Multiple testing correction** (optional, configurable): Apply Benjamini-Hochberg FDR correction on the combined p-values. Default: no correction (exploratory mode). The user can enable it and set a threshold via `config.yaml`.

#### Step 4: GO Semantic Similarity Clustering

1. Pre-filter: retain only GO terms with combined p-value below a configurable threshold (default: 0.05).
2. Compute pairwise semantic similarity between all pre-filtered GO terms using **Lin similarity** via the `goatools` library.
3. Information content is computed from the *Drosophila melanogaster* Gene Annotation File (GAF), which the tool auto-downloads from a standard source (GO Consortium or FlyBase). The download URL is configurable in `config.yaml`. HTTP requests must include a browser-like User-Agent header because the Gene Ontology server rejects Python's default user-agent.
4. The GO OBO file (ontology graph) is also auto-downloaded. The download URL is configurable in `config.yaml`. The same User-Agent requirement applies.
5. **Clustering method**: Hierarchical agglomerative clustering on the similarity matrix, cut at a configurable similarity threshold (default: 0.7).
6. **Representative selection**: Within each cluster, select the GO term with the lowest combined Fisher p-value as the representative.
7. Store the cluster assignments and representative terms.

**Fallback**: Clustering can be disabled via `config.yaml` (`clustering: enabled: false`). When disabled, the tool shows the top N GO terms by combined p-value without redundancy reduction, with a note in the figure legend that redundant terms may be present.

#### Step 5: Visualization

Generate a horizontal bar plot:

- **Y-axis**: Representative GO term names (formatted as readable labels, not GO IDs), ordered by combined Fisher p-value (most significant at top).
- **X-axis**: -log10(combined p-value).
- **Number of bars**: Top N representative terms (default: 20, configurable).
- **Bar color**: Mapped to the number of mutant lines contributing (i.e., how many lines had p < 1.0 for that term), using a sequential colormap. This adds a recurrence dimension.
- **Annotation**: Display the number of contributing mutant lines as text on or next to each bar.
- **Threshold line**: A vertical dashed line at -log10(0.05) to indicate nominal significance.
- **Clean, minimal styling** consistent with the dot plot figures and suitable for publication.

---

## 7. Configuration

### 7.1 Interface

The tool has no required arguments. The GO term category mapping file for Figure 1 is an optional CLI argument retained for backward compatibility. The preferred approach is to specify cherry-pick categories in `config.yaml`.

All tunable parameters are controlled through an optional `config.yaml` file. If a `config.yaml` file is present in the project directory, it is loaded automatically. If absent, all parameters use built-in defaults.

### 7.2 Fixed Conventions (Not Configurable)

- Input data is always read from the `data/` folder in the project directory.
- All outputs are written to the `output/` subdirectory in the project directory.
- Output formats are always PDF, PNG, and SVG for all figures.
- The `notes.md` and intermediate TSV files are always produced.

### 7.3 Configurable Parameters

The following parameters are configurable via `config.yaml`. All have sensible defaults.

**Cherry-pick categories (Figure 1):**

| Parameter | Default | Description |
|---|---|---|
| Cherry-pick categories list | (empty) | List of category entries, each with a parent GO term ID and display label. When non-empty, Figure 1 is produced using ontology-based term resolution |

Each entry in the list has:
- `go_id`: A parent GO term ID (e.g. `GO:0005739`). All descendants in the GO hierarchy are included.
- `label`: Display name for the category box (e.g. "Mitochondria").

**Dot plot parameters (Figures 1 and 2):**

| Parameter | Default | Description |
|---|---|---|
| FDR threshold | 0.05 | Significance threshold for dot presence |
| Number of top GO terms (Figure 2) | 20 | Number of terms to display in the unbiased dot plot |
| Number of auto-groups (Figure 2) | 4 | Number of unsupervised clusters for Figure 2 |

**Meta-analysis parameters (Figure 3):**

| Parameter | Default | Description |
|---|---|---|
| Zero p-value pseudocount | 1e-10 | Replacement for nominal p-values of exactly 0.0 |
| Apply FDR correction | false | Whether to apply Benjamini-Hochberg FDR on combined p-values |
| FDR threshold (meta-analysis) | 0.25 | FDR threshold (only used if FDR correction is enabled) |
| Pre-filter p-value | 0.05 | Only cluster GO terms with combined p below this |
| Top N bars | 20 | Number of bars to show in the bar plot |

**Clustering parameters (Figure 3):**

| Parameter | Default | Description |
|---|---|---|
| Clustering enabled | true | Set to false to skip GO semantic clustering |
| Similarity metric | Lin | Similarity metric for GO term comparison |
| Similarity threshold | 0.7 | Clustering cut height (0-1, higher = fewer clusters) |
| GO OBO URL | (GO Consortium default) | URL for auto-downloading the GO ontology file |
| GAF URL | (GO Consortium/FlyBase default) | URL for auto-downloading the Drosophila GAF |

**Plot appearance parameters (all figures):**

| Parameter | Default | Description |
|---|---|---|
| DPI | 300 | Resolution for PNG output |
| Font family | Arial | Font used in all figures |
| Bar plot colormap | YlOrRd | Colormap for bar color (recurrence encoding) |
| Bar plot figure width | 10 | Figure 3 width in inches |
| Bar plot figure height | 8 | Figure 3 height in inches |
| Label max length | 60 | Truncate GO term names longer than this (chars) |
| Show significance line | true | Vertical dashed line at -log10(0.05) in bar plot |
| Show recurrence annotation | true | Display contributing line count on bars |

---

## 8. Constraints and Non-Goals

- The tool does not perform GSEA itself. It only consumes existing GSEA output.
- The tool does not perform any ortholog mapping or cross-species translation.
- The tool does not filter GO terms based on any external gene list such as SFARI.
- The tool does not modify or reformat the input files in any way.
- The tool produces up to three figures, one notes file, and two intermediate TSV files per run.
- The meta-analysis is not directional -- it combines nominal p-values without regard to the sign of enrichment. Directional meta-analysis is out of scope for this version.
- Interactive HTML output (e.g., Plotly) is out of scope.
- Support for non-GO gene set databases (KEGG, Reactome) is out of scope.
- Weighted Fisher's method (weighting lines by data quality or sample size) is out of scope.

---

## 9. Reference Figure

The visual target for Figures 1 and 2 is **Figure 3a of Gordon et al. 2024**. Key features to replicate:

- Dot plot grid layout with mutants on X and GO terms on Y
- Category grouping boxes with bold labels to the right
- Diverging red-blue color scheme for NES
- Variable dot size encoding significance
- Clean, minimal axis styling consistent with a Nature-family journal figure
- GO terms sorted within categories by enrichment magnitude
- No gridlines within the plot area; only the category boxes provide visual structure

---

## 10. License

MIT License. The project is released under the MIT license, permitting unrestricted use, modification, and distribution with minimal restrictions.

---

*End of specification.*
