# GSEA Proteomics Visualizer

## 1. Problem and Purpose

Researchers studying Autism Spectrum Disorder (ASD) using Drosophila melanogaster fly models run Gene Set Enrichment Analysis (GSEA) on each mutant line independently. This produces per-mutant enrichment results across hundreds of Gene Ontology (GO) biological pathways. No existing tool assembles these per-mutant outputs into a single cohort-level summary figure suitable for manuscript submission.

This tool fills that gap. It reads GSEA output files from an arbitrary number of mutant subfolders, identifies which biological processes are consistently enriched or depleted across the cohort, and produces three publication-quality figures:

- **Figure 1** (optional): A hypothesis-driven dot plot grouping user-curated GO terms into biologically motivated categories. Categories can be specified either via a mapping TSV file or via `cherry_pick_categories` in `config.yaml` (which uses GO ontology ancestry resolution).
- **Figure 2** (always produced): An unbiased dot plot using data-driven GO term selection and unsupervised clustering, requiring no prior biological knowledge.
- **Figure 3** (always produced): A horizontal bar plot summarizing cohort-level pathway dysregulation using Fisher's combined probability method.

The dot plot format is modeled on Figure 3a of Gordon et al. 2024 to enable direct visual comparison between fly proteomics data and human transcriptomics findings. All figures are produced in PDF, PNG, and SVG formats, ready for journal submission without post-processing.

## 2. Intended Users and Needs

This tool is designed for proteomics researchers in academic biology laboratories working on Drosophila models of ASD. Users have strong expertise in fly genetics, proteomics workflows, and GSEA interpretation, but are not professional software developers.

The tool meets the following user needs:

- Visualize whether a mutant cohort converges on shared biological pathways at the protein level, in a format directly comparable to published literature.
- Aggregate evidence across all mutant lines using a principled statistical method (Fisher's combined probability test).
- Reduce redundancy among top pathways using GO semantic similarity clustering.
- Generate manuscript-ready figure legends and materials-and-methods text automatically.
- Operate through a clean command-line interface and optional YAML configuration file, with no need to modify source code.

## 3. Assumptions and Limitations

**Operating environment assumptions:**

- Python 3.11 or later is required.
- The Conda package manager is required for environment setup.
- The tool reads GSEA preranked output files; it does not perform GSEA itself.
- Input data must be organized in a `data/` subdirectory of the project directory. Each mutant has its own subfolder named `<mutant_id>.GseaPreranked.<timestamp>`.
- Each mutant subfolder must contain exactly one positive enrichment TSV (`gsea_report_for_na_pos_*.tsv`) and one negative enrichment TSV (`gsea_report_for_na_neg_*.tsv`).
- A minimum of 2 mutant lines with valid data is required.

**Known limitations:**

- The tool does not perform GSEA itself. It only consumes existing GSEA preranked output.
- The meta-analysis (Fisher's method) is not directional -- it combines nominal p-values regardless of the sign of enrichment.
- GO semantic similarity clustering (for Figure 3) requires downloading the GO ontology file and the Drosophila GAF annotation file from the internet on first use. A network connection is required unless cached files are already present in the `cache/` directory.
- The tool handles GO terms from all three GO namespaces (Biological Process, Cellular Component, Molecular Function) without filtering.
- Interactive HTML output is not supported.
- Non-GO gene set databases (KEGG, Reactome) are not supported.

## 4. Installation Instructions

### Prerequisites

Conda must be installed. Download Miniconda from https://docs.conda.io/en/latest/miniconda.html if not already installed.

### macOS

```bash
# 1. Clone or download the repository
git clone <repository-url>
cd GSEAV2

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate gseav2

# 4. Install the package
pip install -e .

# 5. Verify installation
gsea-tool --help
```

### Linux

```bash
# 1. Clone or download the repository
git clone <repository-url>
cd GSEAV2

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate gseav2

# 4. Install the package
pip install -e .

# 5. Verify installation
gsea-tool --help
```

### Windows

```
# Open Anaconda Prompt (not regular Command Prompt)

# 1. Clone or download the repository
git clone <repository-url>
cd GSEAV2

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate gseav2

# 4. Install the package
pip install -e .

# 5. Verify installation
gsea-tool --help
```

**Note for Windows users:** Use forward slashes or double backslashes in file paths when specifying the mapping file argument.

## 5. Usage

### Directory Setup

Before running, organize your data directory as follows:

```
your-project-directory/
    data/
        mutant_A.GseaPreranked.1234567890/
            gsea_report_for_na_pos_1234567890.tsv
            gsea_report_for_na_neg_1234567890.tsv
        mutant_B.GseaPreranked.1234567890/
            gsea_report_for_na_pos_1234567890.tsv
            gsea_report_for_na_neg_1234567890.tsv
    config.yaml           (optional, for custom settings including Figure 1 categories)
    category_mapping.tsv  (optional, legacy fallback for Figure 1)
```

### Running the Tool

**Produce Figures 2 and 3 only (default, no Figure 1):**

```bash
cd your-project-directory
conda activate gseav2
gsea-tool
```

**Produce all three figures including Figure 1 (recommended: via config.yaml):**

Add `cherry_pick` entries to `config.yaml` (see Section 6 below), then run:

```bash
gsea-tool
```

The tool uses the GO ontology to automatically resolve all descendant GO terms of each parent GO ID you specify, then intersects them with your GSEA results. This is the recommended approach.

**Produce Figure 1 via TSV mapping file (legacy fallback):**

```bash
gsea-tool category_mapping.tsv
```

This approach is retained for backward compatibility and for cases where automatic ontology resolution is not desired. See `category_mapping.tsv.example` for the expected format.

**Precedence rule:** If both `cherry_pick` entries exist in `config.yaml` and a mapping TSV file is provided as a CLI argument, the config-based ontology approach takes precedence and a warning is printed to stderr.

### Category Mapping File Format (TSV fallback for Figure 1)

The mapping file is a two-column TSV (tab-separated) with GO term name in the first column and category name in the second:

```
MITOCHONDRIAL RESPIRATORY CHAIN COMPLEX ASSEMBLY	Mitochondria
MITOCHONDRION ORGANIZATION	Mitochondria
CYTOPLASMIC TRANSLATION	Translation
TRANSLATION	Translation
G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY	GPCR
SYNAPSE ORGANIZATION	Synapse
SYNAPSE	Synapse
```

Lines starting with `#` are treated as comments. Term names are matched case-insensitively against the GSEA results. Category names are user-defined and determine the grouping boxes in the figure.

See `category_mapping.tsv.example` in the repository for a complete example.

### Output

All outputs are written to `output/` in the project directory:

- `figure1_cherry_picked.{pdf,png,svg}` -- hypothesis-driven dot plot (if cherry_pick config or mapping file provided)
- `figure2_unbiased.{pdf,png,svg}` -- data-driven dot plot
- `figure3_meta_analysis.{pdf,png,svg}` -- meta-analysis bar plot
- `pvalue_matrix.tsv` -- raw p-value matrix for all GO terms
- `fisher_combined_pvalues.tsv` -- Fisher combined p-values with cluster assignments
- `notes.md` -- figure legends and materials-and-methods text for manuscript

On success, the tool prints a summary to the terminal showing the number of mutants processed and the output file paths.

## 6. Configuration

Create a `config.yaml` file in your project directory to customize parameters. If no `config.yaml` is present, all defaults are used. See also `config.yaml.example` for a fully commented template.

### Cherry-pick categories (Figure 1)

The preferred way to produce Figure 1 is to define categories directly in `config.yaml`. Each entry specifies a parent GO term ID; the tool resolves all descendant terms from the GO ontology automatically:

```yaml
cherry_pick:
  - go_id: "GO:0005739"
    label: "Mitochondria"
  - go_id: "GO:0006412"
    label: "Translation"
  - go_id: "GO:0007186"
    label: "GPCR"
  - go_id: "GO:0045202"
    label: "Synapse"
```

Each entry has:
- `go_id`: A parent GO term ID. All descendants in the GO hierarchy are included.
- `label`: Display name for the category box in the figure.

The order of entries determines the order of category boxes. Categories with zero matching terms after intersection with GSEA results are silently omitted.

When `cherry_pick` is empty or absent and no mapping TSV file is provided, Figure 1 is not produced.

### Dot plot parameters (Figures 1 and 2)

```yaml
dot_plot:
  fdr_threshold: 0.05   # Significance threshold for dot presence
  top_n: 20             # Number of top GO terms in Figure 2
  n_groups: 4           # Number of unsupervised clusters for Figure 2
  random_seed: 42       # Random seed for reproducibility
```

### Fisher meta-analysis parameters (Figure 3)

```yaml
fisher:
  pseudocount: 1.0e-10     # Pseudocount to avoid log(0)
  apply_fdr: false          # Apply BH-FDR correction to combined p-values
  fdr_threshold: 0.25       # FDR threshold (only used when apply_fdr is true)
  prefilter_pvalue: 0.05    # Pre-filter threshold for clustering input
  top_n_bars: 20            # Number of bars in Figure 3
```

### GO semantic similarity clustering

```yaml
clustering:
  enabled: true               # Set to false to skip clustering
  similarity_metric: "Lin"    # Similarity metric
  similarity_threshold: 0.7   # Threshold for grouping similar terms
  go_obo_url: "https://current.geneontology.org/ontology/go-basic.obo"
  # gaf_url: ""               # Defaults to Drosophila GAF
```

### Plot appearance (all figures)

```yaml
plot:
  dpi: 300
  font_family: "Arial"
  bar_colormap: "YlOrRd"
  bar_figure_width: 10.0
  bar_figure_height: 8.0
  label_max_length: 60
  show_significance_line: true
  show_recurrence_annotation: true
```

The generated `notes.md` in the output directory also contains a full configuration guide with the values used for each run.

## 7. Input Data Format

Each mutant subfolder must contain:
- `gsea_report_for_na_pos_*.tsv` -- positive enrichment report
- `gsea_report_for_na_neg_*.tsv` -- negative enrichment report

These are standard GSEA preranked output files. The tool reads the NAME, NES, FDR q-val, NOM p-val, and SIZE columns.

## 8. Output Files

The tool writes the following files to `output/`:

| File | Description |
|------|-------------|
| `figure1_cherry_picked.{pdf,png,svg}` | Hypothesis-driven dot plot (if produced) |
| `figure2_unbiased.{pdf,png,svg}` | Unbiased selection dot plot |
| `figure3_meta_analysis.{pdf,png,svg}` | Meta-analysis bar plot |
| `pvalue_matrix.tsv` | GO term x mutant nominal p-value matrix |
| `fisher_combined_pvalues.tsv` | Fisher combined p-values with cluster assignments |
| `notes.md` | Figure legends, methods text, and reproducibility notes |

## 9. Requirements

All dependencies are listed in `environment.yml`. The main requirements are:

- Python >= 3.11
- matplotlib >= 3.7
- numpy >= 1.24
- scipy >= 1.10
- pyyaml >= 6.0
- pytest >= 7.0 (for running tests)

## 10. Tips

- **Adjusting the FDR threshold:** Lower the default FDR threshold (0.05) to show fewer, more significant dots; raise it to show more. Edit `config.yaml`: `dot_plot: {fdr_threshold: 0.1}`.
- **Changing the number of terms in Figure 2:** Edit `config.yaml`: `dot_plot: {top_n: 30}` to show 30 terms instead of 20.
- **Disabling GO clustering for Figure 3:** Set `clustering: {enabled: false}` in `config.yaml` to skip the semantic similarity step and use raw top-N terms directly.
- **Changing figure dimensions for Figure 3:** Edit `config.yaml`: `plot: {bar_figure_width: 12, bar_figure_height: 10}`.
- **Caching ontology files:** The GO OBO file and Drosophila GAF file are downloaded once and cached in `cache/`. Re-runs use the cached files. Delete the cache directory to force a fresh download.
- **Reproducibility:** The tool produces identical output for the same input and configuration. The random seed is fixed at 42 by default and recorded in `notes.md`.
- **Category mapping case:** GO term names in the mapping file are matched case-insensitively against the GSEA results. Upper or lower case both work.

## 11. Troubleshooting

**"data/ directory does not exist" error:**
You must invoke the tool from the directory that contains the `data/` folder, not from inside `data/` or from the parent directory. Run `ls` to confirm `data/` is present in the current directory.

**"Insufficient mutant lines" error:**
The tool requires at least 2 mutant lines with valid data. Check that your `data/` directory contains at least 2 subfolders named `<id>.GseaPreranked.*`.

**"Missing positive report file" or "Missing negative report file":**
Each mutant subfolder must contain exactly one file matching `gsea_report_for_na_pos_*.tsv` and one matching `gsea_report_for_na_neg_*.tsv`. Check the file names in the subfolder.

**"No GO terms pass the FDR threshold" (Figure 2 fails):**
The default FDR threshold is 0.05. If your data has few significant terms, try relaxing the threshold in `config.yaml`: `dot_plot: {fdr_threshold: 0.25}`.

**"No GO terms have combined p-value below the pre-filter threshold" (clustering fails):**
This means no GO terms reached Fisher-combined p < 0.05. Try disabling clustering: set `clustering: {enabled: false}` in `config.yaml`.

**Download failure for OBO or GAF files:**
The tool requires internet access for the first run. If behind a firewall, manually download the files:
- GO OBO: https://current.geneontology.org/ontology/go-basic.obo -> save to `cache/go-basic.obo`
- GAF: http://current.geneontology.org/annotations/fb.gaf.gz -> save to `cache/fb.gaf.gz`

**ImportError or ModuleNotFoundError:**
Ensure the conda environment is activated (`conda activate gseav2`) and the package is installed (`pip install -e .`). Run `conda env list` to confirm the environment exists.

**Wrong Python version:**
The tool requires Python 3.11. Run `python --version` to check. If incorrect, ensure the conda environment is activated.

**config.yaml type errors:**
All config values must be the correct type. Write `fdr_threshold: 0.05` (number), not `fdr_threshold: "0.05"` (string). Boolean values must be `true` or `false` (lowercase, no quotes).

## 12. Reference

Figure format modeled on: Gordon et al. 2024, Figure 3a.

## 13. License

This software is distributed under the MIT License.

See the LICENSE file for the full license text.

Copyright (c) 2026
