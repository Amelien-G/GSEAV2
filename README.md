# GSEA Proteomics Visualizer

A command-line tool for generating publication-quality cohort-level summary figures from GSEA preranked output, designed for Drosophila melanogaster ASD proteomics studies.

## Overview

This tool ingests per-mutant GSEA preranked output files, performs cohort-level meta-analysis, and produces three figures:

- **Figure 1** (optional): Hypothesis-driven dot plot showing GO terms from user-specified biological categories
- **Figure 2**: Unbiased dot plot showing the top data-driven GO terms grouped by hierarchical clustering
- **Figure 3**: Meta-analysis bar plot showing representative dysregulated pathways from Fisher's combined probability test

## Installation

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate gseav2
pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Usage

Run the tool from your project directory (the directory containing your `data/` folder):

```bash
# Produce Figure 2 and Figure 3 only (unbiased + meta-analysis)
gsea-tool

# Also produce Figure 1 using a category mapping TSV file
gsea-tool path/to/category_mapping.tsv
```

The tool expects a `data/` directory containing one subdirectory per mutant line. Each mutant subdirectory name must follow the format `<mutant_id>.GseaPreranked.<timestamp>` and must contain exactly one positive and one negative GSEA report TSV file.

Output files are written to `output/` in the current working directory.

## Input Data Format

Each mutant subfolder must contain:
- `gsea_report_for_na_pos_*.tsv` -- positive enrichment report
- `gsea_report_for_na_neg_*.tsv` -- negative enrichment report

These are standard GSEA preranked output files. The tool reads the NAME, NES, FDR q-val, NOM p-val, and SIZE columns.

## Output Files

The tool writes the following files to `output/`:

| File | Description |
|------|-------------|
| `figure1_cherry_picked.{pdf,png,svg}` | Hypothesis-driven dot plot (if produced) |
| `figure2_unbiased.{pdf,png,svg}` | Unbiased selection dot plot |
| `figure3_meta_analysis.{pdf,png,svg}` | Meta-analysis bar plot |
| `pvalue_matrix.tsv` | GO term x mutant nominal p-value matrix |
| `fisher_combined_pvalues.tsv` | Fisher combined p-values with cluster assignments |
| `notes.md` | Figure legends, methods text, and reproducibility notes |

## Configuration

See `config.yaml.example` for a fully documented example with all available options and their defaults.

## Requirements

All dependencies are listed in `environment.yml`. The main requirements are:

- Python >= 3.11
- matplotlib >= 3.7
- numpy >= 1.24
- scipy >= 1.10
- pyyaml >= 6.0
- pytest >= 7.0 (for running tests)

## Reference

Figure format modeled on: Gordon et al. 2024, Figure 3a.
