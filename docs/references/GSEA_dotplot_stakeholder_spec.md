# Stakeholder Specification: GSEA Proteomics Dot Plot Visualizer

**Version:** 1.0  
**Status:** Final  
**Authors:** Stakeholder dialogue  

---

## 1. Purpose

This document specifies the requirements for a command-line tool that ingests GSEA proteomics output from multiple fly mutant experiments and produces two publication-quality dot plot figures. The visual target for both figures is **Figure 3a of Gordon et al. 2024** (bioRxiv doi: 10.1101/2024.04.01.587492). The figures must match that reference as closely as possible in terms of layout, visual encoding, and aesthetic quality.

---

## 2. Context

The experiment consists of a full shotgun proteomics analysis of a cohort of *Drosophila melanogaster* mutants, where each mutant carries a gene anticipated to be associated with Autism Spectrum Disorder (ASD). For each mutant, a GSEA preranked analysis has been run, producing enrichment scores for Gene Ontology terms in both the upregulated and downregulated directions. The goal is to visualize which biological processes are enriched or depleted across the mutant cohort in a compact, grouped, and publication-ready format.

---

## 3. Input Data

### 3.1 Folder Structure

The tool receives a single top-level folder path as its only required input. This folder contains one subfolder per mutant experiment. The tool must traverse exactly one level deep — no deeper.

Each level-1 subfolder follows the naming convention:

```
<mutant_id>.GseaPreranked.<timestamp>
```

The mutant identifier is the portion of the folder name that precedes the `.GseaPreranked` suffix. This identifier is used as the column label in the output figures.

Inside each subfolder, the tool must locate exactly two files:

- One file matching the pattern: `gsea_report_for_na_neg_*.tsv` — representing downregulated enrichment results
- One file matching the pattern: `gsea_report_for_na_pos_*.tsv` — representing upregulated enrichment results

All other files in the subfolder are ignored. Any subfolders nested deeper than level 1 are ignored.

### 3.2 File Format

Each TSV file is a standard GSEA preranked report containing at minimum the following columns:

- `NAME`: GO term identifier and label concatenated (e.g. `GO:0007602 PHOTOTRANSDUCTION`)
- `NES`: Normalized Enrichment Score (negative for downregulated file, positive for upregulated file)
- `FDR q-val`: False Discovery Rate adjusted p-value
- `SIZE`: Number of genes in the gene set

### 3.3 Error Handling

The tool must halt with a descriptive error message under the following conditions:

- A level-1 subfolder contains zero files matching the `neg` pattern
- A level-1 subfolder contains zero files matching the `pos` pattern
- A level-1 subfolder contains more than one file matching the `neg` pattern
- A level-1 subfolder contains more than one file matching the `pos` pattern

No other condition should cause the tool to halt. Non-conforming content within otherwise valid subfolders is silently ignored.

---

## 4. Output Figures

The tool produces two independent figures, described below. Both figures share the same visual language, derived from Figure 3a of Gordon et al. 2024.

### 4.1 Shared Visual Encoding (Both Figures)

- **X-axis:** One column per mutant, labeled with the mutant identifier extracted from the subfolder name. Column order should reflect the alphabetical order of mutant identifiers, or be configurable.
- **Y-axis:** GO terms, grouped into labeled categories. Each category is visually delimited by a surrounding box or bracket, with the category name displayed as a label to the right of the box — matching the layout of Figure 3a.
- **Dot color:** Encodes the NES value on a diverging colormap. Red indicates positive NES (upregulation); blue indicates negative NES (downregulation). The color scale is symmetric around zero. A colorbar legend is included.
- **Dot size:** Encodes statistical significance as -log10(FDR q-val). Larger dots indicate higher significance. A size legend is included.
- **Dot presence:** A dot is shown only if the GO term reached a significance threshold of FDR < 0.05 in that mutant. If a GO term is not significant in a given mutant, the cell is empty (no dot).
- **GO term labels:** Displayed on the Y-axis. The GO term name only is shown (not the GO ID). Labels should be legible and not overlap.
- **Category boxes:** Drawn as visible rectangles or brackets enclosing the GO terms belonging to each category, with the category name rendered in bold to the right of the box, vertically centered. This grouping structure is the primary organizational feature of the figure and must be faithfully reproduced.
- **Figure quality:** Output must be suitable for publication. Resolution of at least 300 DPI. Output format is PDF and PNG.

---

### 4.2 Figure 1 — Cherry-Picked Categories

This figure displays a curated set of GO terms organized into exactly four user-defined biological categories.

**Categories:**
1. Mitochondria
2. Translation
3. GPCR
4. Synapse

The mapping of specific GO terms to categories is provided by the user as a configuration input — a simple structured file (e.g. plain text or tabular) listing GO term names and their assigned category. The tool reads this mapping and uses it to filter and organize the Y-axis. GO terms present in the GSEA results but absent from the mapping are ignored. GO terms present in the mapping but absent from any GSEA result are also ignored silently.

The order of GO terms within each category box on the Y-axis should be determined by mean absolute NES across all mutants, descending — so the most consistently enriched terms appear at the top of each box.

---

### 4.3 Figure 2 — Unbiased Top Categories

This figure displays GO terms selected automatically from the data, with no biological pre-filtering or prior knowledge applied. The selection is driven purely by statistical criteria.

**Selection logic:**

1. Pool all GO terms from all mutants (both upregulated and downregulated files) that pass the FDR < 0.05 threshold in at least one mutant.
2. Rank all terms by their maximum absolute NES observed across the mutant cohort.
3. Remove redundant terms: if two GO term names share substantial lexical overlap, retain only the one with the higher maximum absolute NES.
4. From the deduplicated ranked list, select the top N terms, where N is a configurable parameter with a default of 20.
5. Automatically assign selected terms to 4 or 5 groups based on unsupervised clustering of their NES profiles across mutants. The number of groups is configurable with a default of 4.
6. Each group is labeled automatically with the name of its most representative term (highest mean absolute NES within the group).

This figure requires no user-provided mapping file.

---

## 5. Configuration

The tool must accept the following inputs:

| Input | Type | Required | Description |
|---|---|---|---|
| Top-level GSEA folder path | Argument | Yes | Path to the folder containing mutant subfolders |
| GO term category mapping file | Argument | Yes for Figure 1, not used for Figure 2 | File mapping GO term names to one of the four categories |
| FDR threshold | Parameter | No | Default: 0.05 |
| Number of top GO terms (Figure 2) | Parameter | No | Default: 20 |
| Number of auto-groups (Figure 2) | Parameter | No | Default: 4 |
| Output directory | Parameter | No | Default: current working directory |
| Figure format | Parameter | No | Default: both PDF and PNG |

---

## 6. Constraints and Non-Goals

- The tool does not perform GSEA itself. It only consumes existing GSEA output.
- The tool does not perform any ortholog mapping or cross-species translation.
- The tool does not filter GO terms based on any external gene list such as SFARI.
- The tool does not modify or reformat the input files in any way.
- The tool produces exactly two figures per run. It does not produce intermediate tables or supplementary outputs unless explicitly added in a future version.

---

## 7. Reference Figure

The visual target is **Figure 3a of Gordon et al. 2024**. Key features to replicate:

- Dot plot grid layout with mutants on X and GO terms on Y
- Category grouping boxes with bold labels to the right
- Diverging red-blue color scheme for NES
- Variable dot size encoding significance
- Clean, minimal axis styling consistent with a Nature-family journal figure
- GO terms sorted within categories by enrichment magnitude
- No gridlines within the plot area; only the category boxes provide visual structure

---

*End of specification.*
