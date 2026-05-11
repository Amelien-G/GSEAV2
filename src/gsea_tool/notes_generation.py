"""Unit 9 -- Notes Generation."""

from pathlib import Path
from dataclasses import dataclass
import sys
import platform

from gsea_tool.data_ingestion import CohortData
from gsea_tool.configuration import ToolConfig
from gsea_tool.unbiased import UnbiasedSelectionStats
from gsea_tool.dot_plot import DotPlotResult
from gsea_tool.meta_analysis import FisherResult
from gsea_tool.go_clustering import ClusteringResult
from gsea_tool.bar_plot import BarPlotResult
from gsea_tool.go_tree import GoTreeResult


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
    fig1b_result: GoTreeResult | None = None  # GO tree paired with Figure 1
    fig2b_result: GoTreeResult | None = None  # GO tree paired with Figure 2


def get_dependency_versions() -> dict[str, str]:
    """Collect version strings for all key dependencies (Python, matplotlib, pandas, scipy, numpy, goatools, pyyaml)."""
    versions: dict[str, str] = {}

    versions["Python"] = platform.python_version()

    try:
        import matplotlib
        versions["matplotlib"] = matplotlib.__version__
    except (ImportError, AttributeError):
        versions["matplotlib"] = "unknown"

    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except (ImportError, AttributeError):
        versions["pandas"] = "unknown"

    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except (ImportError, AttributeError):
        versions["scipy"] = "unknown"

    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except (ImportError, AttributeError):
        versions["numpy"] = "unknown"

    try:
        import goatools
        versions["goatools"] = goatools.__version__
    except (ImportError, AttributeError):
        versions["goatools"] = "unknown"

    try:
        import yaml
        versions["PyYAML"] = yaml.__version__
    except (ImportError, AttributeError):
        versions["PyYAML"] = "unknown"

    return versions


def format_figure_legends(notes_input: NotesInput) -> str:
    """Generate the figure legend text section for all produced figures."""
    ni = notes_input
    cfg = ni.config
    fdr_thresh = cfg.dot_plot.fdr_threshold
    n_mutants = len(ni.cohort.mutant_ids)
    sections = []

    sections.append("## Figure Legends")
    sections.append("")

    # Figure 1 legend (only if produced)
    if ni.fig1_result is not None:
        fig1 = ni.fig1_result
        sections.append("### Figure 1: Cherry-Pick Dot Plot")
        sections.append("")
        sections.append(
            f"Figure 1 displays a grouped dot plot of selected GO terms across "
            f"{fig1.n_mutants} mutant lines. "
            f"Dot color represents the normalized enrichment score (NES) using a "
            f"diverging red-blue color scale, where red indicates positive enrichment "
            f"and blue indicates negative enrichment. "
            f"Dot size represents statistical significance as -log10(FDR). "
            f"Empty cells indicate that the term was not significant in that mutant "
            f"(FDR >= {fdr_thresh}). "
            f"Category boxes group related GO terms, with {fig1.n_categories} categories shown. "
            f"An FDR threshold of {fdr_thresh} was used to determine significance. "
            f"A total of {fig1.n_terms_displayed} GO terms are displayed."
        )
        sections.append("")

        if ni.fig1_method == "ontology":
            cherry_picks = cfg.cherry_pick_categories
            parent_descriptions = []
            for cp in cherry_picks:
                parent_descriptions.append(f"{cp.go_id} ({cp.label})")
            parents_str = ", ".join(parent_descriptions)
            sections.append(
                f"Categories were resolved via GO ontology ancestry from the following "
                f"parent GO IDs: {parents_str}."
            )
        elif ni.fig1_method == "tsv":
            sections.append(
                "Categories were resolved via a user-supplied category mapping file."
            )
        sections.append("")

    # Figure 1B legend (GO hierarchy tree paired with Figure 1)
    if ni.fig1b_result is not None:
        f1b = ni.fig1b_result
        sections.append("### Figure 1B: Cherry-Picked GO Term Hierarchy")
        sections.append("")
        sections.append(
            f"Figure 1B displays the minimal Gene Ontology hierarchy (Steiner tree) "
            f"connecting the {f1b.n_leaf_terms} cherry-picked GO terms shown in Figure 1 "
            f"to their namespace roots. Each plotted term is rendered as a leaf (bold "
            f"label); only essential branching-point ancestors are shown as internal nodes "
            f"(regular labels) -- {f1b.n_internal_nodes} essential ancestors are displayed "
            f"and {f1b.n_internal_nodes_pruned} non-branching intermediate ancestors were "
            f"collapsed for legibility. Solid edges denote direct `is_a` parent relationships; "
            f"dashed edges denote transitive `is_a+` paths where collapsed ancestors lie "
            f"between the connected nodes. Terms are split into {f1b.n_namespaces} "
            f"per-namespace file(s), one per Gene Ontology namespace (biological process / "
            f"molecular function / cellular component) so dense panels can use the canvas "
            f"width their widest row requires. No statistical encoding is applied to nodes; "
            f"the figure summarizes structural relationships only."
        )
        sections.append("")

    # Figure 2 legend
    fig2 = ni.fig2_result
    sections.append("### Figure 2: Unbiased Selection Dot Plot")
    sections.append("")
    sections.append(
        f"Figure 2 displays a grouped dot plot of the top unbiased-selected GO terms "
        f"across {fig2.n_mutants} mutant lines. "
        f"Dot color represents the normalized enrichment score (NES) using a "
        f"diverging red-blue color scale, where red indicates positive enrichment "
        f"and blue indicates negative enrichment. "
        f"Dot size represents statistical significance as -log10(FDR). "
        f"Empty cells indicate that the term was not significant in that mutant "
        f"(FDR >= {fdr_thresh}). "
        f"Category boxes group GO terms into {fig2.n_categories} clusters identified "
        f"by hierarchical clustering. "
        f"An FDR threshold of {fdr_thresh} was used to determine significance. "
        f"A total of {fig2.n_terms_displayed} GO terms are displayed."
    )
    sections.append("")

    # Figure 2B legend (GO hierarchy tree paired with Figure 2)
    if ni.fig2b_result is not None:
        f2b = ni.fig2b_result
        sections.append("### Figure 2B: Unbiased GO Term Hierarchy")
        sections.append("")
        sections.append(
            f"Figure 2B displays the minimal Gene Ontology hierarchy (Steiner tree) "
            f"connecting the {f2b.n_leaf_terms} unbiased-selected GO terms shown in Figure 2 "
            f"to their namespace roots. Each plotted term is rendered as a leaf (bold "
            f"label); only essential branching-point ancestors are shown as internal nodes "
            f"(regular labels) -- {f2b.n_internal_nodes} essential ancestors are displayed "
            f"and {f2b.n_internal_nodes_pruned} non-branching intermediate ancestors were "
            f"collapsed for legibility. Solid edges denote direct `is_a` parent relationships; "
            f"dashed edges denote transitive `is_a+` paths where collapsed ancestors lie "
            f"between the connected nodes. Terms are split into {f2b.n_namespaces} "
            f"per-namespace file(s), one per Gene Ontology namespace (biological process / "
            f"molecular function / cellular component) so dense panels can use the canvas "
            f"width their widest row requires. No statistical encoding is applied to nodes; "
            f"the figure summarizes structural relationships only."
        )
        sections.append("")

    # Figure 3 legend
    fig3 = ni.fig3_result
    sections.append("### Figure 3: Meta-Analysis Bar Plot")
    sections.append("")
    sections.append(
        f"Figure 3 displays the top {fig3.n_bars} representative dysregulated pathways "
        f"identified by meta-analysis across {fig3.n_mutants} mutant lines. "
        f"Bar length encodes -log10(combined p-value) from Fisher's combined probability test. "
        f"Bar color encodes the number of contributing mutant lines. "
        f"The statistical method used was Fisher's combined probability test, which "
        f"combines nominal p-values across all mutant lines for each GO term."
    )
    if fig3.clustering_was_used:
        cr = ni.clustering_result
        sections.append(
            f" GO semantic similarity clustering was applied "
            f"(using {cr.similarity_metric} similarity with a threshold of "
            f"{cr.similarity_threshold}) to reduce redundancy among significant terms, "
            f"and one representative term was selected per cluster (the term with the "
            f"lowest combined p-value)."
        )
    else:
        sections.append(
            " GO semantic similarity clustering was not applied; "
            "the top terms by combined p-value are shown directly."
        )
    sections.append("")

    return "\n".join(sections)


def format_methods_text(notes_input: NotesInput) -> str:
    """Generate the unified materials and methods text section."""
    ni = notes_input
    cfg = ni.config
    stats = ni.unbiased_stats
    versions = get_dependency_versions()
    sections = []

    sections.append("## Materials and Methods")
    sections.append("")

    # GSEA consumption
    sections.append(
        "Gene Set Enrichment Analysis (GSEA) preranked output was consumed as input "
        "to this analysis pipeline; GSEA itself was not executed as part of this tool. "
        "Positive and negative enrichment report files were parsed and merged for each "
        "mutant line, with conflicts resolved by retaining the entry with the smaller "
        "nominal p-value."
    )
    sections.append("")

    # Figure 1 methods
    if ni.fig1_result is not None:
        sections.append("### Figure 1: Cherry-Pick Selection")
        sections.append("")
        if ni.fig1_method == "ontology":
            cherry_picks = cfg.cherry_pick_categories
            parent_descriptions = []
            for cp in cherry_picks:
                parent_descriptions.append(f"{cp.go_id} ({cp.label})")
            parents_str = ", ".join(parent_descriptions)
            sections.append(
                f"For Figure 1, GO terms were selected using ontology-based resolution. "
                f"Parent GO IDs were specified in the configuration: {parents_str}. "
                f"Descendant terms present in the cohort data were identified by traversing "
                f"the GO ontology hierarchy and grouped under their respective parent categories. "
                f"An FDR threshold of {cfg.dot_plot.fdr_threshold} was applied."
            )
        elif ni.fig1_method == "tsv":
            sections.append(
                f"For Figure 1, GO terms were selected using a manual category mapping "
                f"provided via a user-supplied TSV file. Each row in the mapping file "
                f"specifies a GO term and its assigned category. "
                f"An FDR threshold of {cfg.dot_plot.fdr_threshold} was applied."
            )
        sections.append("")

    # Figure 2 methods
    sections.append("### Figure 2: Unbiased Selection and Clustering")
    sections.append("")
    sections.append(
        f"For Figure 2, GO terms passing the FDR threshold of {cfg.dot_plot.fdr_threshold} "
        f"in at least one mutant were pooled and ranked by maximum absolute NES across "
        f"all mutant lines. Redundant terms were removed using word-set Jaccard similarity "
        f"with a threshold of 0.5: for each pair of terms whose word sets had Jaccard "
        f"similarity greater than 0.5, only the higher-ranked term was retained. "
        f"The top {cfg.dot_plot.top_n} terms (or fewer if insufficient terms remained) "
        f"were selected. These terms were then clustered into {stats.n_clusters} groups "
        f"using hierarchical agglomerative clustering with Ward linkage "
        f"({stats.clustering_algorithm}) on the NES profile matrix (terms as rows, "
        f"mutants as columns, missing values treated as 0.0). "
        f"A random seed of {stats.random_seed} was used for reproducibility. "
        f"Each cluster was labeled with the term having the highest mean absolute NES "
        f"within that cluster."
    )
    sections.append("")

    # Figure 3 methods
    sections.append("### Figure 3: Meta-Analysis")
    sections.append("")
    sections.append(
        f"For Figure 3, Fisher's combined probability test was applied to merge "
        f"nominal p-values across all {len(ni.cohort.mutant_ids)} mutant lines. "
        f"Positive and negative enrichment tables were merged per mutant. "
        f"For GO terms absent in a given mutant, a p-value of 1.0 was imputed. "
        f"The Fisher statistic was computed as X^2 = -2 * sum(ln(p_i)), and the "
        f"combined p-value was obtained from a chi-squared distribution with "
        f"{2 * len(ni.cohort.mutant_ids)} degrees of freedom "
        f"(2k, where k = {len(ni.cohort.mutant_ids)} mutant lines)."
    )
    sections.append("")

    if ni.clustering_result is not None:
        cr = ni.clustering_result
        sections.append(
            f"GO semantic similarity clustering was applied to reduce redundancy among "
            f"significant terms. GO terms with combined p-value below the pre-filter "
            f"threshold of {cfg.fisher.prefilter_pvalue} were retained for clustering. "
            f"Pairwise semantic similarity was computed using the {cr.similarity_metric} "
            f"similarity metric, with information content derived from Gene Ontology "
            f"annotations (GAF file). Hierarchical agglomerative clustering with average "
            f"linkage was performed, and the dendrogram was cut at a similarity threshold "
            f"of {cr.similarity_threshold}. Within each cluster, the GO term with the "
            f"lowest combined p-value was selected as the representative. "
            f"The top {cfg.fisher.top_n_bars} representative terms were displayed."
        )
    else:
        sections.append(
            f"GO semantic similarity clustering was not applied. The top "
            f"{cfg.fisher.top_n_bars} GO terms ranked by combined p-value were "
            f"displayed directly without redundancy reduction."
        )
    sections.append("")

    # GO hierarchy figures (Figure 1B and Figure 2B) methods
    if ni.fig1b_result is not None or ni.fig2b_result is not None:
        sections.append("### Figures 1B and 2B: GO Term Hierarchy Trees")
        sections.append("")
        produced = []
        if ni.fig1b_result is not None:
            produced.append("Figure 1B (paired with Figure 1)")
        if ni.fig2b_result is not None:
            produced.append("Figure 2B (paired with Figure 2)")
        produced_str = " and ".join(produced)
        sections.append(
            f"For {produced_str}, the GO terms displayed in the corresponding "
            f"dot plot were resolved to their Gene Ontology IDs and combined with "
            f"the union of their is_a ancestors up to the namespace root, walked "
            f"recursively against the same OBO file used for ontology resolution "
            f"({cfg.clustering.go_obo_url}). The resulting node set was reduced to a "
            f"minimal Steiner tree on the plotted leaves and namespace roots (BUG-005), "
            f"keeping only branching-point ancestors that connect two or more disjoint "
            f"leaf clades; non-branching intermediate ancestors were collapsed for "
            f"legibility, and edges crossing collapsed nodes are drawn dashed to indicate "
            f"a transitive `is_a+` relationship (solid edges denote direct `is_a` parents). "
            f"The reduced graph was partitioned by GO namespace (biological_process / "
            f"molecular_function / cellular_component) and rendered as a top-down "
            f"hierarchical tree per namespace, written to one file per populated namespace "
            f"with its own canvas size. Depth was computed as the longest path to a root "
            f"in the reduced graph and within-row x-positions assigned by parent-mean "
            f"barycenter. Plotted terms appear as bold leaves; essential ancestor terms "
            f"appear as regular-weight internal nodes. Long labels are wrapped onto "
            f"multiple lines rather than truncated. Only is_a relationships are used; "
            f"part_of relationships are not represented."
        )
        sections.append("")

    # Software dependencies
    sections.append("### Software Dependencies")
    sections.append("")
    dep_lines = []
    for pkg, ver in versions.items():
        dep_lines.append(f"{pkg} {ver}")
    sections.append(
        "The analysis was performed using the following software: " +
        ", ".join(dep_lines) + "."
    )
    sections.append("")

    return "\n".join(sections)


def format_summary_statistics(notes_input: NotesInput) -> str:
    """Generate the summary statistics section."""
    ni = notes_input
    stats = ni.unbiased_stats
    sections = []

    sections.append("## Summary Statistics")
    sections.append("")

    n_mutants = len(ni.cohort.mutant_ids)
    total_go_terms = len(ni.cohort.all_term_names)

    sections.append(f"Number of mutants analyzed: {n_mutants}.")
    sections.append("")
    sections.append(f"Total unique GO terms in the input data: {total_go_terms}.")
    sections.append("")
    sections.append(
        f"Number of GO terms passing the FDR threshold "
        f"({ni.config.dot_plot.fdr_threshold}) in at least one mutant: "
        f"{stats.total_significant_terms}."
    )
    sections.append("")

    # Terms displayed per figure
    if ni.fig1_result is not None:
        sections.append(
            f"Number of GO terms displayed in Figure 1: {ni.fig1_result.n_terms_displayed}."
        )
        sections.append("")

    sections.append(
        f"Number of GO terms displayed in Figure 2: {ni.fig2_result.n_terms_displayed}."
    )
    sections.append("")
    sections.append(
        f"Number of GO terms displayed in Figure 3: {ni.fig3_result.n_bars}."
    )
    sections.append("")

    # Fisher pre-filter
    if ni.clustering_result is not None:
        sections.append(
            f"Number of GO terms passing the Fisher pre-filter "
            f"(combined p-value < {ni.config.fisher.prefilter_pvalue}): "
            f"{ni.clustering_result.n_prefiltered}."
        )
        sections.append("")
        sections.append(
            f"Number of semantic clusters formed: {ni.clustering_result.n_clusters}."
        )
        sections.append("")
    else:
        # Report Fisher pre-filter count even without clustering
        prefilter_count = sum(
            1 for p in ni.fisher_result.combined_pvalues.values()
            if p < ni.config.fisher.prefilter_pvalue
        )
        sections.append(
            f"Number of GO terms passing the Fisher pre-filter "
            f"(combined p-value < {ni.config.fisher.prefilter_pvalue}): "
            f"{prefilter_count}."
        )
        sections.append("")
        sections.append(
            "GO semantic similarity clustering was not applied."
        )
        sections.append("")

    # GO hierarchy summary stats (Figures 1B and 2B)
    if ni.fig1b_result is not None:
        f1b = ni.fig1b_result
        sections.append(
            f"Figure 1B GO-tree: {f1b.n_leaf_terms} leaf terms, "
            f"{f1b.n_internal_nodes} essential ancestor nodes shown "
            f"({f1b.n_internal_nodes_pruned} non-branching ancestors collapsed), "
            f"{f1b.n_namespaces} namespace panel(s)."
        )
        sections.append("")
    if ni.fig2b_result is not None:
        f2b = ni.fig2b_result
        sections.append(
            f"Figure 2B GO-tree: {f2b.n_leaf_terms} leaf terms, "
            f"{f2b.n_internal_nodes} essential ancestor nodes shown "
            f"({f2b.n_internal_nodes_pruned} non-branching ancestors collapsed), "
            f"{f2b.n_namespaces} namespace panel(s)."
        )
        sections.append("")

    return "\n".join(sections)


def format_reproducibility_note(notes_input: NotesInput) -> str:
    """Generate the reproducibility note with seeds and versions."""
    ni = notes_input
    cfg = ni.config
    stats = ni.unbiased_stats
    versions = get_dependency_versions()
    sections = []

    sections.append("## Reproducibility Note")
    sections.append("")

    sections.append(
        f"The random seed used for Figure 2 hierarchical clustering was "
        f"{stats.random_seed}."
    )
    sections.append("")

    # Software versions
    sections.append("Software versions used in this run:")
    sections.append("")
    for pkg, ver in versions.items():
        sections.append(f"- {pkg}: {ver}")
    sections.append("")

    # Configuration parameters
    sections.append("Configuration parameters used:")
    sections.append("")
    sections.append(f"- dot_plot.fdr_threshold: {cfg.dot_plot.fdr_threshold}")
    sections.append(f"- dot_plot.top_n: {cfg.dot_plot.top_n}")
    sections.append(f"- dot_plot.n_groups: {cfg.dot_plot.n_groups}")
    sections.append(f"- dot_plot.random_seed: {cfg.dot_plot.random_seed}")
    sections.append(f"- fisher.pseudocount: {cfg.fisher.pseudocount}")
    sections.append(f"- fisher.apply_fdr: {cfg.fisher.apply_fdr}")
    sections.append(f"- fisher.fdr_threshold: {cfg.fisher.fdr_threshold}")
    sections.append(f"- fisher.prefilter_pvalue: {cfg.fisher.prefilter_pvalue}")
    sections.append(f"- fisher.top_n_bars: {cfg.fisher.top_n_bars}")
    sections.append(f"- clustering.enabled: {cfg.clustering.enabled}")
    sections.append(f"- clustering.similarity_metric: {cfg.clustering.similarity_metric}")
    sections.append(f"- clustering.similarity_threshold: {cfg.clustering.similarity_threshold}")
    sections.append(f"- plot.dpi: {cfg.plot_appearance.dpi}")
    sections.append(f"- plot.font_family: {cfg.plot_appearance.font_family}")
    sections.append(f"- plot.bar_colormap: {cfg.plot_appearance.bar_colormap}")
    sections.append(f"- plot.bar_figure_width: {cfg.plot_appearance.bar_figure_width}")
    sections.append(f"- plot.bar_figure_height: {cfg.plot_appearance.bar_figure_height}")
    sections.append(f"- plot.label_max_length: {cfg.plot_appearance.label_max_length}")
    sections.append(f"- plot.show_significance_line: {cfg.plot_appearance.show_significance_line}")
    sections.append(f"- plot.show_recurrence_annotation: {cfg.plot_appearance.show_recurrence_annotation}")
    sections.append("")

    return "\n".join(sections)


def format_config_guide(notes_input: NotesInput) -> str:
    """Generate the configuration guide section describing all config.yaml parameters."""
    sections = []

    sections.append("## Configuration Guide")
    sections.append("")
    sections.append(
        "All parameters can be set in a config.yaml file placed in the project "
        "directory. If config.yaml is absent, all parameters use their default values. "
        "Below is a description of each parameter organized by section."
    )
    sections.append("")

    # cherry_pick section
    sections.append("### cherry_pick")
    sections.append("")
    sections.append(
        "A list of GO term categories for Figure 1 cherry-pick selection. "
        "Each entry requires a go_id (format: GO:NNNNNNN) and a label (human-readable "
        "category name). When provided, Figure 1 is generated using ontology-based "
        "resolution from these parent GO IDs. Default: empty list (Figure 1 is only "
        "produced if cherry-pick categories or a mapping file are provided)."
    )
    sections.append("")

    # dot_plot section
    sections.append("### dot_plot")
    sections.append("")
    sections.append(
        "- fdr_threshold: FDR significance threshold for dot plot rendering. "
        "Terms with FDR >= this value are shown as empty cells. Default: 0.05."
    )
    sections.append(
        "- top_n: Maximum number of GO terms to display in the unbiased selection "
        "(Figure 2). Default: 20."
    )
    sections.append(
        "- n_groups: Number of clusters for hierarchical grouping of terms in "
        "Figure 2. Default: 4."
    )
    sections.append(
        "- random_seed: Random seed for reproducibility of clustering in Figure 2. "
        "Default: 42."
    )
    sections.append("")

    # fisher section
    sections.append("### fisher")
    sections.append("")
    sections.append(
        "- pseudocount: Small value added to replace nominal p-values of exactly 0.0 "
        "to avoid log(0). Default: 1e-10."
    )
    sections.append(
        "- apply_fdr: Whether to apply Benjamini-Hochberg FDR correction to the "
        "combined p-values. Default: false."
    )
    sections.append(
        "- fdr_threshold: FDR threshold for filtering combined p-values when "
        "apply_fdr is true. Default: 0.25."
    )
    sections.append(
        "- prefilter_pvalue: Combined p-value threshold for pre-filtering GO terms "
        "before semantic clustering. Default: 0.05."
    )
    sections.append(
        "- top_n_bars: Maximum number of bars to display in the Figure 3 bar plot. "
        "Default: 20."
    )
    sections.append("")

    # clustering section
    sections.append("### clustering")
    sections.append("")
    sections.append(
        "- enabled: Whether to apply GO semantic similarity clustering for Figure 3 "
        "redundancy reduction. Default: true."
    )
    sections.append(
        "- similarity_metric: Semantic similarity metric to use. Default: Lin."
    )
    sections.append(
        "- similarity_threshold: Similarity threshold for cutting the clustering "
        "dendrogram. Higher values produce more clusters. Default: 0.7."
    )
    sections.append(
        "- go_obo_url: URL for the Gene Ontology OBO file. Default: "
        "https://current.geneontology.org/ontology/go-basic.obo."
    )
    sections.append(
        "- gaf_url: URL for the Gene Annotation File (GAF). Default: "
        "https://current.geneontology.org/annotations/fb.gaf.gz."
    )
    sections.append("")

    # plot section
    sections.append("### plot")
    sections.append("")
    sections.append(
        "- dpi: Resolution in dots per inch for PNG output. Default: 300."
    )
    sections.append(
        "- font_family: Font family for all plot text. Default: Arial."
    )
    sections.append(
        "- bar_colormap: Matplotlib colormap name for bar plot colors encoding "
        "number of contributing lines. Default: YlOrRd."
    )
    sections.append(
        "- bar_figure_width: Width of the Figure 3 bar plot in inches. Default: 10.0."
    )
    sections.append(
        "- bar_figure_height: Height of the Figure 3 bar plot in inches. Default: 8.0."
    )
    sections.append(
        "- label_max_length: Maximum character length for bar plot Y-axis labels "
        "before truncation. Default: 60."
    )
    sections.append(
        "- show_significance_line: Whether to display the p = 0.05 significance "
        "reference line on the bar plot. Default: true."
    )
    sections.append(
        "- show_recurrence_annotation: Whether to annotate each bar with the number "
        "of contributing mutant lines. Default: true."
    )
    sections.append("")

    return "\n".join(sections)


def generate_notes(
    notes_input: NotesInput,
    output_dir: Path,
) -> Path:
    """Generate notes.md and write it to output_dir.

    Returns the path to the written file.
    """
    assert output_dir.is_dir(), "Output directory must exist"

    sections = []

    sections.append("# Analysis Notes")
    sections.append("")

    sections.append(format_figure_legends(notes_input))
    sections.append(format_methods_text(notes_input))
    sections.append(format_summary_statistics(notes_input))
    sections.append(format_reproducibility_note(notes_input))
    sections.append(format_config_guide(notes_input))

    content = "\n".join(sections)

    notes_path = output_dir / "notes.md"
    try:
        notes_path.write_text(content, encoding="utf-8")
    except OSError as e:
        raise OSError(f"Cannot write notes.md to {output_dir}: {e}") from e

    assert notes_path.exists(), "notes.md must be written"
    assert notes_path.name == "notes.md", "Output filename must be notes.md"

    return notes_path
