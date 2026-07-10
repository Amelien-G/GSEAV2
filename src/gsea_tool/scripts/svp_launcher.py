"""Unit 10 -- Orchestration: top-level entry point for the GSEA analysis tool."""

from pathlib import Path
import argparse
import sys


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Optional argument:
        mapping_file: Path to the GO term category mapping file.
                      If provided, Figure 1 is produced.
                      If omitted, only Figures 2 and 3 are produced.
    """
    parser = argparse.ArgumentParser(
        description="GSEA cohort analysis tool: dot plots and meta-analysis."
    )
    parser.add_argument(
        "mapping_file",
        nargs="?",
        default=None,
        help="Path to the GO term category mapping file (optional). "
             "If provided, Figure 1 (cherry-picked terms) is produced.",
    )
    return parser


def resolve_paths(project_dir: Path, mapping_file: str | None) -> tuple[Path, Path, Path, Path | None]:
    """Resolve and validate the data directory, output directory, cache directory, and optional mapping file path.

    Returns (data_dir, output_dir, cache_dir, mapping_path_or_none).
    data_dir is always <project_dir>/data/.
    output_dir is always <project_dir>/output/ (created if absent).
    cache_dir is always <project_dir>/cache/ (created if absent).

    Raises FileNotFoundError if data_dir does not exist.
    Raises FileNotFoundError if mapping_file is specified but does not exist.
    """
    data_dir = project_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = project_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    mapping_path: Path | None = None
    if mapping_file is not None:
        mapping_path = Path(mapping_file)
        if not mapping_path.is_file():
            raise FileNotFoundError(f"Mapping file does not exist: {mapping_path}")

    return data_dir, output_dir, cache_dir, mapping_path


def main() -> None:
    """Top-level entry point. Parses arguments and orchestrates all units.

    Exit code 0 on success, 1 on any error (with message printed to stderr).
    """
    try:
        _run()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run() -> None:
    """Internal orchestration logic, separated for clean error handling."""
    from gsea_tool.data_ingestion import ingest_data, CohortData
    from gsea_tool.configuration import load_config, ToolConfig, CherryPickCategory
    from gsea_tool.cherry_picked import (
        parse_category_mapping,
        select_cherry_picked_terms,
        resolve_categories_from_ontology,
        CategoryGroup,
    )
    from gsea_tool.unbiased import select_unbiased_terms
    from gsea_tool.dot_plot import render_dot_plot, DotPlotResult
    from gsea_tool.meta_analysis import run_fisher_analysis, FisherResult
    from gsea_tool.go_clustering import run_semantic_clustering, download_or_load_obo, ClusteringResult
    from gsea_tool.bar_plot import render_bar_plot, BarPlotResult
    from gsea_tool.notes_generation import generate_notes, NotesInput
    from gsea_tool.go_tree import render_go_tree, GoTreeResult

    # Parse CLI arguments
    parser = build_argument_parser()
    args = parser.parse_args()

    # Resolve project directory: __file__ is src/gsea_tool/scripts/svp_launcher.py
    # .parent x4 = scripts -> gsea_tool -> src -> project root
    project_dir = Path(__file__).resolve().parent.parent.parent.parent

    # Resolve paths
    data_dir, output_dir, cache_dir, mapping_path = resolve_paths(
        project_dir, args.mapping_file
    )

    # Step 1: Load configuration (Unit 2)
    config = load_config(project_dir)

    # Step 2: Ingest data (Unit 1)
    cohort = ingest_data(data_dir)

    # Determine Figure 1 approach: dual-path logic
    # Config cherry_pick_categories take precedence over mapping TSV file
    has_config_categories = len(config.cherry_pick_categories) > 0
    has_mapping_file = mapping_path is not None

    fig1_groups: list[CategoryGroup] | None = None
    fig1_method: str | None = None

    if has_config_categories and has_mapping_file:
        # Both present: config takes precedence, warn to stderr
        print(
            "Warning: Both cherry_pick_categories in config.yaml and a mapping file "
            "were provided. Using config-based ontology resolution (config takes precedence).",
            file=sys.stderr,
        )

    # Always resolve the OBO file: it is needed for the GO-hierarchy figures
    # (Figure 1B / Figure 2B), in addition to the optional ontology cherry-pick
    # path and the optional Unit 7 clustering path.
    obo_path = download_or_load_obo(
        config.clustering.go_obo_url,
        cache_dir,
        local_path=config.clustering.go_obo_path,
    )

    if has_config_categories:
        fig1_groups = resolve_categories_from_ontology(
            cohort, config.cherry_pick_categories, obo_path
        )
        fig1_method = "ontology"
    elif has_mapping_file:
        # TSV path: use parse_category_mapping + select_cherry_picked_terms
        assert mapping_path is not None
        term_to_category = parse_category_mapping(mapping_path)
        fig1_groups = select_cherry_picked_terms(cohort, term_to_category)
        fig1_method = "tsv"

    # Step 3: Render Figure 1 (Unit 5, conditional) and Figure 1B (Unit 11, conditional)
    fig1_result: DotPlotResult | None = None
    fig1b_result: GoTreeResult | None = None
    if fig1_groups is not None and len(fig1_groups) > 0:
        fig1_result = render_dot_plot(
            cohort=cohort,
            groups=fig1_groups,
            fdr_threshold=config.dot_plot.fdr_threshold,
            output_stem="figure1_cherry_picked",
            output_dir=output_dir,
            dpi=config.plot_appearance.dpi,
            font_family=config.plot_appearance.font_family,
            title="Figure 1: Cherry-Picked GO Terms",
        )
        fig1b_result = render_go_tree(
            groups=fig1_groups,
            cohort=cohort,
            obo_path=obo_path,
            output_stem="figure1B_cherry_picked_tree",
            output_dir=output_dir,
            dpi=config.plot_appearance.dpi,
            font_family=config.plot_appearance.font_family,
            title="Figure 1B: Cherry-Picked GO Term Hierarchy",
            label_max_chars=config.go_tree.label_max_chars,
            show_namespace_root=config.go_tree.show_namespace_root,
        )

    # Step 4: Select unbiased terms (Unit 4) for Figure 2
    unbiased_groups, unbiased_stats = select_unbiased_terms(
        cohort,
        fdr_threshold=config.dot_plot.fdr_threshold,
        top_n=config.dot_plot.top_n,
        n_groups=config.dot_plot.n_groups,
        random_seed=config.dot_plot.random_seed,
    )

    # Step 5: Render Figure 2 (Unit 5) and Figure 2B (Unit 11)
    fig2_result = render_dot_plot(
        cohort=cohort,
        groups=unbiased_groups,
        fdr_threshold=config.dot_plot.fdr_threshold,
        output_stem="figure2_unbiased",
        output_dir=output_dir,
        dpi=config.plot_appearance.dpi,
        font_family=config.plot_appearance.font_family,
        title="Figure 2: Unbiased GO Term Selection",
    )
    fig2b_result = render_go_tree(
        groups=unbiased_groups,
        cohort=cohort,
        obo_path=obo_path,
        output_stem="figure2B_unbiased_tree",
        output_dir=output_dir,
        dpi=config.plot_appearance.dpi,
        font_family=config.plot_appearance.font_family,
        title="Figure 2B: Unbiased Top GO Term Hierarchy",
        label_max_chars=config.go_tree.label_max_chars,
        show_namespace_root=config.go_tree.show_namespace_root,
    )

    # Step 6: Run Fisher analysis (Unit 6)
    fisher_result = run_fisher_analysis(
        cohort=cohort,
        config=config.fisher,
        output_dir=output_dir,
        clustering_enabled=config.clustering.enabled,
    )

    # Step 7: Optionally run GO clustering (Unit 7)
    clustering_result: ClusteringResult | None = None
    if config.clustering.enabled:
        clustering_result = run_semantic_clustering(
            fisher_result=fisher_result,
            config=config.clustering,
            output_dir=output_dir,
            cache_dir=cache_dir,
        )

    # Step 8: Render Figure 3 (Unit 8)
    fig3_result = render_bar_plot(
        fisher_result=fisher_result,
        clustering_result=clustering_result,
        fisher_config=config.fisher,
        plot_config=config.plot_appearance,
        output_dir=output_dir,
        output_stem="figure3_meta_analysis",
    )

    # Step 9: Generate notes.md (Unit 9)
    notes_input = NotesInput(
        cohort=cohort,
        config=config,
        fig1_result=fig1_result,
        fig1_method=fig1_method if fig1_result is not None else None,
        fig2_result=fig2_result,
        fig3_result=fig3_result,
        unbiased_stats=unbiased_stats,
        fisher_result=fisher_result,
        clustering_result=clustering_result,
        fig1b_result=fig1b_result,
        fig2b_result=fig2b_result,
    )
    notes_path = generate_notes(notes_input, output_dir)

    # Print success summary to stdout
    n_mutants = len(cohort.mutant_ids)
    figures_produced = ["Figure 2", "Figure 2B", "Figure 3"]
    if fig1_result is not None:
        figures_produced.insert(0, "Figure 1B")
        figures_produced.insert(0, "Figure 1")

    print(f"Analysis complete: {n_mutants} mutants processed.")
    print(f"Figures produced: {', '.join(figures_produced)}.")
    print(f"Output directory: {output_dir}")
    print(f"Output files:")
    if fig1_result is not None:
        print(f"  - {fig1_result.pdf_path}")
        print(f"  - {fig1_result.png_path}")
        print(f"  - {fig1_result.svg_path}")
    if fig1b_result is not None:
        for p in fig1b_result.pdf_paths.values():
            print(f"  - {p}")
        for p in fig1b_result.png_paths.values():
            print(f"  - {p}")
        for p in fig1b_result.svg_paths.values():
            print(f"  - {p}")
    print(f"  - {fig2_result.pdf_path}")
    print(f"  - {fig2_result.png_path}")
    print(f"  - {fig2_result.svg_path}")
    for p in fig2b_result.pdf_paths.values():
        print(f"  - {p}")
    for p in fig2b_result.png_paths.values():
        print(f"  - {p}")
    for p in fig2b_result.svg_paths.values():
        print(f"  - {p}")
    print(f"  - {fig3_result.pdf_path}")
    print(f"  - {fig3_result.png_path}")
    print(f"  - {fig3_result.svg_path}")
    print(f"  - {output_dir / 'pvalue_matrix.tsv'}")
    print(f"  - {output_dir / 'fisher_combined_pvalues.tsv'}")
    print(f"  - {notes_path}")


if __name__ == "__main__":
    main()
