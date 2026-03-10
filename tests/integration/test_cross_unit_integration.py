"""
Integration test suite for the GSEA Proteomics Dot Plot Visualizer.

These tests verify cross-unit interactions that no single unit owns.
They exercise the full data pipeline from ingestion through rendering
and notes generation, checking that units compose correctly.

REGRESSION TEST INCLUSION:
    When running integration tests, the test command MUST include both
    tests/integration/ and tests/regressions/ directories:
        python -m pytest tests/integration/ tests/regressions/ -v

DATA ASSUMPTIONS:
    All synthetic test data is constructed to mimic the real GSEA preranked
    output format as documented in the stakeholder spec and the example data
    in data/example_input/. GO term names are uppercase. TSV files include
    the HTML artifact header and trailing tab characters. Mutant folder names
    follow the <mutant_id>.GseaPreranked.<timestamp> convention.
"""

import math
import os
import textwrap
from pathlib import Path

import numpy as np
import pytest

# --- Unit 1: Data Ingestion ---
from gsea_tool.data_ingestion import (
    CohortData,
    DataIngestionError,
    MutantProfile,
    TermRecord,
    discover_mutant_folders,
    ingest_data,
    locate_report_files,
    merge_pos_neg,
    parse_gsea_report,
)

# --- Unit 2: Configuration ---
from gsea_tool.configuration import (
    ClusteringConfig,
    ConfigError,
    DotPlotConfig,
    FisherConfig,
    PlotAppearanceConfig,
    ToolConfig,
    load_config,
    validate_config,
)

# --- Unit 3: Cherry-Picked Term Selection ---
from gsea_tool.cherry_picked import (
    CategoryGroup,
    MappingFileError,
    parse_category_mapping,
    select_cherry_picked_terms,
)

# --- Unit 4: Unbiased Term Selection ---
from gsea_tool.unbiased import (
    UnbiasedSelectionStats,
    cluster_terms,
    pool_significant_terms,
    remove_redundant_terms,
    select_top_n,
    select_unbiased_terms,
)

# --- Unit 5: Dot Plot Rendering ---
from gsea_tool.dot_plot import (
    DotPlotResult,
    build_dot_grid,
    draw_category_boxes,
    render_dot_plot,
)

# --- Unit 6: Meta-Analysis Computation ---
from gsea_tool.meta_analysis import (
    FisherResult,
    build_pvalue_dict_per_mutant,
    build_pvalue_matrix,
    compute_fisher_combined,
    run_fisher_analysis,
    write_fisher_results_tsv,
    write_pvalue_matrix_tsv,
)

# --- Unit 7: GO Semantic Clustering ---
from gsea_tool.go_clustering import (
    ClusteringResult,
    cluster_by_similarity,
    select_representatives,
    write_fisher_results_with_clusters_tsv,
)

# --- Unit 8: Bar Plot Rendering ---
from gsea_tool.bar_plot import (
    BarPlotResult,
    render_bar_plot,
    select_bar_data,
)

# --- Unit 9: Notes Generation ---
from gsea_tool.notes_generation import (
    NotesInput,
    format_config_guide,
    format_figure_legends,
    format_methods_text,
    format_reproducibility_note,
    format_summary_statistics,
    generate_notes,
    get_dependency_versions,
)

# --- Unit 10: Orchestration ---
from gsea_tool.scripts.svp_launcher import (
    build_argument_parser,
    resolve_paths,
)


# ============================================================================
# Shared test data helpers
# ============================================================================

# TSV header matching real GSEA preranked output format, including the HTML
# artifact in the second column and a trailing tab.
_TSV_HEADER = (
    "NAME\tGS<br> follow link to MSigDB\tGS DETAILS\tSIZE\tES\tNES\t"
    "NOM p-val\tFDR q-val\tFWER p-val\tRANK AT MAX\tLEADING EDGE\t"
)


def _make_tsv_row(
    go_id: str,
    term_name: str,
    size: int,
    es: float,
    nes: float,
    nom_pval: float,
    fdr: float,
    fwer: float = 0.0,
    rank: int = 500,
    leading_edge: str = "tags=50%, list=25%, signal=60%",
) -> str:
    """Build one TSV data row in the GSEA report format with trailing tab."""
    name_field = f"{go_id} {term_name}"
    gs_detail = f"{go_id} {term_name}"
    return (
        f"{name_field}\t{gs_detail}\t\t{size}\t{es}\t{nes}\t"
        f"{nom_pval}\t{fdr}\t{fwer}\t{rank}\t{leading_edge}\t"
    )


def _write_tsv(path: Path, rows: list[str]) -> None:
    """Write a complete TSV file with header and data rows."""
    lines = [_TSV_HEADER] + rows
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_mutant_folder(
    data_dir: Path,
    mutant_id: str,
    pos_rows: list[str],
    neg_rows: list[str],
    timestamp: str = "1681387666846",
) -> Path:
    """Create a single mutant folder with pos and neg TSV files."""
    folder_name = f"{mutant_id}.GseaPreranked.{timestamp}"
    folder = data_dir / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    _write_tsv(folder / f"gsea_report_for_na_pos_{timestamp}.tsv", pos_rows)
    _write_tsv(folder / f"gsea_report_for_na_neg_{timestamp}.tsv", neg_rows)
    return folder


def _build_synthetic_cohort_on_disk(data_dir: Path) -> None:
    """Create a synthetic 3-mutant cohort on disk with biologically plausible data.

    Mutants: alpha, beta, gamma
    GO terms span all four cherry-pick categories plus extra terms for unbiased selection.
    NES and FDR values are chosen to exercise the FDR threshold logic and to produce
    non-trivial Fisher combined p-values.
    """
    # ---- Mutant alpha ----
    alpha_pos = [
        _make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.45, 1.85, 0.001, 0.02),
        _make_tsv_row("GO:0006412", "TRANSLATION", 80, 0.52, 2.10, 0.0, 0.01),
        _make_tsv_row("GO:0007186", "G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY", 60, 0.38, 1.55, 0.005, 0.04),
        _make_tsv_row("GO:0045202", "SYNAPSE", 50, 0.42, 1.70, 0.002, 0.03),
        _make_tsv_row("GO:0006338", "CHROMATIN REMODELING", 45, 0.54, 1.96, 0.0, 0.04),
    ]
    alpha_neg = [
        _make_tsv_row("GO:0005777", "PEROXISOME", 52, -0.52, -2.22, 0.0, 0.003),
        _make_tsv_row("GO:0022626", "CYTOSOLIC RIBOSOME", 73, -0.47, -2.04, 0.0, 0.012),
        _make_tsv_row("GO:0005576", "EXTRACELLULAR REGION", 96, -0.43, -1.95, 0.0, 0.019),
    ]
    _create_mutant_folder(data_dir, "alpha", alpha_pos, alpha_neg, "1000000000001")

    # ---- Mutant beta ----
    beta_pos = [
        _make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.40, 1.60, 0.003, 0.035),
        _make_tsv_row("GO:0006412", "TRANSLATION", 80, 0.48, 1.90, 0.001, 0.025),
        _make_tsv_row("GO:0045202", "SYNAPSE", 50, 0.50, 2.00, 0.0, 0.015),
        _make_tsv_row("GO:0006338", "CHROMATIN REMODELING", 45, 0.35, 1.40, 0.01, 0.06),
    ]
    beta_neg = [
        _make_tsv_row("GO:0005777", "PEROXISOME", 52, -0.45, -1.90, 0.002, 0.015),
        _make_tsv_row("GO:0022626", "CYTOSOLIC RIBOSOME", 73, -0.50, -2.10, 0.0, 0.008),
        _make_tsv_row("GO:0007186", "G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY", 60, -0.30, -1.20, 0.02, 0.10),
    ]
    _create_mutant_folder(data_dir, "beta", beta_pos, beta_neg, "1000000000002")

    # ---- Mutant gamma ----
    gamma_pos = [
        _make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.55, 2.20, 0.0, 0.008),
        _make_tsv_row("GO:0006412", "TRANSLATION", 80, 0.60, 2.40, 0.0, 0.005),
        _make_tsv_row("GO:0007186", "G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY", 60, 0.45, 1.80, 0.001, 0.02),
        _make_tsv_row("GO:0045202", "SYNAPSE", 50, 0.38, 1.50, 0.005, 0.045),
        _make_tsv_row("GO:0006338", "CHROMATIN REMODELING", 45, 0.58, 2.30, 0.0, 0.01),
        _make_tsv_row("GO:0003682", "CHROMATIN BINDING", 80, 0.51, 2.06, 0.0, 0.015),
    ]
    gamma_neg = [
        _make_tsv_row("GO:0005777", "PEROXISOME", 52, -0.60, -2.50, 0.0, 0.001),
        _make_tsv_row("GO:0022626", "CYTOSOLIC RIBOSOME", 73, -0.55, -2.30, 0.0, 0.004),
        _make_tsv_row("GO:0005576", "EXTRACELLULAR REGION", 96, -0.48, -2.00, 0.0, 0.010),
    ]
    _create_mutant_folder(data_dir, "gamma", gamma_pos, gamma_neg, "1000000000003")


def _build_cohort_programmatic() -> CohortData:
    """Build a CohortData object programmatically (without touching disk).

    Used for tests that need a CohortData but do not need to exercise ingestion.
    Contains the same conceptual data as _build_synthetic_cohort_on_disk.
    """
    mutant_ids = ["alpha", "beta", "gamma"]
    profiles = {}
    all_term_names = set()
    all_go_ids = set()

    # alpha profile
    alpha_records = {
        "MITOCHONDRION": TermRecord("MITOCHONDRION", "GO:0005739", 1.85, 0.02, 0.001, 100),
        "TRANSLATION": TermRecord("TRANSLATION", "GO:0006412", 2.10, 0.01, 0.0, 80),
        "G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY": TermRecord(
            "G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY", "GO:0007186", 1.55, 0.04, 0.005, 60
        ),
        "SYNAPSE": TermRecord("SYNAPSE", "GO:0045202", 1.70, 0.03, 0.002, 50),
        "CHROMATIN REMODELING": TermRecord("CHROMATIN REMODELING", "GO:0006338", 1.96, 0.04, 0.0, 45),
        "PEROXISOME": TermRecord("PEROXISOME", "GO:0005777", -2.22, 0.003, 0.0, 52),
        "CYTOSOLIC RIBOSOME": TermRecord("CYTOSOLIC RIBOSOME", "GO:0022626", -2.04, 0.012, 0.0, 73),
        "EXTRACELLULAR REGION": TermRecord("EXTRACELLULAR REGION", "GO:0005576", -1.95, 0.019, 0.0, 96),
    }
    profiles["alpha"] = MutantProfile("alpha", alpha_records)

    # beta profile
    beta_records = {
        "MITOCHONDRION": TermRecord("MITOCHONDRION", "GO:0005739", 1.60, 0.035, 0.003, 100),
        "TRANSLATION": TermRecord("TRANSLATION", "GO:0006412", 1.90, 0.025, 0.001, 80),
        "SYNAPSE": TermRecord("SYNAPSE", "GO:0045202", 2.00, 0.015, 0.0, 50),
        "CHROMATIN REMODELING": TermRecord("CHROMATIN REMODELING", "GO:0006338", 1.40, 0.06, 0.01, 45),
        "PEROXISOME": TermRecord("PEROXISOME", "GO:0005777", -1.90, 0.015, 0.002, 52),
        "CYTOSOLIC RIBOSOME": TermRecord("CYTOSOLIC RIBOSOME", "GO:0022626", -2.10, 0.008, 0.0, 73),
        "G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY": TermRecord(
            "G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY", "GO:0007186", -1.20, 0.10, 0.02, 60
        ),
    }
    profiles["beta"] = MutantProfile("beta", beta_records)

    # gamma profile
    gamma_records = {
        "MITOCHONDRION": TermRecord("MITOCHONDRION", "GO:0005739", 2.20, 0.008, 0.0, 100),
        "TRANSLATION": TermRecord("TRANSLATION", "GO:0006412", 2.40, 0.005, 0.0, 80),
        "G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY": TermRecord(
            "G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY", "GO:0007186", 1.80, 0.02, 0.001, 60
        ),
        "SYNAPSE": TermRecord("SYNAPSE", "GO:0045202", 1.50, 0.045, 0.005, 50),
        "CHROMATIN REMODELING": TermRecord("CHROMATIN REMODELING", "GO:0006338", 2.30, 0.01, 0.0, 45),
        "CHROMATIN BINDING": TermRecord("CHROMATIN BINDING", "GO:0003682", 2.06, 0.015, 0.0, 80),
        "PEROXISOME": TermRecord("PEROXISOME", "GO:0005777", -2.50, 0.001, 0.0, 52),
        "CYTOSOLIC RIBOSOME": TermRecord("CYTOSOLIC RIBOSOME", "GO:0022626", -2.30, 0.004, 0.0, 73),
        "EXTRACELLULAR REGION": TermRecord("EXTRACELLULAR REGION", "GO:0005576", -2.00, 0.010, 0.0, 96),
    }
    profiles["gamma"] = MutantProfile("gamma", gamma_records)

    for profile in profiles.values():
        for rec in profile.records.values():
            all_term_names.add(rec.term_name)
            all_go_ids.add(rec.go_id)

    return CohortData(
        mutant_ids=mutant_ids,
        profiles=profiles,
        all_term_names=all_term_names,
        all_go_ids=all_go_ids,
    )


def _create_mapping_file(path: Path) -> None:
    """Write a category mapping file for the cherry-picked terms."""
    content = textwrap.dedent("""\
        MITOCHONDRION\tMitochondria
        TRANSLATION\tTranslation
        G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY\tGPCR
        SYNAPSE\tSynapse
    """)
    path.write_text(content, encoding="utf-8")


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory with data/ and output/ subdirectories."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return tmp_path


@pytest.fixture
def cohort_on_disk(tmp_project):
    """Create a synthetic cohort on disk and return (data_dir, output_dir)."""
    data_dir = tmp_project / "data"
    _build_synthetic_cohort_on_disk(data_dir)
    return data_dir, tmp_project / "output"


@pytest.fixture
def cohort():
    """Build a CohortData object programmatically (no disk I/O)."""
    return _build_cohort_programmatic()


@pytest.fixture
def default_config():
    """Return a ToolConfig with all default values."""
    return ToolConfig()


# ============================================================================
# 1. Data flow: Ingestion -> Cherry-Picked Selection (Units 1 + 3)
# ============================================================================


class TestIngestionToCheryPickedSelection:
    """Verify that CohortData produced by Unit 1 flows correctly into Unit 3's
    cherry-picked term selection logic."""

    def test_ingested_data_feeds_cherry_picked_selection(self, cohort_on_disk):
        """Data ingested from disk produces CohortData that Unit 3 can use
        to select and group cherry-picked terms correctly."""
        data_dir, _ = cohort_on_disk
        cohort = ingest_data(data_dir)

        # Write a mapping file
        mapping_path = data_dir.parent / "mapping.tsv"
        _create_mapping_file(mapping_path)
        term_to_category = parse_category_mapping(mapping_path)

        groups = select_cherry_picked_terms(cohort, term_to_category)

        # The cohort has terms in all 4 categories
        assert len(groups) == 4
        category_names = [g.category_name for g in groups]
        assert category_names == ["Mitochondria", "Translation", "GPCR", "Synapse"]

        # Each group must have exactly 1 term (since we mapped one term per category)
        for group in groups:
            assert len(group.term_names) == 1

    def test_ingestion_term_names_are_uppercase_for_mapping_match(self, cohort_on_disk):
        """Unit 1 uppercases term names, which is required for Unit 3 mapping
        lookup (also uppercase keys) to match correctly."""
        data_dir, _ = cohort_on_disk
        cohort = ingest_data(data_dir)

        # All ingested term names must be uppercase
        for profile in cohort.profiles.values():
            for term_name in profile.records:
                assert term_name == term_name.upper(), (
                    f"Term name '{term_name}' is not uppercase"
                )

        # The mapping file uses uppercase keys
        mapping_path = data_dir.parent / "mapping.tsv"
        _create_mapping_file(mapping_path)
        term_to_category = parse_category_mapping(mapping_path)
        for key in term_to_category:
            assert key == key.upper()

    def test_unmapped_terms_excluded_from_cherry_pick(self, cohort):
        """Terms present in cohort but absent from mapping are excluded."""
        # Map only MITOCHONDRION, leaving all other terms unmapped
        term_to_category = {"MITOCHONDRION": "Mitochondria"}
        groups = select_cherry_picked_terms(cohort, term_to_category)

        assert len(groups) == 1
        assert groups[0].category_name == "Mitochondria"
        assert groups[0].term_names == ["MITOCHONDRION"]


# ============================================================================
# 2. Data flow: Ingestion -> Unbiased Selection (Units 1 + 4)
# ============================================================================


class TestIngestionToUnbiasedSelection:
    """Verify that CohortData feeds correctly into the unbiased term selection
    pipeline, producing valid CategoryGroups for dot plot rendering."""

    def test_ingested_data_feeds_unbiased_selection(self, cohort_on_disk):
        """Data ingested from disk flows through the full unbiased selection
        pipeline (pool, dedup, top-N, cluster) and produces valid groups."""
        data_dir, _ = cohort_on_disk
        cohort = ingest_data(data_dir)

        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )

        assert len(groups) <= 2
        assert len(groups) > 0
        total_terms = sum(len(g.term_names) for g in groups)
        assert total_terms <= 5
        assert stats.random_seed == 42
        assert stats.clustering_algorithm == "scipy.cluster.hierarchy (Ward linkage)"

    def test_fdr_threshold_from_config_applied_to_unbiased_selection(self, cohort):
        """The FDR threshold must be consistently applied: the same threshold
        used for ingestion filtering must be used for unbiased pooling."""
        config = ToolConfig(dot_plot=DotPlotConfig(fdr_threshold=0.01))

        # With strict FDR=0.01, fewer terms should pass
        pooled_strict = pool_significant_terms(cohort, fdr_threshold=0.01)
        pooled_lenient = pool_significant_terms(cohort, fdr_threshold=0.05)

        assert len(pooled_strict) <= len(pooled_lenient), (
            "Stricter FDR threshold should yield fewer or equal pooled terms"
        )

    def test_unbiased_selection_determinism_with_same_seed(self, cohort):
        """Running unbiased selection twice with the same seed must produce
        identical results (reproducibility requirement from spec)."""
        groups1, stats1 = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        groups2, stats2 = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )

        assert len(groups1) == len(groups2)
        for g1, g2 in zip(groups1, groups2):
            assert g1.category_name == g2.category_name
            assert g1.term_names == g2.term_names


# ============================================================================
# 3. Data flow: Selection -> Dot Plot Rendering (Units 3/4 + 5)
# ============================================================================


class TestSelectionToDotPlotRendering:
    """Verify that CategoryGroups from either cherry-picked or unbiased
    selection feed correctly into the dot plot renderer."""

    def test_cherry_picked_groups_render_dot_plot(self, cohort, tmp_path):
        """Cherry-picked CategoryGroups produce a valid dot plot with correct
        file outputs and metadata."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mapping = {
            "MITOCHONDRION": "Mitochondria",
            "TRANSLATION": "Translation",
            "G PROTEIN-COUPLED RECEPTOR SIGNALING PATHWAY": "GPCR",
            "SYNAPSE": "Synapse",
        }
        groups = select_cherry_picked_terms(cohort, mapping)

        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="figure1_cherry_picked",
            output_dir=output_dir,
        )

        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()
        assert result.n_terms_displayed == sum(len(g.term_names) for g in groups)
        assert result.n_categories == len(groups)
        assert result.n_mutants == len(cohort.mutant_ids)

    def test_unbiased_groups_render_dot_plot(self, cohort, tmp_path):
        """Unbiased CategoryGroups produce a valid dot plot with correct
        file outputs and metadata."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )

        result = render_dot_plot(
            cohort=cohort,
            groups=groups,
            fdr_threshold=0.05,
            output_stem="figure2_unbiased",
            output_dir=output_dir,
        )

        assert result.pdf_path.exists()
        assert result.png_path.exists()
        assert result.svg_path.exists()
        assert result.n_terms_displayed == sum(len(g.term_names) for g in groups)
        assert result.n_mutants == 3

    def test_dot_grid_uses_correct_fdr_for_absence_encoding(self, cohort):
        """build_dot_grid must produce None for cells where FDR >= threshold,
        matching the spec requirement that empty cells indicate non-significance."""
        groups = [
            CategoryGroup(category_name="Test", term_names=["CHROMATIN REMODELING"]),
        ]

        # CHROMATIN REMODELING has FDR=0.04 in alpha, 0.06 in beta, 0.01 in gamma
        nes_matrix, sig_matrix, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, fdr_threshold=0.05
        )

        assert mutant_labels == ["alpha", "beta", "gamma"]
        assert term_labels == ["CHROMATIN REMODELING"]

        # alpha: FDR=0.04 < 0.05 -> should be present
        assert nes_matrix[0][0] is not None
        # beta: FDR=0.06 >= 0.05 -> should be absent (None)
        assert nes_matrix[0][1] is None
        # gamma: FDR=0.01 < 0.05 -> should be present
        assert nes_matrix[0][2] is not None


# ============================================================================
# 4. Data flow: Ingestion -> Fisher Meta-Analysis (Units 1 + 6)
# ============================================================================


class TestIngestionToFisherAnalysis:
    """Verify data flows correctly from ingestion through Fisher's combined
    probability test."""

    def test_ingested_cohort_produces_valid_fisher_result(self, cohort_on_disk):
        """Data ingested from disk produces a valid FisherResult when fed
        into run_fisher_analysis."""
        data_dir, output_dir = cohort_on_disk
        cohort = ingest_data(data_dir)
        config = FisherConfig()

        result = run_fisher_analysis(
            cohort=cohort,
            config=config,
            output_dir=output_dir,
            clustering_enabled=False,
        )

        # All GO IDs from cohort must appear in Fisher result
        assert set(result.go_ids) == cohort.all_go_ids
        assert result.n_mutants == len(cohort.mutant_ids)
        assert result.pvalue_matrix.shape == (len(result.go_id_order), len(result.mutant_ids))

        # All combined p-values must be in [0, 1]
        for go_id, pval in result.combined_pvalues.items():
            assert 0.0 <= pval <= 1.0, f"Combined p-value for {go_id} out of range: {pval}"

        # TSV files must be written
        assert (output_dir / "pvalue_matrix.tsv").exists()
        assert (output_dir / "fisher_combined_pvalues.tsv").exists()

    def test_pseudocount_replaces_zero_pvals_before_fisher(self, cohort):
        """Zero NOM p-values from ingestion are replaced with pseudocount
        before Fisher computation, preventing log(0) errors."""
        config = FisherConfig(pseudocount=1e-10)
        per_mutant = build_pvalue_dict_per_mutant(cohort, config.pseudocount)

        # Check that no p-value is exactly 0.0
        for mutant_id, pvals in per_mutant.items():
            for go_id, pval in pvals.items():
                assert pval > 0.0, (
                    f"p-value for {go_id} in {mutant_id} should not be zero "
                    f"after pseudocount replacement"
                )

    def test_missing_terms_imputed_as_one_in_pvalue_matrix(self, cohort):
        """GO terms not present in a mutant are imputed as p=1.0 in the
        p-value matrix, ensuring Fisher's test handles missing data correctly."""
        per_mutant = build_pvalue_dict_per_mutant(cohort, pseudocount=1e-10)
        matrix, go_id_order = build_pvalue_matrix(per_mutant, cohort.mutant_ids)

        # CHROMATIN BINDING (GO:0003682) is only in gamma
        go_idx = go_id_order.index("GO:0003682")
        alpha_idx = cohort.mutant_ids.index("alpha")
        beta_idx = cohort.mutant_ids.index("beta")

        assert matrix[go_idx, alpha_idx] == 1.0, "Missing term should be imputed as 1.0"
        assert matrix[go_idx, beta_idx] == 1.0, "Missing term should be imputed as 1.0"

    def test_fisher_combined_pvalue_mathematically_correct(self, cohort):
        """Verify Fisher's combined p-value is mathematically correct for a
        known case. For a term present in all mutants with known p-values,
        the combined p-value should match the chi-squared survival function."""
        from scipy.stats import chi2

        per_mutant = build_pvalue_dict_per_mutant(cohort, pseudocount=1e-10)
        matrix, go_id_order = build_pvalue_matrix(per_mutant, cohort.mutant_ids)

        # PEROXISOME (GO:0005777) is present in all 3 mutants
        go_idx = go_id_order.index("GO:0005777")
        pvals_row = matrix[go_idx, :]

        # Manual Fisher statistic
        fisher_stat = -2.0 * np.sum(np.log(pvals_row))
        df = 2 * len(cohort.mutant_ids)
        expected_combined = chi2.sf(fisher_stat, df)

        # Compute via the function
        combined = compute_fisher_combined(matrix, len(cohort.mutant_ids))
        actual_combined = combined[go_idx]

        assert abs(actual_combined - expected_combined) < 1e-12, (
            f"Fisher combined p-value mismatch: expected {expected_combined}, got {actual_combined}"
        )


# ============================================================================
# 5. Data flow: Fisher -> Bar Plot Rendering (Units 6 + 8)
# ============================================================================


class TestFisherToBarPlotRendering:
    """Verify that FisherResult feeds correctly into bar plot rendering
    when clustering is disabled."""

    def test_fisher_result_renders_bar_plot_without_clustering(self, cohort, tmp_path):
        """FisherResult directly feeds bar plot rendering when clustering
        is disabled, producing correct output files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort=cohort,
            config=FisherConfig(),
            output_dir=output_dir,
            clustering_enabled=False,
        )

        bar_result = render_bar_plot(
            fisher_result=fisher_result,
            clustering_result=None,
            fisher_config=FisherConfig(top_n_bars=5),
            plot_config=PlotAppearanceConfig(),
            output_dir=output_dir,
        )

        assert bar_result.pdf_path.exists()
        assert bar_result.png_path.exists()
        assert bar_result.svg_path.exists()
        assert bar_result.n_bars <= 5
        assert bar_result.n_bars > 0
        assert bar_result.clustering_was_used is False
        assert bar_result.n_mutants == 3

    def test_bar_data_selection_respects_top_n_limit(self, cohort, tmp_path):
        """select_bar_data respects the top_n_bars limit from FisherConfig."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort=cohort,
            config=FisherConfig(),
            output_dir=output_dir,
            clustering_enabled=False,
        )

        term_names, neg_log_pvals, n_contributing = select_bar_data(
            fisher_result, None, top_n=3
        )

        assert len(term_names) == 3
        assert len(neg_log_pvals) == 3
        assert len(n_contributing) == 3

        # All neg_log values should be positive (p-values < 1)
        for v in neg_log_pvals:
            assert v >= 0.0

    def test_bar_data_ordered_by_significance(self, cohort, tmp_path):
        """Bar data must be ordered by combined p-value (most significant first),
        meaning neg_log_pvalues should be in descending order."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort=cohort,
            config=FisherConfig(),
            output_dir=output_dir,
            clustering_enabled=False,
        )

        term_names, neg_log_pvals, n_contributing = select_bar_data(
            fisher_result, None, top_n=20
        )

        # neg_log_pvalues should be in descending order (most significant first)
        for i in range(len(neg_log_pvals) - 1):
            assert neg_log_pvals[i] >= neg_log_pvals[i + 1], (
                f"Bar data not ordered by significance: "
                f"{neg_log_pvals[i]} < {neg_log_pvals[i+1]}"
            )


# ============================================================================
# 6. Data flow: Fisher -> Clustering -> Bar Plot (Units 6 + 7 + 8)
# ============================================================================


class TestFisherToClusteringToBarPlot:
    """Verify that FisherResult -> ClusteringResult -> BarPlot chain works
    correctly when clustering is enabled."""

    def test_clustering_result_feeds_bar_plot(self, cohort, tmp_path):
        """A manually constructed ClusteringResult feeds correctly into
        bar plot rendering."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort=cohort,
            config=FisherConfig(),
            output_dir=output_dir,
            clustering_enabled=True,
        )

        # Manually construct a ClusteringResult (bypassing actual OBO/GAF downloads)
        # Select the top 3 GO IDs by combined p-value as representatives
        sorted_go_ids = sorted(
            fisher_result.combined_pvalues.keys(),
            key=lambda gid: fisher_result.combined_pvalues[gid],
        )
        top3 = sorted_go_ids[:3]

        clustering_result = ClusteringResult(
            representatives=top3,
            representative_names=[fisher_result.go_id_to_name.get(gid, "") for gid in top3],
            representative_pvalues=[fisher_result.combined_pvalues[gid] for gid in top3],
            representative_n_contributing=[fisher_result.n_contributing[gid] for gid in top3],
            cluster_assignments={gid: i for i, gid in enumerate(sorted_go_ids)},
            n_clusters=len(sorted_go_ids),
            n_prefiltered=len(sorted_go_ids),
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )

        bar_result = render_bar_plot(
            fisher_result=fisher_result,
            clustering_result=clustering_result,
            fisher_config=FisherConfig(top_n_bars=3),
            plot_config=PlotAppearanceConfig(),
            output_dir=output_dir,
        )

        assert bar_result.clustering_was_used is True
        assert bar_result.n_bars <= 3
        assert bar_result.pdf_path.exists()

    def test_select_bar_data_uses_clustering_representatives(self, cohort, tmp_path):
        """When clustering_result is provided, select_bar_data must use
        the representative terms, not all Fisher terms."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort=cohort,
            config=FisherConfig(),
            output_dir=output_dir,
            clustering_enabled=False,
        )

        # Create a clustering result with only 2 representatives
        sorted_go_ids = sorted(
            fisher_result.combined_pvalues.keys(),
            key=lambda gid: fisher_result.combined_pvalues[gid],
        )
        reps = sorted_go_ids[:2]

        clustering_result = ClusteringResult(
            representatives=reps,
            representative_names=[fisher_result.go_id_to_name.get(gid, "") for gid in reps],
            representative_pvalues=[fisher_result.combined_pvalues[gid] for gid in reps],
            representative_n_contributing=[fisher_result.n_contributing[gid] for gid in reps],
            cluster_assignments={gid: i for i, gid in enumerate(sorted_go_ids)},
            n_clusters=2,
            n_prefiltered=len(sorted_go_ids),
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )

        term_names, neg_log_pvals, n_contributing = select_bar_data(
            fisher_result, clustering_result, top_n=20
        )

        # Should return exactly 2 terms (the representatives), not all Fisher terms
        assert len(term_names) == 2

    def test_clustering_tsv_output_consistency(self, cohort, tmp_path):
        """write_fisher_results_with_clusters_tsv output must contain all
        pre-filtered GO IDs and correctly mark representatives."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort=cohort,
            config=FisherConfig(),
            output_dir=output_dir,
            clustering_enabled=False,
        )

        sorted_go_ids = sorted(
            fisher_result.combined_pvalues.keys(),
            key=lambda gid: fisher_result.combined_pvalues[gid],
        )
        reps = sorted_go_ids[:2]

        clustering_result = ClusteringResult(
            representatives=reps,
            representative_names=[fisher_result.go_id_to_name.get(gid, "") for gid in reps],
            representative_pvalues=[fisher_result.combined_pvalues[gid] for gid in reps],
            representative_n_contributing=[fisher_result.n_contributing[gid] for gid in reps],
            cluster_assignments={gid: i % 2 for i, gid in enumerate(sorted_go_ids)},
            n_clusters=2,
            n_prefiltered=len(sorted_go_ids),
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )

        tsv_path = write_fisher_results_with_clusters_tsv(
            fisher_result, clustering_result, output_dir
        )

        assert tsv_path.exists()
        content = tsv_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # Header + one line per GO ID with cluster assignment
        assert len(lines) == len(sorted_go_ids) + 1

        # Check header
        header = lines[0]
        assert "Cluster" in header
        assert "Representative" in header


# ============================================================================
# 7. Cross-cutting: Configuration -> All downstream units (Unit 2 + others)
# ============================================================================


class TestConfigurationPropagation:
    """Verify that configuration values propagate correctly to all consuming units."""

    def test_custom_fdr_threshold_propagates_through_pipeline(self, cohort, tmp_path):
        """A custom FDR threshold from config propagates to both unbiased
        selection and dot plot rendering, affecting which cells appear."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Use a very strict threshold that excludes some terms
        strict_fdr = 0.01
        groups_strict, stats_strict = select_unbiased_terms(
            cohort, fdr_threshold=strict_fdr, top_n=5, n_groups=2, random_seed=42
        )

        # Use a lenient threshold
        lenient_fdr = 0.05
        groups_lenient, stats_lenient = select_unbiased_terms(
            cohort, fdr_threshold=lenient_fdr, top_n=5, n_groups=2, random_seed=42
        )

        # Stricter threshold should yield fewer or equal significant terms
        assert stats_strict.total_significant_terms <= stats_lenient.total_significant_terms

    def test_config_yaml_values_used_by_orchestration_path(self, tmp_project):
        """A config.yaml in the project directory is correctly loaded and
        its values are accessible to downstream units."""
        # Write a config.yaml with custom values
        config_yaml = tmp_project / "config.yaml"
        config_yaml.write_text(textwrap.dedent("""\
            dot_plot:
              fdr_threshold: 0.01
              top_n: 10
              n_groups: 3
              random_seed: 123
            fisher:
              pseudocount: 0.0001
              top_n_bars: 15
            clustering:
              enabled: false
            plot:
              dpi: 150
        """), encoding="utf-8")

        config = load_config(tmp_project)

        assert config.dot_plot.fdr_threshold == 0.01
        assert config.dot_plot.top_n == 10
        assert config.dot_plot.n_groups == 3
        assert config.dot_plot.random_seed == 123
        assert config.fisher.pseudocount == 0.0001
        assert config.fisher.top_n_bars == 15
        assert config.clustering.enabled is False
        assert config.plot_appearance.dpi == 150


# ============================================================================
# 8. Notes Generation: consumes outputs from all rendering units (Unit 9)
# ============================================================================


class TestNotesGenerationIntegration:
    """Verify that notes generation correctly consumes results from all
    upstream units and produces a coherent notes.md file."""

    def test_notes_generation_with_all_figures(self, cohort, tmp_path):
        """Notes generation produces a complete notes.md when all three
        figures are present (Figure 1 cherry-picked, Figure 2 unbiased,
        Figure 3 meta-analysis)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = ToolConfig()

        # Render Figure 1
        mapping = {
            "MITOCHONDRION": "Mitochondria",
            "TRANSLATION": "Translation",
        }
        cherry_groups = select_cherry_picked_terms(cohort, mapping)
        fig1_result = render_dot_plot(
            cohort, cherry_groups, 0.05, "figure1_cherry_picked", output_dir
        )

        # Render Figure 2
        unbiased_groups, unbiased_stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, unbiased_groups, 0.05, "figure2_unbiased", output_dir
        )

        # Run Fisher and render Figure 3
        fisher_result = run_fisher_analysis(cohort, FisherConfig(), output_dir, False)
        fig3_result = render_bar_plot(
            fisher_result, None, FisherConfig(top_n_bars=5),
            PlotAppearanceConfig(), output_dir
        )

        # Generate notes
        notes_input = NotesInput(
            cohort=cohort,
            config=config,
            fig1_result=fig1_result,
            fig1_method="tsv",
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=unbiased_stats,
            fisher_result=fisher_result,
            clustering_result=None,
        )
        notes_path = generate_notes(notes_input, output_dir)

        assert notes_path.exists()
        assert notes_path.name == "notes.md"

        content = notes_path.read_text(encoding="utf-8")
        # Must contain sections from all formatting functions
        assert "Figure 1" in content
        assert "Figure 2" in content
        assert "Figure 3" in content
        assert "Materials and Methods" in content
        assert "Summary Statistics" in content
        assert "Reproducibility Note" in content
        assert "Configuration Guide" in content

    def test_notes_without_figure1(self, cohort, tmp_path):
        """Notes generation works correctly when Figure 1 is omitted
        (no mapping file provided)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        unbiased_groups, unbiased_stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, unbiased_groups, 0.05, "figure2_unbiased", output_dir
        )
        fisher_result = run_fisher_analysis(cohort, FisherConfig(), output_dir, False)
        fig3_result = render_bar_plot(
            fisher_result, None, FisherConfig(top_n_bars=5),
            PlotAppearanceConfig(), output_dir
        )

        notes_input = NotesInput(
            cohort=cohort,
            config=ToolConfig(),
            fig1_result=None,
            fig1_method=None,
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=unbiased_stats,
            fisher_result=fisher_result,
            clustering_result=None,
        )
        notes_path = generate_notes(notes_input, output_dir)

        content = notes_path.read_text(encoding="utf-8")
        # Figure 1 legend should NOT be present
        assert "Figure 1:" not in content
        # Figure 2 and 3 should still be present
        assert "Figure 2" in content
        assert "Figure 3" in content

    def test_notes_reports_correct_mutant_count(self, cohort, tmp_path):
        """Notes summary statistics must report the correct number of mutants
        from the cohort data."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        unbiased_groups, unbiased_stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, unbiased_groups, 0.05, "figure2_unbiased", output_dir
        )
        fisher_result = run_fisher_analysis(cohort, FisherConfig(), output_dir, False)
        fig3_result = render_bar_plot(
            fisher_result, None, FisherConfig(top_n_bars=5),
            PlotAppearanceConfig(), output_dir
        )

        notes_input = NotesInput(
            cohort=cohort,
            config=ToolConfig(),
            fig1_result=None,
            fig1_method=None,
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=unbiased_stats,
            fisher_result=fisher_result,
            clustering_result=None,
        )

        stats_text = format_summary_statistics(notes_input)
        assert "Number of mutants analyzed: 3" in stats_text


# ============================================================================
# 9. Error propagation across unit boundaries
# ============================================================================


class TestErrorPropagation:
    """Test that errors from one unit propagate correctly to consuming units."""

    def test_ingestion_error_prevents_downstream_processing(self, tmp_path):
        """DataIngestionError from Unit 1 must not be silently swallowed --
        it should propagate to any caller attempting to use the data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create only one mutant folder (less than the required minimum of 2)
        _create_mutant_folder(
            data_dir, "single_mutant",
            [_make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.45, 1.85, 0.001, 0.02)],
            [_make_tsv_row("GO:0005777", "PEROXISOME", 52, -0.52, -2.22, 0.0, 0.003)],
        )

        with pytest.raises(DataIngestionError, match="at least 2"):
            ingest_data(data_dir)

    def test_missing_report_file_error_propagation(self, tmp_path):
        """Missing report files in a mutant folder should raise
        DataIngestionError that propagates through ingest_data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create two mutant folders but one is missing files
        _create_mutant_folder(
            data_dir, "alpha",
            [_make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.45, 1.85, 0.001, 0.02)],
            [_make_tsv_row("GO:0005777", "PEROXISOME", 52, -0.52, -2.22, 0.0, 0.003)],
        )

        # Create folder without the pos file
        broken_folder = data_dir / "broken.GseaPreranked.12345"
        broken_folder.mkdir()
        _write_tsv(
            broken_folder / "gsea_report_for_na_neg_12345.tsv",
            [_make_tsv_row("GO:0005739", "MITOCHONDRION", 100, -0.45, -1.85, 0.001, 0.02)],
        )

        with pytest.raises(DataIngestionError, match="Missing positive"):
            ingest_data(data_dir)

    def test_insufficient_significant_terms_raises_in_unbiased_selection(self, tmp_path):
        """If no terms pass the FDR threshold, unbiased selection should
        raise an informative error."""
        # Build a cohort where no terms pass FDR=0.001
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # All terms have FDR > 0.5
        high_fdr_rows = [
            _make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.45, 1.85, 0.5, 0.8),
        ]
        _create_mutant_folder(data_dir, "x", high_fdr_rows, high_fdr_rows, "001")
        _create_mutant_folder(data_dir, "y", high_fdr_rows, high_fdr_rows, "002")

        cohort = ingest_data(data_dir)

        with pytest.raises(ValueError, match="Insufficient significant terms"):
            select_unbiased_terms(cohort, fdr_threshold=0.001, top_n=5, n_groups=2)

    def test_resolve_paths_error_for_missing_data_dir(self, tmp_path):
        """resolve_paths raises FileNotFoundError when data/ does not exist,
        blocking all downstream processing."""
        with pytest.raises(FileNotFoundError, match="Data directory"):
            resolve_paths(tmp_path, None)


# ============================================================================
# 10. Orchestration path resolution (Unit 10)
# ============================================================================


class TestOrchestrationPathResolution:
    """Verify the orchestration unit correctly resolves paths for all units."""

    def test_resolve_paths_creates_output_and_cache_dirs(self, tmp_path):
        """resolve_paths must create output/ and cache/ if they don't exist."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result_data, result_output, result_cache, result_mapping = resolve_paths(
            tmp_path, None
        )

        assert result_data == data_dir
        assert result_output.exists()
        assert result_cache.exists()
        assert result_mapping is None

    def test_resolve_paths_with_mapping_file(self, tmp_path):
        """resolve_paths correctly resolves a provided mapping file path."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        mapping_file = tmp_path / "mapping.tsv"
        _create_mapping_file(mapping_file)

        _, _, _, mapping_path = resolve_paths(tmp_path, str(mapping_file))

        assert mapping_path == mapping_file

    def test_resolve_paths_rejects_nonexistent_mapping(self, tmp_path):
        """resolve_paths raises FileNotFoundError for a mapping file that
        does not exist."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Mapping file"):
            resolve_paths(tmp_path, "/nonexistent/mapping.tsv")


# ============================================================================
# 11. Resource contention: shared output directory (Units 5, 6, 8, 9)
# ============================================================================


class TestSharedOutputDirectory:
    """Test that multiple units writing to the same output directory do not
    clobber each other's files."""

    def test_all_output_files_coexist_in_same_directory(self, cohort, tmp_path):
        """Dot plot, bar plot, Fisher TSV, and notes.md all coexist in
        the same output directory without clobbering."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = ToolConfig()

        # Render Figure 2
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, groups, 0.05, "figure2_unbiased", output_dir
        )

        # Run Fisher (writes pvalue_matrix.tsv and fisher_combined_pvalues.tsv)
        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, clustering_enabled=False
        )

        # Render Figure 3
        fig3_result = render_bar_plot(
            fisher_result, None, FisherConfig(top_n_bars=5),
            PlotAppearanceConfig(), output_dir
        )

        # Generate notes
        notes_input = NotesInput(
            cohort=cohort, config=config,
            fig1_result=None, fig1_method=None,
            fig2_result=fig2_result, fig3_result=fig3_result,
            unbiased_stats=stats, fisher_result=fisher_result,
            clustering_result=None,
        )
        generate_notes(notes_input, output_dir)

        # All expected files must exist
        expected_files = [
            "figure2_unbiased.pdf",
            "figure2_unbiased.png",
            "figure2_unbiased.svg",
            "figure3_meta_analysis.pdf",
            "figure3_meta_analysis.png",
            "figure3_meta_analysis.svg",
            "pvalue_matrix.tsv",
            "fisher_combined_pvalues.tsv",
            "notes.md",
        ]
        for filename in expected_files:
            assert (output_dir / filename).exists(), f"Missing output file: {filename}"


# ============================================================================
# 12. Merge conflict resolution across pos/neg files (Unit 1 cross-file logic)
# ============================================================================


class TestMergeConflictResolution:
    """Test the cross-file merge behavior when a term appears in both pos
    and neg files (conflict resolution logic)."""

    def test_duplicate_term_resolved_by_lower_pvalue(self, tmp_path):
        """When a term appears in both pos and neg files, the record with
        the lower nominal p-value is retained. This affects downstream NES
        values used by all other units."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create pos file with MITOCHONDRION having high p-value
        pos_rows = [
            _make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.45, 1.85, 0.05, 0.02),
        ]
        # Create neg file with MITOCHONDRION having lower p-value
        neg_rows = [
            _make_tsv_row("GO:0005739", "MITOCHONDRION", 100, -0.60, -2.50, 0.001, 0.003),
        ]

        _create_mutant_folder(data_dir, "conflict_a", pos_rows, neg_rows, "001")
        _create_mutant_folder(data_dir, "conflict_b",
                              [_make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.30, 1.20, 0.01, 0.04)],
                              [_make_tsv_row("GO:0006412", "TRANSLATION", 80, -0.40, -1.60, 0.01, 0.03)],
                              "002")

        cohort = ingest_data(data_dir)

        # For conflict_a, the neg record (p=0.001) should win over pos (p=0.05)
        conflict_a_record = cohort.profiles["conflict_a"].records["MITOCHONDRION"]
        assert conflict_a_record.nes < 0, (
            "Neg record should win because it has lower p-value, so NES should be negative"
        )
        assert conflict_a_record.nom_pval == 0.001

    def test_merge_resolution_propagates_correctly_to_fisher(self, tmp_path):
        """The merge-resolved NOM p-val flows correctly into Fisher's method.
        The retained p-value (not the discarded one) should appear in the
        p-value matrix."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Term with conflicting entries: pos has higher p-val, neg has lower
        pos_rows = [
            _make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.45, 1.85, 0.1, 0.02),
        ]
        neg_rows = [
            _make_tsv_row("GO:0005739", "MITOCHONDRION", 100, -0.60, -2.50, 0.002, 0.003),
        ]

        _create_mutant_folder(data_dir, "m1", pos_rows, neg_rows, "001")
        _create_mutant_folder(data_dir, "m2",
                              [_make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.30, 1.20, 0.01, 0.04)],
                              [], "002")

        cohort = ingest_data(data_dir)
        per_mutant = build_pvalue_dict_per_mutant(cohort, pseudocount=1e-10)

        # The retained p-value for m1 should be 0.002 (from neg), not 0.1 (from pos)
        assert abs(per_mutant["m1"]["GO:0005739"] - 0.002) < 1e-12


# ============================================================================
# 13. Timing/ordering: mutant ID sort order consistency
# ============================================================================


class TestSortOrderConsistency:
    """Verify that mutant ID ordering is consistent across all units that
    use it -- ingestion, dot plot X-axis, Fisher matrix columns, notes."""

    def test_mutant_ids_alphabetical_across_pipeline(self, cohort_on_disk):
        """Mutant IDs must be alphabetically sorted everywhere they appear."""
        data_dir, output_dir = cohort_on_disk

        cohort = ingest_data(data_dir)

        # Unit 1: mutant_ids must be sorted
        assert cohort.mutant_ids == sorted(cohort.mutant_ids)

        # Unit 5: dot grid mutant_labels must be sorted
        groups = [CategoryGroup("Test", ["MITOCHONDRION"])]
        _, _, _, mutant_labels = build_dot_grid(cohort, groups, 0.05)
        assert mutant_labels == sorted(mutant_labels)

        # Unit 6: Fisher matrix columns must match sorted mutant_ids
        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, False
        )
        assert fisher_result.mutant_ids == sorted(fisher_result.mutant_ids)

    def test_go_term_grouping_order_preserved_in_dot_grid(self, cohort):
        """The order of terms within CategoryGroups must be preserved
        when building the dot grid Y-axis labels."""
        groups = [
            CategoryGroup("Cat1", ["TRANSLATION", "MITOCHONDRION"]),
            CategoryGroup("Cat2", ["SYNAPSE", "PEROXISOME"]),
        ]

        _, _, term_labels, _ = build_dot_grid(cohort, groups, 0.05)

        assert term_labels == ["TRANSLATION", "MITOCHONDRION", "SYNAPSE", "PEROXISOME"]


# ============================================================================
# 14. Emergent behavior: unbiased selection + rendering composition
# ============================================================================


class TestEmergentBehavior:
    """Test for behaviors that only arise when units work together."""

    def test_redundancy_removal_affects_clustering_input(self, cohort):
        """Redundancy removal in Unit 4 reduces the number of terms passed
        to clustering, which can change the clustering outcome. This is
        emergent because neither unit alone controls the final result."""
        pooled = pool_significant_terms(cohort, fdr_threshold=0.05)
        deduped = remove_redundant_terms(pooled)

        # Deduplication should not increase term count
        assert len(deduped) <= len(pooled)

        # If any terms were removed, they should be lexically similar to survivors
        removed_terms = set(pooled.keys()) - set(deduped.keys())
        for removed in removed_terms:
            words_removed = set(removed.split())
            found_similar = False
            for survivor in deduped:
                words_survivor = set(survivor.split())
                if words_removed and words_survivor:
                    jaccard = len(words_removed & words_survivor) / len(words_removed | words_survivor)
                    if jaccard > 0.5:
                        found_similar = True
                        break
            assert found_similar, (
                f"Removed term '{removed}' has no lexically similar survivor"
            )

    def test_fisher_pvalue_matrix_shape_matches_cohort_dimensions(self, cohort, tmp_path):
        """The Fisher p-value matrix dimensions must match (n_go_terms x n_mutants),
        where n_go_terms is derived from ingestion and n_mutants from the cohort.
        This is cross-unit: ingestion defines the data universe, Fisher builds
        the matrix."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, False
        )

        assert fisher_result.pvalue_matrix.shape[0] == len(fisher_result.go_id_order)
        assert fisher_result.pvalue_matrix.shape[1] == len(cohort.mutant_ids)
        assert fisher_result.n_mutants == len(cohort.mutant_ids)

    def test_dot_plot_empty_cells_match_fisher_missing_imputation(self, cohort):
        """Dot plot empty cells (FDR >= threshold) and Fisher's p=1.0 imputation
        for missing terms should be conceptually aligned. A term that is absent
        from a mutant shows as empty in the dot plot and as p=1.0 in Fisher."""
        # CHROMATIN BINDING is only in gamma
        groups = [CategoryGroup("Test", ["CHROMATIN BINDING"])]
        nes_matrix, _, _, mutant_labels = build_dot_grid(cohort, groups, fdr_threshold=0.05)

        alpha_idx = mutant_labels.index("alpha")
        beta_idx = mutant_labels.index("beta")

        # Dot plot: should be None for alpha and beta (term not in their profiles)
        assert nes_matrix[0][alpha_idx] is None
        assert nes_matrix[0][beta_idx] is None

        # Fisher: should be 1.0 for alpha and beta
        per_mutant = build_pvalue_dict_per_mutant(cohort, pseudocount=1e-10)
        matrix, go_id_order = build_pvalue_matrix(per_mutant, cohort.mutant_ids)

        go_idx = go_id_order.index("GO:0003682")
        assert matrix[go_idx, cohort.mutant_ids.index("alpha")] == 1.0
        assert matrix[go_idx, cohort.mutant_ids.index("beta")] == 1.0


# ============================================================================
# 15. End-to-end test: full pipeline from disk to outputs
# ============================================================================


class TestEndToEnd:
    """End-to-end tests that validate complete input-to-output scenarios
    described in the stakeholder spec. These tests check domain-meaningful
    output values, not just types and shapes."""

    def test_full_pipeline_without_mapping_file(self, tmp_path):
        """Run the complete pipeline (Figures 2 and 3 only) from raw TSV files
        on disk through to final output files. Validates domain-meaningful
        output values including NES color encoding, FDR-based significance
        filtering, Fisher's combined p-values, and notes content.

        This test catches double-normalization, unit conversion errors, and
        scientifically meaningless outputs that would not be caught by
        contract-only checking.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _build_synthetic_cohort_on_disk(data_dir)

        # -- Step 1: Ingest data (Unit 1) --
        cohort = ingest_data(data_dir)

        assert len(cohort.mutant_ids) == 3
        assert cohort.mutant_ids == ["alpha", "beta", "gamma"]

        # Validate specific ingested values for domain correctness
        alpha_mito = cohort.profiles["alpha"].records["MITOCHONDRION"]
        assert alpha_mito.nes == pytest.approx(1.85)
        assert alpha_mito.fdr == pytest.approx(0.02)
        assert alpha_mito.nom_pval == pytest.approx(0.001)
        assert alpha_mito.go_id == "GO:0005739"

        # Verify NES sign is preserved correctly (negative for neg records)
        alpha_perox = cohort.profiles["alpha"].records["PEROXISOME"]
        assert alpha_perox.nes < 0, "Negative GSEA report should produce negative NES"
        assert alpha_perox.nes == pytest.approx(-2.22)

        # -- Step 2: Load config (Unit 2) --
        config = ToolConfig()

        # -- Step 3: Unbiased term selection (Unit 4) --
        groups, stats = select_unbiased_terms(
            cohort,
            fdr_threshold=config.dot_plot.fdr_threshold,
            top_n=config.dot_plot.top_n,
            n_groups=config.dot_plot.n_groups,
            random_seed=config.dot_plot.random_seed,
        )

        assert len(groups) > 0
        total_selected = sum(len(g.term_names) for g in groups)
        assert total_selected > 0

        # TRANSLATION and CYTOSOLIC RIBOSOME should have highest max abs NES
        # TRANSLATION: max abs NES across mutants = 2.40 (gamma)
        # CHROMATIN REMODELING: max abs NES = 2.30 (gamma)
        # PEROXISOME: max abs NES = 2.50 (gamma)
        # CYTOSOLIC RIBOSOME: max abs NES = 2.30 (gamma)
        # These should be among the selected terms
        all_selected_terms = set()
        for g in groups:
            all_selected_terms.update(g.term_names)

        # The terms with highest max abs NES should be selected
        # PEROXISOME (2.50), TRANSLATION (2.40), CHROMATIN REMODELING (2.30),
        # CYTOSOLIC RIBOSOME (2.30)
        assert "PEROXISOME" in all_selected_terms, (
            "PEROXISOME (max abs NES=2.50) should be in the top selected terms"
        )
        assert "TRANSLATION" in all_selected_terms, (
            "TRANSLATION (max abs NES=2.40) should be in the top selected terms"
        )

        # -- Step 4: Render Figure 2 (Unit 5) --
        fig2_result = render_dot_plot(
            cohort, groups, config.dot_plot.fdr_threshold,
            "figure2_unbiased", output_dir,
            dpi=config.plot_appearance.dpi,
            font_family=config.plot_appearance.font_family,
        )

        assert fig2_result.pdf_path.exists()
        assert fig2_result.png_path.exists()
        assert fig2_result.svg_path.exists()
        assert fig2_result.n_mutants == 3
        assert fig2_result.n_terms_displayed == total_selected

        # Validate the dot grid data for domain correctness
        nes_matrix, sig_matrix, term_labels, mutant_labels = build_dot_grid(
            cohort, groups, config.dot_plot.fdr_threshold
        )

        # Check that NES values in the grid match the ingested data
        if "MITOCHONDRION" in term_labels:
            mito_idx = term_labels.index("MITOCHONDRION")
            gamma_idx = mutant_labels.index("gamma")
            assert nes_matrix[mito_idx][gamma_idx] == pytest.approx(2.20), (
                "NES for MITOCHONDRION in gamma should be 2.20 (no normalization/transform)"
            )

        # Check that -log10(FDR) is computed correctly for significance encoding
        if "PEROXISOME" in term_labels:
            perox_idx = term_labels.index("PEROXISOME")
            alpha_idx = mutant_labels.index("alpha")
            expected_sig = -math.log10(0.003)
            if sig_matrix[perox_idx][alpha_idx] is not None:
                assert sig_matrix[perox_idx][alpha_idx] == pytest.approx(expected_sig, rel=0.01), (
                    f"Significance encoding should be -log10(0.003) = {expected_sig}"
                )

        # -- Step 5: Fisher meta-analysis (Unit 6) --
        fisher_result = run_fisher_analysis(
            cohort, config.fisher, output_dir, clustering_enabled=False
        )

        assert (output_dir / "pvalue_matrix.tsv").exists()
        assert (output_dir / "fisher_combined_pvalues.tsv").exists()

        # Validate Fisher combined p-values are domain-meaningful
        # Terms present in all 3 mutants with very low p-values should have
        # very significant combined p-values
        # PEROXISOME is present in all 3 mutants with p=0.0 -> pseudocount
        perox_combined = fisher_result.combined_pvalues["GO:0005777"]
        assert perox_combined < 0.01, (
            f"PEROXISOME combined p-value should be very significant (got {perox_combined})"
        )

        # CHROMATIN BINDING is only in gamma -> should have less significant
        # combined p-value due to p=1.0 imputation for alpha and beta
        chromatin_binding_combined = fisher_result.combined_pvalues["GO:0003682"]
        assert chromatin_binding_combined > perox_combined, (
            "CHROMATIN BINDING (1 mutant) should have less significant combined "
            "p-value than PEROXISOME (3 mutants)"
        )

        # Validate contributing mutant counts
        assert fisher_result.n_contributing["GO:0005777"] == 3  # PEROXISOME in all 3
        assert fisher_result.n_contributing["GO:0003682"] == 1  # CHROMATIN BINDING only in gamma

        # -- Step 6: Render Figure 3 (Unit 8) --
        fig3_result = render_bar_plot(
            fisher_result, None, config.fisher,
            config.plot_appearance, output_dir
        )

        assert fig3_result.pdf_path.exists()
        assert fig3_result.n_bars > 0
        assert fig3_result.n_mutants == 3
        assert fig3_result.clustering_was_used is False

        # -- Step 7: Generate notes (Unit 9) --
        notes_input = NotesInput(
            cohort=cohort, config=config,
            fig1_result=None,
            fig1_method=None,
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=stats,
            fisher_result=fisher_result,
            clustering_result=None,
        )
        notes_path = generate_notes(notes_input, output_dir)

        assert notes_path.exists()
        notes_content = notes_path.read_text(encoding="utf-8")

        # Validate domain-meaningful content in notes
        assert "3 mutant lines" in notes_content or "3 mutants" in notes_content, (
            "Notes must mention the correct number of mutant lines"
        )
        assert "Fisher" in notes_content, (
            "Notes must describe Fisher's combined probability test"
        )
        assert "chi-squared" in notes_content or "chi-square" in notes_content, (
            "Notes must describe the chi-squared distribution used"
        )
        # Degrees of freedom should be 2*3 = 6
        assert "6 degrees of freedom" in notes_content, (
            "Notes must report correct degrees of freedom (2k = 2*3 = 6)"
        )
        assert "random seed" in notes_content.lower() or "random_seed" in notes_content.lower(), (
            "Notes must include the random seed for reproducibility"
        )

    def test_full_pipeline_with_mapping_file(self, tmp_path):
        """Run the complete pipeline with a mapping file (all 3 figures).
        Validates that Figure 1 cherry-picked dot plot is produced alongside
        Figures 2 and 3, and that notes include all three figure legends."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _build_synthetic_cohort_on_disk(data_dir)

        mapping_file = tmp_path / "mapping.tsv"
        _create_mapping_file(mapping_file)

        # -- Ingest --
        cohort = ingest_data(data_dir)
        config = ToolConfig()

        # -- Cherry-picked selection and Figure 1 --
        term_to_category = parse_category_mapping(mapping_file)
        cherry_groups = select_cherry_picked_terms(cohort, term_to_category)

        assert len(cherry_groups) == 4  # All 4 categories have terms

        fig1_result = render_dot_plot(
            cohort, cherry_groups, config.dot_plot.fdr_threshold,
            "figure1_cherry_picked", output_dir
        )

        # Figure 1 must exist
        assert fig1_result.pdf_path.exists()
        assert fig1_result.n_categories == 4
        assert fig1_result.n_mutants == 3

        # -- Unbiased selection and Figure 2 --
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, groups, 0.05, "figure2_unbiased", output_dir
        )

        # -- Fisher and Figure 3 --
        fisher_result = run_fisher_analysis(cohort, config.fisher, output_dir, False)
        fig3_result = render_bar_plot(
            fisher_result, None, config.fisher, config.plot_appearance, output_dir
        )

        # -- Notes --
        notes_input = NotesInput(
            cohort=cohort, config=config,
            fig1_result=fig1_result,
            fig1_method="tsv",
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=stats,
            fisher_result=fisher_result,
            clustering_result=None,
        )
        notes_path = generate_notes(notes_input, output_dir)

        content = notes_path.read_text(encoding="utf-8")
        # All three figure legends must be present
        assert "Figure 1" in content
        assert "Figure 2" in content
        assert "Figure 3" in content
        # Cherry-picked method description
        assert "cherry-pick" in content.lower() or "cherry" in content.lower()

        # All expected output files must exist
        expected_files = [
            "figure1_cherry_picked.pdf", "figure1_cherry_picked.png", "figure1_cherry_picked.svg",
            "figure2_unbiased.pdf", "figure2_unbiased.png", "figure2_unbiased.svg",
            "figure3_meta_analysis.pdf", "figure3_meta_analysis.png", "figure3_meta_analysis.svg",
            "pvalue_matrix.tsv", "fisher_combined_pvalues.tsv", "notes.md",
        ]
        for f in expected_files:
            assert (output_dir / f).exists(), f"Missing: {f}"

    def test_pvalue_matrix_tsv_content_correctness(self, cohort, tmp_path):
        """Validate that the pvalue_matrix.tsv file contains correct domain
        values: pseudocount-replaced p-values where NOM p-val was 0.0, and
        1.0 for missing GO terms in a mutant."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(pseudocount=1e-10), output_dir, False
        )

        tsv_content = (output_dir / "pvalue_matrix.tsv").read_text(encoding="utf-8")
        lines = tsv_content.strip().split("\n")

        # Parse header
        header = lines[0].split("\t")
        assert header[0] == "GO_ID"
        assert header[1] == "Term_Name"
        assert header[2:] == cohort.mutant_ids

        # Find CHROMATIN BINDING row (GO:0003682, only in gamma)
        for line in lines[1:]:
            fields = line.split("\t")
            if fields[0] == "GO:0003682":
                alpha_pval = float(fields[2])  # alpha
                beta_pval = float(fields[3])   # beta
                gamma_pval = float(fields[4])  # gamma

                assert alpha_pval == 1.0, "Missing term imputed as 1.0"
                assert beta_pval == 1.0, "Missing term imputed as 1.0"
                assert gamma_pval < 1.0, "Present term should have actual p-value"
                break
        else:
            pytest.fail("GO:0003682 not found in pvalue_matrix.tsv")

    def test_nes_sign_preserved_end_to_end(self, tmp_path):
        """Verify that NES sign (positive vs negative) is preserved correctly
        from TSV ingestion through to the dot grid. This catches potential
        double-negation or absolute-value bugs in the pipeline."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create data with known NES signs
        pos_rows = [
            _make_tsv_row("GO:0005739", "MITOCHONDRION", 100, 0.45, 1.85, 0.001, 0.02),
        ]
        neg_rows = [
            _make_tsv_row("GO:0005777", "PEROXISOME", 52, -0.52, -2.22, 0.0, 0.003),
        ]

        _create_mutant_folder(data_dir, "a1", pos_rows, neg_rows, "001")
        _create_mutant_folder(data_dir, "a2", pos_rows, neg_rows, "002")

        cohort = ingest_data(data_dir)

        # Verify positive NES is positive
        assert cohort.profiles["a1"].records["MITOCHONDRION"].nes > 0
        # Verify negative NES is negative
        assert cohort.profiles["a1"].records["PEROXISOME"].nes < 0

        # Build dot grid and verify NES sign preserved
        groups = [
            CategoryGroup("Test", ["MITOCHONDRION", "PEROXISOME"]),
        ]
        nes_matrix, _, term_labels, mutant_labels = build_dot_grid(cohort, groups, 0.05)

        mito_idx = term_labels.index("MITOCHONDRION")
        perox_idx = term_labels.index("PEROXISOME")
        a1_idx = mutant_labels.index("a1")

        assert nes_matrix[mito_idx][a1_idx] > 0, "Positive NES must be positive in dot grid"
        assert nes_matrix[perox_idx][a1_idx] < 0, "Negative NES must be negative in dot grid"

    def test_fisher_degrees_of_freedom_correct(self, cohort, tmp_path):
        """Verify Fisher's test uses correct degrees of freedom (2k) where
        k is the number of mutant lines. This catches a common implementation
        error of using k instead of 2k."""
        from scipy.stats import chi2

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(pseudocount=1e-10), output_dir, False
        )

        # Manually compute for one GO term to verify df=2k
        k = len(cohort.mutant_ids)
        df = 2 * k

        go_idx = fisher_result.go_id_order.index("GO:0005777")
        row = fisher_result.pvalue_matrix[go_idx, :]
        stat = -2.0 * np.sum(np.log(row))

        expected_p = chi2.sf(stat, df)
        actual_p = fisher_result.combined_pvalues["GO:0005777"]

        assert actual_p == pytest.approx(expected_p, rel=1e-10), (
            f"df={df} (2k where k={k}): expected p={expected_p}, got p={actual_p}"
        )

        # Verify it does NOT match the wrong df (k instead of 2k)
        # Use relative comparison because both p-values can be very small
        wrong_p = chi2.sf(stat, k)
        if expected_p > 0 and wrong_p > 0:
            relative_diff = abs(expected_p - wrong_p) / max(expected_p, wrong_p)
            assert relative_diff > 0.01, (
                f"Combined p-value should differ between df=2k and df=k: "
                f"correct={expected_p}, wrong={wrong_p}, rel_diff={relative_diff}"
            )


# ============================================================================
# 16. Data consistency: GO ID <-> term name mapping consistency
# ============================================================================


class TestGoIdTermNameConsistency:
    """Verify that GO ID to term name mapping is consistent across units."""

    def test_fisher_go_id_to_name_matches_ingestion(self, cohort, tmp_path):
        """The GO ID -> term name mapping in FisherResult must match
        the mapping from ingested data."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, False
        )

        # Build expected mapping from cohort
        expected_mapping = {}
        for profile in cohort.profiles.values():
            for rec in profile.records.values():
                if rec.go_id not in expected_mapping:
                    expected_mapping[rec.go_id] = rec.term_name

        # Fisher's mapping should match
        for go_id, name in fisher_result.go_id_to_name.items():
            assert go_id in expected_mapping, f"Unknown GO ID in Fisher result: {go_id}"
            assert name == expected_mapping[go_id], (
                f"Name mismatch for {go_id}: Fisher has '{name}', "
                f"ingestion has '{expected_mapping[go_id]}'"
            )

    def test_all_go_ids_in_fisher_match_cohort(self, cohort, tmp_path):
        """Every GO ID in the Fisher result must come from the cohort data."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, False
        )

        assert set(fisher_result.go_ids) == cohort.all_go_ids


# ============================================================================
# 17. Configuration defaults: no config.yaml produces valid pipeline run
# ============================================================================


class TestDefaultConfigurationPipeline:
    """Verify that the pipeline runs correctly with all default configuration
    values (no config.yaml present)."""

    def test_default_config_produces_valid_pipeline(self, tmp_project):
        """Without a config.yaml, load_config returns defaults that produce
        a valid configuration for all downstream units."""
        config = load_config(tmp_project)

        assert config.dot_plot.fdr_threshold == 0.05
        assert config.dot_plot.top_n == 20
        assert config.dot_plot.n_groups == 4
        assert config.dot_plot.random_seed == 42
        assert config.fisher.pseudocount == 1e-10
        assert config.fisher.top_n_bars == 20
        assert config.clustering.enabled is True
        assert config.plot_appearance.dpi == 300


# ============================================================================
# 18. Notes generation with clustering enabled (Units 7 + 9)
# ============================================================================


class TestNotesWithClusteringEnabled:
    """Verify that notes generation correctly incorporates clustering
    metadata when a ClusteringResult is provided."""

    def test_notes_include_clustering_details_when_enabled(self, cohort, tmp_path):
        """When clustering_result is not None, the notes must include
        clustering-specific information: similarity metric, threshold,
        number of clusters, and representative selection description."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run Fisher
        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, clustering_enabled=True
        )

        # Construct a synthetic ClusteringResult
        sorted_go_ids = sorted(
            fisher_result.combined_pvalues.keys(),
            key=lambda gid: fisher_result.combined_pvalues[gid],
        )
        reps = sorted_go_ids[:3]

        clustering_result = ClusteringResult(
            representatives=reps,
            representative_names=[fisher_result.go_id_to_name.get(gid, "") for gid in reps],
            representative_pvalues=[fisher_result.combined_pvalues[gid] for gid in reps],
            representative_n_contributing=[fisher_result.n_contributing[gid] for gid in reps],
            cluster_assignments={gid: i % 3 for i, gid in enumerate(sorted_go_ids)},
            n_clusters=3,
            n_prefiltered=len(sorted_go_ids),
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )

        # Generate Figure 2 and Figure 3
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, groups, 0.05, "figure2_unbiased", output_dir
        )
        fig3_result = render_bar_plot(
            fisher_result, clustering_result,
            FisherConfig(top_n_bars=5), PlotAppearanceConfig(), output_dir
        )

        notes_input = NotesInput(
            cohort=cohort,
            config=ToolConfig(),
            fig1_result=None,
            fig1_method=None,
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=stats,
            fisher_result=fisher_result,
            clustering_result=clustering_result,
        )
        notes_path = generate_notes(notes_input, output_dir)
        content = notes_path.read_text(encoding="utf-8")

        # Clustering-specific content must appear
        assert "Lin" in content, "Notes must mention the Lin similarity metric"
        assert "0.7" in content, "Notes must mention the similarity threshold"
        assert "semantic" in content.lower(), (
            "Notes must mention semantic similarity clustering"
        )

    def test_notes_summary_includes_cluster_count(self, cohort, tmp_path):
        """Summary statistics section must report the number of semantic
        clusters when clustering was applied."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, clustering_enabled=False
        )

        sorted_go_ids = sorted(
            fisher_result.combined_pvalues.keys(),
            key=lambda gid: fisher_result.combined_pvalues[gid],
        )
        reps = sorted_go_ids[:2]

        clustering_result = ClusteringResult(
            representatives=reps,
            representative_names=[fisher_result.go_id_to_name.get(gid, "") for gid in reps],
            representative_pvalues=[fisher_result.combined_pvalues[gid] for gid in reps],
            representative_n_contributing=[fisher_result.n_contributing[gid] for gid in reps],
            cluster_assignments={gid: i % 2 for i, gid in enumerate(sorted_go_ids)},
            n_clusters=2,
            n_prefiltered=len(sorted_go_ids),
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )

        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, groups, 0.05, "figure2_unbiased", output_dir
        )
        fig3_result = render_bar_plot(
            fisher_result, clustering_result,
            FisherConfig(top_n_bars=5), PlotAppearanceConfig(), output_dir
        )

        notes_input = NotesInput(
            cohort=cohort,
            config=ToolConfig(),
            fig1_result=None,
            fig1_method=None,
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=stats,
            fisher_result=fisher_result,
            clustering_result=clustering_result,
        )

        stats_text = format_summary_statistics(notes_input)
        assert "2" in stats_text, "Summary must mention the number of clusters"
        assert "cluster" in stats_text.lower(), "Summary must mention clusters"


# ============================================================================
# 19. Fisher with BH-FDR correction (Units 1 + 6)
# ============================================================================


class TestFisherWithFDRCorrection:
    """Verify Fisher analysis with Benjamini-Hochberg FDR correction enabled."""

    def test_fdr_corrected_pvalues_are_weakly_larger(self, cohort, tmp_path):
        """BH-FDR corrected p-values must be >= the uncorrected combined
        p-values for every GO term."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config_with_fdr = FisherConfig(apply_fdr=True)
        fisher_result = run_fisher_analysis(
            cohort, config_with_fdr, output_dir, clustering_enabled=False
        )

        assert fisher_result.corrected_pvalues is not None

        for go_id in fisher_result.go_ids:
            raw = fisher_result.combined_pvalues[go_id]
            corrected = fisher_result.corrected_pvalues[go_id]
            assert corrected >= raw - 1e-15, (
                f"Corrected p-value ({corrected}) < raw ({raw}) for {go_id}"
            )
            assert 0.0 <= corrected <= 1.0, (
                f"Corrected p-value out of range for {go_id}: {corrected}"
            )

    def test_fdr_disabled_returns_none_corrected(self, cohort, tmp_path):
        """When apply_fdr is False, corrected_pvalues must be None."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config_no_fdr = FisherConfig(apply_fdr=False)
        fisher_result = run_fisher_analysis(
            cohort, config_no_fdr, output_dir, clustering_enabled=False
        )

        assert fisher_result.corrected_pvalues is None


# ============================================================================
# 20. Config cherry_pick_categories -> ontology resolution (Units 2 + 3)
# ============================================================================


class TestConfigCherryPickToOntologyResolution:
    """Verify that cherry_pick_categories from config flow correctly into
    ontology-based category resolution."""

    def test_cherry_pick_categories_validated_before_resolution(self, tmp_path):
        """Config validation ensures cherry_pick GO IDs match GO:NNNNNNN
        format before they reach the ontology resolver."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(textwrap.dedent("""\
            cherry_pick:
              - go_id: "INVALID"
                label: "Bad Category"
        """), encoding="utf-8")

        with pytest.raises(ConfigError, match="GO:"):
            load_config(tmp_path)

    def test_cherry_pick_empty_label_rejected(self, tmp_path):
        """Config validation rejects cherry_pick entries with empty labels."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(textwrap.dedent("""\
            cherry_pick:
              - go_id: "GO:0005739"
                label: "   "
        """), encoding="utf-8")

        with pytest.raises(ConfigError, match="non-empty"):
            load_config(tmp_path)

    def test_cherry_pick_categories_loaded_as_correct_types(self, tmp_path):
        """CherryPickCategory objects from config have correct go_id and label
        types that Unit 3 expects."""
        from gsea_tool.configuration import CherryPickCategory as CPC

        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(textwrap.dedent("""\
            cherry_pick:
              - go_id: "GO:0005739"
                label: "Mitochondria"
              - go_id: "GO:0006412"
                label: "Translation"
        """), encoding="utf-8")

        config = load_config(tmp_path)
        assert len(config.cherry_pick_categories) == 2
        for cat in config.cherry_pick_categories:
            assert isinstance(cat, CPC)
            assert isinstance(cat.go_id, str)
            assert isinstance(cat.label, str)
            assert cat.go_id.startswith("GO:")


# ============================================================================
# 21. Fisher clustering_enabled flag controls TSV output (Units 6 + 10)
# ============================================================================


class TestFisherClusteringEnabledFlag:
    """Verify that the clustering_enabled flag passed to run_fisher_analysis
    correctly controls whether fisher_combined_pvalues.tsv is written."""

    def test_clustering_disabled_writes_fisher_tsv(self, cohort, tmp_path):
        """When clustering is disabled, run_fisher_analysis must write
        fisher_combined_pvalues.tsv directly."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        run_fisher_analysis(cohort, FisherConfig(), output_dir, clustering_enabled=False)

        assert (output_dir / "fisher_combined_pvalues.tsv").exists()

    def test_clustering_enabled_skips_fisher_tsv(self, cohort, tmp_path):
        """When clustering is enabled, run_fisher_analysis must NOT write
        fisher_combined_pvalues.tsv (the clustering unit writes it instead)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        run_fisher_analysis(cohort, FisherConfig(), output_dir, clustering_enabled=True)

        # pvalue_matrix.tsv is always written
        assert (output_dir / "pvalue_matrix.tsv").exists()
        # fisher_combined_pvalues.tsv should NOT exist yet
        assert not (output_dir / "fisher_combined_pvalues.tsv").exists()


# ============================================================================
# 22. Bar plot label truncation with PlotAppearanceConfig (Units 2 + 8)
# ============================================================================


class TestBarPlotLabelTruncation:
    """Verify that bar plot label truncation respects the config parameter."""

    def test_long_term_names_truncated_in_bar_data(self, cohort, tmp_path):
        """select_bar_data returns full names; render_bar_plot handles
        truncation via PlotAppearanceConfig.label_max_length. Verify
        the bar plot renders without error when labels exceed the max."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, clustering_enabled=False
        )

        # Use a very short label_max_length to force truncation
        plot_config = PlotAppearanceConfig(label_max_length=10)
        bar_result = render_bar_plot(
            fisher_result, None, FisherConfig(top_n_bars=5),
            plot_config, output_dir
        )

        assert bar_result.pdf_path.exists()
        assert bar_result.n_bars > 0


# ============================================================================
# 23. Reproducibility: identical inputs yield identical Fisher results
# ============================================================================


class TestFisherReproducibility:
    """Verify that Fisher analysis is deterministic."""

    def test_fisher_deterministic_across_runs(self, cohort, tmp_path):
        """Running Fisher analysis twice on the same data must produce
        identical combined p-values."""
        output_dir1 = tmp_path / "output1"
        output_dir1.mkdir()
        output_dir2 = tmp_path / "output2"
        output_dir2.mkdir()

        result1 = run_fisher_analysis(cohort, FisherConfig(), output_dir1, False)
        result2 = run_fisher_analysis(cohort, FisherConfig(), output_dir2, False)

        assert result1.go_id_order == result2.go_id_order
        for go_id in result1.go_ids:
            assert result1.combined_pvalues[go_id] == result2.combined_pvalues[go_id]


# ============================================================================
# 24. Clustering representative selection from Fisher results (Units 6 + 7)
# ============================================================================


class TestClusteringRepresentativeSelection:
    """Verify that select_representatives correctly uses Fisher combined
    p-values to choose cluster representatives."""

    def test_representatives_ordered_by_combined_pvalue(self, cohort, tmp_path):
        """select_representatives must order representatives by combined
        p-value ascending, using FisherResult data."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, clustering_enabled=False
        )

        go_ids = sorted(fisher_result.combined_pvalues.keys())
        # Create 3 artificial clusters
        clusters = [
            [0, 1, 2],
            [3, 4],
            [5, 6, 7],
        ]
        # Trim to available GO IDs
        n = len(go_ids)
        clusters = []
        cluster_size = max(1, n // 3)
        for i in range(0, n, cluster_size):
            clusters.append(list(range(i, min(i + cluster_size, n))))

        result = select_representatives(clusters, go_ids, fisher_result)

        # Representatives must be ordered by combined p-value ascending
        for i in range(len(result.representative_pvalues) - 1):
            assert result.representative_pvalues[i] <= result.representative_pvalues[i + 1], (
                f"Representatives not ordered by p-value: "
                f"{result.representative_pvalues[i]} > {result.representative_pvalues[i+1]}"
            )

        # Each representative must be from a different cluster
        rep_clusters = set()
        for rep_go_id in result.representatives:
            cluster_idx = result.cluster_assignments[rep_go_id]
            rep_clusters.add(cluster_idx)
        assert len(rep_clusters) == result.n_clusters

    def test_representative_names_match_fisher_go_id_to_name(self, cohort, tmp_path):
        """Representative names in ClusteringResult must match the
        go_id_to_name mapping from FisherResult."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, clustering_enabled=False
        )

        go_ids = sorted(fisher_result.combined_pvalues.keys())
        clusters = [[i] for i in range(len(go_ids))]

        result = select_representatives(clusters, go_ids, fisher_result)

        for go_id, name in zip(result.representatives, result.representative_names):
            expected_name = fisher_result.go_id_to_name.get(go_id, "")
            assert name == expected_name, (
                f"Name mismatch for {go_id}: clustering has '{name}', "
                f"Fisher has '{expected_name}'"
            )


# ============================================================================
# 25. Config -> Fisher pseudocount -> p-value matrix (Units 2 + 6)
# ============================================================================


class TestConfigPseudocountPropagation:
    """Verify that the pseudocount from config propagates correctly through
    the Fisher pipeline, affecting the p-value matrix values."""

    def test_different_pseudocounts_produce_different_matrices(self, cohort, tmp_path):
        """Different pseudocount values should produce different p-value
        matrices for terms that originally had NOM p-val = 0.0."""
        output_dir1 = tmp_path / "out1"
        output_dir1.mkdir()
        output_dir2 = tmp_path / "out2"
        output_dir2.mkdir()

        result1 = run_fisher_analysis(
            cohort, FisherConfig(pseudocount=1e-10), output_dir1, False
        )
        result2 = run_fisher_analysis(
            cohort, FisherConfig(pseudocount=1e-5), output_dir2, False
        )

        # Matrices should differ because zero p-values are replaced
        # with different pseudocounts
        assert not np.array_equal(result1.pvalue_matrix, result2.pvalue_matrix), (
            "Different pseudocounts should produce different p-value matrices"
        )

        # Combined p-values should also differ
        some_go_id = result1.go_ids[0]
        # They might be very close but should not be identical
        # since pseudocount affects log computation
        p1 = result1.combined_pvalues[some_go_id]
        p2 = result2.combined_pvalues[some_go_id]
        # For GO terms where all mutants had p=0, the difference will be large
        # For others it may be the same. Just check they are both valid.
        assert 0.0 <= p1 <= 1.0
        assert 0.0 <= p2 <= 1.0


# ============================================================================
# 26. N_contributing consistency across Fisher and bar plot (Units 6 + 8)
# ============================================================================


class TestNContributingConsistency:
    """Verify that the n_contributing counts from Fisher flow correctly
    into bar plot data selection."""

    def test_n_contributing_matches_actual_mutant_presence(self, cohort, tmp_path):
        """The n_contributing count for each GO term must equal the number
        of mutants where that term has p < 1.0 in the p-value matrix."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, clustering_enabled=False
        )

        for i, go_id in enumerate(fisher_result.go_id_order):
            row = fisher_result.pvalue_matrix[i, :]
            expected_count = int(np.sum(row < 1.0))
            actual_count = fisher_result.n_contributing[go_id]
            assert actual_count == expected_count, (
                f"n_contributing mismatch for {go_id}: "
                f"expected {expected_count}, got {actual_count}"
            )

    def test_bar_plot_contrib_counts_from_fisher(self, cohort, tmp_path):
        """Bar plot data selection preserves the n_contributing values
        from Fisher result when clustering is disabled."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, clustering_enabled=False
        )

        term_names, neg_log_pvals, n_contributing = select_bar_data(
            fisher_result, None, top_n=20
        )

        # Each contributing count must be between 1 and n_mutants
        for count in n_contributing:
            assert 1 <= count <= fisher_result.n_mutants, (
                f"Contributing count {count} out of valid range [1, {fisher_result.n_mutants}]"
            )


# ============================================================================
# 27. Dot plot rendering with all-empty grid (edge case, Units 4 + 5)
# ============================================================================


class TestDotPlotEdgeCases:
    """Test edge cases in the dot plot rendering pipeline."""

    def test_dot_plot_renders_when_all_cells_empty(self, tmp_path):
        """If all terms in a group have FDR >= threshold in all mutants,
        the dot plot should still render (with no visible dots)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Build a cohort where all FDR values are above 0.01
        records_a = {
            "TERM_A": TermRecord("TERM_A", "GO:0000001", 1.5, 0.5, 0.1, 50),
        }
        records_b = {
            "TERM_A": TermRecord("TERM_A", "GO:0000001", 1.2, 0.6, 0.2, 50),
        }
        cohort = CohortData(
            mutant_ids=["ma", "mb"],
            profiles={
                "ma": MutantProfile("ma", records_a),
                "mb": MutantProfile("mb", records_b),
            },
            all_term_names={"TERM_A"},
            all_go_ids={"GO:0000001"},
        )

        groups = [CategoryGroup("Test", ["TERM_A"])]

        # Use strict FDR threshold that excludes everything
        result = render_dot_plot(
            cohort, groups, fdr_threshold=0.01,
            output_stem="test_empty", output_dir=output_dir
        )

        assert result.pdf_path.exists()
        assert result.n_terms_displayed == 1

        # Verify the grid is actually all empty
        nes_matrix, _, _, _ = build_dot_grid(cohort, groups, 0.01)
        for row in nes_matrix:
            for cell in row:
                assert cell is None, "All cells should be empty with strict FDR"


# ============================================================================
# 28. Notes fig1_method propagation (Units 3 + 9 + 10)
# ============================================================================


class TestNotesFig1MethodPropagation:
    """Verify that the fig1_method value correctly influences the notes
    content, describing how Figure 1 categories were resolved."""

    def test_tsv_method_described_in_notes(self, cohort, tmp_path):
        """When fig1_method is 'tsv', notes describe manual category mapping."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mapping = {"MITOCHONDRION": "Mitochondria", "TRANSLATION": "Translation"}
        cherry_groups = select_cherry_picked_terms(cohort, mapping)
        fig1_result = render_dot_plot(
            cohort, cherry_groups, 0.05, "figure1_cherry_picked", output_dir
        )

        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, groups, 0.05, "figure2_unbiased", output_dir
        )
        fisher_result = run_fisher_analysis(cohort, FisherConfig(), output_dir, False)
        fig3_result = render_bar_plot(
            fisher_result, None, FisherConfig(top_n_bars=5),
            PlotAppearanceConfig(), output_dir
        )

        notes_input = NotesInput(
            cohort=cohort,
            config=ToolConfig(),
            fig1_result=fig1_result,
            fig1_method="tsv",
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=stats,
            fisher_result=fisher_result,
            clustering_result=None,
        )

        legend_text = format_figure_legends(notes_input)
        assert "mapping" in legend_text.lower(), (
            "TSV method should mention 'mapping' in legend"
        )

    def test_ontology_method_described_in_notes(self, cohort, tmp_path):
        """When fig1_method is 'ontology', notes describe ontology-based
        resolution with parent GO IDs."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Use config with cherry_pick categories
        from gsea_tool.configuration import CherryPickCategory
        config = ToolConfig(
            cherry_pick_categories=[
                CherryPickCategory(go_id="GO:0005739", label="Mitochondria"),
            ]
        )

        mapping = {"MITOCHONDRION": "Mitochondria"}
        cherry_groups = select_cherry_picked_terms(cohort, mapping)
        fig1_result = render_dot_plot(
            cohort, cherry_groups, 0.05, "figure1_cherry_picked", output_dir
        )

        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, groups, 0.05, "figure2_unbiased", output_dir
        )
        fisher_result = run_fisher_analysis(cohort, FisherConfig(), output_dir, False)
        fig3_result = render_bar_plot(
            fisher_result, None, FisherConfig(top_n_bars=5),
            PlotAppearanceConfig(), output_dir
        )

        notes_input = NotesInput(
            cohort=cohort,
            config=config,
            fig1_result=fig1_result,
            fig1_method="ontology",
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=stats,
            fisher_result=fisher_result,
            clustering_result=None,
        )

        legend_text = format_figure_legends(notes_input)
        assert "ontology" in legend_text.lower(), (
            "Ontology method should mention 'ontology' in legend"
        )
        assert "GO:0005739" in legend_text, (
            "Ontology method should mention the parent GO ID"
        )

    def test_no_fig1_method_when_fig1_absent(self, cohort, tmp_path):
        """When fig1_result is None, fig1_method must also be None, and
        notes must not contain a Figure 1 legend section."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, groups, 0.05, "figure2_unbiased", output_dir
        )
        fisher_result = run_fisher_analysis(cohort, FisherConfig(), output_dir, False)
        fig3_result = render_bar_plot(
            fisher_result, None, FisherConfig(top_n_bars=5),
            PlotAppearanceConfig(), output_dir
        )

        notes_input = NotesInput(
            cohort=cohort,
            config=ToolConfig(),
            fig1_result=None,
            fig1_method=None,
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=stats,
            fisher_result=fisher_result,
            clustering_result=None,
        )

        legend_text = format_figure_legends(notes_input)
        assert "Cherry-Pick" not in legend_text, (
            "No Figure 1 legend when fig1_result is None"
        )


# ============================================================================
# 29. Config invalid values rejected before reaching downstream units
# ============================================================================


class TestConfigValidationBlocksDownstream:
    """Verify that invalid config values are rejected during config loading,
    preventing them from reaching downstream units with invalid state."""

    def test_negative_fdr_threshold_rejected(self, tmp_path):
        """A negative FDR threshold must be rejected by config validation."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(textwrap.dedent("""\
            dot_plot:
              fdr_threshold: -0.1
        """), encoding="utf-8")

        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_zero_top_n_rejected(self, tmp_path):
        """top_n=0 must be rejected by config validation."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(textwrap.dedent("""\
            dot_plot:
              top_n: 0
        """), encoding="utf-8")

        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_zero_pseudocount_rejected(self, tmp_path):
        """pseudocount=0 must be rejected by config validation."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(textwrap.dedent("""\
            fisher:
              pseudocount: 0.0
        """), encoding="utf-8")

        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_similarity_threshold_over_one_rejected(self, tmp_path):
        """similarity_threshold > 1.0 must be rejected."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(textwrap.dedent("""\
            clustering:
              similarity_threshold: 1.5
        """), encoding="utf-8")

        with pytest.raises(ConfigError):
            load_config(tmp_path)


# ============================================================================
# 30. End-to-end with bar plot from clustering representatives (Units 6+7+8)
# ============================================================================


class TestEndToEndWithClusteringBarPlot:
    """End-to-end test for the Figure 3 path when clustering is used."""

    def test_bar_plot_from_clustering_has_correct_term_count(self, cohort, tmp_path):
        """When clustering produces N representatives, the bar plot must
        display at most min(N, top_n_bars) bars."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, clustering_enabled=True
        )

        sorted_go_ids = sorted(
            fisher_result.combined_pvalues.keys(),
            key=lambda gid: fisher_result.combined_pvalues[gid],
        )
        # Create 4 cluster representatives
        reps = sorted_go_ids[:4]

        clustering_result = ClusteringResult(
            representatives=reps,
            representative_names=[fisher_result.go_id_to_name.get(gid, "") for gid in reps],
            representative_pvalues=[fisher_result.combined_pvalues[gid] for gid in reps],
            representative_n_contributing=[fisher_result.n_contributing[gid] for gid in reps],
            cluster_assignments={gid: i % 4 for i, gid in enumerate(sorted_go_ids)},
            n_clusters=4,
            n_prefiltered=len(sorted_go_ids),
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )

        # top_n_bars=3 should limit to 3 bars
        bar_result = render_bar_plot(
            fisher_result, clustering_result,
            FisherConfig(top_n_bars=3), PlotAppearanceConfig(), output_dir
        )

        assert bar_result.n_bars == 3
        assert bar_result.clustering_was_used is True
        assert bar_result.pdf_path.exists()

    def test_bar_data_from_clustering_uses_representative_pvalues(self, cohort, tmp_path):
        """select_bar_data must use the representative p-values from
        ClusteringResult, not recompute from Fisher combined p-values."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fisher_result = run_fisher_analysis(
            cohort, FisherConfig(), output_dir, clustering_enabled=False
        )

        sorted_go_ids = sorted(
            fisher_result.combined_pvalues.keys(),
            key=lambda gid: fisher_result.combined_pvalues[gid],
        )
        reps = sorted_go_ids[:2]

        clustering_result = ClusteringResult(
            representatives=reps,
            representative_names=[fisher_result.go_id_to_name.get(gid, "") for gid in reps],
            representative_pvalues=[fisher_result.combined_pvalues[gid] for gid in reps],
            representative_n_contributing=[fisher_result.n_contributing[gid] for gid in reps],
            cluster_assignments={gid: i % 2 for i, gid in enumerate(sorted_go_ids)},
            n_clusters=2,
            n_prefiltered=len(sorted_go_ids),
            similarity_metric="Lin",
            similarity_threshold=0.7,
        )

        term_names, neg_log_pvals, n_contributing = select_bar_data(
            fisher_result, clustering_result, top_n=20
        )

        assert len(term_names) == 2
        # Verify the p-values match the representatives
        for i, go_id in enumerate(reps):
            expected_neg_log = -math.log10(fisher_result.combined_pvalues[go_id]) \
                if fisher_result.combined_pvalues[go_id] > 0 else 300.0
            assert neg_log_pvals[i] == pytest.approx(expected_neg_log, rel=1e-6)


# ============================================================================
# 31. Dependency version collection for notes (Unit 9)
# ============================================================================


class TestDependencyVersionCollection:
    """Verify that get_dependency_versions returns valid version strings
    for all required packages."""

    def test_all_required_packages_reported(self):
        """get_dependency_versions must include Python, matplotlib, scipy,
        numpy, and PyYAML at minimum."""
        versions = get_dependency_versions()

        required = ["Python", "matplotlib", "scipy", "numpy", "PyYAML"]
        for pkg in required:
            assert pkg in versions, f"Missing required package: {pkg}"
            assert len(versions[pkg]) > 0, f"Empty version for {pkg}"

    def test_versions_appear_in_reproducibility_note(self, cohort, tmp_path):
        """The version strings must appear in the reproducibility section
        of the notes output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=5, n_groups=2, random_seed=42
        )
        fig2_result = render_dot_plot(
            cohort, groups, 0.05, "figure2_unbiased", output_dir
        )
        fisher_result = run_fisher_analysis(cohort, FisherConfig(), output_dir, False)
        fig3_result = render_bar_plot(
            fisher_result, None, FisherConfig(top_n_bars=5),
            PlotAppearanceConfig(), output_dir
        )

        notes_input = NotesInput(
            cohort=cohort,
            config=ToolConfig(),
            fig1_result=None,
            fig1_method=None,
            fig2_result=fig2_result,
            fig3_result=fig3_result,
            unbiased_stats=stats,
            fisher_result=fisher_result,
            clustering_result=None,
        )

        repro_text = format_reproducibility_note(notes_input)

        versions = get_dependency_versions()
        for pkg in ["Python", "matplotlib", "scipy", "numpy"]:
            assert versions[pkg] in repro_text, (
                f"Version of {pkg} ({versions[pkg]}) not found in reproducibility note"
            )


# ============================================================================
# 32. Unbiased selection edge case: fewer terms than n_groups (Units 1 + 4)
# ============================================================================


class TestUnbiasedSelectionEdgeCases:
    """Test edge cases in unbiased selection that arise from data properties."""

    def test_n_groups_adjusted_when_fewer_terms_available(self):
        """When fewer terms pass the FDR threshold than n_groups, the
        pipeline should adjust n_groups downward rather than crash."""
        # Build a minimal cohort with exactly 3 significant terms
        records_a = {
            "TERM_X": TermRecord("TERM_X", "GO:0000001", 2.0, 0.01, 0.001, 50),
            "TERM_Y": TermRecord("TERM_Y", "GO:0000002", 1.8, 0.02, 0.002, 40),
            "TERM_Z": TermRecord("TERM_Z", "GO:0000003", 1.5, 0.03, 0.005, 30),
        }
        records_b = {
            "TERM_X": TermRecord("TERM_X", "GO:0000001", 1.9, 0.015, 0.002, 50),
            "TERM_Y": TermRecord("TERM_Y", "GO:0000002", 1.7, 0.025, 0.003, 40),
            "TERM_Z": TermRecord("TERM_Z", "GO:0000003", 1.4, 0.035, 0.006, 30),
        }
        cohort = CohortData(
            mutant_ids=["ma", "mb"],
            profiles={
                "ma": MutantProfile("ma", records_a),
                "mb": MutantProfile("mb", records_b),
            },
            all_term_names={"TERM_X", "TERM_Y", "TERM_Z"},
            all_go_ids={"GO:0000001", "GO:0000002", "GO:0000003"},
        )

        # Request 3 groups with top_n=3 -- exactly matches available terms
        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=3, n_groups=3, random_seed=42
        )

        assert len(groups) <= 3
        assert len(groups) > 0
        total_terms = sum(len(g.term_names) for g in groups)
        assert total_terms == 3

    def test_single_term_produces_single_group(self):
        """When only one term passes filtering, it should form a single
        group without clustering errors."""
        records_a = {
            "SOLE_TERM": TermRecord("SOLE_TERM", "GO:0000001", 3.0, 0.001, 0.0001, 50),
        }
        records_b = {
            "SOLE_TERM": TermRecord("SOLE_TERM", "GO:0000001", 2.8, 0.002, 0.0002, 50),
        }
        cohort = CohortData(
            mutant_ids=["ma", "mb"],
            profiles={
                "ma": MutantProfile("ma", records_a),
                "mb": MutantProfile("mb", records_b),
            },
            all_term_names={"SOLE_TERM"},
            all_go_ids={"GO:0000001"},
        )

        groups, stats = select_unbiased_terms(
            cohort, fdr_threshold=0.05, top_n=10, n_groups=1, random_seed=42
        )

        assert len(groups) == 1
        assert groups[0].term_names == ["SOLE_TERM"]
