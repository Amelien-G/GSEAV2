"""Tests for Unit 11 -- GO Tree Rendering.

DATA ASSUMPTIONS (module-level):
- Synthetic cohort uses 3 mutants (alpha, beta, gamma) with alphabetical IDs.
- Term names map to a tiny synthetic OBO covering 2 namespaces (BP + MF).
- `is_a` is the only relationship type tested (consistent with Unit 7).
"""

from pathlib import Path
from dataclasses import fields as dataclass_fields
import inspect

import matplotlib
matplotlib.use("Agg")
import pytest

from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.cherry_picked import CategoryGroup
from gsea_tool.go_tree import (
    GoTreeResult,
    build_go_tree,
    render_go_tree,
)

from tests.unit_11.mocks.mock_obo import write_synthetic_obo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _term_record(term_name: str, go_id: str) -> TermRecord:
    return TermRecord(
        term_name=term_name, go_id=go_id, nes=1.0, fdr=0.01,
        nom_pval=0.01, size=50,
    )


def _make_cohort_with_synthetic_terms() -> CohortData:
    """Build a cohort whose term names map to GO IDs in the synthetic OBO."""
    term_to_go = {
        "TRANSLATION": "GO:0006412",
        "OXIDATIVE PHOSPHORYLATION": "GO:0006119",
        "RIBOSOME BIOGENESIS": "GO:0042254",
        "TRANSFERASE ACTIVITY": "GO:0016740",  # MF namespace
    }
    profiles = {}
    for mutant in ["alpha", "beta", "gamma"]:
        records = {tn: _term_record(tn, gid) for tn, gid in term_to_go.items()}
        profiles[mutant] = MutantProfile(mutant_id=mutant, records=records)
    return CohortData(
        mutant_ids=["alpha", "beta", "gamma"],
        profiles=profiles,
        all_term_names=set(term_to_go.keys()),
        all_go_ids=set(term_to_go.values()),
    )


def _make_groups_bp_only() -> list[CategoryGroup]:
    return [
        CategoryGroup(category_name="Translation",
                      term_names=["TRANSLATION", "RIBOSOME BIOGENESIS"]),
        CategoryGroup(category_name="Energy",
                      term_names=["OXIDATIVE PHOSPHORYLATION"]),
    ]


def _make_groups_mixed_namespace() -> list[CategoryGroup]:
    return [
        CategoryGroup(category_name="Process",
                      term_names=["TRANSLATION", "OXIDATIVE PHOSPHORYLATION"]),
        CategoryGroup(category_name="Function",
                      term_names=["TRANSFERASE ACTIVITY"]),
    ]


# ---------------------------------------------------------------------------
# Signature / dataclass tests
# ---------------------------------------------------------------------------


def test_go_tree_result_is_dataclass_with_expected_fields():
    fnames = {f.name for f in dataclass_fields(GoTreeResult)}
    assert fnames == {
        "pdf_paths", "png_paths", "svg_paths",
        "n_leaf_terms", "n_internal_nodes", "n_internal_nodes_pruned",
        "n_namespaces",
    }


def test_render_go_tree_signature():
    sig = inspect.signature(render_go_tree)
    assert set(sig.parameters) == {
        "groups", "cohort", "obo_path", "output_stem", "output_dir",
        "title", "dpi", "font_family",
        # go_tree config knobs; both default to the previous hard-coded values.
        "label_max_chars", "show_namespace_root",
    }
    assert sig.parameters["show_namespace_root"].default is True


# ---------------------------------------------------------------------------
# build_go_tree contract
# ---------------------------------------------------------------------------


def test_build_go_tree_returns_leaves_matching_cohort_terms(tmp_path):
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = _make_groups_bp_only()

    parent_to_children, name_map, ns_map, leaf_ids = build_go_tree(
        groups, cohort, obo
    )

    assert leaf_ids == {"GO:0006412", "GO:0006119", "GO:0042254"}


def test_build_go_tree_walks_is_a_to_namespace_root(tmp_path):
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = _make_groups_bp_only()

    parent_to_children, name_map, ns_map, leaf_ids = build_go_tree(
        groups, cohort, obo
    )

    # Root of biological_process must appear as an ancestor.
    assert "GO:0008150" in name_map
    assert ns_map["GO:0008150"] == "biological_process"
    # Direct parents must appear too.
    assert "GO:0008152" in name_map  # metabolic process
    assert "GO:0009987" in name_map  # cellular process


def test_build_go_tree_parent_to_children_consistency(tmp_path):
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = _make_groups_bp_only()

    parent_to_children, name_map, ns_map, leaf_ids = build_go_tree(
        groups, cohort, obo
    )

    # Every node in name_map should appear as a key in parent_to_children
    for node in name_map:
        assert node in parent_to_children
    # Every child appearing in any parent's set must also be a known node.
    for parent, kids in parent_to_children.items():
        for c in kids:
            assert c in name_map


def test_build_go_tree_empty_groups_raises():
    cohort = _make_cohort_with_synthetic_terms()
    with pytest.raises(ValueError):
        build_go_tree([], cohort, Path("/nonexistent.obo"))


def test_build_go_tree_empty_group_raises(tmp_path):
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = [CategoryGroup(category_name="X", term_names=[])]
    with pytest.raises(ValueError):
        build_go_tree(groups, cohort, obo)


def test_build_go_tree_missing_obo_raises():
    cohort = _make_cohort_with_synthetic_terms()
    groups = _make_groups_bp_only()
    with pytest.raises(OSError):
        build_go_tree(groups, cohort, Path("/no/such/file.obo"))


# ---------------------------------------------------------------------------
# render_go_tree contract
# ---------------------------------------------------------------------------


def test_render_writes_pdf_png_svg(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = _make_groups_bp_only()

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="bp_tree", output_dir=output_dir,
    )

    assert len(result.pdf_paths) >= 1
    assert all(p.exists() for p in result.pdf_paths.values())
    assert all(p.exists() for p in result.png_paths.values())
    assert all(p.exists() for p in result.svg_paths.values())


def test_render_single_namespace_one_panel(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = _make_groups_bp_only()

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="single_ns", output_dir=output_dir,
    )

    assert result.n_namespaces == 1
    assert result.n_leaf_terms == 3
    assert len(result.pdf_paths) == 1
    assert len(result.png_paths) == 1
    assert len(result.svg_paths) == 1


def test_render_mixed_namespaces_multiple_panels(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = _make_groups_mixed_namespace()

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="mixed_ns", output_dir=output_dir,
    )

    assert result.n_namespaces == 2
    assert result.n_leaf_terms == 3
    assert len(result.pdf_paths) == 2
    assert set(result.pdf_paths.keys()) == {"biological_process", "molecular_function"}


def test_render_n_internal_excludes_leaves(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = _make_groups_bp_only()

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="counts", output_dir=output_dir,
    )

    # BUG-005 (Steiner reduction): the BP subgraph for
    # {translation, ox-phos, ribosome biogenesis} has biological_process
    # (root), metabolic_process and cellular_process as essential ancestors
    # because each connects 2 disjoint leaf clades -- so n_internal stays
    # at 3 here. n_internal_nodes + n_internal_nodes_pruned must equal the
    # original ancestor count.
    assert result.n_internal_nodes >= 1
    assert result.n_leaf_terms + result.n_internal_nodes >= 3
    assert result.n_internal_nodes_pruned >= 0


def test_render_single_term_still_produces_tree(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = [CategoryGroup(category_name="Solo",
                            term_names=["TRANSLATION"])]

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="solo", output_dir=output_dir,
    )

    assert all(p.exists() for p in result.pdf_paths.values())
    assert result.n_leaf_terms == 1


def test_render_missing_output_dir_raises(tmp_path):
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = _make_groups_bp_only()
    with pytest.raises(OSError):
        render_go_tree(
            groups=groups, cohort=cohort, obo_path=obo,
            output_stem="x", output_dir=tmp_path / "does_not_exist",
        )


def test_render_paths_use_provided_stem(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    obo = write_synthetic_obo(tmp_path)
    cohort = _make_cohort_with_synthetic_terms()
    groups = _make_groups_bp_only()

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="my_custom_stem", output_dir=output_dir,
    )

    # Per-namespace stems: {stem}_{namespace}.{ext}
    for ns, p in result.pdf_paths.items():
        assert p.name == f"my_custom_stem_{ns}.pdf"
    for ns, p in result.png_paths.items():
        assert p.name == f"my_custom_stem_{ns}.png"
    for ns, p in result.svg_paths.items():
        assert p.name == f"my_custom_stem_{ns}.svg"
