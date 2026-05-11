"""Regression test locking BUG-005: GO-tree figures must show only the
Steiner-tree of plotted leaves plus namespace roots, with dashed edges
where intermediate ancestors were collapsed.

Pre-BUG-005 behavior: render_go_tree displayed the FULL is_a ancestor
closure of every plotted leaf. Long single-child chains (a leaf chained
through 5-7 non-branching intermediate terms before joining anything
else) made the BP panel visually unreadable.

Post-fix behavior:
- A node is kept iff it is a plotted leaf, a namespace root, or a true
  branching point with >= 2 direct children whose dominated-leaf subsets
  are non-empty and distinct.
- Edges between kept nodes that correspond to a direct is_a parent in
  the original DAG are drawn solid (matplotlib linestyle "-").
- Edges between kept nodes that cross one or more collapsed ancestors
  are drawn dashed (matplotlib linestyle "--").
- GoTreeResult reports n_internal_nodes (essential ancestors shown) and
  n_internal_nodes_pruned (non-branching ancestors collapsed); the sum
  equals the original full-closure ancestor count.

This test locks the algorithm with three synthetic ontology shapes
(single-leaf chain, two-leaves-shared-parent, disjoint subtrees) and
verifies the BP-cherry-pick fixture from BUG-004 now collapses its
long passthrough chain.
"""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.cherry_picked import CategoryGroup
from gsea_tool.go_tree import (
    render_go_tree,
    _compute_essential_nodes,
    _compute_reduced_edges,
)


def _build_record(term_name: str, go_id: str) -> TermRecord:
    return TermRecord(
        term_name=term_name, go_id=go_id, nes=1.0, fdr=0.01,
        nom_pval=0.01, size=50,
    )


def _build_cohort(name_to_go: dict[str, str]) -> CohortData:
    profiles = {}
    for mutant in ["alpha", "beta", "gamma"]:
        records = {tn: _build_record(tn, gid) for tn, gid in name_to_go.items()}
        profiles[mutant] = MutantProfile(mutant_id=mutant, records=records)
    return CohortData(
        mutant_ids=["alpha", "beta", "gamma"],
        profiles=profiles,
        all_term_names=set(name_to_go.keys()),
        all_go_ids=set(name_to_go.values()),
    )


def _write_obo(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "fixture.obo"
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Algorithm unit tests: _compute_essential_nodes + _compute_reduced_edges
# ---------------------------------------------------------------------------


def test_single_leaf_deep_chain_collapses_to_leaf_plus_root():
    """A single leaf with 4 chained ancestors should reduce to {leaf, root}.

    DAG: root -> A -> B -> C -> D -> leaf
    Expected essential = {root, leaf}. Reduced edge (leaf, root) is dashed.
    """
    p2c = {
        "ROOT": {"A"},
        "A": {"B"},
        "B": {"C"},
        "C": {"D"},
        "D": {"LEAF"},
        "LEAF": set(),
    }
    namespace = {"ROOT", "A", "B", "C", "D", "LEAF"}
    leaves = {"LEAF"}

    essential = _compute_essential_nodes(p2c, leaves, namespace)
    assert essential == {"ROOT", "LEAF"}

    edges = _compute_reduced_edges(p2c, essential, namespace)
    assert edges == {("LEAF", "ROOT"): False}  # dashed


def test_two_leaves_shared_parent_keeps_parent_collapses_above():
    """Two leaves sharing one common parent: parent is essential (branching).

    DAG: root -> A -> B -> parent
                              -> leaf1
                              -> leaf2
    Expected essential = {root, parent, leaf1, leaf2}.
    leaf -> parent edges are solid. parent -> root edge is dashed (A, B collapsed).
    """
    p2c = {
        "ROOT": {"A"},
        "A": {"B"},
        "B": {"PARENT"},
        "PARENT": {"LEAF1", "LEAF2"},
        "LEAF1": set(),
        "LEAF2": set(),
    }
    namespace = set(p2c.keys())
    leaves = {"LEAF1", "LEAF2"}

    essential = _compute_essential_nodes(p2c, leaves, namespace)
    assert essential == {"ROOT", "PARENT", "LEAF1", "LEAF2"}

    edges = _compute_reduced_edges(p2c, essential, namespace)
    assert edges == {
        ("LEAF1", "PARENT"): True,    # direct is_a
        ("LEAF2", "PARENT"): True,    # direct is_a
        ("PARENT", "ROOT"): False,    # dashed: collapses A, B
    }


def test_disjoint_subtrees_essential_lca_kept():
    """Two leaves in disjoint subtrees: their LCA is kept.

    DAG: root -> X -> leaf1
              -> Y -> leaf2
    Both X and Y have only one leaf in their subtree (disjoint). Root is
    the LCA -- but root is already kept as a namespace root. X and Y are
    single-child internal nodes -> NOT essential, collapsed.
    Expected essential = {root, leaf1, leaf2}.
    """
    p2c = {
        "ROOT": {"X", "Y"},
        "X": {"LEAF1"},
        "Y": {"LEAF2"},
        "LEAF1": set(),
        "LEAF2": set(),
    }
    namespace = set(p2c.keys())
    leaves = {"LEAF1", "LEAF2"}

    essential = _compute_essential_nodes(p2c, leaves, namespace)
    assert essential == {"ROOT", "LEAF1", "LEAF2"}

    edges = _compute_reduced_edges(p2c, essential, namespace)
    assert edges == {
        ("LEAF1", "ROOT"): False,  # dashed: X collapsed
        ("LEAF2", "ROOT"): False,  # dashed: Y collapsed
    }


def test_branching_point_kept_when_two_distinct_leaf_subsets():
    """A node with two children, each with a distinct leaf, IS essential.

    DAG: root -> M (branching) -> leaf1
                                -> leaf2
    Expected essential = {root, M, leaf1, leaf2}.
    leaf -> M edges solid. M -> root edge solid (direct is_a).
    """
    p2c = {
        "ROOT": {"M"},
        "M": {"LEAF1", "LEAF2"},
        "LEAF1": set(),
        "LEAF2": set(),
    }
    namespace = set(p2c.keys())
    leaves = {"LEAF1", "LEAF2"}

    essential = _compute_essential_nodes(p2c, leaves, namespace)
    assert essential == {"ROOT", "M", "LEAF1", "LEAF2"}

    edges = _compute_reduced_edges(p2c, essential, namespace)
    assert edges == {
        ("LEAF1", "M"): True,
        ("LEAF2", "M"): True,
        ("M", "ROOT"): True,
    }


# ---------------------------------------------------------------------------
# Real cherry-pick OBO (mirrors BUG-004 fixture) collapses the long chain.
# ---------------------------------------------------------------------------


_DENSE_OBO = """format-version: 1.2
ontology: bug005-regression

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0009987
name: cellular process
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0008152
name: metabolic process
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0009056
name: catabolic process
namespace: biological_process
is_a: GO:0008152 ! metabolic process

[Term]
id: GO:0030163
name: protein catabolic process
namespace: biological_process
is_a: GO:0009056 ! catabolic process

[Term]
id: GO:0019941
name: modification-dependent protein catabolic process
namespace: biological_process
is_a: GO:0030163 ! protein catabolic process

[Term]
id: GO:0006511
name: ubiquitin-dependent protein catabolic process
namespace: biological_process
is_a: GO:0019941 ! modification-dependent protein catabolic process

[Term]
id: GO:0043161
name: proteasome-mediated ubiquitin-dependent protein catabolic process
namespace: biological_process
is_a: GO:0006511 ! ubiquitin-dependent protein catabolic process

[Term]
id: GO:0061136
name: regulation of proteasomal protein catabolic process
namespace: biological_process
is_a: GO:0030163 ! protein catabolic process

[Term]
id: GO:1903050
name: regulation of proteasomal ubiquitin-dependent protein catabolic process
namespace: biological_process
is_a: GO:0061136 ! regulation of proteasomal protein catabolic process

[Term]
id: GO:0032436
name: positive regulation of proteasomal ubiquitin-dependent protein catabolic process
namespace: biological_process
is_a: GO:1903050 ! regulation of proteasomal ubiquitin-dependent protein catabolic process

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0005488
name: binding
namespace: molecular_function
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0003723
name: RNA binding
namespace: molecular_function
is_a: GO:0005488 ! binding
"""


def _cherry_picked_cohort() -> CohortData:
    return _build_cohort({
        "PROTEASOME-MEDIATED UBIQUITIN-DEPENDENT PROTEIN CATABOLIC PROCESS": "GO:0043161",
        "POSITIVE REGULATION OF PROTEASOMAL UBIQUITIN-DEPENDENT PROTEIN CATABOLIC PROCESS": "GO:0032436",
        "RNA BINDING": "GO:0003723",
    })


def test_full_cherry_pick_fixture_prunes_passthrough_chain(tmp_path):
    """The BUG-004 fixture should now have multiple ancestors pruned by
    Steiner reduction, and the BP panel should be much smaller than the
    pre-fix full-closure version (which had ~9 ancestors)."""
    obo = _write_obo(tmp_path, _DENSE_OBO)
    cohort = _cherry_picked_cohort()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    groups = [CategoryGroup(
        category_name="cherry",
        term_names=list(cohort.all_term_names),
    )]

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="bug005_cherry", output_dir=output_dir,
    )

    # 3 leaves, 2 namespaces (BP + MF).
    assert result.n_leaf_terms == 3
    assert result.n_namespaces == 2

    # BP has 8 ancestors in the full closure (everything from 0030163 up to
    # 0008150 inclusive, plus the GO:0019941 / 0006511 / 0061136 / 1903050
    # chain). Of those, only 0030163 (protein catabolic process, where the
    # 2 BP leaves diverge) and 0008150 (root) are essential. The remaining
    # 6 should be pruned. MF has 2 ancestors (0005488 binding, 0003674 root);
    # only the root is essential, so 1 is pruned. Total pruned >= 6+1 = 7.
    assert result.n_internal_nodes_pruned >= 6, (
        f"Expected >= 6 pruned ancestors, got {result.n_internal_nodes_pruned}"
    )
    # Total internal essential should be small (2 BP essential ancestors
    # + 1 MF root) = 3.
    assert result.n_internal_nodes == 3, (
        f"Expected 3 essential internal nodes, got {result.n_internal_nodes}"
    )
    # Conservation: essential + pruned == full closure size.
    # Full closure for this fixture: 8 BP ancestors + 2 MF ancestors = 10.
    assert result.n_internal_nodes + result.n_internal_nodes_pruned == 10


# ---------------------------------------------------------------------------
# SVG visual smoke tests: dashed strokes must appear, leaf text must remain.
# ---------------------------------------------------------------------------


def test_rendered_svg_contains_dashed_strokes_for_collapsed_edges(tmp_path, monkeypatch):
    """matplotlib emits dashed-line paths with stroke-dasharray in SVG.
    At least one such pattern must appear in the BP panel because the
    cherry-pick fixture has multiple collapsed-ancestor edges."""
    monkeypatch.setitem(matplotlib.rcParams, "svg.fonttype", "none")

    obo = _write_obo(tmp_path, _DENSE_OBO)
    cohort = _cherry_picked_cohort()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    groups = [CategoryGroup(
        category_name="cherry",
        term_names=list(cohort.all_term_names),
    )]

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="bug005_dashed", output_dir=output_dir,
    )

    bp_svg = result.svg_paths["biological_process"].read_text(encoding="utf-8")
    # matplotlib renders dashed strokes either via stroke-dasharray or via
    # a "DashedLine" / dashed pattern class. Check for the canonical SVG
    # property name.
    assert "stroke-dasharray" in bp_svg, (
        "No dashed strokes in BP panel SVG -- collapsed-ancestor edges may not "
        "be using linestyle='--'"
    )

    # Leaf names must still be readable as wrapped chunks.
    for chunk in ["proteasome-mediated", "catabolic", "process", "binding"]:
        assert chunk in bp_svg or chunk in result.svg_paths["molecular_function"].read_text(
            encoding="utf-8"
        ), f"chunk {chunk!r} missing from rendered SVGs"


def test_rendered_svg_drops_pruned_ancestor_names(tmp_path, monkeypatch):
    """Pruned ancestor names like 'modification-dependent protein catabolic
    process' must NOT appear as labels in the rendered SVG -- they were
    collapsed."""
    monkeypatch.setitem(matplotlib.rcParams, "svg.fonttype", "none")

    obo = _write_obo(tmp_path, _DENSE_OBO)
    cohort = _cherry_picked_cohort()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    groups = [CategoryGroup(
        category_name="cherry",
        term_names=list(cohort.all_term_names),
    )]

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="bug005_drop", output_dir=output_dir,
    )

    bp_svg = result.svg_paths["biological_process"].read_text(encoding="utf-8")
    # Strip XML so we only search rendered text.
    plain = re.sub(r"<[^>]+>", " ", bp_svg)

    # "modification-dependent" is a non-branching ancestor of one leaf only
    # -> should be pruned. Same for "ubiquitin-dependent" as an ancestor
    # node (the leaf names DO contain those words, so we look for the
    # standalone ancestor term "modification-dependent" without "protein").
    # Conservative check: the pruned term should not appear as a standalone
    # ancestor label.
    assert "modification-dependent" not in plain or (
        "PROTEASOME-MEDIATED UBIQUITIN-DEPENDENT" in plain  # leaf has hyphenated phrase
    )
