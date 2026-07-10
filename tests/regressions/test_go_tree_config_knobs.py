"""Regression test for the go_tree config knobs.

Two knobs were deferred in references/lessons_learned.md ("if users complain"):

  * go_tree.label_max_chars      -- was hard-coded as Unit 11 _LABEL_WRAP_WIDTH
  * go_tree.show_namespace_root  -- namespace roots were always drawn

Both are now config-driven. The defaults reproduce the previous hard-coded
behaviour exactly, so existing figures are unchanged; these tests pin both the
defaults and the non-default behaviour.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.configuration import GoTreeConfig, ToolConfig
from gsea_tool.cherry_picked import CategoryGroup
from gsea_tool.go_tree import (
    _LABEL_WRAP_WIDTH,
    _drop_namespace_roots,
    _wrap_label,
    render_go_tree,
)

# root -> mid -> {leaf1, leaf2}; mid is a true branching point.
_OBO = """format-version: 1.2

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0030163
name: protein catabolic process
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0043161
name: proteasome-mediated ubiquitin-dependent protein catabolic process
namespace: biological_process
is_a: GO:0030163 ! protein catabolic process

[Term]
id: GO:0032436
name: positive regulation of proteasomal ubiquitin-dependent protein catabolic process
namespace: biological_process
is_a: GO:0030163 ! protein catabolic process
"""


def _build_cohort() -> CohortData:
    name_to_go = {
        "PROTEASOME-MEDIATED UBIQUITIN-DEPENDENT PROTEIN CATABOLIC PROCESS": "GO:0043161",
        "POSITIVE REGULATION OF PROTEASOMAL UBIQUITIN-DEPENDENT PROTEIN CATABOLIC PROCESS": "GO:0032436",
    }
    profiles = {}
    for mutant in ["alpha", "beta"]:
        records = {
            tn: TermRecord(term_name=tn, go_id=gid, nes=1.0, fdr=0.01,
                           nom_pval=0.01, size=50)
            for tn, gid in name_to_go.items()
        }
        profiles[mutant] = MutantProfile(mutant_id=mutant, records=records)
    return CohortData(
        mutant_ids=["alpha", "beta"],
        profiles=profiles,
        all_term_names=set(name_to_go.keys()),
        all_go_ids=set(name_to_go.values()),
    )


def _render(tmp_path, **kwargs):
    obo = tmp_path / "fixture.obo"
    obo.write_text(_OBO, encoding="utf-8")
    out = tmp_path / "output"
    out.mkdir(exist_ok=True)
    cohort = _build_cohort()
    groups = [CategoryGroup(category_name="cherry",
                            term_names=list(cohort.all_term_names))]
    return render_go_tree(groups=groups, cohort=cohort, obo_path=obo,
                          output_stem="knobs", output_dir=out, **kwargs)


class TestConfigDefaults:
    """Defaults must reproduce the previous hard-coded behaviour."""

    def test_go_tree_config_defaults(self):
        cfg = GoTreeConfig()
        assert cfg.label_max_chars == _LABEL_WRAP_WIDTH
        assert cfg.show_namespace_root is True

    def test_tool_config_exposes_go_tree(self):
        assert isinstance(ToolConfig().go_tree, GoTreeConfig)


class TestLabelMaxChars:

    def test_default_wrap_width_matches_module_constant(self):
        text = "a" * 10 + " " + "b" * 10 + " " + "c" * 10
        assert _wrap_label(text) == _wrap_label(text, wrap_width=_LABEL_WRAP_WIDTH)

    def test_narrower_width_produces_more_lines(self):
        text = "positive regulation of proteasomal protein catabolic process"
        wide = _wrap_label(text, wrap_width=40)
        narrow = _wrap_label(text, wrap_width=12)
        assert narrow.count("\n") > wide.count("\n")

    def test_wrap_width_respected(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        for line in _wrap_label(text, wrap_width=15).split("\n"):
            assert len(line) <= 15

    def test_render_accepts_label_max_chars(self, tmp_path):
        result = _render(tmp_path, label_max_chars=12)
        assert result.n_leaf_terms == 2


class TestShowNamespaceRoot:

    def test_root_dropped_and_children_promoted(self):
        essential = {"ROOT", "MID", "LEAF1", "LEAF2"}
        edges = {("MID", "ROOT"): False, ("LEAF1", "MID"): True, ("LEAF2", "MID"): True}
        leaves = {"LEAF1", "LEAF2"}
        nodes, kept = _drop_namespace_roots(essential, edges, leaves)
        assert nodes == {"MID", "LEAF1", "LEAF2"}
        assert ("MID", "ROOT") not in kept
        assert ("LEAF1", "MID") in kept and ("LEAF2", "MID") in kept

    def test_root_that_is_a_leaf_is_never_dropped(self):
        """A namespace root that is itself a plotted term carries data."""
        essential = {"ROOT", "LEAF1"}
        edges = {("LEAF1", "ROOT"): False}
        leaves = {"LEAF1", "ROOT"}
        nodes, kept = _drop_namespace_roots(essential, edges, leaves)
        assert nodes == {"ROOT", "LEAF1"}
        assert kept == edges

    def test_default_render_keeps_root(self, tmp_path):
        result = _render(tmp_path)
        # root (GO:0008150) + branching point (GO:0030163) are both internal
        assert result.n_internal_nodes == 2

    def test_show_namespace_root_false_drops_root(self, tmp_path):
        result = _render(tmp_path, show_namespace_root=False)
        # root dropped; only the branching point remains internal
        assert result.n_internal_nodes == 1

    def test_leaves_survive_root_drop(self, tmp_path):
        assert _render(tmp_path, show_namespace_root=False).n_leaf_terms == 2
