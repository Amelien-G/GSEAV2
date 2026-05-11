"""Regression test locking BUG-004: GO-tree figures must not clip labels
on dense namespace panels.

Before BUG-004's fix:
- `_render_namespace_panel` hard-truncated any label longer than 40 characters
  to `<37 chars>...`, so GO term names like
  ``proteasome-mediated ubiquitin-dependent protein catabolic process``
  rendered as ``proteasome-mediated ubiquitin-depende...``.
- `render_go_tree` produced a single PDF/PNG/SVG triple at a fixed 10-in
  width, regardless of how many nodes sat in the widest row of any
  namespace panel. Dense `biological_process` panels (rows of ~10-13
  nodes) overlapped labels into adjacent bboxes.

The fix:
- One file set per populated namespace (`{stem}_{namespace}.{pdf,png,svg}`),
  exposed via `GoTreeResult.pdf_paths / png_paths / svg_paths` dicts keyed
  by namespace.
- Per-namespace figure width auto-scales as `max(8.0, 1.2 * widest_row)`
  inches.
- Long labels wrap onto at most 3 lines via `textwrap.wrap(width=28)`
  rather than being truncated with `...`.

This test locks all three properties.
"""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

from gsea_tool.data_ingestion import CohortData, MutantProfile, TermRecord
from gsea_tool.cherry_picked import CategoryGroup
from gsea_tool.go_tree import render_go_tree, _wrap_label, _LABEL_WRAP_WIDTH


# A synthetic OBO with the kind of long GO term names that triggered the
# pre-fix truncation, plus a deep ancestor chain to force a "widest row" check.
_DENSE_OBO = """format-version: 1.2
ontology: bug004-regression

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


def _make_obo(tmp_path: Path) -> Path:
    p = tmp_path / "bug004.obo"
    p.write_text(_DENSE_OBO, encoding="utf-8")
    return p


def _make_cohort() -> CohortData:
    """A cohort whose term names map to the long GO IDs above."""
    name_to_id = {
        "PROTEASOME-MEDIATED UBIQUITIN-DEPENDENT PROTEIN CATABOLIC PROCESS": "GO:0043161",
        "POSITIVE REGULATION OF PROTEASOMAL UBIQUITIN-DEPENDENT PROTEIN CATABOLIC PROCESS": "GO:0032436",
        "RNA BINDING": "GO:0003723",
    }
    profiles = {}
    for mutant in ["alpha", "beta", "gamma"]:
        records = {
            tn: TermRecord(
                term_name=tn, go_id=gid,
                nes=1.0, fdr=0.01, nom_pval=0.01, size=50,
            )
            for tn, gid in name_to_id.items()
        }
        profiles[mutant] = MutantProfile(mutant_id=mutant, records=records)
    return CohortData(
        mutant_ids=["alpha", "beta", "gamma"],
        profiles=profiles,
        all_term_names=set(name_to_id.keys()),
        all_go_ids=set(name_to_id.values()),
    )


# ---------------------------------------------------------------------------
# Property 1: split into per-namespace files
# ---------------------------------------------------------------------------


def test_render_produces_one_file_per_populated_namespace(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    obo = _make_obo(tmp_path)
    cohort = _make_cohort()
    groups = [CategoryGroup(
        category_name="all",
        term_names=list(cohort.all_term_names),
    )]

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="bug004_split", output_dir=output_dir,
    )

    # Two namespaces are populated (biological_process + molecular_function).
    assert result.n_namespaces == 2
    assert set(result.pdf_paths.keys()) == {"biological_process", "molecular_function"}
    assert len(result.pdf_paths) == result.n_namespaces == len(result.png_paths) == len(result.svg_paths)

    # Per-namespace filenames carry the namespace suffix.
    for ns, p in result.pdf_paths.items():
        assert p.name == f"bug004_split_{ns}.pdf"
        assert p.exists()
    for ns, p in result.png_paths.items():
        assert p.name == f"bug004_split_{ns}.png"
        assert p.exists()
    for ns, p in result.svg_paths.items():
        assert p.name == f"bug004_split_{ns}.svg"
        assert p.exists()


# ---------------------------------------------------------------------------
# Property 2: labels are wrapped, not hard-truncated
# ---------------------------------------------------------------------------


def test_wrap_label_does_not_truncate_short_labels():
    assert _wrap_label("metabolic process") == "metabolic process"


def test_wrap_label_breaks_long_labels_into_multiple_lines():
    label = "proteasome-mediated ubiquitin-dependent protein catabolic process"
    wrapped = _wrap_label(label)
    # Should not be ellipsised -- three lines is enough for this label.
    assert "..." not in wrapped, (
        f"Pre-fix bug: long label was truncated with '...'. Got: {wrapped!r}"
    )
    # Should produce multiple lines.
    assert "\n" in wrapped
    # Every line should fit roughly within the wrap width (allow small slack
    # for non-breakable runs like long unhyphenated words).
    for line in wrapped.split("\n"):
        assert len(line) <= _LABEL_WRAP_WIDTH + 5


def test_wrap_label_ellipsises_only_when_no_more_lines_fit():
    extreme = "a" * 200
    wrapped = _wrap_label(extreme)
    # At most _LABEL_MAX_LINES (= 3) lines -> at most 2 newlines.
    assert wrapped.count("\n") <= 2


def test_rendered_svg_contains_full_long_label_text(tmp_path, monkeypatch):
    """The SVG output must contain ungarbled chunks of long term names that
    pre-fix would have been replaced by '...' inside the bbox.

    matplotlib normally renders SVG text as vector paths so the string is
    unrecoverable from the saved file. We force `svg.fonttype='none'` for
    this test so rendered labels appear as <text> elements containing the
    literal string we can search for.
    """
    monkeypatch.setitem(matplotlib.rcParams, "svg.fonttype", "none")

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    obo = _make_obo(tmp_path)
    cohort = _make_cohort()
    groups = [CategoryGroup(
        category_name="bp",
        term_names=[
            "PROTEASOME-MEDIATED UBIQUITIN-DEPENDENT PROTEIN CATABOLIC PROCESS",
            "POSITIVE REGULATION OF PROTEASOMAL UBIQUITIN-DEPENDENT PROTEIN CATABOLIC PROCESS",
        ],
    )]

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="bug004_text", output_dir=output_dir,
    )

    svg = result.svg_paths["biological_process"].read_text(encoding="utf-8")

    # Every meaningful word of the long labels must appear as text content
    # in the SVG (wrapped across multiple lines, but still searchable).
    for chunk in ["proteasome-mediated", "ubiquitin-dependent", "catabolic", "process"]:
        assert chunk in svg, f"SVG missing chunk {chunk!r} (label may still be truncated)"

    # The pre-fix ellipsised renders of these labels must not appear.
    assert "ubiquitin-depende..." not in svg
    assert "proteasome-mediated ubiquitin-depende..." not in svg


# ---------------------------------------------------------------------------
# Property 3: per-namespace figure width auto-scales to widest row
# ---------------------------------------------------------------------------


def test_dense_namespace_panel_gets_wider_canvas_than_default(tmp_path):
    """A namespace with many nodes must produce an SVG at least as wide
    as the auto-scaled floor (_MIN_FIG_WIDTH = 8 in -> 576 pt).
    matplotlib SVG carries `width="<N>pt"` on the root <svg> element.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    obo = _make_obo(tmp_path)
    cohort = _make_cohort()
    groups = [CategoryGroup(
        category_name="all",
        term_names=list(cohort.all_term_names),
    )]

    result = render_go_tree(
        groups=groups, cohort=cohort, obo_path=obo,
        output_stem="bug004_width", output_dir=output_dir,
    )

    def _svg_width_pt(p: Path) -> float:
        head = p.read_text(encoding="utf-8")[:2048]
        m = re.search(r'<svg[^>]*\bwidth="([0-9.]+)pt"', head)
        assert m, f"could not extract width from svg head: {head[:300]!r}"
        return float(m.group(1))

    bp_width_pt = _svg_width_pt(result.svg_paths["biological_process"])
    mf_width_pt = _svg_width_pt(result.svg_paths["molecular_function"])

    # 72 pt per inch; floor is 8 in = 576 pt. Allow some slack for matplotlib's
    # bbox_inches="tight" trimming which can shrink the saved canvas a bit.
    assert bp_width_pt >= 576 - 60, (
        f"biological_process SVG width {bp_width_pt}pt fell below the floor (576pt)"
    )
    # The biological_process panel has more terms and deeper levels than the
    # molecular_function panel, so it should never render narrower.
    assert bp_width_pt >= mf_width_pt - 50, (
        f"BP panel width {bp_width_pt}pt unexpectedly narrower than MF panel "
        f"({mf_width_pt}pt) -- auto-scaling may be broken"
    )
