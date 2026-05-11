"""Unit 11 -- GO Tree Rendering.

Renders a GO-term hierarchy figure paired with a dot plot. The "leaves" of
the tree are the GO terms displayed in the dot plot; "internal" nodes are
their is_a ancestors walked up to the namespace root. One file per
populated GO namespace (biological_process / molecular_function /
cellular_component) is written so the per-namespace figure can use the
canvas width its widest row requires without clipping labels.
"""

from pathlib import Path
from dataclasses import dataclass
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gsea_tool.data_ingestion import CohortData
from gsea_tool.cherry_picked import CategoryGroup
from gsea_tool.go_clustering import _parse_obo, _get_ancestors


# Drawing parameters tuned for publication output.
_NODE_FONT_SIZE = 7.0
_LEAF_FONT_WEIGHT = "bold"
_INTERNAL_FONT_WEIGHT = "normal"
_EDGE_COLOR = "#666666"
_EDGE_WIDTH = 0.8
_NAMESPACE_ORDER = ("biological_process", "molecular_function", "cellular_component")

# Label wrapping policy (BUG-004): wrap onto at most 3 lines around _LABEL_WRAP_WIDTH
# characters per line, with a column allocation of _PER_NODE_INCHES on the canvas.
_LABEL_WRAP_WIDTH = 28
_LABEL_MAX_LINES = 3
_PER_NODE_INCHES = 1.2
_MIN_FIG_WIDTH = 8.0
_MIN_FIG_HEIGHT = 3.5
_HEIGHT_PER_LEVEL = 1.0


@dataclass
class GoTreeResult:
    """Metadata about a rendered GO-tree figure, for notes.md consumption.

    Each path dict is keyed by GO namespace
    (``biological_process``/``molecular_function``/``cellular_component``).
    Only populated namespaces appear in the dicts.
    """
    pdf_paths: dict[str, Path]
    png_paths: dict[str, Path]
    svg_paths: dict[str, Path]
    n_leaf_terms: int
    n_internal_nodes: int
    n_namespaces: int


def _wrap_label(text: str) -> str:
    """Wrap a GO term label onto at most _LABEL_MAX_LINES lines.

    BUG-004 (post-delivery): the previous implementation truncated labels
    longer than 40 chars with an ellipsis, which made dense panels
    (especially biological_process) unreadable. We now wrap labels across
    lines instead. If wrapping still overflows _LABEL_MAX_LINES, the final
    line is ellipsised so the bbox stays bounded.
    """
    lines = textwrap.wrap(
        text,
        width=_LABEL_WRAP_WIDTH,
        break_long_words=False,
        break_on_hyphens=True,
    ) or [text]
    if len(lines) > _LABEL_MAX_LINES:
        kept = lines[: _LABEL_MAX_LINES]
        last = kept[-1]
        # Trim last line so trailing "..." fits within wrap width.
        if len(last) > _LABEL_WRAP_WIDTH - 3:
            last = last[: _LABEL_WRAP_WIDTH - 3].rstrip()
        kept[-1] = last + "..."
        lines = kept
    return "\n".join(lines)


def _term_name_to_go_id(cohort: CohortData) -> dict[str, str]:
    """Build a uppercase term-name -> GO ID lookup from the cohort.

    Term names are unique within a cohort (Unit 1 invariant); pick the first
    profile that has the term.
    """
    mapping: dict[str, str] = {}
    for profile in cohort.profiles.values():
        for term_name, record in profile.records.items():
            if term_name not in mapping:
                mapping[term_name] = record.go_id
    return mapping


def build_go_tree(
    groups: list[CategoryGroup],
    cohort: CohortData,
    obo_path: Path,
) -> tuple[
    dict[str, set[str]],
    dict[str, str],
    dict[str, str],
    set[str],
]:
    """Walk is_a parents from each plotted GO term up to namespace roots.

    Returns:
        parent_to_children: GO ID -> set of immediate is_a children present in the tree.
        go_id_to_name: GO ID -> human-readable term name.
        go_id_to_namespace: GO ID -> "biological_process" / "molecular_function" / "cellular_component".
        leaf_go_ids: the set of plotted GO IDs (the "leaves").
    """
    if len(groups) == 0:
        raise ValueError("Empty groups list")
    if not all(len(g.term_names) > 0 for g in groups):
        raise ValueError("No empty groups passed to renderer")
    if not obo_path.exists():
        raise OSError(f"OBO file not found: {obo_path}")

    obo_terms = _parse_obo(obo_path)
    name_to_go = _term_name_to_go_id(cohort)

    leaf_go_ids: set[str] = set()
    for group in groups:
        for term_name in group.term_names:
            go_id = name_to_go.get(term_name)
            if go_id is not None and go_id in obo_terms:
                leaf_go_ids.add(go_id)

    # Compute the union of leaves + their is_a ancestors via Unit 7's helper.
    cache: dict[str, set[str]] = {}
    all_node_ids: set[str] = set()
    for go_id in leaf_go_ids:
        all_node_ids |= _get_ancestors(go_id, obo_terms, cache)

    # Filter to terms present in OBO (the cache may include phantom IDs if
    # a parent reference is missing — we drop them rather than fail loudly).
    all_node_ids = {gid for gid in all_node_ids if gid in obo_terms}

    parent_to_children: dict[str, set[str]] = {gid: set() for gid in all_node_ids}
    for child_id in all_node_ids:
        for parent_id in obo_terms[child_id].get("is_a", []):
            if parent_id in all_node_ids:
                parent_to_children[parent_id].add(child_id)

    go_id_to_name = {gid: obo_terms[gid].get("name", gid) for gid in all_node_ids}
    go_id_to_namespace = {
        gid: obo_terms[gid].get("namespace", "") for gid in all_node_ids
    }

    return parent_to_children, go_id_to_name, go_id_to_namespace, leaf_go_ids


def _depth_from_roots(
    parent_to_children: dict[str, set[str]],
    namespace_nodes: set[str],
) -> tuple[dict[str, int], list[str]]:
    """Compute depth (longest path from any root) for each node in a single namespace.

    A "root" is a node with no parent inside namespace_nodes. Returns
    (depth_map, root_list).
    """
    # Build child -> parents reverse map within this namespace.
    parents_of: dict[str, set[str]] = {n: set() for n in namespace_nodes}
    for p, kids in parent_to_children.items():
        if p not in namespace_nodes:
            continue
        for c in kids:
            if c in namespace_nodes:
                parents_of[c].add(p)

    roots = sorted(n for n in namespace_nodes if not parents_of[n])

    # Longest-path depth via memoized DFS (DAG, so no cycles by construction).
    depth: dict[str, int] = {}

    def _depth(node: str) -> int:
        if node in depth:
            return depth[node]
        if not parents_of[node]:
            depth[node] = 0
            return 0
        depth[node] = 1 + max(_depth(p) for p in parents_of[node])
        return depth[node]

    for n in namespace_nodes:
        _depth(n)

    return depth, roots


def _layout_namespace(
    nodes: set[str],
    parent_to_children: dict[str, set[str]],
) -> tuple[dict[str, tuple[float, float]], int]:
    """Compute (x, y) coordinates for each node, top-down by depth.

    Depth 0 (roots) sits at the top. Within each row, x is assigned via a
    single barycenter pass: each node placed at the mean x of its parents
    in the prior row, then collisions are resolved by stable sort + spacing.
    Returns (positions, n_levels).
    """
    if not nodes:
        return {}, 0

    depth, roots = _depth_from_roots(parent_to_children, nodes)
    max_depth = max(depth.values())

    # Group by depth
    by_depth: dict[int, list[str]] = {d: [] for d in range(max_depth + 1)}
    for node, d in depth.items():
        by_depth[d].append(node)

    # Sort roots alphabetically for deterministic layout.
    by_depth[0].sort()

    pos: dict[str, tuple[float, float]] = {}

    # Roots: spread evenly across [0, len-1].
    n_roots = len(by_depth[0])
    for i, r in enumerate(by_depth[0]):
        x = float(i) if n_roots > 1 else 0.0
        pos[r] = (x, float(max_depth))   # y inverted later

    # Levels below roots: barycenter on parent x-values.
    for d in range(1, max_depth + 1):
        children = by_depth[d]
        if not children:
            continue

        # Build local parents-of-this-row from parent_to_children
        bary: list[tuple[float, str]] = []
        for c in children:
            parent_xs = []
            for p, kids in parent_to_children.items():
                if c in kids and p in pos:
                    parent_xs.append(pos[p][0])
            x_guess = sum(parent_xs) / len(parent_xs) if parent_xs else 0.0
            bary.append((x_guess, c))
        bary.sort()

        # Resolve collisions: enforce minimum spacing of 1.0 between adjacent xs
        last_x = -float("inf")
        for x_guess, c in bary:
            x = max(x_guess, last_x + 1.0)
            pos[c] = (x, float(max_depth - d))
            last_x = x

    return pos, max_depth + 1


def _max_row_width(
    pos: dict[str, tuple[float, float]],
) -> int:
    """Return the count of nodes in the most-populated row (depth band)."""
    if not pos:
        return 0
    by_row: dict[float, int] = {}
    for _, (_, y) in pos.items():
        by_row[y] = by_row.get(y, 0) + 1
    return max(by_row.values())


def _render_namespace_panel(
    ax,
    nodes: set[str],
    leaf_go_ids: set[str],
    parent_to_children: dict[str, set[str]],
    go_id_to_name: dict[str, str],
    namespace: str,
) -> dict[str, tuple[float, float]]:
    """Draw one namespace panel. Returns the positions used (for sizing decisions)."""
    pos, n_levels = _layout_namespace(nodes, parent_to_children)
    if not pos:
        ax.text(0.5, 0.5, f"(no terms in {namespace})", ha="center", va="center",
                transform=ax.transAxes, fontsize=_NODE_FONT_SIZE)
        ax.axis("off")
        return pos

    # Draw edges first (right-angle elbow: vertical drop from parent, then horizontal)
    for parent, kids in parent_to_children.items():
        if parent not in pos:
            continue
        for kid in kids:
            if kid not in pos:
                continue
            px, py = pos[parent]
            cx, cy = pos[kid]
            # Elbow: down to mid-y, then over, then down
            mid_y = (py + cy) / 2.0
            ax.plot([px, px], [py, mid_y], color=_EDGE_COLOR, linewidth=_EDGE_WIDTH, zorder=1)
            ax.plot([px, cx], [mid_y, mid_y], color=_EDGE_COLOR, linewidth=_EDGE_WIDTH, zorder=1)
            ax.plot([cx, cx], [mid_y, cy], color=_EDGE_COLOR, linewidth=_EDGE_WIDTH, zorder=1)

    # Draw nodes as text labels. BUG-004: wrap rather than truncate so full
    # GO term names remain readable in dense panels.
    for go_id, (x, y) in pos.items():
        is_leaf = go_id in leaf_go_ids
        weight = _LEAF_FONT_WEIGHT if is_leaf else _INTERNAL_FONT_WEIGHT
        raw_label = go_id_to_name.get(go_id, go_id)
        label = _wrap_label(raw_label)
        ax.text(
            x, y,
            label,
            ha="center", va="center",
            fontsize=_NODE_FONT_SIZE,
            fontweight=weight,
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", edgecolor=_EDGE_COLOR,
                      linewidth=0.5),
            zorder=5,
        )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.set_title(namespace.replace("_", " ").title(),
                 fontsize=_NODE_FONT_SIZE + 2, fontweight="bold", loc="left")
    ax.axis("off")
    return pos


def render_go_tree(
    groups: list[CategoryGroup],
    cohort: CohortData,
    obo_path: Path,
    output_stem: str,
    output_dir: Path,
    title: str = "",
    dpi: int = 300,
    font_family: str = "Arial",
) -> GoTreeResult:
    """Render a GO-term hierarchy figure to PDF, PNG, and SVG.

    One file set ({output_stem}_{namespace}.{pdf,png,svg}) is written per
    populated GO namespace. Each per-namespace figure auto-scales its width
    to the widest row in that namespace so labels fit without truncation
    (BUG-004).

    Args:
        groups: ordered list of CategoryGroups (the same list passed to render_dot_plot).
        cohort: the cohort, used to map term names to GO IDs.
        obo_path: local path to the GO OBO file.
        output_stem: base filename without extension. The namespace is
            appended to produce per-namespace files.
        output_dir: directory to write output files.
        title: optional figure-level title; namespace is appended in each file.
        dpi: PNG resolution.
        font_family: matplotlib font family for all text.

    Returns:
        GoTreeResult with per-namespace path dicts and summary counts.
    """
    if not output_dir.is_dir():
        raise OSError(f"Output directory does not exist: {output_dir}")

    parent_to_children, go_id_to_name, go_id_to_namespace, leaf_go_ids = build_go_tree(
        groups, cohort, obo_path
    )

    all_nodes = set(go_id_to_name.keys())
    n_internal = len(all_nodes) - len(leaf_go_ids)

    # Partition nodes by namespace; preserve canonical ordering.
    by_namespace: dict[str, set[str]] = {}
    for gid in all_nodes:
        ns = go_id_to_namespace.get(gid, "") or "unknown"
        by_namespace.setdefault(ns, set()).add(gid)

    ordered_namespaces = [
        ns for ns in _NAMESPACE_ORDER if ns in by_namespace
    ] + sorted(ns for ns in by_namespace if ns not in _NAMESPACE_ORDER)

    if not ordered_namespaces:
        # Defensive: should not happen because build_go_tree validates inputs.
        raise ValueError("No nodes found to render")

    plt.rcParams["font.family"] = font_family

    pdf_paths: dict[str, Path] = {}
    png_paths: dict[str, Path] = {}
    svg_paths: dict[str, Path] = {}

    for ns in ordered_namespaces:
        # Layout once to learn the dimensions, then size the canvas before drawing.
        pos, n_levels = _layout_namespace(by_namespace[ns], parent_to_children)
        widest_row = _max_row_width(pos)
        fig_width = max(_MIN_FIG_WIDTH, _PER_NODE_INCHES * max(widest_row, 1))
        fig_height = max(_MIN_FIG_HEIGHT, 1.0 + _HEIGHT_PER_LEVEL * max(n_levels, 1))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        _render_namespace_panel(
            ax,
            by_namespace[ns],
            leaf_go_ids,
            parent_to_children,
            go_id_to_name,
            ns,
        )

        if title:
            fig.suptitle(
                f"{title} -- {ns.replace('_', ' ').title()}",
                fontsize=12,
                fontweight="bold",
            )

        fig.patch.set_facecolor("white")
        plt.tight_layout()

        pdf_path = output_dir / f"{output_stem}_{ns}.pdf"
        png_path = output_dir / f"{output_stem}_{ns}.png"
        svg_path = output_dir / f"{output_stem}_{ns}.svg"

        try:
            fig.savefig(str(pdf_path), format="pdf", dpi=dpi, bbox_inches="tight")
            fig.savefig(str(png_path), format="png", dpi=dpi, bbox_inches="tight")
            fig.savefig(str(svg_path), format="svg", dpi=dpi, bbox_inches="tight")
        except Exception as e:
            plt.close(fig)
            raise OSError(f"Failed to write GO tree files: {e}") from e
        plt.close(fig)

        pdf_paths[ns] = pdf_path
        png_paths[ns] = png_path
        svg_paths[ns] = svg_path

    return GoTreeResult(
        pdf_paths=pdf_paths,
        png_paths=png_paths,
        svg_paths=svg_paths,
        n_leaf_terms=len(leaf_go_ids),
        n_internal_nodes=n_internal,
        n_namespaces=len(ordered_namespaces),
    )
