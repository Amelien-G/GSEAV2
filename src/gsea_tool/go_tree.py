"""Unit 11 -- GO Tree Rendering.

Renders a GO-term hierarchy figure paired with a dot plot. The "leaves" of
the tree are the GO terms displayed in the dot plot; "internal" nodes are
their is_a ancestors walked up to the namespace root. One panel per GO
namespace (BP / MF / CC) is stacked vertically when the leaf set spans
more than one namespace.
"""

from pathlib import Path
from dataclasses import dataclass

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


@dataclass
class GoTreeResult:
    """Metadata about a rendered GO-tree figure, for notes.md consumption."""
    pdf_path: Path
    png_path: Path
    svg_path: Path
    n_leaf_terms: int
    n_internal_nodes: int
    n_namespaces: int


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


def _render_namespace_panel(
    ax,
    nodes: set[str],
    leaf_go_ids: set[str],
    parent_to_children: dict[str, set[str]],
    go_id_to_name: dict[str, str],
    namespace: str,
) -> None:
    """Draw one namespace panel."""
    pos, n_levels = _layout_namespace(nodes, parent_to_children)
    if not pos:
        ax.text(0.5, 0.5, f"(no terms in {namespace})", ha="center", va="center",
                transform=ax.transAxes, fontsize=_NODE_FONT_SIZE)
        ax.axis("off")
        return

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

    # Draw nodes as text labels
    for go_id, (x, y) in pos.items():
        is_leaf = go_id in leaf_go_ids
        weight = _LEAF_FONT_WEIGHT if is_leaf else _INTERNAL_FONT_WEIGHT
        label = go_id_to_name.get(go_id, go_id)
        # Truncate very long labels for readability
        if len(label) > 40:
            label = label[:37] + "..."
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

    Args:
        groups: ordered list of CategoryGroups (the same list passed to render_dot_plot).
        cohort: the cohort, used to map term names to GO IDs.
        obo_path: local path to the GO OBO file.
        output_stem: base filename without extension.
        output_dir: directory to write output files.
        title: optional figure-level title.
        dpi: PNG resolution.
        font_family: matplotlib font family for all text.

    Returns:
        GoTreeResult with paths and summary counts.
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

    n_panels = len(ordered_namespaces)
    if n_panels == 0:
        # Defensive: should not happen because build_go_tree validates inputs.
        raise ValueError("No nodes found to render")

    plt.rcParams["font.family"] = font_family
    fig_height = max(3.0, sum(2.5 for _ in ordered_namespaces))
    fig_width = 10.0
    fig, axes = plt.subplots(
        nrows=n_panels, ncols=1,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for ax_row, ns in zip(axes, ordered_namespaces):
        ax = ax_row[0]
        _render_namespace_panel(
            ax,
            by_namespace[ns],
            leaf_go_ids,
            parent_to_children,
            go_id_to_name,
            ns,
        )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    pdf_path = output_dir / f"{output_stem}.pdf"
    png_path = output_dir / f"{output_stem}.png"
    svg_path = output_dir / f"{output_stem}.svg"

    try:
        fig.savefig(str(pdf_path), format="pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(str(png_path), format="png", dpi=dpi, bbox_inches="tight")
        fig.savefig(str(svg_path), format="svg", dpi=dpi, bbox_inches="tight")
    except Exception as e:
        raise OSError(f"Failed to write GO tree files: {e}") from e
    finally:
        plt.close(fig)

    return GoTreeResult(
        pdf_path=pdf_path,
        png_path=png_path,
        svg_path=svg_path,
        n_leaf_terms=len(leaf_go_ids),
        n_internal_nodes=n_internal,
        n_namespaces=n_panels,
    )
