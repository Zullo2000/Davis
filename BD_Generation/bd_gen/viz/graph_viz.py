"""Visualization for bubble diagram graphs.

Draws bubble diagrams as labeled networkx graphs with room-type-colored
nodes and spatially-labeled edges. Supports single graphs and grids.

Compatible with any matplotlib backend (inline for notebooks, Agg for scripts).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

from bd_gen.data.vocab import EDGE_TYPES

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# Room type → color mapping (13 types)
ROOM_COLORS: dict[int, str] = {
    0: "#FF6B6B",   # LivingRoom — red
    1: "#4ECDC4",   # MasterRoom — teal
    2: "#FFE66D",   # Kitchen — yellow
    3: "#95E1D3",   # Bathroom — light green
    4: "#F38181",   # DiningRoom — salmon
    5: "#AA96DA",   # ChildRoom — purple
    6: "#6C5B7B",   # StudyRoom — dark purple
    7: "#C06C84",   # SecondRoom — pink
    8: "#F8B500",   # GuestRoom — gold
    9: "#00B8A9",   # Balcony — aqua
    10: "#F6416C",  # Entrance — hot pink
    11: "#A8A8A8",  # Storage — gray
    12: "#555555",  # Wall-in — dark gray
}

# Short labels for compact display
_SHORT_LABELS: dict[int, str] = {
    0: "Liv",
    1: "Mstr",
    2: "Kit",
    3: "Bath",
    4: "Din",
    5: "Chld",
    6: "Stdy",
    7: "2nd",
    8: "Gst",
    9: "Bal",
    10: "Ent",
    11: "Stor",
    12: "Wall",
}


def draw_bubble_diagram(
    graph_dict: dict,
    ax: Axes | None = None,
    node_size: int = 800,
    font_size: int = 8,
    title: str | None = None,
) -> Figure:
    """Draw a single bubble diagram as a labeled graph.

    Args:
        graph_dict: Dict with ``num_rooms``, ``node_types``,
            ``edge_triples`` keys (output of ``detokenize``).
        ax: Matplotlib axes to draw on. If ``None``, a new figure
            is created.
        node_size: Size of node circles.
        font_size: Font size for labels.
        title: Optional title for the subplot.

    Returns:
        The matplotlib Figure containing the drawing.
    """
    num_rooms = graph_dict["num_rooms"]
    node_types = graph_dict["node_types"]
    edge_triples = graph_dict["edge_triples"]

    # Build networkx graph
    G = nx.Graph()
    for k in range(num_rooms):
        G.add_node(k, room_type=node_types[k])

    for i, j, rel in edge_triples:
        G.add_edge(i, j, rel_type=rel)

    # Create figure if needed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        created_fig = True
    else:
        fig = ax.get_figure()

    if num_rooms == 0:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                transform=ax.transAxes, fontsize=font_size)
        if title:
            ax.set_title(title, fontsize=font_size + 1)
        ax.axis("off")
        return fig

    # Layout
    if num_rooms == 1:
        pos = {0: (0.5, 0.5)}
    else:
        pos = nx.spring_layout(G, seed=42, k=2.0 / math.sqrt(num_rooms))

    # Node colors
    node_colors = [ROOM_COLORS.get(node_types[k], "#CCCCCC") for k in range(num_rooms)]

    # Node labels (short names)
    labels = {
        k: _SHORT_LABELS.get(node_types[k], str(node_types[k]))
        for k in range(num_rooms)
    }

    # Draw
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=labels,
        font_size=font_size,
        font_weight="bold",
    )

    if edge_triples:
        edge_labels = {}
        for i, j, rel in edge_triples:
            if 0 <= rel < len(EDGE_TYPES):
                edge_labels[(i, j)] = EDGE_TYPES[rel]
            else:
                edge_labels[(i, j)] = str(rel)

        nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, alpha=0.7)
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax,
            edge_labels=edge_labels,
            font_size=max(font_size - 2, 5),
            font_color="darkblue",
        )

    if title:
        ax.set_title(title, fontsize=font_size + 1)
    ax.axis("off")

    if created_fig:
        fig.tight_layout()

    return fig


def draw_bubble_diagram_grid(
    graph_dicts: list[dict],
    ncols: int = 4,
    figsize: tuple[int, int] | None = None,
    titles: list[str] | None = None,
) -> Figure:
    """Draw a grid of bubble diagrams.

    Args:
        graph_dicts: List of graph dicts to visualize.
        ncols: Number of columns in the grid.
        figsize: Figure size ``(width, height)``. Defaults to auto-sized.
        titles: Optional list of titles, one per graph.

    Returns:
        The matplotlib Figure containing the grid.
    """
    n = len(graph_dicts)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.text(0.5, 0.5, "No samples", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes for uniform iteration
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx < n:
            t = titles[idx] if titles and idx < len(titles) else f"Sample {idx}"
            draw_bubble_diagram(graph_dicts[idx], ax=ax, title=t,
                                node_size=500, font_size=7)
        else:
            ax.axis("off")

    fig.tight_layout()
    return fig
