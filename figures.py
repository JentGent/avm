"""Functions for making figures."""

import networkx as nx
import avm
from typing import Literal
import cmasher
import matplotlib.pyplot as plt
import numpy as np

def display(graph: nx.Graph, node_pos={}, title: str = None, cmap_min: float = None, cmap_max: float = None, color_is_flow: bool = True, label: Literal["name", "flow", "pressure", None] = "name", fill_by_flow = True):
    """Displays the graph.

    Args:
        graph: The graph.
        intranidal_nodes: The nodes in the nidus. This is only for color-coding.
        node_pos: A dictionary of node : [x, y] positions.
        title: Optional title for the graph.
        cmap_min: Optional minimum value for the color map range; if None, defaults to minimum flow/pressure value.
        cmap_max: Like `cmap_min`, optional maximum value.
        color_is_flow: If `True`, the color will represent flow, and the edge thickness will represent pressure; if `False`, it will be the other way around.
        label: Specifies whether/what to label the vessels.
        fill_by_flow: If `True` and `graph` backfilling has been calculated, nodes reached by the flow algorithm will be blue; otherwise, nodes filled by the pressure algorithm will be blue.
    """
    pos = nx.spring_layout(graph, 1, node_pos, node_pos.keys(), seed = 1)

    # Fix node positions
    xs, ys = [coords[0] for node, coords in pos.items() if node not in node_pos], [
        coords[1] for node, coords in pos.items() if node not in node_pos]
    if xs:
        mx, my = min(xs), min(ys)
        Mx, My = (max(xs) - mx) or 1, (max(ys) - my) or 1
        for node, coords in pos.items():
            if node not in node_pos.keys():
                pos[node] = [0.1 + 0.55 * (coords[0] - mx) / Mx, 1.15 * (coords[1] - my) / My - 0.75]

    # Nodes
    node_colors = {
        node: ("lightblue" if filled else "pink") for node, filled in graph.nodes("reached" if fill_by_flow else "filled")
    }
    nx.draw_networkx_nodes(graph, pos, node_size=10, node_color=[node_colors[node] for node in graph.nodes()])
    nx.draw_networkx_labels(graph, pos, labels = { node: "" if isinstance(node, (int, float)) else node for node in graph.nodes})

    # Edges
    pressures = [edge[2]["pressure"] for edge in graph.edges(data=True)]
    flows = [edge[2]["flow"] for edge in graph.edges(data=True)]
    min_pressure, max_pressure = min(pressures), max(pressures)
    min_flow, max_flow = min(flows), max(flows)

    if color_is_flow:
        edge_widths = [np.interp(edge[2]["pressure"], [min_pressure, max_pressure], [1, 1]) for edge in graph.edges(data=True)]
        edge_colors = [edge[2]["flow"] for edge in graph.edges(data=True)]
    else:
        edge_widths = [np.interp(edge[2]["flow"], [min_flow, max_flow], [1, 1]) for edge in graph.edges(data=True)]
        edge_colors = [edge[2]["pressure"] for edge in graph.edges(data=True)]
    nx.draw_networkx_edges(graph, pos, node_size = 10, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.cool if color_is_flow else cmasher.get_sub_cmap(plt.cm.Reds, 0.3, 1), edge_vmin=min(edge_colors) if cmap_min is None else cmap_min, edge_vmax=max(edge_colors) if cmap_max is None else cmap_max)
    edge_labels = False
    match label:
        case "name":
            edge_labels = {(edge[0], edge[1]): edge[2]["label"] for edge in graph.edges(data=True)}
        case "flow":
            edge_labels = {(edge[0], edge[1]): str(round(edge[2]["flow"], 3)) for edge in graph.edges(data=True)}
        case "pressure":
            edge_labels = {(edge[0], edge[1]): str(round(edge[2]["pressure"], 3)) for edge in graph.edges(data=True)}
    if edge_labels: nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    if color_is_flow:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=min(edge_colors) if cmap_min is None else cmap_min, vmax=max(edge_colors) if cmap_max is None else cmap_max))
        sm.set_array([])
        colorbar = plt.colorbar(sm, ax = plt.gca(), label="Flow (mL/min)")
    else:
        sm = plt.cm.ScalarMappable(cmap=cmasher.get_sub_cmap(plt.cm.Reds, 0.3, 1), norm=plt.Normalize(vmin=min_pressure if cmap_min is None else cmap_min, vmax=max_pressure if cmap_max is None else cmap_max))
        sm.set_array([])
        colorbar = plt.colorbar(sm, ax = plt.gca(), label="Pressure (mm Hg)")
    # plt.title(title)
    return colorbar


def bw(graph: nx.Graph, node_pos={}):
    """
    """
    pos = nx.spring_layout(graph, 1, node_pos, node_pos.keys(), seed = 1)

    # Fix node positions
    xs, ys = [coords[0] for node, coords in pos.items() if node not in node_pos], [
        coords[1] for node, coords in pos.items() if node not in node_pos]
    if xs:
        mx, my = min(xs), min(ys)
        Mx, My = (max(xs) - mx) or 1, (max(ys) - my) or 1
        for node, coords in pos.items():
            if node not in node_pos.keys():
                pos[node] = [0.1 + 0.55 * (coords[0] - mx) / Mx, 1.15 * (coords[1] - my) / My - 0.75]

    # Nodes
    nx.draw_networkx_nodes(graph, pos, node_size=10, node_color="black")

    # Edges
    nx.draw_networkx_edges(graph, pos, node_size = 10, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.cool if color_is_flow else cmasher.get_sub_cmap(plt.cm.Reds, 0.3, 1), edge_vmin=min(edge_colors) if cmap_min is None else cmap_min, edge_vmax=max(edge_colors) if cmap_max is None else cmap_max)
    edge_labels = False
    match label:
        case "name":
            edge_labels = {(edge[0], edge[1]): edge[2]["label"] for edge in graph.edges(data=True)}
        case "flow":
            edge_labels = {(edge[0], edge[1]): str(round(edge[2]["flow"], 3)) for edge in graph.edges(data=True)}
        case "pressure":
            edge_labels = {(edge[0], edge[1]): str(round(edge[2]["pressure"], 3)) for edge in graph.edges(data=True)}
    if edge_labels: nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    if color_is_flow:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=min(edge_colors) if cmap_min is None else cmap_min, vmax=max(edge_colors) if cmap_max is None else cmap_max))
        sm.set_array([])
        colorbar = plt.colorbar(sm, ax = plt.gca(), label="Flow (mL/min)")
    else:
        sm = plt.cm.ScalarMappable(cmap=cmasher.get_sub_cmap(plt.cm.Reds, 0.3, 1), norm=plt.Normalize(vmin=min_pressure if cmap_min is None else cmap_min, vmax=max_pressure if cmap_max is None else cmap_max))
        sm.set_array([])
        colorbar = plt.colorbar(sm, ax = plt.gca(), label="Pressure (mm Hg)")
    # plt.title(title)
    return colorbar
