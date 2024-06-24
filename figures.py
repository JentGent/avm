"""Functions for making figures."""

import networkx as nx
import avm
from typing import Literal
import cmasher
import matplotlib
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


def display_filling(graph: nx.Graph, node_pos={}, title: str = None, cmap_min: float = None, cmap_max: float = None, color: Literal["flow", "pressure", "filling"] = "flow", label: Literal["name", "flow", "pressure", None] = "name", fill_by_flow = True):
    """Displays the graph.

    Args:
        graph: The graph.
        intranidal_nodes: The nodes in the nidus. This is only for color-coding.
        node_pos: A dictionary of node : [x, y] positions.
        title: Optional title for the graph.
        cmap_min: Optional minimum value for the color map range; if None, defaults to minimum flow/pressure value.
        cmap_max: Like `cmap_min`, optional maximum value.
        color: If `"flow"`, the color will represent flow, and the edge thickness will represent pressure; if `"pressure"`, it will be the other way around. If `"filling"`, vessels that the injection reached will be colored.
        label: Specifies whether/what to label the vessels.
        fill_by_flow: If `True` and `graph` backfilling has been calculated, nodes reached by the flow algorithm will be blue; otherwise, nodes filled by the pressure algorithm will be blue.
    """
    filled_color = plt.cm.Blues(0.99)
    unfilled_color = (1, 0.9, 0.9)
    pos = nx.spring_layout(graph, 1, node_pos, node_pos.keys(), seed = 1)

    # Fix node positions
    xs, ys = [coords[0] for node, coords in pos.items() if node not in node_pos], [coords[1] for node, coords in pos.items() if node not in node_pos]

    if xs:

        mx, my = min(xs), min(ys)
        Mx, My = (max(xs) - mx) or 1, (max(ys) - my) or 1

        for node, coords in pos.items():
            if node not in node_pos.keys():
                pos[node] = [0.1 + 0.55 * (coords[0] - mx) / Mx, 1.15 * (coords[1] - my) / My - 0.75]

    # Edges
    edge_order = list(graph.edges(data=True))
    edge_order = sorted(edge_order, key=lambda edge: 2 if edge[2]["type"] == avm.vessel.fistulous else 1 if "reached" in edge[2] and edge[2]["reached"] else 0)
    edge_colors = [filled_color if data.get("reached") else unfilled_color for u, v, data in edge_order]
    edges = nx.draw_networkx_edges(graph, pos, node_size=0, edgelist=edge_order, edge_color=edge_colors, edge_cmap=plt.cm.Blues, edge_vmin=min(edge_colors) if cmap_min is None else cmap_min, edge_vmax=max(edge_colors) if cmap_max is None else cmap_max)

    for i, edge in enumerate(edges):
        is_fistulous = edge_order[i][2]["type"] == avm.vessel.fistulous
        is_intranidal = edge_order[i][2]["type"] in [avm.vessel.plexiform, avm.vessel.fistulous]
        edge.set(arrowstyle=matplotlib.patches.ArrowStyle('simple', head_length=1.5 if is_fistulous else 1 if is_intranidal else 2, head_width=1.5 if is_fistulous else 1 if is_intranidal else 2, tail_width=0.6 if is_fistulous else 0.4 if is_intranidal else 0.8))
        if "occluded" in edge_order[i][2] and edge_order[i][2]["occluded"]:
            edge.set_color("white")
            graph.edges[edge_order[i][0], edge_order[i][1]]["font"] = { "size": 30 }
        edge.set_edgecolor("white" if edge_order[i][2]["type"] == avm.vessel.fistulous else "black")
        edge.set_linewidth(2 if edge_order[i][2]["type"] == avm.vessel.fistulous else 0.3)
    
    edge_labels = {(edge[0], edge[1]): edge[2]["label"] for edge in edge_order if "occluded" not in edge[2] or not edge[2]["occluded"]}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=15)
    edge_labels = {(edge[0], edge[1]): "✖" for edge in edge_order if "occluded" in edge[2] and edge[2]["occluded"]}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=30, rotate=False, bbox={ "alpha": 0 }, verticalalignment="center_baseline")
    plt.text(200 - 0, -500 - 10, "AF1", fontsize=15, horizontalalignment="right", verticalalignment="top")
    plt.text(200 - 0, -370 - 10, "AF2", fontsize=15, horizontalalignment="right", verticalalignment="top")
    plt.text(200 - 0, -200 + 10, "AF3", fontsize=15, horizontalalignment="right", verticalalignment="bottom")
    plt.text(350 - 10, -150, "AF4", fontsize=15, horizontalalignment="right", verticalalignment="center")
        
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    colorbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
    colorbar.set_label("Flow (mL/min)", size=20, labelpad=20)



def display_flow(graph: nx.Graph, node_pos={}, title: str = None, cmap_min: float = 0, cmap_max: float = 5, color: Literal["flow", "pressure", "filling"] = "flow", label: Literal["name", "flow", "pressure", None] = "name", fill_by_flow = True):
    """Displays the graph.

    Args:
        graph: The graph.
        intranidal_nodes: The nodes in the nidus. This is only for color-coding.
        node_pos: A dictionary of node : [x, y] positions.
        title: Optional title for the graph.
        cmap_min: Optional minimum value for the color map range; if None, defaults to minimum flow/pressure value.
        cmap_max: Like `cmap_min`, optional maximum value.
        color: If `"flow"`, the color will represent flow, and the edge thickness will represent pressure; if `"pressure"`, it will be the other way around. If `"filling"`, vessels that the injection reached will be colored.
        label: Specifies whether/what to label the vessels.
        fill_by_flow: If `True` and `graph` backfilling has been calculated, nodes reached by the flow algorithm will be blue; otherwise, nodes filled by the pressure algorithm will be blue.
    """
    pos = nx.spring_layout(graph, 1, node_pos, node_pos.keys(), seed = 1)

    # Fix node positions
    xs, ys = [coords[0] for node, coords in pos.items() if node not in node_pos], [coords[1] for node, coords in pos.items() if node not in node_pos]

    if xs:

        mx, my = min(xs), min(ys)
        Mx, My = (max(xs) - mx) or 1, (max(ys) - my) or 1

        for node, coords in pos.items():
            if node not in node_pos.keys():
                pos[node] = [0.1 + 0.55 * (coords[0] - mx) / Mx, 1.15 * (coords[1] - my) / My - 0.75]

    # Edges
    edge_order = list(graph.edges(data=True))
    edge_order = sorted(edge_order, key=lambda edge: 0 if edge[2]["type"] == avm.vessel.fistulous else -1)
    edge_colors = [edge[2]["flow"] if "flow" in edge[2] else 0 for edge in edge_order]
    edges = nx.draw_networkx_edges(graph, pos, node_size=0, edgelist=edge_order, edge_color=edge_colors, edge_cmap=plt.cm.Blues, edge_vmin=min(edge_colors) if cmap_min is None else cmap_min, edge_vmax=max(edge_colors) if cmap_max is None else cmap_max)

    for i, edge in enumerate(edges):
        is_fistulous = edge_order[i][2]["type"] == avm.vessel.fistulous
        is_intranidal = edge_order[i][2]["type"] in [avm.vessel.plexiform, avm.vessel.fistulous]
        edge.set(arrowstyle=matplotlib.patches.ArrowStyle('simple', head_length=1.5 if is_fistulous else 1 if is_intranidal else 2, head_width=1.5 if is_fistulous else 1 if is_intranidal else 2, tail_width=0.6 if is_fistulous else 0.4 if is_intranidal else 0.8))
        if "occluded" in edge_order[i][2] and edge_order[i][2]["occluded"]:
            edge.set_color("white")
            graph.edges[edge_order[i][0], edge_order[i][1]]["font"] = { "size": 30 }
        edge.set_edgecolor("white" if edge_order[i][2]["type"] == avm.vessel.fistulous else "black")
        edge.set_linewidth(2 if edge_order[i][2]["type"] == avm.vessel.fistulous else 0.3)
    
    edge_labels = {(edge[0], edge[1]): edge[2]["label"] for edge in edge_order if "occluded" not in edge[2] or not edge[2]["occluded"]}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=15)
    edge_labels = {(edge[0], edge[1]): "✖" for edge in edge_order if "occluded" in edge[2] and edge[2]["occluded"]}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=30, rotate=False, bbox={ "alpha": 0 }, verticalalignment="center_baseline")
    plt.text(200 - 0, -500 - 10, "AF1", fontsize=15, horizontalalignment="right", verticalalignment="top")
    plt.text(200 - 0, -370 - 10, "AF2", fontsize=15, horizontalalignment="right", verticalalignment="top")
    plt.text(200 - 0, -200 + 10, "AF3", fontsize=15, horizontalalignment="right", verticalalignment="bottom")
    plt.text(350 - 10, -150, "AF4", fontsize=15, horizontalalignment="right", verticalalignment="center")
        

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_colors) if cmap_min is None else cmap_min, vmax=max(edge_colors) if cmap_max is None else cmap_max))
    sm.set_array([])
    colorbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
    colorbar.set_label("Flow (mL/min)", size=20, labelpad=20)
    colorbar.ax.tick_params(labelsize=20, length=10)
    ticks = colorbar.ax.get_yticklabels()
    ticks[-1].set_text("≥" + ticks[-1].get_text())
    colorbar.ax.set_yticklabels(ticks)



def display_pressure(graph: nx.Graph, node_pos={}, title: str = None, cmap_min: float = 0, cmap_max: float = 20, color: Literal["flow", "pressure", "filling"] = "flow", label: Literal["name", "flow", "pressure", None] = "name", fill_by_flow = True):
    """Displays the graph.

    Args:
        graph: The graph.
        intranidal_nodes: The nodes in the nidus. This is only for color-coding.
        node_pos: A dictionary of node : [x, y] positions.
        title: Optional title for the graph.
        cmap_min: Optional minimum value for the color map range; if None, defaults to minimum flow/pressure value.
        cmap_max: Like `cmap_min`, optional maximum value.
        color: If `"flow"`, the color will represent flow, and the edge thickness will represent pressure; if `"pressure"`, it will be the other way around. If `"filling"`, vessels that the injection reached will be colored.
        label: Specifies whether/what to label the vessels.
        fill_by_flow: If `True` and `graph` backfilling has been calculated, nodes reached by the flow algorithm will be blue; otherwise, nodes filled by the pressure algorithm will be blue.
    """
    pos = nx.spring_layout(graph, 1, node_pos, node_pos.keys(), seed = 1)

    # Fix node positions
    xs, ys = [coords[0] for node, coords in pos.items() if node not in node_pos], [coords[1] for node, coords in pos.items() if node not in node_pos]

    if xs:

        mx, my = min(xs), min(ys)
        Mx, My = (max(xs) - mx) or 1, (max(ys) - my) or 1

        for node, coords in pos.items():
            if node not in node_pos.keys():
                pos[node] = [0.1 + 0.55 * (coords[0] - mx) / Mx, 1.15 * (coords[1] - my) / My - 0.75]

    # Edges
    edge_order = list(graph.edges(data=True))
    edge_order = sorted(edge_order, key=lambda edge: 0 if edge[2]["type"] == avm.vessel.fistulous else -1)
    edge_colors = [edge[2]["pressure"] for edge in edge_order]
    
    cmap_min, cmap_max = min(edge_colors) if cmap_min is None else cmap_min, max(edge_colors) if cmap_max is None else cmap_max
    edges = nx.draw_networkx_edges(graph, pos, node_size=0, edgelist=edge_order, edge_color=edge_colors, edge_cmap=plt.cm.Reds, edge_vmin=cmap_min, edge_vmax=cmap_max)

    for i, edge in enumerate(edges):
        is_fistulous = edge_order[i][2]["type"] == avm.vessel.fistulous
        is_intranidal = edge_order[i][2]["type"] in [avm.vessel.plexiform, avm.vessel.fistulous]
        edge.set(arrowstyle=matplotlib.patches.ArrowStyle('simple', head_length=1.5 if is_fistulous else 1 if is_intranidal else 2, head_width=1.5 if is_fistulous else 1 if is_intranidal else 2, tail_width=0.6 if is_fistulous else 0.4 if is_intranidal else 0.8))
        edge.set_edgecolor("white" if edge_order[i][2]["type"] == avm.vessel.fistulous else "black")
        edge.set_linewidth(2 if edge_order[i][2]["type"] == avm.vessel.fistulous else 0.3)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(edge_colors) if cmap_min is None else cmap_min, vmax=cmap_max))
    sm.set_array([])
    colorbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
    colorbar.set_label("Pressure (mmHg)", size=20, labelpad=20)
    colorbar.ax.tick_params(labelsize=20, length=10)
    ticks = colorbar.ax.get_yticklabels()
    ticks[-1].set_text("≥" + ticks[-1].get_text())
    colorbar.ax.set_yticklabels(ticks)

    edge_labels = False

    match label:

        case "name":
            edge_labels = {(edge[0], edge[1]): edge[2]["label"] for edge in edge_order}

        case "flow":
            edge_labels = {(edge[0], edge[1]): str(round(edge[2]["flow"], 3)) for edge in edge_order}

        case "pressure":
            edge_labels = {(edge[0], edge[1]): str(round(edge[2]["pressure"], 3)) for edge in edge_order}

    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=15)
        # nx.draw_networkx_labels(graph, pos)
        plt.text(200 - 0, -500 - 10, "AF1", fontsize=15, horizontalalignment="right", verticalalignment="top")
        plt.text(200 - 0, -370 - 10, "AF2", fontsize=15, horizontalalignment="right", verticalalignment="top")
        plt.text(200 - 0, -200 + 10, "AF3", fontsize=15, horizontalalignment="right", verticalalignment="bottom")
        plt.text(350 - 10, -150, "AF4", fontsize=15, horizontalalignment="right", verticalalignment="center")

