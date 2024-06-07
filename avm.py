import random
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Literal
import math
from collections import deque
import cmasher
from skimage.filters import threshold_otsu

# VISCOSITY is the viscosity of blood in Poise.
VISCOSITY = 0.035

# If PREDEFINED_RESISTANCE is True, radius and length data is ignored, and the given resistance values are used; otherwise, radius and length are plugged into the Hagen–Poiseuille equation: resistance = (8 * length * viscosity) / (pi r^4)
PREDEFINED_RESISTANCE = True

# x mmHg = x * MMHG_TO_DYN_PER_SQUARE_CM dyn/cm^2.
MMHG_TO_DYN_PER_SQUARE_CM = 1333.22

# x dyn s/cm^5 = x * DYN_S_PER_CM5_TO_MMHG_MIN_PER_ML mmHg min/mL.
DYN_S_PER_CM5_TO_MMHG_MIN_PER_ML = 0.00001250102

# ABS_PRESSURE indicates whether or not to display calculated absolute pressures (NON-FUNCTIONAL).
ABS_PRESSURE = False

# SOLVE_MODE indicates what solver to use when solving Kirchhoff's circuit equations for flow
# 1 is numpy.linalg.lstsq - very accurate, but slow
# 2 is scipy.sparse.spsolve - less accurate, but fast
# simulate_batch can only use numpy.linalg.lstsq
SOLVE_MODE = 1

# ROUND_DECIMALS specifies how many decimals to round the calculated stats to.
ROUND_DECIMALS = 4

# vessel has vessel type IDs.
class vessel:
    other = 0
    fistulous = 1
    plexiform = 2
    feeder = 3
    drainer = 4

# NODE_POS_TEMPLATE lists the positions of extranidal nodes in the graph.
NODE_POS_TEMPLATE = {
    1: [10, -200],
    2: [50, -250],
    3: [75, -300],
    4: [50, -200],
    5: [100, -100],
    6: [150, -275],
    7: [150, -120],
    8: [500, -120],
    9: [350, -100],
    10: [550, -100],
    11: [580, -300],
    12: [580, -590],
    13: [300, -590],
    "SP": [10, -590],
    "AF1": [200, -500],
    "AF2": [200, -370],
    "AF3": [200, -200],
    "AF4": [350, -150],
    "DV1": [500, -543],
    "DV2": [500, -350],
    "DV3": [500, -250],
}

# VESSELS_TEMPLATE elements are formatted like [first node, second node, radius (cm), length (cm), resistance (mmHg * min / mL), label, type (optional)].
VESSELS_TEMPLATE = [
    [13, "SP", 0.75, 10, 3.2, "superior vena cava"],
    ["SP", 1, 1, 10, 1, "aortic arch"],
    [1, 2, 0.35, 10, 67.9, "SCA"],
    [1, 4, 0.35, 10, 67.9, "CCA"],
    [4, 5, 0.2, 10, 637.5, "ECA"],
    [5, 9, 0.01, 0.1, 1000000, ""],
    [9, 10, 0.01, 0.1, 1000000, ""],
    [10, 11, 0.125, 10, 4177.9, ""],
    [12, 13, 0.4, 20, 79.7, "jugular veins"],
    [4, 6, 0.25, 20, 522, "ICA"],
    [2, 3, 0.15, 25, 5037, "VA"],
    [6, 7, 0.1, 10, 10200, ""],
    [7, 8, 0.01, 0.1, 1000000, ""],
    [8, 11, 0.125, 10, 4177.9, ""],
    [11, 12, 0.25, 10, 261, "dural venous sinuses"],

    [3, "AF1", 0.125, 5.2, 2210, "PCA", vessel.feeder],
    [6, "AF2", 0.15, 3.7, 745.5, "MCA", vessel.feeder],
    [6, "AF3", 0.025, 3.7, 15725000, "ACA", vessel.feeder],
    [9, "AF4", 0.0125, 3, 12750000, "TFA", vessel.feeder],

    ["DV1", 11, 0.25, 5, 130.5, "", vessel.drainer],
    ["DV2", 11, 0.25, 5, 130.5, "", vessel.drainer],
    ["DV3", 11, 0.25, 5, 130.5, "", vessel.drainer],
]

for v in VESSELS_TEMPLATE:
    v[4] *= DYN_S_PER_CM5_TO_MMHG_MIN_PER_ML


def calc_resistance(radius: float, length: float) -> float:
    """Uses the Hagen-Poiseuille equation to calculate resistance: resistance = (8 * length * viscosity) / (pi r^4)
    
    Args:
        radius: The radius of the blood vessel in cm.
        length: The length of the blood vessel in cm.
    
    Returns:
        resistance: The resistance of the blood vessel in mmHg * min / mL
    """
    return 8 * length * VISCOSITY / np.pi / (radius ** 4) * DYN_S_PER_CM5_TO_MMHG_MIN_PER_ML


def pressure_forward(graph: nx.DiGraph, node, pressure, pressures: dict[str, float]):
    """Steps through the graph to calculate absolute pressure.

    Args:
        graph: The graph.
        node: The name of the original node.
        pressure The pressure of the original node.
        pressures: A dictionary of known node : pressure values.
    """
    nx.set_node_attributes(graph, {node: pressure}, "pressure")
    edges = graph.edges([node])
    nx.set_edge_attributes(graph, {
                           edge: pressure for edge in edges if graph.edges[edge]["start"] == node}, "pressure")
    
    for edge in edges:

        next_node = edge[1]

        if "pressure" not in graph.nodes[next_node] and graph.edges[edge]["start"] == node:
            next_pressure = pressure + graph.edges[edge]["pressure"] * (-1 if graph.edges[edge]["start"] == node else 1)
            pressure_forward(graph, next_node, next_pressure, pressures)


def calc_pressures(graph: nx.Graph, pressures: dict[str, float]):
    """Begins the calculation of absolute pressure through the graph.

    Args:
        graph: The graph.
        pressures: A dictionary of known node : pressure values.
    """
    pressure_forward(graph, "SP", pressures["SP"], pressures)


def pressure_change(first_digraph: nx.DiGraph, second_digraph: nx.DiGraph) -> list[float]:
    """Computes the difference in pressure from graph 1 to graph 2."""
    percent_differences = []

    for (node1, node2, pressure) in first_digraph.edges.data("pressure"):

        if node2 in second_digraph[node1]:
            percent_differences.append((second_digraph[node1][node2]["pressure"] - pressure) / pressure * 100)

        else:
            percent_differences.append((-second_digraph[node2][node1]["pressure"] - pressure) / pressure * 100)

    return percent_differences


def display(graph: nx.Graph, node_pos={}, title: str = None, cmap_min: float = None, cmap_max: float = None, color: Literal["flow", "pressure", "filling"] = "flow", label: Literal["name", "flow", "pressure", None] = "name", fill_by_flow = True):
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

    # Nodes
    node_colors = {
        node: ("lightblue" if filled else "pink") for node, filled in graph.nodes("reached" if fill_by_flow else "filled")
    }
    
    nx.draw_networkx_nodes(graph, pos, node_size=100, node_color=[node_colors[node] for node in graph.nodes()])
    nx.draw_networkx_labels(graph, pos, labels = { node: "" if isinstance(node, (int, float)) else node for node in graph.nodes})

    # Edges
    pressures = [edge[2]["pressure"] for edge in graph.edges(data=True)]
    flows = [edge[2]["flow"] for edge in graph.edges(data=True)]
    min_pressure, max_pressure = min(pressures), max(pressures)
    min_flow, max_flow = min(flows), max(flows)

    match color:

        case "flow":
            edge_colors = [edge[2]["flow"] for edge in graph.edges(data=True)]
            edge_widths = [np.interp(edge[2]["pressure"], [min_pressure, max_pressure], [1, 10]) for edge in graph.edges(data=True)]
            nx.draw_networkx_edges(graph, pos, node_size = 300, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.cool, edge_vmin=min(edge_colors) if cmap_min is None else cmap_min, edge_vmax=max(edge_colors) if cmap_max is None else cmap_max)

        case "pressure":
            edge_colors = [edge[2]["pressure"] for edge in graph.edges(data=True)]
            edge_widths = [np.interp(edge[2]["flow"], [min_flow, max_flow], [1, 10]) for edge in graph.edges(data=True)]
            nx.draw_networkx_edges(graph, pos, node_size = 300, width=edge_widths, edge_color=edge_colors, edge_cmap=cmasher.get_sub_cmap(plt.cm.Reds, 0.3, 1), edge_vmin=min(edge_colors) if cmap_min is None else cmap_min, edge_vmax=max(edge_colors) if cmap_max is None else cmap_max)

        case "filling":
            edge_colors = [edge[2]["reached"] if "reached" in edge[2] else 0 for edge in graph.edges(data=True)]
            edge_widths = 1
            nx.draw_networkx_edges(graph, pos, node_size = 100, width=edge_widths, edge_color=edge_colors, edge_cmap=cmasher.get_sub_cmap(plt.cm.Blues, 0.3, 1), edge_vmin=0, edge_vmax=1)

    edge_labels = False

    match label:

        case "name":
            edge_labels = {(edge[0], edge[1]): edge[2]["label"] for edge in graph.edges(data=True)}

        case "flow":
            edge_labels = {(edge[0], edge[1]): str(round(edge[2]["flow"], 3)) for edge in graph.edges(data=True)}

        case "pressure":
            edge_labels = {(edge[0], edge[1]): str(round(edge[2]["pressure"], 3)) for edge in graph.edges(data=True)}

    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels)

    match color:

        case "flow":
            sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=min(edge_colors) if cmap_min is None else cmap_min, vmax=max(edge_colors) if cmap_max is None else cmap_max))
            sm.set_array([])
            colorbar = plt.colorbar(sm, ax = plt.gca(), label="Flow (mL/min)")

        case "pressure":
            sm = plt.cm.ScalarMappable(cmap=cmasher.get_sub_cmap(plt.cm.Reds, 0.3, 1), norm=plt.Normalize(vmin=min_pressure if cmap_min is None else cmap_min, vmax=max_pressure if cmap_max is None else cmap_max))
            sm.set_array([])
            colorbar = plt.colorbar(sm, ax = plt.gca(), label="Pressure (mm Hg)")

        case "filling":
            sm = plt.cm.ScalarMappable(cmap=cmasher.get_sub_cmap(plt.cm.Blues, 0.3, 1), norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            colorbar = plt.colorbar(sm, ax = plt.gca(), label="Filling")

    # plt.title(title)
    return colorbar


def add_edge_to_graph(graph: nx.Graph, start, end, radius = 0.01325, length = 0.5, resistance = None, label = "", type = vessel.plexiform):
    graph.add_edge(
        start,
        end,
        start=start,
        end=end,
        radius=radius,
        length=length,
        resistance=calc_resistance(radius, length) if resistance is None else resistance,
        label=label,
        type=type
    )


def edges_to_graph(edges: list) -> nx.Graph:
    """Converts a list of edges to a graph in the right format ([starting node, ending node, radius (cm), length (cm), resistance (mmHg * min / mL), label]).

    Args:
        edges: List of edges.

    Returns:
        graph: NetworkX graph generated from edges.
    """
    graph = nx.Graph()
    graph.add_edges_from([(
        edge[0],
        edge[1],
        {
            "start": edge[0],
            "end": edge[1],
            "radius": edge[2],
            "length": edge[3],
            "resistance": edge[4] if PREDEFINED_RESISTANCE and edge[4] else calc_resistance(edge[2], edge[3]),
            "label": edge[5],
            "type": edge[6] if len(edge) > 6 else vessel.other
        }) for edge in edges
    ])
    return graph


def simulate_batch(graph: nx.Graph, intranidal_nodes: list, p_exts: list[dict[str, float]], return_error: bool = False) -> tuple[np.ndarray, np.ndarray, list[tuple], list[nx.DiGraph], float]:
    """Simulates the nidus with a batch of different sets of pressure values.

    Args:
        graph: Graph from `edges_to_graph()` or `generate_nidus()`.
        intranidal_nodes: List of nodes in the nidus to generate edges between.
        p_ext: A dictionary of known node : pressure values in mmHg.
        num_nidi: Number of nidi to simulate.
        num_expected_edges: Expected number of edges to generate in the nidus.
        return_error: Whether to return the absolute pressure error from solving Kirchoff's circuit equations.

    Returns:
        flows: The calculated flow values in mL/min.
        pressures: The calculated pressure gradients in mmHg.
        all_edges: The list of edges in the same order as the flow and pressure arrays.
        graphs: The directed graphs of flow (mL/min) and pressure (mmHg).
        error (if `return_error` is True): Error.
    """
    all_edges = [(edge[2]["start"], edge[2]["end"]) for edge in graph.edges(data=True)]
    flows, error = calc_flow_batch(graph, all_edges, p_exts)
    
    # flow = difference in pressure between ends / resistance
    # Hagen-Poiseuille equation
    pressures = flows * np.tile(np.array([graph.edges[edge]["resistance"] for edge in all_edges]), (len(p_exts), 1)).T

    graphs = []

    for j, p_ext in enumerate(p_exts):

        result = graph.copy()

        nx.set_edge_attributes(result, {edge: flows[i][j] for i, edge in enumerate(all_edges)}, "flow")
        nx.set_edge_attributes(result, {edge: pressures[i][j] for i, edge in enumerate(all_edges)}, "pressure")

        result = nx.DiGraph([(
            edge[2]["end" if edge[2]["flow"] < 0 else "start"],
            edge[2]["start" if edge[2]["flow"] < 0 else "end"],
            {
                "radius": edge[2]["radius"],
                "length": edge[2]["length"],
                "resistance": edge[2]["resistance"],
                "label": edge[2]["label"],
                "flow": abs(edge[2]["flow"]),
                "pressure": abs(edge[2]["pressure"]),
                "type": edge[2]["type"]
            }) for edge in result.edges(data=True)
        ])

        graphs.append(result)
    
    if return_error:
        return flows, pressures, all_edges, graphs, error
    
    return flows, pressures, all_edges, graphs


def simulate(graph: nx.Graph, intranidal_nodes: list, p_ext: dict[str, float], return_error: bool = False) -> tuple[np.ndarray, np.ndarray, list[tuple], nx.Graph, float]:
    """Simulates the nidus.

    Args:
        graph: Graph from `edges_to_graph()` or `generate_nidus()`.
        intranidal_nodes: List of nodes in the nidus to generate edges between.
        p_ext: A dictionary of known node : pressure values in mmHg.
        num_nidi: Number of nidi to simulate.
        num_expected_edges: Expected number of edges to generate in the nidus.
        return_error: Whether to return the absolute pressure error from solving Kirchoff's circuit equations.

    Returns:
        flow: The calculated flow values in mL/min.
        pressure: The calculated pressure gradients in mmHg.
        all_edges: The list of edges in the same order as the flow and pressure arrays.
        graph: The directed graph of flow (mL/min) and pressure (mmHg).
        error (if `return_error` is True): Error.
    """
    flow, pressure, all_edges, error = get_Q_and_P(graph, p_ext)

    graph = nx.DiGraph([(
        edge[2]["end" if edge[2]["flow"] < 0 else "start"],
        edge[2]["start" if edge[2]["flow"] < 0 else "end"],
        {
            "radius": edge[2]["radius"],
            "length": edge[2]["length"],
            "resistance": edge[2]["resistance"],
            "label": edge[2]["label"],
            "flow": abs(edge[2]["flow"]),
            "pressure": abs(edge[2]["pressure"]),
            "type": edge[2]["type"]
        }) for edge in graph.edges(data=True)
    ])

    if return_error:
        return flow, pressure, all_edges, graph, error
    
    return flow, pressure, all_edges, graph


def get_nidus(digraph: nx.DiGraph) -> nx.DiGraph:
    """Returns a new graph that consists only of fistulous and plexiform edges of the given graph.
    
    Args:
        digraph
    
    Returns:
        nidus
    """
    nidus = nx.DiGraph([edge for edge in digraph.edges(data=True) if edge[2]["type"] in (vessel.fistulous, vessel.plexiform)])

    for node, attrs in digraph.nodes(data=True):
        if node in nidus.nodes:
            nidus.add_node(node, **attrs)

    return nidus

def calc_filling_bfs(digraph: nx.DiGraph, intranidal_nodes, num_intranidal_vessels, injection_location, no_injection_digraph: nx.DiGraph = None, allow_upstream = False) -> float:
    """Calculates the percent of the nidus that an injection reaches.
    
    Args:
        digraph: Graph after complete simulation (with calculated flow and pressure attributes).
        intranidal_nodes: List of nodes in the nidus (the algorithm will only traverse vessels whose ends are in `intranidal_nodes`).
        injection_location: The name of the node that is injected into.
        no_injection_digraph: Graph after complete simulation of a pressure set without injection.
        allow_upstream: If `True`, blood can travel against the direction of the flow if the pressure difference is low enough.
    
    Returns:
        percent_filled: Percent of nidus nodes to which there exists a directed path from the injection location.
    """
    if allow_upstream:
        no_injection_nidus = get_nidus(no_injection_digraph)
        nidus = get_nidus(digraph)
        all_changes = pressure_change(no_injection_nidus, nidus)
        changes = [change / 100 for change in all_changes if change < 0 and change > -100]
        threshold = threshold_otsu(np.array(changes)) if len(changes) else -10000

    nx.set_node_attributes(digraph, False, "reached")
    nx.set_edge_attributes(digraph, False, "reached")

    digraph.nodes[injection_location]["reached"] = True
    queue = deque([injection_location])
    reached = 0

    while queue:

        n = len(queue)

        for _ in range(n):

            node = queue.popleft()

            for self_node, next_node in digraph.out_edges(node):

                if next_node not in intranidal_nodes or digraph[self_node][next_node]["reached"]:
                    continue

                digraph.nodes[next_node]["reached"] = True
                digraph[self_node][next_node]["reached"] = True
                reached += 1
                queue.append(next_node)

            if allow_upstream:

                for next_node, self_node in digraph.in_edges(node):

                    if next_node not in intranidal_nodes or digraph[next_node][self_node]["reached"]:
                        continue

                    if no_injection_digraph.has_edge(next_node, self_node):
                        if (digraph[next_node][self_node]["pressure"] - no_injection_digraph[next_node][self_node]["pressure"]) / no_injection_digraph[next_node][self_node]["pressure"] < threshold:
                            digraph.nodes[next_node]["reached"] = True
                            digraph[next_node][self_node]["reached"] = True
                            reached += 1
                            queue.append(next_node)

                    else:
                        if (-digraph[next_node][self_node]["pressure"] - no_injection_digraph[self_node][next_node]["pressure"]) / no_injection_digraph[self_node][next_node]["pressure"] < threshold:
                            digraph.nodes[next_node]["reached"] = True
                            digraph[next_node][self_node]["reached"] = True
                            reached += 1
                            queue.append(next_node)

    return reached / num_intranidal_vessels * 100

def calc_filling_post(digraph: nx.DiGraph, intranidal_nodes, num_intranidal_vessels, injection_location, no_injection_digraph: nx.DiGraph = None) -> float:
    """Calculates the percent of the nidus that an injection reaches.
    
    Args:
        digraph: Graph after complete simulation (with calculated flow and pressure attributes).
        intranidal_nodes: List of nodes in the nidus (the algorithm will only traverse vessels whose ends are in `intranidal_nodes`).
        injection_location: The name of the node that is injected into.
        no_injection_digraph: Graph after complete simulation of a pressure set without injection.
        allow_upstream: If `True`, blood can travel against the direction of the flow if the pressure difference is low enough.
    
    Returns:
        percent_filled: Percent of nidus nodes to which there exists a directed path from the injection location.
        percent_filled_post: Percent of nidus nodes which are filled after the injection is complete.
    """
    nx.set_node_attributes(digraph, False, "reached")
    nx.set_edge_attributes(digraph, False, "reached")

    digraph.nodes[injection_location]["reached"] = True
    queue = deque([injection_location])
    reached_nodes = set(injection_location)
    reached = 0

    while queue:
        
        n = len(queue)
        
        for _ in range(n):

            node = queue.popleft()

            for self_node, next_node in digraph.out_edges(node):
                if next_node not in intranidal_nodes or digraph[self_node][next_node]["reached"]:
                    continue

                digraph.nodes[next_node]["reached"] = True
                digraph[self_node][next_node]["reached"] = True
                reached += 1
                reached_nodes.add(next_node)
                queue.append(next_node)

    queue = deque(reached_nodes)
    reached_post = reached

    while queue:

        n = len(queue)

        for _ in range(n):

            node = queue.popleft()

            for self_node, next_node in no_injection_digraph.out_edges(node):

                if next_node not in intranidal_nodes or next_node in reached_nodes:
                    continue

                digraph.nodes[next_node]["reached"] = True

                if next_node in digraph[self_node]:
                    digraph[self_node][next_node]["reached"] = True

                else:
                    digraph[next_node][self_node]["reached"] = True

                reached_post += 1
                reached_nodes.add(next_node)
                queue.append(next_node)

    return reached / num_intranidal_vessels * 100, reached_post / num_intranidal_vessels * 100


def calc_filling(digraph: nx.DiGraph, intranidal_nodes, pressure, all_edges, injection_pressure_mmHg, num_intranidal_vessels) -> float:
    """Calculates the percent of nidus vessels that are back-filled.
    
    Args:
        digraph: Graph after complete simulation (with calculated flow and pressure attributes).
        intranidal_nodes: List of nodes in the nidus.
        pressure: List/Array of calculated pressure values (in mmHg) in a network.
        all_edges: List of (start, end) edge tuples corresponding to the `pressure` array.
        injection_pressure_mmHg: Pressure, in mmHg, of the injection.
        num_intranidal_vessels: Number of vessels in the nidus.
    
    Returns:
        percent_filled: Percent of nidus vessels whose pressure is less than the injection pressure.
    """
    nx.set_node_attributes(digraph, False, "filled")

    filled = 0
    
    for i, p in enumerate(pressure):
        if all_edges[i][0] in intranidal_nodes and all_edges[i][1] in intranidal_nodes and p < injection_pressure_mmHg:
            digraph.nodes[all_edges[i][0]]["filled"] = True
            digraph.nodes[all_edges[i][1]]["filled"] = True
            filled += 1

    return filled / num_intranidal_vessels * 100

def compute_rupture_risk(graph, p_min_mmHg):
    """Computes and prints the rupture risk for each vessel.
    
    Args:
        graph: Graph after complete simulation (with calculated flow and pressure attributes).
        p_min_mmHg: CVP pressure in mmHg.
    """
    pressures = []

    for _, _, attr in graph.edges(data=True):
        if attr["type"] == vessel.fistulous or attr["type"] == vessel.plexiform:
            pressures.append(attr["pressure"])

    p_max_mmHg = 74
    p_min_mmHg = 5
    risks = []

    for pressure in pressures:
        risk = math.log(abs(pressure) / p_min_mmHg) / math.log(p_max_mmHg / p_min_mmHg) * 100
        risk = max(0, min(risk, 100))
        risks.append(risk)

    return min(risks), np.mean(risks), max(risks)


def get_stats(digraph: nx.DiGraph, no_injection_digraph: nx.DiGraph = None, p_min_mmHg = 6, injection_pressure_mmHg = 0, injection_location = 0):
    """Returns stats for different vessels (count and min/mean/max/total flow/pressure of all/fistulous/plexiform/feeder/drainer vessels) given the `graph` result of `simulate()`. Writes some of the computed stats into the provided digraph.
    
    Args:
        digraph: Graph (result of `simulate()`).
        no_injection_digraph: Graph after complete simulation of a pressure set without injection.
        p_min_mmHg: CVP pressure in mmHg.
        injection_pressure_mmHg: Pressure, in mmHg, of the injection.
        injection_location: The name of the node that is injected into.
    
    Returns:
        stats: A dictionary of stats.
    """
    types = {
        vessel_type: {
            "count": 0,
            "name": name,
            "flow": { "total": 0, "min": float("inf"), "max": float("-inf"), "units": "mL/min" },
            "pressure": { "total": 0, "min": float("inf"), "max": float("-inf"), "units": "mmHg" },
            "resistance": { "total": 0, "min": float("inf"), "max": float("-inf"), "units": "mmHg*min/mL" },
            "radius": { "total": 0, "min": float("inf"), "max": float("-inf"), "units": "cm" },
            "length": { "total": 0, "min": float("inf"), "max": float("-inf"), "units": "cm" }
        } for vessel_type, name in zip((None, vessel.fistulous, vessel.plexiform, (vessel.fistulous, vessel.plexiform), vessel.feeder, vessel.drainer), ("vessels", "fistulous", "plexiform", "intranidal", "feeder", "drainer"))
    }

    nodes = set()
    intranidal_nodes = set()
    feeder_total, drainer_total = 0, 0

    for n1, n2, attr in digraph.edges(data=True):

        nodes.add(n1)
        nodes.add(n2)

        if attr["type"] == vessel.feeder:
            feeder_total += attr["flow"] * (1 if isinstance(n2, str) and n2[:2] == "AF" else -1)

        if attr["type"] == vessel.drainer:
            drainer_total += attr["flow"] * (1 if isinstance(n1, str) and n1[:2] == "DV" else -1)
        
        if attr["type"] in [vessel.fistulous, vessel.plexiform]:
            intranidal_nodes.add(n1)
            intranidal_nodes.add(n2)

        types[None]["count"] += 1

        for stat, value in types[None].items():
            if stat not in ["count", "name"]:
                value["total"] += attr[stat]
                value["min"] = min(value["min"], attr[stat])
                value["max"] = max(value["max"], attr[stat])

        for vessel_types in types:

            if attr["type"] == vessel_types or (isinstance(vessel_types, tuple) and attr["type"] in vessel_types):

                types[vessel_types]["count"] += 1

                for stat, value in types[vessel_types].items():

                    if stat not in ["count", "name"]:
                        value["total"] += attr[stat]
                        value["min"] = min(value["min"], attr[stat])
                        value["max"] = max(value["max"], attr[stat])
    
    min_risk, mean_risk, max_risk = compute_rupture_risk(digraph, p_min_mmHg)

    num_intranidal_vessels = types[vessel.fistulous]["count"] + types[vessel.plexiform]["count"]

    stats = {
        "Num intranidal nodes": len(intranidal_nodes),
        "Num nodes": len(nodes),
        "Num intranidal vessels": num_intranidal_vessels,
        "Injection location": injection_location,
        "Injection pressure (mmHg)": injection_pressure_mmHg,
    }

    for vessel_type, attrs in types.items():

        stats[f"Num {attrs['name']}"] = attrs["count"]

        for stat, value in attrs.items():

            if stat not in ["count", "name"]:
                title = f"{'' if attrs['name'] == 'vessels' else attrs['name'] + ' '}{stat} ".capitalize()
                units = f" ({value['units']})"
                stats[title + "min" + units] = round(value["min"], ROUND_DECIMALS) if attrs["count"] else 0
                stats[title + "mean" + units] = round(value["total"] / attrs["count"], ROUND_DECIMALS) if attrs["count"] else 0
                stats[title + "max" + units] = round(value["max"], ROUND_DECIMALS) if attrs["count"] else 0
    
    filling, filling_post = calc_filling_post(digraph, intranidal_nodes, types[(vessel.fistulous, vessel.plexiform)]["count"], injection_location, no_injection_digraph) if injection_location else (0, 0)
    # filling = 0 if injection_location is None else calc_filling_bfs(digraph, intranidal_nodes, types[(vessel.fistulous, vessel.plexiform)]["count"], injection_location)

    return stats | {
        "Feeder total flow (mL/min)": feeder_total,
        "Drainer total flow (mL/min)": drainer_total,
        "Min rupture risk (%)": round(min_risk, ROUND_DECIMALS),
        "Mean rupture risk (%)": round(mean_risk, ROUND_DECIMALS),
        "Max rupture risk (%)": round(max_risk, ROUND_DECIMALS),
        "Percent filled (%)": round(filling, ROUND_DECIMALS),
        "Percent filled post-injection (%)": round(filling_post, ROUND_DECIMALS),
    }


def calc_flow_batch(graph: nx.Graph, all_edges, p_exts) -> np.ndarray:
    """Calculates flow and pressure differences for each vessel.

    Args:
        graph: The graph.
        all_edges: A list of (starting node, ending node) tuples.
        p_ext: A dictionary of known node : pressure values in mmHg.

    Returns:
        flow: The calculated flow values in mL/min.
    """
    Rv = []
    ΔΔP = []
    num_edges = len(all_edges)

    # First law: sum of flow towards each node is 0
    ΔΔP = [[0 for _ in p_exts] for i in range(graph.number_of_nodes())]

    for i, node in enumerate(graph.nodes()):

        edges = graph.edges([node])
        flow = [0 for _ in range(num_edges)]

        for edge in edges:

            if edge in all_edges:
                index = all_edges.index(edge)
                flow[index] = 1

            else:
                index = all_edges.index((edge[1], edge[0]))
                flow[index] = -1

        Rv.append(flow)

    # Second law: total pressure change after traversing any closed loop is 0
    # ==> The only nonzero values of ΔΔP are those where there is an external pressure source
    for cycle in nx.cycle_basis(graph):  # nx.recursive_simple_cycles(graph)

        total_pressures = [0 for _ in p_exts]
        flow = [0 for _ in range(num_edges)]

        for i, node1 in enumerate(cycle):

            node2 = cycle[(i + 1) % len(cycle)]
            edge = (node1, node2)

            if edge in all_edges:
                index = all_edges.index(edge)
                flow[index] = -graph.edges[edge]["resistance"]

            else:
                index = all_edges.index((edge[1], edge[0]))
                flow[index] = graph.edges[edge]["resistance"]

            for j, p_ext in enumerate(p_exts):

                if edge in p_ext:
                    total_pressures[j] -= p_ext[edge]

                elif (node2, node1) in p_ext:
                    total_pressures[j] += p_ext[(node2, node1)]

        Rv.append(flow)
        ΔΔP.append(total_pressures)
    
    result = np.linalg.lstsq(Rv, ΔΔP, rcond=None)

    return result[0], np.sum(np.abs(Rv @ result[0] - ΔΔP))


def calc_flow(graph: nx.Graph, all_edges, p_ext) -> np.ndarray:
    """Calculates flow and pressure differences for each vessel.

    Args:
        graph: The graph.
        all_edges: A list of (starting node, ending node) tuples.
        p_ext: A dictionary of known node : pressure values in mmHg.

    Returns:
        flow: The calculated flow values in mL/min.
    """
    Rv = []
    ΔΔP = []
    num_edges = len(all_edges)

    # First law: sum of flow towards each node is 0
    ΔΔP = [0 for i in range(graph.number_of_nodes())]

    for i, node in enumerate(graph.nodes()):

        edges = graph.edges([node])
        flow = [0 for _ in range(num_edges)]

        for edge in edges:

            if edge in all_edges:
                index = all_edges.index(edge)
                flow[index] = 1

            else:
                index = all_edges.index((edge[1], edge[0]))
                flow[index] = -1

        Rv.append(flow)

    # Second law: total pressure change after traversing any closed loop is 0
    # ==> The only nonzero values of ΔΔP are those where there is an external pressure source
    for cycle in nx.cycle_basis(graph):  # nx.recursive_simple_cycles(graph)

        total_pressure = 0
        flow = [0 for _ in range(num_edges)]

        for i, node1 in enumerate(cycle):

            node2 = cycle[(i + 1) % len(cycle)]
            edge = (node1, node2)

            if edge in all_edges:
                index = all_edges.index(edge)
                flow[index] = -graph.edges[edge]["resistance"]
                
            else:
                index = all_edges.index((edge[1], edge[0]))
                flow[index] = graph.edges[edge]["resistance"]

            if edge in p_ext:
                total_pressure -= p_ext[edge]

            elif (node2, node1) in p_ext:
                total_pressure += p_ext[(node2, node1)]

        Rv.append(flow)
        ΔΔP.append(total_pressure)
    
    match SOLVE_MODE:

        case 1:
            result = np.linalg.lstsq(Rv, ΔΔP, rcond=None)
            return result[0].flatten(), np.sum(np.abs(Rv @ result[0] - ΔΔP))
        
        case 2:
            T = np.array(Rv).T
            result = sp.sparse.linalg.spsolve(T @ Rv, T @ ΔΔP)
            return result.flatten(), np.sum(np.abs(Rv @ result - ΔΔP))


def get_Q_and_P(graph: nx.Graph, p_ext) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Sets flows and pressure differences for each edge in the graph.

    Args:
        graph: The graph.
        p_ext: A dictionary of known node : pressure values in mmHg.

    Returns:
        flow: The calculated flow values in mL/min.
        pressure: The calculated pressure gradients in mmHg.
        all_edges: A list of the edges in the same order as the flows and pressures.
    """
    all_edges = [(edge[2]["start"], edge[2]["end"]) for edge in graph.edges(data=True)]
    flow, error = calc_flow(graph, all_edges, p_ext)
    
    # flow = difference in pressure between ends / resistance
    # Hagen-Poiseuille equation
    pressure = flow * np.array([graph.edges[edge]["resistance"] for edge in all_edges])
    nx.set_edge_attributes(graph, {edge: flow[i] for i, edge in enumerate(all_edges)}, "flow")
    nx.set_edge_attributes(graph, {edge: pressure[i] for i, edge in enumerate(all_edges)}, "pressure")
    
    return flow, pressure, all_edges, error
