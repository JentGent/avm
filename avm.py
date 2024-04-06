import random
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Literal
import math

# VISCOSITY is the viscosity of blood in Poise.
VISCOSITY = 0.035

# If PREDEFINED_RESISTANCE is True, radius and length data is ignored, and the given resistance values are used; otherwise, radius and length are plugged into the Hagen–Poiseuille equation: resistance = (8 * length * viscosity) / (pi r^4)
PREDEFINED_RESISTANCE = True

# x mmHg = x * MMHG_TO_DYNCM dyn/cm^2.
MMHG_TO_DYN_PER_SQUARE_CM = 1333.22

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

# VESSELS_TEMPLATE elements are formatted like [first node, second node, radius (cm), length (cm), resistance (dyn s / cm^5), label, type (optional)].
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

def calc_resistance(radius: float, length: float) -> float:
    """Uses the Hagen-Poiseuille equation to calculate resistance: resistance = (8 * length * viscosity) / (pi r^4)
    
    Args:
        radius: The radius of the blood vessel in cm.
        length: The length of the blood vessel in cm.
    
    Returns:
        resistance: The resistance of the blood vessel in dyn * s / cm^5.
    """
    return 8 * length * VISCOSITY / np.pi / (radius ** 4)


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


def display(graph: nx.Graph, intranidal_nodes: list = [], node_pos={}, title: str = None, cmap_min: float = None, cmap_max: float = None, color_is_flow: bool = True, label: Literal["name", "flow", "pressure", None] = "name"):
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
    if ABS_PRESSURE:
        node_colors = {
            node: pressure or 0 for node, pressure in graph.nodes("pressure")
        }
    else:
        node_colors = {
            node: "lightgreen" if node in intranidal_nodes else "pink" for node in graph.nodes()
        }
    nx.draw_networkx_nodes(graph, pos, node_color=[node_colors[node] for node in graph.nodes()])
    nx.draw_networkx_labels(graph, pos)

    # Edges
    pressures = [edge[2]["pressure"] for edge in graph.edges(data=True)]
    flows = [edge[2]["flow"] for edge in graph.edges(data=True)]
    min_pressure, max_pressure = min(pressures), max(pressures)
    min_flow, max_flow = min(flows), max(flows)

    if color_is_flow:
        edge_widths = [np.interp(edge[2]["pressure"], [min_pressure, max_pressure], [1, 5]) for edge in graph.edges(data=True)]
        edge_colors = [edge[2]["flow"] for edge in graph.edges(data=True)]
    else:
        edge_widths = [np.interp(edge[2]["flow"], [min_flow, max_flow], [1, 5]) for edge in graph.edges(data=True)]
        edge_colors = [edge[2]["pressure"] for edge in graph.edges(data=True)]
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.cool if color_is_flow else plt.cm.Reds, edge_vmin=min(edge_colors) if cmap_min is None else cmap_min, edge_vmax=max(edge_colors) if cmap_max is None else cmap_max)
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
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min_pressure if cmap_min is None else cmap_min, vmax=max_pressure if cmap_max is None else cmap_max))
        sm.set_array([])
        colorbar = plt.colorbar(sm, ax = plt.gca(), label="Pressure (mm Hg)")
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
    """Converts a list of edges to a graph in the right format ([starting node, ending node, radius (cm), length (cm), resistance (dyn * s / cm^5), label]).

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
            "resistance": edge[4] if PREDEFINED_RESISTANCE else calc_resistance(edge[2], edge[3]),
            "label": edge[5],
            "type": edge[6] if len(edge) > 6 else vessel.other
        }) for edge in edges
    ])
    return graph


def simulate_batch(graph: nx.Graph, intranidal_nodes: list, p_exts: list[dict[str, float]], return_error: bool = False) -> tuple[np.ndarray, np.ndarray, list[tuple], list[nx.Graph], float]:
    """Simulates the nidus with a batch of different sets of pressure values.

    Args:
        graph: Graph from `edges_to_graph()` or `generate_nidus()`.
        intranidal_nodes: List of nodes in the nidus to generate edges between.
        p_ext: A dictionary of known node : pressure values in dyn / cm^2.
        num_nidi: Number of nidi to simulate.
        num_expected_edges: Expected number of edges to generate in the nidus.
        return_error: Whether to return the absolute pressure error from solving Kirchoff's circuit equations.

    Returns:
        flows: The calculated flow values in cm^3/s.
        pressures: The calculated pressure gradients in dyn / cm^2.
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
                "flow": abs(edge[2]["flow"]) * 60,  # x cm^3/s = 60x mL/min
                "pressure": abs(edge[2]["pressure"]) / MMHG_TO_DYN_PER_SQUARE_CM,
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
        p_ext: A dictionary of known node : pressure values in dyn / cm^2.
        num_nidi: Number of nidi to simulate.
        num_expected_edges: Expected number of edges to generate in the nidus.
        return_error: Whether to return the absolute pressure error from solving Kirchoff's circuit equations.

    Returns:
        flow: The calculated flow values in cm^3/s.
        pressure: The calculated pressure gradients in dyn / cm^2.
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
            "flow": abs(edge[2]["flow"]) * 60,  # x cm^3/s = 60x mL/min
            "pressure": abs(edge[2]["pressure"]) / MMHG_TO_DYN_PER_SQUARE_CM,
            "type": edge[2]["type"]
        }) for edge in graph.edges(data=True)
    ])
    if return_error:
        return flow, pressure, all_edges, graph, error
    return flow, pressure, all_edges, graph

def calc_filling(graph: nx.Graph, intranidal_nodes, pressure, all_edges, injection_pressure, num_intranidal_vessels):
    """Calculates the percent of nidus vessels that are back-filled."""
    filled = 0
    for i, p in enumerate(pressure):
        if all_edges[i][0] in intranidal_nodes and all_edges[i][1] in intranidal_nodes and p < injection_pressure:
            filled += 1
    return filled / num_intranidal_vessels * 100

def compute_rupture_risk(graph, p_min):
    """Computes and prints the rupture risk for each vessel."""
    pressures = []
    for _, _, attr in graph.edges(data=True):
        if attr["type"] == vessel.fistulous or attr["type"] == vessel.plexiform:
            pressures.append(attr["pressure"])
    p_max = 74  # mmHg
    p_min /= MMHG_TO_DYN_PER_SQUARE_CM
    risks = []
    for pressure in pressures:
        risk = math.log(abs(pressure) / p_min) / math.log(p_max / p_min) * 100
        risk = max(0, min(risk, 100))
        risks.append(risk)
    return np.mean(risks), max(risks)

def get_stats(graph: nx.DiGraph, p_min = 6):
    """Returns stats for different vessels (count and min/mean/max/total flow/pressure of all/fistulous/plexiform/feeder/drainer vessels) given the `graph` result of `simulate()`.
    
    Args:
        graph: Graph (result of `simulate()`).
    
    Returns:
        stats: A dictionary of stats.
    """
    count = 0
    flow_total, flow_min, flow_max = 0, float('inf'), float('-inf')
    pressure_total, pressure_min, pressure_max = 0, float('inf'), float('-inf')
    resistance_total, resistance_min, resistance_max = 0, float('inf'), float('-inf')
    radius_total, radius_min, radius_max = 0, float('inf'), float('-inf')
    length_total, length_min, length_max = 0, float('inf'), float('-inf')

    fi_count = 0
    fi_flow_total, fi_flow_min, fi_flow_max = 0, flow_min, flow_max
    fi_pressure_total, fi_pressure_min, fi_pressure_max = 0, flow_min, flow_max
    fi_resistance_total, fi_resistance_min, fi_resistance_max = 0, flow_min, flow_max
    fi_radius_total, fi_radius_min, fi_radius_max = 0, flow_min, flow_max
    fi_length_total, fi_length_min, fi_length_max = 0, flow_min, flow_max
    
    pl_count = 0
    pl_flow_total, pl_flow_min, pl_flow_max = 0, flow_min, flow_max
    pl_pressure_total, pl_pressure_min, pl_pressure_max = 0, flow_min, flow_max
    pl_resistance_total, pl_resistance_min, pl_resistance_max = 0, flow_min, flow_max
    pl_radius_total, pl_radius_min, pl_radius_max = 0, flow_min, flow_max
    pl_length_total, pl_length_min, pl_length_max = 0, flow_min, flow_max
    
    fe_count = 0
    fe_flow_total, fe_flow_min, fe_flow_max = 0, flow_min, flow_max
    fe_pressure_total, fe_pressure_min, fe_pressure_max = 0, flow_min, flow_max
    fe_resistance_total, fe_resistance_min, fe_resistance_max = 0, flow_min, flow_max
    fe_radius_total, fe_radius_min, fe_radius_max = 0, flow_min, flow_max
    fe_length_total, fe_length_min, fe_length_max = 0, flow_min, flow_max
    
    dr_count = 0
    dr_flow_total, dr_flow_min, dr_flow_max = 0, flow_min, flow_max
    dr_pressure_total, dr_pressure_min, dr_pressure_max = 0, flow_min, flow_max
    dr_resistance_total, dr_resistance_min, dr_resistance_max = 0, flow_min, flow_max
    dr_radius_total, dr_radius_min, dr_radius_max = 0, flow_min, flow_max
    dr_length_total, dr_length_min, dr_length_max = 0, flow_min, flow_max

    nodes = set()
    intranidal_nodes = set()

    for n1, n2, attr in graph.edges(data=True):
        nodes.add(n1)
        nodes.add(n2)
        count += 1
        flow_total += attr["flow"]
        flow_min = min(flow_min, attr["flow"])
        flow_max = max(flow_max, attr["flow"])
        pressure_total += attr["pressure"]
        pressure_min = min(pressure_min, attr["pressure"])
        pressure_max = max(pressure_max, attr["pressure"])
        resistance_total += attr["resistance"]
        resistance_min = min(resistance_min, attr["resistance"])
        resistance_max = max(resistance_max, attr["resistance"])
        radius_total += attr["radius"]
        radius_min = min(radius_min, attr["radius"])
        radius_max = max(radius_max, attr["radius"])
        length_total += attr["length"]
        length_min = min(length_min, attr["length"])
        length_max = max(length_max, attr["length"])
        if attr["type"] == vessel.fistulous:
            intranidal_nodes.add(n1)
            intranidal_nodes.add(n2)
            fi_count += 1
            fi_flow_total += attr["flow"]
            fi_flow_min = min(fi_flow_min, attr["flow"])
            fi_flow_max = max(fi_flow_max, attr["flow"])
            fi_pressure_total += attr["pressure"]
            fi_pressure_min = min(fi_pressure_min, attr["pressure"])
            fi_pressure_max = max(fi_pressure_max, attr["pressure"])
            fi_resistance_total += attr["resistance"]
            fi_resistance_min = min(fi_resistance_min, attr["resistance"])
            fi_resistance_max = max(fi_resistance_max, attr["resistance"])
            fi_radius_total += attr["radius"]
            fi_radius_min = min(fi_radius_min, attr["radius"])
            fi_radius_max = max(fi_radius_max, attr["radius"])
            fi_length_total += attr["length"]
            fi_length_min = min(fi_length_min, attr["length"])
            fi_length_max = max(fi_length_max, attr["length"])
        elif attr["type"] == vessel.plexiform:
            intranidal_nodes.add(n1)
            intranidal_nodes.add(n2)
            pl_count += 1
            pl_flow_total += attr["flow"]
            pl_flow_min = min(pl_flow_min, attr["flow"])
            pl_flow_max = max(pl_flow_max, attr["flow"])
            pl_pressure_total += attr["pressure"]
            pl_pressure_min = min(pl_pressure_min, attr["pressure"])
            pl_pressure_max = max(pl_pressure_max, attr["pressure"])
            pl_resistance_total += attr["resistance"]
            pl_resistance_min = min(pl_resistance_min, attr["resistance"])
            pl_resistance_max = max(pl_resistance_max, attr["resistance"])
            pl_radius_total += attr["radius"]
            pl_radius_min = min(pl_radius_min, attr["radius"])
            pl_radius_max = max(pl_radius_max, attr["radius"])
            pl_length_total += attr["length"]
            pl_length_min = min(pl_length_min, attr["length"])
            pl_length_max = max(pl_length_max, attr["length"])
        elif attr["type"] == vessel.feeder:
            fe_count += 1
            fe_flow_total += attr["flow"]
            fe_flow_min = min(fe_flow_min, attr["flow"])
            fe_flow_max = max(fe_flow_max, attr["flow"])
            fe_pressure_total += attr["pressure"]
            fe_pressure_min = min(fe_pressure_min, attr["pressure"])
            fe_pressure_max = max(fe_pressure_max, attr["pressure"])
            fe_resistance_total += attr["resistance"]
            fe_resistance_min = min(fe_resistance_min, attr["resistance"])
            fe_resistance_max = max(fe_resistance_max, attr["resistance"])
            fe_radius_total += attr["radius"]
            fe_radius_min = min(fe_radius_min, attr["radius"])
            fe_radius_max = max(fe_radius_max, attr["radius"])
            fe_length_total += attr["length"]
            fe_length_min = min(fe_length_min, attr["length"])
            fe_length_max = max(fe_length_max, attr["length"])
        elif attr["type"] == vessel.drainer:
            dr_count += 1
            dr_flow_total += attr["flow"]
            dr_flow_min = min(dr_flow_min, attr["flow"])
            dr_flow_max = max(dr_flow_max, attr["flow"])
            dr_pressure_total += attr["pressure"]
            dr_pressure_min = min(dr_pressure_min, attr["pressure"])
            dr_pressure_max = max(dr_pressure_max, attr["pressure"])
            dr_resistance_total += attr["resistance"]
            dr_resistance_min = min(dr_resistance_min, attr["resistance"])
            dr_resistance_max = max(dr_resistance_max, attr["resistance"])
            dr_radius_total += attr["radius"]
            dr_radius_min = min(dr_radius_min, attr["radius"])
            dr_radius_max = max(dr_radius_max, attr["radius"])
            dr_length_total += attr["length"]
            dr_length_min = min(dr_length_min, attr["length"])
            dr_length_max = max(dr_length_max, attr["length"])
    
    mean_risk, max_risk = compute_rupture_risk(graph, p_min)

    return {
        "Num intranidal nodes": len(intranidal_nodes),
        "Num nodes": len(nodes),
        "Mean rupture risk (%)": mean_risk,
        "Max rupture risk (%)": max_risk,
        "Num intranidal vessels": fi_count + pl_count,
        "Num vessels": count,

        "Flow min (mL/min)": round(flow_min, ROUND_DECIMALS) if count else 0,
        "Flow mean (mL/min)": round(flow_total / count, ROUND_DECIMALS) if count else 0,
        "Flow max (mL/min)": round(flow_max, ROUND_DECIMALS) if count else 0,

        "Pressure min (mmHg)": round(pressure_min, ROUND_DECIMALS) if count else 0,
        "Pressure mean (mmHg)": round(pressure_total / count, ROUND_DECIMALS) if count else 0,
        "Pressure max (mmHg)": round(pressure_max, ROUND_DECIMALS) if count else 0,

        "Resistance min (dyn*s/cm^5)": round(resistance_min, ROUND_DECIMALS) if count else 0,
        "Resistance mean (dyn*s/cm^5)": round(resistance_total / count, ROUND_DECIMALS) if count else 0,
        "Resistance max (dyn*s/cm^5)": round(resistance_max, ROUND_DECIMALS) if count else 0,

        "Radius min (cm)": round(radius_min, ROUND_DECIMALS) if count else 0,
        "Radius mean (cm)": round(radius_total / count, ROUND_DECIMALS) if count else 0,
        "Radius max (cm)": round(radius_max, ROUND_DECIMALS) if count else 0,

        "Length min (cm)": round(length_min, ROUND_DECIMALS) if count else 0,
        "Length mean (cm)": round(length_total / count, ROUND_DECIMALS) if count else 0,
        "Length max (cm)": round(length_max, ROUND_DECIMALS) if count else 0,

        "Num fistulous": fi_count,

        "Fistulous flow min (mL/min)": round(fi_flow_min, ROUND_DECIMALS) if fi_count else 0,
        "Fistulous flow mean (mL/min)": round(fi_flow_total / fi_count, ROUND_DECIMALS) if fi_count else 0,
        "Fistulous flow max (mL/min)": round(fi_flow_max, ROUND_DECIMALS) if fi_count else 0,

        "Fistulous pressure min (mmHg)": round(fi_pressure_min, ROUND_DECIMALS) if fi_count else 0,
        "Fistulous pressure mean (mmHg)": round(fi_pressure_total / fi_count, ROUND_DECIMALS) if fi_count else 0,
        "Fistulous pressure max (mmHg)": round(fi_pressure_max, ROUND_DECIMALS) if fi_count else 0,

        "Fistulous resistance min (dyn*s/cm^5)": round(fi_resistance_min, ROUND_DECIMALS) if fi_count else 0,
        "Fistulous resistance mean (dyn*s/cm^5)": round(fi_resistance_total / fi_count, ROUND_DECIMALS) if fi_count else 0,
        "Fistulous resistance max (dyn*s/cm^5)": round(fi_resistance_max, ROUND_DECIMALS) if fi_count else 0,

        "Fistulous radius min (cm)": round(fi_radius_min, ROUND_DECIMALS) if fi_count else 0,
        "Fistulous radius mean (cm)": round(fi_radius_total / fi_count, ROUND_DECIMALS) if fi_count else 0,
        "Fistulous radius max (cm)": round(fi_radius_max, ROUND_DECIMALS) if fi_count else 0,

        "Fistulous length min (cm)": round(fi_length_min, ROUND_DECIMALS) if fi_count else 0,
        "Fistulous length mean (cm)": round(fi_length_total / fi_count, ROUND_DECIMALS) if fi_count else 0,
        "Fistulous length max (cm)": round(fi_length_max, ROUND_DECIMALS) if fi_count else 0,

        "Num plexiform": pl_count,

        "Plexiform flow min (mL/min)": round(pl_flow_min, ROUND_DECIMALS) if pl_count else 0,
        "Plexiform flow mean (mL/min)": round(pl_flow_total / pl_count, ROUND_DECIMALS) if pl_count else 0,
        "Plexiform flow max (mL/min)": round(pl_flow_max, ROUND_DECIMALS) if pl_count else 0,

        "Plexiform pressure min (mmHg)": round(pl_pressure_min, ROUND_DECIMALS) if pl_count else 0,
        "Plexiform pressure mean (mmHg)": round(pl_pressure_total / pl_count, ROUND_DECIMALS) if pl_count else 0,
        "Plexiform pressure max (mmHg)": round(pl_pressure_max, ROUND_DECIMALS) if pl_count else 0,

        "Plexiform resistance min (dyn*s/cm^5)": round(pl_resistance_min, ROUND_DECIMALS) if pl_count else 0,
        "Plexiform resistance mean (dyn*s/cm^5)": round(pl_resistance_total / pl_count, ROUND_DECIMALS) if pl_count else 0,
        "Plexiform resistance max (dyn*s/cm^5)": round(pl_resistance_max, ROUND_DECIMALS) if pl_count else 0,

        "Plexiform radius min (cm)": round(pl_radius_min, ROUND_DECIMALS) if pl_count else 0,
        "Plexiform radius mean (cm)": round(pl_radius_total / pl_count, ROUND_DECIMALS) if pl_count else 0,
        "Plexiform radius max (cm)": round(pl_radius_max, ROUND_DECIMALS) if pl_count else 0,

        "Plexiform length min (cm)": round(pl_length_min, ROUND_DECIMALS) if pl_count else 0,
        "Plexiform length mean (cm)": round(pl_length_total / pl_count, ROUND_DECIMALS) if pl_count else 0,
        "Plexiform length max (cm)": round(pl_length_max, ROUND_DECIMALS) if pl_count else 0,

        "Num feeder": fe_count,

        "Feeder flow min (mL/min)": round(fe_flow_min, ROUND_DECIMALS) if fe_count else 0,
        "Feeder flow mean (mL/min)": round(fe_flow_total / fe_count, ROUND_DECIMALS) if fe_count else 0,
        "Feeder flow max (mL/min)": round(fe_flow_max, ROUND_DECIMALS) if fe_count else 0,

        "Feeder pressure min (mmHg)": round(fe_pressure_min, ROUND_DECIMALS) if fe_count else 0,
        "Feeder pressure mean (mmHg)": round(fe_pressure_total / fe_count, ROUND_DECIMALS) if fe_count else 0,
        "Feeder pressure max (mmHg)": round(fe_pressure_max, ROUND_DECIMALS) if fe_count else 0,

        "Feeder resistance min (dyn*s/cm^5)": round(fe_resistance_min, ROUND_DECIMALS) if fe_count else 0,
        "Feeder resistance mean (dyn*s/cm^5)": round(fe_resistance_total / fe_count, ROUND_DECIMALS) if fe_count else 0,
        "Feeder resistance max (dyn*s/cm^5)": round(fe_resistance_max, ROUND_DECIMALS) if fe_count else 0,

        "Feeder radius min (cm)": round(fe_radius_min, ROUND_DECIMALS) if fe_count else 0,
        "Feeder radius mean (cm)": round(fe_radius_total / fe_count, ROUND_DECIMALS) if fe_count else 0,
        "Feeder radius max (cm)": round(fe_radius_max, ROUND_DECIMALS) if fe_count else 0,

        "Feeder length min (cm)": round(fe_length_min, ROUND_DECIMALS) if fe_count else 0,
        "Feeder length mean (cm)": round(fe_length_total / fe_count, ROUND_DECIMALS) if fe_count else 0,
        "Feeder length max (cm)": round(fe_length_max, ROUND_DECIMALS) if fe_count else 0,

        "Feeder total flow (mL/min)": round(fe_flow_total, ROUND_DECIMALS),

        "Num drainer": dr_count,

        "Drainer flow min (mL/min)": round(dr_flow_min, ROUND_DECIMALS) if dr_count else 0,
        "Drainer flow mean (mL/min)": round(dr_flow_total / dr_count, ROUND_DECIMALS) if dr_count else 0,
        "Drainer flow max (mL/min)": round(dr_flow_max, ROUND_DECIMALS) if dr_count else 0,

        "Drainer pressure min (mmHg)": round(dr_pressure_min, ROUND_DECIMALS) if dr_count else 0,
        "Drainer pressure mean (mmHg)": round(dr_pressure_total / dr_count, ROUND_DECIMALS) if dr_count else 0,
        "Drainer pressure max (mmHg)": round(dr_pressure_max, ROUND_DECIMALS) if dr_count else 0,

        "Drainer resistance min (dyn*s/cm^5)": round(dr_resistance_min, ROUND_DECIMALS) if dr_count else 0,
        "Drainer resistance mean (dyn*s/cm^5)": round(dr_resistance_total / dr_count, ROUND_DECIMALS) if dr_count else 0,
        "Drainer resistance max (dyn*s/cm^5)": round(dr_resistance_max, ROUND_DECIMALS) if dr_count else 0,

        "Drainer radius min (cm)": round(dr_radius_min, ROUND_DECIMALS) if dr_count else 0,
        "Drainer radius mean (cm)": round(dr_radius_total / dr_count, ROUND_DECIMALS) if dr_count else 0,
        "Drainer radius max (cm)": round(dr_radius_max, ROUND_DECIMALS) if dr_count else 0,

        "Drainer length min (cm)": round(dr_length_min, ROUND_DECIMALS) if dr_count else 0,
        "Drainer length mean (cm)": round(dr_length_total / dr_count, ROUND_DECIMALS) if dr_count else 0,
        "Drainer length max (cm)": round(dr_length_max, ROUND_DECIMALS) if dr_count else 0,

        "Drainer total flow (mL/min)": round(dr_flow_total, ROUND_DECIMALS),
    }

def calc_flow_batch(graph: nx.Graph, all_edges, p_exts) -> np.ndarray:
    """Calculates flow and pressure differences for each vessel.

    Args:
        graph: The graph.
        all_edges: A list of (starting node, ending node) tuples, with resistances in dyn * s / cm^5.
        p_ext: A dictionary of known node : pressure values in dyn / cm^2.

    Returns:
        flow: The calculated flow values in cm^3 / s.
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
        all_edges: A list of (starting node, ending node) tuples, with resistances in dyn * s / cm^5.
        p_ext: A dictionary of known node : pressure values in dyn / cm^2.

    Returns:
        flow: The calculated flow values in cm^3 / s.
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
        p_ext: A dictionary of known node : pressure values in dyn / cm^2.

    Returns:
        flow: The calculated flow values in cm^3 / s
        pressure: The calculated pressure gradients in dyn / cm^2.
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
