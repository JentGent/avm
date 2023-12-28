from scipy.special import comb
import random
import networkx as nx
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from scipy.optimize import lsq_linear

# ONLY_INTRANIDAL indicates whether or not to display extranidal nodes in the graph.
ONLY_INTRANIDAL = False

# VISCOSITY is the viscosity of blood in Poise.
VISCOSITY = 0.04

# MMHG_TO_DYNCM is the conversion factor for converting between mmHg and dynes/cm^2.
MMHG_TO_DYNCM = 1333.22

# ABS_PRESSURE indicates whether or not to display calculated absolute pressures.
ABS_PRESSURE = False

# When LABEL is True, the edge labels are displayed. When it is False, the flow and pressure information is displayed.
LABEL = False


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
            next_pressure = pressure + \
                graph.edges[edge]["Δpressure"] * \
                (-1 if graph.edges[edge]["start"] == node else 1)
            pressure_forward(graph, next_node, next_pressure, pressures)


def calc_pressures(graph: nx.Graph, pressures: dict[str, float]):
    """Begins the calculation of absolute pressure through the graph.

    Args:
        graph: The graph.
        pressures: A dictionary of known node : pressure values.
    """
    pressure_forward(graph, "SP", pressures["SP"], pressures)


def display(graph: nx.Graph, intranidal_nodes: list = [], node_pos={}):
    """Displays the graph.

    Args:
        graph: The graph.
        intranidal_nodes: The nodes in the nidus.
        node_pos: A dictionary of node : [x, y] positions.
    """
    pos = nx.spring_layout(graph, 1, node_pos, node_pos.keys())

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
    pressures = [edge[2]["Δpressure"] for edge in graph.edges(data=True)]
    min_pressure = min(pressures)
    max_pressure = max(pressures)
    edge_widths = [np.interp(edge[2]["Δpressure"], [min_pressure, max_pressure], [0.5, 5]) for edge in graph.edges(data=True)]
    edge_colors = [edge[2]["flow"] for edge in graph.edges(data=True)]
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.cool)
    if LABEL:
        edge_labels = {(edge[0], edge[1]): edge[2]["label"] for i, edge in enumerate(graph.edges(data=True))}
    else:
        edge_labels = {(edge[0], edge[1]): str(round(edge[2]["flow"], 3)) + "\n" + str(round(edge[2]["Δpressure"] * (-1 if edge[2]["flow"] < 0 else 1), 3)) for edge in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
    sm.set_array([])
    plt.colorbar(sm, label="Flow (mL/min)")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min_pressure, vmax=max_pressure))
    sm.set_array([])
    plt.colorbar(sm, label="Pressure (mm Hg)")
    plt.show()


def edges_to_graph(edges: list) -> nx.Graph:
    """Converts a list of edges to a graph in the right format ([starting node, ending node, radius, length, resistance, label]).

    Args:
        edges: List of edges.

    Returns:
        graph: NetworkX graph generated from edges.
    """
    graph = nx.Graph()
    graph.add_edges_from([(
        edge[0], edge[1], {
            "start": edge[0], "end": edge[1],
            "radius": edge[2], "length": edge[3],
            "resistance": edge[4], "label": edge[5],
            "locked": True
        }) for edge in edges])
    return graph


def generate_nidus(graph: nx.Graph, intranidal_nodes: list, num_expected_edges: int) -> nx.Graph:
    """Generates a new graph with a random nidus from a given graph and its intranidal nodes.

    Args:
        graph: The original graph.
        intranidal_nodes: List of the nodes between which to generate edges.
        num_expected_edges: Expected number of edges to generate.

    Returns:
        graph: The graph with a random nidus formation.
    """
    num_intranidal_verts = len(intranidal_nodes)
    probability = num_expected_edges / comb(num_intranidal_verts, 2)
    graph = graph.copy()
    for j in range(num_intranidal_verts - 1):
        for k in range(j + 1, num_intranidal_verts):
            if random.random() < probability:
                radius, length = random.random() * (0.5 - 0.05) + 0.05, random.random() * (5 - 1) + 1
                graph.add_edge(
                    intranidal_nodes[j], intranidal_nodes[k],
                    start=intranidal_nodes[j], end=intranidal_nodes[k],
                    radius=radius, length=length,
                    resistance=8 * VISCOSITY / np.pi * length / (radius ** 4), label=""
                )
    return graph


def simulate(graph: nx.Graph, intranidal_nodes: list, p_ext: dict[str, float]) -> tuple[np.ndarray, np.ndarray, nx.Graph]:
    """Simulates the nidus.

    Args:
        graph: Graph from `edges_to_graph()` or `generate_nidus()`.
        intranidal_nodes: List of nodes in the nidus to generate edges between.
        p_ext: A dictionary of known node : pressure values.
        num_nidi: Number of nidi to simulate.
        num_expected_edges: Expected number of edges to generate in the nidus.

    Returns:
        flow: The calculated flow values.
        pressure: The calculated pressure gradients.
        graph: The directed graph of flow and pressure.
    """
    flow, pressure, all_edges = get_Q_and_P(graph, p_ext)
    if ABS_PRESSURE:
        calc_pressures(graph, p_ext)

    pnodes = [node for node in graph.nodes("pressure") if not ONLY_INTRANIDAL or node[0] in intranidal_nodes]
    graph = nx.DiGraph(
        [
            (edge[2]["end" if edge[2]["flow"] < 0 else "start"], edge[2]["start" if edge[2]["flow"] < 0 else "end"],
                {
                    "radius": edge[2]["radius"],
                    "length": edge[2]["length"],
                    "resistance": edge[2]["resistance"],
                    "label": edge[2]["label"],
                    "flow": abs(edge[2]["flow"]) * 60,  # cm^3/s = 60 mL/min
                    "Δpressure": abs(edge[2]["Δpressure"]) / MMHG_TO_DYNCM
            }) for i, edge in enumerate(graph.edges(data=True))
            if not ONLY_INTRANIDAL or (edge[0] in intranidal_nodes and edge[1] in intranidal_nodes)
        ]
    )
    graph.add_nodes_from(
        [(node[0], {"pressure": node[1]}) for node in pnodes])
    return flow, pressure, graph


def calc_flow(graph: nx.Graph, all_edges, p_ext) -> np.ndarray:
    """Calculates flow and pressure differences for each vessel.

    Args:
        graph: The graph.
        all_edges: A list of (starting node, ending node) tuples.
        p_ext: A dictionary of known node : pressure values.

    Returns:
        flow: The calculated flow values.
    """
    Rv = []
    ΔΔP = []
    num_edges = len(all_edges)

    # First law: sum of flow towards each node is 0
    ΔΔP = [0 for i in range(graph.number_of_nodes())]
    for i in graph.nodes():
        edges = graph.edges([i])
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
                flow[index] = graph.edges[edge]["resistance"]
            else:
                index = all_edges.index((edge[1], edge[0]))
                flow[index] = -graph.edges[edge]["resistance"]
            if edge in p_ext:
                total_pressure += p_ext[edge]
            elif (node2, node1) in p_ext:
                total_pressure -= p_ext[(node2, node1)]
        Rv.append(flow)
        ΔΔP.append(total_pressure)

    result = np.linalg.lstsq(np.array(Rv), np.array([ΔΔP]).T, rcond=None)
    print(result[1])
    return result[0].flatten()


def get_Q_and_P(graph: nx.Graph, p_ext) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Sets flows and pressure differences for each edge in the graph.

    Args:
        graph: The graph.
        p_ext: A dictionary of known node : pressure values.

    Returns:
        flow: The calculated flow values.
        pressure: The calculated pressure gradients.
        all_edges: A list of the edges in the same order as the flows and pressures.
    """
    all_edges = [(edge[2]["start"], edge[2]["end"])
                 for edge in graph.edges(data=True)]
    flow = calc_flow(graph, all_edges, p_ext)

    # flow = difference in pressure between ends / resistance
    # Hagen-Poiseuille equation
    pressure = flow * np.array([graph.edges[edge]["resistance"]
                               for edge in all_edges])
    nx.set_edge_attributes(
        graph, {edge: flow[i] for i, edge in enumerate(all_edges)}, "flow")
    nx.set_edge_attributes(
        graph, {edge: pressure[i] for i, edge in enumerate(all_edges)}, "Δpressure")
    return flow, pressure, all_edges
