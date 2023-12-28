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

# NODE_POS lists the positions of specific nodes in the graph.
NODE_POS = {
    1: [-1, 0.5],
    2: [-0.5, 0.4],
    3: [-0.25, 0],
    4: [-0.5, 0.5],
    5: [-0.25, 1],
    6: [-0.15, 0.2],
    7: [-0.15, 0.95],
    8: [0.85, 0.95],
    9: [0.5, 1],
    10: [0.95, 1],
    11: [1, 0],
    12: [1, -0.5],
    13: [0.5, -1],
    14: [0.1875, -0.1],
    15: [0.375, 0.1],
    16: [0.5625, 0.3],
    "CVP": [1, -1],
    "SP": [-1, -1],
    "AF1": [0, -0.5],
    "AF2": [0, 0],
    "AF3": [0, 0.5],
    "AF4": [0.5, 0.5],
    "DV1": [0.75, -0.85],
    "DV2": [0.75, 0],
    "DV3": [0.75, 0.4],
}

# VESSELS is formatted like [first node, second node, radius, length, resistance, label].
VESSELS = [
    [1, 2, 0, 0, 67.9, "R1"],
    [1, 4, 0, 0, 67.9, "R2"],
    [2, 3, 0, 0, 5037, "R3"],
    [3, 12, 0, 0, 2210, "R4"],
    [4, 5, 0, 0, 637.5, "R5"],
    [5, 9, 0, 0, 1000000, "R6"],
    [9, 18, 0, 0, 12750000, "R7"],
    [9, 10, 0, 0, 1000000, "R8"],
    [10, 11, 0, 0, 4177.9, "R9"],
    [4, 6, 0, 0, 522, "R10"],
    [6, 14, 0, 0, 745.5, "R11"],
    [6, 13, 0, 0, 15725000, "R12"],
    [6, 7, 0, 0, 10200, "R13"],
    [7, 8, 0, 0, 1000000, "R14"],
    [8, 11, 0, 0, 4177.9, "R15"],
]

# PRESSURES is a dictionary of known node : pressure values.
PRESSURES = {
    # "SP": 74 * MMHG_TO_DYNCM,
    # "AF1": 47 * MMHG_TO_DYNCM,
    # "AF2": 47 * MMHG_TO_DYNCM,
    # "AF3": 50 * MMHG_TO_DYNCM,
    # "AF4": 50 * MMHG_TO_DYNCM,
    # "DV1": 17 * MMHG_TO_DYNCM,
    # "DV2": 17 * MMHG_TO_DYNCM,
    # "DV3": 17 * MMHG_TO_DYNCM,
    # "CVP": 5 * MMHG_TO_DYNCM
    
    # (32, "SP"): 74 * MMHG_TO_DYNCM,
    ("SP", 1): 74 * MMHG_TO_DYNCM,
    (3, 12): 47 * MMHG_TO_DYNCM,
    (6, 14): 47 * MMHG_TO_DYNCM,
    (6, 13): 50 * MMHG_TO_DYNCM,
    (9, 18): 50 * MMHG_TO_DYNCM,
    (30, 11): 17 * MMHG_TO_DYNCM,
    (29, 11): 17 * MMHG_TO_DYNCM,
    (31, 32): 5 * MMHG_TO_DYNCM
}

# INTRANIDAL_NODES is a list of nodes in the nidus.
INTRANIDAL_NODES = ["AF1", "AF2", "AF3", "AF4"] + \
    list(range(14, 14 + 19 - 4 - 3)) + ["DV1", "DV2", "DV3"]
# INTRANIDAL_NODES = ["AF1", "AF2", "AF3", "AF4"] + list(range(14, 17)) + ["DV1", "DV2", "DV3"]
# INTRANIDAL_NODES = ["AF1", "AF2", "AF3", "AF4", "DV1", "DV2", "DV3"]
# INTRANIDAL_NODES = []


def pressure_forward(graph: nx.DiGraph, node, pressure, pressures: dict[str, float]):
    """Step through the graph to calculate absolute pressure.

    Args:
        graph (nx.DiGraph): The graph.
        node (any): The name of the original node.
        pressure (float): The pressure of the original node.
        pressures (dict[str, float]): A dictionary of known node : pressure values.
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
    """Begin the calculation of absolute pressure through the graph.

    Args:
        graph (nx.DiGraph): The graph.
        pressures (dict[str, float]): A dictionary of known node : pressure values.
    """
    pressure_forward(graph, "SP", pressures["SP"], pressures)
    # for node, pressure in pressures.items():
    #     if graph.has_node(node):
    #         pressure_forward(graph, node, pressure, pressures)


def display(graph: nx.Graph, intranidal_nodes: list = [], flow=True):
    """Display the graph.

    Args:
        graph (nx.Graph): The graph.
        intranidal_nodes (list): The nodes in the nidus.
    """
    pos = nx.spring_layout(graph, 1, NODE_POS, NODE_POS.keys())

    # Fix node positions
    xs, ys = [coords[0] for node, coords in pos.items() if node not in NODE_POS], [
        coords[1] for node, coords in pos.items() if node not in NODE_POS]
    if xs:
        mx, my = min(xs), min(ys)
        Mx, My = (max(xs) - mx) or 1, (max(ys) - my) or 1
        for node, coords in pos.items():
            if node not in NODE_POS.keys():
                pos[node] = [0.1 + 0.55 * (coords[0] - mx) / Mx, 1.15 * (coords[1] - my) / My - 0.75]

    # Nodes
    node_colors = {
        node: "lightgreen" if node in intranidal_nodes else "pink" for node in graph.nodes()
        # node: pressure or 0 for node, pressure in graph.nodes("pressure")
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
    # edge_labels = { (edge[0], edge[1]): edge[2]["label"] for i, edge in enumerate(graph.edges(data = True)) }
    # edge_labels = { (edge[0], edge[1]): ((edge[2]["label"] + "\n") if edge[2]["label"] else "") + str(round(abs(flow[i, 0]), 3)) + "\n" + str(round(pressure[i, 0] * (-1 if flow[i, 0] < 0 else 1), 3)) for i, edge in enumerate(graph.edges(data = True)) }
    edge_labels = { (edge[0], edge[1]): str(round(edge[2]["flow"], 3)) + "\n" + str(round(edge[2]["Δpressure"] * (-1 if edge[2]["flow"] < 0 else 1), 3)) for edge in graph.edges(data = True) }
    # edge_labels = {(edge[0], edge[1]): str(round(edge[2]["flow"], 3)) for edge in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
    sm.set_array([])
    plt.colorbar(sm, label="Flow (mL/min)")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min_pressure, vmax=max_pressure))
    sm.set_array([])
    plt.colorbar(sm, label="Pressure (mm Hg)")
    plt.show()


def edges_to_graph(edges: list):
    """Convert a list of edges to a graph in the right format ([starting node, ending node, radius, length, resistance, label]).

    Args:
        edges (list): List of edges.
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


def generate_nidus(graph: nx.Graph, intranidal_nodes: list, num_expected_edges: int):
    """Generate a new graph with a random nidus from a given graph and its intranidal nodes.

    Args:
        graph (nx.Graph): The original graph.
        intranidal_nodes (list): List of the nodes between which to generate edges.
        num_expected_edges (int): Expected number of edges to generate.
    """
    num_intranidal_verts = len(intranidal_nodes)
    probability = num_expected_edges / comb(num_intranidal_verts, 2)
    graph = graph.copy()
    for j in range(num_intranidal_verts - 1):
        for k in range(2):
            unconnected = [node for node in intranidal_nodes if not graph.has_edge(intranidal_nodes[j], node) and node is not intranidal_nodes[j]]
            connect = random.choice(unconnected)
            graph.add_edge(
                intranidal_nodes[j], connect,
                start=intranidal_nodes[j], end=connect,
                radius=0.05, length=5,
                resistance=8 * VISCOSITY / np.pi * 5 / (0.05 ** 4), label=""
            )

        # for k in range(j + 1, num_intranidal_verts):
        #     if random.random() < probability:
        #         # radius, length = random.random() * (0.5 - 0.05) + 0.05, random.random() * (5 - 1) + 1
        #         radius, length = 0.05, 1
        #         graph.add_edge(
        #             intranidal_nodes[j], intranidal_nodes[k],
        #             start = intranidal_nodes[j], end = intranidal_nodes[k],
        #             radius = radius, length = length,
        #             resistance = 8 * VISCOSITY / np.pi * length / (radius ** 4), label = ""
        #         )
    return graph


def simulate(graph: nx.Graph, intranidal_nodes: list, p_ext: dict[str, float]):
    """Simulate the nidus.

    Args:
        graph (nx.Graph): Graph from `edges_to_graph()` or `generate_nidus()`.
        intranidal_nodes (list): List of nodes in the nidus to generate edges between.
        p_ext (dict[str, float]): A dictionary of known node : pressure values.
        num_nidi (int): Number of nidi to simulate.
        num_expected_edges (int): Expected number of edges to generate in the nidus.
    """
    flow, pressure, all_edges = get_Q_and_P(graph, p_ext)
    # calc_pressures(graph, p_ext)

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

# Rv * Q = ΔΔP
# Matrix product representing satisfaction of Kirchoff's laws (solve to calculate flow Q)


def calc_flow(graph: nx.Graph, all_edges, p_ext):
    """Calculate flow and pressure differences for each vessel.

    Args:
        graph (nx.Graph): The graph.
        all_edges (list[tuple]): A list of (starting node, ending node) tuples.
        p_ext (dict[str, float]): A dictionary of known node : pressure values.
    """
    Rv = []
    ΔΔP = []
    num_edges = len(all_edges)

    # First law: sum of flow towards each node is 0
    # ΔΔP = [0 for i in range(graph.number_of_nodes() - len(p_ext))]
    ΔΔP = [0 for i in range(graph.number_of_nodes())]
    for i in graph.nodes():
        # if i in p_ext:
        #     continue
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
        # ΔΔP.append(0)

    # assigned pressures
    # nodes_of_p_ext = list(p_ext.keys())
    # for i, start in enumerate(nodes_of_p_ext[:-1]):
    #     for end in nodes_of_p_ext[i + 1:]:
    #         path = nx.shortest_path(graph, start, end)
    #         ΔΔP.append(p_ext[start] - p_ext[end])
    #         flow = [0 for _ in range(num_edges)]
    #         for k, node1 in enumerate(path[:-1]):
    #             edge = (node1, path[k + 1])
    #             if edge in all_edges:
    #                 index = all_edges.index(edge)
    #                 flow[index] = graph.edges[edge]["resistance"]
    #             else:
    #                 index = all_edges.index((edge[1], edge[0]))
    #                 flow[index] = -graph.edges[edge]["resistance"]
    #         Rv.append(flow)

    result = np.linalg.lstsq(np.array(Rv), np.array([ΔΔP]).T, rcond=None)
    # result = lsq_linear(np.array(Rv), np.array(ΔΔP), bounds = ([0 if "locked" in graph.edges[edge] else -np.inf for edge in all_edges], [np.inf for edge in all_edges]))
    print(result[1])
    return result[0].flatten()


def get_Q_and_P(graph: nx.Graph, p_ext):
    """Set flows and pressure differences for each edge in the graph.

    Args:
        graph (nx.Graph): The graph.
        p_ext (dict[str, float]): A dictionary of known node : pressure values.
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


def main():
    graph = edges_to_graph(VESSELS)
    with_nidus = generate_nidus(graph, INTRANIDAL_NODES, 93)
    flow, pressure, graph = simulate(with_nidus, INTRANIDAL_NODES, PRESSURES)
    print(f"Minimum flow: {np.min(np.abs(flow * 60))} mL/min")
    print(f"Maximum flow: {np.max(np.abs(flow * 60))} mL/min")
    print(f"Average flow: {np.average(np.abs(flow * 60))} mL/min")
    print(f"Minimum pressure: {np.min(np.abs(pressure / MMHG_TO_DYNCM))}")
    print(f"Maximum pressure: {np.max(np.abs(pressure / MMHG_TO_DYNCM))}")
    print(f"Average pressure: {np.average(np.abs(pressure / MMHG_TO_DYNCM))}")
    display(graph, INTRANIDAL_NODES)


if __name__ == "__main__":
    main()
