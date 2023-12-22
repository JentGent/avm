from scipy.special import comb
import random
import networkx as nx
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from scipy.optimize import lsq_linear

VISCOSITY = 0.04
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

def pressure_forward(graph: nx.DiGraph, node, pressure, pressures: dict[str, float]):
    nx.set_node_attributes(graph, { node: pressure }, "pressure")
    edges = graph.edges([node])
    nx.set_edge_attributes(graph, { edge: pressure for edge in edges if graph.edges[edge]["start"] == node}, "pressure")
    for edge in edges:
        next_node = edge[1]
        if "pressure" not in graph.nodes[next_node]:
            next_pressure = pressure + graph.edges[edge]["Δpressure"] * (-1 if graph.edges[edge]["start"] == node else 1)
            pressure_forward(graph, next_node, next_pressure, pressures)

def calc_pressures(graph: nx.Graph, pressures: dict[str, float]):
    pressure_forward(graph, "SP", pressures["SP"], pressures)
    # for node, pressure in pressures.items():
    #     if graph.has_node(node):
    #         pressure_forward(graph, node, pressure, pressures)

def display(graph: nx.Graph):
    pos = nx.spring_layout(graph, 1, NODE_POS, NODE_POS.keys())
    xs, ys = [coords[0] for node, coords in pos.items() if node not in NODE_POS], [coords[1] for node, coords in pos.items() if node not in NODE_POS]
    if xs:
        mx, my = min(xs), min(ys)
        Mx, My = (max(xs) - mx) or 1, (max(ys) - my) or 1
        for node, coords in pos.items():
            if node not in NODE_POS.keys():
                pos[node] = [0.1 + 0.55 * (coords[0] - mx) / Mx, 1.15 * (coords[1] - my) / My - 0.75]
    node_colors = {
        # node: "lightgreen" if node in extranidal_nodes else "pink" for node in graph.nodes()
        node: pressure or 0 for node, pressure in graph.nodes("pressure")
    }
    edge_widths = [(edge[2]["Δpressure"] * 0.05 + 1) if "Δpressure" in edge[2] else 1 for edge in graph.edges(data = True)]
    edge_colors = [abs(edge[2]["flow"]) if "flow" in edge[2] else 1 for edge in graph.edges(data = True)]
    node_displays = nx.draw_networkx_nodes(graph, pos, node_color = [node_colors[node] for node in graph.nodes()], cmap = plt.cm.Reds)
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, width = edge_widths, edge_color = edge_colors, edge_cmap = plt.cm.cool)
    # edge_labels = { (edge[0], edge[1]): edge[2]["label"] for i, edge in enumerate(graph.edges(data = True)) }
    # edge_labels = { (edge[0], edge[1]): ((edge[2]["label"] + "\n") if edge[2]["label"] else "") + str(round(abs(flow[i, 0]), 3)) + "\n" + str(round(pressure[i, 0] * (-1 if flow[i, 0] < 0 else 1), 3)) for i, edge in enumerate(graph.edges(data = True)) }
    # edge_labels = { (edge[0], edge[1]): str(round(abs(edge[2]["flow"]), 3)) + "\n" + str(round(edge[2]["Δpressure"] * (-1 if edge[2]["flow"] < 0 else 1), 3)) for edge in graph.edges(data = True) }
    edge_labels = { (edge[0], edge[1]): str(round(abs(edge[2]["flow"]), 3)) for edge in graph.edges(data = True) }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    sm = plt.cm.ScalarMappable(cmap = plt.cm.cool, norm = plt.Normalize(vmin = min(edge_colors), vmax = max(edge_colors)))
    sm.set_array([])
    plt.colorbar(sm, label = "Flow (mL/min)")
    sm = plt.cm.ScalarMappable(cmap = plt.cm.Reds, norm = plt.Normalize(vmin = min(node_colors.values()), vmax = max(node_colors.values())))
    sm.set_array([])
    plt.colorbar(sm, label = "Pressure (mm Hg)")
    plt.show()

def simulate(intranidal_nodes: list, edges: list[list], pressures: dict[str, float], num_nidi: int, num_expected_edges: int):
    num_intranidal_verts = len(intranidal_nodes)
    probability = num_expected_edges / comb(num_intranidal_verts, 2)
    flow_pressures = []
    for i in range(num_nidi):
        graph = nx.Graph()
        for j in range(num_intranidal_verts - 1):
            # for k in range(2):
            #     connect = random.randint(0, num_intranidal_verts - 1)
            #     while graph.has_edge(intranidal_nodes[j], intranidal_nodes[connect]) or connect == j: connect = random.randint(0, num_intranidal_verts - 1)
            #     graph.add_edge(
            #         intranidal_nodes[j], intranidal_nodes[connect],
            #         start = intranidal_nodes[j], end = intranidal_nodes[connect],
            #         radius = 0.05, length = 5,
            #         resistance = 8 * VISCOSITY / np.pi * 5 / (0.05 ** 4), label = ""
            #     )

            for k in range(j + 1, num_intranidal_verts):
                if random.random() < probability:
                    radius, length = random.random() * (0.5 - 0.05) + 0.05, random.random() * (5 - 1) + 1
                    graph.add_edge(
                        intranidal_nodes[j], intranidal_nodes[k],
                        start = intranidal_nodes[j], end = intranidal_nodes[k],
                        radius = radius, length = length,
                        resistance = 8 * VISCOSITY / np.pi * length / (radius ** 4), label = ""
                    )
        graph.add_edges_from([(
            edge[0], edge[1], {
                "start": edge[0], "end": edge[1],
                "radius": edge[2], "length": edge[3],
                "resistance": edge[4], "label": edge[5],
                "locked": True
            }) for edge in edges])
        flow, pressure, all_edges = get_Q_and_P(graph, pressures)
        flow_pressures.append((flow, pressure))
        calc_pressures(graph, pressures)
        
        pnodes = [node for node in graph.nodes("pressure") if node[0] in intranidal_nodes]
        graph = nx.DiGraph(
            [
                (edge[2]["end" if edge[2]["flow"] < 0 else "start"], edge[2]["start" if edge[2]["flow"] < 0 else "end"],
                {
                    "radius": edge[2]["radius"],
                    "length": edge[2]["length"],
                    "resistance": edge[2]["resistance"],
                    "label": edge[2]["label"],
                    "flow": abs(edge[2]["flow"]) * 60, # cm^3/s = 60 mL/min
                    "Δpressure": abs(edge[2]["Δpressure"])
                }) for i, edge in enumerate(graph.edges(data = True))
                #  if edge[0] in intranidal_nodes and edge[1] in intranidal_nodes
            ]
        )
        graph.add_nodes_from([(node[0], { "pressure": node[1] }) for node in pnodes])
        # print([(graph.nodes[node]["pressure"], pressures[node]) for node in pressures])
        print([(graph.nodes[node], pressures[node]) for node in pressures if node in graph.nodes])
        display(graph)
    return flow_pressures

# Rv * Q = ΔΔP
# Matrix product representing satisfaction of Kirchoff's laws (solve to calculate flow Q)
def calc_flow(graph: nx.Graph, all_edges, p_ext):
    Rv = []
    ΔΔP = []
    num_edges = len(all_edges)

    # First law: sum of flow towards each node is 0
    ΔΔP = [0 for i in range(graph.number_of_nodes() - len(p_ext))]
    # ΔΔP = [0 for i in range(graph.number_of_nodes())]
    for i in graph.nodes():
        if i in p_ext:
            continue
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
    for cycle in nx.cycle_basis(graph): # nx.recursive_simple_cycles(graph)
        total_pressure = 0
        flow = [0 for _ in range(num_edges)]
        for i, node1 in enumerate(cycle):
            node2 = cycle[(i + 1) % len(cycle)]
            edge = (node1, node2)
            if edge in all_edges:
                index = all_edges.index(edge)
                flow[index] = graph.edges[edge]["resistance"]
                total_pressure += p_ext[node1] if node1 in p_ext else 0
            else:
                index = all_edges.index((edge[1], edge[0]))
                flow[index] = -graph.edges[edge]["resistance"]
                total_pressure -= p_ext[node1] if node1 in p_ext else 0
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
    
    result = np.linalg.lstsq(np.array(Rv), np.array([ΔΔP]).T, rcond = None)
    # result = lsq_linear(np.array(Rv), np.array(ΔΔP), bounds = ([0 if "locked" in graph.edges[edge] else -np.inf for edge in all_edges], [np.inf for edge in all_edges]))
    print(result[1])
    return result[0].flatten()


def get_Q_and_P(graph: nx.Graph, p_ext):
    all_edges = [(edge[2]["start"], edge[2]["end"]) for edge in graph.edges(data = True)]
    flow = calc_flow(graph, all_edges, p_ext)

    # flow = difference in pressure between ends / resistance
    # Hagen-Poiseuille equation
    pressure = flow * np.array([graph.edges[edge]["resistance"] for edge in all_edges])
    nx.set_edge_attributes(graph, { edge: flow[i] for i, edge in enumerate(all_edges) }, "flow")
    nx.set_edge_attributes(graph, { edge: pressure[i] for i, edge in enumerate(all_edges) }, "Δpressure")
    return flow, pressure, all_edges

# [first node, second node, resistance, label]
vessels = [
    # Cardiovasculature
    [13, "SP", 0.75, 10, 32, "superior vena cava"], # from heart up
    ["SP", 1, 1, 10, 1, "aortic arch"], # from brain to heart
    [1, 2, 0.35, 10, 67.9, "SCA"], # subclavian artery

    # Head and neck vasculature
        # Neck and extracranial circulation
        [1, 4, 0.35, 10, 67.9, "CCA"], # common cartid artery
        [4, 5, 0.2, 10, 637.5, "ECA"], # external carotid artery
        [5, 9, 0.01, 0.1, 1000000, ""], # capillary bed data not given; arbitrary data
        [9, 10, 0.01, 0.1, 1000000, ""],
        [10, 11, 0.125, 10, 4177.9, ""],
        [12, "CVP", 0.4, 10, 79.7 / 2, "CVP"], # central venous pressure
        ["CVP", 13, 0.4, 10, 79.7 / 2, "jugular veins"], # i divided the length in 2; the original paper omits the CVP node in the model data; external pressure seems to have been defined separately from nodes?
        # Intracranial circulation
        [4, 6, 0.25, 20, 522, "ICA"], #internal carotid artery
        [2, 3, 0.15, 25, 5037, "VA"], # vertebral artery
        [6, 7, 0.1, 10, 10200, ""],
        [7, 8, 0.01, 0.1, 1000000, ""],
        [8, 11, 0.125, 10, 4177.9, ""],
        [11, 12, 0.25, 10, 261, "dural venous sinuses"],
    
    # AVM vasculature
        # Major arterial feeders
        [3, "AF1", 0.125, 5.2, 2210, "PCA"], # posterior cerebral artery
        [6, "AF2", 0.15, 3.7, 745.5, "MCA"], # middle cerebral artery
        # Minor arterial feeders
        [6, "AF3", 0.025, 3.7, 15725000, "ACA"], # anterior cerebral artery
        [9, "AF4", 0.0125, 3, 12750000, "TFA"], # transdural feeding artery
        # Fistulous nidus vessels
        ["AF2", 14, 0.1, 4, 4080, ""],
        [14, 15, 0.1, 4, 4080, ""],
        [15, 16, 0.1, 4, 4080, ""],
        [16, "DV2", 0.1, 4, 4080, ""],
        # Draining veins
        ["DV1", 11, 0.25, 5, 130.5, ""],
        ["DV2", 11, 0.25, 5, 130.5, ""],
        ["DV3", 11, 0.25, 5, 130.5, ""],
]
pressures = {
    "SP": 74,
    "AF1": 47,
    "AF2": 47,
    "AF3": 50,
    "AF4": 50,
    "DV1": 17,
    "DV2": 17,
    "DV3": 17,
    "CVP": 5
}
intranidal_nodes = ["AF1", "AF2", "AF3", "AF4"] + list(range(14, 14 + 57 - 4 - 3)) + ["DV1", "DV2", "DV3"]
# intranidal_nodes = ["AF1", "AF2", "AF3", "AF4"] + list(range(14, 17)) + ["DV1", "DV2", "DV3"]
# intranidal_nodes = ["AF1", "AF2", "AF3", "AF4", "DV1", "DV2", "DV3"]
# intranidal_nodes = []
# extranidal_nodes = list(set(vessel[0] for vessel in vessels) | set(vessel[1] for vessel in vessels))
print(simulate(intranidal_nodes, vessels, pressures, 1, 93))
