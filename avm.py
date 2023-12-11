from scipy.special import comb
import random
import networkx as nx
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

INTRANIDAL_RESISTANCE = 1.5

# [intranidal nodes, extranidal]
# [extranidal edges, intranidal]

def simulate(num_intranidal_verts, num_extranidal_verts, extranidal_edges, resistances, p_ext, num_nidi, num_expected_edges):
    probability = num_expected_edges / comb(num_intranidal_verts, 2)
    flow_pressures = []
    for i in range(num_nidi):
        intranidal_edges = []
        for j in range(num_intranidal_verts - 1):
            for k in range(j + 1, num_intranidal_verts):
                if random.random() < probability:
                    intranidal_edges.append((j, k))
        graph = nx.Graph(extranidal_edges + intranidal_edges)
        p_ext = p_ext + [0 for i in range(len(p_ext), graph.number_of_edges())]
        flow, pressure = get_Q_and_P(graph, num_intranidal_verts, num_extranidal_verts, resistances + [INTRANIDAL_RESISTANCE for i in range(len(resistances), graph.number_of_edges())], p_ext)
        flow_pressures.append((flow, pressure))
        pos = nx.spring_layout(graph)
        colors = ["red" if i < num_intranidal_verts else "green" for i in range(num_intranidal_verts + num_extranidal_verts)]
        nx.draw_networkx_nodes(graph, pos, node_color = [colors[node] for node in graph.nodes()])
        nx.draw_networkx_labels(graph, pos)
        nx.draw_networkx_edges(graph, pos)
        edge_labels = { edge: str(round(flow[i, 0], 3)) + "\n" + str(round(pressure[i, 0], 3)) for i, edge in enumerate(graph.edges()) }
        nx.draw_networkx_edge_labels(graph, pos, edge_labels)
        plt.show()
    return flow_pressures

# Rv * Q = ΔΔP
# Matrix product representing satisfaction of Kirchoff's laws (solve to calculate flow Q)
def get_ΔΔP(graph: nx.Graph, num_intranidal_verts, num_extranidal_verts, cycles, resistances, p_ext):
    # First law: sum of flow towards each node is 0
    # Second law: total pressure change after traversing any closed loop is 0
    # ==> The only nonzero values of ΔΔP are those where there is an external pressure source
    all_edges = list(graph.edges())
    ΔΔP = [0 for i in range(num_intranidal_verts)]
    for cycle in cycles:
        total_pressure = 0
        for i, node1 in enumerate(cycle):
            edge = (node1, cycle[(i + 1) % len(cycle)])
            total_pressure += p_ext[all_edges.index(edge if edge in all_edges else (edge[1], edge[0]))]
        ΔΔP.append(total_pressure)
    return np.array([ΔΔP]).T

def get_Rv(graph: nx.Graph, num_intranidal_verts, num_extranidal_verts, cycles, resistances, p_ext):
    Rv = []
    all_edges = list(graph.edges())
    num_edges = len(all_edges)

    # first law
    for i in range(num_intranidal_verts):
        edges = graph.edges([i])
        flow = [0 for _ in range(num_edges)]
        for edge in edges:
            if edge in all_edges:
                index = all_edges.index(edge)
                flow[index] += 1
            else:
                index = all_edges.index((edge[1], edge[0]))
                flow[index] -= 1
        Rv.append(flow)
    
    # second law
    for cycle in cycles:
        flow = [0 for _ in range(num_edges)]
        for i, node1 in enumerate(cycle):
            node2 = cycle[(i + 1) % len(cycle)]
            edge = (node1, node2)
            if edge in all_edges:
                index = all_edges.index(edge)
                flow[index] += resistances[index]
            else:
                index = all_edges.index((edge[1], edge[0]))
                flow[index] -= resistances[index]
        Rv.append(flow)
    return np.array(Rv)


def get_Q_and_P(graph: nx.Graph, num_intranidal_verts, num_extranidal_verts, resistances, p_ext):
    # cycles = nx.recursive_simple_cycles(graph)
    cycles = nx.cycle_basis(graph)
    ΔΔP = get_ΔΔP(graph, num_intranidal_verts, num_extranidal_verts, cycles, resistances, p_ext)
    Rv = get_Rv(graph, num_intranidal_verts, num_extranidal_verts, cycles, resistances, p_ext)
    flow = np.linalg.lstsq(Rv, ΔΔP)[0]

    # flow = difference in pressure between ends / resistance
    # Hagen-Poiseuille equation
    pressure = flow * np.array([resistances]).T
    return flow, pressure

NUM_INTRA = 10
NUM_EXTRA = 3
print(simulate(NUM_INTRA, NUM_EXTRA, [(NUM_INTRA, 0), (NUM_INTRA + 1, 1), (NUM_INTRA - 1, NUM_INTRA + 2)], [1, 0.5, 2], [2, 1, -1], 1, 20))
