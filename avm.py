from scipy.special import comb
import random
import networkx as nx
import cupy as cp

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
        flow_pressures.append(get_Q_and_P(graph, resistances, p_ext))
    return flow_pressures

# Rv * Q = ΔΔP
# Matrix product representing satisfaction of Kirchoff's laws (solve to calculate flow Q)
def get_Rv(cycles, resistances, p_ext):


def get_Q_and_P(graph: nx.Graph, resistances, p_ext):
    cycles = nx.cycle_basis(graph)
    ΔΔP = get_ΔΔP(cycles)
    Rv = get_Rv(cycles, resistances, p_ext)
    flow = cp.linalg.lstsq(Rv, ΔΔP)

    # flow = difference in pressure between ends / resistance
    # Hagen-Poiseuille equation
    pressure = flow * resistances
    return flow, pressure
