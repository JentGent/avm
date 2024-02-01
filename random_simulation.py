"""Inputs for the simulation of the network described in the 1996 paper."""

import avm
import numpy as np

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
    13: [1, -1],
    14: [0, -0.5], # AF1
    15: [0, 0], # AF2
    16: [0, 0.5], # AF3
    17: [0.5, 0.5], # AF4

    18: [0.1875, -0.1], # Fistulous
    19: [0.375, 0.1],
    20: [0.5625, 0.2],
    
    21: [0.75, -0.85], # DV1
    22: [0.75, 0], # DV2
    23: [0.75, 0.4], # DV3

    "SP": [-1, -1],
}

# VESSELS is formatted like [first node, second node, radius, length, resistance, label, fistulous (optional)].
VESSELS = [
    # Cardiovasculature
    [13, "SP", 0.75, 10, 3.2, "superior vena cava"],  # from heart up
    ["SP", 1, 1, 10, 1, "aortic arch"],  # from brain to heart
    [1, 2, 0.35, 10, 67.9, "SCA"],  # subclavian artery

    # Head and neck vasculature
    # Neck and extracranial circulation
    [1, 4, 0.35, 10, 67.9, "CCA"],  # common cartid artery
    [4, 5, 0.2, 10, 637.5, "ECA"],  # external carotid artery
    # capillary bed data not given; arbitrary data
    [5, 9, 0.01, 0.1, 1000000, ""],
    [9, 10, 0.01, 0.1, 1000000, ""],
    [10, 11, 0.125, 10, 4177.9, ""],
    [12, 13, 0.4, 20, 79.7, "jugular veins"],  # central venous pressure
    # Intracranial circulation
    [4, 6, 0.25, 20, 522, "ICA"],  # internal carotid artery
    [2, 3, 0.15, 25, 5037, "VA"],  # vertebral artery
    [6, 7, 0.1, 10, 10200, ""],
    [7, 8, 0.01, 0.1, 1000000, ""],
    [8, 11, 0.125, 10, 4177.9, ""],
    [11, 12, 0.25, 10, 261, "dural venous sinuses"],

    # AVM vasculature
    # Major arterial feeders
    [3, 14, 0.125, 5.2, 2210, "PCA"],  # posterior cerebral artery
    [6, 15, 0.15, 3.7, 745.5, "MCA"],  # middle cerebral artery
    # Minor arterial feeders
    [6, 16, 0.025, 3.7, 15725000, "ACA"],  # anterior cerebral artery
    [9, 17, 0.0125, 3, 12750000, "TFA"],  # transdural feeding artery
    # Fistulous nidus vessels
    [15, 18, 0.1, 4, 40800, "", True],
    [18, 19, 0.1, 4, 40800, "", True],
    [19, 20, 0.1, 4, 40800, "", True],
    [20, 22, 0.1, 4, 40800, "", True],
    # Draining veins
    [21, 11, 0.25, 5, 130.5, ""],
    [22, 11, 0.25, 5, 130.5, ""],
    [23, 11, 0.25, 5, 130.5, ""],
]

# PRESSURES is a dictionary of known node : pressure values.
PRESSURES = {
    (13, "SP"): 74 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (3, 14): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (6, 15): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (6, 16): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (9, 17): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (21, 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (22, 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (23, 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (12, 13): 5 * avm.MMHG_TO_DYN_PER_SQUARE_CM
}

# NUM_INTRANIDAL_NODES is the number of intranidal nodes. (minimum 10)
NUM_INTRANIDAL_NODES = 57

# INTRANIDAL_NODES is a list of nodes in the nidus.
INTRANIDAL_NODES = list(range(14, 14 + NUM_INTRANIDAL_NODES))


def main():
    network = avm.edges_to_graph(VESSELS)
    network = avm.generate_nidus(network, INTRANIDAL_NODES, 1000)
    flow, pressure, _, graph = avm.simulate(network, INTRANIDAL_NODES, PRESSURES)

    # print(f"Number of flow values: {flow.shape}")
    # print(f"Number of edges: {network.number_of_edges()}")
    # print(f"Flows: {np.round(flow * 60, 3)}")
    print(f"Vessel flow range: ({np.min(np.abs(flow * 60))}, {np.max(np.abs(flow * 60))}) mL/min")
    print(f"Average flow: {np.average(np.abs(flow * 60))} mL/min")
    print(f"Total flow through nidus (out): {graph[21][11]['flow'] + graph[22][11]['flow'] + graph[23][11]['flow']} mL/min")
    print(f"Total flow through nidus (in): {graph[3][14]['flow'] + graph[6][15]['flow'] + graph[6][16]['flow'] + graph[9][17]['flow']} mL/min")

    # print(f"Minimum pressure: {np.min(np.abs(pressure / avm.MMHG_TO_DYNCM))} mmHg")
    # print(f"Maximum pressure: {np.max(np.abs(pressure / avm.MMHG_TO_DYNCM))} mmHg")
    # print(f"Average pressure: {np.average(np.abs(pressure / avm.MMHG_TO_DYNCM))} mmHg")

    fistulous = [edge[2] for edge in graph.edges(data = True) if "fistulous" in edge[2] and edge[2]["fistulous"]]
    fistulous_pressures = [edge["Δpressure"] for edge in fistulous]
    fistulous_flows = [edge["flow"] for edge in fistulous]
    print(f"Number of fistulous vessels: {len(fistulous)}")
    print(f"Fistulous flow range: ({min(fistulous_flows)}, {max(fistulous_flows)}) mL/min")
    print(f"Fistulous flow average: {np.average(fistulous_flows)} mL/min")
    print(f"Fistulous pressure range: ({min(fistulous_pressures)}, {max(fistulous_pressures)}) mmHg")
    print(f"Fistulous pressure average: ({np.average(fistulous_pressures)}) mmHg")

    plexiform = [edge[2] for edge in graph.edges(data = True) if "plexiform" in edge[2] and edge[2]["plexiform"]]
    plexiform_pressures = [edge["Δpressure"] for edge in plexiform]
    plexiform_flows = [edge["flow"] for edge in plexiform]
    print(f"Number of plexiform vessels: {len(plexiform)}")
    print(f"Plexiform flow range: ({min(plexiform_flows)}, {max(plexiform_flows)}) mL/min")
    print(f"Plexiform flow average: {np.average(plexiform_flows)} mL/min")
    print(f"Plexiform pressure range: ({min(plexiform_pressures)}, {max(plexiform_pressures)}) mmHg")
    print(f"Plexiform pressure average: ({np.average(plexiform_pressures)}) mmHg")
    # avm.display(graph, INTRANIDAL_NODES, NODE_POS)


if __name__ == "__main__":
    main()
