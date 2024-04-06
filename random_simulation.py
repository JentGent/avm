"""Generates a random nidus."""

import avm
import numpy as np
import matplotlib.pyplot as plt
import generate

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
    [3, 14, 0.125, 5.2, 2210, "PCA", avm.vessel.feeder],  # posterior cerebral artery
    [6, 15, 0.15, 3.7, 745.5, "MCA", avm.vessel.feeder],  # middle cerebral artery
    # Minor arterial feeders
    [6, 16, 0.025, 3.7, 15725000, "ACA", avm.vessel.feeder],  # anterior cerebral artery
    [9, 17, 0.0125, 3, 12750000, "TFA", avm.vessel.feeder],  # transdural feeding artery
    # Fistulous nidus vessels
    [15, 18, 0.1, 4, 40800, "", True, avm.vessel.fistulous],
    [18, 19, 0.1, 4, 40800, "", True, avm.vessel.fistulous],
    [19, 20, 0.1, 4, 40800, "", True, avm.vessel.fistulous],
    [20, 22, 0.1, 4, 40800, "", True, avm.vessel.fistulous],
    # Draining veins
    [21, 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
    [22, 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
    [23, 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
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
    network = generate.gilbert(network, INTRANIDAL_NODES, 1000)
    flow, pressure, _, graph = avm.simulate(network, INTRANIDAL_NODES, PRESSURES)
    for key, value in avm.get_stats(graph).items():
        print(f"{key}: {value}")
    avm.display(graph, INTRANIDAL_NODES, NODE_POS)
    plt.show()


if __name__ == "__main__":
    main()
