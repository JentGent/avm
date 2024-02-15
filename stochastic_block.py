"""Uses a stochastic block model to randomly generate a nidus."""

import avm
import numpy as np
from scipy.special import comb
import scipy.optimize as spop
import matplotlib.pyplot as plt

# FISTULOUS_RESISTANCE is just for testing purposes; the reported value in the paper is 4080.
FISTULOUS_RESISTANCE = 4080

# NUM_INTRANIDAL_NODES is the number of intranidal nodes. (minimum 10)
NUM_INTRANIDAL_NODES = 57

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
    
    18: [0.75, -0.85], # DV1
    19: [0.75, 0], # DV2
    20: [0.75, 0.4], # DV3

    21: [0.1875, -0.1], # Fistulous
    22: [0.375, 0.1],
    23: [0.5625, 0.2],

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
    [15, 21, 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    [21, 22, 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    [22, 23, 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    [23, 19, 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    # Draining veins
    [18, 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
    [19, 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
    [20, 11, 0.25, 5, 130.5, "", avm.vessel.drainer],

    [14, 24, 0.05, 5, 81600, "", avm.vessel.plexiform],
    [15, 25, 0.05, 5, 81600, "", avm.vessel.plexiform],
    [16, 26, 0.05, 5, 81600, "", avm.vessel.plexiform],
    [17, 27, 0.05, 5, 81600, "", avm.vessel.plexiform],

    [21 + NUM_INTRANIDAL_NODES - 1, 18, 0.05, 5, 81600, "", avm.vessel.plexiform],
    [21 + NUM_INTRANIDAL_NODES - 2, 19, 0.05, 5, 81600, "", avm.vessel.plexiform],
    [21 + NUM_INTRANIDAL_NODES - 3, 20, 0.05, 5, 81600, "", avm.vessel.plexiform],
]

# PRESSURES is a dictionary of known node : pressure values.
PRESSURES = {
    (13, "SP"): 74 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (3, 14): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (6, 15): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (6, 16): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (9, 17): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (18, 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (19, 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (20, 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (12, 13): 5 * avm.MMHG_TO_DYN_PER_SQUARE_CM
}

# INTRANIDAL_NODES is a list of nodes in the nidus.
INTRANIDAL_NODES = list(range(21, 21 + NUM_INTRANIDAL_NODES))

def calc_expected_edges(p, sizes):
    n = 0
    index = 0
    for i in range(len(sizes)):
        n += p[index] * sizes[i] * (sizes[i] - 1) / 2
        index += 1
        for j in range(i + 1, len(sizes)):
            n += p[index] * sizes[i] * sizes[j]
            index += 1
    return n
def zero_p(res, sizes):
    p = []
    index = 0
    AF, FI, DV = 0, 1, 2
    for block1 in range(len(sizes)):
        for block2 in range(block1, len(sizes)):
            if block1 == AF:
                # p.append(0)
                if block2 not in [AF, FI, DV]:
                    p.append(res[index])
                    index += 1
                else: p.append(0)
            elif block1 == FI:
                if block2 not in [FI, DV]:
                    p.append(res[index])
                    index += 1
                else: p.append(0)
            elif block1 == DV:
                if block2 not in [DV]:
                    p.append(res[index])
                    index += 1
                else: p.append(0)
            else:
                p.append(res[index])
                index += 1
    return p
def generate_stochastic_matrix(sizes: list[int], num_expected_edges: int):
    p = []
    t = num_expected_edges / comb(sum(sizes), 2)
    AF, FI, DV = 0, 1, 2
    index = 0
    M, C, P = 10, 2, 10
    bias = []
    for block1 in range(len(sizes)):
        for block2 in range(block1, len(sizes)):
            if block1 == AF:
                # do_nothing = 0
                if block2 not in [AF, FI, DV]:
                    p.append(t / (C * (2 - block1) ** P + 1))
                    bias.append([index, M * (P ** ((block2 - 2) ** C) - 1)])
                    index += 1
            elif block1 == FI:
                if block2 not in [FI, DV]:
                    p.append(0.2)
                    index += 1
            elif block1 == DV:
                if block2 not in [DV]:
                    p.append(t / (C * (block2 - len(sizes)) ** P + 1))
                    bias.append([index, M * (P ** ((block2 - len(sizes)) ** C) - 1)])
                    index += 1
            else:
                p.append(t / ((block2 - block1) ** P + 1))
                bias.append([index, M * (P ** (block2 - block1) ** C) - 1])
                index += 1
    res = spop.minimize(lambda x: (calc_expected_edges(zero_p(x, sizes), sizes) - num_expected_edges) ** 2 + sum([x[i] * a for i, a in bias]) * 10, p, bounds = spop.Bounds(0, 1))
    print(res)
    p = zero_p(res.x, sizes)
    print(calc_expected_edges(p, sizes))
    index = 0
    matrix = []
    for block1, size1 in enumerate(sizes):
        matrix.append([])
        for block2, size2 in enumerate(sizes[:block1]):
            matrix[block1].append(matrix[block2][block1])
        for block2, size2 in enumerate(sizes[block1:]):
            matrix[block1].append(p[index])
            index += 1
    print(np.array(matrix))
    return matrix

def main():
    network = avm.edges_to_graph(VESSELS)
    # network = avm.generate_nidus(network, INTRANIDAL_NODES, 1000)
    # 47
    blocks = [4, 3, 3] + [10, 10, 10, 10, 7]
    network = avm.generate_nidus_stochastic(network, INTRANIDAL_NODES, blocks, generate_stochastic_matrix(blocks, 1000))
    # network = avm.generate_nidus_linear(network, INTRANIDAL_NODES)
    flow, pressure, _, graph = avm.simulate(network, INTRANIDAL_NODES, PRESSURES)

    for key, value in avm.get_stats(graph).items():
        print(f"{key}: {value}")
    graph.remove_nodes_from([i for i in range(1, 14)] + ["SP"])
    plt.figure(1)
    avm.display(graph, INTRANIDAL_NODES, NODE_POS, color_is_flow = False)
    plt.figure(2)
    avm.display(graph, INTRANIDAL_NODES, NODE_POS)
    plt.show()

if __name__ == "__main__":
    main()
