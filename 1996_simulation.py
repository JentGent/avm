"""Inputs for the simulation of the network described in the 1996 paper."""

import avm
import numpy as np
import matplotlib.pyplot as plt
import random

# NODE_POS lists the positions of specific nodes in the graph.
NODE_POS = {
    1: [22, -161],
    2: [126, -236],
    3: [207, -331],
    4: [127, -152],
    5: [180, -33],
    6: [216, -209],
    7: [242, -62],
    8: [491, -64],
    9: [350, -21],
    10: [544, -22],
    11: [589, -308],
    12: [272, -473],
    13: [274, -160],
    14: [268, -292],
    15: [317, -254],
    16: [325, -303],
    17: [304, -364],
    18: [373, -117],
    19: [381, -207],
    20: [384, -272],
    21: [392, -342],
    22: [390, -426],
    23: [389, -518],
    24: [431, -156],
    25: [441, -240],
    26: [447, -303],
    27: [450, -382],
    28: [441, -466],
    29: [513, -237],
    30: [522, -361],
    31: [579, -565],
    32: [401, -569],
    "SP": [22, -544],
}

# VESSELS is formatted like [first node, second node, radius, length, resistance, label, type (optional)].
VESSELS = [
    [1, 2, 0, 0, 67.9, "R1"],
    [1, 4, 0, 0, 67.9, "R2"],
    [2, 3, 0, 0, 5037, "R3"],
    [3, 12, 0, 0, 2210, "R4", avm.vessel.feeder],
    [4, 5, 0, 0, 637.5, "R5"],
    [5, 9, 0, 0, 1000000, "R6"],
    [9, 18, 0, 0, 12750000, "R7"],
    [9, 10, 0, 0, 1000000, "R8"],
    [10, 11, 0, 0, 4177.9, "R9"],
    [4, 6, 0, 0, 522, "R10"],
    [6, 14, 0, 0, 745.5, "R11", avm.vessel.feeder],
    [6, 13, 0, 0, 15725000, "R12", avm.vessel.feeder],
    [6, 7, 0, 0, 10200, "R13"],
    [7, 8, 0, 0, 1000000, "R14", avm.vessel.feeder],
    [8, 11, 0, 0, 4177.9, "R15"],
    [13, 18, 0, 0, 81600, "R16", avm.vessel.plexiform],
    [13, 19, 0, 0, 81600, "R17", avm.vessel.plexiform],
    [14, 15, 0, 0, 81600, "R18", avm.vessel.plexiform],
    [14, 16, 0, 0, 4080, "R19", avm.vessel.fistulous],  # Fistulous
    [14, 17, 0, 0, 81600, "R20", avm.vessel.plexiform],
    [15, 19, 0, 0, 81600, "R21", avm.vessel.plexiform],
    [15, 20, 0, 0, 81600, "R22", avm.vessel.plexiform],
    [16, 20, 0, 0, 81600, "R23", avm.vessel.plexiform],
    [16, 21, 0, 0, 4080, "R24", avm.vessel.fistulous],  # Fistulous
    [17, 21, 0, 0, 81600, "R25", avm.vessel.plexiform],
    [17, 22, 0, 0, 81600, "R26", avm.vessel.plexiform],
    [12, 22, 0, 0, 81600, "R27", avm.vessel.plexiform],
    [12, 23, 0, 0, 81600, "R28", avm.vessel.plexiform],
    [23, 28, 0, 0, 81600, "R29", avm.vessel.plexiform],
    [22, 28, 0, 0, 81600, "R30", avm.vessel.plexiform],
    [22, 27, 0, 0, 81600, "R31", avm.vessel.plexiform],
    [21, 27, 0, 0, 4080, "R32", avm.vessel.fistulous],  # Fistulous
    [21, 26, 0, 0, 81600, "R33", avm.vessel.plexiform],
    [20, 26, 0, 0, 81600, "R34", avm.vessel.plexiform],
    [20, 25, 0, 0, 81600, "R35", avm.vessel.plexiform],
    [19, 25, 0, 0, 81600, "R36", avm.vessel.plexiform],
    [19, 24, 0, 0, 81600, "R37", avm.vessel.plexiform],
    [18, 24, 0, 0, 81600, "R38", avm.vessel.plexiform],
    [24, 29, 0, 0, 81600, "R39", avm.vessel.plexiform],
    [25, 29, 0, 0, 81600, "R40", avm.vessel.plexiform],
    [26, 29, 0, 0, 81600, "R41", avm.vessel.plexiform],
    [26, 30, 0, 0, 81600, "R42", avm.vessel.plexiform],
    [27, 30, 0, 0, 4080, "R43", avm.vessel.fistulous],  # Fistulous
    [28, 30, 0, 0, 81600, "R44", avm.vessel.plexiform],
    [29, 11, 0, 0, 130.5, "R45", avm.vessel.drainer],
    [30, 11, 0, 0, 130.5, "R46", avm.vessel.drainer],
    [11, 31, 0, 0, 261, "R47"],
    [31, 32, 0, 0, 79.7, "R48"],
    [32, "SP", 0, 0, 3.2, "R49"],
    ["SP", 1, 0, 0, 1, "R50"],
]
for i in range(len(VESSELS)):
    if random.random() < 0.5:
        [VESSELS[i][0], VESSELS[i][1]] = [VESSELS[i][1], VESSELS[i][0]]

# PRESSURES is a dictionary of known node : pressure values.
PRESSURES = {
    (32, "SP"): 74 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (3, 12): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (6, 14): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (6, 13): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (9, 18): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (30, 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (29, 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (31, 32): 5 * avm.MMHG_TO_DYN_PER_SQUARE_CM
}

# INTRANIDAL_NODES is a list of nodes in the nidus.
INTRANIDAL_NODES = ["AF1", "AF2", "AF3", "AF4"] + list(range(12, 31)) + ["DV1", "DV2", "DV3"]


def main():
    network = avm.edges_to_graph(VESSELS)
    flow, pressure, _, graph = avm.simulate(network, [], PRESSURES)
    print(f"Number of flow values before removing duplicates: {flow.shape}")
    flow = flow[np.unique(np.round(flow / 0.00000001) * 0.00000001, return_index=True)[1]]
    flow = np.append(flow, 820.71 / 60)
    print(f"After: {flow.shape}")

    for key, value in avm.get_stats(graph).items():
        print(f"{key}: {value}")
    avm.display(graph, INTRANIDAL_NODES, NODE_POS)
    plt.show()


"""
Vessel flow range: (0.4393220698311891, 820.7097137783559) mL/min
Average flow: 194.9142655521916 mL/min
Total flow through nidus (out): 812.4164478461618 mL/min
Total flow through nidus (in): 812.4164474832979 mL/min
Fistulous flow range: (595.2160496653493, 638.8651971709985) mL/min
Minimum pressure: 0.010259743500426985
Maximum pressure: 99.2215861316942
Average pressure: 23.544885087855988
"""


if __name__ == "__main__":
    main()
