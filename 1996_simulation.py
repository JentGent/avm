"""Inputs for the simulation of the network described in the 1996 paper."""

import avm
import numpy as np

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
    [13, 18, 0, 0, 81600, "R16"],
    [13, 19, 0, 0, 81600, "R17"],
    [14, 15, 0, 0, 81600, "R18"],
    [14, 16, 0, 0, 4080, "R19"],  # Fistulous
    [14, 17, 0, 0, 81600, "R20"],
    [15, 19, 0, 0, 81600, "R21"],
    [15, 20, 0, 0, 81600, "R22"],
    [16, 20, 0, 0, 81600, "R23"],
    [16, 21, 0, 0, 4080, "R24"],  # Fistulous
    [17, 21, 0, 0, 81600, "R25"],
    [17, 22, 0, 0, 81600, "R26"],
    [12, 22, 0, 0, 81600, "R27"],
    [12, 23, 0, 0, 81600, "R28"],
    [23, 28, 0, 0, 81600, "R29"],
    [22, 28, 0, 0, 81600, "R30"],
    [22, 27, 0, 0, 81600, "R31"],
    [21, 27, 0, 0, 4080, "R32"],  # Fistulous
    [21, 26, 0, 0, 81600, "R33"],
    [20, 26, 0, 0, 81600, "R34"],
    [20, 25, 0, 0, 81600, "R35"],
    [19, 25, 0, 0, 81600, "R36"],
    [19, 24, 0, 0, 81600, "R37"],
    [18, 24, 0, 0, 81600, "R38"],
    [24, 29, 0, 0, 81600, "R39"],
    [25, 29, 0, 0, 81600, "R40"],
    [26, 29, 0, 0, 81600, "R41"],
    [26, 30, 0, 0, 81600, "R42"],
    [27, 30, 0, 0, 4080, "R43"],  # Fistulous
    [28, 30, 0, 0, 81600, "R44"],
    [29, 11, 0, 0, 130.5, "R45"],
    [30, 11, 0, 0, 130.5, "R46"],
    [11, 31, 0, 0, 261, "R47"],
    [31, 32, 0, 0, 79.7, "R48"],
    [32, "SP", 0, 0, 3.2, "R49"],
    ["SP", 1, 0, 0, 1, "R50"],
]

# PRESSURES is a dictionary of known node : pressure values.
PRESSURES = {
    (32, "SP"): 74 * avm.MMHG_TO_DYNCM,
    (3, 12): 47 * avm.MMHG_TO_DYNCM,
    (6, 14): 47 * avm.MMHG_TO_DYNCM,
    (6, 13): 50 * avm.MMHG_TO_DYNCM,
    (9, 18): 50 * avm.MMHG_TO_DYNCM,
    (30, 11): 17 * avm.MMHG_TO_DYNCM,
    (29, 11): 17 * avm.MMHG_TO_DYNCM,
    (31, 32): 5 * avm.MMHG_TO_DYNCM
}

# INTRANIDAL_NODES is a list of nodes in the nidus.
INTRANIDAL_NODES = ["AF1", "AF2", "AF3", "AF4"] + list(range(12, 31)) + ["DV1", "DV2", "DV3"]


def main():
    network = avm.edges_to_graph(VESSELS)
    flow, pressure, graph = avm.simulate(network, [], PRESSURES)
    print(len(flow))
    flow = avm.unique_with_tolerance(flow, 0.00000001)
    flow = np.append(flow, 820.71 / 60)
    print(len(flow))

    print(f"Flows: {np.round(flow * 60, 3)}")
    print(f"Vessel flow range: ({np.min(np.abs(flow * 60))}, {np.max(np.abs(flow * 60))}) mL/min")
    print(f"Average flow: {np.average(np.abs(flow * 60))} mL/min")
    print(f"Total flow through nidus (out): {graph[29][11]['flow'] + graph[30][11]['flow']} mL/min")
    print(f"Total flow through nidus (in): {graph[3][12]['flow'] + graph[6][14]['flow'] + graph[6][13]['flow'] + graph[9][18]['flow']} mL/min")
    print(f"Fistulous flow range: ({min(graph[14][16]['flow'], graph[16][21]['flow'], graph[21][27]['flow'], graph[27][30]['flow'])}, {max(graph[14][16]['flow'], graph[16][21]['flow'], graph[21][27]['flow'], graph[27][30]['flow'])}) mL/min")
    print(f"Minimum pressure: {np.min(np.abs(pressure / avm.MMHG_TO_DYNCM))}")
    print(f"Maximum pressure: {np.max(np.abs(pressure / avm.MMHG_TO_DYNCM))}")
    print(f"Average pressure: {np.average(np.abs(pressure / avm.MMHG_TO_DYNCM))}")
    avm.display(graph, INTRANIDAL_NODES, NODE_POS)

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
