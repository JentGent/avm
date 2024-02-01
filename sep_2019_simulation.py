"""Recreates the September 2019 network."""

import avm
import numpy as np

# FISTULOUS_RESISTANCE is just for testing purposes; the reported value in the paper is 4080.
FISTULOUS_RESISTANCE = 4080

# SIMULATIONS lists pressures for the 72 different TRENSH injection simulations.
SIMULATIONS = []

# NODE_POS lists the positions of specific nodes in the graph.
NODE_POS = {
    1: [10, -200],
    2: [50, -250],
    3: [75, -300],
    4: [50, -200],
    5: [100, -100],
    6: [150, -275],
    7: [150, -120],
    8: [500, -120],
    9: [350, -100],
    10: [550, -100],
    11: [580, -300],
    12: [580, -590],
    13: [300, -590],
    14: [275, -370],
    15: [339, -359],
    16: [425, -350],
    17: [266, -160],
    18: [264, -185],
    19: [267, -204],
    20: [265, -224],
    21: [248, -257],
    22: [272, -313],
    23: [275, -346],
    24: [273, -393],
    25: [274, -422],
    26: [272, -466],
    27: [267, -492],
    28: [269, -512],
    29: [267, -531],
    30: [271, -560],
    31: [348, -172],
    32: [347, -193],
    33: [348, -215],
    34: [350, -241],
    35: [337, -271],
    36: [362, -302],
    37: [338, -327],
    38: [350, -383],
    39: [351, -406],
    40: [362, -433],
    41: [358, -456],
    42: [347, -481],
    43: [345, -503],
    44: [348, -524],
    45: [349, -549],
    46: [349, -574],
    47: [430, -161],
    48: [430, -183],
    49: [429, -207],
    50: [433, -232],
    51: [433, -255],
    52: [423, -317],
    53: [426, -375],
    54: [428, -399],
    55: [427, -421],
    56: [427, -469],
    57: [427, -493],
    58: [426, -513],
    59: [426, -541],
    60: [429, -564],
    61: [492, -187],
    62: [499, -408],
    63: [499, -480],
    "SP": [10, -590],
    "AF1": [200, -500],
    "AF2": [200, -370],
    "AF3": [200, -200],
    "AF4": [350, -150],
    "DV1": [501, -543],
    "DV2": [500, -350],
    "DV3": [500, -250],
}

# VESSELS elements are formatted like [first node, second node, radius (cm), length (cm), resistance (dyn s / cm^5), label, fistulous (optional)].
VESSELS = [
    [13, "SP", 0.75, 10, 3.2, "superior vena cava"],
    ["SP", 1, 1, 10, 1, "aortic arch"],
    [1, 2, 0.35, 10, 67.9, "SCA"],
    [1, 4, 0.35, 10, 67.9, "CCA"],
    [4, 5, 0.2, 10, 637.5, "ECA"],
    [5, 9, 0.01, 0.1, 1000000, ""],
    [9, 10, 0.01, 0.1, 1000000, ""],
    [10, 11, 0.125, 10, 4177.9, ""],
    [12, 13, 0.4, 20, 79.7, "jugular veins"],
    [4, 6, 0.25, 20, 522, "ICA"],
    [2, 3, 0.15, 25, 5037, "VA"],
    [6, 7, 0.1, 10, 10200, ""],
    [7, 8, 0.01, 0.1, 1000000, ""],
    [8, 11, 0.125, 10, 4177.9, ""],
    [11, 12, 0.25, 10, 261, "dural venous sinuses"],
    [3, "AF1", 0.125, 5.2, 2210, "PCA"],
    [6, "AF2", 0.15, 3.7, 745.5, "MCA"],
    [6, "AF3", 0.025, 3.7, 15725000, "ACA"],
    [9, "AF4", 0.0125, 3, 12750000, "TFA"],
    ["AF2", 14, 0.1, 4, FISTULOUS_RESISTANCE, "", True],
    [14, 15, 0.1, 4, FISTULOUS_RESISTANCE, "", True],
    [15, 16, 0.1, 4, FISTULOUS_RESISTANCE, "", True],
    [16, "DV2", 0.1, 4, FISTULOUS_RESISTANCE, "", True],
    ["DV1", 11, 0.25, 5, 130.5, ""],
    ["DV2", 11, 0.25, 5, 130.5, ""],
    ["DV3", 11, 0.25, 5, 130.5, ""],
    ["AF3", 17, 0.05, 5, 81600, ""],
    ["AF3", 18, 0.05, 5, 81600, ""],
    ["AF3", 19, 0.05, 5, 81600, ""],
    ["AF3", 20, 0.05, 5, 81600, ""],
    ["AF2", 23, 0.05, 5, 81600, ""],
    ["AF2", 24, 0.05, 5, 81600, ""],
    ["AF1", 27, 0.05, 5, 81600, ""],
    ["AF1", 28, 0.05, 5, 81600, ""],
    ["AF1", 29, 0.05, 5, 81600, ""],
    [17, "AF4", 0.05, 5, 81600, ""],
    [17, 31, 0.05, 5, 81600, ""],
    [18, 31, 0.05, 5, 81600, ""],
    [18, 32, 0.05, 5, 81600, ""],
    [19, 32, 0.05, 5, 81600, ""],
    [19, 33, 0.05, 5, 81600, ""],
    [20, 33, 0.05, 5, 81600, ""],
    [20, 34, 0.05, 5, 81600, ""],
    [21, 34, 0.05, 5, 81600, ""],
    [21, 35, 0.05, 5, 81600, ""],
    [22, 36, 0.05, 5, 81600, ""],
    [22, 37, 0.05, 5, 81600, ""],
    [23, 37, 0.05, 5, 81600, ""],
    [23, 15, 0.05, 5, 81600, ""],
    [14, 38, 0.05, 5, 81600, ""],
    [24, 38, 0.05, 5, 81600, ""],
    [24, 39, 0.05, 5, 81600, ""],
    [25, 39, 0.05, 5, 81600, ""],
    [25, 40, 0.05, 5, 81600, ""],
    [26, 41, 0.05, 5, 81600, ""],
    [26, 42, 0.05, 5, 81600, ""],
    [27, 42, 0.05, 5, 81600, ""],
    [27, 43, 0.05, 5, 81600, ""],
    [28, 43, 0.05, 5, 81600, ""],
    [28, 44, 0.05, 5, 81600, ""],
    [29, 44, 0.05, 5, 81600, ""],
    [29, 45, 0.05, 5, 81600, ""],
    [30, 45, 0.05, 5, 81600, ""],
    [30, 46, 0.05, 5, 81600, ""],
    ["AF4", 47, 0.05, 5, 81600, ""],
    [31, 47, 0.05, 5, 81600, ""],
    [31, 48, 0.05, 5, 81600, ""],
    [32, 48, 0.05, 5, 81600, ""],
    [32, 49, 0.05, 5, 81600, ""],
    [33, 49, 0.05, 5, 81600, ""],
    [33, 50, 0.05, 5, 81600, ""],
    [34, 50, 0.05, 5, 81600, ""],
    [34, 51, 0.05, 5, 81600, ""],
    [35, 51, 0.05, 5, 81600, ""],
    [36, 52, 0.05, 5, 81600, ""],
    [37, 52, 0.05, 5, 81600, ""],
    [37, 16, 0.05, 5, 81600, ""],
    [15, 53, 0.05, 5, 81600, ""],
    [38, 53, 0.05, 5, 81600, ""],
    [38, 54, 0.05, 5, 81600, ""],
    [39, 54, 0.05, 5, 81600, ""],
    [39, 55, 0.05, 5, 81600, ""],
    [40, 55, 0.05, 5, 81600, ""],
    [41, 56, 0.05, 5, 81600, ""],
    [42, 56, 0.05, 5, 81600, ""],
    [42, 57, 0.05, 5, 81600, ""],
    [43, 57, 0.05, 5, 81600, ""],
    [43, 58, 0.05, 5, 81600, ""],
    [44, 58, 0.05, 5, 81600, ""],
    [44, 59, 0.05, 5, 81600, ""],
    [45, 59, 0.05, 5, 81600, ""],
    [45, 60, 0.05, 5, 81600, ""],
    [46, 60, 0.05, 5, 81600, ""],
    [47, 61, 0.05, 5, 81600, ""],
    [48, 61, 0.05, 5, 81600, ""],
    [49, 61, 0.05, 5, 81600, ""],
    [49, "DV3", 0.05, 5, 81600, ""],
    [50, "DV3", 0.05, 5, 81600, ""],
    [51, "DV3", 0.05, 5, 81600, ""],
    [52, "DV2", 0.05, 5, 81600, ""],
    [53, "DV2", 0.05, 5, 81600, ""],
    [53, 62, 0.05, 5, 81600, ""],
    [54, 62, 0.05, 5, 81600, ""],
    [55, 62, 0.05, 5, 81600, ""],
    [56, 63, 0.05, 5, 81600, ""],
    [57, 63, 0.05, 5, 81600, ""],
    [58, 63, 0.05, 5, 81600, ""],
    [58, "DV1", 0.05, 5, 81600, ""],
    [59, "DV1", 0.05, 5, 81600, ""],
    [60, "DV1", 0.05, 5, 81600, ""],
    [17, 37, 0.05, 5, 81600, ""],
    [31, 14, 0.05, 5, 81600, ""],
    [50, "DV2", 0.05, 5, 81600, ""],
    [34, 36, 0.05, 5, 81600, ""],
    ["AF3", 24, 0.05, 5, 81600, ""],
    [21, 24, 0.05, 5, 81600, ""],
    [51, 38, 0.05, 5, 81600, ""],
    [36, 38, 0.05, 5, 81600, ""],
    [15, 28, 0.05, 5, 81600, ""],
    [21, 36, 0.05, 5, 81600, ""],
    [25, 26, 0.05, 5, 81600, ""],
    [55, 56, 0.05, 5, 81600, ""],
    [62, 57, 0.05, 5, 81600, ""],
    [39, 42, 0.05, 5, 81600, ""],
    [16, 44, 0.05, 5, 81600, ""],
]

# Check for duplicate vessels
print("checking vessels...")
any_duplicates = False
for i in range(len(VESSELS) - 1):
    if VESSELS[i][0] == VESSELS[i][1]:
        print(i, VESSELS[i])
        any_duplicates = True
    for j in range(i + 1, len(VESSELS)):
        if (VESSELS[i][0] == VESSELS[j][0] and VESSELS[i][1] == VESSELS[j][1]) or (VESSELS[i][1] == VESSELS[j][0] and VESSELS[i][0] == VESSELS[j][1]):
            print(i, j, VESSELS[i])
            any_duplicates = True
print("DUPLICATE VESSELS" if any_duplicates else "all good")

# PRESSURES is a dictionary of known node : pressure values.
PRESSURES = {
    ("SP", 1): 74 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (3, "AF1"): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (6, "AF2"): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (6, "AF3"): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (9, "AF4"): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    ("DV1", 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    ("DV2", 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    ("DV3", 11): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
    (12, 13): 5 * avm.MMHG_TO_DYN_PER_SQUARE_CM
}

# INTRANIDAL_NODES is a list of nodes in the nidus.
INTRANIDAL_NODES = ["AF1", "AF2", "AF3", "AF4"] + list(range(14, 63 + 1)) + ["DV1", "DV2", "DV3"]


def main():
    network = avm.edges_to_graph(VESSELS)
    flow, pressure, _, graph = avm.simulate(network, [], PRESSURES)
    # print(f"Number of flow values before removing duplicates: {flow.shape}")
    # flow = flow[np.unique(np.round(flow / 0.00000001) * 0.00000001, return_index=True)[1]]
    # print(f"After: {flow.shape}")

    fistulous = [edge[2] for edge in graph.edges(data=True) if edge[2]["resistance"] == FISTULOUS_RESISTANCE]
    fistulous_pressures = [edge["Δpressure"] for edge in fistulous]
    fistulous_flows = [edge["flow"] for edge in fistulous]
    print(f"Number of fistulous vessels: {len(fistulous)}")
    print(f"Fistulous flow range: ({min(fistulous_flows)}, {max(fistulous_flows)}) mL/min")
    print(f"Fistulous flow average: {np.average(fistulous_flows)} mL/min")
    print(f"Fistulous pressure range: ({min(fistulous_pressures)}, {max(fistulous_pressures)}) mmHg")
    print(f"Fistulous pressure average: ({np.average(fistulous_pressures)}) mmHg")

    plexiform = [edge[2] for edge in graph.edges(data=True) if edge[2]["resistance"] == 81600]
    plexiform_pressures = [edge["Δpressure"] for edge in plexiform]
    plexiform_flows = [edge["flow"] for edge in plexiform]
    print(f"Number of plexiform vessels: {len(plexiform)}")
    print(f"Plexiform flow range: ({min(plexiform_flows)}, {max(plexiform_flows)}) mL/min")
    print(f"Plexiform flow average: {np.average(plexiform_flows)} mL/min")
    print(f"Plexiform pressure range: ({min(plexiform_pressures)}, {max(plexiform_pressures)}) mmHg")
    print(f"Plexiform pressure average: ({np.average(plexiform_pressures)}) mmHg")

    # print(f"Flows: {np.round(flow * 60, 3)}")
    print(f"Vessel flow range: ({np.min(np.abs(flow * 60))}, {np.max(np.abs(flow * 60))}) mL/min")
    print(f"Average flow: {np.average(np.abs(flow * 60))} mL/min")
    print(f"Total flow through nidus (out): {graph['DV1'][11]['flow'] + graph['DV2'][11]['flow'] + graph['DV3'][11]['flow']} mL/min")
    print(f"Total flow through nidus (in): {graph[3]['AF1']['flow'] + graph[6]['AF2']['flow'] + graph[6]['AF3']['flow'] + graph[9]['AF4']['flow']} mL/min")
    print(f"Fistulous flow range: ({min(graph['AF2'][14]['flow'], graph[14][15]['flow'], graph[15][16]['flow'], graph[16]['DV2']['flow'])}, {max(graph['AF2'][14]['flow'], graph[14][15]['flow'], graph[15][16]['flow'], graph[16]['DV2']['flow'])}) mL/min")
    print(f"Minimum pressure: {np.min(np.abs(pressure / avm.MMHG_TO_DYN_PER_SQUARE_CM))} mmHg")
    print(f"Maximum pressure: {np.max(np.abs(pressure / avm.MMHG_TO_DYN_PER_SQUARE_CM))} mmHg")
    print(f"Average pressure: {np.average(np.abs(pressure / avm.MMHG_TO_DYN_PER_SQUARE_CM))} mmHg")
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
