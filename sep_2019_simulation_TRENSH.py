"""Recreates the September 2019 network."""

import avm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# FISTULOUS_RESISTANCE is just for testing purposes; the reported value in the paper is 4080.
FISTULOUS_RESISTANCE = 4080

# PLEXIFORM_RESISTANCE is just for testing purposes; the reported value in the paper is 81600.
PLEXIFORM_RESISTANCE = 81600

# SIMULATIONS lists pressures for the 72 different TRENSH injection simulations.
SIMULATIONS = []

# NODE_POS lists the positions of specific nodes in the graph.
NODE_POS = avm.NODE_POS_TEMPLATE | {
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
}

# VESSELS elements are formatted like [first node, second node, radius (cm), length (cm), resistance (dyn s / cm^5), label, fistulous (optional)].
VESSELS = avm.VESSELS_TEMPLATE + [
    ["AF2", 14, 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    [14, 15, 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    [15, 16, 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    [16, "DV2", 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],

    ["AF3", 17, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    ["AF3", 18, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    ["AF3", 19, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    ["AF3", 20, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    ["AF2", 23, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    ["AF2", 24, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    ["AF1", 27, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    ["AF1", 28, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    ["AF1", 29, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [17, "AF4", 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [17, 31, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [18, 31, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [18, 32, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [19, 32, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [19, 33, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [20, 33, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [20, 34, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [21, 34, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [21, 35, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [22, 36, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [22, 37, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [23, 37, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [23, 15, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [14, 38, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [24, 38, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [24, 39, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [25, 39, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [25, 40, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [26, 41, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [26, 42, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [27, 42, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [27, 43, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [28, 43, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [28, 44, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [29, 44, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [29, 45, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [30, 45, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [30, 46, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    ["AF4", 47, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [31, 47, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [31, 48, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [32, 48, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [32, 49, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [33, 49, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [33, 50, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [34, 50, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [34, 51, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [35, 51, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [36, 52, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [37, 52, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [37, 16, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [15, 53, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [38, 53, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [38, 54, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [39, 54, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [39, 55, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [40, 55, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [41, 56, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [42, 56, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [42, 57, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [43, 57, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [43, 58, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [44, 58, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [44, 59, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [45, 59, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [45, 60, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [46, 60, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [47, 61, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [48, 61, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [49, 61, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [49, "DV3", 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [50, "DV3", 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [51, "DV3", 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [52, "DV2", 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [53, "DV2", 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [53, 62, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [54, 62, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [55, 62, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [56, 63, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [57, 63, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [58, 63, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [58, "DV1", 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [59, "DV1", 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [60, "DV1", 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [17, 37, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [31, 14, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [50, "DV2", 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [34, 36, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    ["AF3", 24, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [21, 24, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [51, 38, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [36, 38, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [15, 28, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [21, 36, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [25, 26, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [55, 56, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [62, 57, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [39, 42, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
    [16, 44, 0.05, 5, PLEXIFORM_RESISTANCE, "", avm.vessel.plexiform],
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

injections = {}
for dv in [1, 2, 3]:
    for i in [10, 20, 30]:
        p = 17
        injections[f"DV{dv} {i} mmHg normotension, normal CVP"] = {
            ("SP", 1): 74 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (3, "AF1"): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF2"): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF3"): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (9, "AF4"): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV1", 11): ((p - i) if dv == 1 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV2", 11): ((p - i) if dv == 2 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV3", 11): ((p - i) if dv == 3 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (12, 13): 6 * avm.MMHG_TO_DYN_PER_SQUARE_CM
        }

        p = 15
        injections[f"DV{dv} {i} mmHg minor hypotension, normal CVP"] = {
            ("SP", 1): 70 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (3, "AF1"): 45 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF2"): 45 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF3"): 48 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (9, "AF4"): 48 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV1", 11): ((p - i) if dv == 1 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV2", 11): ((p - i) if dv == 2 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV3", 11): ((p - i) if dv == 3 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (12, 13): 5 * avm.MMHG_TO_DYN_PER_SQUARE_CM
        }

        p = 12
        injections[f"DV{dv} {i} mmHg moderate hypotension, normal CVP"] = {
            ("SP", 1): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (3, "AF1"): 32 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF2"): 32 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF3"): 34 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (9, "AF4"): 34 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV1", 11): ((p - i) if dv == 1 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV2", 11): ((p - i) if dv == 2 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV3", 11): ((p - i) if dv == 3 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (12, 13): 5 * avm.MMHG_TO_DYN_PER_SQUARE_CM
        }

        p = 8
        injections[f"DV{dv} {i} mmHg profound hypotension, normal CVP"] = {
            ("SP", 1): 25 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (3, "AF1"): 15 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF2"): 15 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF3"): 16 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (9, "AF4"): 16 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV1", 11): ((p - i) if dv == 1 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV2", 11): ((p - i) if dv == 2 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV3", 11): ((p - i) if dv == 3 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (12, 13): 4 * avm.MMHG_TO_DYN_PER_SQUARE_CM
        }
        
        p = 22
        injections[f"DV{dv} {i} mmHg normotension, elevated CVP"] = {
            ("SP", 1): 74 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (3, "AF1"): 49 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF2"): 49 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF3"): 52 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (9, "AF4"): 52 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV1", 11): ((p - i) if dv == 1 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV2", 11): ((p - i) if dv == 2 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV3", 11): ((p - i) if dv == 3 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (12, 13): 12 * avm.MMHG_TO_DYN_PER_SQUARE_CM
        }

        p = 19
        injections[f"DV{dv} {i} mmHg minor hypotension, elevated CVP"] = {
            ("SP", 1): 70 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (3, "AF1"): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF2"): 47 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF3"): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (9, "AF4"): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV1", 11): ((p - i) if dv == 1 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV2", 11): ((p - i) if dv == 2 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV3", 11): ((p - i) if dv == 3 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (12, 13): 10 * avm.MMHG_TO_DYN_PER_SQUARE_CM
        }

        p = 14
        injections[f"DV{dv} {i} mmHg moderate hypotension, elevated CVP"] = {
            ("SP", 1): 50 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (3, "AF1"): 33 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF2"): 33 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF3"): 35 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (9, "AF4"): 35 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV1", 11): ((p - i) if dv == 1 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV2", 11): ((p - i) if dv == 2 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV3", 11): ((p - i) if dv == 3 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (12, 13): 8 * avm.MMHG_TO_DYN_PER_SQUARE_CM
        }

        p = 9
        injections[f"DV{dv} {i} mmHg profound hypotension, elevated CVP"] = {
            ("SP", 1): 25 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (3, "AF1"): 16 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF2"): 16 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (6, "AF3"): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (9, "AF4"): 17 * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV1", 11): ((p - i) if dv == 1 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV2", 11): ((p - i) if dv == 2 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            ("DV3", 11): ((p - i) if dv == 3 else p) * avm.MMHG_TO_DYN_PER_SQUARE_CM,
            (12, 13): 6 * avm.MMHG_TO_DYN_PER_SQUARE_CM
        }
injections = list(injections.items())

# INTRANIDAL_NODES is a list of nodes in the nidus.
INTRANIDAL_NODES = ["AF1", "AF2", "AF3", "AF4"] + list(range(14, 63 + 1)) + ["DV1", "DV2", "DV3"]

colorbar = None
def update(frame):
    global colorbar
    network = avm.edges_to_graph(VESSELS)
    label, pressures = injections[frame]
    flow, pressure, _, graph = avm.simulate(network, [], pressures)
    # for key, value in avm.get_stats(graph).items():
    #     print(f"{key}: {value}")
    if colorbar is not None: colorbar.remove()
    plt.cla()
    colorbar = avm.display(graph, INTRANIDAL_NODES, NODE_POS, label, 0, 800)

def main():
    fig, ax = plt.subplots(figsize = (8, 8))
    ax.set_aspect("equal")
    update(0)
    # ani = FuncAnimation(fig, update, frames = len(injections), interval = 100)
    # ani.save("trensh.mp4", writer = "ffmpeg")
    plt.show()



if __name__ == "__main__":
    main()
