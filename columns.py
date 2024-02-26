"""Randomly generates a nidus with nodes along columns for more organized visualization."""

import avm
import numpy as np
import matplotlib.pyplot as plt
import random

avm.PREDEFINED_RESISTANCE = False

# FISTULOUS_RESISTANCE is just for testing purposes; the reported value in the paper is 4080.
FISTULOUS_RESISTANCE = 0

# PLEXIFORM_RESISTANCE is just for testing purposes; the reported value in the paper is 81600.
PLEXIFORM_RESISTANCE = 0

# SIMULATIONS lists pressures for the 72 different TRENSH injection simulations.
SIMULATIONS = []

def lerp(x, a, b, c, d):
    return c + (d - c) * (x - a) / (b - a)
def normint(a, b, mean, sd):
    i = round(random.normalvariate(mean, sd))
    while i < a or i > b:
        i = round(random.normalvariate(mean, sd))
    return i
        
def choose_norm(list, mean, sd):
    return list[normint(0, len(list) - 1, mean, sd)]

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
    # 14: [275, -370],
    # 15: [339, -359],
    # 16: [425, -350],
    "SP": [10, -590],
    "AF1": [200, -500],
    "AF2": [200, -370],
    "AF3": [200, -200],
    "AF4": [350, -150],
    "DV1": [500, -543],
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
    # [1, 4, 0.35, 10, 637.5, "CCA"],
    # [4, 5, 0.2, 10, 1000000, "ECA"],
    # [5, 9, 0.01, 0.1, 1000000, ""],
    # [9, 10, 0.01, 0.1, 4177.9, ""],
    # [12, 13, 0.4, 20, 79.7, "jugular veins"],

    [4, 6, 0.25, 20, 522, "ICA"],
    [2, 3, 0.15, 25, 5037, "VA"],
    [6, 7, 0.1, 10, 10200, ""],
    [7, 8, 0.01, 0.1, 1000000, ""],
    [8, 11, 0.125, 10, 4177.9, ""],
    [11, 12, 0.25, 10, 261, "dural venous sinuses"],

    [3, "AF1", 0.125, 5.2, 2210, "PCA", avm.vessel.feeder],
    [6, "AF2", 0.15, 3.7, 745.5, "MCA", avm.vessel.feeder],
    [6, "AF3", 0.025, 3.7, 15725000, "ACA", avm.vessel.feeder],
    [9, "AF4", 0.0125, 3, 12750000, "TFA", avm.vessel.feeder],

    # ["AF2", 14, 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    # [14, 15, 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    # [15, 16, 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    # [16, "DV2", 0.1, 4, FISTULOUS_RESISTANCE, "", avm.vessel.fistulous],
    ["DV1", 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
    ["DV2", 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
    ["DV3", 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
]

COLUMNS = [
    [2, 3, 2],
    [5, 10, 5],
    [10, 15, 8],
    [15, 20, 10],
    [10, 15, 8],
    [5, 10, 5],
    [2, 3, 2]
]
EDGES = [
    [5, 10, 5],
    [20, 30, 20],
    [30, 40, 20],
    [30, 40, 20],
    [20, 30, 20],
    [5, 10, 5]
]

node_id = 1
while node_id in NODE_POS: node_id += 1
SPACING = 3
for i, groups in enumerate(COLUMNS):
    total = sum(groups) + (len(groups) - 1) * SPACING
    y = 0
    for j, group_size in enumerate(groups):
        nodes = []
        for k in range(group_size):
            NODE_POS[node_id] = [lerp(i, -1, len(COLUMNS), NODE_POS["AF2"][0], NODE_POS["DV2"][0]), lerp(y, -SPACING, total, NODE_POS["AF4"][1], NODE_POS["DV1"][1])]
            nodes.append(node_id)
            y += 1
            node_id += 1
        groups[j] = nodes
        y += SPACING
for i, groups in enumerate(COLUMNS[:-1]):
    for j, group in enumerate(groups):
        for k in range(EDGES[i][j]):
            first = random.randint(0, len(COLUMNS[i][j]) - 1)
            VESSELS.append([COLUMNS[i][j][first], choose_norm(COLUMNS[i + 1][j], first, 2), 0.047, 3, -1, "", avm.vessel.plexiform])
            # VESSELS.append([random.choice(COLUMNS[i][j]), random.choice(COLUMNS[i + 1][j]), 0.43, 2, -1, "", avm.vessel.plexiform])
VESSELS.append(["AF1", 19, 0.047, 3, -1, "", avm.vessel.plexiform])
VESSELS.append(["AF1", 20, 0.047, 3, -1, "", avm.vessel.plexiform])

VESSELS.append(["AF2", 16, 0.047, 3, -1, "", avm.vessel.plexiform])
VESSELS.append(["AF2", 17, 0.047, 3, -1, "", avm.vessel.plexiform])
VESSELS.append(["AF2", 18, 0.047, 3, -1, "", avm.vessel.plexiform])

VESSELS.append(["AF3", 14, 0.047, 3, -1, "", avm.vessel.plexiform])
VESSELS.append(["AF3", 15, 0.047, 3, -1, "", avm.vessel.plexiform])

VESSELS.append(["AF4", 74, 0.047, 3, -1, "", avm.vessel.plexiform])

VESSELS.append(["DV1", 177, 0.047, 3, -1, "", avm.vessel.plexiform])
VESSELS.append(["DV1", 178, 0.047, 3, -1, "", avm.vessel.plexiform])

VESSELS.append(["DV2", 174, 0.047, 3, -1, "", avm.vessel.plexiform])
VESSELS.append(["DV2", 175, 0.047, 3, -1, "", avm.vessel.plexiform])
VESSELS.append(["DV2", 176, 0.047, 3, -1, "", avm.vessel.plexiform])

VESSELS.append(["DV3", 172, 0.047, 3, -1, "", avm.vessel.plexiform])
VESSELS.append(["DV3", 173, 0.047, 3, -1, "", avm.vessel.plexiform])

for i in range(20):
    start = random.choice(["AF1", "AF2", "AF3", "AF4", "DV1", "DV2", "DV3"] + list(range(14, 178 + 1)))
    end = random.choice(["AF1", "AF2", "AF3", "AF4", "DV1", "DV2", "DV3"] + list(range(14, 178 + 1)))
    VESSELS.append([start, end, 0.047, 3, -1, "", avm.vessel.plexiform])

start = "AF2"
for i, column in enumerate(COLUMNS):
    end = normint(0, len(COLUMNS[i][1]) - 1, 1 if start == "AF2" else start, 2)
    VESSELS.append([start if start == "AF2" else COLUMNS[i - 1][1][start], COLUMNS[i][1][end], 0.044, 1, -1, "", avm.vessel.fistulous])
    start = end
VESSELS.append([COLUMNS[-1][1][start], "DV2", 0.5, 1, -1, "", avm.vessel.fistulous])

print("Plexiform resistance: " + str(avm.calc_resistance(0.047, 3)))
print("Fistulous resistance: " + str(avm.calc_resistance(0.044, 1)))

# Check for duplicate vessels
print("checking vessels...")
any_duplicates = False
for i in range(len(VESSELS) - 1):
    if VESSELS[i][0] == VESSELS[i][1]:
        # print(i, VESSELS[i])
        any_duplicates = True
    for j in range(i + 1, len(VESSELS)):
        if (VESSELS[i][0] == VESSELS[j][0] and VESSELS[i][1] == VESSELS[j][1]) or (VESSELS[i][1] == VESSELS[j][0] and VESSELS[i][0] == VESSELS[j][1]):
            # print(i, j, VESSELS[i])
            any_duplicates = True
print("DUPLICATE VESSELS" if any_duplicates else "all good")
print("Number of vessels before removing duplicates: " + str(len(VESSELS)))

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
INTRANIDAL_NODES = ["AF1", "AF2", "AF3", "AF4"] + list(range(14, 178 + 1)) + ["DV1", "DV2", "DV3"]


def main():
    network = avm.edges_to_graph(VESSELS)
    flow, pressure, _, graph = avm.simulate(network, [], PRESSURES)
    for key, value in avm.get_stats(graph).items():
        print(f"{key}: {value}")
    nodes_to_remove = []
    for node in graph.nodes:
        if node not in INTRANIDAL_NODES:
            nodes_to_remove.append(node)
    graph.remove_nodes_from(nodes_to_remove)
    avm.display(graph, INTRANIDAL_NODES, NODE_POS)
    plt.show()


if __name__ == "__main__":
    main()
