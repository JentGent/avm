"""Randomly generates a nidus with nodes arranged in columns."""

import avm
import matplotlib.pyplot as plt
import random
import copy
import math
import pandas as pd
import os
import numpy as np

# SPACING is the size of the space between each compartment, where each unit is the height of one node.
SPACING = 3

# PLEXIFORM_RADIUS is the radius of each plexiform vessel in cm.
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7390970/ says radius is about 50 microns
# https://www.sciencedirect.com/science/article/pii/S1078588417307360 says diameter is 265 microns
# https://sci-hubtw.hkvisa.net/10.1111/j.1440-1827.1981.tb02813.x is the og source that says diameter 265 microns
# https://journals.sagepub.com/doi/epdf/10.1097/00004647-199708000-00009?src=getftr this paper explains that they artificially increased the resistances to account for the resistance lost by approximating the curved vessels as straight
PLEXIFORM_RADIUS = 0.01325
PLEXIFORM_RADIUS_CURR = PLEXIFORM_RADIUS

# PLEXIFORM_LENGTH is the length of each plexiform vessel in cm.
# https://www.sciencedirect.com/science/article/pii/S1350453398000599 says 5 mm as length
PLEXIFORM_LENGTH = 0.5  # No source, 1996 paper had no source either
PLEXIFORM_LENGTH_CURR = PLEXIFORM_LENGTH

# FISTULOUS_RADIUS is the radius of each fistulous vessel in cm.
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9537653/ fistulas with diameter 200 microns
FISTULOUS_RADIUS = 0.05
FISTULOUS_RADIUS_CURR = FISTULOUS_RADIUS

# FISTULOUS_LENGTH is the length of each fistulous vessel in cm.
FISTULOUS_LENGTH = 0.5
FISTULOUS_LENGTH_CURR = FISTULOUS_LENGTH

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
    "SP": [10, -590],
    "AF1": [200, -500],
    "AF2": [200, -370],
    "AF3": [200, -200],
    "AF4": [350, -150],
    "DV1": [500, -543],
    "DV2": [500, -350],
    "DV3": [500, -250],
}

# FIRST_INTRANIDAL_NODE_ID is the ID of the first intranidal node (must be updated with NODE_POS).
FIRST_INTRANIDAL_NODE_ID = 14

# VESSELS elements are formatted like [first node, second node, radius (cm), length (cm), resistance (dyn s / cm^5), label, type (optional)].
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

    [3, "AF1", 0.125, 5.2, 2210, "PCA", avm.vessel.feeder],
    [6, "AF2", 0.15, 3.7, 745.5, "MCA", avm.vessel.feeder],
    [6, "AF3", 0.025, 3.7, 15725000, "ACA", avm.vessel.feeder],
    [9, "AF4", 0.0125, 3, 12750000, "TFA", avm.vessel.feeder],

    ["DV1", 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
    ["DV2", 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
    ["DV3", 11, 0.25, 5, 130.5, "", avm.vessel.drainer],
]

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

PRESSURE_SETS = [
    [74, 47, 47, 50, 50, 17, 17, 17, 6],
    [74, 47, 47, 50, 50, 17, 17, 17, 12],
    [70, 45, 45, 48, 48, 15, 15, 15, 5],
    [70, 45, 45, 48, 48, 15, 15, 15, 10],
    [50, 32, 32, 34, 34, 12, 12, 12, 5],
    [50, 32, 32, 34, 34, 12, 12, 12, 8],
    [25, 15, 15, 16, 16, 8, 8, 8, 4],
    [25, 15, 15, 16, 16, 8, 8, 8, 6],
]


def lerp(x, a, b, c, d):
    """Computes the linear interpolation between c and d, based on the relative position of x between a and b."""
    return c + (d - c) * (x - a) / (b - a)


def normint(a, b, mean, sd):
    """Generates a random integer within the range [a, b], drawn from a normal distribution characterized by a specified mean and standard deviation. The function repeatedly samples from the normal distribution until an integer falling within the specified range is obtained."""
    i = round(random.normalvariate(mean, sd))
    while i < a or i > b:
        i = round(random.normalvariate(mean, sd))
    return i


def choose_norm(list, mean, sd):
    """Selects an element from list using an index generated by normint, which draws from a normal distribution with a specified mean and standard deviation. The index is constrained between 0 and the length of the list minus one, ensuring a valid element from the list is chosen."""
    index = normint(0, len(list) - 1, mean, sd)
    return index, list[index]


def max_int_key(dict):
    """Examines the keys of the dict and returns the max int found."""
    return max(k for k in dict.keys() if type(k) == int)


def add_plexiform(vessels, node1, node2):
    vessels.append([node1, node2, PLEXIFORM_RADIUS_CURR, PLEXIFORM_LENGTH, -1, "", avm.vessel.plexiform])


def add_fistulous(vessels, node1, node2):
    vessels.append([node1, node2, FISTULOUS_RADIUS_CURR, FISTULOUS_LENGTH, -1, "", avm.vessel.fistulous])


def generate_configuration():
    """Specifies the arrangement of the nodes in the nidus.
    The output is a rectangular grid of size m x n.
    m is the number of nidus columns.
    n is the number of compartments in each nidus column.
    Each row represents one nidus column.
    If the first row is [3, 4, 5, 6], this means that the first nidus column has 3 nodes in the first compartment, 4 nodes in the second compartment, etc.
    """
    columns = []
    m = random.randint(5, 10)
    n = random.randint(3, 6)
    columns.append([random.randint(2, 3) for _ in range(n)])
    for i in range(1, m):
        delta = 1
        if i >= math.ceil(m / 2):
            delta = -1
        columns.append([max(columns[-1][i] + random.randint(0, 4) * delta, 2) for i in range(n)])

    global PLEXIFORM_RADIUS_CURR
    global PLEXIFORM_LENGTH_CURR
    global FISTULOUS_RADIUS_CURR
    global FISTULOUS_LENGTH_CURR
    PLEXIFORM_RADIUS_CURR = max(0.0100, random.normalvariate(PLEXIFORM_RADIUS, (PLEXIFORM_RADIUS - 0.0100) / 3))
    PLEXIFORM_LENGTH_CURR = max(0.25, random.normalvariate(PLEXIFORM_LENGTH, (PLEXIFORM_LENGTH - 0.25) / 3))
    FISTULOUS_RADIUS_CURR = max(PLEXIFORM_RADIUS_CURR, random.normalvariate(FISTULOUS_RADIUS, (FISTULOUS_RADIUS - PLEXIFORM_RADIUS_CURR) / 3))
    FISTULOUS_LENGTH_CURR = max(0.25, random.normalvariate(FISTULOUS_LENGTH, (FISTULOUS_LENGTH - 0.25) / 3))

    global PRESSURES
    random_pressure_set = random.choice(PRESSURE_SETS)
    for (key, _), pressure in zip(PRESSURES.items(), random_pressure_set):
        PRESSURES[key] = pressure * avm.MMHG_TO_DYN_PER_SQUARE_CM
    return columns


def generate_nodes():
    """Generates the nodes in the nidus."""
    nodes = generate_configuration()
    node_pos = copy.deepcopy(NODE_POS)
    node_id = max_int_key(node_pos) + 1
    for i, column in enumerate(nodes):
        height = sum(column) + (len(column) - 1) * SPACING  # The height (in number of nodes) of the column
        y = 0
        for j, compartment in enumerate(column):
            compartment_nodes = []
            for _ in range(compartment):
                node_pos[node_id] = [lerp(i, -1, len(nodes), node_pos["AF2"][0], node_pos["DV2"][0]), lerp(y, -SPACING, height, node_pos["AF4"][1], node_pos["DV1"][1])]
                compartment_nodes.append(node_id)
                y += 1
                node_id += 1
            nodes[i][j] = compartment_nodes
            y += SPACING
    return nodes, node_pos


def generate_intranidal_vessels(vessels, nodes):
    """Generates the intranidal vessels. Ensures that each node has at least two edges."""
    for i, column in enumerate(nodes[:-1]):
        for j, _ in enumerate(column):
            small, large = (nodes[i][j].copy(), nodes[i + 1][j].copy()) if len(nodes[i][j]) < len(nodes[i + 1][j]) else (nodes[i + 1][j].copy(), nodes[i][j].copy())
            for node in small:
                partner = random.choice(large)
                large.remove(partner)
                add_plexiform(vessels, node, partner)
            for node in large:
                partner = random.choice(small)
                add_plexiform(vessels, node, partner)
    return vessels


def generate_extranidal_vessels(vessels, nodes):
    """Generates the vessels that connect the intranidal nodes to the feeders and drainers."""
    for i, column in enumerate(nodes):
        for j, compartment in enumerate(column):
            for node_id in compartment:
                if i == 0:
                    if j < (len(column) - 1) * 1/3:
                        add_plexiform(vessels, "AF3", node_id)
                    elif j < (len(column) - 1) * 2/3:
                        add_plexiform(vessels, "AF2", node_id)
                    else:
                        add_plexiform(vessels, "AF1", node_id)
                elif i == len(nodes) - 1:
                    if j < (len(column) - 1) * 1/3:
                        add_plexiform(vessels, "DV3", node_id)
                    elif j < (len(column) - 1) * 2/3:
                        add_plexiform(vessels, "DV2", node_id)
                    else:
                        add_plexiform(vessels, "DV1", node_id)
    add_plexiform(vessels, "AF4", nodes[len(nodes) // 2][0][0])
    if len(nodes) % 2 == 0:
        add_plexiform(vessels, "AF4", nodes[len(nodes) // 2 - 1][0][0])
    return vessels


def generate_cross_compartment_vessels(vessels, nodes, n):
    """Generates n vessels between random nodes of different compartments."""
    if len(nodes[0]) < 2:
        return vessels
    for _ in range(n):
        start_col = random.randint(0, len(nodes) - 2)
        start_compartment = random.randint(0, len(nodes[start_col]) - 1)
        start_node_id = random.choice(nodes[start_col][start_compartment])
        end_compartment = random.choice([i for i in range(len(nodes[start_col + 1])) if i != start_compartment])
        end_node_id = random.choice(nodes[start_col + 1][end_compartment])
        add_plexiform(vessels, start_node_id, end_node_id)
    return vessels


def generate_fistulous_vessels(vessels, nodes):
    """Generates the fistulous vessels as a continuous path through the nidus."""
    compartment = len(nodes[0]) // 2
    start_index = 0
    for i in range(len(nodes)):
        if i == 0:
            start_node_id = "AF2"
            end_index = random.randint(0, len(nodes[i][compartment]) - 1)
            vessels.append(["AF2", nodes[i][compartment][end_index], FISTULOUS_RADIUS, FISTULOUS_LENGTH, -1, "", avm.vessel.fistulous])
            start_index = end_index
        else:
            start_node_id = nodes[i - 1][compartment][start_index]
            end_index = lerp(start_index, 0, len(nodes[i - 1][compartment]) - 1, 0, len(nodes[i][compartment]) - 1)
            end_index, end_node_id = choose_norm(nodes[i][compartment], end_index, 2)
            vessels.append([start_node_id, end_node_id, FISTULOUS_RADIUS, FISTULOUS_LENGTH, -1, "", avm.vessel.fistulous])
            start_index = end_index
    vessels.append([nodes[-1][compartment][start_index], "DV2", FISTULOUS_RADIUS, FISTULOUS_LENGTH, -1, "", avm.vessel.fistulous])
    return vessels


def generate_nidus():
    nodes, node_pos = generate_nodes()
    vessels = copy.deepcopy(VESSELS)
    vessels = generate_intranidal_vessels(vessels, nodes)
    vessels = generate_extranidal_vessels(vessels, nodes)
    vessels = generate_cross_compartment_vessels(vessels, nodes, 20)
    vessels = generate_fistulous_vessels(vessels, nodes)  # Must go last to overwrite previous edges
    return nodes, node_pos, vessels


def print_stats(graph):
    print('Stats are shown as (min, mean, max)\n')
    for key, value in avm.get_stats(graph).items():
        if value:
            print(f"{key}: {value}")
        else:
            print('')
    print(f'Plexiform resistance (dyn * s / cm^5): {avm.calc_resistance(PLEXIFORM_RADIUS, PLEXIFORM_LENGTH)}')
    print(f'Fistulous resistance (dyn * s / cm^5): {avm.calc_resistance(FISTULOUS_RADIUS, FISTULOUS_LENGTH)}')
    mean_risk, max_risk = compute_rupture_risk(graph)
    print(f'Mean rupture risk: {mean_risk}%')
    print(f'Max rupture risk: {max_risk}%')


def compute_rupture_risk(graph):
    """Computes and prints the rupture risk for each vessel."""
    pressures = []
    for _, _, attr in graph.edges(data=True):
        if attr["type"] == avm.vessel.fistulous or attr["type"] == avm.vessel.plexiform:
            pressures.append(attr["pressure"])
    p_max = 74  # mmHg
    p_min = PRESSURES[(12, 13)] / avm.MMHG_TO_DYN_PER_SQUARE_CM
    risks = []
    for pressure in pressures:
        risk = math.log(abs(pressure) / p_min) / math.log(p_max / p_min) * 100
        risk = max(0, min(risk, 100))
        risks.append(risk)
    return np.mean(risks), max(risks)


def extract_stats(graph, nodes, node_pos):
    stats = avm.get_stats(graph)
    stats['Num columns'] = len(nodes)
    stats['Num compartments'] = len(nodes[0])
    stats['Num intranidal nodes'] = max_int_key(node_pos) - FIRST_INTRANIDAL_NODE_ID + 1
    stats['Num intranidal vessels'] = stats['Number of plexiform vessels'] + stats['Number of fistulous vessels']
    stats['Num fistulas'] = stats['Number of fistulous vessels'] / (len(nodes) + 1)
    stats['Fistulous vessel radius (cm)'] = FISTULOUS_RADIUS_CURR
    stats['Fistulous vessel length (cm)'] = FISTULOUS_LENGTH_CURR
    stats['Plexiform vessel radius (cm)'] = PLEXIFORM_RADIUS_CURR
    stats['Plexiform vessel length (cm)'] = PLEXIFORM_LENGTH_CURR

    stats['Systemic pressure (mmHg)'] = PRESSURES[("SP", 1)] / avm.MMHG_TO_DYN_PER_SQUARE_CM
    stats['AF1 (mmHg)'] = PRESSURES[(3, "AF1")] / avm.MMHG_TO_DYN_PER_SQUARE_CM
    stats['AF2 (mmHg)'] = PRESSURES[(6, "AF2")] / avm.MMHG_TO_DYN_PER_SQUARE_CM
    stats['AF3 (mmHg)'] = PRESSURES[(6, "AF3")] / avm.MMHG_TO_DYN_PER_SQUARE_CM
    stats['AF4 (mmHg)'] = PRESSURES[(9, "AF4")] / avm.MMHG_TO_DYN_PER_SQUARE_CM
    stats['DV (mmHg)'] = PRESSURES[("DV1", 11)] / avm.MMHG_TO_DYN_PER_SQUARE_CM
    stats['Central venous pressure (mmHg)'] = PRESSURES[(12, 13)] / avm.MMHG_TO_DYN_PER_SQUARE_CM

    # Outputs
    mean_risk, max_risk = compute_rupture_risk(graph)
    stats['Mean rupture risk (%)'] = mean_risk
    stats['Max rupture risk (%)'] = max_risk
    stats['Total nidal flow (mL/min)'] = stats['Feeder total flow (mL/min)']

    return stats


def main():

    all_stats = []

    for i in range(1):

        avm.PREDEFINED_RESISTANCE = False

        nodes, node_pos, vessels = generate_nidus()
        network = avm.edges_to_graph(vessels)

        _, _, _, graph = avm.simulate(network, [], PRESSURES)
        print_stats(graph)
        print(f'Iteration {i}')

        stats = extract_stats(graph, nodes, node_pos)
        all_stats.append(stats)

        df = pd.DataFrame([stats])

        file_exists = os.path.isfile('graph_stats.csv')
        df.to_csv('graph_stats.csv', mode='a', index=False, header=not file_exists)

        # intranidal_nodes = list(range(FIRST_INTRANIDAL_NODE_ID, max_int_key(node_pos) + 1))
        # avm.display(graph, intranidal_nodes, node_pos, cmap_max=74, color_is_flow=False)
        # plt.show()


if __name__ == "__main__":
    main()
