"""Randomly generates a nidus with nodes arranged in columns."""

from skimage.filters import threshold_otsu
import avm
import matplotlib.pyplot as plt
import random
import copy
import math
import networkx as nx
import pandas as pd
import os
import numpy as np
import generate
import time
from pathlib import Path
from injections import injections
from itertools import combinations
feeder_sets = []
drainer_sets = []
for i in range(1, 5):
    for combination in combinations(["AF1", "AF2", "AF3", "AF4"], i):
        feeder_sets.append(list(combination))
for i in range(1, 4):
    for combination in combinations(["DV1", "DV2", "DV3"], i):
        drainer_sets.append(list(combination))
feeder_sets = [["AF1", "AF2", "AF3", "AF4"]]
drainer_sets = [["DV1", "DV2", "DV3"]]
print(len(feeder_sets))

# CALCULATE_ERROR indicates whether or not Kirchoff law pressure error is calculated for each simulation. This slows down the simulation noticeably. Disable when avm.SOLVE_MODE is set to numpy.lstsq because that is pretty much guaranteed to be accurate.
CALCULATE_ERROR = True

# ITERATIONS is the number of unique graphs to generate.
ITERATIONS = 50

# FILE_NAME is the name of the file (including the ".csv" ending) to save data to.
FILE_NAME = "data.csv"

# FIRST_INTRANIDAL_NODE_ID is the ID of the first intranidal node (must be updated with NODE_POS).
FIRST_INTRANIDAL_NODE_ID = max(k for k in avm.NODE_POS_TEMPLATE.keys() if type(k) == int) + 1

def main():

    avm.PREDEFINED_RESISTANCE = False
    start_time = time.time()

    if os.path.exists(FILE_NAME):
        os.remove(FILE_NAME)

    for i in range(1, 1 + ITERATIONS):
        # injections_without_missing_nodes = injections
        # injection_pressures = list(injections_without_missing_nodes.values())
        # feeders = feeder_sets[0]
        # drainers = drainer_sets[0]
        # best_error = float('inf')
        # best_num_compartments = None
        # best_num_columns = None
        # best_min_compartment_height = None
        # best_max_compartment_height = None
        # best_num_intercompartmental_vessels = None
        # for num_compartments in [3, 4, 5]:
        #     for num_columns in [7, 8, 9]:
        #         for min_compartment_height in [5, 10, 15]:
        #             max_compartment_height = min_compartment_height * 2
        #             for num_intercompartmental_vessels in [65, 70, 75]:

        #                 node_pos = copy.deepcopy(avm.NODE_POS_TEMPLATE)
        #                 network = avm.edges_to_graph(avm.VESSELS_TEMPLATE)
        #                 network, _ = generate.compartments(network, feeders, drainers, FIRST_INTRANIDAL_NODE_ID, node_pos, num_compartments, num_columns, num_intercompartmental_vessels, min_compartment_height, max_compartment_height, fistula_start = "AF2", fistula_end = "DV2")

        #                 flows, pressures, all_edges, graphs, *error = avm.simulate_batch(network, [], injection_pressures, CALCULATE_ERROR)
                        
        #                 net_flow = avm.get_stats(graphs[0], None, 6, 0, None)["Drainer total flow (mL/min)"]
        #                 error = (net_flow - min(450, max(300, net_flow))) ** 2 - avm.get_stats(graphs[1], None, 6, 30, "DV1")["Percent filled (%)"]
                        
        #                 print(f'{error} {best_error}{" best!" if error < best_error else ""}')
        #                 if error < best_error:
        #                     print(f"= {i}, {num_compartments} {num_columns} {min_compartment_height} {num_intercompartmental_vessels}")
        #                     best_error = error
        #                     best_num_compartments = num_compartments
        #                     best_num_columns = num_columns
        #                     best_min_compartment_height = min_compartment_height
        #                     best_max_compartment_height = max_compartment_height
        #                     best_num_intercompartmental_vessels = num_intercompartmental_vessels


        # continue

        feeders = feeder_sets[0]
        drainers = drainer_sets[0]
        
        # injections_without_missing_nodes = { key: value for key, value in injections.items() if key[0] is None or key[0] in feeders + drainers }
        injections_without_missing_nodes = injections
        injection_pressures = list(injections_without_missing_nodes.values())
        all_stats = {}
        # num_compartments = 5
        num_compartments = generate.normint(4, 6)
        # num_compartments = random.randint(4, 6)
        # num_compartments = 5
        num_columns = generate.normint(6, 10)
        # num_columns = random.randint(6, 10)
        # num_columns = 10
        num_intercompartmental_vessels = generate.normint(90, 110)
        # num_intercompartmental_vessels = 90
        print(f"{i}: {num_compartments} compartments, {num_columns} columns")
        # print(f"{i}: {feeders} feeders and {drainers} drainers")

        node_pos = copy.deepcopy(avm.NODE_POS_TEMPLATE)
        network = avm.edges_to_graph(avm.VESSELS_TEMPLATE)
        network, _ = generate.compartments(network, feeders, drainers, FIRST_INTRANIDAL_NODE_ID, node_pos, num_compartments, num_columns, num_intercompartmental_vessels, fistula_start = "AF2", fistula_end = "DV2")

        flows, pressures, all_edges, graphs, *error = avm.simulate_batch(network, [], injection_pressures, CALCULATE_ERROR)
        error = error if error else None

        for j, label in enumerate(injections_without_missing_nodes.keys()):
            no_injection_graph = graphs[list(injections_without_missing_nodes.keys()).index((None, 0, label[2], label[3]))] if (None, 0, label[2], label[3]) in injections_without_missing_nodes else None

            flow = flows[:, j]
            pressure = pressures[:, j]
            graph = graphs[j]

            stats = avm.get_stats(graph, no_injection_graph, abs(injections_without_missing_nodes[label][(12, 13)]), label[1], label[0])
            # print(f'{label}: Filling is {stats["Percent filled (%)"]} %')

            # if label[0] and stats["Percent filled (%)"] > 80:
            #     no_injection_nidus = avm.get_nidus(no_injection_graph)
            #     nidus = avm.get_nidus(graph)
            #     all_changes = avm.pressure_change(no_injection_nidus, nidus)
            #     changes = [change for change in all_changes if change < 0 and change > -100]
            #     plt.figure(figsize=(10, 6))
            #     plt.hist(changes, bins=np.arange(min(changes), max(changes), 5), color="skyblue", edgecolor="black")
            #     plt.axvline(threshold_otsu(np.array(changes)), color="red", linestyle="dashed", linewidth=2)
            #     plt.grid(True)
            #     plt.show()
            #     avm.display(nidus, node_pos, color = "filling", fill_by_flow = True, cmap_min=0, cmap_max=30)
            #     plt.show()

            # avm.display(avm.get_nidus(graph), node_pos, color = "filling", fill_by_flow = True, cmap_min=0, cmap_max=30)
            # avm.display(graph, node_pos, color = "filling", fill_by_flow = True, cmap_min=0, cmap_max=30)
            # plt.show()


            stats["Blood pressure hypotension"] = label[2]
            stats["CVP pressure"] = label[3]
            stats["Num columns"] = num_columns
            stats["Num compartments"] = num_compartments
            stats["Num cross vessels"] = num_intercompartmental_vessels
            stats["AF1"] = 1 if "AF1" in feeders else 0
            stats["AF2"] = 1 if "AF2" in feeders else 0
            stats["AF3"] = 1 if "AF3" in feeders else 0
            stats["AF4"] = 1 if "AF4" in feeders else 0
            # stats["DV1"] = 1 if "DV1" in drainers else 0
            # stats["DV2"] = 1 if "DV2" in drainers else 0
            # stats["DV3"] = 1 if "DV3" in drainers else 0
            stats["Fistula"] = drainers[len(drainers) // 2][2]

            for key, value in stats.items():

                if key not in all_stats:
                    all_stats[key] = []

                all_stats[key].append(value)

            if CALCULATE_ERROR:

                if "Error" not in all_stats:
                    all_stats["Error"] = []

                all_stats["Error"].append(error)

        df = pd.DataFrame(all_stats)
        file_exists = os.path.isfile(FILE_NAME)
        df.to_csv(FILE_NAME, mode="a", index=False, header=not file_exists)

    print(f"{time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
