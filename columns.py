"""Randomly generates a nidus with nodes arranged in columns."""

import avm
import matplotlib.pyplot as plt
import copy
import pandas as pd
import os
import generate
import time
from injections import injections
from itertools import combinations

# CALCULATE_ERROR indicates whether or not Kirchoff law pressure error is calculated for each simulation. This slows down the simulation noticeably. Disable when avm.SOLVE_MODE is set to numpy.lstsq because that is pretty much guaranteed to be accurate.
CALCULATE_ERROR = True

# ITERATIONS is the number of unique graphs to generate.
ITERATIONS = 1

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

        feeders = ['AF1', 'AF2', 'AF3', 'AF4']
        drainers = ['DV1', 'DV2', 'DV3']
    
        injection_pressures = list(injections.values())

        all_stats = {}

        num_compartments = generate.normint(4, 6)
        num_columns = generate.normint(6, 10)
        num_intercompartmental_vessels = generate.normint(90, 110)
        print(f"{i}: {num_compartments} compartments, {num_columns} columns")

        node_pos = copy.deepcopy(avm.NODE_POS_TEMPLATE)
        network = avm.edges_to_graph(avm.VESSELS_TEMPLATE)
        network, _ = generate.compartments(network, feeders, drainers, FIRST_INTRANIDAL_NODE_ID, node_pos, num_compartments, num_columns, num_intercompartmental_vessels, fistula_start = "AF2", fistula_end = "DV2")

        flows, pressures, all_edges, graphs, *error = avm.simulate_batch(network, [], injection_pressures, CALCULATE_ERROR)
        error = error if error else None

        for j, label in enumerate(injections.keys()):

            no_injection_graph = graphs[list(injections.keys()).index((None, 0, label[2], label[3]))] if (None, 0, label[2], label[3]) in injections else None

            flow = flows[:, j]
            pressure = pressures[:, j]
            graph = graphs[j]

            stats = avm.get_stats(graph, no_injection_graph, abs(injections[label][(12, 13)]), label[1], label[0])

            if label[0] == 'DV3' and label[2] == 'profound' and label[3] == 'elevated':
                print(label)
                avm.display(graph, node_pos, color="filling", fill_by_flow=True)
                plt.show()

            stats["Blood pressure hypotension"] = label[2]
            stats["CVP pressure"] = label[3]
            stats["Num columns"] = num_columns
            stats["Num compartments"] = num_compartments
            stats["Num cross vessels"] = num_intercompartmental_vessels
            stats["Fistula compartment number"] = drainers[len(drainers) // 2][2]

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
