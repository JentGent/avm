"""Randomly generates a nidus with nodes arranged in columns."""

import avm
import matplotlib.pyplot as plt
import random
import copy
import math
import pandas as pd
import os
import numpy as np
import generate
import time
from pathlib import Path
from injections import injections

# CALCULATE_ERROR indicates whether or not Kirchoff law pressure error is calculated for each simulation. This slows down the simulation noticeably. Disable when avm.SOLVE_MODE is set to numpy.lstsq because that is pretty much guaranteed to be accurate.
CALCULATE_ERROR = True

# ITERATIONS is the number of unique graphs to generate.
ITERATIONS = 100

# FILE_NAME is the name of the file (including the ".csv" ending) to save data to.
FILE_NAME = "data.csv"

# FIRST_INTRANIDAL_NODE_ID is the ID of the first intranidal node (must be updated with NODE_POS).
FIRST_INTRANIDAL_NODE_ID = max(k for k in avm.NODE_POS_TEMPLATE.keys() if type(k) == int) + 1

def main():

    avm.PREDEFINED_RESISTANCE = False
    start_time = time.time()
    injection_pressures = list(injections.values())

    for i in range(1, 1 + ITERATIONS):

        all_stats = {}
        num_compartments = generate.normint(5, 12)
        num_columns = generate.normint(5, 10)
        num_intercompartmental_vessels = num_compartments * 6
        print(f"{i}: {num_compartments} compartments, {num_columns} columns")

        node_pos = copy.deepcopy(avm.NODE_POS_TEMPLATE)
        network = avm.edges_to_graph(avm.VESSELS_TEMPLATE)
        network, _ = generate.compartments(network, ["AF1", "AF2", "AF3", "AF4"], ["DV1", "DV2", "DV3"], FIRST_INTRANIDAL_NODE_ID, node_pos, num_compartments, num_columns, num_intercompartmental_vessels)

        flows, pressures, all_edges, graphs, *error = avm.simulate_batch(network, [], injection_pressures, CALCULATE_ERROR)
        error = error if error else None

        for j, label in enumerate(injections.keys()):

            flow = flows[:, j]
            pressure = pressures[:, j]
            graph = graphs[j]

            stats = avm.get_stats(graph, abs(injections[label][(12, 13)]), pressure, all_edges, label[1], label[0])
            print(f'{label}: Percent filled using flow formula (%) is {stats["Percent filled using flow formula (%)"]}')

            # avm.display(graph, node_pos, "sdf", color_is_flow = False, cmap_max = 50, fill_by_flow = True)
            # plt.show()

            stats["Blood pressure hypotension"] = label[2]
            stats["CVP pressure"] = label[3]
            stats["Num columns"] = num_columns
            stats["Num compartments"] = num_compartments

            for key, value in stats.items():
                if key not in all_stats: all_stats[key] = []
                all_stats[key].append(value)

            if CALCULATE_ERROR:
                if "Error" not in all_stats: all_stats["Error"] = []
                all_stats["Error"].append(error)

        df = pd.DataFrame(all_stats)
        file_exists = os.path.isfile(FILE_NAME)
        df.to_csv(FILE_NAME, mode="a", index=False, header=not file_exists)

    print(f"{time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
