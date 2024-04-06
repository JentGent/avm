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

# SIMULATE_IN_BATCHES indicates whether to use `simulate_batch` or `simulate`
SIMULATE_IN_BATCHES = True

# ITERATIONS is the number of unique graphs to generate.
ITERATIONS = 10

# FILE_NAME is the name of the file (minus the ".csv" ending) to save data to.
FILE_NAME = "test"

def max_int_key(dict: dict) -> int:
    """Examines the keys of the dict and returns the max int found."""
    return max(k for k in dict.keys() if type(k) == int)

# FIRST_INTRANIDAL_NODE_ID is the ID of the first intranidal node (must be updated with NODE_POS).
FIRST_INTRANIDAL_NODE_ID = max_int_key(avm.NODE_POS_TEMPLATE) + 1

def main():
    avm.PREDEFINED_RESISTANCE = False
    
    start_time = time.time()
    injection_pressures = list(injections.values())
    for i in range(1, 1 + ITERATIONS):
        all_stats = {}
        num_compartments = random.randint(3, 9)
        num_columns = random.randint(5, 10)
        print(f"{i}: {num_compartments} compartments, {num_columns} columns")
        node_pos = copy.deepcopy(avm.NODE_POS_TEMPLATE)
        network = avm.edges_to_graph(avm.VESSELS_TEMPLATE)
        network, _ = generate.compartments(network, ["AF1", "AF2", "AF3", "AF4"], ["DV1", "DV2", "DV3"], FIRST_INTRANIDAL_NODE_ID, node_pos, num_compartments, num_columns, num_compartments * 6)
        if SIMULATE_IN_BATCHES:
            _, _, _, graphs, *error = avm.simulate_batch(network, [], injection_pressures, CALCULATE_ERROR)
            error = error if error else None
        for j, label in enumerate(injections.keys()):
            if SIMULATE_IN_BATCHES:
                stats = avm.get_stats(graphs[j], injections[label][(12, 13)])
            else:
                _, _, _, graph, *error = avm.simulate(network, [], injections[label], CALCULATE_ERROR)
                error = error if error else None
                stats = avm.get_stats(graph, injections[label][(12, 13)])
            stats["Injection location"] = label[0]
            stats["Injection pressure (mmHg)"] = label[1]
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
        file_exists = os.path.isfile(f"{FILE_NAME}.csv")
        df.to_csv(f"{FILE_NAME}.csv", mode="a", index=False, header=not file_exists)
    
    file_path = Path(__file__).parent / f"{FILE_NAME}.csv"
    data = pd.read_csv(file_path)
    print(data["Drainer total flow (mL/min)"].mean())

    print(f"{time.time() - start_time} seconds")

    input("Press return to exit")


if __name__ == "__main__":
    main()
