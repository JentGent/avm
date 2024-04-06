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

def print_stats(graph, p_min):
    print("Stats are shown as (min, mean, max)\n")
    for key, value in avm.get_stats(graph).items():
        if value:
            print(f"{key}: {value}")
        else:
            print("")
    mean_risk, max_risk = compute_rupture_risk(graph, p_min)
    print(f"Mean rupture risk: {mean_risk}%")
    print(f"Max rupture risk: {max_risk}%")


def compute_rupture_risk(graph, p_min):
    """Computes and prints the rupture risk for each vessel."""
    pressures = []
    for _, _, attr in graph.edges(data=True):
        if attr["type"] == avm.vessel.fistulous or attr["type"] == avm.vessel.plexiform:
            pressures.append(attr["pressure"])
    p_max = 74  # mmHg
    p_min /= avm.MMHG_TO_DYN_PER_SQUARE_CM
    risks = []
    for pressure in pressures:
        risk = math.log(abs(pressure) / p_min) / math.log(p_max / p_min) * 100
        risk = max(0, min(risk, 100))
        risks.append(risk)
    return np.mean(risks), max(risks)


def extract_stats(graph, num_columns, node_pos, p_min):
    stats = avm.get_stats(graph)
    stats["Num intranidal nodes"] = max_int_key(node_pos) - FIRST_INTRANIDAL_NODE_ID + 1
    stats["Num intranidal vessels"] = stats["Num plexiform"] + stats["Num fistulous"]
    stats["Num fistulas"] = stats["Num fistulous"] / (num_columns + 1)

    # Outputs
    mean_risk, max_risk = compute_rupture_risk(graph, p_min)
    stats["Mean rupture risk (%)"] = mean_risk
    stats["Max rupture risk (%)"] = max_risk
    stats["Total nidal flow (mL/min)"] = stats["Feeder total flow (mL/min)"]

    return stats

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
                stats = extract_stats(graphs[j], num_columns, node_pos, injections[label][(12, 13)])
            else:
                _, _, _, graph, *error = avm.simulate(network, [], injections[label], CALCULATE_ERROR)
                error = error if error else None
                stats = extract_stats(graph, num_columns, node_pos, injections[label][(12, 13)])
            stats["Label"] = label
            stats["Injection pressure (mmHg)"] = float(label[4:6]) if label[0:2] == "DV" else 0
            stats["Blood pressure"] = label[12:] if label[0:2] == "DV" else label
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
    print(data["Total nidal flow (mL/min)"].mean())

    print(f"{time.time() - start_time} seconds")

    input("Press return to exit")


if __name__ == "__main__":
    main()
