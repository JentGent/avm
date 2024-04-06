"""Makes sure simulate_batch() and simulate() are consistent with each other."""

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
from injections import injections

# CALCULATE_ERROR indicates whether or not Kirchoff law pressure error is calculated for each simulation. This slows down the simulation noticeably. Disable when avm.SOLVE_MODE is set to numpy.lstsq because that is pretty much guaranteed to be accurate.
CALCULATE_ERROR = True

def max_int_key(dict: dict) -> int:
    """Examines the keys of the dict and returns the max int found."""
    return max(k for k in dict.keys() if type(k) == int)

# FIRST_INTRANIDAL_NODE_ID is the ID of the first intranidal node (must be updated with NODE_POS).
FIRST_INTRANIDAL_NODE_ID = max_int_key(avm.NODE_POS_TEMPLATE) + 1


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
    for i in range(1, 1 + 2):
        print(f"Iteration {i}")
        for num_compartments in [3, 6]:
            for num_columns in [5, 10]:
                print(f"{num_compartments} compartments, {num_columns} columns")
                node_pos = copy.deepcopy(avm.NODE_POS_TEMPLATE)
                network = avm.edges_to_graph(avm.VESSELS_TEMPLATE)
                network, _ = generate.compartments(network, ["AF1", "AF2", "AF3", "AF4"], ["DV1", "DV2", "DV3"], FIRST_INTRANIDAL_NODE_ID, node_pos, num_compartments, num_columns, num_compartments * 6)
                _, _, _, graphs, *error = avm.simulate_batch(network, [], injection_pressures, CALCULATE_ERROR)
                error = error if error else None
                for j, label in enumerate(injections.keys()):
                    stats_batch = extract_stats(graphs[j], num_columns, node_pos, injection_pressures[j][(12, 13)])
                    _, _, _, graph, *error = avm.simulate(network, [], injections[label], CALCULATE_ERROR)
                    stats = extract_stats(graph, num_columns, node_pos, injection_pressures[j][(12, 13)])
                    for stat in stats:
                        var1 = stats[stat]
                        var2 = stats_batch[stat]
                        if isinstance(var1, (int, float)) and isinstance(var2, (int, float)):
                            if abs(var1 - var2) > 0.001: print("bad!")
                        else:
                            if not all(abs(a - b) < 0.001 for a, b in zip(var1, var2)): print("wrong!")

    print(f"{time.time() - start_time} seconds")
    print("if there were no 'wrong!' or 'bad!' messages, batch simulation is working")

    input("Press return to exit")


if __name__ == "__main__":
    main()
