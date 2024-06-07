"""Simulates a nidus with nodes arranged in columns."""

from injections import injections
from itertools import combinations
import avm
import copy
import figures
import generate
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

# CALCULATE_ERROR indicates whether or not Kirchoff law pressure error is calculated for each simulation. This slows down the simulation noticeably. Disable when avm.SOLVE_MODE is set to numpy.lstsq because that is pretty much guaranteed to be accurate.
CALCULATE_ERROR = True

# ITERATIONS is the number of unique graphs to generate.
ITERATIONS = 10

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

        num_compartments = generate.normint(3, 6, sd=1)
        num_columns = generate.normint(3, 7, sd=1)
        num_intercompartmental_vessels = num_compartments * num_columns * 2
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

            stats["Blood pressure hypotension"] = label[2]
            stats["CVP pressure"] = label[3]
            stats["Num columns"] = num_columns
            stats["Num compartments"] = num_compartments
            stats["Num cross vessels"] = num_intercompartmental_vessels
            stats["Fistula compartment index"] = drainers[len(drainers) // 2][2]

            for key, value in stats.items():

                if key not in all_stats:
                    all_stats[key] = []

                all_stats[key].append(value)

            if CALCULATE_ERROR:

                if "Error" not in all_stats:
                    all_stats["Error"] = []

                all_stats["Error"].append(error)

            # print(label)

            # # Flow
            # plt.figure(figsize=(1920/100, 1080/100))
            # figures.display_flow(graph, node_pos)
            # plt.show()

            # plt.text(0.01, 0.99, f"{label[2]}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            # plt.text(0.01, 0.96, f"Total Nidal Flow: {int(stats['Feeder total flow (mL/min)'])} mL/min", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            # filename = f"temp/{j:02d}_{label[2]}_{label[0]}_flow.png"
            # plt.savefig(filename)
            # # plt.show()
            # plt.close()

            # # Pressure
            # plt.figure(figsize=(1920/100, 1080/100))
            # figures.display_pressure(graph, node_pos)

            # plt.text(0.01, 0.99, f"{label[2]}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            # plt.text(0.01, 0.96, f"Mean Vessel Rupture Risk: {int(stats['Mean rupture risk (%)'])}%", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            # filename = f"temp/{j:02d}_{label[2]}_{label[0]}_pressure.png"
            # plt.savefig(filename)
            # # plt.show()
            # plt.close()

        df = pd.DataFrame(all_stats)
        file_exists = os.path.isfile(FILE_NAME)
        df.to_csv(FILE_NAME, mode="a", index=False, header=not file_exists)

    print(f"{time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
