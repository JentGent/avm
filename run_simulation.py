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
import logging

# CALCULATE_ERROR indicates whether or not Kirchoff law pressure error is calculated for each simulation. This slows down the simulation noticeably. Disable when avm.SOLVE_MODE is set to numpy.lstsq because that is pretty much guaranteed to be accurate.
CALCULATE_ERROR = False

# ITERATIONS is the number of unique graphs to generate.
ITERATIONS = float('inf')

# FILE_NAME is the name of the file (including the ".csv" ending) to save data to.
FILE_NAME = "data-2024-08-22-mac.csv"

# FIRST_INTRANIDAL_NODE_ID is the ID of the first intranidal node (must be updated with NODE_POS).
FIRST_INTRANIDAL_NODE_ID = max(k for k in avm.NODE_POS_TEMPLATE.keys() if type(k) == int) + 1


def main():

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    avm.PREDEFINED_RESISTANCE = False
    start_time = time.time()

    # if os.path.exists(FILE_NAME):
    #     os.remove(FILE_NAME)

    i = 0
    while i < ITERATIONS:

        i += 1

        feeders = ['AF1', 'AF2', 'AF3', 'AF4']
        drainers = ['DV1', 'DV2', 'DV3']
    
        injection_pressures = list(injections.values())

        all_stats = {}

        num_compartments = generate.normint(3, 6, sd=1)
        num_columns = generate.normint(3, 7, sd=1)
        num_intercompartmental_vessels = num_compartments * num_columns * 2
        logger.info(f"{i}: {num_compartments} compartments, {num_columns} columns")

        node_pos = copy.deepcopy(avm.NODE_POS_TEMPLATE)
        full_network = avm.edges_to_graph(avm.VESSELS_TEMPLATE)
        full_network, _ = generate.compartments(full_network, feeders, drainers, FIRST_INTRANIDAL_NODE_ID, node_pos, num_compartments, num_columns, num_intercompartmental_vessels, fistula_start = "AF2", fistula_end = "DV2")
        feeders = [edge for edge in full_network.edges(data=True) if edge[2]["type"] == avm.vessel.feeder and edge[1] != "AF4"]

        for occluded in [None] + feeders:
            
            if occluded is not None:
                network = full_network.copy()
                network.remove_edge(occluded[0], occluded[1])
            else: network = full_network

            flows, pressures, all_edges, graphs, *error = avm.simulate_batch(network, "SP", 0, injection_pressures, CALCULATE_ERROR)
            error = error if error else None

            for j, label in enumerate(injections.keys()):

                no_injection_graph = graphs[list(injections.keys()).index((None, 0, label[2], label[3], label[4]))]

                flow = flows[:, j]
                pressure = pressures[:, j]
                graph = graphs[j]

                stats = avm.get_stats(graph, no_injection_graph, injection_pressure_mmHg=label[1], injection_location=label[0])

                stats["Blood pressure hypotension"] = label[2]
                stats["CVP pressure"] = label[3]
                stats["Cardiac phase"] = label[4]
                stats["Num columns"] = num_columns
                stats["Num compartments"] = num_compartments
                stats["Num cross vessels"] = num_intercompartmental_vessels
                stats["Fistula compartment index"] = drainers[len(drainers) // 2][2]
                stats["Occluded"] = occluded[1] if occluded else None

                for key, value in stats.items():

                    if key not in all_stats:
                        all_stats[key] = []

                    all_stats[key].append(value)

                if CALCULATE_ERROR:

                    if "Error" not in all_stats:
                        all_stats["Error"] = []

                    all_stats["Error"].append(error)

                if occluded:
                    graph.add_edge(occluded[0], occluded[1], occluded=True, pressure=0, **occluded[2])
                
                # print(label, occluded, stats["Intranidal pressure min (mmHg)"], stats["Intranidal pressure mean (mmHg)"], stats["Intranidal pressure max (mmHg)"], stats["Min rupture risk (%)"], stats["Mean rupture risk (%)"], stats["Max rupture risk (%)"])
                # print([(edge[0], edge[1], edge[2]["pressure"]) for edge in graph.edges(data=True) if edge[2]["type"] == avm.vessel.feeder])
                # print([(edge[0], edge[1], edge[2]["pressure"]) for edge in graph.edges(data=True) if edge[2]["type"] == avm.vessel.drainer])

                injection_location, injection_pressure, hypotension, cvp, cardiacPhase = label

                # Flow
                if False and injection_location == 'DV2' and injection_pressure == 30 and hypotension == 'moderate' and cvp == 'normal' and cardiacPhase == 'average':
                    print(label, occluded, stats["Percent filled post-injection (%)"])
                    plt.figure(figsize=(1920/100, 1080/100))
                    figures.display_filling(graph, node_pos)
                    plt.text(0.01, 0.99, f"Hypotension: {hypotension}\nInjection: {injection_pressure} mmHg\nFilling: {int(stats['Percent filled post-injection (%)'])}%", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')
                    plt.show()

                # plt.text(0.01, 0.99, f"{label[2]}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                # plt.text(0.01, 0.96, f"Total Nidal Flow: {int(stats['Feeder total flow (mL/min)'])} mL/min", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

                # filename = f"temp/{j:02d}_{label[2]}_{label[0]}_flow.png"
                # plt.savefig(filename)
                # # plt.show()
                # plt.close()

                # # Pressure
                # plt.figure(figsize=(1920/100, 1080/100))
                # figures.display_pressure(graph, node_pos)
                # plt.show()

                # plt.text(0.01, 0.99, f"{label[2]}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                # plt.text(0.01, 0.96, f"Mean Vessel Rupture Risk: {int(stats['Mean rupture risk (%)'])}%", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

                # filename = f"temp/{j:02d}_{label[2]}_{label[0]}_pressure.png"
                # plt.savefig(filename)
                # # plt.show()
                # plt.close()

        df = pd.DataFrame(all_stats)
        file_exists = os.path.isfile(FILE_NAME)
        df.to_csv(FILE_NAME, mode="a", index=False, header=not file_exists)

    logger.info(f"{time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
