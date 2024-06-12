"""Randomly generates a nidus with nodes arranged in columns."""

from injections import injections
from itertools import combinations
import avm
import copy
import figures
import generate
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import time

# CALCULATE_ERROR indicates whether or not Kirchoff law pressure error is calculated for each simulation. This slows down the simulation noticeably. Disable when avm.SOLVE_MODE is set to numpy.lstsq because that is pretty much guaranteed to be accurate.
CALCULATE_ERROR = True

# FIRST_INTRANIDAL_NODE_ID is the ID of the first intranidal node (must be updated with NODE_POS).
FIRST_INTRANIDAL_NODE_ID = max(k for k in avm.NODE_POS_TEMPLATE.keys() if type(k) == int) + 1

def main():

    avm.PREDEFINED_RESISTANCE = False
    start_time = time.time()

    feeders = ['AF1', 'AF2', 'AF3', 'AF4']
    drainers = ['DV1', 'DV2', 'DV3']

    injection_pressures = list(injections.values())

    node_pos = copy.deepcopy(avm.NODE_POS_TEMPLATE)
    while True:

        num_compartments = generate.normint(3, 6, sd=1)
        num_columns = generate.normint(3, 7, sd=1)
        num_compartments = 6
        num_columns = 7
        num_intercompartmental_vessels = num_compartments * num_columns * 2
        print(f"{num_compartments} compartments, {num_columns} columns")
        
        network = avm.edges_to_graph(avm.VESSELS_TEMPLATE)
        network, _ = generate.compartments(network, feeders, drainers, FIRST_INTRANIDAL_NODE_ID, node_pos, num_compartments, num_columns, num_intercompartmental_vessels, fistula_start = "AF2", fistula_end = "DV2")

        flows, pressures, all_edges, graphs, *error = avm.simulate_batch(network, [], injection_pressures, CALCULATE_ERROR)
        max_jump, max_fill, prev = 0, 0, 0
        for graph in graphs:
            new = avm.get_stats(graph, graphs[0], 6, 0, "DV3")["Percent filled post-injection (%)"]
            max_fill = max(new, max_fill)
            if new - prev > max_jump:
                max_jump = new - prev
            prev = new
        print(max_jump, max_fill)
        if max_jump < 20 and max_fill > 30 and max_fill < 40: break
    print("generated")
    error = error if error else None

    filled_vessels = set([(11, "DV3")])
    frame = 0
    for j, label in enumerate(injections.keys()):
        (injection_location, injection_pressure, hypotension, cvp) = label

        flow = flows[:, j]
        pressure = pressures[:, j]
        graph = graphs[j]

        vessel_count = 0
        for start, end, type in graph.edges.data("type"):
            if type in [avm.vessel.fistulous, avm.vessel.plexiform, avm.vessel.drainer]:
                vessel_count += 1

        print(label)

        while True:

            new_filled_vessels = set()
            if injection_location:
                for vessel in graph.out_edges(injection_location):
                    new_filled_vessels.add(vessel)
            for vessel in filled_vessels:
                if vessel not in graph.edges: vessel = (vessel[1], vessel[0])
                for next_vessel in graph.out_edges(vessel[1]):
                    if graph.edges[next_vessel]["type"] in [avm.vessel.fistulous, avm.vessel.plexiform, avm.vessel.drainer]:
                        new_filled_vessels.add(next_vessel)
            
            if new_filled_vessels == filled_vessels or injection_location is None:
                plt.figure(figsize=(1920/100, 1080/100))
                nx.set_edge_attributes(graph, False, "reached")
                for vessel in filled_vessels:
                    if vessel not in graph.edges: vessel = (vessel[1], vessel[0])
                    graph.edges[vessel]["reached"] = True
                figures.display_filling(graph, node_pos)
                # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

                plt.text(0.01, 0.99, f"Injection pressure: {int(injection_pressure)} mmHg\nFilling: {int(len(filled_vessels) / vessel_count * 100)}%", transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')

                filename = f"temp/filling_injection/profound_20_DV3/{frame:03d}_{label[2]}_{label[0]}.png"
                plt.savefig(filename)
                # plt.show()
                plt.close()
                frame += 1

            if new_filled_vessels == filled_vessels:
                break
            filled_vessels = new_filled_vessels

    print(f"{time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
