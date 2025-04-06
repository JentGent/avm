"""Generates the frames of an animation where the vessel is colored if it is filled.

To use this, change the constants at the top of the script. Then run."""

import injections
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

# settings
HYPOTENSION = ["normal", "minor", "moderate", "profound"][3]
CVP = ["normal", "elevated"][0]
CARDIAC_PHASE = ["average", "systolic", "diastolic"][0]
OCCLUDED = [None, "AF1", "AF2", "AF3", "AF4"][0]
INJECTION_LOCATION = ["DV1", "DV2", "DV3"][2]
MAX_INJECTION_PRESSURE = 20
MAX_JUMP = 37
DESIRED_FILL = 35

FOLDER = f"temp/filling_injection/{HYPOTENSION}_{CVP}_{CARDIAC_PHASE}_{MAX_INJECTION_PRESSURE}_{INJECTION_LOCATION}_{OCCLUDED}"

def main():

    avm.PREDEFINED_RESISTANCE = False
    start_time = time.time()

    feeders = ['AF1', 'AF2', 'AF3', 'AF4']
    drainers = ['DV1', 'DV2', 'DV3']

    pressure_sets = [injections.generate_pressure_set(INJECTION_LOCATION if injection_pressure else None, injection_pressure, HYPOTENSION, CVP, CARDIAC_PHASE) for injection_pressure in range(0, MAX_INJECTION_PRESSURE + 1)]

    while True:

        node_pos = avm.NODE_POS_TEMPLATE
        num_compartments = generate.normint(3, 6, sd=1)
        num_columns = generate.normint(3, 7, sd=1)
        num_compartments = 6
        num_columns = 7
        num_intercompartmental_vessels = num_compartments * num_columns * 2
        print(f"{num_compartments} compartments, {num_columns} columns")
        
        network = avm.edges_to_graph(avm.VESSELS_TEMPLATE)
        network, _ = generate.compartments(network, feeders, drainers, FIRST_INTRANIDAL_NODE_ID, node_pos, num_compartments, num_columns, num_intercompartmental_vessels, fistula_start = "AF2", fistula_end = "DV2")
        occluded = [edge for edge in network.edges(data=True) if edge[1] == OCCLUDED][0] if OCCLUDED else None
        if occluded:
            network.remove_edge(occluded[0], occluded[1])

        flows, pressures, all_edges, graphs, *error = avm.simulate_batch(network, "SP", 0, pressure_sets, CALCULATE_ERROR)
        max_jump, max_fill, prev = 0, 0, 0

        for injection_pressure, graph in enumerate(graphs):
            new = avm.get_stats(graph, graphs[0], 4, injection_pressure, INJECTION_LOCATION)["Percent filled post-injection (%)"]
            max_fill = max(new, max_fill)
            max_jump = max(new - prev, max_jump)
            prev = new

        # Generate architectures until we find one that satisfies the criteria below
        print(f"Max jump={max_jump}, max fill={max_fill}")
        if max_jump <= MAX_JUMP and max_fill >= DESIRED_FILL:
            break

    print("generated")
    error = error if error else None
    
    if os.path.exists(FOLDER):
        raise FileExistsError(f"Folder already exists at {FOLDER}")
    else:
        os.makedirs(FOLDER)
        print(f"Folder created at {FOLDER}")

    animation_phase = 0
    filled_vessels = set()
    frame = 0
    for injection_pressure in list(range(0, MAX_INJECTION_PRESSURE + 1)) + [0]:

        injection_location = INJECTION_LOCATION if injection_pressure else None
        
        flow = flows[:, injection_pressure]
        pressure = pressures[:, injection_pressure]
        graph = graphs[injection_pressure]

        if occluded:
            graph.add_edge(occluded[0], occluded[1], occluded=True, pressure=0, **occluded[2])

        vessel_count = 0
        for start, end, type in graph.edges.data("type"):
            if type in [avm.vessel.fistulous, avm.vessel.plexiform, avm.vessel.drainer, avm.vessel.feeder]:
                vessel_count += 1

        while True:

            # A fresh set to store which vessels are filled in this frame
            new_filled_vessels = set()

            # If injection is active, seed filling from the injection site
            if injection_location:
                for vessel in graph.out_edges(injection_location):
                    new_filled_vessels.add(vessel)

            # Propagate filling from vessels that were filled from the prev frame. During injection, this is a BFS (starting from the injection site). During exit, this iterates through each vessel that was filled in the prev frame and makes sure its outgoing neighbors (but not necessarily itself) will be filled in the current frame.
            for vessel in filled_vessels:

                # Since filled_vessels edges are stored as non-directional
                if vessel not in graph.edges:
                    vessel = (vessel[1], vessel[0])

                for next_vessel in graph.out_edges(vessel[1]):
                    if graph.edges[next_vessel]["type"] in [avm.vessel.fistulous, avm.vessel.plexiform, avm.vessel.drainer, avm.vessel.feeder]:
                        new_filled_vessels.add(next_vessel)
            
            # Injection Phase: Stop the BFS only when we're no longer adding anything new
            # Exit Phase: Always stop the BFS so that we get a frame-by-frame look
            if new_filled_vessels == filled_vessels or injection_location is None:

                plt.figure(figsize=(1920/100, 1080/100))
                nx.set_edge_attributes(graph, False, "reached")
                for vessel in filled_vessels:
                    if vessel not in graph.edges:
                        vessel = (vessel[1], vessel[0])
                    graph.edges[vessel]["reached"] = True
                figures.display_filling(avm.get_nidus(graph, True), node_pos)
                # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

                plt.text(0.01, 0.99, f"Injection pressure: {int(injection_pressure)} mmHg\nFilling: {len(filled_vessels) / vessel_count * 100:.1f}%", transform=plt.gca().transAxes, fontsize=30, verticalalignment='top')

                filename = FOLDER + f"/{frame:03d}.png"
                
                plt.savefig(filename)
                # plt.show()
                plt.close()
                frame += 1

                # Injection Phase: Now that we've added a frame, break so we can do the next injection pressure
                if injection_location:
                    break

                # Exit Phase: Stop adding new frames if the frame we just saved has no filled vessels
                if len(filled_vessels) == 0:
                    break

            filled_vessels = new_filled_vessels

    print(f"{time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
