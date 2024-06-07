import multiprocessing
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

CALCULATE_ERROR = True
ITERATIONS = 10
FILE_NAME = "data.csv"
FIRST_INTRANIDAL_NODE_ID = max(k for k in avm.NODE_POS_TEMPLATE.keys() if type(k) == int) + 1

def worker(i):
    feeders = ['AF1', 'AF2', 'AF3', 'AF4']
    drainers = ['DV1', 'DV2', 'DV3']
    injection_pressures = list(injections.values())
    all_stats = {}
    
    num_compartments = generate.normint(4, 6)
    num_columns = generate.normint(6, 10)
    num_intercompartmental_vessels = generate.normint(90, 110)
    
    node_pos = copy.deepcopy(avm.NODE_POS_TEMPLATE)
    network = avm.edges_to_graph(avm.VESSELS_TEMPLATE)
    network, _ = generate.compartments(network, feeders, drainers, FIRST_INTRANIDAL_NODE_ID, node_pos, num_compartments, num_columns, num_intercompartmental_vessels, fistula_start="AF2", fistula_end="DV2")
    
    flows, pressures, all_edges, graphs, *error = avm.simulate_batch(network, [], injection_pressures, CALCULATE_ERROR)
    error = error if error else None
    
    for j, label in enumerate(injections.keys()):
        no_injection_graph = graphs[list(injections.keys()).index((None, 0, label[2], label[3]))] if (None, 0, label[2], label[3]) in injections else None
        flow = flows[:, j]
        pressure = pressures[:, j]
        graph = graphs[j]
        stats = avm.get_stats(graph, no_injection_graph, abs(injections[label][(12, 13)]), label[1], label[0])
        
        stats.update({
            "Blood pressure hypotension": label[2],
            "CVP pressure": label[3],
            "Num columns": num_columns,
            "Num compartments": num_compartments,
            "Num cross vessels": num_intercompartmental_vessels,
            "Fistula compartment number": drainers[len(drainers) // 2][2]
        })
        
        if CALCULATE_ERROR:
            stats["Error"] = error
        
        for key, value in stats.items():
            if key not in all_stats:
                all_stats[key] = []
            all_stats[key].append(value)
        
    return all_stats

def main():
    if os.path.exists(FILE_NAME):
        os.remove(FILE_NAME)
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(worker, range(1, 1 + ITERATIONS))
    
    all_stats = {}
    for stats in results:
        for key, value in stats.items():
            if key not in all_stats:
                all_stats[key] = []
            all_stats[key] += value
    
    df = pd.DataFrame(all_stats)
    df.to_csv(FILE_NAME, index=False)
    
    print("Completed in {} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    start_time = time.time()
    main()