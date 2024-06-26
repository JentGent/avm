"""Generates a dict of sets of pressure values for injections in each DV at different blood pressure levels."""

# autoregulation
# pulsatility; phases of cardiac cycle?

# get total flow around 300-450:
# decrease length
# # nodes in column
# # cross-compartmental vessels

# calculate backfilling threshold
# calculate backfilling by comparing to 0 injection

import avm
import numpy as np

templates = {
    # ("normal", "normal", "systolic"): [74, 52, 52, -19, -6],
    # ("minor", "normal", "systolic"): [70, 51, 51, -17, -5],
    # ("moderate", "normal", "systolic"): [50, 34, 34, -13, -5],
    # ("profound", "normal", "systolic"): [25, 17, 17, -9, -4],

    # ("normal", "normal", "average"): [74, 47, 47, -17, -6],
    # ("minor", "normal", "average"): [70, 45, 45, -15, -5],
    # ("moderate", "normal", "average"): [50, 32, 32, -12, -5],
    # ("profound", "normal", "average"): [25, 15, 15, -8, -4],
    
    # ("normal", "elevated", "average"): [74, 47, 47, -22, -12],
    # ("minor", "elevated", "average"): [70, 45, 45, -19, -10],
    # ("moderate", "elevated", "average"): [50, 32, 32, -14, -8],
    ("profound", "elevated", "average"): [25, 15, 15, -9, -6],
    
    # ("normal", "normal", "diastolic"): [74, 43, 43, -15, -6],
    # ("minor", "normal", "diastolic"): [70, 44, 44, -14, -5],
    # ("moderate", "normal", "diastolic"): [50, 30, 30, -11, -5],
    # ("profound", "normal", "diastolic"): [25, 14, 14, -7, -4],
}

injections = {}
for injection_location in [None, "DV3"]:
    for injection_pressure in [20] if injection_location else [0]:
        for (hypotension, cvp, cardiacPhase), pressures in templates.items():
            injections[(injection_location, injection_pressure, hypotension, cvp, cardiacPhase)] = [
                pressures[0],
                pressures[1] + (injection_pressure if injection_location == "AF1" else 0),
                pressures[1] + (injection_pressure if injection_location == "AF2" else 0),
                pressures[2] + (injection_pressure if injection_location == "AF3" else 0),
                pressures[2] + (injection_pressure if injection_location == "AF4" else 0),
                pressures[3] - (injection_pressure if injection_location == "DV1" else 0),
                pressures[3] - (injection_pressure if injection_location == "DV2" else 0),
                pressures[3] - (injection_pressure if injection_location == "DV3" else 0),
                pressures[4]
            ]


print(f"{len(injections)} pressure sets")
for key, pressures in injections.items():
    injections[key] = {
        ("SP", 1): pressures[0],
        (3, "AF1"): pressures[1],
        (6, "AF2"): pressures[2],
        (6, "AF3"): pressures[3],
        (9, "AF4"): pressures[4],
        ("DV1", 11): pressures[5],
        ("DV2", 11): pressures[6],
        ("DV3", 11): pressures[7],
        (12, 13): pressures[8]
    }
