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

# (hypotension, cvp, cardiac_phase): [systemic_pressure, af1_and_af2, af3_and_af4, dv, cvp]
templates = {
    ("normal", "normal", "average"): [74, 47, 47, -17, -6],
    ("minor", "normal", "average"): [70, 45, 45, -15, -5],
    ("moderate", "normal", "average"): [50, 32, 32, -12, -5],
    ("profound", "normal", "average"): [25, 15, 15, -8, -4],
    
    ("normal", "elevated", "average"): [74, 47, 47, -22, -12],
    ("minor", "elevated", "average"): [70, 45, 45, -19, -10],
    ("moderate", "elevated", "average"): [50, 32, 32, -14, -8],
    ("profound", "elevated", "average"): [25, 15, 15, -9, -6],
    
    ("normal", "normal", "systolic"): [74, 52, 52, -19, -6],
    ("minor", "normal", "systolic"): [70, 51, 51, -17, -5],
    ("moderate", "normal", "systolic"): [50, 34, 34, -13, -5],
    ("profound", "normal", "systolic"): [25, 17, 17, -9, -4],

    ("normal", "normal", "diastolic"): [74, 43, 43, -15, -6],
    ("minor", "normal", "diastolic"): [70, 44, 44, -14, -5],
    ("moderate", "normal", "diastolic"): [50, 30, 30, -11, -5],
    ("profound", "normal", "diastolic"): [25, 14, 14, -7, -4],
}

def generate_pressure_set(injection_location, injection_pressure, hypotension, cvp, cardiacPhase):
    template = templates[(hypotension, cvp, cardiacPhase)]
    return {
        ("SP", 1): template[0],
        (3, "AF1"): template[1] + (injection_pressure if injection_location == "AF1" else 0),
        (6, "AF2"): template[1] + (injection_pressure if injection_location == "AF2" else 0),
        (6, "AF3"): template[2] + (injection_pressure if injection_location == "AF3" else 0),
        (9, "AF4"): template[2] + (injection_pressure if injection_location == "AF4" else 0),
        ("DV1", 11): template[3] - (injection_pressure if injection_location == "DV1" else 0),
        ("DV2", 11): template[3] - (injection_pressure if injection_location == "DV2" else 0),
        ("DV3", 11): template[3] - (injection_pressure if injection_location == "DV3" else 0),
        (12, 13): template[4]
    }

injections = {}
for injection_location in [None, "DV1", "DV2", "DV3"]:
    for injection_pressure in [10, 20, 30] if injection_location else [0]:
        for (hypotension, cvp, cardiacPhase), pressures in templates.items():
            key = (injection_location, injection_pressure, hypotension, cvp, cardiacPhase)
            injections[key] = generate_pressure_set(*key)


print(f"{len(injections)} pressure sets")