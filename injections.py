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

templates = {
    ("normal", "normal"): [74, 47, 50, 17, 6],
    ("minor", "normal"): [70, 45, 48, 15, 5],
    ("moderate", "normal"): [50, 32, 34, 12, 5],
    ("profound", "normal"): [25, 15, 16, 8, 4],
    
    ("normal", "elevated"): [74, 47, 50, 17, 0],
    ("minor", "elevated"): [70, 45, 48, 15, 0],
    ("moderate", "elevated"): [50, 32, 34, 12, 2],
    ("profound", "elevated"): [25, 15, 16, 8, 2],
}

# Uncomment these if you only want to generate specific injections
injections = {

    ("", 0, "normal", "normal"): [74, 47, 47, 50, 50, 17, 17, 17, 6],

    ("DV3", 0, "moderate", "elevated"): [50, 32, 32, 34, 34, 12, 12, 12, 2],
    ("DV3", 10, "moderate", "elevated"): [50, 32, 32, 34, 34, 12, 12, 12 - 10, 2],
    ("DV3", 20, "moderate", "elevated"): [50, 32, 32, 34, 34, 12, 12, 12 - 20, 2],
    ("DV3", 30, "moderate", "elevated"): [50, 32, 32, 34, 34, 12, 12, 12 - 30, 2],

    # ("DV1", 20, "normal", "normal"): [74, 47, 47, 50, 50, 17 - 20, 17, 17, 6],
    # ("DV1", 20, "minor", "normal"): [70, 45, 45, 48, 48, 15 - 20, 15, 15, 5],
    # ("DV3", 20, "minor", "normal"): [70, 45, 45, 48, 48, 15, 15, 15 - 20, 5],
    # ("DV1", 20, "profound", "normal"): [25, 15, 15, 16, 16, 8 - 20, 8, 8, 4],
}

# Uncomment this if you want to simulate every possible injection
# for injection_location in [None, "DV1", "DV2", "DV3", "AF1", "AF2", "AF3", "AF4"]:
#     for injection_pressure in ([0, 10, 20, 30] if injection_location[0] == "A" else [10, 20, 30]) if injection_location else [0]:
#         for (hypotension, cvp), pressures in templates.items():
#             injections[(injection_location, injection_pressure, hypotension, cvp)] = [
#                 pressures[0],
#                 pressures[1] + (injection_pressure if injection_location == "AF1" else 0),
#                 pressures[1] + (injection_pressure if injection_location == "AF2" else 0),
#                 pressures[2] + (injection_pressure if injection_location == "AF3" else 0),
#                 pressures[2] + (injection_pressure if injection_location == "AF4" else 0),
#                 pressures[3] - (injection_pressure if injection_location == "DV1" else 0),
#                 pressures[3] - (injection_pressure if injection_location == "DV2" else 0),
#                 pressures[3] - (injection_pressure if injection_location == "DV3" else 0),
#                 pressures[4]
#             ]

print(f"{len(injections)} pressure sets")
for key, pressures in injections.items():
    for index, pressure in enumerate(pressures):
        pressures[index] = pressure * avm.MMHG_TO_DYN_PER_SQUARE_CM
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
