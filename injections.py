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
    ("profound", "normal"): [25, 15, 16, 8, 4],
    ("minor", "normal"): [70, 45, 48, 15, 5],
    ("moderate", "normal"): [50, 32, 34, 12, 5],
    ("normal", "normal"): [74, 47, 50, 17, 6],
    ("normal", "elevated"): [74, 49, 52, 22, 12],
    ("minor", "elevated"): [70, 47, 50, 19, 10],
    ("moderate", "elevated"): [50, 33, 35, 14, 8],
    ("profound", "elevated"): [25, 16, 17, 9, 6],
    ("profound", "normal"): [25, 15, 16, 8, 4],
}

# Uncomment these if you only want to generate specific injections
injections = {
    # ("DV1", 30, "minor", "normal"): [70, 45, 45, 48, 48, 15 - 30, 15, 15, 5],
    # ("DV1", 30, "moderate", "normal"): [50, 32, 32, 34, 34, 12 - 30, 12, 12, 5],
    # ("DV1", 30, "profound", "normal"): [25, 15, 15, 16, 16, 8 - 30, 8, 8, 4],
    # ("DV2", 30, "profound", "normal"): [25, 15, 15, 16, 16, 8, 8 - 30, 8, 4],
    # ("AF2", 30, "profound", "normal"): [25, 15, 15 + 30, 16, 16, 8, 8, 8, 4]
}

# Uncomment this if you want to simulate every possible injection
for injection_location in [None, "DV1", "DV2", "DV3", "AF1", "AF2", "AF3", "AF4"]:
    name = f"DV{injection_location} " if injection_location else ""
    for injection_pressure in [10, 20, 30] if injection_location else [0]:
        for (hypotension, cvp), pressures in templates.items():
            injections[(injection_location, injection_pressure, hypotension, cvp)] = [
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
