"""Generates a dict of 80 sets of pressure values for injections in each DV at different blood pressure levels."""

# try different back-filling algorithms
# show what happens when injecting into feeders; rupture risk?
# autoregulation
# pulsatility; phases of cardiac cycle?

# get total flow around 300-450:
# decrease length
# # nodes in column
# # cross-compartmental vessels
# cross-compartmental vessels are MORE LIKELY to be within near columns; not ONLY

import avm

injections = {}
for dv in [0, 1, 2, 3]:
    name = f"DV{dv} " if dv else ""
    for i in [10, 20, 30] if dv else [0]:
        injection = f"{i} mmHg " if i else ""

        p = 17
        injections[(dv, i, "normal", "normal")] = [74, 47, 47, 50, 50, ((p - i) if dv == 1 else p), ((p - i) if dv == 2 else p), ((p - i) if dv == 3 else p), 6]

        p = 15
        injections[(dv, i, "minor", "normal")] = [70, 45, 45, 48, 48, ((p - i) if dv == 1 else p), ((p - i) if dv == 2 else p), ((p - i) if dv == 3 else p), 5]

        p = 12
        injections[(dv, i, "moderate", "normal")] = [50, 32, 32, 34, 34, ((p - i) if dv == 1 else p), ((p - i) if dv == 2 else p), ((p - i) if dv == 3 else p), 5]

        p = 8
        injections[(dv, i, "profound", "normal")] = [25, 15, 15, 16, 16, ((p - i) if dv == 1 else p), ((p - i) if dv == 2 else p), ((p - i) if dv == 3 else p), 4]
        
        p = 22
        injections[(dv, i, "normal", "elevated")] = [74, 49, 49, 52, 52, ((p - i) if dv == 1 else p), ((p - i) if dv == 2 else p), ((p - i) if dv == 3 else p), 12]

        p = 19
        injections[(dv, i, "minor", "elevated")] = [70, 47, 47, 50, 50, ((p - i) if dv == 1 else p), ((p - i) if dv == 2 else p), ((p - i) if dv == 3 else p), 10]

        p = 14
        injections[(dv, i, "moderate", "elevated")] = [50, 33, 33, 35, 35, ((p - i) if dv == 1 else p), ((p - i) if dv == 2 else p), ((p - i) if dv == 3 else p), 8]

        p = 9
        injections[(dv, i, "profound", "elevated")] = [25, 16, 16, 17, 17, ((p - i) if dv == 1 else p), ((p - i) if dv == 2 else p), ((p - i) if dv == 3 else p), 6]

for key, pressures in injections.items():
    for index, pressure in enumerate(pressures):
        injections[key][index] = pressure * avm.MMHG_TO_DYN_PER_SQUARE_CM
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
