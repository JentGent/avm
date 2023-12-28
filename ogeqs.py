"""This recreates exactly the equations described in the 1996 paper to confirm that we understand the matrix math described in that paper."""

import numpy as np

# FISTULOUS is the indices of the resistances of the fistulous vessels.
FISTULOUS = [19, 24, 32, 43]

# R is the resistances of the vessels.
R = [
    0,
    67.9,  # 1
    67.9,  # 2
    5037,   # 3
    2210,   # 4
    637.5,  # 5
    1000000,   # 6
    12750000,   # 7
    1000000,   # 8
    4177.9,  # 9
    522,   # 10
    745.5,  # 11
    15725000,   # 12
    10200,   # 13
    1000000,   # 14
    4177.9,  # 15
] + [4080 if i in FISTULOUS else 81600 for i in range(16, 44 + 1)] + [
    130.5,  # 45
    130.5,  # 46
    261,   # 47
    79.7,  # 48
    3.2,  # 49
    1,   # 50
]

# E is the external pressure sources in dyn/cm^2.
E = {
    "SP": 74 * 1333.22,
    "AF1": 47 * 1333.22,
    "AF2": 47 * 1333.22,
    "AF3": 50 * 1333.22,
    "AF4": 50 * 1333.22,
    "DV1": 17 * 1333.22,
    "DV2": 17 * 1333.22,
    "CVP": 5 * 1333.22
}

# EQS lists the equations. The first entry of each list is the lefthand side; the second is the right. The left is a dictionary where each key is a flow index and each value is its coefficient.
EQS = [
    [{41: 1, 1: -1, 2: -1}, 0],  # 1
    [{2: 1, 3: -1, 6: -1}, 0],  # 4
    [{6: 1, 7: -1, 8: -1, 9: -1}, 0],  # 6
    [{3: 1, 4: -1, 5: -1}, 0],  # 9
    [{5: 1, 9: 1, 38: 1, 39: 1, 40: -1}, 0],  # 11
    [{1: 1, 21: -1, 22: -1}, 0],  # 12
    [{8: 1, 10: -1, 11: -1}, 0],  # 13
    [{7: 1, 12: -1, 13: -1, 14: -1}, 0],  # 14
    [{12: 1, 15: -1, 16: -1}, 0],  # 15
    [{13: 1, 17: -1, 18: -1}, 0],  # 16
    [{14: 1, 19: -1, 20: -1}, 0],  # 17
    [{10: 1, 4: 1, 31: -1}, 0],  # 18
    [{11: 1, 15: 1, 29: -1, 30: -1}, 0],  # 19
    [{16: 1, 17: 1, 27: -1, 28: -1}, 0],  # 20
    [{18: 1, 19: 1, 25: -1, 26: -1}, 0],  # 21
    [{20: 1, 21: 1, 23: -1, 24: -1}, 0],  # 22
    [{30: 1, 31: 1, 32: -1}, 0],  # 24
    [{28: 1, 29: 1, 33: -1}, 0],  # 25
    [{26: 1, 27: 1, 34: -1, 35: -1}, 0],  # 26
    [{24: 1, 25: 1, 36: -1}, 0],  # 27
    [{22: 1, 23: 1, 37: -1}, 0],  # 28
    [{32: 1, 33: 1, 34: 1, 38: -1}, 0],  # 29
    [{35: 1, 36: 1, 37: 1, 39: -1}, 0],  # 30
    # [{41: 1, 40: -1}, 0],  # 30

    [{3: R[5] + R[6], 4: R[7], 10: -R[16], 8: -R[12], 6: -R[10]}, E['AF4'] - E['AF3']],  # 1
    [{5: R[8] + R[9], 38: -R[45], 32: -R[39], 31: -R[38], 4: -R[7]}, -E["DV2"] - E["AF4"]],  # 2
    [{9: R[13] + R[14] + R[15], 38: -R[45], 32: -R[39], 31: -R[38], 10: -R[16], 8: -R[12]}, -E["DV2"] - E["AF3"]],  # 3
    [{2: R[2], 6: R[10], 7: R[11], 14: R[20], 20: R[26], 21: -R[27], 1: -(R[4] + R[3] + R[1])}, E["AF2"] - E["AF1"]],  # 4
    [{8: R[12], 11: R[17], 15: -R[21], 12: -R[18], 7: -R[11]}, E["AF3"] - E["AF2"]],  # 5
    [{12: R[18], 16: R[22], 17: -R[23], 13: -R[19]}, 0],  # 6
    [{13: R[19], 18: R[24], 19: -R[25], 14: -R[20]}, 0],  # 7
    [{10: R[16], 31: R[38], 30: -R[37], 11: -R[17]}, 0],  # 8
    [{15: R[21], 29: R[36], 28: -R[35], 16: -R[22]}, 0],  # 9
    [{17: R[23], 27: R[34], 26: -R[33], 18: -R[24]}, 0],  # 10
    [{19: R[25], 25: R[32], 24: -R[31], 20: -R[26]}, 0],  # 11
    [{21: R[27], 23: R[30], 22: -(R[29] + R[28])}, 0],  # 12
    [{30: R[37], 32: R[39], 33: -R[40], 29: -R[36]}, 0],  # 13
    [{28: R[35], 33: R[40], 34: -R[41], 27: -R[34]}, 0],  # 14
    [{26: R[33], 35: R[42], 36: -R[43], 25: -R[32]}, 0],  # 15
    [{24: R[31], 36: R[43], 37: -R[44], 23: -R[30]}, 0],  # 16
    [{34: R[41], 38: R[45], 39: -R[46], 35: -R[42]}, E["DV2"] - E["DV1"]],  # 17
    [{1: R[1] + R[3] + R[4], 22: R[28] + R[29], 37: R[44], 39: R[46], 40: R[47] + R[48] + R[49], 41: R[50]}, E["AF1"] + E["DV1"] + E["CVP"] + E["SP"]]
]


def main():
    num_edges = 41
    Rv = np.array([
        [eq[0][edge] if edge in eq[0] else 0 for edge in range(1, num_edges + 1)] for eq in EQS
    ])
    ddP = np.array([
        eq[1] for eq in EQS
    ])

    flows, r, N, rank = np.linalg.lstsq(Rv, ddP.T)

    print(f"Flows shape: {flows.shape}")
    print(f"Error: {r}")
    print(f"Flows: {np.round(flows * 60, 3)}")
    print(f"Vessel flow range: {np.min(flows) * 60, np.max(flows) * 60}")
    print(f"Total flow through nidus (out): {(flows[37] + flows[38]) * 60}")
    print(f"Total flow through nidus (in): {(flows[3] + flows[7] + flows[6] + flows[0]) * 60}")
    print(f"Fistulous max flow: {max(flows[12], flows[17], flows[24], flows[35]) * 60}")


if __name__ == "__main__":
    main()
