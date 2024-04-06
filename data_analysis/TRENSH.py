import pandas as pd
from pathlib import Path
import ast
import matplotlib.pyplot as plt

file_path = Path(__file__).parent / "10k_stats.csv"
data = pd.read_csv(file_path)

def get_blood_pressure(label):
    if label[0:2] == "DV": return label[12:]
    else: return label

def get_injection_pressure(label):
    if label[0:2] != "DV": return 0
    else: return label[4:6]

def get_DV(label):
    if label[0:2] == "DV": return label[2]
    else: return 0

data["Blood pressure"] = data["Label"].apply(lambda x: get_blood_pressure(x))
data["Injection pressure"] = data["Label"].apply(lambda x: get_injection_pressure(x))
data["Injection location"] = data["Label"].apply(lambda x: get_DV(x))
data["Average plexiform pressure"] = data["Plexiform pressure (mmHg)"].apply(lambda x: ast.literal_eval(x)[1])
means = data.groupby(["Blood pressure", "Injection pressure", "Injection location"])["Average plexiform pressure"].mean().reset_index()
# table = means.pivot(columns = "Blood pressure", index = "Injection pressure", values = "Average plexiform pressure")
print(means.to_string())

fig, ax = plt.subplots(figsize=(2, 1))
cax = ax.matshow(table, interpolation="nearest", cmap="viridis")
cbar = fig.colorbar(cax)

# Set color bar label
cbar.set_label("Average Intranidal Pressure (mmHg)", fontsize=20)

# Set ticks and labels
ax.set_xticklabels([""] + list(table.columns), rotation=45, ha="left", fontsize=15)
ax.set_yticklabels([""] + list(table.index), fontsize=15)

# Set axis labels
ax.set_xlabel("Blood Pressure Level", fontsize=20)
ax.set_ylabel("Injection Pressure (mmHg)", fontsize=20)

plt.show()
