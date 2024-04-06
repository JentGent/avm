import pandas as pd
from pathlib import Path
import ast
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

file_path = Path(__file__).parent / "10k_stats.csv"
data = pd.read_csv(file_path)

fig, axs = plt.subplots(5, 1, figsize = (10, 25))
titles = [
    "Plexiform pressure vs. # vessels",
    "Fistulous pressure vs. # vessels",
    "Total nidal flow vs. # vessels",
    "Mean rupture risk vs. # vessels",
    "Max rupture risk vs. # vessels"
]
x_labels = ["# intranidal vessels"] * 5
y_labels = ["Plexiform Pressure (mmHg)", "Fistulous Pressure (mmHg)", "Total Nidal Flow (mL/min)", "Mean Rupture Risk (%)", "Max Rupture Risk (%)"]

x_data = data["Num intranidal vessels"]
y_data = [data["Plexiform pressure (mmHg)"].apply(lambda x: ast.literal_eval(x)[1]), data["Fistulous pressure (mmHg)"].apply(lambda x: ast.literal_eval(x)[1]), data["Total nidal flow (mL/min)"], data["Mean rupture risk (%)"], data["Max rupture risk (%)"]]

for ax, title, x_label, y_label, y in zip(axs, titles, x_labels, y_labels, y_data):
    ax.scatter(x_data, y, alpha = 0.3, s = 3)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

plt.tight_layout()
plt.show()
