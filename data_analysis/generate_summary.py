import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data = pd.read_csv(Path(__file__).parent.parent / 'datasets/11083.csv')

DV = "Percent filled (%)"
DV = "Percent filled post-injection (%)"
# DV = "Drainer total flow (mL/min)"
# DV = "Mean rupture risk (%)"
# DV = "Max rupture risk (%)"

filtered_data = data
filtered_data = data[
    data["Injection location"].notna()
    & data["Injection location"].str.startswith('D')
]
# print(filtered_data)

# Calculate the statistics for each group
grouped_data = filtered_data.groupby(
    ["Injection pressure (mmHg)", "Injection location", "CVP pressure", "Blood pressure hypotension"]
).agg(
    Mean=pd.NamedAgg(column=DV, aggfunc='mean'),
    SD=pd.NamedAgg(column=DV, aggfunc='std'),
    N=pd.NamedAgg(column=DV, aggfunc='count')
).reset_index()

# Reshape the data for specific format - 6 columns, 3 rows
pivot_data = grouped_data.pivot(index="Injection pressure (mmHg)", columns=["Injection location", "CVP pressure", "Blood pressure hypotension"], values=['Mean', 'SD', 'N'])
pivot_data.columns = [' '.join(col).strip() for col in pivot_data.columns.values]
print(grouped_data)
print(pivot_data)
pivot_data = pivot_data[[
    "Mean DV1 normal normal", "SD DV1 normal normal", "N DV1 normal normal", "Mean DV2 normal normal", "SD DV2 normal normal", "N DV2 normal normal", "Mean DV3 normal normal", "SD DV3 normal normal", "N DV3 normal normal", "Mean DV1 elevated normal", "SD DV1 elevated normal", "N DV1 elevated normal", "Mean DV2 elevated normal", "SD DV2 elevated normal", "N DV2 elevated normal", "Mean DV3 elevated normal", "SD DV3 elevated normal", "N DV3 elevated normal",
    
    "Mean DV1 normal minor", "SD DV1 normal minor", "N DV1 normal minor", "Mean DV2 normal minor", "SD DV2 normal minor", "N DV2 normal minor", "Mean DV3 normal minor", "SD DV3 normal minor", "N DV3 normal minor", "Mean DV1 elevated minor", "SD DV1 elevated minor", "N DV1 elevated minor", "Mean DV2 elevated minor", "SD DV2 elevated minor", "N DV2 elevated minor", "Mean DV3 elevated minor", "SD DV3 elevated minor", "N DV3 elevated minor",
    
    "Mean DV1 normal moderate", "SD DV1 normal moderate", "N DV1 normal moderate", "Mean DV2 normal moderate", "SD DV2 normal moderate", "N DV2 normal moderate", "Mean DV3 normal moderate", "SD DV3 normal moderate", "N DV3 normal moderate", "Mean DV1 elevated moderate", "SD DV1 elevated moderate", "N DV1 elevated moderate", "Mean DV2 elevated moderate", "SD DV2 elevated moderate", "N DV2 elevated moderate", "Mean DV3 elevated moderate", "SD DV3 elevated moderate", "N DV3 elevated moderate",
    
    "Mean DV1 normal profound", "SD DV1 normal profound", "N DV1 normal profound", "Mean DV2 normal profound", "SD DV2 normal profound", "N DV2 normal profound", "Mean DV3 normal profound", "SD DV3 normal profound", "N DV3 normal profound", "Mean DV1 elevated profound", "SD DV1 elevated profound", "N DV1 elevated profound", "Mean DV2 elevated profound", "SD DV2 elevated profound", "N DV2 elevated profound", "Mean DV3 elevated profound", "SD DV3 elevated profound", "N DV3 elevated profound",
    # "Mean normal normal", "SD normal normal", "N normal normal", "Mean elevated normal", "SD elevated normal", "N elevated normal",
    # "Mean normal minor", "SD normal minor", "N normal minor", "Mean elevated minor", "SD elevated minor", "N elevated minor",
    # "Mean normal moderate", "SD normal moderate", "N normal moderate", "Mean elevated moderate", "SD elevated moderate", "N elevated moderate",
    # "Mean normal profound", "SD normal profound", "N normal profound", "Mean elevated profound", "SD elevated profound", "N elevated profound"
]]

# Save to CSV
csv_path = Path(__file__).parent / 'grouped_data.csv'
pivot_data.to_csv(csv_path, index=True)

print("CSV file has been saved to:", csv_path)