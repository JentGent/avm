import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data = pd.read_csv(Path(__file__).parent / 'data .csv')

DV = "Max rupture risk (%)"

# Filter data to include only data where the Injection location starts with 'D'
filtered_data = data[
    # data["Injection location"].notna() &
    (data["Injection location"].isna() | data["Injection location"].str.startswith('D'))
    # data["Blood pressure hypotension"].str.lower().eq(["normal", "minor", "moderate", "profound"][3])
]

# Calculate the statistics for each group
grouped_data = filtered_data.groupby(
    ["Injection pressure (mmHg)", "CVP pressure", "Blood pressure hypotension"]
).agg(
    Mean=pd.NamedAgg(column=DV, aggfunc='mean'),
    SD=pd.NamedAgg(column=DV, aggfunc='std'),
    N=pd.NamedAgg(column=DV, aggfunc='count')
).reset_index()

# Reshape the data for specific format - 6 columns, 3 rows
pivot_data = grouped_data.pivot(index='Injection pressure (mmHg)', columns=['CVP pressure', 'Blood pressure hypotension'], values=['Mean', 'SD', 'N'])
pivot_data.columns = [' '.join(col).strip() for col in pivot_data.columns.values]
print(grouped_data)
pivot_data = pivot_data[[
    "Mean normal normal", "SD normal normal", "N normal normal", "Mean elevated normal", "SD elevated normal", "N elevated normal",
    "Mean normal minor", "SD normal minor", "N normal minor", "Mean elevated minor", "SD elevated minor", "N elevated minor",
    "Mean normal moderate", "SD normal moderate", "N normal moderate", "Mean elevated moderate", "SD elevated moderate", "N elevated moderate",
    "Mean normal profound", "SD normal profound", "N normal profound", "Mean elevated profound", "SD elevated profound", "N elevated profound"
]]

# Save to CSV
csv_path = Path(__file__).parent / 'grouped_data.csv'
pivot_data.to_csv(csv_path, index=True)

print("CSV file has been saved to:", csv_path)