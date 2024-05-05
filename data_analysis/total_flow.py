import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data = pd.read_csv(Path(__file__).parent / 'data.csv')

# Filter data to include only data where the Injection location starts with 'D'
filtered_data = data[
    data["Injection location"].notna() &
    data["Injection location"].str.startswith('D')
]

# Calculate the average percent filled for each group
grouped_data = filtered_data.groupby(
    ["Blood pressure hypotension", "Injection pressure (mmHg)", "CVP pressure"]
).agg(
    Total_flow=pd.NamedAgg(column="Feeder total flow (mL/min)", aggfunc='mean')
).reset_index()

# Define order for "Blood pressure hypotension" and "CVP pressure"
hypotension_order = ['normal', 'minor', 'moderate', 'profound']
cvp_order = ['normal', 'elevated']
grouped_data['Blood pressure hypotension'] = pd.Categorical(
    grouped_data['Blood pressure hypotension'],
    categories=hypotension_order,
    ordered=True
)
grouped_data['CVP pressure'] = pd.Categorical(
    grouped_data['CVP pressure'],
    categories=cvp_order,
    ordered=True
)

# Plotting
sns.set_theme(style="whitegrid", font_scale=1.5)

# Create a FacetGrid, mapping data to bar plots
g = sns.FacetGrid(
    grouped_data, 
    col="Blood pressure hypotension", 
    col_order=hypotension_order, 
    height=4, 
    aspect=1.5
)
g.map_dataframe(
    sns.barplot, 
    x="Injection pressure (mmHg)", 
    y="Total_flow", 
    hue="CVP pressure",  # Use CVP pressure for hue
    palette='viridis',  # Color palette for differentiating CVP pressures
    dodge=True  # Ensure bars are dodged (i.e., side by side)
)

# Customize the FacetGrid
g.set_titles("{col_name} hypotension")
g.set_axis_labels("Injection pressure (mmHg)", "Average total nidal flow (mL/min)")
# g.set(ylim=(0, 100))
g.add_legend(title='CVP Pressure')


# Adjust the layout to ensure labels are visible
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.9, top=0.9)

plt.show()


