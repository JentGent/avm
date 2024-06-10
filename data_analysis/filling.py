import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data = pd.read_csv(Path(__file__).parent / 'data.csv')
print(data)
# Filter data to include only data where the Injection location starts with 'D'
filtered_data = data[
    data["Injection location"].notna() &
    data["Injection location"].str.startswith('D')
]

# Calculate the average percent filled for each group
grouped_data = filtered_data.groupby(
    ["Blood pressure hypotension", "Injection pressure (mmHg)", "CVP pressure"]
).agg(
    Average_filling=pd.NamedAgg(column="Percent filled (%)", aggfunc='mean')
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
    aspect=0.5,
    gridspec_kws={'wspace': 0}
)
g.map_dataframe(
    sns.barplot, 
    x="Injection pressure (mmHg)", 
    y="Average_filling", 
    hue="CVP pressure",  # Use CVP pressure for hue
    palette='viridis',  # Color palette for differentiating CVP pressures
    dodge=True  # Ensure bars are dodged (i.e., side by side)
)

# Customize the FacetGrid
# g.set_titles("{col_name} hypotension")
g.set_axis_labels("Injection pressure (mmHg)", "Average filling (%)")
g.set(ylim=(0, 95))
# g.add_legend(title='CVP Pressure')

for ax, title in zip(g.axes.flat, hypotension_order):
    ax.set_title("Normotension" if title == "normal" else (title.capitalize() + " hypotension"))

g.add_legend(title='Central venous pressure')
for t, l in zip(g._legend.texts, ["Normal", "Elevated"]):  # Updated to correct access to texts
    t.set_text(l)


# Adjust the layout to ensure labels are visible
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.9, top=0.9)

plt.show()

