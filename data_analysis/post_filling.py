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
    ["Blood pressure hypotension", "Injection pressure (mmHg)"]
).agg(
    During_injection=pd.NamedAgg(column="Percent filled (%)", aggfunc='mean'),
    After_injection=pd.NamedAgg(column="Percent filled post-injection (%)", aggfunc='mean')
).reset_index()

# Define order for "Blood pressure hypotension"
hypotension_order = ['normal', 'minor', 'moderate', 'profound']
grouped_data['Blood pressure hypotension'] = pd.Categorical(
    grouped_data['Blood pressure hypotension'],
    categories=hypotension_order,
    ordered=True
)

# Melt the dataframe to combine the two types of filling percentages and relabel
melted_data = pd.melt(
    grouped_data,
    id_vars=["Blood pressure hypotension", "Injection pressure (mmHg)"],
    var_name="Fill_type",
    value_name="Filling Percentage"
)

# Relabeling the "Fill_type" values
melted_data['Fill_type'] = melted_data['Fill_type'].map({
    'During_injection': 'During injection',
    'After_injection': 'After injection'
})

# Plotting
sns.set_theme(style="whitegrid", font_scale=1.5)

# Create a FacetGrid, mapping data to bar plots
g = sns.FacetGrid(
    melted_data, 
    col="Blood pressure hypotension", 
    col_order=hypotension_order, 
    height=4, 
    aspect=0.5
)
g.map_dataframe(
    sns.barplot, 
    x="Injection pressure (mmHg)", 
    y="Filling Percentage", 
    hue="Fill_type", 
    palette=['blue', 'orange'],  # Colors for the two fill types
    dodge=True  # Ensure bars are dodged (i.e., side by side)
)

# Customize the FacetGrid
g.set_titles("{col_name} hypotension")
g.set_axis_labels("Injection pressure (mmHg)", "Average filling (%)")
g.set(ylim=(0, 100))
g.add_legend(title='Average Filling Percentage')

# Adjust the layout to ensure labels are visible
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.83, top=0.9)

plt.show()