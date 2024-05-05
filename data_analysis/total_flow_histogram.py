import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data = pd.read_csv(Path(__file__).parent / 'data.csv')

# Apply filters similar to the SQL conditions
filtered_data = data[
    (data["Injection location"].isnull()) &
    (data["Blood pressure hypotension"] == 'normal') &
    (data["CVP pressure"] == 'normal')
]

# Set the seaborn theme for aesthetics
sns.set_theme(style="whitegrid", font_scale=1.5)

# Plot the histogram
plt.figure(figsize=(10, 6))
ax = sns.histplot(filtered_data['Feeder total flow (mL/min)'], kde=False, color='blue', binwidth=10)
plt.grid(True)

# Using plt.setp() to set multiple properties
plt.setp(ax, xlabel='Total nidal flow (mL/min)', ylabel='Count')

# Show the plot
plt.show()


