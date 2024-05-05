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
sns.set_theme()

# Plot the histogram
plt.figure(figsize=(10, 6))
ax = sns.histplot(data['Num vessels'], kde=False, color='blue', binwidth=100)
plt.grid(True)

# Using plt.setp() to set multiple properties
plt.setp(ax, xlabel='Number of vessels', ylabel='Count')

# Show the plot
plt.show()


