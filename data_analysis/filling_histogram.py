import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data = pd.read_csv(Path(__file__).parent / 'data.csv')

# Apply filters similar to the SQL conditions
filtered_data = data[
    (data["Injection location"] == 'DV3') &
    (data["Injection pressure (mmHg)"] == 30) &
    (data["Blood pressure hypotension"] == 'profound') &
    (data["CVP pressure"] == 'elevated')
]
print(filtered_data["Percent filled (%)"])

# Set the seaborn theme for aesthetics
sns.set_theme(style="whitegrid", font_scale=1.5)

# Plot the histogram
plt.figure(figsize=(10, 6))
ax = sns.histplot(filtered_data['Percent filled (%)'], kde=False, color='blue', binwidth=1)
plt.grid(True)

# Using plt.setp() to set multiple properties
plt.setp(ax, xlabel='Percent filled (%)', ylabel='Count', xlim=[0, 100])

# Show the plot
plt.show()


