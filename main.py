import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Scripts import mediaTempVel as mtv
from Scripts import data as dt

# Read data from file
data = dt.data("export.csv")

# Reorganize data
data = np.array(data)

# Change the value of 4° column to Celsius
data[:, 3] = data[:, 3] - 273.15

# Convert the three first columns to mm
data[:, 0:3] *= 1000

# Define the room dimensions
sala = [0, -1000, -200, 200, -200, 200]

# Define the number of cubes
n_cubes = 8000

# Plot the data
plt.show()

# Media Temp Velocity
result = mtv.media_temp_vel(sala, n_cubes, data)

# Creates a list to store the values of i, j, k, temp, vel and count
values = []

# Loop by the keys of i
for i in result.keys():
    # Loop by the keys of j
    for j in result[i].keys():
        # Loop by the keys of k
        for k in result[i][j].keys():
            # Add the values to the list
            values.append([i, j, k, result[i][j][k]['temp'],
                          result[i][j][k]['vel'], result[i][j][k]['count']])

# Criates a DataFrame with the values
df = pd.DataFrame(values, columns=['i', 'j', 'k', 'temp', 'vel', 'count'])

# Convert the values of i, j and k to int, if necessary
df['i'] = df['i'].astype(int)
df['j'] = df['j'].astype(int)
df['k'] = df['k'].astype(int)

# Print the DataFrame
print(df)

# Save the DataFrame to a file
df.to_csv('output.txt', sep='\t', index=False)
