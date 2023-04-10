import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Scripts import mediaTempVel as mtv
from Scripts import data as dt

# Define the room dimensions
sala = [0, -1000, -200, 200, -200, 200]

# Define the number of cubes
n_cubes = 8000

# Define the path to the results folder
results_path = './tube/results/'

# Loop through all files in the results folder
for filename in os.listdir(results_path):
    if filename.endswith('.csv'):
        # Read data from file, skipping the first 6 rows
        filepath = os.path.join(results_path, filename)
        data = pd.read_csv(filepath, skiprows=6, header=None)

        # Convert data to a numpy array
        data = data.values

        # Change the value of 4th column to Celsius
        data[:, 3] = data[:, 3] #- 273.15

        # Convert the first three columns to mm
        data[:, 0:3] *= 1000

        # Plot the data
        plt.show()

        # Media Temp Velocity
        result = mtv.media_temp_vel(sala, n_cubes, data)

        # Create a list to store the values of i, j, k, temp, vel and count
        values = []

        # Loop through the keys of i
        for i in result.keys():
            # Loop through the keys of j
            for j in result[i].keys():
                # Loop through the keys of k
                for k in result[i][j].keys():
                    # Add the values to the list
                    values.append([i, j, k, result[i][j][k]['temp'],
                                  result[i][j][k]['vel'], result[i][j][k]['count']])

        # Create a DataFrame with the values
        df = pd.DataFrame(values, columns=['i', 'j', 'k', 'temp', 'vel', 'count'])

        # Convert the values of i, j and k to int, if necessary
        df['i'] = df['i'].astype(int)
        df['j'] = df['j'].astype(int)
        df['k'] = df['k'].astype(int)

        # Print the DataFrame
        print(df)

        # Save the DataFrame to a file
        output_filename = os.path.splitext(filename)[0] + '.csv'
        output_filepath = os.path.join('./tube/outputs/', output_filename)
        df.to_csv(output_filepath, sep='\t', index=False)

# Define the path to the results folder
results_path = './tube/trainingResults/'

# Loop through all files in the results folder
for filename in os.listdir(results_path):
    if filename.endswith('.csv'):
        # Read data from file, skipping the first 6 rows
        filepath = os.path.join(results_path, filename)
        data = pd.read_csv(filepath, skiprows=6, header=None)

        # Convert data to a numpy array
        data = data.values

        # Change the value of 4th column to Celsius
        data[:, 3] = data[:, 3] #- 273.15

        # Convert the first three columns to mm
        data[:, 0:3] *= 1000

        # Plot the data
        plt.show()

        # Media Temp Velocity
        result = mtv.media_temp_vel(sala, n_cubes, data)

        # Create a list to store the values of i, j, k, temp, vel and count
        values = []

        # Loop through the keys of i
        for i in result.keys():
            # Loop through the keys of j
            for j in result[i].keys():
                # Loop through the keys of k
                for k in result[i][j].keys():
                    # Add the values to the list
                    values.append([i, j, k, result[i][j][k]['temp'],
                                  result[i][j][k]['vel'], result[i][j][k]['count']])

        # Create a DataFrame with the values
        df = pd.DataFrame(values, columns=['i', 'j', 'k', 'temp', 'vel', 'count'])

        # Convert the values of i, j and k to int, if necessary
        df['i'] = df['i'].astype(int)
        df['j'] = df['j'].astype(int)
        df['k'] = df['k'].astype(int)

        # Print the DataFrame
        print(df)

        # Save the DataFrame to a file
        output_filename = os.path.splitext(filename)[0] + '.csv'
        output_filepath = os.path.join('./tube/traningOutputs/', output_filename)
        df.to_csv(output_filepath, sep='\t', index=False)
