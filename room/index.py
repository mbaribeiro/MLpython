import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from math import *


# Define the parameters of the optimizer
optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.005)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.001)

# Define the Callback to print the epoch


class PrintEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, ax):
        self.losses = []
        self.ax = ax
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.ax.plot(self.losses)
        self.ax.set_title(
            f"Epoch {epoch+1}/{self.params['epochs']}, loss: {logs['loss']:.4f}")
        plt.pause(0.01)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

epoch_callback_temp = PrintEpochCallback(axs[0])
epoch_callback_vel = PrintEpochCallback(axs[1])

# Load input data
with open('./room/inputs/inputs.csv', 'r') as file:
    # Skip the header
    file.readline()
    # Read data lines
    data = []
    for line in file:
        values = [float(x) for x in line.strip().split(',')]
        data.append(values)
    # Convert data to numpy array
    data = np.array(data)
    print("reading inputs.csv...")

# Separate Temperature and Velocity data
inputs = data[:, :3]

# Substract the temperature
inputs[:, 0] = inputs[:, 0] - 15

# Load input data
with open('./room/inputs/inputs.csv', 'r') as f:
    # Skip the header
    next(f)
    data1 = []
    data2 = []
    for line in f:
        # Read input values
        inTemp, inVel = map(str, line.strip().split(','))

        # Load output data
        output_file = f'./room/outputs/T{inTemp}V{inVel}.csv'
        print("reading " + output_file + "...")
        df = pd.read_csv(output_file, delimiter='\t')
        coordenates = df.iloc[:, 0:3].values.tolist()
        data_line1 = df.iloc[:, 3].astype(str).tolist()
        data_line2 = df.iloc[:, 4].astype(str).tolist()
        data1 = data1 + [data_line1]
        data2 = data2 + [data_line2]

    # Separate input and output data
    outputs1 = data1
    outputs2 = data2

    outputs1 = np.array(outputs1, dtype=float)
    outputs2 = np.array(outputs2, dtype=float)

    # Remove the corresponding columns in coordinates that have all zeros in both outputs
    coordenates = np.array(coordenates)
    coordenates = coordenates[~np.all(outputs1 == 0, axis=0)]

    # Remove the columns with all zeros in both outputs
    outputs1 = outputs1[:, ~np.all(outputs1 == 0, axis=0)]
    outputs2 = outputs2[:, ~np.all(outputs2 == 0, axis=0)]

    # Create the model
modelTemp = tf.keras.Sequential([
    tf.keras.layers.Dense(
        round(len(outputs1[0])/exp(4*len(inputs[0]))), input_shape=(len(inputs[0]),)),
    tf.keras.layers.Dense(
        round(len(outputs1[0])/exp(3*len(inputs[0]))), activation="relu"),
    tf.keras.layers.Dense(
        round(len(outputs1[0])/exp(2*len(inputs[0]))), activation="relu"),
    tf.keras.layers.Dense(
        round(len(outputs1[0])/exp(1*len(inputs[0]))), activation="relu"),
    tf.keras.layers.Dense(len(outputs1[0]), activation='linear')
])

modelVel = tf.keras.Sequential([
    tf.keras.layers.Dense(round(len(
        outputs2[0])/exp(2*len(inputs[0]))), input_shape=(len(inputs[0]),), activation='relu'),
    tf.keras.layers.Dense(
        round(len(outputs2[0])/exp(1*len(inputs[0]))), activation='relu'),
    tf.keras.layers.Dense(len(outputs2[0]), activation='linear')
])

# Compile the model
modelTemp.compile(loss='mean_squared_error', optimizer=optimizer1)
modelVel.compile(loss='mean_squared_error', optimizer=optimizer2)

# Train the modelTemp
modelTemp.fit(inputs, outputs1, epochs=1500, batch_size=32,
              verbose=0, callbacks=[epoch_callback_temp])

# Train the modelVel
modelVel.fit(inputs, outputs2, epochs=300, batch_size=32,
             verbose=0, callbacks=[epoch_callback_vel])

# Test the model
predicted_inputs = [[15, 1], [18, 5], [20, 3]]
test_inputs = np.array(predicted_inputs)
test_inputs[:, 0] = test_inputs[:, 0] - 15
predicted_outputsTemp = modelTemp.predict(test_inputs)
predicted_outputsVel = modelVel.predict(test_inputs)

# Save predicted outputs
for i in range(len(predicted_inputs)):
    inTemp, inVel = predicted_inputs[i]
    with open(f'./room/predicteds/T{inTemp}V{inVel}.csv', 'w') as out_file:
        out_file.write('i,j,k,temp,vel,count\n')
        predicteds = '\n'.join([f"{a},{b},{c}" for a, b, c in zip(
            coordenates, predicted_outputsTemp[i], predicted_outputsVel[i])]).replace('[', '').replace(']', '')
        out_file.write(predicteds)

        output_file = f'./room/trainingOutputs/T{inTemp}V{inVel}.csv'
        df = pd.read_csv(output_file, delimiter='\t')

        # remove the lines in the output file that have all zeros in both outputs
        df = df[~np.all(df.iloc[:, 3:5] == 0, axis=1)]

        # Plot the predicted outputs and the real outputs with scatter in 3d with temperature and velocity
        fig1 = plt.figure()

        # Criate a boolean mask to remove the points with temperature less than a number
        mask = np.where(coordenates[:, 2] < 15)

        # Define the minimum and maximum values for the colorbar
        minTemp = min(min(predicted_outputsTemp[i]), min(df.iloc[:, 3]))
        maxTemp = max(max(predicted_outputsTemp[i]), max(df.iloc[:, 3]))
        minVel = min(min(predicted_outputsVel[i]), min(df.iloc[:, 4]))
        maxVel = max(max(predicted_outputsVel[i]), max(df.iloc[:, 4]))

        # First grath (Predicted Temp)
        ax1 = fig1.add_subplot(221, projection='3d')
        surf = ax1.scatter(coordenates[:, 0][mask], coordenates[:, 2][mask],
                           coordenates[:, 1][mask], c=predicted_outputsTemp[i][mask], cmap='jet')
        ax1.set_xlabel('X Label')
        ax1.set_ylabel('Y Label')
        ax1.set_zlabel('Z Label')
        ax1.set_title('Predicted Temp')

        # Second grath (Real Temp)
        ax2 = fig1.add_subplot(222, projection='3d')
        c_array = np.array(df.iloc[:, 3])
        surf = ax2.scatter(coordenates[:, 0][mask], coordenates[:, 2]
                           [mask], coordenates[:, 1][mask], c=c_array[mask], cmap='jet')
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        plt.colorbar(surf, ax=[ax1, ax2])
        ax2.set_title('Real Temp')

        # Third grath (Predicted Vel)
        ax3 = fig1.add_subplot(223, projection='3d')
        surf = ax3.scatter(coordenates[:, 0][mask], coordenates[:, 2][mask],
                           coordenates[:, 1][mask], c=predicted_outputsVel[i][mask], cmap='jet')
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Y Label')
        ax3.set_zlabel('Z Label')
        ax3.set_title('Predicted Vel')

        # Fourth grath (Real Vel)
        ax4 = fig1.add_subplot(224, projection='3d')
        c_array = np.array(df.iloc[:, 4])
        surf = ax4.scatter(coordenates[:, 0][mask], coordenates[:, 2]
                           [mask], coordenates[:, 1][mask], c=c_array[mask], cmap='jet')
        ax4.set_xlabel('X Label')
        ax4.set_ylabel('Y Label')
        ax4.set_zlabel('Z Label')
        plt.colorbar(surf, ax=[ax3, ax4])
        ax4.set_title('Real Vel')

        # Plot the error between the predicted outputs and the real outputs
        fig2 = plt.figure()

        ax5 = fig2.add_subplot(121, projection='3d')
        c_array = np.array(df.iloc[:, 3])
        c = predicted_outputsTemp[i][mask]
        diff1 = abs(c_array[mask] - c)
        surf = ax5.scatter(coordenates[:, 0][mask], coordenates[:, 2]
                           [mask], coordenates[:, 1][mask], c=diff1, cmap='jet')
        ax5.set_xlabel('X Label')
        ax5.set_ylabel('Y Label')
        ax5.set_zlabel('Z Label')
        plt.colorbar(surf, ax=ax5)
        ax5.set_title('Error Temp')

        ax6 = fig2.add_subplot(122,  projection='3d')
        c_array = np.array(df.iloc[:, 4])
        c = predicted_outputsVel[i][mask]
        diff2 = abs(c_array[mask] - c)
        surf = ax6.scatter(coordenates[:, 0][mask], coordenates[:, 2]
                           [mask], coordenates[:, 1][mask], c=diff2, cmap='jet')
        ax6.set_xlabel('X Label')
        ax6.set_ylabel('Y Label')
        ax6.set_zlabel('Z Label')
        plt.colorbar(surf, ax=ax6)
        ax6.set_title('Error Vel')

        # Plot only the 5% of the points with the highest error
        fig3 = plt.figure()

        ax7 = fig3.add_subplot(121, projection='3d')
        c_array = np.array(df.iloc[:, 3])
        c = predicted_outputsTemp[i][mask]
        diff1 = abs(c_array[mask] - c)
        diff1 = np.array(diff1)
        diff1 = diff1.flatten()
        diff1.sort()
        diff1 = diff1[::-1]
        diff1 = diff1[:int(len(diff1)*0.05)]
        mask2 = np.isin(abs(c_array[mask] - c), diff1)
        surf = ax7.scatter(coordenates[:, 0][mask][mask2], coordenates[:, 2]
                           [mask][mask2], coordenates[:, 1][mask][mask2], c=diff1, cmap='jet')
        ax7.set_xlabel('X Label')
        ax7.set_ylabel('Y Label')
        ax7.set_zlabel('Z Label')
        plt.colorbar(surf, ax=ax7)
        ax7.set_title('Error Temp')

        ax8 = fig3.add_subplot(122,  projection='3d')
        c_array = np.array(df.iloc[:, 4])
        c = predicted_outputsVel[i][mask]
        diff2 = abs(c_array[mask] - c)
        diff2 = np.array(diff2)
        diff2 = diff2.flatten()
        diff2.sort()
        diff2 = diff2[::-1]
        diff2 = diff2[:int(len(diff2)*0.05)]
        mask2 = np.isin(abs(c_array[mask] - c), diff2)
        surf = ax8.scatter(coordenates[:, 0][mask][mask2], coordenates[:, 2]
                           [mask][mask2], coordenates[:, 1][mask][mask2], c=diff2, cmap='jet')
        ax8.set_xlabel('X Label')
        ax8.set_ylabel('Y Label')
        ax8.set_zlabel('Z Label')
        plt.colorbar(surf, ax=ax8)
        ax8.set_title('Error Vel')

        plt.show()

# Plot the predicted outputs and the real outputs
for i in range(len(predicted_inputs)):
    inTemp, inVel = predicted_inputs[i]
    output_file = f'./room/trainingOutputs/T{inTemp}V{inVel}.csv'
    df = pd.read_csv(output_file, delimiter='\t')

    # Temperature
    fig, (ax1, ax2) = plt.subplots(1, 2)

    output_list = df.iloc[:, 3].values.tolist()

    nonzero_indices = [i for i in range(
        len(output_list)) if output_list[i] != 0]
    output_list_filt = [output_list[i] for i in nonzero_indices]

    m, b = np.polyfit(output_list_filt, predicted_outputsTemp[i], 1)
    x = np.array(output_list_filt)
    ax1.scatter(output_list_filt, predicted_outputsTemp[i])
    ax1.plot(x, m*x + b, color='red')

    result = r2_score(output_list_filt, predicted_outputsTemp[i])
    ax1.text(
        0.1, 0.9, "r-squared = {:.3f}".format(result), transform=ax1.transAxes)

    print("R² (Temperature) = " + str(result))
    ax1.set_title('Temperature [K]')
    ax1.set_ylabel('Predicted [K]')
    ax1.set_xlabel('Real [K]')

    # calculate the mse
    mse = tf.keras.losses.mean_squared_error(
        output_list_filt, predicted_outputsTemp[i])
    print("MSE (Temperature) = " + str(mse))

    # Velocity
    output_list = df.iloc[:, 4].values.tolist()

    nonzero_indices = [i for i in range(
        len(output_list)) if output_list[i] != 0]
    output_list_filt = [output_list[i] for i in nonzero_indices]

    m, b = np.polyfit(output_list_filt, predicted_outputsVel[i], 1)
    x = np.array(output_list_filt)
    ax2.scatter(output_list_filt, predicted_outputsVel[i])
    ax2.plot(x, m*x + b, color='red')

    result = r2_score(output_list_filt, predicted_outputsVel[i])
    ax2.text(
        0.1, 0.9, "r-squared = {:.3f}".format(result), transform=ax2.transAxes)

    print("R² (Velocity) = " + str(result))
    ax2.set_title('Velocity [m s^-1]')
    ax2.set_ylabel('Predicted [m s^-1]')
    ax2.set_xlabel('Real [m s^-1]')

    # calculate the mse
    mse = tf.keras.losses.mean_squared_error(
        output_list_filt, predicted_outputsVel[i])
    print("MSE (Velocity) = " + str(mse))

    plt.show()

# Save the models
modelTemp.save('room/models/modelTemp.h5')
modelVel.save('room/models/modelVel.h5')
