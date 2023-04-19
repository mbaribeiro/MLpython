import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.005)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.001)

# Create the model
modelTemp = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(
        2,), activation="relu", use_bias=True, bias_initializer='zeros'),
    tf.keras.layers.Dense(64, activation="relu",
                          use_bias=True, bias_initializer='zeros'),
    tf.keras.layers.Dense(512, activation="relu",
                          use_bias=True, bias_initializer='zeros'),
    tf.keras.layers.Dense(4096, activation="relu",
                          use_bias=True, bias_initializer='zeros'),
    tf.keras.layers.Dense(40500, activation='linear')
])

modelVel = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(
        2,), activation='relu', use_bias=True, bias_initializer='zeros'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(40500, activation='linear')
])


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


# Compile the model
modelTemp.compile(loss='mean_squared_error', optimizer=optimizer1)
modelVel.compile(loss='mean_squared_error', optimizer=optimizer2)

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

# Separate input and output data
inputs = data[:, :3]

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

    # Train the modelTemp
    modelTemp.fit(inputs, outputs1, epochs=200, batch_size=32,
                  verbose=0, callbacks=[epoch_callback_temp])

    # Train the modelVel
    modelVel.fit(inputs, outputs2, epochs=50, batch_size=32,
                 verbose=0, callbacks=[epoch_callback_vel])

    # Test the model
    predicted_inputs = [[15, 1], [18, 5], [20, 3]]
    test_inputs = np.array(predicted_inputs)
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
    predicted_outputsTemp_filt = [
        predicted_outputsTemp[i][j] for j in nonzero_indices]

    m, b = np.polyfit(output_list_filt, predicted_outputsTemp_filt, 1)
    x = np.array(output_list_filt)
    ax1.scatter(output_list_filt, predicted_outputsTemp_filt)
    ax1.plot(x, m*x + b, color='red')

    result = r2_score(output_list, predicted_outputsTemp[i])
    ax1.text(
        0.1, 0.9, "r-squared = {:.3f}".format(result), transform=ax1.transAxes)

    print("R² (Temperature) = " + str(result))
    ax1.set_title('Temperature [K]')
    ax1.set_ylabel('Predicted [K]')
    ax1.set_xlabel('Real [K]')

    # calculate the mse
    mse = tf.keras.losses.mean_squared_error(
        output_list, predicted_outputsTemp[i])
    print("MSE (Temperature) = " + str(mse))

    # Velocity
    output_list = df.iloc[:, 4].values.tolist()

    nonzero_indices = [i for i in range(
        len(output_list)) if output_list[i] != 0]
    output_list_filt = [output_list[i] for i in nonzero_indices]
    predicted_outputsVel_filt = [
        predicted_outputsVel[i][j] for j in nonzero_indices]

    m, b = np.polyfit(output_list_filt, predicted_outputsVel_filt, 1)
    x = np.array(output_list_filt)
    ax2.scatter(output_list_filt, predicted_outputsVel_filt)
    ax2.plot(x, m*x + b, color='red')

    result = r2_score(output_list, predicted_outputsVel[i])
    ax2.text(
        0.1, 0.9, "r-squared = {:.3f}".format(result), transform=ax2.transAxes)

    print("R² (Velocity) = " + str(result))
    ax2.set_title('Velocity [m s^-1]')
    ax2.set_ylabel('Predicted [m s^-1]')
    ax2.set_xlabel('Real [m s^-1]')

    # calculate the mse
    mse = tf.keras.losses.mean_squared_error(
        output_list, predicted_outputsVel[i])
    print("MSE (Velocity) = " + str(mse))

    plt.show()
