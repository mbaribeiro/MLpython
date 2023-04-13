import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Create the model
modelTemp = keras.Sequential([
    keras.layers.Dense(64, input_shape=(2,), activation='sigmoid', use_bias=True, bias_initializer='zeros'),
    keras.layers.Dense(1024, activation='sigmoid',use_bias=True, bias_initializer='zeros'),
    keras.layers.Dense(8192, activation='sigmoid',use_bias=True, bias_initializer='zeros'),
    keras.layers.Dense(40500, activation='linear')
])

modelVel = keras.Sequential([
    keras.layers.Dense(64, input_shape=(2,), activation='relu', use_bias=True, bias_initializer='zeros'),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(40500, activation='linear')
])


class PrintEpochCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}/{self.params['epochs']}")


# Compile the model
modelTemp.compile(loss='mean_squared_error')
modelVel.compile(loss='mean_squared_error')

epoch_callback = PrintEpochCallback()

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
    modelTemp.fit(inputs, outputs1, epochs=500, batch_size=2,
                  verbose=0, callbacks=[epoch_callback])

    # Train the modelVel
    modelVel.fit(inputs, outputs2, epochs=500, batch_size=2,
                 verbose=0, callbacks=[epoch_callback])

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
    mse = keras.losses.mean_squared_error(
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
    mse = keras.losses.mean_squared_error(output_list, predicted_outputsVel[i])
    print("MSE (Velocity) = " + str(mse))

    plt.show()
