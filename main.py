import data as dt
import numpy as np

# Read data from file
data = dt.data("export.csv");

# Reorganize data
data = np.array(data)

# Plote a 3d data and utilize the 4° as color, using matplotlib and axes3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Change the value of 4° column to Celsius
data[:,3] = data[:,3] - 273.15

# Plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3], cmap=plt.jet())
ax.set_title('Temperature [K]')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
plt.colorbar(surf)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,4], cmap=plt.jet())
ax.set_title('Velocity [m/s]')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
plt.colorbar(surf)
plt.show()














