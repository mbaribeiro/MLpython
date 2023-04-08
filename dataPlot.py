from Scripts import data as dt
import matplotlib.pyplot as plt

#Import the data
data = dt.data("export.csv")

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