# Interpolate the data with scipy
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator

def interpolate(data, x, y, z):
    # put the available x,y,z data as a numpy array
    dataPoints = np.array([data[:,0], data[:,1], data[:,2]]).T
    dataValuesT = np.array(data[:,3])
    dataValuesV = np.array(data[:,4])
    
    # put the x,y,z data you want to interpolate as a numpy array
    dataRequest = np.array([[x, y, z]])
    
    # return the interpolated values
    return griddata(dataPoints, dataValuesT, dataRequest, method='linear'), griddata(dataPoints, dataValuesV, dataRequest, method='linear')
