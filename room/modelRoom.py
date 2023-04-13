# Read the model save 
import tensorflow as tf
from tensorflow import keras

modelTemp = keras.models.load_model('./room/models/modelTemp.h5')
modelVel = keras.models.load_model('./room/models/modelVel.h5')
