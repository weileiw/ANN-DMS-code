'''
this function bulids a keras sequential model.
author: Wei-Lei Wang, 
Date: May 18, 2020
'''
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

################### keras model with manually seperated data
def myDmsModel(train_features):

    model = keras.Sequential()
    model.add(layers.Dense(128,
                           activation='relu',
                           kernel_regularizer=regularizers.l2(0.001),
                           input_shape=[len(train_features.keys())]))
    model.add(BatchNormalization())
    model.add(Dropout(1/4))
    model.add(layers.Dense(128, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(1/4))
    model.add(layers.Dense(1))
    
    optimizer = tf.keras.optimizers.RMSprop(lr = 0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  metrics=['mean_squared_error',  'mean_absolute_error'])

    return model
    

