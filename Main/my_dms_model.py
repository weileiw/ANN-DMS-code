'''
this function bulids a keras sequential model.
author: Wei-Lei Wang, 
Date: May 18, 2020
'''
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

################### print out some dots on screen while the model is training 
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')
################### print out some dots on screen while the model is training

################### keras model with manually seperated data
def myDmsModel(train_features, train_labels,
               internal_features, intest_labels,
               saved_model_path,seed_value):

    # Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
     # Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    #  Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    #  Set `tensorflow` pseudo-random generator at a fixed value
    tf.set_random_seed(seed_value)

    model = keras.Sequential()
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                           input_shape=[len(train_features.keys())]))
    model.add(BatchNormalization())
    model.add(Dropout(1/4))
    model.add(layers.Dense(128, activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(1/4))
    model.add(layers.Dense(1))
    
    optimizer = tf.keras.optimizers.RMSprop(lr = 0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  metrics=['mean_squared_error',  'mean_absolute_error'])

    EPOCHS = 200
    # Set callback functions to early stop training and save the best model so far
    callbacks = [PrintDot(),
                 keras.callbacks.EarlyStopping(monitor='val_loss',  patience=20,  mode='min'),
                 keras.callbacks.ModelCheckpoint(filepath=saved_model_path,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 mode='min')]
    
    history = model.fit(train_features,
                        train_labels,
                        epochs=EPOCHS,
                        batch_size=64,
                        validation_data=(internal_features,intest_labels),
                        verbose=0,
                        callbacks=callbacks)

    return history

