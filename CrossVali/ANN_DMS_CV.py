''' 
This model is based on Tensorflow Keras framework
The purpose of this function is to test n-fold validation models.
As we stated in the paper that the DMS data is ordered according 
contributor, a cross-validation without shuffling leads to under-
or over-trained model. (shuffling can be turned on and off in KFold
function)

Turn shuffling on to test how shuffle can influence the results.
author: Wei-Lei Wang on Mar. 04, 2020. 
'''

import os
import time
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import sklearn.metrics, math

from tensorflow import keras
from keras import regularizers
from keras import backend as K
from sklearn.externals import joblib
from addAttribs import addAttribs
from my_dms_model import myDmsModel
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
warnings.filterwarnings(action='ignore',message='^internal gelsd')


# 1. Configure a new global `tensorflow` session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=32,
                              inter_op_parallelism_threads=32)
sess = tf.Session(graph=tf.get_default_graph(),
                  config=session_conf)
K.set_session(sess)

# ignore divide by zero warning.
np.seterr(divide='ignore',invalid='ignore')

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
        
########################    set up model path    #############################        
PROJECT_ROOT_DIR = ''
# define fucntion to get data
DMS_PATH = os.path.join("../datasets","DMS")
def fetch_DMS_data(DMS_path=DMS_PATH):
    if not os.path.isdir(DMS_path):
        os.makedirs(DMS_path)
        
        fetch_DMS_data()

def load_DMS_data(DMS_path=DMS_PATH):
    csv_path=os.path.join(DMS_path, "PMEL_NAMMES_May2020.csv")
    return pd.read_csv(csv_path)

###################
MODEL_PATH = os.path.join(PROJECT_ROOT_DIR,"CrossVali")
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)

# path and filename of saved model
model_Fname = MODEL_PATH + "/PMEL_NAMMES.h5"
scaler_Fname = MODEL_PATH + "/PMEL_NAMMES.save"
########################    set up model path    #############################

# get data
raw_data_DMS = load_DMS_data()
[raw_data_DMS.pop(x) for x in ["DMSPt","Chl"]] # get rid of in-situ Chl and DMSP
DMS_raw = raw_data_DMS.copy()
# remove last Unnamed row.
DMS_raw.drop(DMS_raw.columns[DMS_raw.columns.str.contains('unnamed',case=False)],
             axis=1, inplace=True)


#######################         clean up         #############################
DMS_clean = DMS_raw.reset_index(drop=True).copy()
# clean up data
DMS_clean = DMS_clean[DMS_clean["SSS"]>=30] 
DMS_clean = DMS_clean[DMS_clean["POC"]<=750] # 500
DMS_clean = DMS_clean[DMS_clean["PO4"]>=0.01] # 0.01
DMS_clean = DMS_clean[DMS_clean["NO3"]>=0.01] # 0.01
DMS_clean = DMS_clean[DMS_clean["SiO"]>=0.10] # 0.10
DMS_clean = DMS_clean[DMS_clean["swDMS"]<=100] # 0.01
DMS_clean = DMS_clean[DMS_clean["swDMS"]>=0.01] # 0.0
DMS_clean = DMS_clean[DMS_clean["Chl_sat"]<=40]
DMS_clean = DMS_clean[DMS_clean["Chl_sat"]>0.01]

# drop off nans
DMS_clean = DMS_clean.dropna()
DMS_clean = DMS_clean.reset_index(drop=True)
########################         clean up         #############################


# call add addAttibus function to add time, location, MLD weighted PAR features
DMS_extra_attribs = addAttribs(DMS_clean)

#######################        log transform     #############################

DMS_log = DMS_extra_attribs.copy()
DMS_log["PAR"] = DMS_extra_attribs.PAR
DMS_log["SSS"] = np.log(DMS_extra_attribs.SSS)
DMS_log["SRD"] = np.log(DMS_extra_attribs.SRD)
DMS_log["MLD"] = np.log(DMS_extra_attribs.MLD)
DMS_log["SiO"] = np.log(DMS_extra_attribs.SiO)
DMS_log["PO4"] = np.log(DMS_extra_attribs.PO4)
DMS_log["NO3"] = np.log(DMS_extra_attribs.NO3)
DMS_log["PIC"] = np.log(DMS_extra_attribs.PIC)
DMS_log["POC"] = np.log(DMS_extra_attribs.POC)
DMS_log["swDMS"] = np.log(DMS_extra_attribs.swDMS)
DMS_log["SST"] = np.log(DMS_extra_attribs.SST+273.15)
DMS_log["Chl_sat"] = np.log(DMS_extra_attribs.Chl_sat)

DMS_log = DMS_log[np.isfinite(DMS_log).all(1)]
DMS_log = DMS_log.reset_index(drop=True)
#######################         log transform         #############################


DMS_clean_extra_attribs = DMS_log.copy()
[DMS_clean_extra_attribs.pop(x) for x in ["DOY","latitude","longitude","Time","Kd490",
                                          "SRD","PIC","POC","Time1","Time2"]]
# data for training
DMS_train_set = DMS_clean_extra_attribs.copy()

print()
print("training data points : ", len(DMS_train_set))
print()

train_labels = DMS_train_set.pop('swDMS')

# normalize only environmental parameters
train_norm = DMS_train_set[DMS_train_set.columns]

# normalize training data and then put it back
min_max_scaler = MinMaxScaler(feature_range=(-1,1)).fit(train_norm)
joblib.dump(min_max_scaler, scaler_Fname)

x_train_norm = min_max_scaler.transform(train_norm)
training_norm_col = pd.DataFrame(x_train_norm,
                                 index=train_norm.index,
                                 columns=train_norm.columns)
DMS_train_set.update(training_norm_col)

for col in training_norm_col.columns:
    print(col)


###############################  set up callbacks ##############################
callbacks = [PrintDot(),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=30,
                                           mode='auto'),
             keras.callbacks.ModelCheckpoint(filepath=model_Fname,
                                             monitor='val_loss',
                                             save_best_only=True,
                                             mode='auto')]
###############################  set up callbacks #############################


#############################  K-fold validation  #############################
fold_no = 1

# define per-fold performance containers
rmse_train_per_fold = []
rmse_test_per_fold  = [] 

X = DMS_train_set.values
Y = train_labels.values

# here you may turn shuffle on and off to test how shuffle inflence your results.
for train_indx, test_indx in KFold(n_splits=10,shuffle=False).split(X):

    x_train,x_test=X[train_indx],X[test_indx]
    y_train,y_test=Y[train_indx],Y[test_indx]

    # Generate a print on screen
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    # Set callback functions to early stop training and save the best model so far

    # Fit data to model
    model = None #cleaning the NN
    model=myDmsModel(DMS_train_set)
    model.fit(x_train,
              y_train,
              verbose=0,
              epochs=200,
              batch_size=64,
              callbacks=callbacks,
              validation_split = 0.2)


    # Generate generalization metrics
    print()
    scores1 = model.evaluate(X[train_indx], Y[train_indx], verbose=0)
    print(f'Train score for fold {fold_no}: {model.metrics_names[1]} of {scores1[1]}; {model.metrics_names[2]} of {scores1[2]}')
    rmse_train_per_fold.append(scores1[1])

    scores2 = model.evaluate(x_test,y_test, verbose=0)
    print(f'test. score for fold {fold_no}: {model.metrics_names[1]} of {scores2[1]}; {model.metrics_names[2]} of {scores2[2]}')
    rmse_test_per_fold.append(scores2[1])
    print()
    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(rmse_test_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - RMSE: {rmse_test_per_fold[i]}')
    print('------------------------------------------------------------------------')

print('Average scores for all folds:')
print(f'> RMSE: {np.mean(rmse_test_per_fold)} (+- {np.std(rmse_test_per_fold)})')
print('------------------------------------------------------------------------')


