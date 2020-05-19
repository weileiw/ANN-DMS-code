'''
Main function to fine tune the ANN model. The top 10 model parameterizations
were chosen according to parameterization significant test.
To tune the model, the "fname" should be changed mannually.
author: Wei-Lei Wang
data: May 18, 2020
'''
import os
import re
import time
import pickle
import warnings
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import sklearn.metrics, math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from addAttribs import addAttribs
from itertools import combinations
from my_dms_model import myDmsModel
from sklearn.externals import joblib
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split


#############   Change this name to adjust parameter combs ##########

fname = 'SAL_SST_SiO_Chl'

#############   Change this name to adjust parameter combs ##########


#######################    set up Envi.      #######################
warnings.filterwarnings(action='ignore',message='^internal gelsd')
# Seed value (can actually be different for each attribution step)
seed_value = 64

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=64,
                              inter_op_parallelism_threads=64)
sess = tf.Session(graph=tf.get_default_graph(),
                  config=session_conf)
K.set_session(sess)
# ignore divide by zero warning.
np.seterr(divide='ignore',invalid='ignore')
#######################    set up Envi.      #######################


#######################    set up model path  #######################
PROJECT_ROOT_DIR = ''
MODEL_PATH = os.path.join(PROJECT_ROOT_DIR,"PMEL_NAMMES")

if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)


# path and filename of saved model
saved_model_path = MODEL_PATH + "/"+fname+".h5"
scaler_filename1 = MODEL_PATH + "/"+fname+".save"
#######################    set up model path  #######################


########################        get data     ########################
DMS_PATH = os.path.join("../datasets","DMS")
def fetch_DMS_data(DMS_path=DMS_PATH):
    if not os.path.isdir(DMS_path):
        os.makedirs(DMS_path)

fetch_DMS_data()

def load_DMS_data(DMS_path=DMS_PATH):
    csv_path=os.path.join(DMS_path,"PMEL_NAMMES_May2020.csv")
    return pd.read_csv(csv_path)

raw_data_DMS = load_DMS_data()
raw_data_DMS.rename({'SSS':'SAL'}, axis = 1, inplace = True)    
# print(raw_data_DMS.describe().T)
########################        get data     ########################


########################   keep only useful columne  ################
DMS_raw = raw_data_DMS.filter(['latitude', 'longitude','swDMS','PAR','MLD','SAL','SST',
                               'SiO','PO4','NO3','DOY','Time','Chl_sat','Kd490'])
DMS_raw.rename({'Chl_sat': 'Chl'}, axis=1, inplace=True);
# convert degree C to absolute temp to avoid lossing data when logtransform.
DMS_raw.SST = DMS_raw.SST+273.15
# remove last Unnamed row.
DMS_raw.drop(DMS_raw.columns[DMS_raw.columns.str.contains('unnamed',case=False)],
        axis=1, inplace=True)
########################   keep only useful columne  ################


#######################         shuffle         #############################
DMS_raw = shuffle(DMS_raw, random_state = seed_value)
#######################         shuffle         #############################


#######################         clean up         #############################
# the loaded dataset has been clean for too low/high DMS and coastal DMS measurements(SAL<30)
DMS_clean = DMS_raw.reset_index(drop=True).copy()

DMS_clean = DMS_clean[DMS_clean["MLD"]<=150] # remove 479 data points
DMS_clean = DMS_clean[DMS_clean["PO4"]>=0.01] # 0.01
DMS_clean = DMS_clean[DMS_clean["NO3"]>=0.01] # 0.01
DMS_clean = DMS_clean[DMS_clean["SiO"]>=0.10] # 0.10
# 50 data points with Chla>20; 1 data points with Chla<=0.01
DMS_clean = DMS_clean[(DMS_clean["Chl"]>=0.01) & (DMS_clean["Chl"]<20)] # 0.01

# drop off nans
DMS_clean = DMS_clean.dropna()
DMS_clean = DMS_clean.reset_index(drop=True)
#######################         clean up         #############################

# print(DMS_clean.describe().T)
# call add addAttibus function to add time, location, MLD weighted PAR features
DMS_extra_attribs = addAttribs(DMS_clean)


#######################         log transform         #############################
DMS_log = DMS_extra_attribs.copy()
DMS_log["SAL"] = np.log(DMS_extra_attribs.SAL)
DMS_log["SST"] = np.log(DMS_extra_attribs.SST)

# DMS_log["PAR"] = np.log(DMS_clean.PAR) #no logtransform is better
DMS_log["MLD"] = np.log(DMS_extra_attribs.MLD)
DMS_log["SiO"] = np.log(DMS_extra_attribs.SiO)
DMS_log["PO4"] = np.log(DMS_extra_attribs.PO4)
DMS_log["NO3"] = np.log(DMS_extra_attribs.NO3)
DMS_log["SRD"] = np.log(DMS_extra_attribs.SRD)
DMS_log["swDMS"] = np.log(DMS_extra_attribs.swDMS)
DMS_log["Chl"] = np.log(DMS_extra_attribs.Chl)

DMS_log = DMS_log[np.isfinite(DMS_log).all(1)]
DMS_log = DMS_log.reset_index(drop=True)
#######################         log transform         #############################


#######################    get rid of redundant parameters ########################
# get rid of parameters that are not used in the model
all_cols = ["DOY", "latitude","longitude", "Time", "Kd490", "Time1",
            "Time2", "SRD","MLD","SST","SAL","PAR","SiO","PO4","NO3","Chl"]

used_cols = re.split('_',fname)
junk_cols = [item for item in all_cols if item not in used_cols]

train_set = DMS_log.copy()
[train_set.pop(x) for x in junk_cols]
################# get target feature
DMS = train_set.pop('swDMS')

train_features, internal_features, train_labels, intest_labels = train_test_split(train_set, DMS, test_size=0.20)

for col in internal_features.columns:
    print(col)
    
# normalize only environmental parameters
if 'Time1' in train_features.columns:
    nidx = 7
else:
    nidx = 5

# normalize only environmental parameters
if len(train_features.columns)-nidx > 0:
    train_norm = train_features[train_features.columns[0:len(internal_features.columns)-nidx]]
    intest_norm = internal_features[internal_features.columns[0:len(internal_features.columns)-nidx]]
    
    #  get scaler and save it;
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_norm)
    joblib.dump(min_max_scaler, scaler_filename1)
    # normalize training data and then put it back
    x_train_norm = min_max_scaler.transform(train_norm)
    training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns)
    train_features.update(training_norm_col)
    # print(train_features.describe().T)
    
    x_intest_norm = min_max_scaler.transform(intest_norm)
    intest_norm_col = pd.DataFrame(x_intest_norm, index=intest_norm.index, columns=intest_norm.columns)
    internal_features.update(intest_norm_col)
    # print(internal_features.describe().T)

    
    ## train the model
    history = myDmsModel(train_features, train_labels, internal_features,
                         intest_labels, saved_model_path, seed_value)

    # get predictions
    loaded_model = load_model(saved_model_path)
    intest_predictions  = loaded_model.predict(internal_features).flatten()
    train_predictions   = loaded_model.predict(train_features).flatten()

    # put data in dataframe and save it
    dat1 = {"train_labels": train_labels, "train_pred": train_predictions}
    train = pd.DataFrame(dat1)
    print(train.describe().T)
    dat2 = {"test_labels": intest_labels, "test_pred": intest_predictions}
    tst = pd.DataFrame(dat2)
    print(tst.describe().T)
    
    # plt.scatter(train_labels, train_predictions,  color="red", alpha=0.1);
    # plt.axis('equal')
    # plt.axis('square')
    # plt.xlim([train_labels.min(),train_labels.max()])
    # plt.ylim([train_labels.min(),train_labels.max()])
    # _ = plt.plot([-100, 100], [-100, 100])
    # plt.show()

    # calculate statistics
    train_MAE = sklearn.metrics.mean_absolute_error(train_labels,train_predictions)
    itest_MAE = sklearn.metrics.mean_absolute_error(intest_labels,intest_predictions)
    train_RMSE = math.sqrt(sklearn.metrics.mean_squared_error(train_labels,train_predictions))
    itest_RMSE = math.sqrt(sklearn.metrics.mean_squared_error(intest_labels,intest_predictions))

    print()
    print("Mean absolute error (MAE):      %f" % train_MAE)
    print("Root mean squared error (RMSE): %f" % train_RMSE)

    print()
    print("Mean absolute error (MAE):      %f" % itest_MAE)
    print("Root mean squared error (RMSE): %f" % itest_RMSE)
    print()


print("----------------------------------done!----------------------------------------------------")

