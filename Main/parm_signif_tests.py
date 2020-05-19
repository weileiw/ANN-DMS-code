'''
This functin is to test paramter combinations, so that the models 
can be ranked according to how well they make predictions.
A total of eight parameters are to be tested. The fucntion "combinations"
is used to generate every combination of the eight parameters. In total,
there are 255 test.

Change numCom parameter to set how many parameters in a combination.
e.g. when numCom = 3, the combination function will generate all combinations
with 3 parameters.

The results are used to produce Fig. 2a in the paper.

author: Wei-Lei Wang
date: May 18,2020
'''

import os
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
MODEL_PATH = os.path.join(PROJECT_ROOT_DIR,"Output")

if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)

# path and filename of saved model
saved_model_path = MODEL_PATH + "/Parm_tests_combs4-2.h5"
scaler_filename1 = MODEL_PATH + "/Parm_tests_scaler_combs4-2.save"
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
########################        get data     ########################


########################   keep only useful columne  ################
DMS_raw = raw_data_DMS.filter(['latitude', 'longitude','swDMS','PAR','MLD','SSS','SST',
                               'SiO','PO4','NO3','DOY','Time','Chl_sat','Kd490'])
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
DMS_clean = DMS_clean[(DMS_clean["Chl_sat"]>=0.01) & (DMS_clean["Chl_sat"]<20)] # 0.01

# drop off nans
DMS_clean = DMS_clean.dropna()
DMS_clean = DMS_clean.reset_index(drop=True)
#######################         clean up         #############################


# call add addAttibus function to add time, location, MLD weighted PAR features
DMS_extra_attribs = addAttribs(DMS_clean)


#######################         log transform         #############################
DMS_log = DMS_extra_attribs.copy()
DMS_log["SSS"] = np.log(DMS_extra_attribs.SSS)
DMS_log["SST"] = np.log(DMS_extra_attribs.SST)

# DMS_log["PAR"] = np.log(DMS_clean.PAR) #no logtransform has better distribution
DMS_log["MLD"] = np.log(DMS_extra_attribs.MLD)
DMS_log["SiO"] = np.log(DMS_extra_attribs.SiO)
DMS_log["PO4"] = np.log(DMS_extra_attribs.PO4)
DMS_log["NO3"] = np.log(DMS_extra_attribs.NO3)
DMS_log["SRD"] = np.log(DMS_extra_attribs.SRD)
DMS_log["swDMS"] = np.log(DMS_extra_attribs.swDMS)
DMS_log["Chl_sat"] = np.log(DMS_extra_attribs.Chl_sat)

DMS_log = DMS_log[np.isfinite(DMS_log).all(1)]
DMS_log = DMS_log.reset_index(drop=True)
#######################         log transform         #############################


#######################   get index for external test data   #######################
DMS_log_extra_attribs = DMS_log.copy()
valindex = DMS_log_extra_attribs[
    ((DMS_log_extra_attribs['latitude'] >= 69) & (DMS_log_extra_attribs['latitude'] <= 70)) |
    ((DMS_log_extra_attribs['latitude'] >= 59) & (DMS_log_extra_attribs['latitude'] <= 60)) |
    ((DMS_log_extra_attribs['latitude'] >= 49) & (DMS_log_extra_attribs['latitude'] <= 50)) |
    ((DMS_log_extra_attribs['latitude'] >= 39) & (DMS_log_extra_attribs['latitude'] <= 40)) |
    ((DMS_log_extra_attribs['latitude'] >= 29) & (DMS_log_extra_attribs['latitude'] <= 30)) |
    ((DMS_log_extra_attribs['latitude'] >= 19) & (DMS_log_extra_attribs['latitude'] <= 20)) |
    ((DMS_log_extra_attribs['latitude'] >= 9) & (DMS_log_extra_attribs['latitude'] <= 10)) |
    ((DMS_log_extra_attribs['latitude'] <= 1) & (DMS_log_extra_attribs['latitude'] >= -0)) |
    ((DMS_log_extra_attribs['latitude'] <= -9) & (DMS_log_extra_attribs['latitude'] >= -10)) |
    ((DMS_log_extra_attribs['latitude'] <= -19) & (DMS_log_extra_attribs['latitude'] >= -20)) |
    ((DMS_log_extra_attribs['latitude'] <= -29) & (DMS_log_extra_attribs['latitude'] >= -30)) |
    ((DMS_log_extra_attribs['latitude'] <= -39) & (DMS_log_extra_attribs['latitude'] >= -40)) |
    ((DMS_log_extra_attribs['latitude'] <= -49) & (DMS_log_extra_attribs['latitude'] >= -50)) |
    ((DMS_log_extra_attribs['latitude'] <= -59) & (DMS_log_extra_attribs['latitude'] >= -60)) |
    ((DMS_log_extra_attribs['latitude'] <= -69) & (DMS_log_extra_attribs['latitude'] >= -70))].index
# external testing data
external_test = DMS_log_extra_attribs.iloc[valindex].reset_index(drop=True)
#######################   get index for external test data   #######################


#######################   get index for internal test data   #######################
train_set = DMS_log_extra_attribs.drop(valindex, axis=0).reset_index(drop=True)
testindex = train_set[
    ((train_set['latitude'] >= 64) & (train_set['latitude'] <= 65)) |
    ((train_set['latitude'] >= 54) & (train_set['latitude'] <= 55)) |
    ((train_set['latitude'] >= 44) & (train_set['latitude'] <= 45)) |
    ((train_set['latitude'] >= 34) & (train_set['latitude'] <= 35)) |
    ((train_set['latitude'] >= 24) & (train_set['latitude'] <= 25)) |
    ((train_set['latitude'] >= 14) & (train_set['latitude'] <= 15)) |
    ((train_set['latitude'] >= 4) & (train_set['latitude'] <= 5)) |
    ((train_set['latitude'] <= -4) & (train_set['latitude'] >= -5)) |
    ((train_set['latitude'] <= -14) & (train_set['latitude'] >= -15)) |
    ((train_set['latitude'] <= -24) & (train_set['latitude'] >= -25)) |
    ((train_set['latitude'] <= -34) & (train_set['latitude'] >= -35)) |
    ((train_set['latitude'] <= -44) & (train_set['latitude'] >= -45)) |
    ((train_set['latitude'] <= -56) & (train_set['latitude'] >= -55)) |
    ((train_set['latitude'] <= -64) & (train_set['latitude'] >= -65))].index

# internal testing data
internal_test = train_set.iloc[testindex].reset_index(drop=True)
#######################   get index for internal test data   #######################


#  data for training
DMS_train_set = train_set.drop(testindex, axis=0).reset_index(drop=True)


####################### check the number of data points for each sets ##############
print("All data points : ", len(DMS_train_set) + len(internal_test) + len(external_test))
print("training data points : ", len(DMS_train_set))
print("internal testing data points : ", len(internal_test))
print("external validation data points: ", len(external_test))
####################### check the number of data points for each sets ###############


#######################    iteratively test parameter combinations     ###############
# get rid of parameters that are not used in the model
rm_cols1 = ["DOY", "latitude","longitude", "Time", "Kd490", "Time1", "Time2","SRD"]
# a list of parameters to be tested.
val_col = ["MLD","SST","SSS","PAR","SiO","PO4","NO3","Chl_sat"]
ErrorData = pd.DataFrame() # DataFrame for storing errors
numCom = 3 # number of environmental parameters in each combination
ErrorFile = '/ErrorCombs'+str(numCom)+'.csv'
# search all possible combinations.
combs = combinations(val_col, (len(val_col)-numCom))
itr = 1
for rm_cols2 in combs:
    print(itr)
    itr = itr + 1
    # a list of parameters that not used 
    rm_cols =  rm_cols1 + list(rm_cols2)
    train_features = DMS_train_set.copy()
    external_features = external_test.copy()
    internal_features = internal_test.copy()

    [train_features.pop(x) for x in rm_cols]
    [external_features.pop(x) for x in rm_cols]
    [internal_features.pop(x) for x in rm_cols]

    ################# get target feature
    train_labels = train_features.pop('swDMS')
    extest_labels  = external_features.pop('swDMS')
    intest_labels  = internal_features.pop('swDMS')
    for col in external_features.columns:
        print(col)

    # normalize only environmental parameters
    if 'Time1' in train_features.columns:
        nidx = 7
    else:
        nidx = 5

    # normalize only environmental parameters
    if len(train_features.columns)-nidx > 0:
        train_norm = train_features[train_features.columns[0:len(external_features.columns)-nidx]]
        extest_norm = external_features[external_features.columns[0:len(external_features.columns)-nidx]]
        intest_norm = internal_features[internal_features.columns[0:len(internal_features.columns)-nidx]]

        #  get scaler and save it;
        min_max_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_norm)
        joblib.dump(min_max_scaler, scaler_filename1)
        # normalize training data and then put it back
        x_train_norm = min_max_scaler.transform(train_norm)
        training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns)
        train_features.update(training_norm_col)
        # print(train_features.describe().T)
        
        x_extest_norm = min_max_scaler.transform(extest_norm)
        extest_norm_col = pd.DataFrame(x_extest_norm, index=extest_norm.index, columns=extest_norm.columns)
        external_features.update(extest_norm_col)
        # print(external_features.describe().T)
        
        x_intest_norm = min_max_scaler.transform(intest_norm)
        intest_norm_col = pd.DataFrame(x_intest_norm, index=intest_norm.index, columns=intest_norm.columns)
        internal_features.update(intest_norm_col)
        # print(internal_features.describe().T)

    
    ## train the model
    history = myDmsModel(train_features, train_labels, internal_features,
                         intest_labels,saved_model_path,seed_value)

    # get predictions
    loaded_model = load_model(saved_model_path)
    extest_predictions  = loaded_model.predict(external_features).flatten()
    intest_predictions  = loaded_model.predict(internal_features).flatten()
    train_predictions   = loaded_model.predict(train_features).flatten()

    # calculate statistics
    train_MAE = sklearn.metrics.mean_absolute_error(train_labels,train_predictions)
    xtest_MAE = sklearn.metrics.mean_absolute_error(extest_labels,extest_predictions)
    train_RMSE = math.sqrt(sklearn.metrics.mean_squared_error(train_labels,train_predictions))
    xtest_RMSE = math.sqrt(sklearn.metrics.mean_squared_error(extest_labels,extest_predictions))

    # reserve only environmental parameters and save them in the .csv file
    columnsNamesArr = external_features.columns.values
    trash = (list(columnsNamesArr))
    junk = ['latlon1', 'latlon2', 'latlon3', 'DOY1', 'DOY2']
    predictors = [item for item in trash if item not in junk]
    hh = '+'.join(predictors)

    # print out results on screen
    print()
    print('----------------------results for  '+hh+'   _____________________')
    print()
    print("Mean absolute error (MAE):      %f" % train_MAE)
    print("Root mean squared error (RMSE): %f" % train_RMSE)
    print("R square (R^2):                 %f" % sklearn.metrics.r2_score(train_labels,train_predictions))
    print()
    print("Mean absolute error (MAE):      %f" % xtest_MAE)
    print("Root mean squared error (RMSE): %f" % xtest_RMSE)
    print("R square (R^2):                 %f" % sklearn.metrics.r2_score(extest_labels,extest_predictions))
    print('----------------------results for  '+hh+'   _____________________')
    print()

    out_dict = {'Pred':hh,'TMAE': train_MAE, 'XMAE':xtest_MAE,'TRMSE':train_RMSE,'XRMSE':xtest_RMSE}
    df_tmp = pd.DataFrame(out_dict, index=[0])
    ErrorData = ErrorData.append(df_tmp, ignore_index = True) # ig

ErrorData.to_csv(MODEL_PATH+ErrorFile)
print("----------------------------------done!----------------------------------------------------")

