'''
Apply ANN model to NAAMES predictor fields to make predictions.
'''

import re
import os
import sys
import scipy
import keras
import pickle
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from numpy import ma
from numpy import mat
from mystat import rsquared
from addAttribs import addAttribs
from sklearn.externals import joblib
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, TransformerMixin
# from matplotlib.mlab import bivariate_normal

# define function for model evaluation
def evaluate(test_labels,test_pred):
    R2 = rsquared(test_labels,test_pred)
    print('Model Performance')
    print("R2 score is ",R2)
    return R2

DATA_PATH = os.path.join('datasets','DMS')
filename = DATA_PATH+'/NAMMES_May2020.csv'

DMS_para = pd.read_csv(filename)


########################   keep only useful columne  ################
DMS_raw = DMS_para.filter(['latitude', 'longitude','swDMS','PAR','MLD','SSS','SST',
                           'SiO','PO4','NO3','DOY','Time','Chl_sat','Kd490'])
DMS_raw.rename({'Chl_sat': 'Chl','SSS':'SAL'}, axis=1, inplace=True);
# convert degree C to absolute temp to avoid lossing data when logtransform.
DMS_raw.SST = DMS_raw.SST+273.15
# remove last Unnamed row.
DMS_raw.drop(DMS_raw.columns[DMS_raw.columns.str.contains('unnamed',case=False)],
        axis=1, inplace=True)
########################   keep only useful columne  ################

# clean up data
envi_clean = DMS_raw.copy()
envi_clean.loc[envi_clean.SAL <   30, 'SAL'] = np.nan
envi_clean.loc[envi_clean.MLD >  150, 'MLD'] = np.nan
envi_clean.loc[envi_clean.SST < -200, 'SST'] = np.nan
envi_clean.loc[envi_clean.Chl < 0.01, 'Chl'] = np.nan
envi_clean.loc[envi_clean.PO4 <= 0.01,'PO4'] = np.nan
envi_clean.loc[envi_clean.SiO <= 0.1, 'SiO'] =  np.nan

ikeep = np.isfinite(envi_clean).all(1)
envi_parm = envi_clean[ikeep].reset_index(drop=True)
DMS_insi = envi_parm.pop('swDMS').reset_index(drop=True)

# DMS_obs_mod = DMS_insi.to_numpy().flatten()

################### top 10 models named after their parameters  ###########
filenames = {'SAL_SST_SiO_Chl',
             'SAL_SST_PO4_Chl',
             'MLD_SAL_PO4',
             'PAR_MLD_SAL_SST_SiO_PO4',
             'PAR_MLD_SST_SiO_PO4',
             'PAR_MLD_SST_Chl',
             'PAR_MLD_SST_SiO',
             'PAR_MLD_SAL_SST',
             'MLD_SST_PO4',
             'MLD_SST'}

############################   loop through every model  ####################
DMS_pred = pd.DataFrame()
ji = 1
for fname in filenames:

    envi_keep = envi_parm.copy()

    scaler_filename  = './PMEL_NAMMES/' + fname + '.save'
    saved_model_path = './PMEL_NAMMES/' + fname + '.h5'
    loaded_model = load_model(saved_model_path)
    min_max_scaler = joblib.load(scaler_filename)
    warnings.filterwarnings(action='ignore',message='^internal gelsd')

    predictors = re.split('_',fname) + ['latitude','longitude','DOY']
    columnsNamesArr = envi_keep.columns.values
    AllFeatures = (list(columnsNamesArr))
    Junk = [item for item in AllFeatures if item not in predictors]
    [envi_keep.pop(x) for x in Junk]
    
    # call add addAttibus function to add time, location, MLD weighted PAR features
    envi_extra_attribs = addAttribs(envi_keep)
    rmCols = ['latitude','longitude','DOY']

    [envi_extra_attribs.pop(x) for x in rmCols]
        
    envi_log = envi_extra_attribs.copy()
    if 'PAR' in envi_extra_attribs.columns:
        envi_log["PAR"] = envi_extra_attribs.PAR
    if 'MLD' in envi_extra_attribs.columns:
        envi_log["MLD"] = np.log(envi_extra_attribs.MLD)
    if 'Chl' in envi_extra_attribs.columns:
        envi_log["Chl"] = np.log(envi_extra_attribs.Chl)
    if 'SAL' in envi_extra_attribs.columns:
        envi_log["SAL"] = np.log(envi_extra_attribs.SAL)
    if 'SST' in envi_extra_attribs.columns:
        envi_log["SST"] = np.log(envi_extra_attribs.SST)
    if 'PO4' in envi_extra_attribs.columns:
        envi_log["PO4"] = np.log(envi_extra_attribs.PO4)
    if 'SiO' in envi_extra_attribs.columns:
        envi_log["SiO"] = np.log(envi_extra_attribs.SiO)
    

    # normanlize data
    if 'Time1' in envi_log.columns:
        nidx = 7
    else:
        nidx = 5

    # normalize only environmental parameters
    if len(envi_log.columns)-nidx > 0:
        envi_norm = envi_log[envi_log.columns[0:len(envi_log.columns)-nidx]]
    print(fname)
    print()
    for col in envi_norm.columns:
        print(col)
        
    # normalize training data and then put it back
    x_envi_norm = min_max_scaler.transform(envi_norm)
    envi_norm_col = pd.DataFrame(x_envi_norm,
                                 index=envi_norm.index,
                                 columns=envi_norm.columns)

    envi_log.update(envi_norm_col)
    # print(envi_log.describe().T)
    # envi_clean.hist(bins=50, figsize=(20,15))
    # plt.show()

    DMS_tmp = loaded_model.predict(envi_log).flatten()
    evaluate(np.log(DMS_insi),DMS_tmp)

    TabName = 'model'+str(ji);
    DMS_pred[TabName] = np.exp(DMS_tmp)
    ji = ji+1

envi_parm['swDMS'] = DMS_insi
envi_parm['Predictions'] = DMS_pred.mean(axis=1)
envi_parm['STD'] = DMS_pred.std(axis=1)
envi_parm.to_csv('NAAMES_with_predictions.csv')

    
