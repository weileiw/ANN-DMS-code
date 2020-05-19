import re
import os
import sys
import scipy
import pickle
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from numpy import ma
from numpy import mat
from addAttribs import addAttribs
from scipy.io import loadmat, savemat
from geoplots import geoplot
from sklearn.externals import joblib
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, TransformerMixin
# from matplotlib.mlab import bivariate_normal

lon = np.arange(0.5, 360.5, 1)
lat = np.arange(-89.5, 90.5, 1)
Lon, Lat = np.meshgrid(lon, lat)
warnings.filterwarnings(action='ignore',message='^internal gelsd')

# load the model and saved scaler
fname = 'SAL_SST_PO4_Chl'
    
saved_model_path = "./PMEL_NAMMES/"+fname+".h5"
scaler_filename1 = "./PMEL_NAMMES/"+fname+".save"
min_max_scaler = joblib.load(scaler_filename1)
loaded_model = load_model(saved_model_path)

# output model path
fname_out = "./output_data/"+fname


data_path = '/glade/u/home/weileiw/DATA/'
# connect data path with file names
chl_file = os.path.join(data_path,"Chl_180x360x12.mat")
par_file = os.path.join(data_path,'PAR_180x360x12.mat');
mld_file = os.path.join(data_path,"MIMOC_MLD_180x360x12.mat")
sst_file = os.path.join(data_path,"surface_tempobs_180x360x12.mat")
sal_file = os.path.join(data_path,"surface_Sobs_180x360x12.mat")
po4_file = os.path.join(data_path,'surface_po4obs_180x360x12.mat')
sio_file = os.path.join(data_path,'surface_sio4obs_180x360x12.mat')

PAR_data = loadmat(par_file)
MLD_data = loadmat(mld_file)
SST_data = loadmat(sst_file)
SAL_data = loadmat(sal_file)
Chl_data = loadmat(chl_file)
po4_data = loadmat(po4_file)
sio_data = loadmat(sio_file)

for month in range(1,13):
    
    
    MLD = MLD_data['MLD'][:,:,month-1]
    SST = SST_data['SST'][:,:,month-1]
    SAL = SAL_data['SAL'][:,:,month-1]
    PO4 = po4_data['PO4'][:,:,month-1]
    SiO = sio_data['SiO'][:,:,month-1]
    Chl = Chl_data['Chl'][:,:,month-1]
    PAR = PAR_data['PAR'][:,:,month-1]
    
    envi_raw = pd.DataFrame({"latitude":Lat.flatten(),
                             "longitude":Lon.flatten(),
                             "PAR":PAR.flatten(),
                             "MLD":MLD.flatten(),
                             "SAL":SAL.flatten(),
                             "SST":SST.flatten(),
                             "SiO":SiO.flatten(),
                             "PO4":PO4.flatten(),
                             "Chl":Chl.flatten(),
                             "DOY":np.zeros(len(Lat.flatten()))+month*30-15})
    
    # print(envi_raw.describe().T)
    
    predictors = re.split('_',fname) + ['latitude','longitude','DOY']
    columnsNamesArr = envi_raw.columns.values
    AllFeatures = (list(columnsNamesArr))
    Junk = [item for item in AllFeatures if item not in predictors]
    [envi_raw.pop(x) for x in Junk]
    # print(envi_raw.describe().T)
    
    # clean up data
    envi_clean = envi_raw.copy()
    if 'SAL' in envi_clean.columns:
        envi_clean.loc[envi_clean.SAL < 30.0, 'SAL'] = np.nan
    if 'MLD' in envi_clean.columns:
        envi_clean.loc[envi_clean.MLD > 150, 'MLD'] = np.nan
    if 'SST' in envi_clean.columns:
        envi_clean.loc[envi_clean.SST < -20,  'SST'] = np.nan
    if 'Chl' in envi_clean.columns:
        envi_clean.loc[envi_clean.Chl < 0.01, 'Chl'] = np.nan
    if 'PO4' in envi_clean.columns:
        envi_clean.loc[envi_clean.PO4 <= 0.01, 'PO4'] = 0.01
    if 'SiO' in envi_clean.columns:
        envi_clean.loc[envi_clean.SiO <= 0.1, 'SiO'] = 0.1

    ikeep = np.isfinite(envi_clean).all(1)
    envi_keep = envi_clean[ikeep]
    # print(envi_keep.describe().T)

    # call add addAttibus function to add time, location, MLD weighted PAR features
    envi_extra_attribs = addAttribs(envi_keep)
    #if 'SAL' in fname:
    rmCols = ['latitude','longitude','DOY']
    #else:
     #   rmCols = ['SAL','latitude','longitude','DOY']

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
        envi_log["SST"] = np.log(envi_extra_attribs.SST+273.15)
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
    
    DMS_pred = loaded_model.predict(envi_log)
    
    DMS2d = Lon+np.nan
    DMS2d_tmp = DMS2d.flatten()
    DMS2d_tmp[ikeep] = np.exp(DMS_pred.flatten())
    DMS2d = DMS2d_tmp.reshape(180,360)
    
    if "DMS2d_all" not in locals():
        DMS2d_all = DMS2d
    else:
        DMS2d_all = np.dstack((DMS2d_all,DMS2d))
        
savemat(fname_out, mdict={'DMS': DMS2d_all}, appendmat=True,
        format='5', long_field_names=False,
        do_compression=False, oned_as='row')

del DMS2d_all
