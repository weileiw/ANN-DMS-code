''' 
this function transforms coordinate and time parameters 
using sine and cosine function. It also optionally adds 
mixed layer depth weighted light parameters.
author: Wei-Lei Wang
data: May 18, 2020
'''
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin

def addAttribs(FrameIn):
    # add appropriate attributes according to input Frame
    if ('Time' in FrameIn.columns) & ('Kd490' in FrameIn.columns):
        add_SRD = True
        add_Time = True
        ilat, ilon, iPAR, iMLD, iDOY, iTime, iKd = [list(FrameIn.columns).index(col)
                                                    for col in("latitude","longitude","PAR",
                                                               "MLD","DOY","Time","Kd490")]
    elif ('Time' not in FrameIn.columns) & ('Kd490' in FrameIn.columns):
        add_SRD = True
        add_Time = False
        ilat, ilon, iPAR, iMLD, iDOY, iKd = [list(FrameIn.columns).index(col)
                                             for col in("latitude","longitude","PAR",
                                                        "MLD","DOY","Kd490")]
    else:
        add_SRD = False
        add_Time = False
        ilat, ilon, iDOY = [list(FrameIn.columns).index(col)
                            for col in("latitude","longitude","DOY")]
                                             

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_MLD_normalized_PAR = True): # no *args or **kargs
            self.add_MLD_normalized_PAR = add_MLD_normalized_PAR
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            latlon1 = np.sin(X[:,ilat] * np.pi / 180)
            latlon2 = np.sin(X[:,ilon] * np.pi / 180) * np.cos(X[:,ilat] * np.pi / 180)
            latlon3 =-np.cos(X[:,ilon] * np.pi / 180) * np.cos(X[:,ilat] * np.pi / 180)
            DOY1 = np.cos(X[:,iDOY] * (2 * np.pi) / 366)  # Month_ix is day of year
            DOY2 = np.sin(X[:,iDOY] * (2 * np.pi) / 366)  # Month_ix is day of year
            if 'iTime' in locals():
                Time1 = np.cos(X[:,iTime] * (2 * np.pi))
                Time2 = np.sin(X[:,iTime] * (2 * np.pi))
                if self.add_MLD_normalized_PAR:
                    MLD_normalized_PAR = X[:, iPAR] / (X[:, iMLD] * X[:, iKd]) * (1 - np.exp(-X[:, iKd] * X[:, iMLD]))
                    return np.c_[X, MLD_normalized_PAR, latlon1, latlon2, latlon3, DOY1, DOY2, Time1, Time2]
                else:
                    return np.c_[X, latlon1, latlon2, latlon3, DOY1, DOY2, Time1, Time2]
            else:
                if self.add_MLD_normalized_PAR:
                    MLD_normalized_PAR = X[:, iPAR] / (X[:, iMLD] * X[:, iKd]) * (1 - np.exp(-X[:, iKd] * X[:, iMLD]))
                    return np.c_[X, MLD_normalized_PAR, latlon1, latlon2, latlon3, DOY1, DOY2]
                else:
                    return np.c_[X, latlon1, latlon2, latlon3, DOY1, DOY2]

                    
                    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(add_MLD_normalized_PAR=add_SRD))])
    
    FrameOut = num_pipeline.fit_transform(FrameIn)
    # Put the data into a DateFrame again to remove unuse data
    if add_SRD & add_Time:
        FrameOut = pd.DataFrame(FrameOut,
                                columns=list(FrameIn.columns)+['SRD',"latlon1","latlon2",
                                                               "latlon3","DOY1","DOY2","Time1","Time2"],
                                index=FrameIn.index)
    elif add_SRD & (not add_Time):
        FrameOut = pd.DataFrame(FrameOut,
                                columns=list(FrameIn.columns)+['SRD',"latlon1","latlon2",
                                                               "latlon3","DOY1","DOY2"],
                                index=FrameIn.index)
    else:
        FrameOut = pd.DataFrame(FrameOut,
                                columns=list(FrameIn.columns)+["latlon1","latlon2",
                                                               "latlon3","DOY1","DOY2"],
                                index=FrameIn.index)

    return FrameOut



