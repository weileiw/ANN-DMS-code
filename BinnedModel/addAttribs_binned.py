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
    '''add time and location signature, impute data, and add MLD depth normalized PAR '''
    ilat, ilon, iPAR, iMLD, iMON, iKd = [list(FrameIn.columns).index(col)
                                         for col in("latitude","longitude","PAR",
                                                    "MLD","MON","Kd490")]
    
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_MLD_normalized_PAR = True): # no *args or **kargs
            self.add_MLD_normalized_PAR = add_MLD_normalized_PAR
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            latlon1 = np.sin(X[:,ilat] * np.pi / 180)
            latlon2 = np.sin(X[:,ilon] * np.pi / 180) * np.cos(X[:,ilat] * np.pi / 180)
            latlon3 =-np.cos(X[:,ilon] * np.pi / 180) * np.cos(X[:,ilat] * np.pi / 180)
            MON1 = np.cos(X[:,iMON] * (2 * np.pi) / 12)  # Month_ix is day of year
            MON2 = np.sin(X[:,iMON] * (2 * np.pi) / 12)  # Month_ix is day of year
            if self.add_MLD_normalized_PAR:
                MLD_normalized_PAR = X[:, iPAR] / (X[:, iMLD] * X[:, iKd]) * (1 - np.exp(-X[:, iKd] * X[:, iMLD]))
                return np.c_[X, MLD_normalized_PAR, latlon1, latlon2, latlon3, MON1, MON2]
            else:
                return np.c_[X, latlon1, latlon2, latlon3, MON1, MON2]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(add_MLD_normalized_PAR=True))])
    
    FrameOut = num_pipeline.fit_transform(FrameIn)
    # Put the data into a DateFrame again to remove unuse data
    FrameOut = pd.DataFrame(FrameOut,
                            columns=list(FrameIn.columns)+['SRD',"latlon1","latlon2","latlon3","MON1","MON2"],
                            index=FrameIn.index)

    return FrameOut



