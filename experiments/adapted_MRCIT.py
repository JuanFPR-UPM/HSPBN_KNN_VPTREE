import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../external/AAAI2022_HCM/'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pybnesian import IndependenceTest
from RCIT import RCITIndepTest
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn_pandas import DataFrameMapper
import pandas as pd
import numpy as np
from astropy import stats


def normalize_biweight(x, eps=1e-10):
    median = np.median(x)
    scale = stats.biweight.biweight_scale(x)
    if np.std(x) < 1e+2 or np.isnan(scale) or scale < 1e-4:
        norm = (x-np.mean(x))/np.std(x)
    else:
        norm = (x - median) / (scale + eps)
    return norm


def data_preprocess(df):

    def _encoding(i):
        if df.iloc[:, i].dtype == 'O' or df.iloc[:, i].dtype.name == 'category':
            tempX = df.iloc[:, i].values.reshape(-1, 1)
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(tempX)
            out = enc.transform(tempX).toarray()
        else:
            out = df.iloc[:, i].values.reshape(-1, 1)
        return out
    p = df.shape[1]
    X_encode = [_encoding(i) for i in np.arange(p)]
    return X_encode


def data_processing(df, cat_index, normalize='biweight'):
    columns = df.columns
    if normalize == 'biweight':
        BiweightScaler = FunctionTransformer(normalize_biweight)
        standardize = [(col, None) if col in cat_index else
                       ([col], BiweightScaler) for col in columns]
        x_mapper = DataFrameMapper(standardize)
        df = x_mapper.fit_transform(df).astype('float32')
        df = pd.DataFrame(df, columns=columns)
    elif normalize == 'standard':
        standardize = [(col, None) if col in cat_index else
                       ([col], StandardScaler()) for col in columns]
        x_mapper = DataFrameMapper(standardize)
        df = x_mapper.fit_transform(df).astype('float32')
        df = pd.DataFrame(df, columns=columns)
    else:
        raise NotImplementedError(
            f"currently we only support 'biweight' and 'standard'.")
    # encode
    df[cat_index] = df[cat_index].astype(object)
    X_encode = data_preprocess(df)
    return df, X_encode


class MRCIT(IndependenceTest):

    def __init__(self, df, num_f=100, num_f2=10, normalize='biweight'):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.variables = df.columns.tolist()
        df.columns = [int(x) for x in df.columns]
        cat_index = np.where((df.dtypes == 'category') |
                             (df.dtypes == 'object'))[0]
        self.df, self.X_encode = data_processing(
            df, cat_index, normalize=normalize)
        self.test = RCITIndepTest(suffStat=self.X_encode,
                                  num_f=num_f, num_f2=num_f2)

    def num_variables(self):
        return len(self.variables)

    def variable_names(self):
        return self.variables

    def has_variables(self, vars):
        return set(vars).issubset(set(self.variables))

    def name(self, index):
        return self.variables[index]

    def pvalue(self, x, y, z):

        # assuming column names are encoded to integers
        z = [int(col) for col in z] if z is not None else []
        return self.test.fit(int(x), int(y), z)
