import numpy as np
from sklearn.preprocessing import OneHotEncoder
from astropy import stats


def normalize_biweight(x, eps=1e-10):
    median = np.median(x)
    scale = stats.biweight.biweight_scale(x)
    if np.std(x) < 1e+2 or np.isnan(scale) or scale < 1e-4:
        norm =  (x-np.mean(x))/np.std(x)
    else:
        norm = (x - median) / (scale + eps)
    return norm


def normalize(x):
    norm = lambda x: (x-np.mean(x))/np.std(x)
    return np.apply_along_axis(norm, 0, x)


def data_preprocess(df):

    def _encoding(i):
        if df.iloc[:,i].dtype == 'O' or df.iloc[:, i].dtype.name == 'category':
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

