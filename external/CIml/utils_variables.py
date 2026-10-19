import numpy as np
from sklearn.preprocessing import StandardScaler


def get_var_type(V, unique_val_thr = 10):

    '''
    Decides if the input variable is continuous or categorical
    V (numpy/pandas): the input variable
    unique_val_thr(int): optional, the maximum unique number of values to be considered categorical
    Returns
    var_type (str): categorical or continuous
    '''

    if len(np.unique(V)) < unique_val_thr:  # number of unique values < threshold
        var_type = 'categorical'
    else:
        var_type = 'continuous'
    return var_type



def preprocess_data(data, x_col='X', y_col='Y', z_cols=None):
    """
    Standardizes continuous variables for baseline CI tests.
    """
    if z_cols is None:
        z_cols = []

    prep_data = data.copy()
    continuous_vars = []

    if get_var_type(data[x_col].to_numpy().reshape(-1, 1)) == 'continuous':
        continuous_vars.append(x_col)
    if get_var_type(data[y_col].to_numpy().reshape(-1, 1)) == 'continuous':
        continuous_vars.append(y_col)
    for col in z_cols:
        if get_var_type(data[col].to_numpy().reshape(-1, 1)) == 'continuous':
            continuous_vars.append(col)

    if continuous_vars:
        scaler = StandardScaler()
        prep_data.loc[:, continuous_vars] = scaler.fit_transform(prep_data[continuous_vars])

    return prep_data

