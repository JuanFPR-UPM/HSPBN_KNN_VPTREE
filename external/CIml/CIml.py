import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, PowerTransformer, FunctionTransformer, OrdinalEncoder)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn import clone
from xgboost import XGBRegressor, XGBClassifier
from sklearn import metrics
from .utils_variables import get_var_type


def brier_multi(targets, probs):
    return np.mean(np.sum((probs - targets)**2, axis=1))


def evaluate_performance(y_true, y_pred, metric):

    if metric == 'neg_mse':
        perf = - metrics.mean_squared_error(y_true, y_pred)

    elif metric == 'neg_brier':
        # this works only in this case
        categories = [[x for x in range(y_pred.shape[1])]]
        enc = OneHotEncoder(categories=categories, handle_unknown='ignore')
        t2 = enc.fit_transform(pd.DataFrame(y_true)).toarray()
        perf = - brier_multi(t2, y_pred)

    else:
        raise ValueError('not supported metric')

    return perf


def decide_split(Y_, classification):
    """Decides Cross-Validation strategy based on sample size."""

    if Y_.shape[0] <= 1000:
        if classification:
            kf_ = StratifiedKFold(n_splits=10,  shuffle=True)
            indices = list(kf_.split(Y_, Y_))
        else:
            kf_ = KFold(n_splits=10,  shuffle=True)
            indices = list(kf_.split(Y_))

        pred_methods = ['XGB', 'LR', 'SVM']

    else:
        if classification:
            kf_ = StratifiedKFold(n_splits=5,  shuffle=True)
            indices = list(kf_.split(Y_, Y_))
        else:
            kf_ = KFold(n_splits=5,  shuffle=True)
            indices = list(kf_.split(Y_))

        pred_methods = ['XGB', 'LR']

    return indices, pred_methods


def get_scale_configs(continuous_features, categorical_features,  scaling_methods_con, scaling_methods_cat):
    """ColumnTransformers for numerical and categorical combinations."""

    fs_transformations = []

    scaler_map = {
        'std': StandardScaler,
        'minmax': MinMaxScaler,
        'power': PowerTransformer,
        'none': FunctionTransformer
    }

    scaler_map_cat = {
        'onehot': OneHotEncoder,
        'ordinal': OrdinalEncoder
    }

    methods_con = scaling_methods_con if len(continuous_features)>0 else [None]
    methods_cat = scaling_methods_cat if len(categorical_features)>0 else [None]

    for sc_con in methods_con:
        for sc_cat in methods_cat:
            transformers = []

            if len(continuous_features)>0:
                con_scaler_class = scaler_map.get(sc_con, FunctionTransformer)
                transformers.append(('num', con_scaler_class(), continuous_features))

            if len(categorical_features)>0:
                cat_scaler_class = scaler_map_cat.get(sc_cat, FunctionTransformer)
                transformers.append(('cat', cat_scaler_class(), categorical_features))

            if transformers:
                ct = ColumnTransformer(
                    transformers=transformers,
                    remainder='passthrough',
                    verbose_feature_names_out=False
                )
                fs_transformations.append(ct)
            else:
                # if both feature lists are empty
                fs_transformations.append(FunctionTransformer())

    return fs_transformations


def get_pred_configs(task_type, pred_methods):
    """Instantiates configurations (model+hyperparameters)"""

    models = []
    for method in pred_methods:

        if method == 'XGB':
            if task_type == 'regression':
                for rate in [0.01, 0.05, 0.1, 0.2]:
                    for depth in [3,4,5]:
                        models.append(XGBRegressor(learning_rate=rate, max_depth=depth, n_jobs=1))
                models.append(XGBRegressor())
            else:
                for rate in [0.01, 0.05, 0.1, 0.2]:
                    for depth in [3,4,5]:
                        models.append(XGBClassifier(learning_rate=rate, max_depth=depth, n_jobs=1))
                models.append(XGBClassifier())

        elif method == 'LR':
            if task_type == 'regression':
                models.append(LinearRegression(n_jobs=1))

        elif method == 'SVM':
            for kernel in ['rbf', 'poly']:
                for gamma_ in [0.001, 0.01]:
                    if task_type == 'regression':
                        models.append(SVR(kernel=kernel, C=1, gamma=gamma_, epsilon=0.05, shrinking=False))
                    else:
                        models.append(SVC(kernel=kernel, C=1, gamma=gamma_, shrinking=False, probability=True))

    return models


def permute_X (X):
    """Permutes the input variable X"""
    idx_rows = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    X_permuted = X[idx_rows,:]
    return X_permuted


def fit_predict(pred_config, train_X, train_y, test_X, classification):
    """Fits on train data and predicts on test data."""
    model_ = clone(pred_config)
    model_.fit(train_X, train_y.values.ravel())
    if classification:
        y_pred = model_.predict_proba(test_X)
    else:
        y_pred = model_.predict(test_X).reshape(-1,1)

    return y_pred


def get_XZ_prediction(XZ, Y, cur_indices, classification, config):
    """ Fits the model with the given configuration on train data, predicts on test data"""
    (train_idx, test_idx) = cur_indices
    Ytrain = Y.iloc[train_idx, :]
    XZtrain = XZ.iloc[train_idx, :]
    XZtest = XZ.iloc[test_idx, :]
    y_pred = fit_predict(config, XZtrain, Ytrain, XZtest, classification)

    return y_pred


def get_XpZ_prediction(X, Z, Y, cur_indices, classification, config):
    " Fits the model with the given configuration on train data, predicts on test data"
    (train_idx, test_idx) = cur_indices

    # Y ~Xpermuted, Z
    Xptrain = permute_X(X.iloc[train_idx, :].to_numpy())
    Xptest = permute_X(X.iloc[test_idx, :].to_numpy())

    Xp_tr = pd.DataFrame(Xptrain, columns=X.columns, index=X.index[train_idx])
    Xp_te = pd.DataFrame(Xptest, columns=X.columns, index=X.index[test_idx])

    XpZtrain = pd.concat((Xp_tr, Z.iloc[train_idx, :]), axis=1)
    XpZtest = pd.concat((Xp_te, Z.iloc[test_idx, :]), axis=1)

    y_pred = fit_predict(config, XpZtrain, Y.iloc[train_idx, :], XpZtest, classification)

    return y_pred


def get_pvalue(null_stat, obs_stat):
    "Calculate pvalue, add 1 to avoid zeros"
    pvalue = (np.count_nonzero(null_stat <= obs_stat) +1) / (null_stat.shape[0] +1)
    return pvalue


def swap_predictions(y_true, pred_xz, pred_xpz,  metric_name):
    " Swapping heuristic"

    idx_bool = np.random.randint(0, high=2, size=pred_xz.shape[0], dtype=bool)
    idx = np.where(idx_bool)[0]

    swap_1 = pred_xz.copy()
    swap_1[idx,:] = pred_xpz[idx,:].copy()
    swap_2 = pred_xpz.copy()
    swap_2[idx,:] = pred_xz[idx,:].copy()

    perf_xz_swap = evaluate_performance(y_true, swap_1, metric_name)
    perf_xpz_swap = evaluate_performance(y_true, swap_2, metric_name)

    null_stat = perf_xpz_swap - perf_xz_swap

    return null_stat


def nullVector_perFold(pred_xz, pred_xpz, Ytest, metric_name, S):
    """Computes observed statistic and statistics under the null for a single fold."""

    perf_xz = evaluate_performance(Ytest, pred_xz, metric_name)
    perf_xpz = evaluate_performance(Ytest, pred_xpz, metric_name)

    # Observed statistic
    obs_stat = perf_xpz - perf_xz

    # Approximate the distribution of the statistic under H0: swapping technique
    null_stats = np.zeros(S)
    for i in range(S):
        null_stats[i]= swap_predictions(Ytest, pred_xz, pred_xpz, metric_name)

    return obs_stat, null_stats


def create_pipelines(XZ,  task_type, scaling_methods, scaling_methods_cat, pred_methods):

    XZ_types = np.array([get_var_type(XZ[col]) for col in XZ.columns])
    continuous_features = XZ.columns[XZ_types=='continuous']
    categorical_features = XZ.columns[XZ_types=='categorical']

    transformations = get_scale_configs(continuous_features, categorical_features, scaling_methods, scaling_methods_cat)
    models = get_pred_configs(task_type, pred_methods)
    pipelines = []
    simple_scalers = (StandardScaler, MinMaxScaler)
    none_scaler = (FunctionTransformer,)

    xgb_types = (XGBClassifier, XGBRegressor)
    svm_types = (SVR, SVC)
    for ct_ in transformations:
        for model in models:
            is_xgb = isinstance(model, xgb_types)
            is_svm = isinstance(model, svm_types)

            has_simple_scaler = any(
                isinstance(trans_tuple[1], simple_scalers)
                for trans_tuple in ct_.transformers
            )

            has_none_scaler = any(
                isinstance(trans_tuple[1], none_scaler)
                for trans_tuple in ct_.transformers
            )

            if is_xgb and has_simple_scaler:
                continue

            if is_svm and has_none_scaler:
                continue

            pipelines.append(Pipeline(steps=[('prep', ct_), ('pred', model)]))

    return pipelines


def CIml(X_np, Y_np, Z_np,  **params):
    """
        Conditional Independence Testing.

        Parameters
        ----------
        X_np (numpy): (n_samples, 1)
        Y_np (numpy): (n_samples, 1)
        Z_np (numpy): (n_samples, n_features_Z)
        **params (dictionary):
            Optional keyword arguments:
            - S (int): Number of swapping iterations for null distribution (default: 1000).
            - scaling (list of str): Scalers for numerical features (default: ['none', 'std', 'power']).
            - scaling_cat (list of str): Encoders for categorical features (default: ['onehot']).

        Returns
        -------
        pvalue (float)

        Notes:
        1. Categorical Features:
           - Must be integer-encoded (e.g., 0, 1, 2, ...).
           - Rare levels must appear in all cross-validation folds.
        2. No missing values
        """


    # params
    S = params.get('S', 1000)
    scaling_methods = params.get('scaling', ['none', 'std', 'power'])
    scaling_methods_cat = params.get('scaling_cat', ['onehot'])

    # Data Preparation
    X = pd.DataFrame(X_np, columns=['X'])
    Y = pd.DataFrame(Y_np, columns=['Y'])
    Z = pd.DataFrame(Z_np, columns=['Z'+str(i) for i in range(Z_np.shape[1])])
    XZ = pd.concat((X, Z), axis=1)

    target_is = get_var_type(Y)
    metric_name = 'neg_mse' if target_is == 'continuous' else 'neg_brier'
    task_type = 'regression' if target_is == 'continuous' else 'classification'
    classification = False if target_is == 'continuous' else True

    # CV split
    indices, pred_methods = decide_split(Y, classification)
    n_indices = len(indices)

    # Configurations
    pipelines = create_pipelines(XZ, task_type, scaling_methods, scaling_methods_cat, pred_methods)
    n_configs = len(pipelines)

    # Ouf-of-sample predictions for each configuration / pipeline
    predictions_XZ = [
        Parallel(n_jobs=-1)(
            delayed(get_XZ_prediction)(XZ, Y, indices[f], classification, config)
            for f in range(len(indices))
        )
        for config in pipelines
    ]

    # Predictive performance
    perf_cv = np.zeros((n_configs, n_indices))
    for c in range(n_configs):
        for k in range(n_indices):
            perf_cv[c, k] = evaluate_performance(Y.iloc[indices[k][1], :], predictions_XZ[c][k], metric_name)

    # Find best configuration for each fold
    c_bests=[]
    for k in range(len(indices)):
        rest_folds = [i for i in range(len(indices)) if i != k]
        c_best = np.argmax(np.mean(perf_cv[:, rest_folds], axis=1))
        c_bests.append(c_best)

    # Get predictions for Y ~Xpermuted, Z
    predictions_XpZ = Parallel(n_jobs=-1)(
            delayed(get_XpZ_prediction)(X, Z, Y, indices[f], classification, pipelines[c_bests[f]])
            for f in range(len(indices)))

    res = Parallel(n_jobs=-1)(
            delayed(nullVector_perFold)(
                predictions_XZ[c_bests[k]][k],
                predictions_XpZ[k],
                Y.iloc[indices[k][1], :],
                metric_name,
                S)
            for k in range(n_indices))

    obs_stats, null_stats = zip(*res)
    null_stats_np = np.zeros((S, n_indices))
    for k in range(n_indices):
        null_stats_np[:,k]=null_stats[k]

    # Average statistics
    mean_obs = np.mean(obs_stats)
    mean_null = np.mean(null_stats_np, axis=1)

    # P-value calculation
    pvalue = get_pvalue(mean_null, mean_obs)

    return pvalue



def CIml_sym(X,Y,Z, **params):
    """
        Symmetric Conditional Independence Test wrapper.
        Evaluates both directions (X ⫫ Y | Z and Y ⫫ X | Z) and combines p-values
        Parameters
        ----------
        X, Y, Z (numpy) : vectors / matrices

        Returns
        -------
        pmm (float): combined p-value
    """

    pval1 = CIml(X, Y, Z, **params)
    pval2 = CIml(Y, X, Z, **params)

    minpval = np.min((pval1, pval2))
    maxpval = np.max((pval1, pval2))
    pmm = np.min((2 * minpval, maxpval))

    return pmm


