
# ==============================================================================
# Pillai's Trace CI Test Implementation
#
# Source: Adapted from the pgmpy library (https://github.com/pgmpy/pgmpy)
# Copyright (c) Ankur Ankan and pgmpy contributors
# License: MIT License
# ==============================================================================


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA


def _get_predictions(X, Y, Z, data, **kwargs):
    """
    Function to get predictions using XGBoost for `ci_pillai`.
    """
    # Step 0: Check if XGboost is installed.
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as e:
        raise ImportError(
            f"{e}. xgboost is required for using pillai_trace test. Please install using: pip install xgboost"
        ) from None

    # Step 1: Check if any of the conditional variables are categorical
    if any(data.loc[:, Z].dtypes == "category"):
        enable_categorical = True
    else:
        enable_categorical = False

    # Step 2: Check variable type of X, choose estimator, and compute predictions.
    if data.loc[:, X].dtype == "category":
        clf_x = XGBClassifier(
            enable_categorical=enable_categorical,
            seed=kwargs.get("seed"),
            random_state=kwargs.get("seed"),
        )
        x, x_cat_index = pd.factorize(data.loc[:, X])
        clf_x.fit(data.loc[:, Z], x)
        pred_x = clf_x.predict_proba(data.loc[:, Z])
    else:
        clf_x = XGBRegressor(
            enable_categorical=enable_categorical,
            seed=kwargs.get("seed"),
            random_state=kwargs.get("seed"),
        )
        x = data.loc[:, X]
        x_cat_index = None
        clf_x.fit(data.loc[:, Z], x)
        pred_x = clf_x.predict(data.loc[:, Z])

    # Step 3: Check variable type of Y, choose estimator, and compute predictions.
    if data.loc[:, Y].dtype == "category":
        clf_y = XGBClassifier(
            enable_categorical=enable_categorical,
            seed=kwargs.get("seed"),
            random_state=kwargs.get("seed"),
        )
        y, y_cat_index = pd.factorize(data.loc[:, Y])
        clf_y.fit(data.loc[:, Z], y)
        pred_y = clf_y.predict_proba(data.loc[:, Z])
    else:
        clf_y = XGBRegressor(
            enable_categorical=enable_categorical,
            seed=kwargs.get("seed"),
            random_state=kwargs.get("seed"),
        )
        y = data.loc[:, Y]
        y_cat_index = None
        clf_y.fit(data.loc[:, Z], y)
        pred_y = clf_y.predict(data.loc[:, Z])

    # Step 4: Return the predictions.
    return (pred_x, pred_y, x_cat_index, y_cat_index)

def pillai_trace(X, Y, Z, data, boolean=True, **kwargs):
    """
    A mixed-data residualization based conditional independence test[1].

    Uses XGBoost estimator to compute LS residuals[2], and then does an
    association test (Pillai's Trace) on the residuals.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \u27c2 Y | Z

    Y: str
        The second variable for testing the independence condition X \u27c2 Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \u27c2 Y | Z

    data: pandas.DataFrame
        The dataset in which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the pearson correlation coefficient and p_value
            of the test.

    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True if p-value >= significance_level, else False. If
        boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value)

    References
    ----------
    [1] Ankan, Ankur, and Johannes Textor.
        "A simple unified approach to testing high-dimensional"
        "conditional independences for categorical and ordinal data."
        Proceedings of the AAAI Conference on Artificial Intelligence.
    [2] Li, C.; and Shepherd, B. E. 2010.
      Test of Association Between Two Ordinal Variables While Adjusting for Covariates.
      Journal of the American Statistical Association.
    [3] Muller, K. E. and Peterson B. L. (1984) Practical Methods for computing power
      in testing the multivariate general linear hypothesis.
      Computational Statistics & Data Analysis.
    """
    # Step 1: Test if the inputs are correct
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}"
        )

    # Step 1.1: If no conditional variables are specified, use a constant value.
    if len(Z) == 0:
        Z = ["cont_Z"]
        data = data.assign(cont_Z=np.ones(data.shape[0]))

    # Step 2: Get the predictions
    pred_x, pred_y, x_cat_index, y_cat_index = _get_predictions(X, Y, Z, data, **kwargs)

    # Step 3: Compute the residuals
    if data.loc[:, X].dtype == "category":
        x = pd.get_dummies(data.loc[:, X]).loc[
            :, x_cat_index.categories[x_cat_index.codes]
        ]
        # Drop last column to avoid multicollinearity
        res_x = (x - pred_x).iloc[:, :-1]
    else:
        res_x = data.loc[:, X] - pred_x

    if data.loc[:, Y].dtype == "category":
        y = pd.get_dummies(data.loc[:, Y]).loc[
            :, y_cat_index.categories[y_cat_index.codes]
        ]
        # Drop last column to avoid multicollinearity
        res_y = (y - pred_y).iloc[:, :-1]
    else:
        res_y = data.loc[:, Y] - pred_y

    # Step 4: Compute Pillai's trace.
    if isinstance(res_x, pd.Series):
        res_x = res_x.to_frame()
    if isinstance(res_y, pd.Series):
        res_y = res_y.to_frame()

    cca = CCA(scale=False, n_components=min(res_x.shape[1], res_y.shape[1]))
    res_x_c, res_y_c = cca.fit_transform(res_x, res_y)

    cancor = []
    for i in range(min(res_x.shape[1], res_y.shape[1])):
        cancor.append(np.corrcoef(res_x_c[:, [i]].T, res_y_c[:, [i]].T)[0, 1])

    coef = (np.array(cancor) ** 2).sum()

    # Step 5: Compute p-value using f-approximation [3].
    s = min(res_x.shape[1], res_y.shape[1])
    df1 = res_x.shape[1] * res_y.shape[1]
    df2 = s * (data.shape[0] - 1 + s - res_x.shape[1] - res_y.shape[1])
    f_stat = (coef / df1) * (df2 / (s - coef))
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    # Step 6: Return
    if boolean:
        if p_value >= kwargs["significance_level"]:
            return True
        else:
            return False
    else:
        return coef, p_value


