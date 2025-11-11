#!/usr/bin/env python
# coding: utf-8


from ucimlrepo import fetch_ucirepo
from pmlb import fetch_data
import os
import pandas as pd
import numpy as np


try:
    os.mkdir('./datasets')
except:
    pass

try:
    os.mkdir('./datasets/real')
except:
    pass


datasets_uci = {
    "Iris": 53,
    "Rice (Cammeo and Osmancik)": 545,
    "Concrete Compressive Strength": 165,
    "Glass Identification": 42,
    "Connectionist Bench (Sonar, Mines vs. Rocks)": 151,
    "Blood Transfusion Service Center": 176,
    "Airfoil Self-Noise": 291,
    "User Knowledge Modeling": 257,
    "Abalone": 1,
    "Liver Disorders": 60,
    "Auto MPG":	9,
    "Real Estate Valuation":	477,
    "HCV data":	571,
    "Combined Cycle Power Plant": 294,
    "Statlog (Australian Credit Approval)": 143,

}

datasets_pmlb = [
    "519_vinnie",
    "522_pm10",
    "529_pollen",
    "547_no2",
    "560_bodyfat",
    "666_rmftsa_ladata",
    "690_visualizing_galaxy",
    "712_chscase_geyser1",
    "analcatdata_lawsuit",
    "banana",
    "bupa",
    "cars",
    "collins",
    "haberman",
    "ionosphere",
    "irish",
    "new_thyroid",
    "penguins",
    "prnn_crabs",
    "profb",
    "sonar",
    "vehicle",
    "vowel",
    "wine_recognition"]


def linear_dependent_features(dataset):
    to_delete = set()
    dataset = dataset.select_dtypes('float64')
    if not dataset.empty:
        rank = np.linalg.matrix_rank(dataset.cov())
        df_copy = dataset.copy()
        if rank < dataset.shape[1]:
            for c in dataset.columns:
                new_df = df_copy.drop(c, axis=1)
                new_rank = np.linalg.matrix_rank(new_df.cov())

                if rank <= new_rank:
                    to_delete.add(c)
                    df_copy = new_df

    return to_delete


raw_dataframes = dict()
print('Downloading UCI datasets...')
for ds_name, ds_id in datasets_uci.items():
    api_df = fetch_ucirepo(id=ds_id)
    df = pd.concat([api_df.data.features, api_df.data.targets], axis=1)
    raw_dataframes[ds_name] = df

print('Downloading PMLB datasets...')
for ds_name in datasets_pmlb:
    df = fetch_data(ds_name)
    raw_dataframes[ds_name] = df

print('Preprocessing datasets...')
for ds_name, df in raw_dataframes.items():

    df = df.dropna(axis=1, thresh=int(0.3*len(df)))
    df = df.dropna(axis=0)

    # remove constant columns
    index_constant = np.where(df.nunique() == 1)[0]
    constant_columns = [df.columns[i] for i in index_constant]
    df = df.drop(columns=constant_columns, axis=1)

    cat_data = df.select_dtypes('object').astype('category')
    for c in cat_data:
        df = df.assign(**{c: cat_data[c]})

    # induce as categoricals numeric columns with less than 10 values
    index_low_cats = np.where(df.select_dtypes('int64').nunique() < 10)[0]
    for integer_cat_column in index_low_cats:
        df[df.columns[integer_cat_column]] = df[df.columns[integer_cat_column]].astype(
            'str').astype('category')

    float_data = df.select_dtypes('number').astype('float64')
    for c in float_data:
        df = df.assign(**{c: float_data[c]})

    df.reset_index(drop=True, inplace=True)

    # remove highly correlated columns
    to_remove_features = linear_dependent_features(df)
    df = df.drop(columns=to_remove_features, axis=1)

    if len(df) < 150 or len(df.columns) < 2 or len(df.select_dtypes('category').columns) == len(df.columns):
        print('Invalid dataset: ' + ds_name)
        continue

    df.to_csv('./datasets/real/' +
              ds_name.replace(' ', '_') + '.csv', index=False)

    print(ds_name, df.size, len(df.select_dtypes('category').columns),
          len(df.select_dtypes('float64').columns))
