#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
import pickle
import pybnesian as pbn
from functools import partial
from adapted_MMCIT import MMCIT
from adapted_CGIT import CGIT
from adapted_DGCIT import DGCIT
from adapted_MRCIT import MRCIT
from sklearn.preprocessing import LabelEncoder


try:
    os.mkdir('./results/')
except:
    pass

try:
    os.mkdir('./results/real/')
except:
    pass

try:
    os.mkdir('./results/real/knncit')
except:
    pass

try:
    os.mkdir('./results/real/mrcit')
except:
    pass

try:
    os.mkdir('./results/real/mmcit')
except:
    pass

try:
    os.mkdir('./results/real/cgit')
except:
    pass

try:
    os.mkdir('./results/real/dgcit')
except:
    pass

try:
    os.mkdir('./results/real/hillclimbing')
except:
    pass


def preprocess_dataset(filename):
    df = pd.read_csv('./datasets/real/' + filename)

    cat_data = df.select_dtypes('object').astype('str').astype('category')
    for c in cat_data:
        df = df.assign(**{c: cat_data[c]})

    # induce as categoricals numeric columns with less than 10 values
    index_low_cats = np.where(df.select_dtypes('number').nunique() < 10)[0]
    for integer_cat_column in index_low_cats:
        df[df.columns[integer_cat_column]] = df[df.columns[integer_cat_column]].astype(
            'str').astype('category')

    float_data = df.select_dtypes('number').astype('float64')
    for c in float_data:
        df = df.assign(**{c: float_data[c]})

    node_children_blacklist = []
    for source in df.columns:
        for target in df.columns:
            if source != target and df[target].dtype == 'category' and df[source].dtype == 'float64':

                node_children_blacklist.append(
                    [source, target])

    return df, node_children_blacklist



indep_test_dict = {
    'knncit': partial(
        pbn.MixedKMutualInformation,
        shuffle_neighbors=5,
        samples=100,
        scaling="normalized_rank",
        gamma_approx=True,
        adaptive_k=True
    ),
    'cgit': CGIT,
    'dgcit': DGCIT,
    'mmcit': MMCIT
}


def learn_pdags_pc(ds_name, df, node_children_blacklist, counter):
    for indep_test_name in indep_test_dict.keys():
        if os.path.isfile(f'./results/real/{indep_test_name}/'+ds_name+'.pkl'):
            continue
        print(counter, f'{indep_test_name}: processing file',
              ds_name, flush=True)

        if indep_test_name != 'knncit':
            indep_test = indep_test_dict[indep_test_name](df=df)
        else:
            indep_test = indep_test_dict[indep_test_name](df=df, k=len(df)//10)

        pdag = pbn.PC().estimate(hypot_test=indep_test, alpha=0.05, allow_bidirected=False,
                                 arc_blacklist=node_children_blacklist, arc_whitelist=[], edge_blacklist=[], edge_whitelist=[], verbose=1)

        with open(f'./results/real/{indep_test_name}/'+ds_name+'.pkl', "wb") as f:
            pickle.dump(pdag, f)


def learn_bn_hillclimbing(ds_name, df, node_children_blacklist, counter):
    if os.path.isfile('./results/real/hillclimbing/'+ds_name+'.pkl'):
        return
    print(counter, 'hillclimbing: processing file', ds_name, flush=True)

    spbn = pbn.SemiparametricBN(nodes=df.columns)
    score = pbn.CVLikelihood(df, k=4)
    op_set = pbn.OperatorPool(
        opsets=([pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()]))

    spbn = pbn.GreedyHillClimbing().estimate(
        start=spbn,
        score=score,
        operators=op_set,
        arc_blacklist=node_children_blacklist,
        arc_whitelist=[],
        max_indegree=10,
        patience=10,
        verbose=1
    )

    with open('./results/real/hillclimbing/'+ds_name+'.pkl', "wb") as f:
        pickle.dump(spbn, f)


def learn_pdag_pc_mrcit(ds_name, df, node_children_blacklist, counter):
    if os.path.isfile('./results/real/mrcit/'+ds_name[:-4]+'.pkl'):
        return
    print(counter, 'mrcit: processing file', ds_name[:-4], flush=True)

    # label-encode the column names and categorical values to be passed to MRCIT R method
    col_dict = {str(k): v for k, v in enumerate(df.columns)}
    df.columns = [str(i) for i, _ in enumerate(df.columns)]
    le = LabelEncoder()
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = le.fit_transform(df[col])
        df[col] = df[col].astype('category')

    blacklist_int = []

    for var1, var2 in node_children_blacklist:
        # find integer column names corresponding to the original names
        col1_int = next(k for k, v in col_dict.items() if v == var1)
        col2_int = next(k for k, v in col_dict.items() if v == var2)
        blacklist_int.append([col1_int, col2_int])

    indep_test = MRCIT(df, num_f2=20)
    pdag = pbn.PC().estimate(hypot_test=indep_test, alpha=0.05, allow_bidirected=False,
                             arc_blacklist=blacklist_int, arc_whitelist=[], edge_blacklist=[], edge_whitelist=[], verbose=1)

    # label-decode
    decoded_pdag = pbn.PartiallyDirectedGraph(
        nodes=[v for v in col_dict.values()])
    for arc in pdag.arcs():
        decoded_pdag.add_arc(col_dict[arc[0]], col_dict[arc[1]])

    for edge in pdag.edges():
        decoded_pdag.add_edge(col_dict[edge[0]], col_dict[edge[1]])

    with open('./results/real/mrcit/'+ds_name[:-4]+'.pkl', "wb") as f:
        pickle.dump(decoded_pdag, f)

directory = os.fsencode('./datasets/real')
filenames = [x for x in enumerate(sorted(os.listdir(directory)))]

for i, filename in filenames:
    ds_name = os.fsdecode(filename)
    df, node_children_blacklist = preprocess_dataset(ds_name)
    
    learn_bn_hillclimbing(ds_name, df, node_children_blacklist, i)
    learn_pdags_pc(ds_name, df, node_children_blacklist, i)
    learn_pdag_pc_mrcit(ds_name, df, node_children_blacklist, i)

