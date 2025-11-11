#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import pickle
import pybnesian as pbn
from functools import partial
from adapted_MMCIT import MMCIT
from adapted_CGIT import CGIT
from adapted_DGCIT import DGCIT
# from adapted_LCIT import Adapted_LCIT
from adapted_MRCIT import MRCIT

try:
    os.mkdir('./results/')
except:
    pass

try:
    os.mkdir('./results/synthetic/')
except:
    pass

try:
    os.mkdir('./results/synthetic/knncit')
except:
    pass

try:
    os.mkdir('./results/synthetic/mrcit')
except:
    pass

try:
    os.mkdir('./results/synthetic/lcit')
except:
    pass

try:
    os.mkdir('./results/synthetic/mmcit')
except:
    pass

try:
    os.mkdir('./results/synthetic/cgit')
except:
    pass

try:
    os.mkdir('./results/synthetic/dgcit')
except:
    pass

try:
    os.mkdir('./results/synthetic/hillclimbing')
except:
    pass

def preprocess_dataset(filename):
    df = pd.read_csv('datasets/synthetic/'+ filename)

    cat_data = df.select_dtypes('int64').astype('str').astype('category')
    for c in cat_data:
        df = df.assign(**{c: cat_data[c]})
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
    'mrcit': partial(MRCIT, num_f2=20),
    'cgit': CGIT,
    'dgcit': DGCIT,
    'mmcit': MMCIT,
    # 'lcit': partial(Adapted_LCIT, n_components=32, hidden_sizes=[4]),
}

def learn_bn_hillclimbing(filename, counter):
    if os.path.isfile('./results/synthetic/hillclimbing/'+filename[:-4]+'.pkl'):
        return
    print(counter, 'hillclimbing: processing file', filename)

    df, node_children_blacklist = preprocess_dataset(filename)

    spbn =  pbn.SemiparametricBN(nodes=df.columns)
    score = pbn.CVLikelihood(df, k=4)
    op_set = pbn.OperatorPool(opsets=([pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()]))

    spbn = pbn.GreedyHillClimbing().estimate(
            start=spbn,
            score=score,
            operators=op_set,
            arc_blacklist=node_children_blacklist,
            arc_whitelist=[],
            max_indegree=4,
            patience=10
        )

    with open('./results/synthetic/hillclimbing/'+filename[:-4]+'.pkl', "wb") as f:
        pickle.dump(spbn, f)


def learn_pdags_pc(filename, counter):
    df, node_children_blacklist = preprocess_dataset(filename)
    for indep_test_name in indep_test_dict.keys():
        if os.path.isfile(f'./results/synthetic/{indep_test_name}/'+filename[:-4]+'.pkl'):
            continue
        print(counter, f'{indep_test_name}: processing file', filename)

        if indep_test_name != 'knncit':
            indep_test = indep_test_dict[indep_test_name](df=df)
        else:
            indep_test = indep_test_dict[indep_test_name](df=df, k=len(df)//10)
        pdag = pbn.PC().estimate(hypot_test=indep_test, alpha=0.05, allow_bidirected=False, arc_blacklist = node_children_blacklist, arc_whitelist = [], edge_blacklist = [], edge_whitelist = [], verbose = 1)

        with open(f'./results/synthetic/{indep_test_name}/'+filename[:-4]+'.pkl', "wb") as f:
            pickle.dump(pdag, f)


directory = os.fsencode('./datasets/synthetic')
filenames = sorted(
    f for f in os.listdir(directory)
    if os.path.isfile(os.path.join(directory, f))
)
for i, filename in enumerate(filenames):
    learn_bn_hillclimbing(os.fsdecode(filename), i)
    learn_pdags_pc(os.fsdecode(filename), i)

