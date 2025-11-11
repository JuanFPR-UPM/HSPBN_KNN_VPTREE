#!/usr/bin/env python
# coding: utf-8

import os
os.environ["OMP_NUM_THREADS"] = '16'
import pandas as pd
import pybnesian as pbn
import pickle

# no need to include Hill Climbing as it learns the HSPBN directly
methodLocDict = {
                '$k$NNCIT': ['./results/synthetic/knncit/', './results/synthetic/knncit/hspbns/'],
                 '$MRCIT$': ['./results/synthetic/mrcit/','./results/synthetic/mrcit/hspbns/'],
                '$MMCIT$': ['./results/synthetic/mmcit/', './results/synthetic/mmcit/hspbns/'],
                 '$CGIT$': ['./results/synthetic/cgit/', './results/synthetic/cgit/hspbns/'],
                 '$DGCIT$': ['./results/synthetic/dgcit/', './results/synthetic/dgcit/hspbns/'],
                }

groundTruthLoc = './hspbns/'


try:
    os.mkdir('./results/synthetic/knncit/hspbns/')
except:
    pass

try:
    os.mkdir('./results/synthetic/mrcit/hspbns/')
except:
    pass

try:
    os.mkdir('./results/synthetic/mmcit/hspbns/')
except:
    pass

try:
    os.mkdir('./results/synthetic/cgit/hspbns/')
except:
    pass

try:
    os.mkdir('./results/synthetic/dgcit/hspbns/')
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



def find_pdag_nodetypes_topological(pdag, df, node_children_blacklist):
    dag = None

    arcs = [[u,v] for u,v in pdag.arcs()]

    for v1, v2 in arcs:
        if pdag.has_arc(v1, v2) and pdag.has_arc(v2, v1):
            pdag.remove_arc(v1, v2)
            pdag.remove_arc(v2, v1)
            pdag.add_edge(v1, v2)

    try:
        dag = pdag.to_dag()   # this raises exception is PDAG is non-extendable
        pdagfalse = pbn.PartiallyDirectedGraph(df.columns)    # already a DAG, athough PDAG C++ class is needed
        for arc in dag.arcs():
            pdagfalse.add_arc(arc[0], arc[1])
        spbn = pbn.SemiparametricBNType().spbn_from_pdag(df, pdagfalse, node_children_blacklist)   #   learn CPD types

    except:
        spbn = pbn.SemiparametricBNType().spbn_from_pdag(df, pdag, node_children_blacklist)   #   induce HSPBN and learn CPD types
    return spbn

entry = []
for graphFile in sorted(os.listdir(groundTruthLoc))[:]:
    splitted = graphFile.split('_')
    n_rep = splitted[0]
    n_vars = splitted[1]
    discrete_ratio = splitted[2].replace('p','.')
    kde_ratio = splitted[3].replace('p','.')
    edge_density = splitted[4].replace('p','.')[:-4]
    print(graphFile)

    for samples in ['150', '375', '750', '1500', '3000']:
    
        df, node_children_blacklist = preprocess_dataset(graphFile[:-4] + '_' + samples + '.csv')
        
        for method in methodLocDict.keys():
            try:  
                with open(methodLocDict[method][1]+ graphFile[:-4]+ '_' + samples+'.pickle', "rb") as f:
                    pickle.load(f)
                    # skip existing HSPBNs

            except FileNotFoundError:
                try:
                    with open(methodLocDict[method][0]+ graphFile[:-4]+ '_' + samples+'.pkl', "rb") as f:
                        learned_graph = pickle.load(f)
                except FileNotFoundError:
                    # in case an independence test did not fully learn all 5400 networks
                    continue

                try:
                    learned_spbn = find_pdag_nodetypes_topological(learned_graph, df, node_children_blacklist)
                except ValueError as e:
                    continue
                print(method)
                learned_spbn.save(methodLocDict[method][1]+ graphFile[:-4]+ '_' + samples, True)