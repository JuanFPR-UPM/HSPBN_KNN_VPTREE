#!/usr/bin/env python
# coding: utf-8

import itertools
import os
from generate_new_bns import ProbabilisticModel

### hyper-parameters, total 5400 datasets for training + 1080 for testing
# number of repetitions
n_reps = (0, 10)

n_samples_train = [150, 375, 750, 1500, 3000]
n_samples_test = 1000

n_vars = [10, 20, 30]

edge_densities = [0.06, 0.1, 0.14, 0.18]

discrete_ratios = [0.1, 0.5, 0.9]

kde_ratios = [0.0, 0.5, 1.0]

try:
    os.mkdir('./datasets')
except:
    pass

try:
    os.mkdir('./datasets/synthetic')
except:
    pass

try:
    os.mkdir('./datasets/synthetic/test')
except:
    pass
try:
    os.mkdir('./hspbns')
except:
    pass


for rep in range(n_reps[0],n_reps[1]):
    sample_seed = 0
    iterParams = [n_vars, discrete_ratios, kde_ratios, edge_densities]
    for i, (n_var, discrete_ratio, kde_ratio, edge_density) in enumerate(itertools.product(*iterParams)):
   
        bb = ProbabilisticModel.generate_new_model(n_var, discrete_ratio, kde_ratio, edge_density, seed=rep + 50 * i)
        output_file_name = '_'.join([str(rep),str(n_var),str(discrete_ratio),str(kde_ratio), str(edge_density)])
        output_file_name = output_file_name.replace('.','p')
        print(output_file_name)

        # skip if done in previous work session
        if os.path.isfile(f"./datasets/synthetic/{output_file_name + '_3000'}.csv") and os.path.isfile(f"./datasets/synthetic/test/{output_file_name + '_1000'}.csv"):
            continue

        elif not os.path.isfile(f"./datasets/synthetic/{output_file_name + '_150'}.csv"):
            bb.save('./hspbns/' + output_file_name + '.pkl')
        # reload spbn if exists
        else:
            bb = ProbabilisticModel.load('./hspbns/' + output_file_name + '.pkl')
        
    
        for samples in n_samples_train:
            sample_output_file_name = output_file_name + '_' + str(samples)
            print(sample_output_file_name)
            sample_seed += 1
            if os.path.isfile(f"./datasets/synthetic/{sample_output_file_name}.csv"):
                continue
            # sample model with different seeds
            df = bb.ground_truth_bn.sample(n=samples, seed=rep + i * sample_seed * 50, ordered=True).to_pandas()
            df.to_csv(f"./datasets/synthetic/{sample_output_file_name}.csv", index=False)

        sample_output_file_name = output_file_name + '_1000'
        print('test/' + sample_output_file_name)
        sample_seed += 1
        if os.path.isfile(f"./datasets/synthetic/test/{sample_output_file_name}.csv"):
            continue
        df = bb.ground_truth_bn.sample(n=n_samples_test, seed=rep + i * sample_seed * 50, ordered=True).to_pandas()
        df.to_csv(f"./datasets/synthetic/test/{sample_output_file_name}.csv", index=False)
            
    
    

