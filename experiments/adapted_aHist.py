from pybnesian import IndependenceTest
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
ro.r('''
require("parallel")
require("SCCI")
require(dplyr)
library("MASS")


Sys.setenv("_R_CHECK_LENGTH_1_CONDITION_" = "true")


source("../external/CMI-adaptive-hist/algorithm/generate_candidate_cuts.R")
source("../external/CMI-adaptive-hist/algorithm/refine_candidate_cuts_exclude_empty_bins.R")
source("../external/CMI-adaptive-hist/algorithm/modified_log.R")
source("../external/CMI-adaptive-hist/algorithm/iterative_cmi_greedy_flexible_parallel.R")

source("../external/CMI-adaptive-hist/algorithm/multi_hist_splitting_seed_based_simple.R")
source("../external/CMI-adaptive-hist/algorithm/oned_hist_iterative.R")
source("../external/CMI-adaptive-hist/algorithm/CMI_estimates.R")
source("../external/CMI-adaptive-hist/algorithm/CMI_pvals.R")
source("../external/CMI-adaptive-hist/algorithm/utils.R")
''')

def data_processing(df, cat_index):
    columns = df.columns

    standardize = [(col, None) if col in cat_index else
                   ([col], MinMaxScaler()) for col in columns]
    x_mapper = DataFrameMapper(standardize)
    df = x_mapper.fit_transform(df).astype('float32')
    df = pd.DataFrame(df, columns=columns)

    return df


class aHist(IndependenceTest):

    def __init__(self, df):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.variables = df.columns.tolist()
        df.columns = [int(x) for x in df.columns]
        cat_index = np.where((df.dtypes == 'category') |
                             (df.dtypes == 'object'))[0]
        df = data_processing(
            df, cat_index)
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
        ro.globalenv['matrix_df'] = df_r

        # Create type vector
        ncol = ro.r['ncol'](ro.globalenv['matrix_df'])[0]
        nrow = ro.r['nrow'](ro.globalenv['matrix_df'])
        type_vec = ro.IntVector([1] * ncol)

        # Check for columns with fewer than 11 unique values
        for i in range(1, ncol + 1):
            col_data = ro.r(f'matrix_df[,{i}]')
            unique_vals = ro.r['unique'](col_data)
            if len(unique_vals) < 11:
                type_vec[i-1] = 0

        self.sufficient_stats = ro.ListVector({
            'dm': ro.globalenv['matrix_df'],
            'type': type_vec,
            'n': nrow
        })

        # Assuming Chisq95 is available in R environment

    def num_variables(self):
        return len(self.variables)

    def variable_names(self):
        return self.variables

    def has_variables(self, vars):

        return set(vars).issubset(set(self.variables))

    def name(self, index):

        return self.variables[index]

    def pvalue(self, x, y, z):

        if z is not None and len(z) > 0:
            if isinstance(z, str):
                z = [z]
        else:
            z = []

        z = ro.IntVector([int(col) + 1 for col in z]) if len(z) > 0 else ro.IntVector([])
        return ro.r['CMIp.Chisq95'](int(x)+ 1, int(y)+ 1, z, self.sufficient_stats)
