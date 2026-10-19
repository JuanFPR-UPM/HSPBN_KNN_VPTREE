import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pybnesian import IndependenceTest
from external.CIml.CIml import CIml_sym

class Adapted_CIml(IndependenceTest):

    def __init__(self, df):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.variables = df.columns.tolist()
        self.df = df.copy()

        for col in self.df.columns:
            col_series = self.df[col]
            if (col_series.dtype == 'object' or 
                isinstance(col_series.dtype, pd.CategoricalDtype) or 
                col_series.dtype.name == 'category'):
                
                self.df[col] = LabelEncoder().fit_transform(col_series)

    def num_variables(self):
        return len(self.variables)

    def variable_names(self):
        return self.variables

    def has_variables(self, vars):
        return set(vars).issubset(set(self.variables))

    def name(self, index):
        return self.variables[index]

    def pvalue(self, x, y, z):

        z = [str(col) for col in z] if z is not None else []
        z_series = self.df[z].values
        
        if z_series.size == 0:
            z_series = np.empty((len(self.df), 0))
      
        return CIml_sym(self.df[str(x)].values, self.df[str(y)].values, z_series)
