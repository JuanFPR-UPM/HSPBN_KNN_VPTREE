import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from pybnesian import IndependenceTest
from external.LCIT.src.citests import LCIT
import pandas as pd
import numpy as np



class Adapted_LCIT(IndependenceTest):

    def __init__(self, df, n_components=32, hidden_sizes=[4]):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.variables = df.columns.tolist()
        self.df = df
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes

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
        batch_norm = True
        # if Z is empty, force unconditional independece test against a constant value
        if z_series.size == 0:
            z_series = np.ones(len(self.df))
            batch_norm = False

        return LCIT(self.df[str(x)].values, self.df[str(y)].values, z_series, n_components=self.n_components, hidden_sizes=self.hidden_sizes, normalize=True, batch_norm=batch_norm)
