import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from pybnesian import IndependenceTest
from external.CIml.function_github_Pillai import pillai_trace

class Pillai(IndependenceTest):

    def __init__(self, df, seed=42):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.variables = df.columns.tolist()
        self.df = df
        self.seed = seed

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
      
        return pillai_trace(x, y, z, self.df, boolean=False, kwargs={'seed':self.seed})[1]
