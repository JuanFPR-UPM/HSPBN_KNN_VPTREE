from pybnesian import IndependenceTest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
ro.r('''require("MXM")''')


class MMCIT(IndependenceTest):

    def __init__(self, df):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.variables = df.columns.tolist()
        self.df = df

        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
        ro.globalenv['df_r'] = df_r
        # Assuming ci.mm is available in in the R environment inside the MXM package

    def num_variables(self):
        return len(self.variables)

    def variable_names(self):
        return self.variables

    def has_variables(self, vars):
        return set(vars).issubset(set(self.variables))

    def name(self, index):
        return self.variables[index]

    def pvalue(self, x, y, z):

        # R method uses column indices instead of names
        x = np.where(self.df.columns == x)[0][0]
        y = np.where(self.df.columns == y)[0][0]

        # format R string with Z column indices
        if z is not None and len(z) > 0:
            if isinstance(z, str):
                z = [z]
            z_str = "c(" + ",".join([str(np.where(self.df.columns ==
                                                  col)[0][0] + 1) for col in z]) + ")"
        else:
            z_str = "NULL"

        r_call = f"ci.mm({int(x)+1}, {int(y)+1}, cs={z_str}, dat=df_r)"

        # evaluate it in R
        res = ro.r(r_call)   # result is an R named numeric vector
        log_pval = res.rx2('logged.p-value')[0]   # extract by name
        return np.exp(log_pval)    # exponentiate logvalue
