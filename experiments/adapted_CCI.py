from pybnesian import IndependenceTest
from pybnesian import IndependenceTest
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
ro.r('''require("CCI")''')


class CCI(IndependenceTest):

    def __init__(self, df, p=None, nperm=200, parametric=True, method='rf', metric='Auto', robust=True):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.variables = df.columns.tolist()
        self.df = df

        if p is None:
            n_samples = df.shape[0]
            if n_samples <= 375:
                self.p = 0.80
            elif n_samples <= 750:
                self.p = 0.65
            else:
                self.p = 0.50
        else:
            self.p = p

        self.nperm = nperm
        self.parametric = "TRUE" if parametric else "FALSE"
        self.robust = "TRUE" if robust else "FALSE"
        self.method = method
        self.metric = metric

        
        # Assuming CCI.test is available in the R environment inside the CCI package

    def num_variables(self):
        return len(self.variables)

    def variable_names(self):
        return self.variables

    def has_variables(self, vars):
        return set(vars).issubset(set(self.variables))

    def name(self, index):
        return self.variables[index]

    def pvalue(self, x, y, z):

        cols_to_keep = [x, y]
        safe_z_names = []

        if z is not None and len(z) > 0:
            if isinstance(z, str):
                z = [z]
            cols_to_keep += z
            safe_z_names = [f"Z{i+1}" for i in range(len(z))]
            # format R string with Z column names
            z_formula = " + ".join([f"{zn}" for zn in safe_z_names])
            formula_str = f"Y ~ X + {z_formula}"
        else:
            formula_str = f"Y ~ X | 1"

        df_subset = self.df[cols_to_keep].copy()
        df_subset.columns = ["Y", "X"] + safe_z_names

        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df_subset)
        ro.globalenv['df_r'] = df_r

        r_command = (
            f"CCI.test(formula = as.formula('{formula_str}'), "
            f"data = df_r, "
            f"p = {self.p}, "
            f"nperm = {self.nperm}, "
            f"parametric = {self.parametric}, "
            f"method = '{self.method}', "
            f"metric = '{self.metric}', "
            f"robust = {self.robust})$p.value"
        )

        # Execute the R code and extract the float p-value
        try:
            pval = ro.r(r_command)[0]
        except Exception as e:
            # Catch the specific zero standard deviation crash from R
            if "zero standard deviation" in str(e):
                # Fallback to empirical p-value (parametric = FALSE)
                fallback_command = (
                    f"CCI.test(formula = as.formula('{formula_str}'), "
                    f"data = df_r, "
                    f"p = {self.p}, "
                    f"nperm = {self.nperm}, "
                    f"parametric = FALSE, "  # OVERRIDE HERE
                    f"method = '{self.method}', "
                    f"metric = '{self.metric}', "
                    f"robust = {self.robust})$p.value"
                )
                pval = ro.r(fallback_command)[0]
        return pval
