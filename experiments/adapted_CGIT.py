from pybnesian import IndependenceTest
from sklearn.preprocessing import LabelEncoder
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
ro.r('''require("bnlearn")''')


class CGIT(IndependenceTest):

    def __init__(self, df):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.variables = df.columns.tolist()
        self.df = df

        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
        ro.globalenv['df_r'] = df_r
        # Assuming ci.test is available in the R environment inside the bnlearn package

    def num_variables(self):
        return len(self.variables)

    def variable_names(self):
        return self.variables

    def has_variables(self, vars):
        return set(vars).issubset(set(self.variables))

    def name(self, index):
        return self.variables[index]

    def pvalue(self, x, y, z):

        ci_test_str = "mi-cg"

        if z is not None and len(z) > 0:
            if isinstance(z, str):
                z = [z]
            # format R string with Z column names
            z_str = "z=c(\"" + "\",\"".join(z) + "\"),"
        else:
            z_str = ""
            z = []

        dtypes = set(str(dt) for dt in self.df[[x, y] + z].dtypes)
        if all(dt == "category" for dt in dtypes):
            ci_test_str = "mi"    # fully discrete
        elif all(dt == "float64" for dt in dtypes):
            ci_test_str = "mi-g"  # fully continuous

        r_call = f"ci.test(x=\"{x}\", y=\"{y}\", {z_str} data=df_r, test=\"{ci_test_str}\")"

        # evaluate it in R
        res = ro.r(r_call)   # result is an R named numeric vector
        pval = res.rx2('p.value')[0]   # extract by name
        return pval
