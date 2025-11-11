import os
# place the device java location
os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-25-oracle-x64/bin/java"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
from jpype import JClass, JObject
import pytetrad.tools.translate as tr
import edu.cmu.tetrad.search.test as test
from pybnesian import IndependenceTest


class DGCIT(IndependenceTest):

    def __init__(self, df):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.variables = df.columns.tolist()

        self.java_data = tr.pandas_data_to_tetrad(df)

        self.test = test.IndTestDegenerateGaussianLrt(self.java_data)

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

        # define Z as a set of Tetrad node objects
        z_set = JClass("java.util.HashSet")()
        for zcol in z:
            z_set.add(JObject(self.java_data.getVariable(
                zcol), JClass("java.lang.Object")))

        # evaluate it in Java
        pval = self.test.checkIndependence(self.java_data.getVariable(
            x), self.java_data.getVariable(y), z_set).getPValue()
        return pval
