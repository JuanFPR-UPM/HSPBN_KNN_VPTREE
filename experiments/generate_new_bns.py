import pybnesian as pbn
import pandas as pd
import pyarrow as pa
import numpy as np
import scipy.special
import itertools
import pickle

ISS_PRIOR_DISCRETE = 3
ISS_PRIOR_NMM = 3

class FixedDiscreteFactorType(pbn.FactorType):
    def __init__(self):
        pbn.FactorType.__init__(self)

    def __str__(self):
        return "FixedDiscreteFactorType"

    def new_factor(self, model, variable, evidence, *args, **kwargs):
        return FixedDiscreteFactor(variable, evidence, *args, **kwargs)

class FixedDiscreteFactor(pbn.Factor):
    def __init__(self, variable, evidence, variable_values, evidence_values, discrete_configs, logprob):
        pbn.Factor.__init__(self, variable, evidence)
        self.variable_values = variable_values
        self.evidence_values = evidence_values
        self.discrete_configs = discrete_configs
        self.logprob = logprob

    @classmethod
    def new_random_cpd(cls, variable, evidence):
        cats = [ProbabilisticModel.discrete_categories[n] for n in evidence]
        discrete_configs = list(itertools.product(*cats))

        num_var_cats = len(ProbabilisticModel.discrete_categories[variable])
        num_configs = len(discrete_configs)

        logprob = np.empty((num_configs, num_var_cats), dtype=float)

        for i in range(num_configs):
            logprob[i,:] = np.log(np.random.dirichlet([ISS_PRIOR_DISCRETE]*num_var_cats))

        return FixedDiscreteFactor(variable,
                                   evidence, 
                                   ProbabilisticModel.discrete_categories[variable],
                                   cats,
                                   discrete_configs,
                                   logprob)

    def data_type(self):
        return pa.dictionary(pa.int8(), pa.utf8())

    def fitted(self):
        return True

    def logl(self, df):
        df_pandas = df.to_pandas()
        ll = np.empty((df_pandas.shape[0],), dtype=float)

        for i,config in enumerate(self.discrete_configs):
            df_config = df_pandas
            
            # Filter by discrete configuration
            for (dv, dc) in zip(self.evidence(), config):
                df_config = df_config.where(df_config[dv] == dc)

            for j,var_value in enumerate(self.variable_values):
                df_value = df_config.where(df_config[self.variable()] == var_value)

                # Filtered rows are NaN
                not_null_indices = df_value.notnull().all(axis=1)

                ll[not_null_indices] = self.logprob[i,j]

        return ll


    def slogl(self, df):
        return self.logl(df).sum()

    def sample(self, n, evidence, seed):
        np.random.seed(seed)
        s = np.full((n,), '', dtype='object')

        if evidence is not None:
            evidence_pandas = evidence.to_pandas()

            for i,config in enumerate(self.discrete_configs):
                ev_config = evidence_pandas
                # Filter by discrete configuration
                for (dv, dc) in zip(self.evidence(), config):
                    ev_config = ev_config.where(ev_config[dv] == dc)
                
                # Filtered rows are NaN
                not_null_indices = ev_config.notnull().all(axis=1)
                # Filter df
                ev_config = ev_config[not_null_indices]

                arr_variable_values = np.array(self.variable_values)

                s[not_null_indices] = arr_variable_values[np.random.choice(len(self.variable_values), 
                                                                p=np.exp(self.logprob[i,:]), 
                                                                size=ev_config.shape[0])]
        else:
            arr_variable_values = np.array(self.variable_values)

            s = arr_variable_values[np.random.choice(len(self.variable_values), 
                                                     p=np.exp(self.logprob[0,:]), 
                                                     size=n)]

        series = pd.Series(s, dtype='category')

        return pa.array(s)

    def type(self):
        return FixedDiscreteFactorType()

    def __getstate_extra__(self):
        return (self.variable_values, self.evidence_values, self.discrete_configs, self.logprob)

    def __setstate_extra__(self, extra):
        self.variable_values = extra[0]
        self.evidence_values = extra[1]
        self.discrete_configs = extra[2]
        self.logprob = extra[3]

class FixedCLGType(pbn.FactorType):
    def __init__(self):
        pbn.FactorType.__init__(self)

    def __str__(self):
        return "FixedCLGType"

    def new_factor(self, model, variable, evidence, *args, **kwargs):
        return NormalMixtureCPD(variable, evidence, *args, **kwargs)

class FixedCLG(pbn.Factor):
    def __init__(self, variable, evidence, discrete_evidence, continuous_evidence, discrete_configs, lgs):
        pbn.Factor.__init__(self, variable, evidence)
        self.discrete_evidence = discrete_evidence
        self.continuous_evidence = continuous_evidence
        self.discrete_configs = discrete_configs
        self._lgs = lgs

    @classmethod
    def new_random_cpd(cls, variable, discrete_evidence, continuous_evidence):
        evidence = discrete_evidence + continuous_evidence

        cats = [ProbabilisticModel.discrete_categories[n] for n in discrete_evidence]
        discrete_configs = list(itertools.product(*cats))

        lgs = {}
        for conf in discrete_configs:
            betas = np.empty((len(continuous_evidence)+1,))
            betas[0] = np.random.normal(0, 2)
            betas[1:] = np.random.choice([1, -1], size=len(continuous_evidence), p=[0.5, 0.5])*\
                        np.random.uniform(0, 5, size=len(continuous_evidence))

            var = 0.2 + np.random.chisquare(1)
            lgs[conf] = pbn.LinearGaussianCPD(variable, continuous_evidence, betas, var)
            
        return FixedCLG(variable, evidence, discrete_evidence, continuous_evidence, discrete_configs, lgs)

    def data_type(self):
        return pa.float64()

    def fitted(self):
        return True

    def logl(self, df):
        df_pandas = df.to_pandas()
        ll = np.empty((df_pandas.shape[0],), dtype=float)

        for config in self.discrete_configs:
            df_config = df_pandas
            # Filter by discrete configuration
            for (dv, dc) in zip(self.discrete_evidence, config):
                df_config = df_config.where(df_config[dv] == dc)
            
            # Filtered rows are NaN
            not_null_indices = df_config.notnull().all(axis=1)
            # Filter df
            df_config = df_config[not_null_indices]

            ll[not_null_indices] = self._lgs[config].logl(df_config)

        return ll

    def slogl(self, df):
        return self.logl(df).sum()

    def sample(self, n, evidence, seed):
        np.random.seed(seed)
        s = np.empty((n,), dtype=float)

        if evidence is not None:
            evidence_pandas = evidence.to_pandas()

            for config in self.discrete_configs:
                ev_config = evidence_pandas
                # Filter by discrete configuration
                for (dv, dc) in zip(self.discrete_evidence, config):
                    ev_config = ev_config.where(ev_config[dv] == dc)
                
                # Filtered rows are NaN
                not_null_indices = ev_config.notnull().all(axis=1)
                # Filter df
                ev_config = ev_config[not_null_indices]

                s[not_null_indices] = self._lgs[config].sample(ev_config.shape[0], ev_config, seed=seed+328)
        else:
            s = self._lgs[()].sample(n, None, seed=seed+389).to_numpy()

        return pa.array(s)

    def type(self):
        return FixedCLGType()

    def __getstate_extra__(self):
        return (self.discrete_evidence, self.continuous_evidence, self.discrete_configs, self._lgs)

    def __setstate_extra__(self, extra):
        self.discrete_evidence = extra[0]
        self.continuous_evidence = extra[1]
        self.discrete_configs = extra[2]
        self._lgs = extra[3]


class NormalMixtureType(pbn.FactorType):
    def __init__(self):
        pbn.FactorType.__init__(self)

    def __str__(self):
        return "NormalMixtureType"

    def new_factor(self, model, variable, evidence, *args, **kwargs):
        return NormalMixtureCPD(variable, evidence, *args, **kwargs)


class NormalMixtureCPD(pbn.Factor):
    def __init__(self, variable, evidence, discrete_evidence, continuous_evidence, discrete_configs, priors, lgs):
        pbn.Factor.__init__(self, variable, evidence)
        self.discrete_evidence = discrete_evidence
        self.continuous_evidence = continuous_evidence
        self.discrete_configs = discrete_configs
        self._priors = priors
        self._lgs = lgs

    @classmethod
    def new_random_cpd(cls, variable, discrete_evidence, continuous_evidence):
        evidence = discrete_evidence + continuous_evidence

        cats = [ProbabilisticModel.discrete_categories[n] for n in discrete_evidence]
        discrete_configs = list(itertools.product(*cats))

        priors = {}
        lgs = {}
        for conf in discrete_configs:
            num_components = np.random.choice([2, 3], size=1, p=[0.45, 0.55])[0]
            priors[conf] = np.random.dirichlet([ISS_PRIOR_NMM]*num_components)

            cpds = []
            for k in range(num_components):
                betas = np.empty((len(continuous_evidence)+1,))
                betas[0] = np.random.normal(0, 2)
                betas[1:] = np.random.choice([1, -1], size=len(continuous_evidence), p=[0.5, 0.5])*\
                            np.random.uniform(0, 5, size=len(continuous_evidence))

                var = 0.2 + np.random.chisquare(1)
                cpds.append(pbn.LinearGaussianCPD(variable, continuous_evidence, betas, var))
            
            lgs[conf] = cpds

        return NormalMixtureCPD(variable, evidence, discrete_evidence, continuous_evidence, discrete_configs, priors, lgs)

    def data_type(self):
        return pa.float64()

    def fitted(self):
        return True

    def logl(self, df):
        df_pandas = df.to_pandas()
        ll = np.empty((df_pandas.shape[0],), dtype=float)

        for config in self.discrete_configs:
            df_config = df_pandas
            # Filter by discrete configuration
            for (dv, dc) in zip(self.discrete_evidence, config):
                df_config = df_config.where(df_config[dv] == dc)
            
            # Filtered rows are NaN
            not_null_indices = df_config.notnull().all(axis=1)
            # Filter df
            df_config = df_config[not_null_indices]

            logpriors = np.log(self._priors[config])
            ll_matrix = np.tile(logpriors, (df_config.shape[0], 1)).T

            for i, lg in enumerate(self._lgs[config]):
                ll_matrix[i,:] += lg.logl(df_config)

            ll[not_null_indices] = scipy.special.logsumexp(ll_matrix, axis=0)

        return ll

    def slogl(self, df):
        return self.logl(df).sum()

    def sample(self, n, evidence, seed):
        np.random.seed(seed)
        s = np.empty((n,), dtype=float)

        if evidence is not None:
            evidence_pandas = evidence.to_pandas()

            for config in self.discrete_configs:
                ev_config = evidence_pandas
                # Filter by discrete configuration
                for (dv, dc) in zip(self.discrete_evidence, config):
                    ev_config = ev_config.where(ev_config[dv] == dc)
                
                # Filtered rows are NaN
                not_null_indices = ev_config.notnull().all(axis=1)
                # Filter df
                ev_config = ev_config[not_null_indices]

                priors = self._priors[config]
                num_components = priors.shape[0]
                component = np.random.choice(num_components, p=priors, size=ev_config.shape[0])

                s_config = np.empty((ev_config.shape[0],), dtype=float)

                ev_config.reset_index(inplace=True)

                for i, cpd in enumerate(self._lgs[config]):
                    component_size = np.sum(component == i)
                    evidence_df = ev_config.iloc[component == i,:]
                    s_config[component == i] = cpd.sample(component_size, evidence_df, seed=seed+357).to_numpy()

                s[not_null_indices] = s_config
        else:
            priors = self._priors[()]
            num_components = priors.shape[0]
            component = np.random.choice(num_components, p=priors, size=n)

            for i, cpd in enumerate(self._lgs[()]):
                component_size = np.sum(component == i)
                evidence_df = evidence.to_pandas().iloc[component == i,:]
                s[component == i] = cpd.sample(component_size, evidence_df, seed=seed+337).to_numpy()

        return pa.array(s)

    def type(self):
        return NormalMixtureType()

    def __getstate_extra__(self):
        return (self.discrete_evidence, self.continuous_evidence, self.discrete_configs,
                self._priors, self._lgs)

    def __setstate_extra__(self, extra):
        self.discrete_evidence = extra[0]
        self.continuous_evidence = extra[1]
        self.discrete_configs = extra[2]
        self._priors = extra[3]
        self._lgs = extra[4]

class ProbabilisticModel:

    def __init__(self, expected_bn, ground_truth_bn):
        self.expected_bn = expected_bn
        self.ground_truth_bn = ground_truth_bn

    @classmethod
    def generate_structure(cls, num_nodes, ratio_discrete, ratio_kde, edge_density, seed=0):
        np.random.seed(seed)
        nodes = [str(n) for n in range(num_nodes)]
        discrete_nodes = [nodes[i] for i in range(len(nodes)) if i < len(nodes)*ratio_discrete]
        
        discrete_categories = dict()
        for n in discrete_nodes:
            node_cats = [n + str(i) for i in range(np.random.randint(2, 6))]
            discrete_categories[n] = node_cats
        continuous_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i] not in discrete_nodes]
        ProbabilisticModel.discrete_nodes = discrete_nodes
        ProbabilisticModel.discrete_categories = discrete_categories
        ProbabilisticModel.continuous_nodes = continuous_nodes

        continuous_node_types = np.asarray([pbn.LinearGaussianCPDType(), pbn.CKDEType()])
        cc_index = [0 if i < len(continuous_nodes)*(1-ratio_kde) else 1 for i in range(len(continuous_nodes))]
        node_types = list(continuous_node_types[cc_index])

        bn = pbn.SemiparametricBN(discrete_nodes + continuous_nodes, [(n, pbn.DiscreteFactorType()) for n in discrete_nodes] +\
                                                                     [(n, nt) for n, nt in zip(continuous_nodes, node_types)])
        
        parent_count = {n: 0 for n in nodes}
        children_count = {n: 0 for n in nodes}
        max_in_degrees = {n: np.random.randint(1, 5) for n in nodes}
        max_out_degrees = {n: np.random.randint(5, 9) for n in nodes}
        
        def check_in_degree(s, d):
            if parent_count[d] >= max_in_degrees[d] or children_count[s] >= max_out_degrees[s]:
                return
            bn.add_arc(s, d)
            parent_count[d] += 1
            children_count[s] += 1

        # Helper to sample arcs from candidate edges
        def sample_and_add_edges(candidates):
            num_arcs = len(candidates)
            active_mask = np.random.choice(2, size=num_arcs, p=[1 - edge_density, edge_density])
            for i in range(num_arcs):
                if active_mask[i] == 1:
                    s, d = candidates[i]
                    check_in_degree(s, d)

        # Edges: discrete → discrete
        arcs = [(s, d) for i, s in enumerate(discrete_nodes[:-1]) for d in discrete_nodes[i+1:]]
        sample_and_add_edges(arcs)

        # Edges: discrete → continuous
        arcs = [(s, d) for s in discrete_nodes for d in continuous_nodes]
        sample_and_add_edges(arcs)

        # Edges: continuous → continuous
        arcs = [(s, d) for i, s in enumerate(continuous_nodes[:-1]) for d in continuous_nodes[i+1:]]
        sample_and_add_edges(arcs)

        return bn

    @classmethod
    def generate_discrete_parameters(cls, structure, node):
        parents = structure.parents(node)

        return FixedDiscreteFactor.new_random_cpd(node, parents)

    @classmethod
    def generate_clg_parameters(cls, structure, node):
        parents = structure.parents(node)

        discrete_evidence = list(set(parents).intersection(set(ProbabilisticModel.discrete_nodes)))
        continuous_evidence = list(set(parents).intersection(set(ProbabilisticModel.continuous_nodes)))

        return FixedCLG.new_random_cpd(node, discrete_evidence, continuous_evidence)

    @classmethod
    def generate_ckde_parameters(cls, structure, node):
        parents = structure.parents(node)

        discrete_evidence = list(set(parents).intersection(set(ProbabilisticModel.discrete_nodes)))
        continuous_evidence = list(set(parents).intersection(set(ProbabilisticModel.continuous_nodes)))

        return NormalMixtureCPD.new_random_cpd(node, discrete_evidence, continuous_evidence)

    @classmethod
    def generate_parameters(cls, structure, seed=0):
        np.random.seed(seed)
        default_factor_type = {pa.float64(): [FixedCLGType(), NormalMixtureType()],
                               pa.dictionary(pa.int8(), pa.utf8()): [FixedDiscreteFactorType()]}

        ground_truth = pbn.HeterogeneousBN(default_factor_type, structure.graph())
        cpds = []

        for node in structure.nodes():
            node_type = structure.node_type(node)

            if node_type == pbn.DiscreteFactorType():
                cpds.append(ProbabilisticModel.generate_discrete_parameters(structure, node))
            elif node_type == pbn.LinearGaussianCPDType():
                cpds.append(ProbabilisticModel.generate_clg_parameters(structure, node))
            elif node_type == pbn.CKDEType():
                cpds.append(ProbabilisticModel.generate_ckde_parameters(structure, node))

        ground_truth.add_cpds(cpds)

        return ground_truth

    @classmethod
    def generate_new_model(cls, num_nodes, ratio_discrete, ratio_kde, edge_density, seed=0):
        expected_bn = ProbabilisticModel.generate_structure(num_nodes, ratio_discrete, ratio_kde, edge_density, seed=seed)
        ground_truth_bn = ProbabilisticModel.generate_parameters(expected_bn, seed=seed+2000)
        return ProbabilisticModel(expected_bn, ground_truth_bn)

    def save(self, name):
        self.ground_truth_bn.include_cpd = True

        with open(name, 'wb') as f:
            pickle.dump((self.expected_bn, self.ground_truth_bn), f)

    @classmethod
    def load(cls, name):
        with open(name, 'rb') as f:
            expected_bn, ground_truth_bn = pickle.load(f)

        return ProbabilisticModel(expected_bn, ground_truth_bn)


