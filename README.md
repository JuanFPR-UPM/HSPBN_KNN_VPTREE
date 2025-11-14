# Constraint-based learning of HSPBNs

This repository contains the code and experiments for the paper:

"*Constraint-based learning of hybrid semi-parametric Bayesian networks*",

which has been submitted to [Information Sciences (Elsevier)](https://www.sciencedirect.com/journal/information-sciences).

Additionally, it contains the implementation of ${k}\text{NNCIT}$: a non-parametric permutation test of conditional independence based on $\text{MS}_{0\text{-}\mathcal{\infty}}$ [1], and accelerated with VP-trees [2] and moment matching.

## If you are only going to use ${k}\text{NNCIT}$

For this you only need to compile from scratch the PyBNesian module provided in this repository. Detailed instructions can be found on its [README.md](/pybnesian/README.md). After installing the "Build from source dependencies" (Python >= 3.6, C++17 compatible compiler & CMake), you have to run:

```
git clone https://github.com/JuanFPR-UPM/HSPBN_KNN_VPTREE
cd pybnesian
python setup.py install
```

### Usage example

```python
import numpy as np
import pandas as pd
from pybnesian import MixedKMutualInformation

rng = np.random.default_rng(42)
num_samples = 1000
num_categories = 5
m_perm = 150

# X and Y are unconditionally dependent
X = rng.integers(0, num_categories, size=num_samples)
Y = rng.uniform(X, X + 2.0)

Z_cont = rng.standard_normal((num_samples, 1))
Z_disc = np.zeros((num_samples, 2), dtype=int)
for d in range(2):
    probs = rng.dirichlet(np.ones(num_categories))
    Z_disc[:, d] = rng.choice(num_categories, size=num_samples, p=probs)

df = pd.DataFrame([X, Y] + Z_cont.T.tolist() + Z_disc.T.tolist()).T
for i in range(len(df.columns)):
    # PyBNesian accepts discrete variables as pandas.category
    if i == 0 or i > 2:
        col = df.columns[i]
        df[col] = df[col].astype(str).astype('category')
x = str(df.columns[0])
y = str(df.columns[1])
z = [str(x) for x in df.columns[2:]]

ci_test = MixedKMutualInformation(df=df, k=num_samples//10, scaling="normalized_rank", adaptive_k=True, shuffle_neighbors=5, samples=m_perm, gamma_approx=True, seed=42)

print("Unconditional MI")
print(ci_test.mi(x, y))
print("Conditional MI")
print(ci_test.mi(x, y, z))
print("Unconditional independence test p-value")
print(ci_test.pvalue(x, y))
print("Conditional independence test p-value")
print(ci_test.pvalue(x, y, z))

```
### Hyper-parameters


| Name | Type | Description |
|------|------|-------------|
| `df` | DataFrame | Data on which to calculate the independence tests. |
| `k` | int | Number of neighbors in the k-NN model used to estimate the mutual information. |
| `samples` | int | Number of conditional permutations. |
| `shuffle_neighbors` | int | Number of neighbors used for the conditional permutation. |
| `scaling` | str | Transformation for continuous variables into the `[0,1]` range. Options: `"normalized_rank"` or `"min_max"`. |
| `gamma_approx` | bool | If `True`, the p-value is approximated by fitting a gamma distribution to the first three moments of the permutation statistics. If `False`, the crude Monte Carlo estimation is computed.|
| `adaptive_k` | bool | If `True`, upper-bounds both `k` and `shuffle_neighbors` with the minimum discrete configuration size (as in [1]). If `False`, allows the k-NN model distance to cross boundaries between discrete values.|
| `tree_leafsize` | int | Maximum size for VP-Tree leaves before switching to brute force search. |
| `seed` | int or None | Random seed. If `None`, a random seed is generated. |


## If you want to reproduce all the experiments

The experiments involve several libraries that may have conflicting dependencies between them. The following steps should ease their installation.

* For ${k}\text{NNCIT}$ (Python/C++) -> PyBNesian - Install as explained above.
* For DGCIT [3] (Java) -> Py-Tetrad 0.1.2 - Need to install JDK-25 and JPype1. More info in their [repository](https://github.com/cmu-phil/py-tetrad/tree/56dd9ec5e0e97230ed037f2c61dd9364e978197a).
```python
pip install JPype1
pip install git+https://github.com/cmu-phil/py-tetrad@56dd9ec5e0e97230ed037f2c61dd9364e978197a
```
* For CGIT [4] (R) -> bnlearn (Currently version 5.1).
```r
install.packages("bnlearn")
```
* For MMCIT [5] (R) -> MXM 1.5.5 - Deprecated in CRAN, provided in this repo as a compressed file with minor changes. Original package [here](https://cran.r-project.org/src/contrib/Archive/MXM/).
```r
install.packages("./external/MXM/MXM_1.5.5.tar.gz", repos = NULL, type="source")
```
* For MRCIT [6] (Python/R) -> Essential files and dependencies in ./external/AAAI2022_HCM/, install as explained in their [README.md](/external/AAAI2022_HCM/README.md). Original repository [here](https://github.com/DAMO-DI-ML/AAAI2022-HCM).
```r
install.packages("/ossfs/workspace/pai-algo-dev/momentchi2_0.1.5.tar", repos=NULL, type="source")
```
```python
cd external/AAAI2022_HCM/
pip install requirements.txt
```
* For the original $\text{MS}_{0\text{-}\mathcal{\infty}}$ implementation [1] (Python) -> Essential files and dependencies in ./external/cmiknnmixed/, install as explained in their [README.md](external/cmiknnmixed/README.md).
Note that this newer version of the estimator was retrieved from their Python library [Tigramite](https://github.com/jakobrunge/tigramite/blob/0d0c321bddfde5832fc97b8a21ae7bc2220a652d/tigramite/independence_tests/cmiknn_mixed.py). Original repository [here](https://github.com/oanaipopescu/cmiknnmixed).
```python
cd external/cmiknnmixed/
pip install requirements.txt
```
* For LCIT [7] (Python) -> Essential files and dependencies in ./external/LCIT/, install as explained in their [README.md](/external/LCIT/README.md). Original repository [here](https://github.com/baosws/LCIT).
```python
pip install numpy pandas scikit-learn scipy torch==1.12 pytorch-lightning==1.5.3
```

* Additionally, the experiments have dependencies that must be installed:
```python
pip install -r requirements.txt
```

The recommended and working environments for the experiments in the paper have been: 
* A single Python 3.10.0 installation for executing LCIT + PyTorch.
* A second Python 3.13.5 installation for everything else.
* R version 4.5.1 and latest Java LTS release JDK-25.

### Dataset availability
Although all datasets are either publicly available or reproducible, we provide all the experimental data and learned models as they may take some time to be generated. They can be downloaded as a .zip file [here](https://cig.fi.upm.es/wp-content/uploads/experiments.zip), and must be decompressed under the [experiments/](/experiments) directory of this project.

## Author

    - Juan Fernández del Pozo Romero j.fernandezdelpozo@upm.es

## License

This project is licensed under the MIT License - see the LICENSE file for details.

For additional modules under the /external/ directory - see the THIRD_PARTY_LICENSES file.


## Acknowledgments

The experiments in this repository were inspired by: 
* [mCMIkNN](https://github.com/hpi-epic/mCMIkNN) [8]
* [HSPBNs](https://github.com/davenza/HSPBN-Experiments) [9]

Additionally, this repository extends the PyBNesian implementation in [https://repo.hca.bsc.es/gitlab/aingura-public/pybnesian.git](https://repo.hca.bsc.es/gitlab/aingura-public/pybnesian.git), with authors:

    - Erik Blázquez erik.blazquez@bsc.es
    - Gaizka Virumbrales gvirumbrales@ainguraiiot.com
    - Filippo Mantovani filippo.mantovani@bsc.es
    - Javier Diaz jdiaz@ainguraiiot.com

Being all extensions of the original work [10] (https://github.com/davenza/PyBNesian) authored by:

    - David Atienza datienza@fi.upm.es

## References

[1] O.-I. Popescu, A. Gerhardus, M. Rabel, J. Runge. Non-parametric conditional independence testing for mixed continuous-categorical variables: A novel method and numerical evaluation. In *Proceedings of the 4th Conference on Causal Learning and Reasoning*, PMLR, vol. 275, pp. 406–450, 2025.

[2] P. N. Yianilos. Data structures and algorithms for nearest neighbor search in general metric spaces. In *Proceedings of the 4th Annual ACM-SIAM Symposium on Discrete Algorithms*, Society for Industrial and Applied Mathematics, pp. 311–321, 1993.

[3] B. Andrews, J. Ramsey, G. F. Cooper. Learning high-dimensional directed acyclic graphs with mixed data-types. In *Proceedings of Machine Learning Research*, PMLR, vol. 104, pp. 4–21, 2019.

[4] M. Scutari. Learning Bayesian networks with the bnlearn R package. *Journal of Statistical Software*, vol. 35, no. 3, pp. 1–22, 2010.

[5] M. Tsagris, G. Borboudakis, V. Lagani, I. Tsamardinos. Constraint-based causal discovery with mixed data. *International Journal of Data Science and Analytics*, vol. 6, no. 1, pp. 19–30, 2018.

[6] Y. Li, R. Xia, C. Liu, L. Sun. A hybrid causal structure learning algorithm for mixed-type data. In *Proceedings of the AAAI Conference on Artificial Intelligence*, AAAI Press, vol. 36, no. 7, pp. 7435–7443, 2022.

[7] B. Duong, T. Nguyen. Normalizing flows for conditional independence testing. *Knowledge and Information Systems*, vol. 66, no. 1, pp. 357–380, 2024.

[8] J. Huegle, C. Hagedorn, R. Schlosser. A kNN-Based non-parametric conditional independence test for mixed data and application in causal discovery. In *Machine Learning and Knowledge Discovery in Databases: Research Track*, Springer, pp. 541–558, 2023.

[9] D. Atienza, P. Larrañaga, C. Bielza. Hybrid semiparametric Bayesian networks. *TEST*, vol. 31, no. 2, pp. 299–327, 2022.

[10]  D. Atienza, C. Bielza, P. Larrañaga. PyBNesian: An extensible python package for Bayesian networks. *Neurocomputing*, vol. 504, pp. 204–209, 2022.