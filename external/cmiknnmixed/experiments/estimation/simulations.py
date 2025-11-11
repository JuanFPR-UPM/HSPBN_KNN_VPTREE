from __future__ import print_function
import numpy as np
import pandas as pd

from sklearn import datasets


# partly adapted from Mesner and Shalizi (2019)

def corrUnif(n, dim, num_classes=3, trials=5):
    disc_unif = np.random.choice(list(range(num_classes)), size=n)
    cont_unif = list()
    for itr in disc_unif:
        cont_unif.append(np.random.uniform(itr, itr + 2, 1)[0])
    dat = {'x': disc_unif, 'y': cont_unif}
    for itr in range(dim):
        dat[str(itr)] = np.random.binomial(trials, 0.5, n)
    df = pd.DataFrame(data=dat)
    info = np.log(num_classes) - (num_classes - 1) * np.log(2) / num_classes
    name = 'Variables mixed and dependent'
    type_array = np.zeros((n, dim + 2))
    type_array[:, 0] = 1
    type_array[:, 2:] = 1
    xyz = np.asarray([0, 1] + [2] * dim)
    return df, type_array, info, name, xyz


def run_corrUnif_ZMADG_1d_Ber(n):
    num_classes = 5
    # the number of trials doesn't matter
    trials = 1
    d = 1
    return corrUnif(n, dim=d,
                    num_classes=num_classes,
                    trials=trials)


def run_corrUnif_ZMADG_nd_Ber(n):
    num_classes = 5
    # the number of trials doesn't matter
    trials = 1
    d = 3
    return corrUnif(n, dim=d,
                    num_classes=num_classes,
                    trials=trials)


def cindep(n, dim, num_classes=0):
    beta = 10
    p = 0.5
    exp_dist = np.random.exponential(scale=1. / beta, size=n)
    pois_dist = list()
    binom_dist = list()
    for itr in exp_dist:
        pois = np.random.poisson(itr, size=1)[0]
        pois_dist.append(pois)
        binom_dist.append(np.random.binomial(pois + num_classes, p, size=1)[0])
    dat = {'exp': exp_dist, 'binom': binom_dist, 'pois': pois_dist}
    for itr in range(dim - 1):
        dat[str(itr)] = np.random.normal(0, 1, n)
    df = pd.DataFrame(data=dat)
    info = 0.
    name = 'Variables Mixed and Conditionally Independent'
    type_array = np.zeros((n, dim + 2))
    type_array[:, 1] = 1
    type_array[:, 2] = 1
    xyz = np.asarray([0, 1] + [2] * dim)
    return df, type_array, info, name, xyz


def run_cindep_MS(n):
    d = 1
    return cindep(n, d)


def run_cindep_nd(n):
    d = 3
    return cindep(n, d)


def confounder_normal_std1(n, num_classes=2):
    # d -> normal
    #   -> normal
    z = np.random.choice(list(range(num_classes)), size=n)
    x_vals, y_vals = [], []
    for itr in z:
        x = np.random.normal(itr, 1, 1)[0]
        y = np.random.normal(itr, 1, 1)[0]
        x_vals.append(x)
        y_vals.append(y)
    data = {'x': x_vals, 'y': y_vals, 'z': z}
    df = pd.DataFrame(data=data)
    info = 0.
    name = 'Confounder discrete to normal'
    type_array = np.zeros((n, 3))
    type_array[:, 2] = 1
    xyz = np.asarray([0, 1, 2])
    return df, type_array, info, name, xyz


def run_confounder_normal_ZMADG(n):
    num_classes = 9
    return confounder_normal_std1(n, num_classes=num_classes)


def confounder(n, num_classes=2):
    # d -> unif
    #   -> unif
    z = np.random.choice(list(range(num_classes)), size=n)
    x_vals, y_vals = [], []
    for itr in z:
        x = np.random.uniform(0, itr, 1)[0]
        y = np.random.uniform(itr, itr + 1, 1)[0]
        x_vals.append(x)
        y_vals.append(y)
    data = {'x': x_vals, 'y': y_vals, 'z': z}
    df = pd.DataFrame(data=data)
    info = 0.
    name = 'Confounder discrete to continuous uniform'
    type_array = np.zeros((n, 3))
    type_array[:, 2] = 1
    xyz = np.asarray([0, 1, 2])
    return df, type_array, info, name, xyz


def confounder_from_uniform(n, num_classes=2):
    # unif -> unif
    #      -> disc
    z = np.random.choice(list(range(num_classes)), size=n)
    x_vals, y_vals = [], []
    for itr in z:
        x = np.random.normal(0.1 * itr, itr + 1, 1)[0]
        y = np.random.uniform(0.2 * itr, itr + 1, 1)[0]
        x_vals.append(x)
        y_vals.append(y)
    data = {'x': x_vals, 'y': y_vals, 'z': z}
    df = pd.DataFrame(data=data)
    info = 0.
    name = 'Confounder discrete to normal'
    type_array = np.zeros((n, 3))
    type_array[:, 2] = 1
    xyz = np.asarray([0, 1, 2])
    return df, type_array, info, name, xyz


def mi_multivar_mixture(n, p=0.3):
    z = np.random.binomial(1, p, size=n)
    mean = [0, 0]
    cov = [[1, 0.6], [0.6, 1]]
    x_vals, y_vals = [], []
    x_types, y_types = [], []
    for itr in z:
        if itr == 0:
            xy = np.random.multivariate_normal(mean, cov, size=1)
            x_vals.append(xy[0, 0])
            y_vals.append(xy[0, 1])
            x_types.append(0)
            y_types.append(0)
        else:
            x = np.random.choice(list(range(5)), size=1)[0]
            y = np.random.uniform(x, x + 2, size=1)[0]
            x_vals.append(x)
            y_vals.append(y)
            x_types.append(1)
            y_types.append(0)

    data = {'x': x_vals, 'y': y_vals, 'z': z}
    df = pd.DataFrame(data=data)
    info = (1 - p) * (-np.log(1 - 0.36) / 2) + p * (np.log(5) - 4 / 5 * np.log(2))
    name = 'Z-dependent clusters with mixture data'
    z_types = np.ones((n, 1))
    x_types = np.expand_dims(np.asarray(x_types), -1)
    y_types = np.expand_dims(np.asarray(y_types), -1)
    type_array = np.concatenate([x_types, y_types, z_types], axis=-1)
    xyz = np.asarray([0, 1, 2])

    return df, type_array, info, name, xyz
