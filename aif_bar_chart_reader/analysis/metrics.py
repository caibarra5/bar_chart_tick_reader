import numpy as np


def expectation_and_variance(p, values):
    p = np.asarray(p, dtype=float)
    mean = np.sum(p * values)
    var = np.sum(p * (values - mean) ** 2)
    return float(mean), float(var)


def entropy(p, eps=1e-16):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))
