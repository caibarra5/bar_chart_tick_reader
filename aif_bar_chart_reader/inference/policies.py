# filename: inference/policies.py
import numpy as np


def build_policies(Ncoarse):
    policies = []
    n_factors = 5

    for q in range(Ncoarse):
        pol = np.zeros((1, n_factors), dtype=int)
        pol[0, 2] = 0
        pol[0, 3] = q
        policies.append(pol)

    for q in range(Ncoarse):
        pol = np.zeros((1, n_factors), dtype=int)
        pol[0, 2] = 1
        pol[0, 3] = q
        policies.append(pol)

    for rep in range(Ncoarse + 1):
        pol = np.zeros((1, n_factors), dtype=int)
        pol[0, 2] = 2
        pol[0, 4] = rep
        policies.append(pol)

    return policies
