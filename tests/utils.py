"""helpers for test
"""

import numpy as np

def check(a, mi, ma, shape=False):
    if shape is False:
        return a.min_index == mi and a.max_index == ma
    return a.min_index == mi and a.max_index == ma and a.shape == shape

def equal(a, b):
    return np.all(a==b)

def same_size(a, b, shape=False):
    if shape is False:
        return a.min_index == b.min_index and a.max_index == b.max_index
    else:
        return a.min_index == b.min_index and a.max_index == b.max_index and a.shape == b.shape