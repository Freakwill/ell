#!/usr/bin/env python3

"""Just for convolution
"""

from ell import *
from utils import *

def test_conv1d():
    a = Ell2d(np.ones((3,4)), min_index=-1)
    b = Ell1d([-1/2,0,1/3])
    c = a.conv1d(b, axis=0).conv1d(b, axis=1)
    d = a @ b.tensor()

    assert same_size(c,d)

    a = MultiEll2d(np.ones((3,4,2)), min_index=-1)
    b = Ell1d([-1/2,0,1/3])
    c = a.conv1d(b, axis=0).conv1d(b, axis=1)
    d = a @ b.tensor()

    assert same_size(c,d, True)

test_conv1d()