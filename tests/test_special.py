from ell import *
import numpy as np


def test_zero():
    o=Ell1d.zero(min_index=-10, max_index=10)
    o[::2]=2
    a=Ell1d.zero(min_index=-5, max_index=5)
    a[:]=o[::2]
    assert True

def test_zero_2d():
    o=Ell2d.zero(min_index=(-10,-10), max_index=(10,10))
    o[::2,::2]=2
    a=Ell2d.zero(min_index=(-5,-5), max_index=(5,5))
    a[:,:]=o[::2,::2]
    assert True

def test_unit_2d():
    o=Ell2d.unit(min_index=(-10,-10), max_index=(10,10))
    o[::2,::2]=2
    a=Ell2d.unit(min_index=(-5,-5), max_index=(5,5))
    a[:,:]=o[::2,::2]
    assert True

test_unit_2d()