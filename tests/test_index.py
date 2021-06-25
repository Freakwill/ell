from ell import *
import numpy as np


def test_index():
    o=Ell1d.zero(min_index=-10, max_index=10)
    o.set_min_index(-5)
    assert o.min_index == -5 and o.max_index == 15
    o=Ell1d.zero(min_index=-10, max_index=10)
    o.inc_min_index(5)
    assert o.min_index == -5 and o.max_index == 15

def test_index_2d():
    o=Ell1d.zero(min_index=-10, max_index=10).tensor()
    o.set_min_index((-5,-4))
    assert o.min_index == (-5, -4) and o.max_index == (15, 16)

def test_index_m2d():
    o=MultiEll2d.zero(min_index=-10, max_index=10, n_values=3)
    o.set_min_index((-5,-4))
    assert o.min_index == (-5, -4) and o.max_index == (15, 16)

def test_index_filter():
    f = Filter.from_name('db3')
    assert f.min_index == 0
    f = Filter.from_name('db3').tensor()
    assert f.min_index == (0,0)

def test_common_index():

    a = Ell1d([1,2,3,4])
    b = Ell1d([2,3,4,5,5,6], min_index=-3)
    mi, ma = common_index(a.index_pair, b.index_pair)
    assert mi == b.min_index and ma == a.max_index
    a = Ell1d([1,2,3,4])
    b = Ell1d([2,3,4,5,5,6], min_index=-3)
    a = a.tensor()
    b = b.tensor()
    mi, ma = common_index(a.index_pair, b.index_pair)
    assert np.all(np.equal(mi, b.min_index)) and np.all(np.equal(ma, a.max_index))


test_common_index()
