from ell import *
import numpy as np
from utils import *

_filter = Filter.from_name('db2')

def test_copy():
    b = Ell1d([0,1,2,3])
    assert hasattr(b, 'min_index')
    a = b.copy()
    assert hasattr(a, 'min_index')
    b = _filter
    assert hasattr(b, 'min_index')
    a = b.copy()
    assert hasattr(a, 'min_index')
    a = a.resize(min_index=-2, max_index=2)
    assert a.min_index == -2 and a.max_index == 2 and b.min_index ==0 and b.max_index == 3

def test_copy2d():
    a = Ell1d([1,2,3,4]).tensor()
    cpy = a.copy()
    c = cpy @ _filter
    assert (a.min_index, a.max_index) == ((0,0), (3,3))

def test_up_sample():
    b = _filter.tensor()
    a = b.up_sample()
    assert equal(a.shape, np.multiply(b.shape, 2)-1)
    assert a.min_index == tuple(np.multiply(b.min_index, 2)) and equal(a.max_index, np.multiply(b.max_index, 2))

def test_up_sample_m2d():
    b = MultiEll2d(np.ones((4,5,3)))
    a = b.up_sample(step=2)
    assert equal(a.shape, np.multiply(b.shape, 2)-1)
    assert equal(a.min_index, np.multiply(b.min_index, 2)) and equal(a.max_index, np.multiply(b.max_index, 2))

def test_up_sample_m2d():
    b = MultiEll2d(np.ones((4,5,3)))
    a = b.up_sample(step=2)
    assert equal(a.shape, np.multiply(b.shape, 2)-1)
    assert equal(a.min_index, np.multiply(b.min_index, 2)) and equal(a.max_index, np.multiply(b.max_index, 2))
    b = MultiEll2d(np.ones((4,5,3)))
    a = b.up_sample(step=2, axis=1)
    assert equal(a.shape[1], np.multiply(b.shape[1], 2)-1)
    assert equal(a.min_index[1], np.multiply(b.min_index[1], 2)) and equal(a.max_index[1], np.multiply(b.max_index[1], 2))


def test_down_sample_m2d():
    b = MultiEll2d(np.ones((4,5,3)))
    a = b.down_sample(step=2, axis=0).down_sample(step=2, axis=1)
    assert equal(a.min_index, np.floor(np.divide(b.min_index, 2)))

    a = b.down_sample(step=2, axis=0)
    assert a.min_index[0] == b.min_index[0] // 2
    a = a.down_sample(step=2, axis=1)
    assert a.min_index[1] == b.min_index[1] // 2


def test_alt():
    min_index = np.array([-2,3])
    max_index = np.array([4,6])
    res = np.array([[-1,  1, -1,  1, -1,  1, -1],
 [ 1, -1,  1, -1,  1, -1,  1],
 [-1,  1, -1,  1, -1,  1, -1],
 [ 1, -1,  1, -1,  1, -1,  1]])
    assert np.all(altArray(min_index, max_index) == res)

def test_trancate():
    a = Ell1d([1,2,3,4,5,6,-3,-4,-1], min_index=-3)
    a = a.truncate(2, soft=True)
    assert a == Ell1d([0, 0, 1, 2, 3, 4, -1, -2, 0], min_index=-3)

def test_sub():
    a = Ell1d([1,2,3,4])
    b = Ell1d([2,3,4,5,5,6], min_index=-3)
    c = Ell1d([-2.0, -3.0, -4.0, -4.0, -3.0, -3.0, 4.0], min_index=-3)
    assert (a-b == c)
    a = 2
    b = Ell1d([2,3,4,5,5,6], min_index=-3)
    c = Ell1d([0, -1, -2, -3, -3, -4], min_index=-3)
    assert a-b==c


def test_fillzeros():
    a = Ell1d([1,2,3,4]).tensor()
    a= a.fill_zero(n_zeros=-3, axis=0)
    a= a.fill_zero(n_zeros=-3, axis=1)
    assert np.all(np.equal(a.shape, (7,7)))


def test_resize():
    a = Ell1d([1,2,3,4,5,6])
    a = a.resize(-3,3)
    assert check(a, -3,3,(7,))

    a = Ell1d([1,2,3,4,5,6]).tensor()
    assert check(a, (0,0), (5,5), (6,6))
    a = a.resize(np.array([-3,-3]), np.array([3,3]))
    assert check(a, (-3,-3), (3,3), (7,7))


def test_conv():
    a = Ell1d([1,2,3,4,5,6])
    b = Ell1d([1,2,3,4,5,6], min_index=-2)
    c = a @ b
    assert c.min_index==-2 and c.max_index==8

def test_conv_2d():
    a = Ell1d([1,2,3,4,5,6]).tensor()
    b = Ell1d([1,2,3,4,5,6]).refl().tensor()
    c = a @ b
    assert c.min_index==(-5,-5) and c.max_index==(5,5)

def test_sub_2d():
    a = Ell2d(np.ones((5,4)))
    b = Ell2d(np.ones((5,4)), min_index=-2)
    c = b-a
    assert c.min_index == (-2,-2) and c.max_index == (4, 3)

def test_mul_2d():
    a = Ell2d(np.ones((5,4)))
    b = Ell2d(np.ones((5,4)), min_index=-2)
    c = b*a
    assert c.min_index == (-2,-2) and c.max_index == (4, 3)

def test_sub_m2d():
    a = MultiEll2d(np.ones((5,4,3)))
    b = MultiEll2d(np.ones((5,4,3)), min_index=-2)
    c = b-a
    assert c.min_index == (-2,-2) and c.max_index == (4, 3)

def test_resize_m2d():
    a = MultiEll2d(np.ones((5,4,3)), min_index=-2)
    c = a.resize(max_index=(4,3))
    assert c.min_index == (-2,-2) and c.max_index == (4, 3)

def test_dot():
    a = MultiEll2d(np.ones((5,4,3)), min_index=-2)
    assert a.dot(a) == 60
    a = Ell1d(np.ones(5), min_index=-2)
    assert a.dot(a) == 5

def test_make_multi():
    a = Ell2d(np.ones((5,4)), min_index=-2)
    a = MultiEll2d.make_multi(a)
    assert a.ndim == 2

def test_add_x():
    a = Ell2d(np.ones((5,4)), min_index=-2)
    b = MultiEll2d(np.ones((5,4,3)), min_index=0)
    c = a +b
    assert isinstance(c, MultiEll2d) and c.shape == (7, 6)


