from ell import *


def check(a, mi, ma, shape):
    return np.all(a.min_index == mi) and np.all(a.max_index == ma) and np.all(np.equal(a.shape, shape))


def test_copy():
    b = d_filter
    a = b.copy()
    a = a.resize(min_index=-2, max_index=2)
    assert a.min_index == -2 and a.max_index == 2 and b.min_index ==0 and b.max_index == 3

def test_copy2d():
    a = Ell1d([1,2,3,4]).tensor()
    cpy = a.copy()
    c = cpy @ d_filter
    assert np.all(np.equal(a.min_index, 0)) and np.all(np.equal(a.max_index, 3))

def test_sample():
    b = d_filter.tensor()
    a=b.up_sample()
    assert np.all(a.min_index == b.min_index*2) and np.all(a.max_index == b.max_index*2)


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
    assert a-b==c
    a = 2
    b = Ell1d([2,3,4,5,5,6], min_index=-3)
    c = Ell1d([0, -1, -2, -3, -3, -4], min_index=-3)
    assert a-b==c

def test_common_index():

    a = Ell1d([1,2,3,4])
    b = Ell1d([2,3,4,5,5,6], min_index=-3)
    a = a.tensor()
    b = b.tensor()
    mi, ma = common_index(a.index_pair, b.index_pair)
    assert np.all(np.equal(mi, b.min_index)) and np.all(np.equal(ma, a.max_index))


def test_fillzeros():
    a = Ell1d([1,2,3,4]).tensor()
    a= a.fill_zero(n_zeros=-3, axis=0)
    a= a.fill_zero(n_zeros=-3, axis=1)
    assert np.all(np.equal(a.shape, (7,7)))


def test_resize():
    a = Ell1d([1,2,3,4,5,6]).tensor()
    a = a.resize(np.array([-3,-3]), np.array([3,3]))
    assert check(a, (-3,-3), (3,3), (7,7))
