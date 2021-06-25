from ell import *
import numpy as np

_filter = Filter.from_name('db4')

def check(a, mi, ma, shape):
    return np.all(a.min_index == mi) and np.all(a.max_index == ma) and np.all(np.equal(a.shape, shape))


def test_reduce_1d():
    signal = Ell1d(np.sin(np.linspace(0,3,100)))
    reduced = signal.reduce(_filter)
    min_index = signal.min_index - _filter.max_index
    max_index = signal.max_index - _filter.min_index
    assert np.all(np.abs(reduced.min_index) == np.abs(min_index) // 2)
    assert np.all(np.abs(reduced.max_index) == np.abs(max_index) // 2)

def test_filter_1d():
    signal = Ell1d(np.sin(np.linspace(0,3,10)))
    reduced = signal.filter(_filter)
    assert reduced.min_index==-6 and reduced.max_index==15

def test_reduce_2d():
    signal = Ell1d(np.sin(np.linspace(0,3,10))).tensor()
    reduced = signal.reduce(_filter)
    min_index = np.subtract(signal.min_index, _filter.max_index)
    max_index = np.subtract(signal.max_index, _filter.min_index)
    assert np.all(np.abs(reduced.min_index) == np.abs(min_index) // 2) and np.all(np.abs(reduced.max_index) == np.abs(max_index) // 2)
    signal = Ell1d(np.sin(np.linspace(0,3,10))).tensor()
    reduced1 = signal.reduce(_filter, axis=0).reduce(_filter, axis=1)
    reduced2 = signal.reduce(_filter)
    reduced3 = signal.reduce(_filter.tensor())
    print(f"{reduced1:s}, {reduced2:s}, {reduced3:s}")
    assert reduced1.shape == reduced2.shape

def test_filter_2d():
    signal = Ell1d(np.sin(np.linspace(0,3,10))).tensor()
    reduced = signal.filter(_filter)
    assert reduced.min_index==(-6,-6) and reduced.max_index==(15,15)

def test_filter_m2d():
    signal = MultiEll2d(np.ones((10,20,3)))
    expanded = signal.filter(_filter)
    assert expanded.min_index==(-6,-6) and expanded.max_index==(15,25)
    reduced = signal.reduce(_filter)
    expanded= reduced.expand(_filter)

test_filter_m2d()
