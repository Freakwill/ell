#!/usr/bin/env python3

"""Test methods about image process

Make sure the existance of the images
"""


from ell import *
import numpy as np

_filter = Filter.from_name('db4')

def test_resize():
    chennal=0
    c = ImageRGB.open('src/lenna.jpg')
    d=c.resize(minInd=(-100,-100), maxInd=(100,100))
    d.to_image()
    assert True

def test_quantize():
    im = ImageRGB.open('src/lenna.jpg')
    d = im.quantize(128)
    d.to_image()
    assert True

def test_convolve():
    im = ImageRGB.open('src/lenna.jpg')
    d = (im @ _filter.H).D
    # print(f"{d:i}, {d.shape}")
    assert True

def test_filter():
    im = ImageRGB.open('src/lenna.jpg')
    rec = (im @ _filter.H).D.U @ _filter

    assert True

def test_rec():
    im = ImageRGB.open('src/lenna.jpg')
    def _f(im, h1, h2=None):
        if h2 is None: h2 = h1
        return (im.conv1d(h1.H, axis=0).conv1d(h2.H, axis=1)).P.conv1d(h1, axis=0).conv1d(h2, axis=1)
    rec = _f(im, _filter) + _f(im, _filter.H) + _f(im, _filter, _filter.H) + _f(im, _filter.H, _filter)

    assert True

def test_rec2():
    im = ImageRGB.open('../src/lenna.jpg')
    def _f(im, h1, h2=None):
        if h2 is None: h2 = h1
        # return (im @ h1.tensor(h2).H).P @ h1.tensor(h2)
        return (im.conv1d(h1.H, axis=0).conv1d(h2.H, axis=1)).P.conv1d(h1, axis=0).conv1d(h2, axis=1)
    im1 = _f(im, _filter)
    rec1 = _f(im1, _filter) + _f(im1, _filter.H) + _f(im1, _filter, _filter.H) + _f(im1, _filter.H, _filter)
    rec2 = rec1 + _f(im, _filter.H) + _f(im, _filter, _filter.H) + _f(im, _filter.H, _filter)
    assert True

def test_rec3():
    im = ImageRGB.open('src/lenna.jpg')
    def _f(im, h1, h2=None):
        if h2 is None: h2 = h1
        f = h1.tensor(h2)
        return im.reduce(f).expand(f)
    im1 = im.reduce(_filter)
    rec1 = _f(im1, _filter) + _f(im1, _filter.H) + _f(im1, _filter, _filter.H) + _f(im1, _filter.H, _filter)
    rec2 = rec1.expand(_filter) + _f(im, _filter.H) + _f(im, _filter, _filter.H) + _f(im, _filter.H, _filter)

    assert True

