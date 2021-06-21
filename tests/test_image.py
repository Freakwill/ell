#!/usr/bin/env python3

from ell import *
import numpy as np
from PIL import Image

_filter = Filter.from_name('db4')

def test_resize():
    chennal=0
    im = Image.open('../lenna.jpg')
    c = ImageRGB.from_image(im)
    d=c.resize(minInd=(-100,-100), maxInd=(100,100))
    d.to_image()
    assert True

def test_quantize():
    im = Image.open('../lenna.jpg')
    im = ImageRGB.from_image(im)
    d = im.quantize(128)
    d.to_image()
    assert True

def test_convolve():
    im = Image.open('../lenna.jpg')
    im = ImageRGB.from_image(im)
    d = (im @ _filter.H).D
    print(f"{d:i}, {d.shape}")
    assert True

test_convolve().show()