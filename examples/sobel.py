#!/usr/bin/env python3

"""Sobel edge detection
"""

from ell import *
import numpy as np

s1 = Ell1d([-1,-2,-1], min_index=-1)
s2 = Ell1d([1,0,-1], min_index=-1)

im = ImageRGB.open('lenna.jpg')
sx = im.conv1d(s1, axis=0).conv1d(s2, axis=1)
sy = im.conv1d(s2, axis=0).conv1d(s1, axis=1)
G = np.hypot(sx, sy)
G = G.minmaxmap()
G = G.truncate(30)
G.resize_as(im).imshow()
