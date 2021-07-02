#!/usr/bin/env python3

"""Sobel edge detection
"""

from ell import *

s1 = Ell1d([-1,-2,-1])
s2 = Ell1d([1,0,-1])

im = ImageRGB.open('lenna.jpg')
sim = im.conv1d(s1, axis=0).conv1d(s2, axis=1)

# <=> sim = im @ s1.tensor(s2) but might more slow

sim.imshow()
