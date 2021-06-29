#!/usr/bin/env python3

"""Sobel edge detection
"""

from ell import *

s1 = Ell1d([-1,-2,-1])
s2 = Ell1d([1,0,-1])

im = ImageGray.open('lenna.jpg')
s_im = im.conv1d(s1, axis=0).conv1d(s2, axis=1)

# <=> s_im = im @ s1.tensor(s2) but might more slow

(s_im).to_image().show()
