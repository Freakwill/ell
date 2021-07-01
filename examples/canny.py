#!/usr/bin/env python3

"""Canny Algo. for Edge Detection

See https://en.wikipedia.org/wiki/Canny_edge_detector
"""

import numpy as np
from ell import *

_setitem = np.ndarray.__setitem__

im = ImageGray.open('src/lenna.jpg')

# Gaussian filter
gm = 1/ 159 * Ell2d([[2, 4, 5, 4, 2],
[4, 9, 12, 9, 4],
[5, 12, 15, 12, 5],
[4, 9, 12, 9, 4],
[2, 4, 5, 4, 2]])
im1 = im @ gm

# Finding the intensity gradient of the image (by Sobel)
s1 = Ell1d([-1,-2,-1])
s2 = Ell1d([1,0,-1])
Gx = im.conv1d(s1, axis=0).conv1d(s2, axis=1)
Gy = im.conv1d(s2, axis=0).conv1d(s1, axis=1)
G = np.hypot(Gx, Gy).resize_as(im1)
Theta = np.abs(np.arctan2(Gy, Gx)).resize_as(im1)
Theta = np.round(Theta / (np.pi/4))

_G = np.asarray(G)
_Theta = np.asarray(Theta)

# Lower bound cut-off suppression
im2 = im1.copy()
r, c= _G.shape
for i in range(1,r-1):
    for j in range(1,c-1):
        if _Theta[i,j]==0 or _Theta[i,j]==4:
            if not (_G[i, j] > _G[i, j-1] and _G[i, j] > _G[i, j+1]):
                _setitem(im2, (i,j), 0)
        elif _Theta[i,j]==1:
            if not (_G[i, j] > _G[i-1, j+1] and _G[i, j] > _G[i+1, j-1]):
                _setitem(im2, (i,j), 0)
        elif _Theta[i,j]==2:
            if not (_G[i, j] > _G[i-1, j] and _G[i, j] > _G[i+1, j]):
                _setitem(im2, (i,j), 0)
        elif _Theta[i,j]==3:
            if not (_G[i, j] > _G[i+1, j+1] and _G[i, j] > _G[i-1, j-1]):
                _setitem(im2, (i,j), 0)


# Double threshold
weak, strong = 25, 75

im3 = im2.copy()
np.putmask(im3, _G<weak, 0)

# Edge tracking by hysteresis

im4 = im3.copy()
for i in range(1,r-1):
    for j in range(1,c-1):
        if _G[i, j]<weak:
            _setitem(im4, (i,j), 0)
        elif np.all(_G[i-1:i+2,j-1:j+2]<strong):
            _setitem(im4, (i,j), 0)


import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('Canny Algo. for Edge Detection')
ax = fig.subplots(2,3)
for i in range(2):
    for j in range(3):
        ax[i,j].axis('off')
ax[0,0].imshow(im.to_image(), cmap='gray')
ax[0,0].set_title('Original image')
ax[0,1].imshow(im1.to_image(), cmap='gray')
ax[0,1].set_title('Gaussian filter')
ax[0,2].imshow((G.translate() + im1).to_image(), cmap='gray')
ax[0,2].set_title('Intensity gradient')
ax[1,0].imshow(im2.to_image(), cmap='gray')
ax[1,0].set_title('Non-maximum suppression')
ax[1,1].imshow(im3.to_image(), cmap='gray')
ax[1,1].set_title('Double thresholding')
ax[1,2].imshow(im4.to_image(), cmap='gray')
ax[1,2].set_title('Hysteresis')
plt.show()

