#!/usr/bin/env python3

from ell import *
import numpy as np

level = 6
_filter = Filter.from_name('db10')

from PIL import Image

a = Image.open('apple2.jpeg')
b = Image.open('orange2.jpeg').resize(a.size)

def lrmerge(im1, im2, loc=None):
    xsize1, ysize1 = im1.size
    xsize2, ysize2 = im2.size
    if loc is None:
        loc = xsize1 // 2
    elif loc <1:
        loc = int(xsize1 * loc)
    box1 = (0, 0, loc, ysize1)
    im1 = im1.crop(box1)
    im2.paste(im1, box1)
    return im2

c = lrmerge(a, b)
a = ImageRGB.from_image(a)
b = ImageRGB.from_image(b)
c = ImageRGB.from_image(c)

_, Lb, _, _ = b.pyramid(_filter, level=level)

def key(L):
    def _key(d):
        k = d.shape[1] // 2
        dd = np.asarray(d)
        # dd[:,k-100:k] = 0.1*dd[:,k-100:k]+0.9*np.ndarray.__getitem__(L, np.s_[:,k-100:k])
        dd[:,k:] = np.ndarray.__getitem__(L, np.s_[:,k:])
        dd[:,k-1:k+2] = dd[:,k-1:k+2] * (np.abs(dd[:,k-1:k+2])>120)
        d = d.__class__(dd, min_index=d.min_index, max_index=d.max_index)
        return d.truncate(10)
    return _key


_, _, _, gauss = a.pyramid(_filter, level=level, op=tuple(map(key, Lb)), resize=True)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('Image Mosaics')
ax = fig.subplots(1,2)
ax[0].imshow(c.to_image())
ax[1].imshow(gauss[0].minmaxmap().to_image())
ax[0].set_title('catenated Image')
ax[0].axis('off')
ax[1].set_title('Mosaic Image')
ax[1].axis('off')
plt.show()

