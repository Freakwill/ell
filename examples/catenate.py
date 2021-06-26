#!/usr/bin/env python3

from ell import *
from ell.utils import max0
import numpy as np

level = 6
_filter = Filter.from_name('db6')

from PIL import Image

a = Image.open('apple2.jpeg')
b = Image.open('orange2.jpeg').resize(a.size)

# a = a.crop((0, 0, a.size[0]-1, a.size[1]))
# a.save('apple2.jpeg')
# raise

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

def tbmerge(im1, im2, loc=None):
    '''Merge top part of `im1` and bottum part of `im2`

    See also lrmerge
    '''
    xsize1, ysize1 = im1.size
    xsize2, ysize2 = im2.size
    if loc is None:
        loc = ysize1 // 2
    elif loc <1:
        loc = int(ysize1 * loc)
    box1 = (0, 0, xsize1, loc)
    im1 = im1.crop(box1)
    im2.paste(im1, box1)
    return im2

c = tbmerge(a, b)
a = ImageRGB.from_image(a)
b = ImageRGB.from_image(b)

_, Lb, _, _ = b.pyramid(_filter, level=level)

def key(k_L):
    k, L = k_L
    def _key(d):
        k = d.shape[0] // 2+1
        dd = np.asarray(d)
        r = min(max(int(k * 0.05), 2), 10)
        dd[k+r:,:] = np.ndarray.__getitem__(L, np.s_[k+r:,:])
        dd[k-r:k+r,:] = 0.5*dd[k-r:k+r,:]+0.5*np.ndarray.__getitem__(L, np.s_[k-r:k+r,:])
        r //=2
        if k<level:
            dd[k-r:k+r,:] = max0(dd[k-r:k+r,:], 100)
        else:
            dd[k-r:k+r,:] = max0(dd[k-r:k+r,:], 50)
        d = d.__class__(dd, min_index=d.min_index, max_index=d.max_index)
        return d
    return _key

_, _, _, gauss = a.pyramid(_filter, level=level, op=tuple(map(key, list(zip(range(level+1), Lb)))), resize=True)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('Image Mosaics')
ax = fig.subplots(1,2)
ax[0].imshow(c)
ax[1].imshow(gauss[0].to_image(masked=True))
ax[0].set_title('catenated Image')
ax[0].axis('off')
ax[1].set_title('Mosaic Image')
ax[1].axis('off')
plt.show()

