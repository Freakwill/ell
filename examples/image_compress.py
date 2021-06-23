#!/usr/bin/env python3

from ell import *
import numpy as np
from ell.utils import max0


def mallat_rec(tree, dual_filter, level=0):
    if tree.depth() == 1:
        n1 = tree.get_node(level+1)
        n11, n12, n13 = (tree.get_node((level+1, k)) for k in range(1,4))
        return n1.data.expand(dual_filter)\
        + n11.data.expand(dual_filter.tensor(dual_filter.check()))\
        + n12.data.expand(dual_filter.check().tensor(dual_filter))\
        + n13.data.expand(dual_filter.check())
    else:
        m = mallat_rec(tree.subtree(level+1), dual_filter, level=level+1)
        n11, n12, n13 = (tree.get_node((level+1, k)) for k in range(1,4))
        return m.expand(dual_filter)\
        + n11.data.expand(dual_filter.tensor(dual_filter.check()))\
        + n12.data.expand(dual_filter.check().tensor(dual_filter))\
        + n13.data.expand(dual_filter.check())

level = 4
_filter = Filter.from_name('db5')

a = ImageRGB.open('lenna.jpg')
t = a.mallat_tensor(_filter, level=level)

def key(d):
    k = np.prod(d.shape)
    k = int(np.round(k *0.05))
    th = [np.ndarray.__getitem__(np.sort(np.asarray(d[:,:,ch]), axis=None), -k) for ch in range(3)]
    return d.truncate(th)
        

t = t.apply_data(key)
# print(t)

ret= mallat_rec(t, dual_filter=_filter)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('Image Compression')
ax = fig.subplots(1,2)
ax[0].imshow(a.to_image())
ax[1].imshow(ret.minmaxmap().resize_as(a).to_image())
ax[0].set_title('Origial Image')
ax[0].axis('off')
ax[1].set_title('Compressed Image (<10%)')
ax[1].axis('off')
plt.show()

