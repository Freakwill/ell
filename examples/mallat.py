#!/usr/bin/env python3

# Mallat algorithm

from ell import *
import numpy as np
from ell.utils import max0

def draw(tree, fig):
    # draw the tree type of coefs of mallat decomp.
    import matplotlib.gridspec as gridspec
    level = tree._level
    gs = gridspec.GridSpec(2**level, 2**level, figure=fig)
    # gs.update(left=0.05, right=0.45, top=0.02, bottom=0.01, wspace=0.05)
    def _draw(tree, level, gs, offset=0):
        if level == 0:
            n = tree.get_node(offset)
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(n.data.minmaxmap().to_image(resize=True))
            ax0.axis('off')
            ax0.set_title(n.tag)
        else:
            n11, n12, n13 = (tree.get_node((offset+1, k)) for k in range(1,4))
            ax11 = fig.add_subplot(gs[:2**(level-1), 2**(level-1):2**level])
            ax11.imshow(n11.data.to_image(resize=True))
            ax11.axis('off')
            ax11.set_title(n11.tag)
            ax12 = fig.add_subplot(gs[2**(level-1):2**level, :2**(level-1)])
            ax12.imshow(n12.data.to_image(resize=True))
            ax12.axis('off')
            ax12.set_title(n12.tag)
            ax13 = fig.add_subplot(gs[2**(level-1):2**level, 2**(level-1):2**level])
            ax13.imshow(n13.data.to_image(resize=True))
            ax13.axis('off')
            ax13.set_title(n13.tag)
            _draw(tree, level=level-1, gs=gs, offset=offset+1)
    _draw(tree, level, gs)
    fig.suptitle("Mallat Algo. for Tensor Wavelets")

level = 2
_filter = Filter.from_name('db2')
a = ImageRGB.open('lenna.jpg')
t = a.mallat_tensor(_filter, level=level)

import matplotlib.pyplot as plt
fig = plt.figure(constrained_layout=True)
draw(t, fig)
plt.show()
