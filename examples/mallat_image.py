#!/usr/bin/env python3

from ell import *
import numpy as np
from ell.utils import max0

from PIL import Image

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


def draw(tree, fig):
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

im = Image.open('lenna.jpg')

level = 2
_filter = Filter.from_name('db5')

a = ImageRGB.from_image(im)
t = a.mallat_tensor(_filter, level=level)

print(t)

# ret= mallat_rec(t, dual_filter=_filter)
# ret.to_image(resize=True).show()

import matplotlib.pyplot as plt
fig = plt.figure(constrained_layout=True)
draw(t, fig)
plt.show()
