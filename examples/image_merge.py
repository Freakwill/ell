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


im1 = Image.open('car1.jpg')
im2 = Image.open('car2.jpg')
im2 = im2.resize(im1.size)
level = 2
_filter = Filter.from_name('db5')

a = ImageRGB.from_image(im1)
t1 = a.mallat_tensor(_filter, level=level)

b = ImageRGB.from_image(im2)
t2 = b.mallat_tensor(_filter, level=level)

def merge(t1, t2):
    def _merge(a, b, w, threshold=0.1):
        """Merge two 2D array according to local energy.
        
        [description]
        
        Arguments:
            a, b {2d array} -- 2D arraies, such as images
            w {array} -- weight
            threshold {number0~1}
        
        Returns:
            2d array
        """

        # define local energy with weight `w`
        Ea = a**2 @ w
        Eb = b**2 @ w
        Eab = a*b @ w
        m = 2*Eab /(Ea+Eb)
        w = max0(m-threshold)/(2*(1-threshold))
        w=w.resize_as(a)
        return np.max([a, b], axis=0) - w * np.abs(a-b)
    
    w = Ell1d([0.05,0.1,0.05,0.6,0.05,0.1,0.05])
    for n1, n2 in zip(t1.all_nodes_itr(), t2.all_nodes_itr()):
        if n1.data is not None:
            n1.data = _merge(n1.data, n2.data, w)
    return t1

t = merge(t1, t2)

ret= mallat_rec(t, dual_filter=_filter)
ret.resize_as(a).to_image().show(title='Merge of Images')
