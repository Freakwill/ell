#!/usr/bin/env python3

from ell import *

level = 7
ds = 2**(level/2) * compose(d_filter, level=level)
ds2d = ds.tensor()
dw = 2**.5 * d_filter.check().up_sample(2**level) @ ds
dw2d = dw.tensor()

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

ax=fig.add_subplot(221)
ds.plot(axes=ax, irange=(0, 3))
ax.set_title('Daubechies Scaling Function')
ax=fig.add_subplot(222)
dw.plot(axes=ax, irange=(-2,1))
ax.set_title('Daubechies Wavelet')
ax=fig.add_subplot(223, projection='3d')
ds2d.plot(axes=ax, irange=((0,3),(0,3)), cmap=cm.coolwarm)
ax.set_title('2D Daubechies Scaling Function')
ax=fig.add_subplot(224, projection='3d')
dw2d.plot(axes=ax, irange=((-2,1),(-2,1)), cmap=cm.coolwarm)
ax.set_title('2D Daubechies Wavelet')
plt.show()