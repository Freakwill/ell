#!/usr/bin/env python3

from ell import *

filter = Filter.from_name('db4')
f, ws = filter.fourier(N=100)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ws, f)
plt.show()