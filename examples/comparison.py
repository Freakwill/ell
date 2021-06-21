#!/usr/bin/env python3


import pandas as pd
from ell import *

code = '161725'
data = pd.read_pickle(code)
y = data['累计净值'].values[::-1]

level = 3
y = Ell1D(y)
ylow = y.filter(Filter.from_name('db4'), level=level, resize=True)

L = len(y)
fy=np.fft.fft(y)
sy = np.sort(abs(fy))
thr = sy[-L//(2**level)]
fy[abs(fy)<thr]=0
fy[fy>thr]=fy[fy>thr]-thr
fy[fy<-thr]=fy[fy<-thr]+thr
iy = np.fft.ifft(fy)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
y.plot(axes=ax, marker='.')
ylow.plot(axes=ax, linestyle='--')
ax.plot(iy, '--')
ax.legend(('Original Signal', 'Wavelet Filter', 'Fourier Filter'))
ax.set_title(f'Filters will compress the signal {2**level} times')
plt.show()