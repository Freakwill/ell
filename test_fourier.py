#!/usr/bin/env python3

"""Draw the graph of Daubeches' scaling function and wavelet
and their Fourier transform.
"""

from ell import *
from mymath.fft import *

d_filter = Filter.from_name('db2')

phi, psi, t1, t2 = d_filter.scaling_wavelet(level=12)

def plot(ax, fx, xlim, *args, **kwargs):
    L = len(fx)
    ax.plot(np.linspace(*xlim, L), fx, *args, **kwargs)


import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('Scaling function and wavelet')
ax = fig.subplots(2,2)
plot(ax[0, 0], phi, t1)
ax[0, 0].set_title(r'$\phi$')
plot(ax[0, 1], psi, t2)
ax[0, 1].set_title(r'$\psi$')
Fphi, w = cft(phi, t1, wrange=(-30,30), Nw=1000)
ax[1, 0].plot(w, Fphi)
ax[1, 0].set_title(r'$\hat{\phi}$')
Fpsi, w = cft(psi, t2, wrange=(-30,30), Nw=1000)
ax[1,1].plot(w, Fpsi)
ax[1, 1].set_title(r'$\hat{\psi}$')
plt.show()