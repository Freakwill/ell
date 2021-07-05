#!/usr/bin/env python3

r"""Poisson Editing

# Poisson Editing

## Poisson Editing

mask of gradient: m

mixed gradient: $v=m\circ\nabla u_1+(1-m)\circ\nabla u_2=m\circ\nabla (u_1-u_2)+\nabla u_2$

$Eu=\frac{1}{2}(\|\nabla u-v\|_2^2+\|u-u_2\|_B^2)$ where $B$ is diag

poisson edit of $u_1, u_2$:

$\min Eu \sim q(B+\nabla^T\nabla, Bu_2+\nabla^Tv)= q(B-\Delta , Bu_2-w)$

where $w=\nabla m \cdot \nabla (u_1-u_2)+m\circ\Delta (u_1-u_2)+ \Delta u_2$

if m=constant then $w=\Delta(mu_1+(1-m)u_2)$

For theory see 
https://lemonzi.files.wordpress.com/2013/01/fcsa_lab3.pdf
https://piazza.com/class_profile/get_resource/hz5ykuetdmr53k/i0zbj9rijcs7m7
or https://www.cs.tau.ac.il/~dcor/Graphics/adv-slides/PoissonImageEditing06.pdf

For conjugate gradient see https://en.wikipedia.org/wiki/Conjugate_gradient_method
"""

import numpy as np
from ell import *
from PIL import *

D = Ell1d([1,-1])
L = Ell2d([[0,1,0],[1,-4,1],[0,1,0]], min_index=(-1,-1))

# embed u1 to u2
u1 = Image.open('lenna.jpg').resize((250,250))
u2 = Image.open('apple3.jpg').resize((250,250))
# u2 = u2.crop((110, 0, u2.size[0], u2.size[1]))
# u2.save('apple3.jpg')
# raise
u1 = ImageRGB.from_image(u1)
u2 = ImageRGB.from_image(u2)

r, c = u1.shape
chi = np.outer((120< np.arange(r)) * (np.arange(r)<185), (125< np.arange(c)) * (np.arange(c)<170))
m = 0.6*chi
m = Ell2d(m)

d = u1 - u2

w = d.conv1d(D, axis=0) * m.conv1d(D, axis=0) + d.conv1d(D, axis=1) * m.conv1d(D, axis=1)  +  (d @ L) * m + (u2 @ L)
B= 1- chi
B = Ell2d(B)

u = u1
# u.imshow()
b = B * u2 - w
Au = B * u - u @ L
r1 = b - Au
p = r1
s1 = r1.dot(r1)
for _ in range(10):
    Ap = B* p - p @ L
    alpha = s1 / p.dot(Ap)
    u = u + alpha * p
    r2 = r1 - alpha * Ap
    s2 = r2.dot(r2)
    p = r2 + s2/s1*p

    r1 = r2
    s1 = s2
    u = u.resize_as(u1)

u.minmaxmap().imshow()
