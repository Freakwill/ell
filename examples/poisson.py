
import numpy as np
from ell import *
from PIL import *

D = Ell1d([-1,1])
L = Ell2d([[0,1,0],[1,-4,1],[0,1,0]])

u1 = Image.open('apple1.jpeg').resize((200,200))
u2 = Image.open('lenna.jpg').resize((200,200))
u1 = ImageGray.from_image(u1)
u2 = ImageGray.from_image(u2)

r, c = u1.shape
m = 1.0 - np.outer((100< np.arange(r)) * (np.arange(r)<150), (100< np.arange(c)) * (np.arange(c)<150))
m = Ell2d(m)
d = u1 - u2
w = m.conv1d(D, axis=0) * d.conv1d(D, axis=0) + m.conv1d(D, axis=1) * d.conv1d(D, axis=1) + m * (d @ L) + (u2 @ L)
B=1-m

u = (u1+u2)/2
# u.imshow()
b = B * u2 + w
Au = B * u + u @ L
r1 = b - Au
p = r1
for _ in range(50):
    Ap = B * p + p @ L
    alpha = r1.dot(r1) / p.dot(Ap)
    u += alpha * p
    r2 = r1 - alpha * Ap
    p = r2 + r2.norm()/r1.norm()*p
    u = u.resize_as(u1)

u.imshow()

