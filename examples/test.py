
from ell import *

a = ImageRGB.open('car1.jpg')
b = ImageRGB.open('car2.jpg')

w = Ell1d([0.05,0.1,0.05,0.6,0.05,0.1,0.05])
def _merge(a, b, w):
    Ea = a**2 @ w
    Eb = b**2 @ w
    Eab = a*b @ w
    m = 2*Eab /(Ea+Eb)
    alpha = 0.1
    w = max0(m-alpha)/(2*(1-alpha))
    w=w.resize_as(a)
    return np.max([a, b], axis=0) - w * np.abs(a-b)

c = _merge(a,b,w)
print(c.to_image().show())