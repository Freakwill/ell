
from ell import *

gm = Ell2d([[2, 4, 5, 4, 2],
[4, 9, 12, 9, 4],
[5, 12, 15, 12, 5],
[4, 9, 12, 9, 4],
[2, 4, 5, 4, 2]]) / 159

print(gm)

im = ImageRGB.open('src/lenna.jpg')

(im-im @ gm).to_image().show()

