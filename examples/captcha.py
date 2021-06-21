#!/usr/bin/env python3

from ell import *
import numpy as np

from PIL import Image, ImageEnhance

im = Image.open('CAPTCHA.png').convert('L')
c0 = Ell2D(255-np.asarray(im, dtype=np.float64))

level = 1
filter = Filter.from_name('db10')

rate = 0.7
def op(d):
    s = np.sort(np.abs(d).ravel())
    L = len(s)
    t = s.tolist()[int(L * rate)]
    return d.truncate(t, soft=False)


c = c0-c0.filter(filter.check().tensor(filter), level=level, resize=True) - -c0.filter(filter.check().tensor(), level=level, resize=True)
c /= (c.max() / 255)
c=c.truncate(150, soft=False)
im = np.asarray(c, dtype='uint8')

im = Image.fromarray(255-im, mode='L')
im.save('1.png')
# im = Image.open("1.png")    # 打开截图

sharpness = ImageEnhance.Contrast(im.convert('L'))  # 对比度增强
im = sharpness.enhance(3.0)  # 3.0为图像的饱和度

import pytesseract
cc = pytesseract.image_to_string(im, config='--psm 7').strip()
cc = ''.join(c for c in cc if c.isdigit() or c.isalpha())
print(cc)

