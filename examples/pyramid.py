#!/usr/bin/env python3

from ell import *
import numpy as np
from PIL import Image

def pyramid1_demo(c, level=3, q=16):
    # in gray mode
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(level+1, 4)
    fig.suptitle(f"Pyramid Algorithm, level={level}")

    for i in range(level+1):
        ax[i, 0].set_ylabel(f'level-{i}')

    for i in range(level+1):
        for j in range(4):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

    pyramid = c.pyramid(filter, level=level, op=lambda obj: obj.quantize(q), resize=True)
    names = ('Gaussian', 'Lapacian', 'Rec Lapacian', 'Rec Gaussian')
    for j, name in enumerate(names):
        ax[0, j].set_title(name)
        for i, c in enumerate(pyramid[j]):
            ax[i,j].imshow(c.to_image(), cmap='gray')

    plt.show()

def pyramid_demo(c, level=3, q=128):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(level+1, 4)
    fig.suptitle(f"Pyramid Algorithm, level={level}")

    for i in range(level+1):
        ax[i, 0].set_ylabel(f'level-{i}')

    for i in range(level+1):
        for j in range(4):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

    pyramid = c.pyramid(filter, level=level, op=lambda obj: obj.quantize(q), resize=True)
    names = ('Gaussian', 'Lapacian', 'Rec Lapacian', 'Rec Gaussian')
    for j, name in enumerate(names):
        ax[0, j].set_title(name)
        for i, c in enumerate(pyramid[j]):
            if (j != 1 and j != 2) or i==level:
                c = c.minmaxmap()
            ax[i,j].imshow(c.to_image(resize=True))

    plt.show()

im = Image.open('lenna.jpg')
filter = Filter.from_name('db4')
c = ImageRGB.from_image(im)
pyramid_demo(c)