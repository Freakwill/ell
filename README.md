# Ell



The space of sequences, and the operators on it.

$\ell$ is latex code of l, offen used as the space of sequences.

And Ell takes the symbol as its logo:

 ![](src/ell-logo.png)

## Concepts

A sequence is an array (`numpy.ndarray`) with start-index and end-index. I wll call it an *ell* in the context.

When adding or other operator acting on two sequences, you have two fit their indexes, that has been done by a decorator silently.



the space of sequences is a type of normal (Banach) space, also a type of *-normal (Banach) algebra



## Motivations

to implement the algorithms of wavelet analysis.

## Requirements

mainly and heavily requires `numpy`

for image classes, it also needs  [pillow](https://pillow.readthedocs.io/en/stable/)



## Download

Plz download it from github. It is not loaded up to pypi currently.

## Main classes

### Basic classes

`BaseEll`: Base calss of all space of sequences

`Ell1d, Ell2d`: sequences on $\Z$ and $\Z^2$

`Ellnd`: higher-dim sequances

`BaseMultiEll`: multi-values version of `BaseEll`, similarly it has following two subclasses

`MultiEll1d, MultiEll2d`: sequances with multi-values, such an image which is an instance of `MultiEll2d` with 3 values.



#### some mixin class

`AsReal` let the star-operator equal to `refl` operator, but one have to guarantee that no complex number joins the operations.

`ImageLike`: only for image class

### Applied classes

`Filter < Ell1d`

`ImageRGB < MultiEll2d`



### Examples

Before test the examples, plz import classes with`from ell import *`

#### basic operation

Ell is a subclass of numpy.ndarray, so it inherits all methods of ndarray. Functions defined on arrayes can act on ells and return Ell objects in most case.

```python
a = Ell1d([1,2,3,4])
b = Ell1d([2,3,4,5,5,6], min_index=-3)
c = Ell1d([-2.0, -3.0, -4.0, -4.0, -3.0, -3.0, 4.0], min_index=-3)
assert a-b==c

# tensor prod of Ell1d
assert isinstance(a.tensor(), Ell2d)
```



Only difference is that in Ell, we use `@` to implement convolution, as in following example.

```python
from ell import *

gm = 1/ 159 * Ell2d([[2, 4, 5, 4, 2],
[4, 9, 12, 9, 4],
[5, 12, 15, 12, 5],
[4, 9, 12, 9, 4],
[2, 4, 5, 4, 2]])  # gaussian mask

im = ImageRGB.open('src/lenna.jpg')

(im-im @ gm).to_image().show()

```



### draw wavelets

Run following script to draw in scaling function and wavelet wrt the filter.

```python
#!/usr/bin/env python3

"""Draw the graph of Daubeches' scaling function and wavelet.
"""

from ell import *

d_filter = Filter.from_name('db2')

phi, psi, t1, t2 = d_filter.scaling_wavelet(level=12)

def plot(ax, fx, xlim, *args, **kwargs):
    L = len(fx)
    ax.plot(np.linspace(*xlim, L), fx, *args, **kwargs)


import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('Scaling function and wavelet')
ax = fig.subplots(2)
plot(ax[0], phi, t1)
ax[0].set_title(r'$\phi$')
plot(ax[1], psi, t2)
ax[1].set_title(r'$\psi$')
plt.show()
```

![](src/daubechies-wavelet-fourier.png)



#### image process

```python
im ImageRGB.open('lenna.jpg')
im = im @ Ell1d([1/2,-1, 1/2])
# equiv. to im = im @ Ell1d([1/2,-1, 1/2]).tensor()
im.to_image().show()

# filtering by wavelets
im = ImageRGB.open('lenna.jpg')

# reducing with db2 wavelet
im = (im @ Filter.from_name('db2').H).D
# implement of Hx = D(x*h~)
# <==> im = im.reduce(Filter.from_name('db2'))
im.to_image().show()

# expanding
im = im.U @ Filter.from_name('db2')
# <==> im = im.expand(Filter.from_name('db2'))
im.to_image().show()

# to execute the two steps <==> call im = im.ezfilter(Filter.from_name('db2'))
```



### Experiments

There are some experiements in `examples/` most of whom are related to wavelets. Our ambition is to replace [pywavlets](http://pywavelets.readthedocs.io/en/latest/)

`pyramid.py` is recommanded to run to have a look at pyramid algorithm which is one of the goals to developing Ell.

## TO-DO

- [ ] define filter banks
- [x] design a logo
- [x] index should be a tuple in feature
- [ ] periodic sequences.
- [ ] audio process

