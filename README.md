# Ell



The space of sequences, and the operators on it.



$\ell$ is latex code of l, offen used as the space of sequences.



logo: 

## Concepts

A sequence is an array with start-index and end-index,

When adding or other operator acting on two sequences, you have two fit their indexes, that has been done by a decorator silently.



the space of sequences is a type of normal (Banach) space, also a type of *-normal (Banach) algebra

## Main classes

### Basic classes

`BaseEll`: Base calss of all space of sequences

`Ell1d, Ell2d`: sequences on $\Z$ and $\Z^2$

`Ellnd`: higher-dim sequances

`BaseMultiEll`: multi-values version of `BaseEll`

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

```python
a = Ell1d([1,2,3,4])
b = Ell1d([2,3,4,5,5,6], min_index=-3)
c = Ell1d([-2.0, -3.0, -4.0, -4.0, -3.0, -3.0, 4.0], min_index=-3)
assert a-b==c

# tensor prod of Ell1d
assert isinstance(a.tensor(), Ell2d)
```



#### image process

```python
im ImageRGB.open('lenna.jpg')
im = im @ Ell1d([1/2,-1, 1/2])
# equiv. to im = im @ Ell1d([1/2,-1, 1/2]).tensor()
im.to_image().show()

# filtering by wavelets
im = ImageRGB.open('lenna.jpg')
im = (im @ Filter.from_name('db2').H).D
# implement of Hx = D(x*h~)
# im = im.reduce(Filter.from_name('db2'))

im.to_image().show()
```



### Experiments

There are some experiements in `examples/` most of whom are related to wavelets. Our ambition is to replace [pywavlets](http://pywavelets.readthedocs.io/en/latest/)

## TO-DO

- [ ] define filter banks
- [ ] design a logo
- [ ] index should be a tuple in feature
- [ ] periodic sequences.
- [ ] audio process

