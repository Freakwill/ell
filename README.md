# Ell



The space of sequences, and the operators on it.



$\ell$ is latex code of l, is offen used as the space of sequences.



## Concepts

A sequance is an array with start-index and end-index,

When adding or other operator acting on two sequance, you have two fit their indexes, that has been done by a decorator.

## Main classes

### Basic classes

BaseEll: Base calss of all space of sequances

Ell1d, Ell2d: sequences on $\Z$ and $\Z^2$

Ellnd: higher-dim sequances

BaseMultiEll: multi-values version of `BaseEll`

MultiEll1d, MultiEll2d: sequances with multi-values, such an image which is an instance of `MultiEll2d` with 3 values.



### Applied classes

`Filter < Ell1d`

`ImageRGB < MultiEll2d`



### Examples

#### basic

```python
a = Ell1d([1,2,3,4])
b = Ell1d([2,3,4,5,5,6], min_index=-3)
c = Ell1d([-2.0, -3.0, -4.0, -4.0, -3.0, -3.0, 4.0], min_index=-3)
assert a-b==c
```



#### image process

```python
im = ImageRGB.open('lenna.jpg')
im = im @ Ell1D([1/2,-1, 1/2])
im.to_image().show()
```



### Experiments

There are some experiements in `examples/` most of whom are related to wavelets.

## TO-DO

- [ ] define filter banks