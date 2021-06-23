#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
An open source Python library for working with sequance spaces $\ell^p$

Sequence spaces is a type of linear space in math.

A sequence is represented by two parts:
1. values: np.ndarray
2. min_index, max_index: lower bound and upper bound of the indexes on which the values are non-zero.
By default, the indexes start from 0

-------------------------------
Author: William
From: 2015-07-28 (this work is initialized from 2015 with Matlab)
"""


import copy
from types import MethodType

import numpy as np
import scipy.signal as signal
from .utils import *


COLON = np.s_[:]

_getitem = np.ndarray.__getitem__
_setitem = np.ndarray.__setitem__
_sub = np.ndarray.__sub__
_isub = np.ndarray.__isub__
_rsub = np.ndarray.__rsub__
_add = np.ndarray.__add__
_radd = np.ndarray.__radd__
_iadd = np.ndarray.__iadd__
_mul = np.ndarray.__mul__
_rmul = np.ndarray.__rmul__
_imul = np.ndarray.__imul__
_div = np.ndarray.__truediv__
_rdiv = np.ndarray.__rtruediv__
_idiv = np.ndarray.__itruediv__


def fit(f):
    def _f(obj, other):
        if np.isscalar(other):
            pass
        elif isinstance(other, BaseEll):
            mi, ma = common_index(obj.index_pair, other.index_pair)
            obj = obj.resize(mi, ma)
            other = other.resize(mi, ma)
        return f(obj, other)
    return _f

class BaseEll(np.ndarray):
    """Ell Class: sequence spaces on Z^n
    """

    def __new__(cls, array, min_index=0, max_index=None, *args, **kwargs):

        obj = np.asarray(array).view(cls)
        if isinstance(min_index, int):
            obj.min_index = np.array((min_index,)*obj.ndim)
        else:
            obj.min_index = np.array(min_index)
        if max_index is None:
            obj.max_index = min_index + np.array(obj.shape) - 1
        else:
            obj.max_index = max_index
            shape = obj.max_index - obj.min_index + 1
            obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        elif isinstance(obj, (tuple, list)):
            __array_finalize__(self, np.array(obj)) 
            return
        if isinstance(obj, BaseEll):
            self.min_index = copy.copy(obj.min_index)
            self.max_index = copy.copy(obj.max_index)
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                if not hasattr(self, 'min_index'):
                    self.min_index = 0
                if not hasattr(self, 'max_index'):
                    self.max_index = self.min_index + obj.size - 1
            else:
                if not hasattr(self, 'min_index'):
                    self.min_index = np.zeros(obj.ndim, dtype=int)
                if not hasattr(self, 'max_index'):
                    self.max_index = self.min_index + np.array(obj.shape, dtype=int) - 1
        else:
            raise TypeError('Type of `obj` should be BaseEll | ndarray | tuple | list')


    @classmethod
    def random(cls, min_index=0, *args, **kwargs):
        data = np.random.random(*args, **kwargs)
        return cls(data, min_index=min_index)

    @property
    def min_index(self):
        return self.__min_index
    
    @min_index.setter
    def min_index(self, v):
        self.__min_index = v

    @property
    def max_index(self):
        return self.__max_index

    @max_index.setter
    def max_index(self, v):
        self.__max_index = v

    @property
    def irange(self):
        return tuple(zip(self.min_index, self.max_index))

    @property
    def index_pair(self):
        return self.min_index, self.max_index
    

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):

        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, BaseEll):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, BaseEll):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(self.__class__(result, min_index=self.min_index, max_index=self.max_index)
            if output is None else output for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results


    @classmethod
    def zero(cls, min_index=0, max_index=None, ndim=1):
        if max_index is None:
            if ndim == 1:
                return cls(np.zeros(1), min_index=min_index)
            else:
                return cls(np.zeros((1,)*ndim), min_index=min_index)
        else:
            shape = max_index - min_index + 1
            return cls(np.zeros(shape), min_index=min_index, max_index=max_index)

    @classmethod
    def unit(cls, shape=(1,), ind=0):
        e = cls.zero()
        e[ind] = 1
        return e

    # basic methods
    def __str__(self):
        return f'{self.tolist()} on {self.min_index}:{self.max_index}'

    def __repr__(self):
        return f'{self.__class__}({self.tolist()}, min_index={self.min_index}, max_index={self.max_index})'

    def __format__(self, spec=None):
        if spec is None:
            return str(self)
        elif spec in {'array', 'a'}:
            return super().__format__()
        elif spec in {'index', 'i'}:
            return f'{self.min_index}:{self.max_index}'
        elif spec in {'Shape', 's'}:
            return f'{self.min_index}:{self.max_index}; {self.length}'
        elif spec in {'full', 'f'}:
            return f"""type: {self.__class__}
            coordinates: {self}
index range: {self.min_index}:{self.max_index}
length: {self.size}"""
        else:
            return str(self)

    def iszero(self):
        return np.all(self==0)

    def __eq__(self, other):
        if np.isscalar(other):
            np.all(super().__eq__(other))
        elif isinstance(other, BaseEll):
            return np.all(super().__eq__(other)) and np.all(self.min_index == other.min_index) and np.all(self.max_index == other.max_index)
        else:
            raise TypeError("type of `other` is invalid!")

    @fit
    def __iadd__(self, other):
        return _add(self, other)

    def __add__(self, other):
        cpy = self.copy()
        cpy += other
        return cpy

    @fit
    def __radd__(self, other):
        return _radd(self, other)

    @fit
    def __isub__(self, other):
        return _isub(self, other)

    @fit
    def __rsub__(self, other):
        return _rsub(self, other)

    def __sub__(self, other):
        cpy = self.copy()
        cpy -= other
        return cpy

    @fit
    def __imul__(self, other):
        return _imul(self, other)

    def __mul__(self, other):
        cpy = self.copy()
        cpy *= other
        return cpy

    @fit
    def __rmul__(self, other):
        return _rmul(self, other)


    def __imatmul__(self, other):
        # convolution: size -> size1 + size2 - 1
        raise NotImplementedError

    def __matmul__(self, other):
        if np.isscalar(other):
            return other * self
        elif isinstance(other, BaseEll):
            cpy = self.copy()
            cpy @= other
            return cpy

    @fit
    def __rmatmul__(self, other):
        # convolution: size -> size1 + size2 - 1
        if np.isscalar(other):
            return other * self
        else:
            return _rmatmul(other)

    def fill_zero(self, n_zeros=1, axis=0):
        size = tuple(abs(n_zeros) if a == axis else k for a, k in enumerate(self.shape))
        cpy = self.copy()
        if n_zeros > 0:
            array = np.concatenate([cpy, np.zeros(size)], axis=axis)
            cpy.max_index[axis] += n_zeros
            return self.__class__(array, min_index=cpy.min_index, max_index=cpy.max_index)
        elif n_zeros < 0:
            array = np.concatenate([np.zeros(size), cpy], axis=axis)
            cpy.min_index[axis] += n_zeros
            return self.__class__(array, min_index=cpy.min_index, max_index=cpy.max_index)

    def resize_as(self, other):
        return self.resize(min_index=other.min_index, max_index=other.max_index)

    def resize(self, min_index=None, max_index=None):
        # make self.min_index==min_index, self.max_index==max_index

        if min_index is not None and max_index is not None:
            if np.all(min_index>self.max_index) or np.all(max_index<self.min_index):
                return self.zero()
        else:
            if min_index is None:
                min_index = self.min_index
            if max_index is None:
                max_index = self.max_index

        m = min_index - self.min_index
        M = max_index - self.max_index
        cpy = self.copy()
        for k in range(self.ndim):
            if m[k] < 0:
                cpy = cpy.fill_zero(m[k], axis=k)
            elif m[k] > 0:
                inds = tuple(np.s_[m[k]:] if _ == k else COLON for _ in range(self.ndim))
                cpy = _getitem(cpy, inds)
                cpy.min_index[k] += m[k]
            if M[k] > 0:
                cpy = cpy.fill_zero(M[k], axis=k)
            elif M[k] < 0:
                inds = tuple(np.s_[:M[k]] if _ == k else COLON for _ in range(self.ndim))
                cpy = _getitem(cpy, inds)
                cpy.max_index[k] += M[k]
        return cpy


    # basic operation
    def truncate(self, threshold, soft=False):
        # assert threshold > 0
        cpy = self.copy()
        _setitem(cpy, np.abs(self)<=threshold, 0)
        if soft:
            _setitem(cpy, self>threshold, _getitem(self, self>threshold) - threshold)
            _setitem(cpy, self<-threshold, _getitem(self, self<-threshold) + threshold)
        return cpy

    def translate(self, k=1):
        # translation
        cpy = self.copy()
        cpy.min_index += k
        cpy.max_index += k
        return cpy

    def refl(self):
        # reflecting
        obj = np.flip(self)
        min_index, max_index = -self.max_index, -self.min_index
        return self.__class__(array, min_index=min_index, max_index=max_index)

    def reflect(self):
        # alias of `refl`
        return self.refl()

    @property
    def R(self):
        return self.refl()

    def hermite(self):
        obj = self.refl()
        return np.conj(obj)

    @property
    def H(self):
        return self.hermite()

    @property
    def star(self):
        return self.hermite()
    

    def check(self):
        # check-hat
        # low_filter -> high filter
        return self.star.translate().alt()

    def alt(self):
        # alternating
        M = altArray(self.min_index, self.max_index)
        obj = M * self
        obj.min_index, obj.max_index = self.min_index, self.max_index
        return obj


    def up_sample(self, k=2):
        '''up sampling
        U: z(m:M) => w(2m:2M),  size -> size * 2 - 1'''
        cpy = self.copy()
        if self.ndim == 1:
            raise Exception('Should use the method of Ell1d!')
        elif self.ndim == 2:
            a = np.zeros((k, k))
            a[0,0] = 1
            cpy = np.kron(cpy, a)
            cpy = np.delete(np.delete(cpy, -1, axis=0), -1, axis=1)
        else:
            raise Exception('only for dim<=2');''

        cpy.min_index *= k
        cpy.max_index *= k
        return cpy
    
    def project_sample(self, k=2):
        """project sampling
        P = UD
        """
        return self.down_sample(k=k).up_sample(k=k)


    @property
    def U(self):
        return self.up_sample(k=2)

    @property
    def D(self):
        return self.down_sample(k=2)

    @property
    def P(self):
        return self.project_sample(k=2)
    
    def quantize(self, n=64):
        # quantization
        return np.round(self / n) * n;

    # advanced operators
    def orth_test(self, k=2):
        g = self.check()
        print(f"""
            Result of orthogonality test:
            Low-pass filter: {self}
            High-pass filter: {g}
            Orthogonality of low-pass filter: {self.reduce(self, k=2)}
            Orthogonality of high-pass filter: {g.reduce(g, k=2)}
            Orthogonality of low-pass and high-pass filters: {self.reduce(g, k=2)}
            """)

    def biorth_test(self, dual, k=2):
        g = self.check()
        gg = dual.check()
        print(f"""
            Result of orthogonality test:
            Low-pass filter: {self}
            High-pass filter: {g}
            Dual low-pass filter: {dual}
            Dual high-pass filter: {gg}
            Orthogonality of low-pass filter: {self.reduce(dual, k=2)}
            Orthogonality of high-pass filter: {g.reduce(gg, k=2)}
            Orthogonality of low-pass and dual high-pass filters: {self.reduce(gg, k=2)}
            Orthogonality of dual low-pass and high-pass filters: {dual.reduce(g, k=2)}
            """)


    def expand(self, weight, k=2, level=1):
        # Uc w
        return self.up_sample(k) @ weight

    def reduce(self, weight, k=2, level=1):
        # D(cw*)
        return (self @ weight.H).down_sample(k)

    def ezfilter(self, weight, dual_weight=None, k=2):
        if dual_weight is None:
            dual_weight = weight
        return (self @ weight.H).project_sample(k) @ dual_weight


    def filter(self, weight, dual_weight=None, op=None, k=2, level=1, resize=False):
        if level == 1:
            if dual_weight is None:
                dual_weight = weight.copy()
        else:
            # equivalent filter for high level filting
            weight = compose(weight, level=level)
            if dual_weight is None:
                dual_weight = weight.copy()
            else:
                dual_weight = compose(dual_weight, level=level)
        cpy = self.copy()
        if op is None:
            res = (cpy @ weight.H).project_sample(k**level) @ dual_weight
        elif isinstance(op, str):
            res = getattr(cpy.reduce(weight, k**level), op)().expand(dual_weight, k**level)
        else:
            res = op(cpy.reduce(weight, k**level)).expand(dual_weight, k**level)
        if resize:
            return res.resize_as(self)
        else:
            return res

    def decompose(self, low_filter, dual_low_filter, k=2, level=2):
        assert level > 0 and isinstance(level, int), '`level` should be an integer >=1!'
        if dual_low_filter is None:
            dual_low_filter = low_filter
        high_filter = low_filter.check()
        dual_high_filter = dual_low_filter.check()

        low_band = self.copy()
        high_bands = []
        for l in range(level):
            high_bands.append(low.ezfilter(high_filter, dual_high_filter, k=k))
            low_band = low_band.reduce(low_filter, k=k) # current low-band
        dual_filter = dual_low_filter
        for i, h in enumerate(high_bands[1:], start=1):
            high_bands[i] = h.expand(dual_filter, k=k**i)
            dual_filter = dual_filter.up_sample() @ dual_low_filter
        low_band = low_band.expand(dual_filter, k=k**level)
        return low_band, high_bands


    def pyramid(self, low_filter, dual_low_filter=None, op=None, k=2, level=2, resize=False):
        assert level > 0 and isinstance(level, int), '`level` should be an integer >=1!'
        if dual_low_filter is None:
            dual_low_filter = low_filter
        high_filter = low_filter.check()
        dual_high_filter = dual_low_filter.check()
        
        low = self.copy()
        gauss = [low]
        laplace = []
        for l in range(level):
            low = low.reduce(low_filter, k=k) # current low-band
            gauss.append(low)
            laplace.append(gauss[-2] - low.expand(dual_low_filter, k=k))
        laplace.append(low)

        if op is None:
            rec_laplace = [l.copy() for l in laplace]
        else:
            rec_laplace = list(map(op, laplace))
        rec_gauss = [l.copy() for l in rec_laplace]

        for l in range(level-1, -1, -1):
            low = low.expand(low_filter, k)
            rec_gauss[l] = rec_laplace[l] + rec_gauss[l+1].expand(dual_low_filter)
        if resize:
            laplace[0] = laplace[0].resize_as(self)
            rec_laplace[0] = rec_laplace[0].resize_as(self)
            for k, rg in enumerate(rec_gauss):
                rec_gauss[k] = rg.resize_as(gauss[k])
        return gauss, laplace, rec_laplace, rec_gauss

    def as_real(self):
        self.hermite = MethodType(lambda obj: obj.refl(), self)


class AsReal:
    # just let star-operator == reflecting operator
    @property
    def star(self):
        return self.refl()

class BaseRealEll(AsReal, BaseEll):
    pass


class BaseMultiEll(BaseEll):

    @property
    def ndim(self):
        return super().ndim - 1

    @property
    def n_values(self):
        return super().shape[-1]

    @property
    def shape(self):
        return super().shape[:-1]

    def __array_finalize__(self, obj):
        if obj is None:
            return
        elif isinstance(obj, (tuple, list)):
            __array_finalize__(self, np.array(obj)) 
            return
        if isinstance(obj, BaseEll):
            self.min_index = copy.copy(obj.min_index)
            self.max_index = copy.copy(obj.max_index)
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                raise ValueError('The dim of array must >= 2')
            else:
                if not hasattr(self, 'min_index'):
                    self.min_index = np.zeros(obj.ndim - 1, dtype=int)
                if not hasattr(self, 'max_index'):
                    self.max_index = self.min_index + np.array(obj.shape[:-1], dtype=int) - 1
        else:
            raise TypeError('Type of `obj` should be BaseEll | ndarray | tuple | list')

    def refl(self):
        # reflecting
        obj = np.flip(self, axis=range(self.ndim))
        min_index, max_index = -self.max_index, -self.min_index
        return self.__class__(array, min_index=min_index, max_index=max_index)

    def fill_zero(self, n_zeros=1, axis=0):
        size = tuple(abs(n_zeros) if a == axis else k for a, k in enumerate(self.shape))
        cpy = self.copy()
        if n_zeros > 0:
            array = np.concatenate([cpy, np.zeros(size+(self.n_values,))], axis=axis)
            cpy.max_index[axis] += n_zeros
            return self.__class__(array, min_index=cpy.min_index, max_index=cpy.max_index)
        elif n_zeros < 0:
            array = np.concatenate([np.zeros(size+(self.n_values,)), cpy], axis=axis)
            cpy.min_index[axis] += n_zeros
            return self.__class__(array, min_index=cpy.min_index, max_index=cpy.max_index)


class Ellnd(BaseEll):
    def __new__(cls, array, min_index=0, max_index=None, *args, **kwargs):

        obj = np.asarray(array).view(cls)
        if isinstance(min_index, int):
            obj.min_index = np.array((min_index,)*obj.ndim)
        else:
            obj.min_index = np.array(min_index)
        if max_index is None:
            obj.max_index = min_index + np.array(obj.shape) - 1
        else:
            obj.max_index = max_index
            shape = obj.max_index - obj.min_index + 1
            obj = obj.resize(obj.min_index, max_index)
        return obj

    def __getitem__(self, ind):
        # get one element
        if isinstance(ind, int):
            return self[tuple(np.array((ind,)*self.ndim)-self.min_index)]
        elif isinstance(ind, tuple) and all(map(lambda x: isinstance(x, int), ind)):
            return self[tuple((np.array(ind)-self.min_index))]
        # slice of a seq.
        ss= []
        min_index = []
        max_index = []
        for k, s in enumerate(ind):
            if isinstance(s, int):
                ss.append(s)
            elif isinstance(s, slice):
                start = None if s.start is None else s.start-self.min_index[k]
                stop = None if s.stop is None else s.stop-self.min_index[k]
                ss.append(slice(start, stop, s.step))
                if s.start:
                    min_index.append(s.start)
                else:
                    min_index.append(self.min_index[k])
                if s.stop:
                    max_index.append(s.stop)
                else:
                    max_index.append(self.max_index[k])
            else:
                raise TypeError(f'Each element in `ind` should be an instance of int or slice, but {s} not.')
        array = _getitem(self, tuple(ss))
        return self.__class__(array, min_index=np.array(min_index), max_index=np.array(max_index))


class MultiEllnd(Ellnd, BaseMultiEll):
    pass

class Ell2d(Ellnd):
    
    @classmethod
    def from_image(cls, image, min_index=np.array([0,0]), max_index=None, chennal=0):
        if chennal is None:
            array = np.asarray(image, dtype=np.float64)
        else:
            array = np.asarray(image, dtype=np.float64)[:, :, chennal]
        assert array.ndim == 2, 'Make sure the array representing the image has 2 dim.'
        if max_index is None:
            max_index = min_index + np.array(array.shape) - 1
        return cls(array, min_index=min_index, max_index=max_index)

    def to_image(self, mode='L'):
        from PIL import Image
        obj = np.round(self)
        return Image.fromarray(obj.astype('uint8')).convert(mode)


    def __matmul__(self, other):
        # convolution: size -> size1 + size2 - 1
        if isinstance(other, Ell1d):
            return self.conv_tensor(other)
        elif isinstance(other, Ell2d):
            return self.conv_2d(other)
        else:
            raise TypeError("`other` should be an instance of Ell1d or Ell2d")

    def conv_2d(self, other):
        obj = signal.convolve2d(self, other)
        min_index, max_index = self.min_index+other.min_index, self.max_index+other.max_index
        return self.__class__(obj, min_index=min_index, max_index=max_index)

    def conv_tensor(self, other1, other2=None):
        if other2 is None:
            other2 = other1
            min_index, max_index = other1.min_index, other1.max_index
        else:
            min_index = np.array([other1.min_index, other2.min_index])
            max_index = np.array([other1.max_index, other2.max_index])
        obj = np.apply_along_axis(np.convolve, 0, np.asarray(self), np.asarray(other1))
        array = np.apply_along_axis(np.convolve, 1, np.asarray(obj), np.asarray(other2))
        min_index, max_index = self.min_index + min_index, self.max_index + max_index
        return self.__class__(array, min_index=min_index, max_index=max_index)

    def conv_1d(self, other, axis=0):
        if isinstance(other, Ell1d):
            obj = np.apply_along_axis(np.convolve, axis, np.asarray(self), np.asarray(other))
        else:
            raise TypeError('`other` should be an instance of Ell1d')
        min_index, max_index = self.min_index, self.max_index
        min_index[axis], max_index[axis] = self.min_index[axis] + min_index, self.max_index[axis] + max_index
        return self.__class__(obj, min_index=min_index, max_index=max_index)


    def refl(self):
        # reflecting
        array = _getitem(self, (np.s_[::-1], np.s_[::-1]))
        return self.__class__(array, min_index=-self.max_index, max_index=-self.min_index)

    def up_sample(self, k=2, axis=(0, 1)):
        '''up sampling
        U: z(m:M) => w(2m:2M),  size -> size * 2 - 1
        '''
        if axis is None or axis == (0, 1):
            if isinstance(k, int):
                k = (k, k)
            a = np.zeros(k)
            a[0,0] = 1
            array = np.kron(self, a)
            array = np.delete(np.delete(array, -np.arange(1, k[0]), axis=0), -np.arange(1, k[1]), axis=1)
        elif isinstance(axis, int):
            if axis==1:
                a = np.zeros(k)
                a[0] = 1
            else:
                a = np.zeros((1, k))
                a[0,0] = 1
            array = np.kron(self, a)
            array = np.delete(array, -np.arange(1, k), axis=axis)
        else:
            raise ValueError("Value of `axis` is invalid!")

        return self.__class__(array, min_index=self.min_index * k, max_index=self.max_index * k)

    def down_sample(self, k=2):
        """down sampling
        D: z(m:M) => w([m/2]:[M/2]), size -> [size/2]
        """
        if isinstance(k, int):
            k = (k, k)
        d10, r0 = divround(self.min_index[0], k[0])
        d11, r1 = divround(self.min_index[1], k[1])
        d20, _ = divround(self.max_index[0], k[0])
        d21, _ = divround(self.max_index[1], k[1])
        array = _getitem(self, (np.s_[r0::k[0]], np.s_[r1::k[1]]))
        min_index = np.array([d10, d11])
        max_index = np.array([d20, d21])
        return self.__class__(array, min_index=min_index, max_index=max_index)

    def expand(self, weight, k=2):
        # Uc w
        return self.up_sample(k) @ weight


    def reduce(self, weight, k=2):
        # D(cw*)
        return (self @ weight.H).down_sample(k)

    def plot(self, irange=None, scale=1, axes=None, *args, **kwargs):
        if irange is None:
            irange = self.index_array
        elif isinstance(irange, (tuple, np.ndarray)):
            irange = np.linspace(*irange[0], self.shape[0]), np.linspace(*irange[1], self.shape[1])
        else:
            raise TypeError('The type of `irange` is wrong.')
        irange = np.meshgrid(*irange, indexing='ij')
        axes.plot_surface(*irange, np.asarray(self) * scale, *args, **kwargs)

    @property
    def index_array(self):
        return np.arange(self.min_index[0], self.max_index[0]+1), np.arange(self.min_index[1], self.max_index[1]+1)


class MultiEll2d(Ell2d, BaseMultiEll):
    def up_sample(self, k=2):
        '''up sampling
        U: z(m:M) => w(2m:2M),  size -> size * 2 - 1
        '''
        if isinstance(k, int):
            k = (k, k)
        a = np.zeros(k+(1,))
        a[0,0] = 1
        array = np.kron(np.asarray(self), a)
        array = np.delete(np.delete(array, -np.arange(1, k[0]), axis=0), -np.arange(1, k[1]), axis=1)
        return self.__class__(array, min_index=self.min_index * k, max_index=self.max_index * k)

    def conv_2d(self, other):
        obj = np.dstack([signal.convolve2d(self[:,:,k], other) for k in range(self.n_values)])
        min_index, max_index = self.min_index+other.min_index, self.max_index+other.max_index
        return self.__class__(obj, min_index=min_index, max_index=max_index)


class Ell1d(BaseEll):
    """Space of abstractly summable sequences on Z
    
    Extends:
        BaseEll
    """
    def __new__(cls, array, min_index=0, max_index=None, *args, **kwargs):

        obj = np.asarray(array).view(cls)
        obj.min_index = min_index
        if max_index is None:
            obj.max_index = min_index + obj.length - 1
        else:
            obj.max_index = max_index
        return obj

    def __getitem__(self, ind):
        if isinstance(ind, int):
            return _getitem(self, ind-self.min_index)
        elif isinstance(ind, slice):
            # slice to slice
            start = None if ind.start is None else ind.start-self.min_index
            stop = None if ind.stop is None else ind.stop-self.min_index
            obj = _getitem(self, slice(start, stop, ind.step))
            if ind.start:
                obj.min_index=ind.start-self.min_index
            if ind.stop:
                obj.max_index=ind.stop-self.min_index
            return obj
        elif isinstance(ind, list):
            raise NotImplementedError
        else:
            raise TypeError('Invalid type for `ind`, as the index of the object.')

    def __format__(self, spec=None):
        if spec == 'z':
            return ' + '.join(monomial(c, k, 'z') for c, k in zip(self, range(self.irange[0], self.irange[1]+1)) if c != 0)
        else:
            return super().__format__(spec=spec)


    @property
    def irange(self):
        return (self.min_index, self.max_index)

    @property
    def mirror_range(self):
        return (1-self.max_index, 1-self.min_index)


    def resize(self, min_index, max_index):
        # make self.min_index==min_index, self.max_index==max_index
        if min_index>self.max_index or max_index<self.min_index:
            return self.zero()

        m = min_index - self.min_index
        M = max_index - self.max_index

        cpy = self.copy()
        if m < 0:
            cpy = self.fill_zero(m)
        elif m > 0:
            cpy = _getitem(cpy, slice(m, None))
        if M > 0:
            cpy = cpy.fill_zero(M)
        elif M < 0:
            cpy = _getitem(cpy, slice(None, M))
        cpy.min_index, cpy.max_index = min_index, max_index
        return cpy

    def commonInd(self, *others):
        # common index for some sequence, the new range should contain all ranges
        return min(min([other.min_index for other in others]), self.min_index), \
        max(max([other.max_index for other in others]), self.max_index)


    def fill_zero(self, n_zeros=1):
        if n_zeros > 0:
            array = np.hstack([self, np.zeros(n_zeros)])
            min_index = self.min_index
            max_index = self.max_index + n_zeros
        elif n_zeros < 0:
            array = np.hstack([np.zeros(-n_zeros), self])
            min_index = self.min_index + n_zeros
            max_index = self.max_index
        return self.__class__(array, min_index=min_index, max_index=max_index)

    def __matmul__(self, other):
        # convolution: size -> size1 + size2 - 1
        array = np.convolve(self, other)
        min_index = self.min_index + other.min_index
        max_index = self.max_index + other.max_index
        return self.__class__(array, min_index=min_index, max_index=max_index)

    def alt(self):
        # alternating
        M = altArray1d(self.min_index, self.max_index)
        obj = np.ndarray.__mul__(self, M)
        obj.min_index, obj.max_index = self.min_index, self.max_index
        return obj


    def refl(self):
        # reflecting
        return self.__class__(self[::-1], min_index=-self.max_index, max_index=-self.min_index)

    def down_sample(self, k=2):
        """down sampling
        D: z(m:M) => w([m/2]:[M/2]), size -> [size/2]
        
        Keyword Arguments:
            k {number} -- sampling interval (default: {2})
        """
        d1, r = divround(self.min_index, k)
        d2, _ = divround(self.max_index, k)
        obj = _getitem(self.copy(), np.s_[r::k])
        obj.min_index = d1
        obj.max_index = d2
        return obj

    def up_sample(self, k=2):
        """up sampling, as an inverse of down_sample
        U: z(m:M) => w(2m:2M),  size -> size * 2 - 1

        Example:
        a = Ell1d([1,2,3,4])
        a.up_sample()
        """

        a = np.zeros(k); a[0] = 1
        obj = np.delete(np.kron(self.copy(), a), -np.arange(1, k))
        return self.__class__(obj, min_index=self.min_index*k, max_index=self.max_index*k)

    def project_sample(self, k=2):
        """project sampling
        P = UD
        """
        d, r = divround(self.min_index, k)
        d2, r2 = divround(self.max_index, k)
        cpy = self.zero(min_index=d*k, max_index=d2*k)
        cpy[d*k::k]=self[d*k::k]
        return cpy

    def split_sample(self, k=2):
        """split sampling, split a sequence to (by default) even part and odd part
        
        z -> Dz, DT_1z
        
        Keyword Arguments:
            k {number} -- sampling interval (default: {2})
        """
        return self.down_sample(), self.translate().down_sample()

    @property
    def length(self):
        return self.size

    def tensor(self, other=None):
        if other is None:
            other = self
        obj = np.outer(self, other).view(Ell2d)
        obj.min_index = np.array((self.min_index, other.min_index))
        obj.max_index = np.array((self.max_index, other.max_index))
        return obj

    def fourier(self, N=None, phase='s'):
        L = self.length
        if N is None:
            N = L
        cpy = self.copy()
        if phase == 's':
            phase = -0.5
            cpy *= alt_sequences(L)
        elif phase !=0:
            cpy *= np.exp(-2j * np.pi * phase * np.arange(0, L))
        ws = np.linspace(phase, 1+phase, N, endpoint=False)
        return np.fft.fft(cpy, N) * np.exp(-2j * np.pi * self.min_index * ws), ws

    @classmethod
    def ifourier(cls, f, min_index=0, max_index=None, phase='s', padding='c'):
        N = f.shape[0]
        t = np.linspace(0, 1, N+1, endpoint=False)
        if padding == 'c':
            padding_value = f[-1]

        if max_index is None:
            max_index = N - 1

        assert max_index - min_index < N, 'The nubmer of coefficients must be larger than samples!'

        c = np.fft.ifft(f * np.exp(2j*np.pi*min_index*t))
        c = (c + (padding_value - f[0])/2) / N
        if tstart == 's':
            return cls(c[:max_index-min_index+1] * alt_sequences(max_index-min_index+1, -min_index%2), min_index=min_index, max_index=max_index)
        elif tstart == 0:
            return cls(c[:max_index-min_index+1], min_index=min_index, max_index=max_index)
        else:
            return cls(c[:max_index-min_index+1] * np.exp(2j*np.pi*phase*np.arange(min_index, max_index+1)), min_index=min_index, max_index=max_index)

    @property
    def index_array(self):
        return np.arange(self.min_index, self.max_index+1)

    def plot(self, irange=None, scale=1, axes=None, *args, **kwargs):
        if irange is None:
            irange = self.index_array
        elif isinstance(irange, (tuple, np.ndarray)):
            irange = np.linspace(*irange, self.length)
        else:
            raise TypeError('The type of `irange` is wrong.')
        axes.plot(irange, np.asarray(self) * scale, *args, **kwargs)

    def scaling_function(self, level=10):
        assert level >=7, 'level is not enough'
        return 2**(level/2) * np.asarray(compose(self, level)), self.irange

    def wavelet(self, level=10):
        assert level >=7, 'level is not enough'
        return 2**((level+1)/2) * np.asarray(self.check().up_sample(2**level) @ compose(self, level)), self.irange

    def scaling_wavelet(self, level=10):
        assert level >=7, 'level is not enough'
        t = 2**(level/2) * compose(self, level-1)
        phi = np.asarray(t.up_sample() @ self)
        psi = np.asarray(self.check().up_sample(2**(level-1)) @ t)
        return phi, psi, self.irange, self.mirror_range

    @staticmethod
    def from_function(f, lb=0, ub=1, level=0, *args, **kwargs):
        xs = np.arange(lb, ub, 2**{-level}, *args, **kwargs)
        return Ell1d(f(xs))

def compose(weight, level=1):
    # Calculate U^{n-1}h...Uh h
    if level == 1:
        return weight
    elif level > 1:
        return compose(weight, level-1).up_sample() @ weight
    else:
        raise Exception('`level` is a integer >=1.')
