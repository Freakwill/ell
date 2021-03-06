#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from types import MethodType
from collections.abc import Iterable

import numpy as np
import numpy.linalg as LA
import scipy.signal as signal
from .utils import *
from .errors import *


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
_equal = np.equal


def fit(f):
    # decorator for operators of Ell objects
    def _f(obj, other):
        if isinstance(other, BaseEll):
            mi, ma = common_index(obj.index_pair, other.index_pair)
            obj = obj.resize(mi, ma)
            other = other.resize(mi, ma)
        return f(obj, other)
    return _f

def fit2(f):
    def _f(obj, other):
        if isinstance(other, BaseEll):
            mi, ma = shared_index(obj.index_pair, other.index_pair)
            if np.any(mi>ma):
                return obj.zero()
            else:
                obj = obj.resize(mi, ma)
                other = other.resize(mi, ma)
        return f(obj, other)
    return _f

def fit3(f):
    def _f(obj, other):
        if isinstance(other, BaseEll):
            mi, ma = shared_index(obj.index_pair, other.index_pair)
            if np.any(mi>ma):
                return 0
            else:
                obj = obj.resize(mi, ma)
                other = other.resize(mi, ma)
        return f(obj, other)
    return _f


class BaseEll(np.ndarray):
    """Ell Class: sequence spaces on Z^n
    """

    _ndim = None # default ndim of an ell

    def __new__(cls, array, min_index=0, max_index=None, *args, **kwargs):
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        obj = array.view(cls)
        if isinstance(min_index, (int, Iterable)):
            if max_index is None:
                obj.min_index = min_index
            elif isinstance(max_index, Iterable):
                array = _getitem(array, tuple(slice(0, ma-mi+1) for mi, ma in zip(min_index, max_index)))
                obj = array.view(cls)
                obj._min_index = min_index if np.isscalar(min_index) else tuple(min_index)
                obj._max_index = max_index if np.isscalar(max_index) else tuple(max_index)
            elif isinstance(max_index, int):
                array = _getitem(array, slice(0, max_index-min_index+1))
                obj = array.view(cls)
                obj._min_index = min_index if np.isscalar(min_index) else tuple(min_index)
                obj._max_index = max_index if np.isscalar(max_index) else tuple(max_index)
            else:
                raise TypeError('type of argument `max_index` should be int or iterable object.')
        elif max_index is None:
            raise IndexUnavailableError()
        elif isinstance(max_index, (int, Iterable)):
            obj.max_index = max_index
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        elif isinstance(obj, (tuple, list)):
            self.__array_finalize__(np.array(obj)) 
            return
        if isinstance(obj, BaseEll):
            self._min_index = obj.min_index
            self._max_index = obj.max_index
        elif isinstance(obj, np.ndarray):
            if not hasattr(self, 'min_index'):
                self._min_index = 0
            if not hasattr(self, 'max_index'):
                self._max_index = np.subtract(obj.shape, 1)
        else:
            raise TypeError('Type of `obj` should be BaseEll | ndarray | tuple | list')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        args = []
        for input_ in inputs:
            if isinstance(input_, BaseEll):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.get('out', None)
        if outputs is None:
            outputs = (None,) * ufunc.nout
        else:
            out_args = []
            for output in outputs:
                if isinstance(output, BaseEll):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            output = outputs[0]
            return (self.__class__(results, min_index=self.min_index, max_index=self.max_index)
                if output is None else output)
        else:
            return tuple(self.__class__(result, min_index=self.min_index, max_index=self.max_index)
                if output is None else output for result, output in zip(results, outputs))


    @classmethod
    def random(cls, min_index=0, *args, **kwargs):
        data = np.random.random(*args, **kwargs)
        return cls(data, min_index=min_index)

    @property
    def min_index(self):
        if np.isscalar(self._min_index):
            return (self._min_index,) * self.ndim
        else:
            return self._min_index
    
    @min_index.setter
    def min_index(self, v):
        if v is None: return
        if np.isscalar(v):
            self._min_index = tuple(v for _ in range(self.ndim))
        else:
            self._min_index = tuple(v)
        self._max_index = tuple(np.add(v, self.shape) - 1)

    @property
    def max_index(self):
        if np.isscalar(self._max_index):
            return (self._max_index,) * self.ndim
        else:
            return self._max_index

    @max_index.setter
    def max_index(self, v):
        if v is None: return
        self._max_index = tuple(v)
        self._min_index = tuple(np.subtract(v, self.shape) + 1)


    def set_min_index(self, v, axis=None):
        if axis is None:
            if isinstance(v, int):
                if isinstance(self, Ell1d):
                    self.min_index = v
                else:
                    self.min_index = tuple(v for _ in range(self.ndim))
            else:
                self.min_index = v
        else:
            self.min_index = replace_tuple(self.min_index, v, axis)

    def inc_min_index(self, v, axis=None):
        self.min_index = inc_tuple(self.min_index, v, axis)

    def set_max_index(self, v, axis=None):
        if axis is None:
            if isinstance(v, int):
                if isinstance(self, Ell1d):
                    self.max_index = v
                else:
                    self.max_index = tuple(v for _ in range(self.ndim))
            else:
                self.max_index = v
        else:
            self.max_index = replace_tuple(self.max_index, v, axis)

    def inc_max_index(self, v, axis=None):
        self.max_index = inc_tuple(self.max_index, v, axis)

    @property
    def irange(self):
        return tuple(zip(self.min_index, self.max_index))

    @property
    def index_pair(self):
        return self.min_index, self.max_index

    @classmethod
    def zero(cls, min_index=0, max_index=None, ndim=None):
        ndim = ndim or cls._ndim or 1
        if max_index is None:
            if ndim == 1 or ndim:
                return cls(np.zeros(1), min_index=min_index)
            else:
                return cls(np.zeros((1,)*ndim), min_index=min_index)
        else:
            shape = np.subtract(max_index, min_index) + 1
            if np.isscalar(shape) and ndim>1:
                shape = (shape,)*ndim
            return cls(np.zeros(shape), min_index=min_index)

    @classmethod
    def unit(cls, index=0, ndim=None, *args, **kwargs):
        ndim = ndim or cls._ndim or 1
        if ndim == 1:
            e = cls.zero(ndim=ndim, *args, **kwargs)
            e[index] = 1
            return e
        else:
            if np.isscalar(index):
                index = tuple(index for _ in range(ndim))
            e = cls.zero(ndim=ndim, *args, **kwargs)
            e[index] = 1
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
            return f'{self.min_index}:{self.max_index}; {self.shape}'
        elif spec in {'full', 'f'}:
            return f"""type: {self.__class__}
            coordinates: {self}
index range: {self.min_index}:{self.max_index}
shape: {self.shape}"""
        else:
            return str(self)

    def iszero(self):
        return np.all(self==0)

    def __eq__(self, other):
        if np.isscalar(other):
            np.all(super().__eq__(other))
        elif isinstance(other, BaseEll):
            return np.all(_equal(self, other)) and np.all(_equal(self.min_index, other.min_index)) and np.all(_equal(self.max_index, other.max_index))
        else:
            raise TypeError("type of `other` should be scalar or ell!")

    @fit
    def __iadd__(self, other):
        return _iadd(self, other)

    @fit
    def __add__(self, other):
        return _add(self, other)

    @fit
    def __radd__(self, other):
        return _radd(self, other)

    @fit
    def __isub__(self, other):
        return _isub(self, other)

    @fit
    def __rsub__(self, other):
        return _rsub(self, other)

    @fit
    def __sub__(self, other):
        return _sub(self, other)

    @fit2
    def __imul__(self, other):
        return _imul(self, other)

    @fit
    def __mul__(self, other):
        return _mul(self, other)

    @fit2
    def __rmul__(self, other):
        return _rmul(self, other)


    def __imatmul__(self, other):
        # convolution: size -> size1 + size2 - 1
        return self.conv(other)

    def __matmul__(self, other):
        if np.isscalar(other):
            return other * self
        elif isinstance(other, BaseEll):
            cpy = self.copy()
            cpy @= other
            return cpy

    def __rmatmul__(self, other):
        # convolution: size -> size1 + size2 - 1
        if np.isscalar(other):
            return other * self
        else:
            return _rmatmul(self, other)

    def conv(self, other, mode='full', *args, **kwargs):
        array = scipy.signal.convolve(self, other, mode=mode, *args, **kwargs)
        if mode == 'full':
            _min_index = np.add(self.min_index, other.min_index)
            _max_index = np.add(self.max_index, other.max_index)
        elif mode == 'same':
            _min_index = self.min_index
            _max_index = self.max_index
        else:
            raise ModeError()
        return self.__class__(array, min_index=_min_index, max_index=_max_index)

    @fit3
    def dot(self, other=None):
        # inner prod
        if other is None:
            other = self
        return np.dot(self.ravel(), other.ravel())

    def norm(self, *args, **kwargs):
        return LA.norm(self.ravel(), *args, **kwargs)


    def fill_zero(self, n_zeros=1, axis=0):
        size = tuple(abs(n_zeros) if a == axis else k for a, k in enumerate(self.shape))
        cpy = self.copy()
        if n_zeros > 0:
            array = np.concatenate([cpy, np.zeros(size)], axis=axis)
            cpy.inc_max_index(n_zeros, axis=axis)
            return self.__class__(array, min_index=None, max_index=cpy.max_index)
        elif n_zeros < 0:
            array = np.concatenate([np.zeros(size), cpy], axis=axis)
            cpy.inc_min_index(n_zeros, axis=axis)
            return self.__class__(array, min_index=cpy.min_index)

    def fit_size(self, *args, **kwargs):
        # alias of resize
        return self.resize(*args, **kwargs)

    def fit_size_as(self, *args, **kwargs):
        # alias of resize_as
        return self.resize_as(*args, **kwargs)


    def resize_as(self, other):
        return self.resize(min_index=other.min_index, max_index=other.max_index)

    def resize(self, min_index=None, max_index=None):
        # make self.min_index==min_index, self.max_index==max_index

        if min_index is None:
            min_index = self.min_index
            if max_index is None:
                return self.copy()
        elif max_index is None:
            max_index = self.max_index
        else:
            if np.all(min_index>self.max_index) or np.all(max_index<self.min_index):
                return self.zero()

        m = np.subtract(min_index, self.min_index)
        M = np.subtract(max_index, self.max_index)
        cpy = self.copy()
        for k in range(self.ndim):
            if m[k] < 0:
                cpy = cpy.fill_zero(m[k], axis=k)
            elif m[k] > 0:
                inds = tuple(np.s_[m[k]:] if _ == k else COLON for _ in range(self.ndim))
                cpy = _getitem(cpy, inds)
                cpy.inc_min_index(m[k], axis=k)
            if M[k] > 0:
                cpy = cpy.fill_zero(M[k], axis=k)
            elif M[k] < 0:
                inds = tuple(np.s_[:M[k]] if _ == k else COLON for _ in range(self.ndim))
                cpy = _getitem(cpy, inds)
                cpy.inc_max_index(M[k], axis=k)

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
        cpy.inc_min_index(k)
        return cpy

    def refl(self, axis=None):
        # reflecting
        min_index, max_index = np.negative(self.max_index), np.negative(self.min_index)
        return self.__class__(np.flip(self, axis=axis), min_index=min_index, max_index=max_index)

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

    @property
    def g(self):
        if not hasattr(self, '_g') or self._g is None:
            return self.check()
        else:
            return self._g

    @g.setter
    def g(self, v):
        self._g = v
    
    def alt(self):
        # alternating
        M = altArray(self.min_index, self.max_index)
        obj = M * self
        obj.min_index, obj.max_index = self.min_index, self.max_index
        return obj

    def kron(self, a):
        return np.kron(np.asarray(self), a)

    def up_sample(self, step=2, axis=None):
        """up sampling
        U: z(m:M) => w(2m:2M),  size -> size * 2 - 1
        """

        array = np.asarray(self)
        if axis is None:
            if isinstance(step, int):
                step = tuple(step for _ in range(len(axis)))
            a = np.zeros((step[i] for i in range(self.ndim)))
            a[tuple(0 for i in range(self.ndim))] = 1
            array = np.kron(array, a)
            for a in range(self.ndim):
                array = np.delete(array, -np.arange(1, step[a]), axis=a)
            _min_index = op_tuple(self.min_index, lambda x: np.multiply(x, step), axis=axis)
            _max_index = op_tuple(self.max_index, lambda x: np.multiply(x, step), axis=axis)
        elif isinstance(axis, int):
            assert isinstance(step, int), '`step` should be an integer if axis is an integer!'
            a = np.zeros((step[i] if i == axis else 1 for i in range(self.ndim)))
            a[tuple(0 for i in range(self.ndim))] = 1
            cpy = self.__class__(cpy.kron(a))
            cpy = np.delete(np.delete(cpy, -np.arange(1, step), axis=0), -1, axis=1)
            _min_index = op_tuple(self.min_index, lambda x: np.multiply(x, step), axis=axis)
            _max_index = op_tuple(self.max_index, lambda x: np.multiply(x, step), axis=axis)
        elif isinstance(axis, tuple):
            a = np.zeros(tuple(step[i] if i in axis else 1 for i in range(self.ndim)))
            a[tuple(0 for i in range(self.ndim))] = 1
            cpy = self.__class__(cpy.kron(a))
            for i in axis:
                cpy = np.delete(cpy, -np.arange(1, step[i]), axis=i)
            _min_index = op_tuple(self.min_index, lambda x: np.multiply(x, step), axis=axis)
            _max_index = op_tuple(self.max_index, lambda x: np.multiply(x, step), axis=axis)
        else:
            raise TypeError('axis is an instance of int | tuple');
        return self.__class__(array, min_index=_min_index, max_index=_max_index)
    
    def project_sample(self, step=2, axis=None):
        """project sampling
        P = UD
        """
        return self.down_sample(step=step, axis=axis).up_sample(step=step, axis=axis)


    @property
    def U(self):
        return self.up_sample(step=2)

    @property
    def D(self):
        return self.down_sample(step=2)

    @property
    def P(self):
        return self.project_sample(step=2)
    
    def quantize(self, n=64):
        # quantization
        return np.round(self / n) * n;

    # advanced operators
    def orth_test(self, step=2):
        g = self.g
        print(f"""
            Result of orthogonality test:
            Low-pass filter: {self}
            High-pass filter: {g}
            Orthogonality of low-pass filter: {self.reduce(self, step=2)}
            Orthogonality of high-pass filter: {g.reduce(g, step=2)}
            Orthogonality of low-pass and high-pass filters: {self.reduce(g, step=2)}
            """)

    def biorth_test(self, dual, step=2):
        g = self.g
        gg = dual.g
        print(f"""
            Result of orthogonality test:
            Low-pass filter: {self}
            High-pass filter: {g}
            Dual low-pass filter: {dual}
            Dual high-pass filter: {gg}
            Orthogonality of low-pass filter: {self.reduce(dual, step=2)}
            Orthogonality of high-pass filter: {g.reduce(gg, step=2)}
            Orthogonality of low-pass and dual high-pass filters: {self.reduce(gg, step=2)}
            Orthogonality of dual low-pass and high-pass filters: {dual.reduce(g, step=2)}
            """)


    def expand(self, weight, step=2, level=1, axis=None):
        # expand operator: Uc w
        if isinstance(axis, int):
            return self.up_sample(step, axis=axis).conv1d(weight.H, axis=axis)
        elif axis is None:
            return self.up_sample(step=step, axis=axis) @ weight
        elif isinstance(axis, tuple):
            cpy = self.copy()
            for a in axis:
                cpy.down_sample(step, axis=a).conv1d(weight.H, axis=a)
            return cpy
        else:
            raise TypeError('type of `axis` should be int | tuple | None')

    def reduce(self, weight, step=2, level=1, axis=None):
        # reduce operator: c -> D(cw*)
        if isinstance(axis, int):
            return self.conv1d(weight.H, axis=axis).down_sample(step, axis=axis)
        elif axis is None:
            return (self @ weight.H).down_sample(step)
        elif isinstance(axis, tuple):
            cpy = self.copy()
            for a in axis:
                cpy.conv1d(weight.H, axis=a).down_sample(step, axis=a)
            return cpy
        else:
            raise TypeError('type of `axis` should be int | tuple | None')

    def ezfilter(self, weight, dual_weight=None, step=2):
        if dual_weight is None:
            dual_weight = weight
        return (self @ weight.H).project_sample(step=step) @ dual_weight


    def filter(self, weight, dual_weight=None, op=None, step=2, level=1, resize=False):
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
            res = (cpy @ weight.H).project_sample(step**level) @ dual_weight
        elif isinstance(op, str):
            res = getattr(cpy.reduce(weight, step**level), op)().expand(dual_weight, step**level)
        else:
            res = op(cpy.reduce(weight, step**level)).expand(dual_weight, step**level)
        if resize:
            return res.resize_as(self)
        else:
            return res

    def decompose(self, low_filter, dual_low_filter, step=2, level=2):
        assert level > 0 and isinstance(level, int), '`level` should be an integer >=1!'
        if dual_low_filter is None:
            dual_low_filter = low_filter
        high_filter = low_filter.g
        dual_high_filter = dual_low_filter.g

        low_band = self.copy()
        high_bands = []
        for l in range(level):
            high_bands.append(low.ezfilter(high_filter, dual_high_filter, step=step))
            low_band = low_band.reduce(low_filter, step=step) # current low-band
        dual_filter = dual_low_filter
        for i, h in enumerate(high_bands[1:], start=1):
            high_bands[i] = h.expand(dual_filter, step=step**i)
            dual_filter = dual_filter.up_sample(step=step) @ dual_low_filter
        low_band = low_band.expand(dual_filter, step=step**level)
        return low_band, high_bands


    def pyramid(self, low_filter, dual_low_filter=None, op=None, step=2, level=3, resize=False):
        """Pyramid algorithm
        see [Burt & Adelson 1983]
        """
        assert level > 0 and isinstance(level, int), '`level` should be an integer >=1!'
        if dual_low_filter is None:
            dual_low_filter = low_filter
        high_filter = low_filter.g
        dual_high_filter = dual_low_filter.g
        
        low = self.copy()
        gauss = [low]
        laplace = []
        for l in range(level):
            low = low.reduce(low_filter, step=step) # current low-band
            gauss.append(low)
            laplace.append(gauss[-2] - low.expand(dual_low_filter, step=step))
        laplace.append(low)

        if op is None:
            rec_laplace = [l.copy() for l in laplace]
        elif isinstance(op, (tuple, list)):
            if len(op) == level+1:
                rec_laplace = [l if f is None else f(l) for f, l in zip(op, laplace)]
            else:
                raise ValueError(f'the len of `op` should be {level+1}')
        elif callable(op):
            rec_laplace = list(map(op, laplace))
        else:
            raise TypeError('`op` has to be None, function or tuple of functions')

        rec_gauss = [l.copy() for l in rec_laplace]

        for l in range(level-1, -1, -1):
            low = low.expand(low_filter, step)
            rec_gauss[l] = rec_laplace[l] + rec_gauss[l+1].expand(dual_low_filter)
        if resize:
            laplace[0] = laplace[0].resize_as(self)
            rec_laplace[0] = rec_laplace[0].resize_as(self)
            for k, rg in enumerate(rec_gauss):
                rec_gauss[k] = rg.resize_as(gauss[k])
        return gauss, laplace, rec_laplace, rec_gauss

    def as_real(self):
        self.hermite = MethodType(lambda obj: obj.refl(), self)

    def to_multi(self, n_channels=3):
        return np.stack((self,)*n_channels, axis=-1)

    def conv1d(self, other, axis=0, mode='full'):
        """Convolution of Ell and Ell1d

        This is a core method of many operators!
        
        Arguments:
            other {Ell1d} -- a 1d-ell
            mode -- mode of np.convolve, only support "full" and "same"
        
        Keyword Arguments:
            axis {number} -- Axis along which a 1d-ell convolves with the object. (default: {0})
        
        Returns:
            BaseEll
        
        Raises:
            TypeError -- `other` have to be Ell1d
        """

        if isinstance(other, Ell1d) and np.asarray(other).ndim == 1:
            obj = np.apply_along_axis(np.convolve, axis, np.asarray(self), np.asarray(other), mode=mode)
        else:
            raise TypeError('`other` should be an instance of Ell1d')
        if mode == 'full':
            _min_index = inc_tuple(self.min_index, other.min_index, axis=axis)
            _max_index = inc_tuple(self.max_index, other.max_index, axis=axis)
        elif mode == 'same':
            _min_index = self.min_index
            _max_index = self.max_index
        else:
            raise ModeError()
        return self.__class__(obj, min_index=_min_index, max_index=_max_index)


class AsReal:
    """Mixin for real ell

    just let star-operator == reflecting operator
    """
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
    def n_channels(self):
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
            self._min_index = obj.min_index
            self._max_index = obj.max_index
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                raise DimError('>=2', 'Note mutli-values are set in the last dim!')
            else:
                self._min_index = np.zeros(obj.ndim - 1, dtype=int)
                self._max_index = np.subtract(obj.shape[:-1], 1)
        else:
            raise TypeError('Type of `obj` should be BaseEll | ndarray | tuple | list!')

    @classmethod
    def zero(cls, min_index=0, max_index=None, ndim=None, n_channels=3):
        ndim = ndim or cls._ndim or 1

        if max_index is None:
            if ndim == 1:
                return cls(np.zeros((1, n_channels)), min_index=min_index)
            else:
                return cls(np.zeros((1,)*ndim + (n_channels,)), min_index=min_index)
        else:
            shape = np.subtract(max_index, min_index) + 1
            if np.isscalar(shape):
                shape = tuple(shape for _ in range(ndim))
            return cls(np.zeros(shape+ (n_channels,)), min_index=min_index, max_index=max_index)

    def refl(self):
        # reflecting
        return super().refl(axis=range(self.ndim))

    def fill_zero(self, n_zeros=1, axis=0):
        size = tuple(abs(n_zeros) if a == axis else k for a, k in enumerate(self.shape))
        if n_zeros > 0:
            array = np.concatenate([self, np.zeros(size+(self.n_channels,))], axis=axis)
            return self.__class__(array, min_index=None, max_index=inc_tuple(self.max_index, n_zeros, axis=axis))
        elif n_zeros < 0:
            array = np.concatenate([np.zeros(size+(self.n_channels,)), self], axis=axis)
            return self.__class__(array, min_index=inc_tuple(self.min_index, n_zeros, axis=axis))

    def kron(self, a):
        a = a.reshape(a.shape+(1,))
        return np.kron(np.asarray(self), a)

    def get_channal(self, ch=0):
        return Ellnd(_getitem(self, (..., ch)))

    def __add__(self, other):
        if np.isscalar(other) or (isinstance(other, np.ndarray) and other.ndim==1) or isinstance(other, MultiEllnd):
            return super().__add__(other)
        elif equal_ndim(self, other):
            return super().__add__(self.embed(other))
        else:
            raise TypeError('unsupported type!')

    def __radd__(self, other):
        if np.isscalar(other) or (isinstance(other, np.ndarray) and other.ndim==1) or isinstance(other, MultiEllnd):
            return super().__radd__(other)
        elif equal_ndim(self, other):
            return super().__radd__(self.embed(other))
        else:
            raise TypeError('unsupported type!')

    def __sub__(self, other):
        if np.isscalar(other) or (isinstance(other, np.ndarray) and other.ndim==1) or isinstance(other, MultiEllnd):
            return super().__sub__(other)
        elif equal_ndim(self, other):
            return super().__sub__(self.embed(other))
        else:
            raise TypeError('unsupported type!')

    def __rsub__(self, other):
        if np.isscalar(other) or (isinstance(other, np.ndarray) and other.ndim==1) or isinstance(other, MultiEllnd):
            return super().__rsub__(other)
        elif equal_ndim(self, other):
            return super().__rsub__(self.embed(other))
        else:
            raise TypeError('unsupported type!')

    def __mul__(self, other):
        if np.isscalar(other) or (isinstance(other, np.ndarray) and other.ndim==1) or isinstance(other, MultiEllnd):
            return super().__mul__(other)
        elif equal_ndim(self, other):
            return super().__mul__(self.embed(other))
        else:
            raise TypeError('unsupported type!')

    def __rmul__(self, other):
        if np.isscalar(other) or (isinstance(other, np.ndarray) and other.ndim==1) or isinstance(other, MultiEllnd):
            return super().__rmul__(other)
        elif equal_ndim(self, other):
            return super().__rmul__(self.embed(other))
        else:
            raise TypeError('unsupported type!')


    @classmethod
    def embed(cls, other):
        # embed Ell to MultiEll
        return cls(np.expand_dims(np.asarray(other), -1), min_index=other.min_index, max_index=other.max_index)

    @classmethod
    def make_multi(cls, other, n_channels=3):
        if hasattr(other, 'min_index'):
            min_index, max_index = other.min_index, other.max_index
        else:
            min_index, max_index = 0, None
        return cls(np.tile(np.expand_dims(np.asarray(other), -1), n_channels), min_index=min_index, max_index=max_index)


class Ellnd(BaseEll):
    # just define __getitem__

    def __getitem__(self, ind):
        # get ell[ind]
        if isinstance(ind, int):
            if np.any(ind < self.min_index) or np.any(ind > self.max_index):
                return 0
            return self[(ind - self.min_index,)*self.ndim]
        elif isinstance(ind, slice):
            return self[(ind,)*self.ndim]
        elif isinstance(ind, tuple) and all(map(lambda x: isinstance(x, int), ind)):
            return self[tuple((np.array(ind)-self.min_index))]
        elif isinstance(ind, list) and all(map(lambda x: isinstance(x, (int, tuple)), ind)):
            return Ell1d([self[k] for k in ind])
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
                if s.start is None:
                    min_index.append(self.min_index[k])
                else:
                    min_index.append(s.start)
                if s.stop is None:
                    max_index.append(self.max_index[k])
                else:
                    max_index.append(s.stop)
            else:
                raise TypeError(f'Each element in `ind` should be an instance of int or slice, but {s} not.')
        array = _getitem(self, tuple(ss))
        return self.__class__(array, min_index=np.array(min_index), max_index=np.array(max_index))


class MultiEllnd(Ellnd, BaseMultiEll):
    pass

class Ell2d(Ellnd):
    _ndim = 2

    @classmethod
    def from_image(cls, image, min_index=(0,0), max_index=None, chennal=0):
        if chennal is None:
            array = np.asarray(image, dtype=np.float64)
        else:
            array = np.asarray(image, dtype=np.float64)[:, :, chennal]
        if array.ndim !=2:
            assert DimError(2, details='Make sure the array representing the image has 2 dim.')
        if max_index is None:
            max_index = np.add(min_index, array.shape) - 1
        return cls(array, min_index=min_index, max_index=max_index)

    def to_image(self, mode='L'):
        from PIL import Image
        return Image.fromarray(np.round(self).astype('uint8')).convert(mode)


    def __matmul__(self, other):
        # convolution: size -> size1 + size2 - 1
        if isinstance(other, Ell1d):
            return self.conv_tensor(other)
        elif isinstance(other, Ell2d):
            return self.conv2d(other)
        else:
            raise TypeError("`other` should be an instance of Ell1d or Ell2d")

    def conv2d(self, other, mode='full', *args, **kwargs):
        obj = signal.convolve2d(self, other, mode=mode, *args, **kwargs)
        if mode == 'full':
            _min_index = np.add(self.min_index, other.min_index)
            _max_index = np.add(self.max_index, other.max_index)
        elif mode == 'same':
            _min_index = self.min_index
            _max_index = self.max_index
        else:
            raise ModeError()
        return self.__class__(obj, min_index=_min_index, max_index=_max_index)


    def conv_tensor(self, other1, other2=None, mode='full', *args, **kwargs):
        if other2 is None:
            other2 = other1
            min_index = other1.min_index
        else:
            min_index = (other1.min_index, other2.min_index)
        array = np.apply_along_axis(np.convolve, 0, np.asarray(self), np.asarray(other1), mode=mode, *args, **kwargs)
        array = np.apply_along_axis(np.convolve, 1, np.asarray(array), np.asarray(other2), mode=mode, *args, **kwargs)
        min_index = np.add(self.min_index, min_index)
        if mode == 'full':
            min_index = np.add(self.min_index, min_index)
        elif mode == 'same':
            min_index = self.min_index
        else:
            raise ModeError()
        return self.__class__(array, min_index=min_index)

    def up_sample(self, step=2, axis=None):
        '''up sampling
        U: z(m:M) => w(2m:2M),  size -> size * 2 - 1
        '''

        array = np.asarray(self)
        if axis is None:
            if isinstance(step, int):
                step = (step, step)
            a = np.zeros(step)
            a[0,0] = 1
            array = np.delete(np.delete(np.kron(array, a), -np.arange(1, step[0]), axis=0), -np.arange(1, step[1]), axis=1)
            _min_index = op_tuple(self.min_index, lambda x: np.multiply(x, step), axis)
            _max_index = op_tuple(self.max_index, lambda x: np.multiply(x, step), axis)
        elif isinstance(axis, int):
            if axis == 0:
                a = np.zeros((1, step))
                a[0,0] = 1
            if axis == 1:
                a = np.zeros(step)
                a[0] = 1
            else:
                raise ValueError('axis = 0 or 1 if it is an integer.')
            array = np.delete(np.kron(array, a), -np.arange(1, step), axis=axis)
            _min_index = np.multiply(self.min_index, step)
            _max_index = np.multiply(self.max_index, step)
        elif isinstance(axis, tuple):
            if len(axis)==1:
                return self.up_sample(step=step, axis=axis[0])
            elif axis == (0, 1):
                return self.up_sample(step=step, axis=None)
            else:
                raise ValueError('len of `axis` has to be <=2!')
        else:
            raise ValueError("Value of `axis` is invalid!")

        return self.__class__(array, min_index=_min_index, max_index=_max_index)

    def down_sample(self, step=2, axis=None):
        """down sampling
        D: z(m:M) => w([m/2]:[M/2]), size -> [size/2]
        """
        if axis is None:
            if isinstance(step, int):
                step = (step, step)
            d10, r0 = divround(self.min_index[0], step[0])
            d20, _ = divround(self.max_index[0], step[0])
            d11, r1 = divround(self.min_index[1], step[1])
            d21, _ = divround(self.max_index[1], step[1])
            array = _getitem(self, (np.s_[r0::step[0]], np.s_[r1::step[1]]))
            _min_index = (d10, d11)
            _max_index = (d20, d21)
        elif isinstance(axis, tuple):
            if axis == (0,1):
                return self.down_sample(step)
            else:
                return self.down_sample(step, axis[0])
        elif isinstance(axis, int):
            assert isinstance(step, int)
            if axis == 0:
                d10, r0 = divround(self.min_index[0], step)
                d20, _ = divround(self.max_index[0], step)
                array = _getitem(self, (np.s_[r0::step], COLON))
                _min_index = (d10, self.min_index[1])
                _max_index = (d20, self.max_index[1])
            elif axis == 1:
                d11, r1 = divround(self.min_index[1], step)
                d21, _ = divround(self.max_index[1], step)
                array = _getitem(self, (COLON, np.s_[r1::step]))
                _min_index = (self.min_index[0], d11)
                _max_index = (self.max_index[0], d21)
            else:
                raise ValueError('axis = 0 or 1, if it is an integer')
        else:
            raise TypeError('axis has to be an instance of tuple or int')
        return self.__class__(array, min_index=_min_index, max_index=_max_index)


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


class MultiEll2d(MultiEllnd, Ell2d):
    _ndim = 2
    

    def up_sample(self, step=2, axis=None):
        """up sampling
        U: z(m:M) => w(2m:2M),  size -> size * 2 - 1
        """

        array = np.asarray(self)
        if axis is None:
            if isinstance(step, int):
                step = (step, step)
            a = np.zeros(step+(1,))
            a[0, 0, 0] = 1
            _min_index = np.multiply(self.min_index, step)
            _max_index = np.multiply(self.max_index, step)
            array = np.delete(np.delete(np.kron(array, a), -np.arange(1, step[0]), axis=0), -np.arange(1, step[1]), axis=1)
        elif isinstance(axis, int):
            assert isinstance(step, int), 'step should be an instance of int, when axis is an integer.'
            if axis == 0:
                a = np.zeros((step, 1, 1))
                a[0, 0, 0] = 1
            elif axis == 1:
                a = np.zeros((step, 1))
                a[0, 0] = 1
            else:
                raise ValueError('axis = 0 or 1 if it is an integer.')
            array = np.delete(np.kron(array, a), -np.arange(1, step), axis=axis)
            _min_index = op_tuple(self.min_index, lambda x: np.multiply(x, step), axis=axis)
            _max_index = op_tuple(self.max_index, lambda x: np.multiply(x, step), axis=axis)
        elif isinstance(axis, tuple):
            if len(axis)==1:
                return self.up_sample(step=step, axis=axis[0])
            elif axis == (0, 1):
                return self.up_sample(step=step, axis=None)
            else:
                raise ValueError('len of `axis` has to be <=2!')
        else:
            raise TypeError('`axis` is an instance of int | tuple |None');

        return self.__class__(array, min_index=_min_index, max_index=_max_index)

    def conv2d(self, other, mode='full', *args, **kwargs):
        if isinstance(other, MultiEll2d):
            obj = np.dstack([signal.convolve2d(self.get_channal(ch), other.get_channal(ch), mode=mode, *args, **kwargs) for ch in range(self.n_channels)])
        else:
            obj = np.dstack([signal.convolve2d(self.get_channal(ch), other, mode=mode, *args, **kwargs) for ch in range(self.n_channels)])
        
        if mode == 'full':
            _min_index = np.add(self.min_index, other.min_index)
            _max_index = np.add(self.max_index, other.max_index)
        elif mode == 'same':
            _min_index = self.min_index
            _max_index = self.max_index
        else:
            raise ModeError()
        return self.__class__(obj, min_index=_min_index, max_index=_max_index)


class Ell1d(BaseEll):
    """Space of abstractly summable sequences on Z
    
    Extends:
        BaseEll
    """

    ndim = 1

    def __new__(cls, array, min_index=0, max_index=None, *args, **kwargs):
        """Create 1d ell
        
        Arguments:
            array {array like} -- array like, values of the ell
        
        Keyword Arguments:
            min_index {number} -- begining index (default: {0})
            max_index {number} -- ending index (default: {None})
        
        Returns:
            Ell1d
        
        Raises:
            DimError -- the ndim of array does not match 1
            IndexUnavailableError -- did not supply index
            TypeError -- type error of index
        """

        ndim = np.ndim(array)
        if ndim ==2:
            raise DimError(details=f"The dim of the `array` that you provide is {ndim}, you may need MultiEll1d or Ell2d!")
        elif ndim !=1:
            raise DimError(details=f"The dim of the `array` that you provide is {ndim}")

        obj = np.asarray(array).view(cls)

        if np.isscalar(min_index):
            if max_index is None:
                obj.min_index = min_index
            else:
                obj._min_index = min_index
                obj._max_index = max_index
        elif min_index is None:
            if max_index is None:
                raise IndexUnavailableError()
            else:
                obj.max_index = max_index
        else:
            raise TypeError('`max_index` should be an instance of int.')
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
                obj.min_index = ind.start
            else:
                obj.min_index = self.min_index
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

    def __eq__(self, other):
        if np.isscalar(other):
            np.all(super().__eq__(other))
        elif isinstance(other, BaseEll):
            return np.all(_equal(np.asarray(self), np.asarray(other))) and self.min_index == other.min_index and self.max_index == other.max_index
        else:
            raise TypeError("type of `other` is invalid!")

    @property
    def min_index(self):
        return self._min_index

    @min_index.setter
    def min_index(self, v):
        self._min_index = v
        self._max_index = v + self.length - 1

    @property
    def max_index(self):
        return self._max_index

    @max_index.setter
    def max_index(self, v):
        self._max_index = v
        self._min_index = v - self.length + 1


    @property
    def irange(self):
        return self.min_index, self.max_index

    @property
    def mirror_range(self):
        return 1-self.max_index, 1-self.min_index


    def resize(self, min_index=None, max_index=None):
        # make self.min_index==min_index, self.max_index==max_index

        if min_index is None:
            min_index = self.min_index
            if max_index is None:
                return self.copy()
        elif max_index is None:
            max_index = self.max_index
        else:
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
        if min_index is not None:
            cpy.min_index = min_index
        elif max_index is not None:
            cpy.max_index = max_index
        return cpy

    def dot(self, other):
        # inner prod
        mi, ma = shared_index(self.index_pair, other.index_pair)
        if mi > ma:
            return 0
        else:
            obj = self.resize(mi, ma)
            other = other.resize(mi, ma)
            return np.dot(obj, other)

    def norm(self, *args, **kwargs):
        return LA.norm(self, *args, **kwargs)


    def fill_zero(self, n_zeros=1):
        if n_zeros > 0:
            array = np.hstack([self, np.zeros(n_zeros)])
            min_index = self.min_index
        elif n_zeros < 0:
            array = np.hstack([np.zeros(-n_zeros), self])
            min_index = self.min_index + n_zeros
        return self.__class__(array, min_index=min_index)

    def __matmul__(self, other):
        # convolution: size -> size1 + size2 - 1
        assert isinstance(other, Ell1d), 'other should be an instance of Ell1d'
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

    def down_sample(self, step=2, axis=None):
        """down sampling
        D: z(m:M) => w([m/2]:[M/2]), size -> [size/2]
        
        Keyword Arguments:
            step {number} -- sampling interval (default: {2})
        """
        d1, r = divround(self.min_index, step)
        d2, _ = divround(self.max_index, step)
        obj = _getitem(self, np.s_[r::step])
        obj._min_index = d1
        obj._max_index = d2
        return obj

    def up_sample(self, step=2, axis=None):
        """up sampling, as an inverse of down_sample
        U: z(m:M) => w(2m:2M),  size -> size * 2 - 1

        Example:
        a = Ell1d([1,2,3,4])
        a.up_sample()
        """

        a = np.zeros(step); a[0] = 1
        obj = np.delete(np.kron(np.asarray(self), a), -np.arange(1, step))
        return self.__class__(obj, min_index=self.min_index*step, max_index=self.max_index*step)

    def project_sample(self, step=2):
        """project sampling
        P = UD
        """
        d, _ = divround(self.min_index, step)
        d2, _ = divround(self.max_index, step)
        cpy = self.zero(min_index=d*step, max_index=d2*step)
        cpy[d*step::step]=self[d*step::step]
        return cpy

    def split_sample(self, step=2):
        """split sampling, split a sequence to (by default) even part and odd part
        
        z -> Dz, DT_1z
        
        Keyword Arguments:
            step {number} -- sampling interval (default: {2})
        """
        return self.down_sample(), self.translate().down_sample()

    @property
    def length(self):
        return self.shape[0]

    def tensor(self, other=None):
        if other is None:
            other = self
        obj = np.outer(self, other).view(Ell2d)
        obj.min_index = (self.min_index, other.min_index)
        obj.max_index = (self.max_index, other.max_index)
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

    @staticmethod
    def from_function(f, lb=0, ub=1, level=0, *args, **kwargs):
        xs = np.arange(lb, ub, 2**{-level}, *args, **kwargs)
        return Ell1d(f(xs))

def ell(*args, **kwargs):
    return Ellnd(*args, **kwargs)

def ell1d(*args, **kwargs):
    return Ell1d(*args, **kwargs)

